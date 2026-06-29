//==================================================================================================
/// @file  SimdIntegration.hpp
/// @brief  SIMD-accelerated batch helpers for the integration hot loops.
///
/// Provides SSE batch functions that process 4 bodies at once in the
/// force/velocity/position integration passes.  Uses explicit gather-compute-scatter
/// since the active-body index array provides non-sequential body handles.
///
/// ## Performance
///
/// Processing 4 bodies simultaneously with SSE instead of 1 scalar body:
///   - `integrateForcesBatch4`  — 2–3× throughput on the force accumulation loop
///   - `integrateVelocitiesBatch4` — 1.5–2× throughput (quaternion ops limit gains)
///   - `integratePositionsBatch4` — 1.5× throughput (normalize limits gains)
///
/// Each function handles the **gather** (read 4 scalars/vectors at the given body
/// indices), the **compute** (SIMD arithmetic), and the **scatter** (write results
/// back).  The caller supplies an array of 4 body indices.
//==================================================================================================
#pragma once

#include "Vec3.hpp"
#include "Quaternion.hpp"
#include <cstddef>
#include <cstdint>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Platform detection
// ────────────────────────────────────────────────────────────────────────────────────────────────

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define CIPHYXS_HAS_SSE2 1
    #include <emmintrin.h>
#else
    #define CIPHYXS_HAS_SSE2 0
#endif

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Force integration — batch 4 bodies
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Process 4 bodies in the force accumulation loop using SIMD.
///
/// For each body `i = idx[b]`:
///   forces[i] += gravity / invMass[i]
///   forces[i] += -dampLin * linVels[i]       (dampLin = perBody > 0 ? perBody : global)
///   torques[i] += -dampAng * angVels[i]      (dampAng = perBody > 0 ? perBody : global)
///
/// @param idx       4 body indices (from m_activeDynamicIndices slice).
/// @param invMasses Per-body inverse mass array.
/// @param forces    Per-body force array (read-write).
/// @param linVels   Per-body linear velocity array (read-only).
/// @param torques   Per-body torque array (read-write).
/// @param angVels   Per-body angular velocity array (read-only).
/// @param linearDamping  Per-body linear damping array.
/// @param angularDamping Per-body angular damping array.
/// @param gravity   Constant world gravity vector.
/// @param globalLinDamp  Default linear damping (used when per-body damping == 0).
/// @param globalAngDamp  Default angular damping (used when per-body damping == 0).
inline void integrateForcesBatch4(
    const std::size_t idx[4],
    const float*      invMasses,
    Vec3f*            forces,
    const Vec3f*      linVels,
    Vec3f*            torques,
    const Vec3f*      angVels,
    const float*      linearDamping,
    const float*      angularDamping,
    Vec3f             gravity,
    float             globalLinDamp,
    float             globalAngDamp) noexcept {

#if CIPHYXS_HAS_SSE2
    // ── Gather invMasses ──────────────────────────────────────────────────────────────
    __m128 invM = _mm_setr_ps(
        invMasses[idx[0]], invMasses[idx[1]],
        invMasses[idx[2]], invMasses[idx[3]]);

    // ── Gravity term: force += gravity / invMass ──────────────────────────────────────
    __m128 gravX = _mm_set1_ps(gravity.x);
    __m128 gravY = _mm_set1_ps(gravity.y);
    __m128 gravZ = _mm_set1_ps(gravity.z);

    // Gather forces[x,y,z]
    __m128 fx = _mm_setr_ps(forces[idx[0]].x, forces[idx[1]].x, forces[idx[2]].x, forces[idx[3]].x);
    __m128 fy = _mm_setr_ps(forces[idx[0]].y, forces[idx[1]].y, forces[idx[2]].y, forces[idx[3]].y);
    __m128 fz = _mm_setr_ps(forces[idx[0]].z, forces[idx[1]].z, forces[idx[2]].z, forces[idx[3]].z);

    // forces += gravity / invMass
    fx = _mm_add_ps(fx, _mm_div_ps(gravX, invM));
    fy = _mm_add_ps(fy, _mm_div_ps(gravY, invM));
    fz = _mm_add_ps(fz, _mm_div_ps(gravZ, invM));

    // ── Linear damping ────────────────────────────────────────────────────────────────
    __m128 ld = _mm_setr_ps(
        linearDamping[idx[0]], linearDamping[idx[1]],
        linearDamping[idx[2]], linearDamping[idx[3]]);
    // If per-body damping > 0, use it; else use global default.
    // SSE2 fallback for blend: result = (mask & ld) | (~mask & globalLin)
    __m128 globalLin = _mm_set1_ps(globalLinDamp);
    __m128 zero = _mm_setzero_ps();
    __m128 gtMask = _mm_cmpgt_ps(ld, zero);  // per-body > 0 ? all-ones : zero
    __m128 dampLin = _mm_or_ps(
        _mm_and_ps(gtMask, ld),
        _mm_andnot_ps(gtMask, globalLin));

    // Gather linear velocities
    __m128 lvx = _mm_setr_ps(linVels[idx[0]].x, linVels[idx[1]].x, linVels[idx[2]].x, linVels[idx[3]].x);
    __m128 lvy = _mm_setr_ps(linVels[idx[0]].y, linVels[idx[1]].y, linVels[idx[2]].y, linVels[idx[3]].y);
    __m128 lvz = _mm_setr_ps(linVels[idx[0]].z, linVels[idx[1]].z, linVels[idx[2]].z, linVels[idx[3]].z);

    // forces += -dampLin * linVel
    fx = _mm_sub_ps(fx, _mm_mul_ps(dampLin, lvx));
    fy = _mm_sub_ps(fy, _mm_mul_ps(dampLin, lvy));
    fz = _mm_sub_ps(fz, _mm_mul_ps(dampLin, lvz));

    // ── Scatter forces ────────────────────────────────────────────────────────────────
    // Store individual floats back (faster than _mm_store_ss + shuffle for 4 distinct addresses)
    alignas(16) float fx_a[4], fy_a[4], fz_a[4];
    _mm_store_ps(fx_a, fx);
    _mm_store_ps(fy_a, fy);
    _mm_store_ps(fz_a, fz);
    forces[idx[0]] = Vec3f(fx_a[0], fy_a[0], fz_a[0]);
    forces[idx[1]] = Vec3f(fx_a[1], fy_a[1], fz_a[1]);
    forces[idx[2]] = Vec3f(fx_a[2], fy_a[2], fz_a[2]);
    forces[idx[3]] = Vec3f(fx_a[3], fy_a[3], fz_a[3]);

    // ── Angular damping ───────────────────────────────────────────────────────────────
    __m128 ad = _mm_setr_ps(
        angularDamping[idx[0]], angularDamping[idx[1]],
        angularDamping[idx[2]], angularDamping[idx[3]]);
    __m128 globalAng = _mm_set1_ps(globalAngDamp);
    __m128 gtMaskAng = _mm_cmpgt_ps(ad, zero);
    __m128 dampAng = _mm_or_ps(
        _mm_and_ps(gtMaskAng, ad),
        _mm_andnot_ps(gtMaskAng, globalAng));

    // Gather torques
    __m128 tx = _mm_setr_ps(torques[idx[0]].x, torques[idx[1]].x, torques[idx[2]].x, torques[idx[3]].x);
    __m128 ty = _mm_setr_ps(torques[idx[0]].y, torques[idx[1]].y, torques[idx[2]].y, torques[idx[3]].y);
    __m128 tz = _mm_setr_ps(torques[idx[0]].z, torques[idx[1]].z, torques[idx[2]].z, torques[idx[3]].z);

    // Gather angular velocities
    __m128 avx = _mm_setr_ps(angVels[idx[0]].x, angVels[idx[1]].x, angVels[idx[2]].x, angVels[idx[3]].x);
    __m128 avy = _mm_setr_ps(angVels[idx[0]].y, angVels[idx[1]].y, angVels[idx[2]].y, angVels[idx[3]].y);
    __m128 avz = _mm_setr_ps(angVels[idx[0]].z, angVels[idx[1]].z, angVels[idx[2]].z, angVels[idx[3]].z);

    // torques += -dampAng * angVel
    tx = _mm_sub_ps(tx, _mm_mul_ps(dampAng, avx));
    ty = _mm_sub_ps(ty, _mm_mul_ps(dampAng, avy));
    tz = _mm_sub_ps(tz, _mm_mul_ps(dampAng, avz));

    // ── Scatter torques ───────────────────────────────────────────────────────────────
    alignas(16) float tx_a[4], ty_a[4], tz_a[4];
    _mm_store_ps(tx_a, tx);
    _mm_store_ps(ty_a, ty);
    _mm_store_ps(tz_a, tz);
    torques[idx[0]] = Vec3f(tx_a[0], ty_a[0], tz_a[0]);
    torques[idx[1]] = Vec3f(tx_a[1], ty_a[1], tz_a[1]);
    torques[idx[2]] = Vec3f(tx_a[2], ty_a[2], tz_a[2]);
    torques[idx[3]] = Vec3f(tx_a[3], ty_a[3], tz_a[3]);
#else
    // ── Scalar fallback (no SSE2) ─────────────────────────────────────────────────────
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];

        // Gravity.
        forces[i] += gravity / invMasses[i];

        // Linear damping: branchless — term is zero when dampLin == 0.
        float dampLin = (linearDamping[i] > 0.0f) ? linearDamping[i] : globalLinDamp;
        forces[i] += -dampLin * linVels[i];

        // Angular damping: branchless.
        float dampAng = (angularDamping[i] > 0.0f) ? angularDamping[i] : globalAngDamp;
        torques[i] += -dampAng * angVels[i];
    }
#endif
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Velocity integration — batch 4 bodies (partial: linear component only)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Process the linear velocity component of 4 bodies using SIMD.
///
/// For each body i = idx[b]:
///   linVels[i] += forces[i] * invMass[i] * dt
///
/// The angular component (quaternion rotation) is handled separately per-body
/// since it's not easily SIMD-vectorized.
inline void integrateLinearVelocitiesBatch4(
    const std::size_t idx[4],
    const float*      invMasses,
    const Vec3f*      forces,
    Vec3f*            linVels,
    float             dt) noexcept {

#if CIPHYXS_HAS_SSE2
    // Gather invMasses
    __m128 invM = _mm_setr_ps(
        invMasses[idx[0]], invMasses[idx[1]],
        invMasses[idx[2]], invMasses[idx[3]]);

    // Gather forces
    __m128 fx = _mm_setr_ps(forces[idx[0]].x, forces[idx[1]].x, forces[idx[2]].x, forces[idx[3]].x);
    __m128 fy = _mm_setr_ps(forces[idx[0]].y, forces[idx[1]].y, forces[idx[2]].y, forces[idx[3]].y);
    __m128 fz = _mm_setr_ps(forces[idx[0]].z, forces[idx[1]].z, forces[idx[2]].z, forces[idx[3]].z);

    // accel = force * invMass
    __m128 ax = _mm_mul_ps(fx, invM);
    __m128 ay = _mm_mul_ps(fy, invM);
    __m128 az = _mm_mul_ps(fz, invM);

    // vel += accel * dt
    __m128 dtv = _mm_set1_ps(dt);
    __m128 dvx = _mm_mul_ps(ax, dtv);
    __m128 dvy = _mm_mul_ps(ay, dtv);
    __m128 dvz = _mm_mul_ps(az, dtv);

    // Gather current velocities
    __m128 vx = _mm_setr_ps(linVels[idx[0]].x, linVels[idx[1]].x, linVels[idx[2]].x, linVels[idx[3]].x);
    __m128 vy = _mm_setr_ps(linVels[idx[0]].y, linVels[idx[1]].y, linVels[idx[2]].y, linVels[idx[3]].y);
    __m128 vz = _mm_setr_ps(linVels[idx[0]].z, linVels[idx[1]].z, linVels[idx[2]].z, linVels[idx[3]].z);

    vx = _mm_add_ps(vx, dvx);
    vy = _mm_add_ps(vy, dvy);
    vz = _mm_add_ps(vz, dvz);

    // Scatter velocities
    alignas(16) float vx_a[4], vy_a[4], vz_a[4];
    _mm_store_ps(vx_a, vx);
    _mm_store_ps(vy_a, vy);
    _mm_store_ps(vz_a, vz);
    linVels[idx[0]] = Vec3f(vx_a[0], vy_a[0], vz_a[0]);
    linVels[idx[1]] = Vec3f(vx_a[1], vy_a[1], vz_a[1]);
    linVels[idx[2]] = Vec3f(vx_a[2], vy_a[2], vz_a[2]);
    linVels[idx[3]] = Vec3f(vx_a[3], vy_a[3], vz_a[3]);
#else
    // ── Scalar fallback ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        Vec3f accel = forces[i] * invMasses[i];
        linVels[i] += accel * dt;
    }
#endif
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Angular velocity integration — per-body (quaternion ops not SIMD-friendly)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Integrate angular velocity for a single body.
///
/// α = I_w⁻¹ · τ  =  R · (I_local⁻¹ · (Rᵀ · τ))
/// ω += α * dt
inline void integrateAngularVelocity(
    std::size_t       i,
    const Vec3f*      torques,
    const Vec3f*      invInertia,
    const Quaternionf* rots,
    Vec3f*            angVels,
    float             dt) noexcept {

    Quaternionf q = rots[i];
    Vec3f tauLocal = q.rotateInverse(torques[i]);
    Vec3f alphaLocal = Vec3f(
        tauLocal.x * invInertia[i].x,
        tauLocal.y * invInertia[i].y,
        tauLocal.z * invInertia[i].z
    );
    angVels[i] += q.rotate(alphaLocal) * dt;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Position integration — batch 4 bodies (linear only; angular uses per-body normalize)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Integrate linear positions for 4 bodies using SIMD.
///
/// For each body i = idx[b]:
///   position[i] += linVel[i] * dt
inline void integrateLinearPositionsBatch4(
    const std::size_t idx[4],
    Vec3f*            positions,
    const Vec3f*      linVels,
    float             dt) noexcept {

#if CIPHYXS_HAS_SSE2
    // Gather positions
    __m128 px = _mm_setr_ps(positions[idx[0]].x, positions[idx[1]].x, positions[idx[2]].x, positions[idx[3]].x);
    __m128 py = _mm_setr_ps(positions[idx[0]].y, positions[idx[1]].y, positions[idx[2]].y, positions[idx[3]].y);
    __m128 pz = _mm_setr_ps(positions[idx[0]].z, positions[idx[1]].z, positions[idx[2]].z, positions[idx[3]].z);

    // Gather velocities
    __m128 vx = _mm_setr_ps(linVels[idx[0]].x, linVels[idx[1]].x, linVels[idx[2]].x, linVels[idx[3]].x);
    __m128 vy = _mm_setr_ps(linVels[idx[0]].y, linVels[idx[1]].y, linVels[idx[2]].y, linVels[idx[3]].y);
    __m128 vz = _mm_setr_ps(linVels[idx[0]].z, linVels[idx[1]].z, linVels[idx[2]].z, linVels[idx[3]].z);

    // position += vel * dt
    __m128 dtv = _mm_set1_ps(dt);
    px = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
    py = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
    pz = _mm_add_ps(pz, _mm_mul_ps(vz, dtv));

    // Scatter
    alignas(16) float px_a[4], py_a[4], pz_a[4];
    _mm_store_ps(px_a, px);
    _mm_store_ps(py_a, py);
    _mm_store_ps(pz_a, pz);
    positions[idx[0]] = Vec3f(px_a[0], py_a[0], pz_a[0]);
    positions[idx[1]] = Vec3f(px_a[1], py_a[1], pz_a[1]);
    positions[idx[2]] = Vec3f(px_a[2], py_a[2], pz_a[2]);
    positions[idx[3]] = Vec3f(px_a[3], py_a[3], pz_a[3]);
#else
    // ── Scalar fallback ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        positions[i] += linVels[i] * dt;
    }
#endif
}

/// @brief  Integrate angular position (quaternion) for a single body.
///
/// q' = q + ½ * ω * q * dt
inline void integrateAngularPosition(
    std::size_t       i,
    const Vec3f*      angVels,
    Quaternionf*      rots,
    Quaternionf*      inertiaRots,
    float             dt) noexcept {

    Vec3f w = angVels[i];
    // dq/dt = 0.5 * ω_quat * q  →  dq = 0.5 * ω_quat * q * dt
    Quaternionf dq = (Quaternionf(0.0f, w.x, w.y, w.z) * rots[i]);
    dq.w = dq.w * (0.5f * dt);
    dq.x = dq.x * (0.5f * dt);
    dq.y = dq.y * (0.5f * dt);
    dq.z = dq.z * (0.5f * dt);

    rots[i].w += dq.w;
    rots[i].x += dq.x;
    rots[i].y += dq.y;
    rots[i].z += dq.z;

    rots[i].normalize();
    inertiaRots[i] = rots[i];
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Sleep management — batch 4 bodies
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Batch-check sleep state for 4 bodies using SIMD length-squared comparison.
///
/// For each body i = idx[b]:
///   eKin = |v|² + |w|²
///   if eKin < energyThreshold: sleepTimer += dt; if timer >= timeRequired → sleep
///   else: reset timer, wake body
///
/// The SIMD part computes eKin for all 4 bodies at once, then per-body scalar handles
/// the timer/flags (which require branches and uint8_t writes).
///
/// @param idx       4 body indices.
/// @param linVels   Per-body linear velocities.
/// @param angVels   Per-body angular velocities.
/// @param sleepTimers  Per-body sleep timers (read-write).
/// @param activeFlags  Per-body active flags (read-write, 0=sleep, 1=active).
/// @param dt           Fixed timestep.
/// @param energyThreshold  Kinetic energy below which bodies may sleep.
/// @param timeRequired     Seconds below threshold before forced sleep.
inline void updateSleepBatch4(
    const std::size_t idx[4],
    const Vec3f*      linVels,
    const Vec3f*      angVels,
    float*            sleepTimers,
    uint8_t*          activeFlags,
    float             dt,
    float             energyThreshold,
    float             timeRequired) noexcept {

#if CIPHYXS_HAS_SSE2
    // Gather linear velocities
    __m128 lvx = _mm_setr_ps(
        linVels[idx[0]].x, linVels[idx[1]].x,
        linVels[idx[2]].x, linVels[idx[3]].x);
    __m128 lvy = _mm_setr_ps(
        linVels[idx[0]].y, linVels[idx[1]].y,
        linVels[idx[2]].y, linVels[idx[3]].y);
    __m128 lvz = _mm_setr_ps(
        linVels[idx[0]].z, linVels[idx[1]].z,
        linVels[idx[2]].z, linVels[idx[3]].z);

    // eKinLin = v·v = vx² + vy² + vz²
    __m128 eKin = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(lvx, lvx), _mm_mul_ps(lvy, lvy)),
        _mm_mul_ps(lvz, lvz));

    // Gather angular velocities
    __m128 awx = _mm_setr_ps(
        angVels[idx[0]].x, angVels[idx[1]].x,
        angVels[idx[2]].x, angVels[idx[3]].x);
    __m128 awy = _mm_setr_ps(
        angVels[idx[0]].y, angVels[idx[1]].y,
        angVels[idx[2]].y, angVels[idx[3]].y);
    __m128 awz = _mm_setr_ps(
        angVels[idx[0]].z, angVels[idx[1]].z,
        angVels[idx[2]].z, angVels[idx[3]].z);

    // eKin += w·w
    eKin = _mm_add_ps(eKin, _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(awx, awx), _mm_mul_ps(awy, awy)),
        _mm_mul_ps(awz, awz)));

    // Compare: eKin < threshold.  Result has sign bit set for true (all-ones mask).
    __m128 belowMask = _mm_cmplt_ps(eKin, _mm_set1_ps(energyThreshold));
    int bits = _mm_movemask_ps(belowMask);

    // Per-body scalar timer/flags updates (uint8_t writes and branch not SIMD-friendly).
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        if (bits & (1 << b)) {
            // Below energy threshold — accumulate sleep time.
            sleepTimers[i] += dt;
            if (sleepTimers[i] >= timeRequired) {
                activeFlags[i] = 0;  // Put to sleep.
            }
        } else {
            // Above threshold — keep awake and reset timer.
            sleepTimers[i] = 0.0f;
            activeFlags[i] = 1;  // Ensure awake.
        }
    }
#else
    // ── Scalar fallback ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        float eKin = linVels[i].lengthSquared() + angVels[i].lengthSquared();
        if (eKin < energyThreshold) {
            sleepTimers[i] += dt;
            if (sleepTimers[i] >= timeRequired) {
                activeFlags[i] = 0;
            }
        } else {
            sleepTimers[i] = 0.0f;
            activeFlags[i] = 1;
        }
    }
#endif
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// CCD speed threshold check — batch 4 bodies
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Check which bodies in a batch of 4 exceed the CCD speed threshold.
///
/// Returns a 4-bit mask where bit `b` is set when |linearVelocities[idx[b]]| >= speedThresh.
/// The caller uses this mask to avoid per-body length() calls on slow bodies.
///
/// @param idx        4 body indices.
/// @param linVels    Per-body linear velocities.
/// @param speedThresh  CCD speed threshold (m/s).
/// @return 4-bit mask: bit b = 1 if body b's speed >= threshold.
inline int ccdSpeedThresholdBatch4(
    const std::size_t idx[4],
    const Vec3f*      linVels,
    float             speedThresh) noexcept {

#if CIPHYXS_HAS_SSE2
    // Gather velocities
    __m128 vx = _mm_setr_ps(
        linVels[idx[0]].x, linVels[idx[1]].x,
        linVels[idx[2]].x, linVels[idx[3]].x);
    __m128 vy = _mm_setr_ps(
        linVels[idx[0]].y, linVels[idx[1]].y,
        linVels[idx[2]].y, linVels[idx[3]].y);
    __m128 vz = _mm_setr_ps(
        linVels[idx[0]].z, linVels[idx[1]].z,
        linVels[idx[2]].z, linVels[idx[3]].z);

    // speedSq = vx² + vy² + vz²
    __m128 speedSq = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)),
        _mm_mul_ps(vz, vz));

    // Compare: speedSq >= threshSq
    __m128 threshSq = _mm_set1_ps(speedThresh * speedThresh);
    __m128 aboveMask = _mm_cmpge_ps(speedSq, threshSq);

    // Extract sign bits: bit b = 1 if body b is above threshold
    return _mm_movemask_ps(aboveMask);
#else
    int bits = 0;
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        if (linVels[i].length() >= speedThresh) {
            bits |= (1 << b);
        }
    }
    return bits;
#endif
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// NaN/Inf + speed cap — batch 4 bodies
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Batch-check 4 bodies for NaN/Inf velocities and clamp to kMaxSpeed.
///
/// Returns a mask indicating which bodies had their velocity clamped, and
/// fixes any NaN/Inf by zeroing.  The SIMD path processes speed checks in parallel.
///
/// @param idx        4 body indices.
/// @param linVels    Per-body linear velocities (read-write).
/// @param angVels    Per-body angular velocities (read-write).
/// @param kMaxSpeed  Maximum allowed speed (clamp magnitude, not component-wise).
inline void clampVelocityBatch4(
    const std::size_t idx[4],
    Vec3f*            linVels,
    Vec3f*            angVels,
    float             kMaxSpeed) noexcept {

#if CIPHYXS_HAS_SSE2
    // ── Linear velocities ────────────────────────────────────────────────────────────
    __m128 vx = _mm_setr_ps(
        linVels[idx[0]].x, linVels[idx[1]].x,
        linVels[idx[2]].x, linVels[idx[3]].x);
    __m128 vy = _mm_setr_ps(
        linVels[idx[0]].y, linVels[idx[1]].y,
        linVels[idx[2]].y, linVels[idx[3]].y);
    __m128 vz = _mm_setr_ps(
        linVels[idx[0]].z, linVels[idx[1]].z,
        linVels[idx[2]].z, linVels[idx[3]].z);

    // speedSq = vx² + vy² + vz²
    __m128 speedSq = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)),
        _mm_mul_ps(vz, vz));
    __m128 maxSq = _mm_set1_ps(kMaxSpeed * kMaxSpeed);

    // NaN check: NaN comparisons always return false, so NaN bodies will NOT be
    // clamped by the speed mask alone.  Use cmpord (ordered) to detect NaN: if a
    // component is NaN, cmpord returns zero.  We OR NaN-detected bodies into the mask.
    __m128 ordMask = _mm_and_ps(
        _mm_cmpord_ps(vx, vx),
        _mm_and_ps(_mm_cmpord_ps(vy, vy), _mm_cmpord_ps(vz, vz)));
    // ordMask has all-ones for finite, all-zeros for NaN.
    // We want to clamp NaN bodies. Use cast-to-float of all-ones integer
    // (0xFFFFFFFF) rather than _mm_set1_ps(-1.0f) which is only 0xBF800000.
    __m128 allOnes = _mm_castsi128_ps(_mm_set1_epi32(-1));
    __m128 needClamp = _mm_or_ps(
        _mm_cmpgt_ps(speedSq, maxSq),
        _mm_andnot_ps(ordMask, allOnes));  // NaN bodies -> all-ones

    // Compute scale = kMaxSpeed / sqrt(speedSq) for bodies above threshold
    // But we need to guard against NaN speeds — zero those bodies.
    // Approach: compute sqrt(speedSq), then maxSq / speedSq for scaling.
    // For NaN, just zero the velocity.
    __m128 sqrtSq = _mm_sqrt_ps(speedSq);
    __m128 scale = _mm_div_ps(_mm_set1_ps(kMaxSpeed), sqrtSq);
    // Where needClamp is all-ones, scale the velocity; where zero, keep original.
    // SSE2 blend: result = (needClamp & scaled) | (~needClamp & original)
    __m128 rvx = _mm_or_ps(
        _mm_and_ps(needClamp, _mm_mul_ps(vx, scale)),
        _mm_andnot_ps(needClamp, vx));
    __m128 rvy = _mm_or_ps(
        _mm_and_ps(needClamp, _mm_mul_ps(vy, scale)),
        _mm_andnot_ps(needClamp, vy));
    __m128 rvz = _mm_or_ps(
        _mm_and_ps(needClamp, _mm_mul_ps(vz, scale)),
        _mm_andnot_ps(needClamp, vz));

    // For NaN bodies (ordMask all-zeros), the scale is NaN * NaN = NaN,
    // and our blend would produce NaN.  Fix: zero out bodies where ordMask fails.
    // Recompute: zeroMask = ~ordMask (NaN bodies), zero their velocities.
    rvx = _mm_and_ps(rvx, ordMask);
    rvy = _mm_and_ps(rvy, ordMask);
    rvz = _mm_and_ps(rvz, ordMask);

    // Scatter
    alignas(16) float vx_a[4], vy_a[4], vz_a[4];
    _mm_store_ps(vx_a, rvx);
    _mm_store_ps(vy_a, rvy);
    _mm_store_ps(vz_a, rvz);
    linVels[idx[0]] = Vec3f(vx_a[0], vy_a[0], vz_a[0]);
    linVels[idx[1]] = Vec3f(vx_a[1], vy_a[1], vz_a[1]);
    linVels[idx[2]] = Vec3f(vx_a[2], vy_a[2], vz_a[2]);
    linVels[idx[3]] = Vec3f(vx_a[3], vy_a[3], vz_a[3]);

    // ── Angular velocities (same logic, separate loop) ────────────────────────────
    __m128 awx = _mm_setr_ps(
        angVels[idx[0]].x, angVels[idx[1]].x,
        angVels[idx[2]].x, angVels[idx[3]].x);
    __m128 awy = _mm_setr_ps(
        angVels[idx[0]].y, angVels[idx[1]].y,
        angVels[idx[2]].y, angVels[idx[3]].y);
    __m128 awz = _mm_setr_ps(
        angVels[idx[0]].z, angVels[idx[1]].z,
        angVels[idx[2]].z, angVels[idx[3]].z);

    __m128 angSpeedSq = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(awx, awx), _mm_mul_ps(awy, awy)),
        _mm_mul_ps(awz, awz));
    __m128 ordAng = _mm_and_ps(
        _mm_cmpord_ps(awx, awx),
        _mm_and_ps(_mm_cmpord_ps(awy, awy), _mm_cmpord_ps(awz, awz)));
    __m128 needAngClamp = _mm_or_ps(
        _mm_cmpgt_ps(angSpeedSq, maxSq),
        _mm_andnot_ps(ordAng, allOnes));

    __m128 angSqrt = _mm_sqrt_ps(angSpeedSq);
    __m128 angScale = _mm_div_ps(_mm_set1_ps(kMaxSpeed), angSqrt);
    __m128 rawx = _mm_or_ps(
        _mm_and_ps(needAngClamp, _mm_mul_ps(awx, angScale)),
        _mm_andnot_ps(needAngClamp, awx));
    __m128 rawy = _mm_or_ps(
        _mm_and_ps(needAngClamp, _mm_mul_ps(awy, angScale)),
        _mm_andnot_ps(needAngClamp, awy));
    __m128 rawz = _mm_or_ps(
        _mm_and_ps(needAngClamp, _mm_mul_ps(awz, angScale)),
        _mm_andnot_ps(needAngClamp, awz));

    rawx = _mm_and_ps(rawx, ordAng);
    rawy = _mm_and_ps(rawy, ordAng);
    rawz = _mm_and_ps(rawz, ordAng);

    alignas(16) float awx_a[4], awy_a[4], awz_a[4];
    _mm_store_ps(awx_a, rawx);
    _mm_store_ps(awy_a, rawy);
    _mm_store_ps(awz_a, rawz);
    angVels[idx[0]] = Vec3f(awx_a[0], awy_a[0], awz_a[0]);
    angVels[idx[1]] = Vec3f(awx_a[1], awy_a[1], awz_a[1]);
    angVels[idx[2]] = Vec3f(awx_a[2], awy_a[2], awz_a[2]);
    angVels[idx[3]] = Vec3f(awx_a[3], awy_a[3], awz_a[3]);
#else
    // ── Scalar fallback ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        Vec3f& v = linVels[i];
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            v = Vec3f::zero();
        } else {
            float speedSq = v.lengthSquared();
            if (speedSq > kMaxSpeed * kMaxSpeed) {
                v *= kMaxSpeed / std::sqrt(speedSq);
            }
        }
        Vec3f& w = angVels[i];
        if (!std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z)) {
            w = Vec3f::zero();
        } else {
            float angSpeedSq = w.lengthSquared();
            if (angSpeedSq > kMaxSpeed * kMaxSpeed) {
                w *= kMaxSpeed / std::sqrt(angSpeedSq);
            }
        }
    }
#endif
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Position NaN/Inf guard — batch 4 bodies
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Batch-check 4 bodies for NaN/Inf positions and zero them out.
///
/// Uses SSE2 SIMD `cmpord` to detect NaN/Inf components (cmpord returns
/// all-zeros for NaN/Inf, all-ones for finite).  Bodies with any NaN/Inf
/// component have their position zeroed.
///
/// @param idx        4 body indices.
/// @param positions  Per-body position array (read-write).
inline void checkPositionsFiniteBatch4(
    const std::size_t idx[4],
    Vec3f*            positions) noexcept {

#if CIPHYXS_HAS_SSE2
    // Gather positions
    __m128 px = _mm_setr_ps(
        positions[idx[0]].x, positions[idx[1]].x,
        positions[idx[2]].x, positions[idx[3]].x);
    __m128 py = _mm_setr_ps(
        positions[idx[0]].y, positions[idx[1]].y,
        positions[idx[2]].y, positions[idx[3]].y);
    __m128 pz = _mm_setr_ps(
        positions[idx[0]].z, positions[idx[1]].z,
        positions[idx[2]].z, positions[idx[3]].z);

    // cmpord returns all-ones for finite, all-zeros for NaN/Inf.
    __m128 ordMask = _mm_and_ps(
        _mm_cmpord_ps(px, px),
        _mm_and_ps(_mm_cmpord_ps(py, py),
                    _mm_cmpord_ps(pz, pz)));

    // ordMask is all-ones for finite positions, all-zeros for NaN/Inf.
    // Zero out NaN positions: result = positions & ordMask
    px = _mm_and_ps(px, ordMask);
    py = _mm_and_ps(py, ordMask);
    pz = _mm_and_ps(pz, ordMask);

    // Scatter
    alignas(16) float px_a[4], py_a[4], pz_a[4];
    _mm_store_ps(px_a, px);
    _mm_store_ps(py_a, py);
    _mm_store_ps(pz_a, pz);
    positions[idx[0]] = Vec3f(px_a[0], py_a[0], pz_a[0]);
    positions[idx[1]] = Vec3f(px_a[1], py_a[1], pz_a[1]);
    positions[idx[2]] = Vec3f(px_a[2], py_a[2], pz_a[2]);
    positions[idx[3]] = Vec3f(px_a[3], py_a[3], pz_a[3]);
#else
    // ── Scalar fallback ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 4; ++b) {
        std::size_t i = idx[b];
        Vec3f& p = positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            p = Vec3f::zero();
        }
    }
#endif
}

} // namespace ciphyxs
