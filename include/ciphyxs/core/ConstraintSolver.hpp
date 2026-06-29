//==================================================================================================
/// @file  ConstraintSolver.hpp
/// @brief  Sequential-impulse constraint solver with warm starting and friction.
///
/// This is the heart of the physics engine.  The algorithm is the industry-standard
/// **Sequential Impulse** method (Erwin Coumans / Box2D / PhysX lineage):
///
///   for each iteration:
///     for each manifold:
///       for each contact point:
///         1. Compute relative velocity at the contact.
///         2. Compute impulse along the constraint axis (normal + 2 friction tangents).
///         3. Clamp to satisfy the constraint inequality.
///         4. Apply impulse immediately (affects subsequent constraints in the same pass).
///
/// The solver is **deterministic** given a fixed manifold order (iteration over handle indices
/// in ascending order).
//==================================================================================================
#pragma once

#include "../collision/Contact.hpp"
#include "../core/RigidBody.hpp"
#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SolverConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Parameters controlling the sequential-impulse solver behaviour.
struct SolverConfig {
    /// @brief  Number of solver iterations per fixed step (6–12 typical).
    ///         Higher values improve stacking stability but cost O(N·iter).
    std::uint32_t numIterations = 10;

    /// @brief  Baumgarte stabilisation factor for penetration correction [0, 1].
    ///         Penetration bias = -baumgarte * depth / dt.
    ///         A value of 0.10–0.25 works well for most scenes.
    float baumgarte = 0.15f;

    /// @brief  Velocity threshold below which restitution is suppressed (m/s).
    ///         Prevents jittering for objects resting on each other.
    float restitutionThreshold = 1.0f;

    /// @brief  Maximum allowed penetration correction per step (m).
    ///         Prevents large impulses from deep interpenetration.
    float maxPenetrationCorrection = 0.15f;

    /// @brief  Enable warm starting (re-use impulses from the previous frame).
    bool enableWarmStart = true;

    /// @brief  Warm-starting factor — scales down cached impulses to avoid over-correction.
    float warmStartFactor = 0.8f;

    /// @brief  [DEPRECATED] Linear velocity damping factor.
    ///         Damping is now handled by PhysicsWorld using the Rayleigh model (F = -c.v)
    ///         during the force integration phase.  These fields are retained for API compat
    ///         but are no longer read by the solver.
    float linearDamping  = 0.0f;

    /// @brief  [DEPRECATED] Angular velocity damping factor.
    float angularDamping = 0.0f;

    /// @brief  Minimum total contact points before pre-computation is used.
    ///         When totalPoints <= this threshold, the solver skips Phase 2
    ///         (pre-compute scratch data) and computes per-contact data on the fly.
    ///         This avoids the overhead of clearing + resizing m_scratch + 3
    ///         effectiveMass() calls per point when those savings would be tiny.
    ///         
    ///         Default: 4 — large enough to cover most single-box stacks while
    ///         avoiding overhead for tiny contact sets (common in ChaosDensityTest
    ///         where small Voronoi fragments produce 1-2 contacts per pair).
    std::uint32_t minPrecomputationPoints = 4;

    /// @brief  Early-exit convergence threshold (impulse-change sum per iteration).
    ///         When the sum of absolute impulse deltas across all contacts in an
    ///         iteration drops below this threshold, the solver exits early.
    ///         Set to 0 to disable (always run all iterations).
    ///         Default: 0.001 — small enough to not affect stacking quality while
    ///         saving iterations on nearly-settled stacks.
    float earlyExitThreshold = 0.001f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ConstraintSolver
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Solves contact constraints using the sequential-impulse method.
///
/// Operates directly on the SoA `RigidBodyStorage` and a mutable array of
/// `ContactManifold`s (to update warm-starting caches).
class ConstraintSolver {
public:
    /// @brief  Run the solver for one fixed timestep.
    ///
    /// @param dt         Fixed timestep (seconds).
    /// @param manifolds  Contact manifolds (points updated in-place for warm-start cache).
    /// @param bodies     SoA rigid-body storage. Velocities are modified in-place.
    /// @param config     Solver parameters.
    // ─── Pre-computed per-contact scratch data for solver iteration invariance ────────────
    //
    // rA, rB, tangent basis, effective masses, and bias depend only on positions and
    // contact normals, which are CONSTANT during the solver phase (velocities change,
    // but positions stay fixed until after the solver).  Pre-computing ONCE before the
    // main solver loop saves ~120 float operations per contact point per iteration.
    //
    // For 200 boxes × 400 contacts × 10 iterations:  ~430,000 ops saved per frame.
    struct ContactScratch {
        Vec3f         rA, rB;
        Vec3f         tangent[2];
        float         normalMass;
        float         tangentMass[2];
        float         bias;
        float         penetration;
        RigidBodyHandle bodyA, bodyB;
    };

    void solve(float dt,
               std::vector<ContactManifold>& manifolds,
               RigidBodyStorage& bodies,
               const SolverConfig& config) const noexcept {

        if (manifolds.empty()) return;

        auto h = bodies.hot();  // solver-hot view

        // Phase 1: Warm starting.
        if (config.enableWarmStart) {
            warmStart(manifolds, h, config);
        }

        // --- Phase 2: Pre-compute invariant contact data --------------------------------
        // rA, rB, tangent basis, effective masses, and bias depend only on positions
        // and contact normals, which are constant during the solver phase.
        // Pre-computing once saves ~120 ops per contact per iteration.
        std::size_t totalPoints = 0;
        for (auto& manifold : manifolds) totalPoints += manifold.pointCount;

        // --- Scene-adaptive: skip pre-computation for tiny contact sets -----------------
        // When totalPoints <= minPrecomputationPoints, the overhead of clearing and
        // resizing m_scratch plus computing 3 effectiveMass() calls per point exceeds
        // the per-iteration savings.  This is common in ChaosDensityTest where small
        // Voronoi fragments produce 1-2 contact points per pair in tiny islands.
        if (totalPoints > config.minPrecomputationPoints) {
            // Pre-computation path.
            m_scratch.clear();
            m_scratch.resize(totalPoints);

            std::size_t si = 0;
            for (auto& manifold : manifolds) {
                RigidBodyHandle hA = manifold.bodyA;
                RigidBodyHandle hB = manifold.bodyB;

                float invMassA = h.inverseMasses[hA];
                float invMassB = h.inverseMasses[hB];
                Vec3f  invInertiaA = h.inverseInertiaDiag[hA];
                Vec3f  invInertiaB = h.inverseInertiaDiag[hB];
                Quaternionf inertiaRotA = h.inertiaRotations[hA];
                Quaternionf inertiaRotB = h.inertiaRotations[hB];

                for (int p = 0; p < manifold.pointCount; ++p) {
                    auto& pt = manifold.points[p];

                    Vec3f rA = pt.position - h.positions[hA];
                    Vec3f rB = pt.position - h.positions[hB];
                    Vec3f n  = pt.normal;

                    // Reuse cached tangent basis (built once, constant for contact lifetime).
                    Vec3f t1, t2;
                    if (pt.tangent[0].lengthSquared() < 0.5f) {
                        buildTangentBasis(n, t1, t2);
                        pt.tangent[0] = t1;
                        pt.tangent[1] = t2;
                    } else {
                        t1 = pt.tangent[0];
                        t2 = pt.tangent[1];
                    }

                    float nM  = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                              inertiaRotA, inertiaRotB, rA, rB, n);
                    float t1M = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                              inertiaRotA, inertiaRotB, rA, rB, t1);
                    float t2M = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                              inertiaRotA, inertiaRotB, rA, rB, t2);

                    float pen = std::max(0.0f, pt.penetration);
                    float bias = -config.baumgarte
                               * std::min(pen, config.maxPenetrationCorrection) / dt;

                    m_scratch[si++] = {
                        rA, rB,
                        {t1, t2},
                        nM,
                        {t1M, t2M},
                        bias,
                        pt.penetration,
                        hA, hB
                    };
                }
            }

            // --- Phase 3: Main solver loop -- uses pre-computed scratch data ------------
            // Early-exit: track total absolute impulse change per iteration.
            // When the sum drops below earlyExitThreshold, convergence is reached.
            for (std::uint32_t iter = 0; iter < config.numIterations; ++iter) {
                float totalChange = 0.0f;
                std::size_t si = 0;
                for (auto& manifold : manifolds) {
                    float combinedFriction    = manifold.combinedFriction;
                    float combinedRestitution = manifold.combinedRestitution;

                    for (int p = 0; p < manifold.pointCount; ++p) {
                        auto& cd = m_scratch[si++];
                        auto& pt = manifold.points[p];

                        Vec3f n = pt.normal;

                        // Relative velocity at contact.
                        Vec3f vA = h.linearVelocities[cd.bodyA];
                        Vec3f vB = h.linearVelocities[cd.bodyB];
                        Vec3f wA = h.angularVelocities[cd.bodyA];
                        Vec3f wB = h.angularVelocities[cd.bodyB];

                        Vec3f vRel = (vB + wB.cross(cd.rB)) - (vA + wA.cross(cd.rA));
                        float vn = vRel.dot(n);

                        float bias = cd.bias;
                        if (vn < -config.restitutionThreshold) {
                            bias -= combinedRestitution * vn;
                        }

                        float lambdaN = (-(vn + bias)) * cd.normalMass;
                        float newN = pt.normalImpulse + lambdaN;
                        if (newN < 0.0f) newN = 0.0f;
                        float appliedN = newN - pt.normalImpulse;
                        totalChange += std::abs(appliedN);
                        pt.normalImpulse = newN;

                        Vec3f impulseN = n * appliedN;
                        applyImpulse(h, cd.bodyA, cd.bodyB, impulseN, cd.rA, cd.rB);

                        // Friction
                        vA = h.linearVelocities[cd.bodyA];
                        vB = h.linearVelocities[cd.bodyB];
                        wA = h.angularVelocities[cd.bodyA];
                        wB = h.angularVelocities[cd.bodyB];
                        vRel = (vB + wB.cross(cd.rB)) - (vA + wA.cross(cd.rA));

                        Vec3f impulseF = Vec3f::zero();
                        float maxF = combinedFriction * pt.normalImpulse;
                        for (int d = 0; d < 2; ++d) {
                            float vt = vRel.dot(cd.tangent[d]);
                            float lt = -vt * cd.tangentMass[d];
                            float newT = pt.tangentImpulse[d] + lt;
                            newT = std::max(-maxF, std::min(maxF, newT));
                            float appliedT = newT - pt.tangentImpulse[d];
                            totalChange += std::abs(appliedT);
                            pt.tangentImpulse[d] = newT;
                            impulseF += cd.tangent[d] * appliedT;
                        }

                        applyImpulse(h, cd.bodyA, cd.bodyB, impulseF, cd.rA, cd.rB);
                    }
                }
                // Early-exit: if total impulse change is below threshold, converged.
                if (config.earlyExitThreshold > 0.0f && totalChange < config.earlyExitThreshold) {
                    break;
                }
            }
        } else {
            // --- No pre-computation: use per-manifold solveManifold (on-the-fly) -------
            // For tiny contact sets, this avoids scratch overhead with negligible
            // per-iteration cost increase.
            for (std::uint32_t iter = 0; iter < config.numIterations; ++iter) {
                float totalChange = 0.0f;
                for (auto& manifold : manifolds) {
                    totalChange += solveManifold(dt, manifold, h, config);
                }
                // Early-exit: if total impulse change is below threshold, converged.
                if (config.earlyExitThreshold > 0.0f && totalChange < config.earlyExitThreshold) {
                    break;
                }
            }
        }

        // Phase 4: Split-impulse positional correction (improves stacking stability).
        positionalSolve(h, manifolds, config);
    }

    /// @brief  Split-impulse positional correction to resolve residual interpenetration.
    ///
    /// After the velocity-impulse solver, this pass applies direct position corrections
    /// along contact normals to push bodies apart.  Unlike Baumgarte velocity bias, these
    /// corrections do NOT affect velocities for the next step — the correction is applied
    /// directly to positions and rotations only.
    ///
    /// A small penetration slop is allowed to prevent jittering.
    void positionalSolve(RigidBodyHotSpan h,
                         std::vector<ContactManifold>& manifolds,
                         const SolverConfig& config) const noexcept {
        if (manifolds.empty()) return;

        (void)config; // currently unused (uses hard-coded parameters)

        constexpr float kSlop            = 0.005f;  ///< Allowed penetration (m).
        constexpr float kCorrection      = 0.35f;   ///< Fraction of residual penetration fixed per iteration.
        constexpr int   kPositionIters   = 3;       ///< Number of position-correction iterations.

        for (int iter = 0; iter < kPositionIters; ++iter) {
            for (auto& manifold : manifolds) {
                RigidBodyHandle hA = manifold.bodyA;
                RigidBodyHandle hB = manifold.bodyB;

                float invMassA = h.inverseMasses[hA];
                float invMassB = h.inverseMasses[hB];
                Vec3f invInertiaA = h.inverseInertiaDiag[hA];
                Vec3f invInertiaB = h.inverseInertiaDiag[hB];
                Quaternionf inertiaRotA = h.inertiaRotations[hA];
                Quaternionf inertiaRotB = h.inertiaRotations[hB];

                for (int p = 0; p < manifold.pointCount; ++p) {
                    auto& pt = manifold.points[p];
                    if (pt.penetration <= kSlop) continue;

                    Vec3f rA = pt.position - h.positions[hA];
                    Vec3f rB = pt.position - h.positions[hB];
                    Vec3f n  = pt.normal;

                    // Effective mass (same formula as velocity solver).
                    float normalMass = invMassA + invMassB;

                    Vec3f rCrossA = rA.cross(n);
                    Vec3f localA  = inertiaRotA.rotateInverse(rCrossA);
                    normalMass += localA.x * localA.x * invInertiaA.x
                                + localA.y * localA.y * invInertiaA.y
                                + localA.z * localA.z * invInertiaA.z;

                    Vec3f rCrossB = rB.cross(n);
                    Vec3f localB  = inertiaRotB.rotateInverse(rCrossB);
                    normalMass += localB.x * localB.x * invInertiaB.x
                                + localB.y * localB.y * invInertiaB.y
                                + localB.z * localB.z * invInertiaB.z;

                    if (normalMass <= 1e-10f) continue;

                    // Compute the position correction (clamped to avoid overshoot).
                    float depth  = pt.penetration - kSlop;
                    float correction = std::max(depth, 0.0f) * kCorrection;

                    // Distribute the correction as a pseudo-impulse.
                    float impulse = correction / normalMass;
                    Vec3f corrVec = n * impulse;

                    // Apply linear position correction.
                    if (invMassA > 0.0f) {
                        h.positions[hA] -= corrVec * invMassA;
                    }
                    if (invMassB > 0.0f) {
                        h.positions[hB] += corrVec * invMassB;
                    }

                    // Apply angular position correction (small rotation to follow contact torque).
                    // Body A
                    if (rCrossA.lengthSquared() > 1e-12f) {
                        Vec3f torqueA = rA.cross(-corrVec * invMassA);
                        Vec3f localTorqueA = inertiaRotA.rotateInverse(torqueA);
                        Vec3f deltaAngleA(
                            localTorqueA.x * invInertiaA.x,
                            localTorqueA.y * invInertiaA.y,
                            localTorqueA.z * invInertiaA.z);
                        // q' = q + ½ * Δθ * q
                        Quaternionf dqA = Quaternionf(0.0f, deltaAngleA.x, deltaAngleA.y, deltaAngleA.z)
                                        * h.rotations[hA];
                        dqA.w *= 0.5f; dqA.x *= 0.5f; dqA.y *= 0.5f; dqA.z *= 0.5f;
                        h.rotations[hA].w += dqA.w;
                        h.rotations[hA].x += dqA.x;
                        h.rotations[hA].y += dqA.y;
                        h.rotations[hA].z += dqA.z;
                        h.rotations[hA].normalize();
                        h.inertiaRotations[hA] = h.rotations[hA];
                    }

                    // Body B
                    if (rCrossB.lengthSquared() > 1e-12f) {
                        Vec3f torqueB = rB.cross(corrVec * invMassB);
                        Vec3f localTorqueB = inertiaRotB.rotateInverse(torqueB);
                        Vec3f deltaAngleB(
                            localTorqueB.x * invInertiaB.x,
                            localTorqueB.y * invInertiaB.y,
                            localTorqueB.z * invInertiaB.z);
                        Quaternionf dqB = Quaternionf(0.0f, deltaAngleB.x, deltaAngleB.y, deltaAngleB.z)
                                        * h.rotations[hB];
                        dqB.w *= 0.5f; dqB.x *= 0.5f; dqB.y *= 0.5f; dqB.z *= 0.5f;
                        h.rotations[hB].w += dqB.w;
                        h.rotations[hB].x += dqB.x;
                        h.rotations[hB].y += dqB.y;
                        h.rotations[hB].z += dqB.z;
                        h.rotations[hB].normalize();
                        h.inertiaRotations[hB] = h.rotations[hB];
                    }
                }
            }
        }
    }

private:
    // ─── Reusable scratch buffer for pre-computed contact data ──────────────────────────────
    // Avoids per-frame allocation for the pre-computed scratch array.
    mutable std::vector<ContactScratch> m_scratch;  // Reused across solve() calls.

    // ─── Per-contact impulse data (re-computed each frame, cached on stack) ─────────────────

    struct ContactSolverData {
        // Body indices.
        RigidBodyHandle bodyA;
        RigidBodyHandle bodyB;

        // World-space vectors.
        Vec3f rA;          // contact - COM_A
        Vec3f rB;          // contact - COM_B
        Vec3f normal;      // from A → B (normalised)

        // Tangent basis (2 vectors perpendicular to normal).
        Vec3f tangent[2];

        // Effective masses.
        float normalMass;
        float tangentMass[2];

        // Accumulated impulses (for clamping).
        float accumulatedNormalImpulse;
        float accumulatedTangentImpulse[2];

        // Constraint bias (Baumgarte penetration correction + restitution).
        float bias;

        // Friction coefficient.
        float friction;

        // Penetration (used for bias).
        float penetration;
    };

    // ─── Warm starting ──────────────────────────────────────────────────────────────────────────

    void warmStart(std::vector<ContactManifold>& manifolds,
                   RigidBodyHotSpan h,
                   const SolverConfig& config) const noexcept {

        for (auto& manifold : manifolds) {
            for (int p = 0; p < manifold.pointCount; ++p) {
                auto& pt = manifold.points[p];
                if (pt.normalImpulse == 0.0f &&
                    pt.tangentImpulse[0] == 0.0f &&
                    pt.tangentImpulse[1] == 0.0f) continue;

                RigidBodyHandle hA = manifold.bodyA;
                RigidBodyHandle hB = manifold.bodyB;

                // Compute rA, rB.
                Vec3f rA = pt.position - h.positions[hA];
                Vec3f rB = pt.position - h.positions[hB];

                // Apply scaled-down cached impulses.
                float scale = config.warmStartFactor;
                float nImp  = pt.normalImpulse * scale;
                Vec3f impulse = pt.normal * nImp;
                impulse += pt.tangent[0] * pt.tangentImpulse[0] * scale;
                impulse += pt.tangent[1] * pt.tangentImpulse[1] * scale;

                applyImpulse(h, hA, hB, impulse, rA, rB);
            }
        }
    }

    // ─── Solve one manifold ─────────────────────────────────────────────────────────────────────

    /// @brief  Solve one manifold.  Returns the sum of absolute impulse changes for convergence tracking.
    float solveManifold(float dt,
                        ContactManifold& manifold,
                        RigidBodyHotSpan h,
                        const SolverConfig& config) const noexcept {

        RigidBodyHandle hA = manifold.bodyA;
        RigidBodyHandle hB = manifold.bodyB;

        float invMassA = h.inverseMasses[hA];
        float invMassB = h.inverseMasses[hB];
        Vec3f invInertiaA = h.inverseInertiaDiag[hA];
        Vec3f invInertiaB = h.inverseInertiaDiag[hB];
        Quaternionf inertiaRotA = h.inertiaRotations[hA];
        Quaternionf inertiaRotB = h.inertiaRotations[hB];

        float totalChange = 0.0f;

        for (int p = 0; p < manifold.pointCount; ++p) {
            auto& pt = manifold.points[p];

            Vec3f rA = pt.position - h.positions[hA];
            Vec3f rB = pt.position - h.positions[hB];
            Vec3f n  = pt.normal;

            // Reuse cached tangent basis when available.
            // Tangents depend only on the normal, which is constant for the contact
            // lifetime.  The narrowphase sets them once; subsequent solver iterations
            // read them directly — saving 2 cross products + 1 normalize per point
            // per iteration (~30% of solver ALU).
            Vec3f t1, t2;
            if (pt.tangent[0].lengthSquared() < 0.5f) {
                buildTangentBasis(n, t1, t2);
                pt.tangent[0] = t1;
                pt.tangent[1] = t2;
            } else {
                t1 = pt.tangent[0];
                t2 = pt.tangent[1];
            }

            // Compute effective masses.
            float nM = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                     inertiaRotA, inertiaRotB, rA, rB, n);
            float t1M = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                      inertiaRotA, inertiaRotB, rA, rB, t1);
            float t2M = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                      inertiaRotA, inertiaRotB, rA, rB, t2);

            // Relative velocity at contact.
            Vec3f vA = h.linearVelocities[hA];
            Vec3f vB = h.linearVelocities[hB];
            Vec3f wA = h.angularVelocities[hA];
            Vec3f wB = h.angularVelocities[hB];

            Vec3f vRel = (vB + wB.cross(rB)) - (vA + wA.cross(rA));
            float vn = vRel.dot(n);

            // ─── Normal impulse ─────────────────────────────────────────────────────────────
            // Bias = Baumgarte penetration correction (+ restitution on first solver pass).
            //
            // Baumgarte bias is NEGATIVE:  we want the constraint vn + bias >= 0 to produce
            // a separating velocity vn >= -bias = baumgarte * pen / dt > 0.  This pushes
            // the bodies apart at a rate proportional to the penetration depth.
            //
            // Reference: Box2D / Erin Catto uses  bias = -baumgarte * penetration * inv_dt.
            float pen = std::max(0.0f, pt.penetration);
            float bias = -config.baumgarte * std::min(pen, config.maxPenetrationCorrection) / dt;

            // Restitution: only on the first pass (when vn is most negative).
            if (vn < -config.restitutionThreshold) {
                bias -= manifold.combinedRestitution * vn;  // vn < 0 → -vn > 0 → bias more negative
            }

            float lambdaN = (-(vn + bias)) * nM;

            // Clamp: accumulated normal impulse must stay >= 0 (no pulling).
            float newN = pt.normalImpulse + lambdaN;
            if (newN < 0.0f) newN = 0.0f;
            float appliedN = newN - pt.normalImpulse;
            totalChange += std::abs(appliedN);
            pt.normalImpulse = newN;

            Vec3f impulseN = n * appliedN;
            applyImpulse(h, hA, hB, impulseN, rA, rB);

            // ─── Friction impulses ──────────────────────────────────────────────────────────
            // Recompute vRel with the new velocities.
            vA = h.linearVelocities[hA];
            vB = h.linearVelocities[hB];
            wA = h.angularVelocities[hA];
            wB = h.angularVelocities[hB];
            vRel = (vB + wB.cross(rB)) - (vA + wA.cross(rA));

            // Two friction tangent directions.
            Vec3f tangents[2] = {t1, t2};
            float tMasses[2]  = {t1M, t2M};
            float lambdaT[2];
            for (int d = 0; d < 2; ++d) {
                float vt    = vRel.dot(tangents[d]);
                float lt    = -vt * tMasses[d];

                // Accumulate and clamp to friction cone.
                float newT = pt.tangentImpulse[d] + lt;
                float maxF = manifold.combinedFriction * pt.normalImpulse;
                newT = std::max(-maxF, std::min(maxF, newT));
                float appliedT = newT - pt.tangentImpulse[d];
                totalChange += std::abs(appliedT);
                pt.tangentImpulse[d] = newT;
                lambdaT[d] = appliedT;
            }

            Vec3f impulseF = t1 * lambdaT[0] + t2 * lambdaT[1];
            applyImpulse(h, hA, hB, impulseF, rA, rB);

            // Cache tangents for warm start.
            pt.tangent[0] = t1;
            pt.tangent[1] = t2;
        }
        return totalChange;
    }

    // ─── Helpers ────────────────────────────────────────────────────────────────────────────────

    /// @brief  Build an orthonormal basis from a single normal.
    static void buildTangentBasis(const Vec3f& n, Vec3f& t1, Vec3f& t2) noexcept {
        // Choose the axis most perpendicular to n for cross product.
        Vec3f ref = (std::abs(n.x) < 0.9f) ? Vec3f::unitX() : Vec3f::unitY();
        t1 = n.cross(ref).normalized();
        t2 = n.cross(t1).normalized();
    }

    /// @brief  Compute the effective mass along an arbitrary direction `dir`.
    ///
    /// M_eff = 1 / ( m_A⁻¹ + m_B⁻¹ + (rA × dir) · I_A⁻¹ · (rA × dir)
    ///                                + (rB × dir) · I_B⁻¹ · (rB × dir) )
    ///
    /// The rotational term simplifies for diagonal local inertia:
    ///   (r × dir) · I_w⁻¹ · (r × dir)
    ///   = sum_k( i_local_k * (Rᵀ · (r × dir))_k² )
    static float effectiveMass(float invMassA, float invMassB,
                                const Vec3f& invInertiaA, const Vec3f& invInertiaB,
                                const Quaternionf& inertiaRotA, const Quaternionf& inertiaRotB,
                                const Vec3f& rA, const Vec3f& rB,
                                const Vec3f& dir) noexcept {
        float normalMass = invMassA + invMassB;

        // Rotational contribution from body A.
        Vec3f rCrossA = rA.cross(dir);
        Vec3f localA  = inertiaRotA.rotateInverse(rCrossA);
        normalMass += localA.x * localA.x * invInertiaA.x
                    + localA.y * localA.y * invInertiaA.y
                    + localA.z * localA.z * invInertiaA.z;

        // Rotational contribution from body B.
        Vec3f rCrossB = rB.cross(dir);
        Vec3f localB  = inertiaRotB.rotateInverse(rCrossB);
        normalMass += localB.x * localB.x * invInertiaB.x
                    + localB.y * localB.y * invInertiaB.y
                    + localB.z * localB.z * invInertiaB.z;

        // Guard against degenerate configurations (e.g. two static bodies).
        return (normalMass > 1e-10f) ? 1.0f / normalMass : 0.0f;
    }

    /// @brief  Apply an impulse (world-space linear + angular) to a body pair.
    ///
    /// delta_v_A = -impulse * invMassA
    /// delta_ω_A = -I_A⁻¹ · (rA × impulse)
    ///
    /// delta_v_B = +impulse * invMassB
    /// delta_ω_B = +I_B⁻¹ · (rB × impulse)
    static void applyImpulse(RigidBodyHotSpan h,
                              RigidBodyHandle hA, RigidBodyHandle hB,
                              const Vec3f& impulse,
                              const Vec3f& rA, const Vec3f& rB) noexcept {
        // Aligned pointers for SIMD-friendly access.
        // std::assume_aligned<16> tells the compiler it can emit aligned load/store
        // instructions (movaps/movapd) instead of the slower unaligned variants.
        float*      invMasses   = std::assume_aligned<16>(h.inverseMasses.data());
        Vec3f*      linVels     = std::assume_aligned<16>(h.linearVelocities.data());
        Vec3f*      angVels     = std::assume_aligned<16>(h.angularVelocities.data());
        Vec3f*      invInertia  = std::assume_aligned<16>(h.inverseInertiaDiag.data());
        Quaternionf* inertiaRots = std::assume_aligned<16>(h.inertiaRotations.data());
        Quaternionf* rots       = std::assume_aligned<16>(h.rotations.data());

        // Body A
        float invMA = invMasses[hA];
        if (invMA > 0.0f) {
            linVels[hA] -= impulse * invMA;
        }
        Vec3f torqueA = rA.cross(impulse);
        Vec3f localA  = inertiaRots[hA].rotateInverse(torqueA);
        Vec3f alphaA  = Vec3f(localA.x * invInertia[hA].x,
                              localA.y * invInertia[hA].y,
                              localA.z * invInertia[hA].z);
        angVels[hA] -= rots[hA].rotate(alphaA);

        // Body B
        float invMB = invMasses[hB];
        if (invMB > 0.0f) {
            linVels[hB] += impulse * invMB;
        }
        Vec3f torqueB = rB.cross(impulse);
        Vec3f localB  = inertiaRots[hB].rotateInverse(torqueB);
        Vec3f alphaB  = Vec3f(localB.x * invInertia[hB].x,
                              localB.y * invInertia[hB].y,
                              localB.z * invInertia[hB].z);
        angVels[hB] += rots[hB].rotate(alphaB);
    }
};

} // namespace ciphyxs
