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
    ///         Damping is now handled by PhysicsWorld using the Rayleigh model (F = -c·v)
    ///         during the force integration phase.  These fields are retained for API compat
    ///         but are no longer read by the solver.
    float linearDamping  = 0.0f;

    /// @brief  [DEPRECATED] Angular velocity damping factor.
    float angularDamping = 0.0f;
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
    void solve(float dt,
               std::vector<ContactManifold>& manifolds,
               RigidBodyStorage& bodies,
               const SolverConfig& config) const noexcept {

        if (manifolds.empty()) return;

        // Phase 1: Warm starting.
        if (config.enableWarmStart) {
            warmStart(manifolds, bodies, config);
        }

        // Phase 2: Main solver loop.
        for (std::uint32_t iter = 0; iter < config.numIterations; ++iter) {
            for (auto& manifold : manifolds) {
                solveManifold(dt, manifold, bodies, config);
            }
        }

        // Phase 3: Split-impulse positional correction (improves stacking stability).
        positionalSolve(bodies, manifolds, config);
    }

    /// @brief  Split-impulse positional correction to resolve residual interpenetration.
    ///
    /// After the velocity-impulse solver, this pass applies direct position corrections
    /// along contact normals to push bodies apart.  Unlike Baumgarte velocity bias, these
    /// corrections do NOT affect velocities for the next step — the correction is applied
    /// directly to positions and rotations only.
    ///
    /// A small penetration slop is allowed to prevent jittering.
    void positionalSolve(RigidBodyStorage& bodies,
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

                float invMassA = bodies.inverseMasses[hA];
                float invMassB = bodies.inverseMasses[hB];
                Vec3f invInertiaA = bodies.inverseInertiaDiag[hA];
                Vec3f invInertiaB = bodies.inverseInertiaDiag[hB];
                Quaternionf inertiaRotA = bodies.inertiaRotations[hA];
                Quaternionf inertiaRotB = bodies.inertiaRotations[hB];

                for (int p = 0; p < manifold.pointCount; ++p) {
                    auto& pt = manifold.points[p];
                    if (pt.penetration <= kSlop) continue;

                    Vec3f rA = pt.position - bodies.positions[hA];
                    Vec3f rB = pt.position - bodies.positions[hB];
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
                        bodies.positions[hA] -= corrVec * invMassA;
                    }
                    if (invMassB > 0.0f) {
                        bodies.positions[hB] += corrVec * invMassB;
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
                                        * bodies.rotations[hA];
                        dqA.w *= 0.5f; dqA.x *= 0.5f; dqA.y *= 0.5f; dqA.z *= 0.5f;
                        bodies.rotations[hA].w += dqA.w;
                        bodies.rotations[hA].x += dqA.x;
                        bodies.rotations[hA].y += dqA.y;
                        bodies.rotations[hA].z += dqA.z;
                        bodies.rotations[hA].normalize();
                        bodies.inertiaRotations[hA] = bodies.rotations[hA];
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
                                        * bodies.rotations[hB];
                        dqB.w *= 0.5f; dqB.x *= 0.5f; dqB.y *= 0.5f; dqB.z *= 0.5f;
                        bodies.rotations[hB].w += dqB.w;
                        bodies.rotations[hB].x += dqB.x;
                        bodies.rotations[hB].y += dqB.y;
                        bodies.rotations[hB].z += dqB.z;
                        bodies.rotations[hB].normalize();
                        bodies.inertiaRotations[hB] = bodies.rotations[hB];
                    }
                }
            }
        }
    }

private:
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
                   RigidBodyStorage& bodies,
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
                Vec3f rA = pt.position - bodies.positions[hA];
                Vec3f rB = pt.position - bodies.positions[hB];

                // Apply scaled-down cached impulses.
                float scale = config.warmStartFactor;
                float nImp  = pt.normalImpulse * scale;
                Vec3f impulse = pt.normal * nImp;
                impulse += pt.tangent[0] * pt.tangentImpulse[0] * scale;
                impulse += pt.tangent[1] * pt.tangentImpulse[1] * scale;

                applyImpulse(bodies, hA, hB, impulse, rA, rB);
            }
        }
    }

    // ─── Solve one manifold ─────────────────────────────────────────────────────────────────────

    void solveManifold(float dt,
                       ContactManifold& manifold,
                       RigidBodyStorage& bodies,
                       const SolverConfig& config) const noexcept {

        RigidBodyHandle hA = manifold.bodyA;
        RigidBodyHandle hB = manifold.bodyB;

        float invMassA = bodies.inverseMasses[hA];
        float invMassB = bodies.inverseMasses[hB];
        Vec3f invInertiaA = bodies.inverseInertiaDiag[hA];
        Vec3f invInertiaB = bodies.inverseInertiaDiag[hB];
        Quaternionf inertiaRotA = bodies.inertiaRotations[hA];
        Quaternionf inertiaRotB = bodies.inertiaRotations[hB];

        for (int p = 0; p < manifold.pointCount; ++p) {
            auto& pt = manifold.points[p];

            Vec3f rA = pt.position - bodies.positions[hA];
            Vec3f rB = pt.position - bodies.positions[hB];
            Vec3f n  = pt.normal;

            // Build tangent basis from normal.
            Vec3f t1, t2;
            buildTangentBasis(n, t1, t2);

            // Compute effective masses.
            float nM = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                     inertiaRotA, inertiaRotB, rA, rB, n);
            float t1M = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                      inertiaRotA, inertiaRotB, rA, rB, t1);
            float t2M = effectiveMass(invMassA, invMassB, invInertiaA, invInertiaB,
                                      inertiaRotA, inertiaRotB, rA, rB, t2);

            // Relative velocity at contact.
            Vec3f vA = bodies.linearVelocities[hA];
            Vec3f vB = bodies.linearVelocities[hB];
            Vec3f wA = bodies.angularVelocities[hA];
            Vec3f wB = bodies.angularVelocities[hB];

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
            pt.normalImpulse = newN;

            Vec3f impulseN = n * appliedN;
            applyImpulse(bodies, hA, hB, impulseN, rA, rB);

            // ─── Friction impulses ──────────────────────────────────────────────────────────
            // Recompute vRel with the new velocities.
            vA = bodies.linearVelocities[hA];
            vB = bodies.linearVelocities[hB];
            wA = bodies.angularVelocities[hA];
            wB = bodies.angularVelocities[hB];
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
                pt.tangentImpulse[d] = newT;
                lambdaT[d] = appliedT;
            }

            Vec3f impulseF = t1 * lambdaT[0] + t2 * lambdaT[1];
            applyImpulse(bodies, hA, hB, impulseF, rA, rB);

            // Cache tangents for warm start.
            pt.tangent[0] = t1;
            pt.tangent[1] = t2;
        }
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
    static void applyImpulse(RigidBodyStorage& bodies,
                              RigidBodyHandle hA, RigidBodyHandle hB,
                              const Vec3f& impulse,
                              const Vec3f& rA, const Vec3f& rB) noexcept {
        // Body A
        float invMA = bodies.inverseMasses[hA];
        if (invMA > 0.0f) {
            bodies.linearVelocities[hA] -= impulse * invMA;
        }
        Vec3f torqueA = rA.cross(impulse);
        Vec3f localA  = bodies.inertiaRotations[hA].rotateInverse(torqueA);
        Vec3f alphaA  = Vec3f(localA.x * bodies.inverseInertiaDiag[hA].x,
                              localA.y * bodies.inverseInertiaDiag[hA].y,
                              localA.z * bodies.inverseInertiaDiag[hA].z);
        bodies.angularVelocities[hA] -= bodies.rotations[hA].rotate(alphaA);

        // Body B
        float invMB = bodies.inverseMasses[hB];
        if (invMB > 0.0f) {
            bodies.linearVelocities[hB] += impulse * invMB;
        }
        Vec3f torqueB = rB.cross(impulse);
        Vec3f localB  = bodies.inertiaRotations[hB].rotateInverse(torqueB);
        Vec3f alphaB  = Vec3f(localB.x * bodies.inverseInertiaDiag[hB].x,
                              localB.y * bodies.inverseInertiaDiag[hB].y,
                              localB.z * bodies.inverseInertiaDiag[hB].z);
        bodies.angularVelocities[hB] += bodies.rotations[hB].rotate(alphaB);
    }
};

} // namespace ciphyxs
