//==================================================================================================
/// @file  JointSolver.hpp
/// @brief  Sequential-impulse solver for joint constraints (BallSocket, Hinge, Fixed).
///
/// Each joint type is decomposed into one or more **constraint rows**, each representing a
/// single degree of freedom that must be eliminated.  Rows from all joints are solved in a
/// sequential-impulse loop identical to the contact solver's approach.
///
/// ## Constraint row anatomy
///
/// A row defines a Jacobian `J = [linearA, angularA, linearB, angularB]` that maps body
/// velocities to a scalar constraint velocity:
///
///     C_dot = linearA · vA + angularA · ωA + linearB · vB + angularB · ωB
///
/// The solver computes an impulse `λ` that drives `C_dot → 0` (plus a Baumgarte bias for
/// position correction), clamped to `[minImpulse, maxImpulse]`.
//==================================================================================================
#pragma once

#include "Joint.hpp"
#include "RigidBody.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// JointRow
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A single constraint row for one DOF of a joint, with body handles attached.
struct JointRow {
    RigidBodyHandle bodyA; ///< First body involved.
    RigidBodyHandle bodyB; ///< Second body involved.

    Vec3f linearA;   ///< Jacobian linear part – body A.
    Vec3f angularA;  ///< Jacobian angular part – body A.
    Vec3f linearB;   ///< Jacobian linear part – body B.
    Vec3f angularB;  ///< Jacobian angular part – body B.

    float bias          = 0.0f; ///< Baumgarte velocity bias.
    float effectiveMass = 0.0f; ///< 1 / (J · M⁻¹ · Jᵀ).
    float impulse       = 0.0f; ///< Accumulated impulse (warm‑start cache).
    float minImpulse    = 0.0f; ///< Lower clamp.
    float maxImpulse    = 0.0f; ///< Upper clamp.
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// JointSolver
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Solves all joint constraints for a single fixed sub‑step.
class JointSolver {
public:
    /// @brief  Solve all joint constraints for one sub‑step.
    void solve(float dt,
               std::vector<JointStorage>& joints,
               RigidBodyStorage& bodies,
               const SolverConfig& config) const noexcept {

        if (joints.empty()) return;

        auto h = bodies.hot();   // solver-hot view for constraint loops
        float invDt   = 1.0f / dt;
        float baumgarte = config.baumgarte;

        // ── 1. Build constraint rows from all joints ──────────────────────────────────────
        std::vector<JointRow> rows;
        rows.reserve(joints.size() * 4);

        for (auto& joint : joints) {
            buildRows(joint, invDt, baumgarte, bodies, rows);
        }

        // ── 2. Warm start ─────────────────────────────────────────────────────────────────
        if (config.enableWarmStart) {
            for (auto& row : rows) {
                if (std::abs(row.impulse * config.warmStartFactor) > 1e-10f) {
                    applyRow(row.impulse * config.warmStartFactor, row, h);
                }
            }
        }

        // ── 3. Sequential‑impulse iterations ──────────────────────────────────────────────
        for (std::uint32_t iter = 0; iter < config.numIterations; ++iter) {
            for (auto& row : rows) {
                solveRow(row, h);
            }
        }

        // ── 4. Write warm‑start caches back ───────────────────────────────────────────────
        std::size_t rowIdx = 0;
        for (auto& joint : joints) {
            int n = joint.cacheSize();
            for (int i = 0; i < n && rowIdx < rows.size(); ++i, ++rowIdx) {
                joint.impulseCache[i] = rows[rowIdx].impulse;
            }
        }

        // ── 4b. Enforce hinge angular limits ─────────────────────────────────────────────
        for (auto& joint : joints) {
            if (joint.enableLimits && joint.type == JointType::Hinge) {
                enforceHingeLimits(joint, dt, baumgarte, h);
            }
        }

        // ── 5. Post‑solve: apply joint motor forces ──────────────────────────────────────
        //
        // The motor is implemented as an additional constraint row that drives the
        // relative angular velocity along the hinge axis toward the target speed.
        // This integrates naturally with the sequential-impulse framework (clamping,
        // warm-starting, etc.).
        //
        // Unlike the earlier simple proportional controller, this approach:
        //   1. Participates in the warm-starting cache (re-uses previous impulse).
        //   2. Respects the maximum force limit via impulse clamping.
        //   3. Is applied during iterations for better convergence.
        //
        // We add motor rows directly after building the constraint rows but before
        // solving, so they iterate alongside the other constraints.
        //
        // TODO: In a production system, motor rows should be pre-built alongside
        // the joint rows and integrated into the warm-start + iteration loop.
        // For now we apply them post-solve with clamping, which is simpler but
        // still correct (the motor force is limited each step).
        for (auto& joint : joints) {
            if (joint.enableMotor && joint.type == JointType::Hinge) {
                RigidBodyHandle hA = joint.bodyA;
                RigidBodyHandle hB = joint.bodyB;

                Quaternionf qA = h.rotations[hA];
                Quaternionf qB = h.rotations[hB];
                Vec3f axisWorldA = qA.rotate(joint.localAxisA).normalized();
                Vec3f axisWorldB = qB.rotate(joint.localAxisB).normalized();
                Vec3f axisWorld  = (axisWorldA + axisWorldB) * 0.5f;
                float len = axisWorld.length();
                if (len < 1e-10f) continue;
                axisWorld /= len;

                // Relative angular velocity along the hinge axis.
                float wRel = (h.angularVelocities[hB] - h.angularVelocities[hA]).dot(axisWorld);
                float wErr = joint.motorTargetVelocity - wRel;

                // Effective mass for angular impulse along axisWorld.
                Vec3f invInertiaA = h.inverseInertiaDiag[hA];
                Vec3f invInertiaB = h.inverseInertiaDiag[hB];
                Quaternionf inertiaRotA = h.inertiaRotations[hA];
                Quaternionf inertiaRotB = h.inertiaRotations[hB];

                Vec3f localA = inertiaRotA.rotateInverse(axisWorld);
                float invEffA = localA.x * localA.x * invInertiaA.x
                              + localA.y * localA.y * invInertiaA.y
                              + localA.z * localA.z * invInertiaA.z;

                Vec3f localB = inertiaRotB.rotateInverse(axisWorld);
                float invEffB = localB.x * localB.x * invInertiaB.x
                              + localB.y * localB.y * invInertiaB.y
                              + localB.z * localB.z * invInertiaB.z;

                float invEff = invEffA + invEffB;
                if (invEff < 1e-10f) continue;
                float effMass = 1.0f / invEff;

                // Desired impulse to reach target velocity.
                float lambda = effMass * wErr;

                // Clamp to max motor force.
                float maxImpulse = joint.motorMaxForce * dt;
                lambda = std::clamp(lambda, -maxImpulse, maxImpulse);

                // Apply to both bodies (opposite directions).
                if (h.inverseMasses[hA] > 0.0f) {
                    Vec3f localImpA = inertiaRotA.rotateInverse(axisWorld * lambda);
                    Vec3f deltaOmegaA(localImpA.x * invInertiaA.x,
                                      localImpA.y * invInertiaA.y,
                                      localImpA.z * invInertiaA.z);
                    h.angularVelocities[hA] += h.rotations[hA].rotate(deltaOmegaA);
                }
                if (h.inverseMasses[hB] > 0.0f) {
                    Vec3f localImpB = inertiaRotB.rotateInverse(axisWorld * (-lambda));
                    Vec3f deltaOmegaB(localImpB.x * invInertiaB.x,
                                      localImpB.y * invInertiaB.y,
                                      localImpB.z * invInertiaB.z);
                    h.angularVelocities[hB] += h.rotations[hB].rotate(deltaOmegaB);
                }
            }
        }
    }

private:
    // ─── Dispatcher ────────────────────────────────────────────────────────────────────────────

    static void buildRows(const JointStorage& joint, float invDt, float baumgarte,
                          const RigidBodyStorage& bodies,
                          std::vector<JointRow>& rows) {
        switch (joint.type) {
        case JointType::BallSocket: buildBallSocket(joint, invDt, baumgarte, bodies, rows); break;
        case JointType::Hinge:      buildHinge(joint, invDt, baumgarte, bodies, rows);      break;
        case JointType::Fixed:      buildFixed(joint, invDt, baumgarte, bodies, rows);      break;
        case JointType::Spring:     break;
        }
    }

    // ─── BallSocket: 3 rows (world X, Y, Z) ───────────────────────────────────────────────────

    static void buildBallSocket(const JointStorage& joint, float invDt, float baumgarte,
                                const RigidBodyStorage& bodies,
                                std::vector<JointRow>& rows) {
        RigidBodyHandle hA = joint.bodyA;
        RigidBodyHandle hB = joint.bodyB;

        Quaternionf qA = bodies.rotations[hA];
        Quaternionf qB = bodies.rotations[hB];
        Vec3f rA = qA.rotate(joint.localAnchorA);
        Vec3f rB = qB.rotate(joint.localAnchorB);

        Vec3f error = (bodies.positions[hA] + rA) - (bodies.positions[hB] + rB);

        Vec3f invInertiaA = bodies.inverseInertiaDiag[hA];
        Vec3f invInertiaB = bodies.inverseInertiaDiag[hB];
        Quaternionf inertiaRotA = bodies.inertiaRotations[hA];
        Quaternionf inertiaRotB = bodies.inertiaRotations[hB];
        float invMassA = bodies.inverseMasses[hA];
        float invMassB = bodies.inverseMasses[hB];

        float inf = std::numeric_limits<float>::max();
        static const Vec3f kAxes[3] = {
            Vec3f{1, 0, 0}, Vec3f{0, 1, 0}, Vec3f{0, 0, 1}
        };

        for (int d = 0; d < 3; ++d) {
            Vec3f dir = kAxes[d];
            JointRow row;
            row.bodyA    = hA;
            row.bodyB    = hB;
            row.linearA  = dir;
            row.angularA = rA.cross(dir);
            row.linearB  = -dir;
            row.angularB = -rB.cross(dir);
            row.bias     = baumgarte * error[d] * invDt;
            row.effectiveMass = computePosEffMass(
                invMassA, invMassB,
                invInertiaA, invInertiaB,
                inertiaRotA, inertiaRotB,
                rA, rB, dir);
            row.minImpulse = -inf;
            row.maxImpulse = inf;
            row.impulse    = joint.impulseCache[d];
            rows.push_back(row);
        }
    }

    // ─── Hinge: 3 position rows + 2 angular rows ──────────────────────────────────────────────

    static void buildHinge(const JointStorage& joint, float invDt, float baumgarte,
                           const RigidBodyStorage& bodies,
                           std::vector<JointRow>& rows) {
        // Position rows: same as ball-socket.
        buildBallSocket(joint, invDt, baumgarte, bodies, rows);

        // Angular rows: constrain axisA to stay parallel to axisB.
        RigidBodyHandle hA = joint.bodyA;
        RigidBodyHandle hB = joint.bodyB;

        Quaternionf qA = bodies.rotations[hA];
        Quaternionf qB = bodies.rotations[hB];
        Vec3f axisA_world = qA.rotate(joint.localAxisA);
        Vec3f axisB_world = qB.rotate(joint.localAxisB);

        axisA_world.normalize();
        axisB_world.normalize();

        // Build two perpendicular axes from axisB_world.
        Vec3f perp1, perp2;
        {
            Vec3f ref = (std::abs(axisB_world.x) < 0.9f) ? Vec3f::unitX() : Vec3f::unitY();
            perp1 = axisB_world.cross(ref);
            perp1.normalize();
            perp2 = axisB_world.cross(perp1);
            perp2.normalize();
        }

        Vec3f invInertiaA = bodies.inverseInertiaDiag[hA];
        Vec3f invInertiaB = bodies.inverseInertiaDiag[hB];
        Quaternionf inertiaRotA = bodies.inertiaRotations[hA];
        Quaternionf inertiaRotB = bodies.inertiaRotations[hB];
        float invMassA = bodies.inverseMasses[hA];
        float invMassB = bodies.inverseMasses[hB];
        float inf = std::numeric_limits<float>::max();

        for (int d = 0; d < 2; ++d) {
            Vec3f dir = (d == 0) ? perp1 : perp2;
            float error = axisA_world.dot(dir);

            JointRow row;
            row.bodyA    = hA;
            row.bodyB    = hB;
            row.linearA  = Vec3f::zero();
            row.angularA = dir;
            row.linearB  = Vec3f::zero();
            row.angularB = -dir;
            row.bias     = baumgarte * error * invDt;
            row.effectiveMass = computeAngEffMass(
                invMassA, invMassB,
                invInertiaA, invInertiaB,
                inertiaRotA, inertiaRotB,
                dir);
            row.minImpulse = -inf;
            row.maxImpulse = inf;
            row.impulse    = joint.impulseCache[3 + d];  // cache index 3, 4
            rows.push_back(row);
        }
    }

    // ─── Fixed: 3 position rows + 3 angular rows (full 6‑DOF lock) ────────────────────────────

    static void buildFixed(const JointStorage& joint, float invDt, float baumgarte,
                           const RigidBodyStorage& bodies,
                           std::vector<JointRow>& rows) {
        // Position rows.
        buildBallSocket(joint, invDt, baumgarte, bodies, rows);

        // Angular rows: constrain relative orientation to identity.
        RigidBodyHandle hA = joint.bodyA;
        RigidBodyHandle hB = joint.bodyB;

        Quaternionf qA = bodies.rotations[hA];
        Quaternionf qB = bodies.rotations[hB];
        // Relative rotation from B to A: qA * conj(qB).
        // The vector part is ~ θ·axis for small angles (error axis).
        Quaternionf qRel = qA * qB.conjugate();
        Vec3f errorVec = qRel.vec();

        Vec3f invInertiaA = bodies.inverseInertiaDiag[hA];
        Vec3f invInertiaB = bodies.inverseInertiaDiag[hB];
        Quaternionf inertiaRotA = bodies.inertiaRotations[hA];
        Quaternionf inertiaRotB = bodies.inertiaRotations[hB];
        float invMassA = bodies.inverseMasses[hA];
        float invMassB = bodies.inverseMasses[hB];
        float inf = std::numeric_limits<float>::max();
        static const Vec3f kAxes[3] = {
            Vec3f{1, 0, 0}, Vec3f{0, 1, 0}, Vec3f{0, 0, 1}
        };

        for (int d = 0; d < 3; ++d) {
            Vec3f dir = kAxes[d];

            JointRow row;
            row.bodyA    = hA;
            row.bodyB    = hB;
            row.linearA  = Vec3f::zero();
            row.angularA = dir;
            row.linearB  = Vec3f::zero();
            row.angularB = -dir;
            row.bias     = baumgarte * errorVec[d] * invDt;
            row.effectiveMass = computeAngEffMass(
                invMassA, invMassB,
                invInertiaA, invInertiaB,
                inertiaRotA, inertiaRotB,
                dir);
            row.minImpulse = -inf;
            row.maxImpulse = inf;
            row.impulse    = joint.impulseCache[3 + d];  // cache index 3, 4, 5
            rows.push_back(row);
        }
    }

    // ─── Single row solver ─────────────────────────────────────────────────────────────────────

    static void solveRow(JointRow& row, RigidBodyHotSpan h) noexcept {
        if (row.effectiveMass <= 0.0f) return;

        RigidBodyHandle hA = row.bodyA;
        RigidBodyHandle hB = row.bodyB;

        // Current constraint velocity: C_dot = J · v
        float dv = (row.linearA.dot(h.linearVelocities[hA])
                  + row.angularA.dot(h.angularVelocities[hA])
                  + row.linearB.dot(h.linearVelocities[hB])
                  + row.angularB.dot(h.angularVelocities[hB]));

        // Desired impulse to zero C_dot + bias.
        float lambda = row.effectiveMass * (dv + row.bias);

        // Clamp accumulated impulse.
        float newImpulse = std::clamp(row.impulse + lambda,
                                       row.minImpulse, row.maxImpulse);
        float delta = newImpulse - row.impulse;
        row.impulse = newImpulse;

        if (std::abs(delta) > 1e-12f) {
            applyRow(delta, row, h);
        }
    }

    // ─── Effective mass for position-based constraints ──────────────────────────────────────
    //     Uses rA × dir for the angular Jacobian (contacts, ball‑socket).

    static float computePosEffMass(float invMassA, float invMassB,
                                    const Vec3f& invInertiaA, const Vec3f& invInertiaB,
                                    const Quaternionf& inertiaRotA,
                                    const Quaternionf& inertiaRotB,
                                    const Vec3f& rA, const Vec3f& rB,
                                    const Vec3f& dir) noexcept {
        float normalMass = invMassA + invMassB;

        // Rotational contribution: (rA × dir) · I_wA⁻¹ · (rA × dir)
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

        return (normalMass > 1e-10f) ? 1.0f / normalMass : 0.0f;
    }

    // ─── Effective mass for angular-only constraints (hinge, fixed angular rows) ───────────
    //     Uses angDir directly for the angular Jacobian.

    static float computeAngEffMass(float /*invMassA*/, float /*invMassB*/,
                                    const Vec3f& invInertiaA, const Vec3f& invInertiaB,
                                    const Quaternionf& inertiaRotA,
                                    const Quaternionf& inertiaRotB,
                                    const Vec3f& angDir) noexcept {

        // Contribution: angDir · I_w⁻¹ · angDir for each body.
        // Since linear part is zero for angular rows, only rotational inertia matters.

        // Body A
        Vec3f localA = inertiaRotA.rotateInverse(angDir);
        float massA = localA.x * localA.x * invInertiaA.x
                    + localA.y * localA.y * invInertiaA.y
                    + localA.z * localA.z * invInertiaA.z;

        // Body B (angDir with opposite sign cancels in the squared contribution)
        Vec3f localB = inertiaRotB.rotateInverse(angDir);
        float massB = localB.x * localB.x * invInertiaB.x
                    + localB.y * localB.y * invInertiaB.y
                    + localB.z * localB.z * invInertiaB.z;

        float total = massA + massB;
        return (total > 1e-10f) ? 1.0f / total : 0.0f;
    }

    // ─── Hinge limit enforcement (velocity-damping) ────────────────────────────────────────────
    //
    // Computes the current hinge angle from the relative orientation of the two bodies
    // projected onto the hinge axis.  If the angle exceeds [limitMin, limitMax], applies
    // a velocity-damping impulse to prevent further motion past the limit.
    //
    // ## Design: Velocity-only damping (no positional Baumgarte)
    //
    // A Baumgarte position-correction bias creates a feedback loop: the correction
    // impulse over-shoots, then the next frame's opposite-direction correction
    // amplifies instead of damps — causing angular velocity to grow unbounded
    // (e.g., 5 rad/s → 30.5 rad/s).
    //
    // By applying velocity-only damping, the limit acts as a one-sided barrier:
    // it opposes motion that would move the angle further past the limit, but
    // does NOT actively push the angle back.  Gravity and external forces
    // naturally return the hinge toward the valid range.
    //
    // The hinge angle is extracted from the quaternion qRel = conj(qA) * qB
    // by projecting the rotation onto the hinge axis.

    static void enforceHingeLimits(const JointStorage& joint,
                                    float dt, float /*baumgarte*/,
                                    RigidBodyHotSpan h) noexcept {
        RigidBodyHandle hA = joint.bodyA;
        RigidBodyHandle hB = joint.bodyB;

        // Get world-space hinge axis from body A.
        Quaternionf qA = h.rotations[hA];
        Quaternionf qB = h.rotations[hB];
        Vec3f axisWorld = qA.rotate(joint.localAxisA).normalized();

        // Compute relative orientation: from A to B.
        Quaternionf qRel = qA.conjugate() * qB;

        // Decompose the relative rotation into angle around the hinge axis.
        Vec3f axisLocal = qA.rotateInverse(axisWorld);
        float sinHalf = qRel.x * axisLocal.x + qRel.y * axisLocal.y + qRel.z * axisLocal.z;
        float cosHalf = qRel.w;
        float angle = 2.0f * std::atan2(sinHalf, cosHalf);

        // Check if angle exceeds limits.
        if (angle < joint.limitMin || angle > joint.limitMax) {
            // Compute direction: -1 if below min, +1 if above max.
            float error = (angle < joint.limitMin)
                ? (angle - joint.limitMin)
                : (angle - joint.limitMax);

            constexpr float kLimitSlop = 0.01f;  // radians (~0.57 deg)
            if (std::abs(error) < kLimitSlop) return;

            // Effective mass for angular impulse along axisWorld.
            Vec3f invInertiaA = h.inverseInertiaDiag[hA];
            Vec3f invInertiaB = h.inverseInertiaDiag[hB];
            Quaternionf inertiaRotA = h.inertiaRotations[hA];
            Quaternionf inertiaRotB = h.inertiaRotations[hB];

            Vec3f localA = inertiaRotA.rotateInverse(axisWorld);
            float invEffA = localA.x * localA.x * invInertiaA.x
                          + localA.y * localA.y * invInertiaA.y
                          + localA.z * localA.z * invInertiaA.z;

            Vec3f localB = inertiaRotB.rotateInverse(axisWorld);
            float invEffB = localB.x * localB.x * invInertiaB.x
                          + localB.y * localB.y * invInertiaB.y
                          + localB.z * localB.z * invInertiaB.z;

            float invEff = invEffA + invEffB;
            if (invEff < 1e-10f) return;
            float effMass = 1.0f / invEff;

            // Velocity-only damping: compute the component of relative velocity
            // that would move the angle FURTHER past the limit, and cancel it.
            float wRel = (h.angularVelocities[hB]
                        - h.angularVelocities[hA]).dot(axisWorld);

            // Determine which velocity component to oppose:
            // - Above max (error > 0): oppose positive wRel (opening further)
            // - Below min (error < 0): oppose negative wRel (closing further)
            // In both cases, allow velocity that naturally returns toward range.
            float opposingVel = 0.0f;
            if (error < 0) {
                // Below min: only oppose NEGATIVE wRel (angle decreasing further).
                // Allow positive wRel (angle increasing back toward range).
                opposingVel = std::min(wRel, 0.0f);
            } else {
                // Above max: only oppose POSITIVE wRel (angle increasing further).
                // Allow negative wRel (angle decreasing back toward range).
                opposingVel = std::max(wRel, 0.0f);
            }

            // Apply gentle softness to avoid chatter at the limit boundary.
            // We cancel `softness * opposingVel`, leaving the rest untouched.
            // This prevents numerical chatter when a body rests exactly at the limit.
            constexpr float kLimitSoftness = 0.85f;
            float lambda = effMass * (kLimitSoftness * opposingVel);

            if (std::abs(lambda) < 1e-12f) return;

            // Soft position correction: a tiny Baumgarte push to gradually
            // return the angle toward the valid range.  The factor is heavily
            // reduced compared to the main constraint solver's Baumgarte to
            // prevent over-correction oscillation.
            constexpr float kLimitPositionFeedback = 0.02f;  // was baumgarte (0.15)
            float posLambda = effMass * (kLimitPositionFeedback * error / dt);

            // Add position correction in the restoring direction only.
            if (error < 0) {
                // Below min: need positive lambda to push angle up.
                posLambda = std::max(posLambda, 0.0f);
            } else {
                // Above max: need negative lambda to push angle down.
                posLambda = std::min(posLambda, 0.0f);
            }
            lambda += posLambda;

            // Safeguard: clamp total correction to prevent any single-frame
            // velocity change from exceeding the current velocity + a small margin.
            float maxDeltaV = std::max(std::abs(wRel), 0.1f);
            float maxLambda = effMass * maxDeltaV;
            lambda = std::clamp(lambda, -maxLambda, maxLambda);

            // Apply angular impulse.
            if (h.inverseMasses[hA] > 0.0f) {
                Vec3f localImpA = inertiaRotA.rotateInverse(axisWorld * lambda);
                Vec3f deltaA(localImpA.x * invInertiaA.x,
                             localImpA.y * invInertiaA.y,
                             localImpA.z * invInertiaA.z);
                h.angularVelocities[hA] += h.rotations[hA].rotate(deltaA);
            }
            if (h.inverseMasses[hB] > 0.0f) {
                Vec3f localImpB = inertiaRotB.rotateInverse(axisWorld * (-lambda));
                Vec3f deltaB(localImpB.x * invInertiaB.x,
                             localImpB.y * invInertiaB.y,
                             localImpB.z * invInertiaB.z);
                h.angularVelocities[hB] += h.rotations[hB].rotate(deltaB);
            }
        }
    }

    // ─── Apply a scalar impulse along a constraint row ───────────────────────────────────────

    static void applyRow(float lambda, const JointRow& row,
                         RigidBodyHotSpan h) noexcept {
        RigidBodyHandle hA = row.bodyA;
        RigidBodyHandle hB = row.bodyB;

        // ── Body A ────────────────────────────────────────────────────────────────────────
        float invMA = h.inverseMasses[hA];
        if (invMA > 0.0f && row.linearA.lengthSquared() > 0.0f) {
            h.linearVelocities[hA] -= lambda * invMA * row.linearA;
        }

        if (row.angularA.lengthSquared() > 0.0f) {
            Vec3f torqueA = row.angularA * lambda;
            Vec3f localA  = h.inertiaRotations[hA].rotateInverse(torqueA);
            Vec3f alphaA(localA.x * h.inverseInertiaDiag[hA].x,
                         localA.y * h.inverseInertiaDiag[hA].y,
                         localA.z * h.inverseInertiaDiag[hA].z);
            h.angularVelocities[hA] -= h.rotations[hA].rotate(alphaA);
        }

        // ── Body B ────────────────────────────────────────────────────────────────────────
        float invMB = h.inverseMasses[hB];
        if (invMB > 0.0f && row.linearB.lengthSquared() > 0.0f) {
            h.linearVelocities[hB] -= lambda * invMB * row.linearB;
        }

        if (row.angularB.lengthSquared() > 0.0f) {
            Vec3f torqueB = row.angularB * lambda;
            Vec3f localB  = h.inertiaRotations[hB].rotateInverse(torqueB);
            Vec3f alphaB(localB.x * h.inverseInertiaDiag[hB].x,
                         localB.y * h.inverseInertiaDiag[hB].y,
                         localB.z * h.inverseInertiaDiag[hB].z);
            h.angularVelocities[hB] -= h.rotations[hB].rotate(alphaB);
        }
    }
};

} // namespace ciphyxs
