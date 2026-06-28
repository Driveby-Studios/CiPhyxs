//==================================================================================================
/// @file  CharacterController.hpp
/// @brief  Kinematic character controller with sliding, step-up/down, and slope handling.
///
/// ## Design
///
/// The CharacterController manages a kinematic capsule body inside the PhysicsWorld and moves
/// it each frame using **sweep-based collision resolution** — the same technique used by PhysX
/// (PxCapsuleController) and Bullet (btKinematicCharacterController):
///
///   1. Apply gravity to vertical velocity.
///   2. Try **step-up** — sweep up, forward, then down.  If the forward sweep is clear and
///      ground is found within `stepHeight` after stepping down, accept the step.
///   3. If step-up fails, perform **sliding movement** — sweep along the velocity vector,
///      project remaining displacement onto the hit surface normal, repeat up to 4 iterations.
///   4. Update grounded state via a short downward sweep.
///   5. Write the final position back to the kinematic body so the engine's contact solver
///      correctly handles character↔dynamic interactions.
///
/// ## Usage
///
/// @code
///   CharacterController ctrl(world);
///   ctrl.setMoveDirection(Vec3f(0, 0, 1));  // walk forward
///
///   // Call each fixed step (or from an ISolverHook).
///   for (int i = 0; i < 60; ++i) {
///       ctrl.update(1.0f / 60.0f);
///       world.step(1.0f / 60.0f);   // step AFTER updating the character
///   }
/// @endcode
//==================================================================================================
#pragma once

#include "../core/PhysicsWorld.hpp"
#include "../collision/Shape.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// CharacterController::Config
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning parameters for the kinematic character controller.
struct CharacterControllerConfig {
    /// @brief  Capsule radius (m).
    float radius = 0.4f;

    /// @brief  Total capsule height including the hemispherical end-caps (m).
    float height = 1.8f;

    /// @brief  Maximum step height the character can climb (m).
    float stepHeight = 0.3f;

    /// @brief  Maximum walkable slope angle (degrees).  Slopes steeper than this cause sliding.
    float maxSlopeAngle = 50.0f;

    /// @brief  Extra clearance around the capsule for numerical robustness (m).
    float skinWidth = 0.01f;

    /// @brief  Gravity multiplier (1.0 = normal gravity).
    float gravityScale = 1.0f;

    /// @brief  Upward velocity applied on jump (m/s).
    float jumpSpeed = 5.0f;

    /// @brief  Maximum horizontal speed (m/s).
    float maxSpeed = 5.0f;

    /// @brief  Horizontal acceleration (m/s²).
    float acceleration = 12.0f;

    /// @brief  Horizontal friction deceleration (m/s²) when no input.
    float friction = 8.0f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// CharacterController
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Kinematic character controller with sweep-based collision resolution.
///
/// The controller owns a kinematic capsule body inside the PhysicsWorld.  Call `update()` each
/// **fixed timestep** (either from an `ISolverHook` or before `world.step()`).  The kinematic
/// body is moved to the computed safe position so the engine's contact solver can push dynamic
/// objects out of the character.
///
/// @note  This is a **ghost + kinematic** hybrid: the character uses swept queries to avoid
///        embedding in static geometry, while its kinematic body provides correct collision
///        response against dynamic objects via the engine's constraint solver.
class CharacterController {
public:
    /// @brief  Construct and register the character in the world.
    /// @param world  PhysicsWorld that will own the character's kinematic body and capsule shape.
    /// @param config  Tuning parameters (radius, height, step height, etc.).
    CharacterController(PhysicsWorld& world, const CharacterControllerConfig& config = {})
        : m_world(&world), m_config(config) {
        // Compute capsule half-height from the total height and radius.
        float halfHeight = (config.height * 0.5f) - config.radius;
        if (halfHeight < 0.01f) halfHeight = 0.01f;

        // Create the capsule shape in the world.
        m_capsuleShape = world.createShape(Capsule{config.radius, halfHeight});

        // Create a kinematic body for the character.
        RigidBodyDesc desc;
        desc.motionType  = MotionType::Kinematic;
        desc.setShape(m_capsuleShape);
        desc.position    = config.radius > 0.0f
            ? Vec3f(0.0f, config.height * 0.5f, 0.0f)   // stand on ground
            : Vec3f::zero();
        desc.restitution = 0.0f;
        desc.friction    = 0.0f;
        m_body = world.createBody(desc);

        // Initial position matches the body's spawn position.
        m_position  = desc.position;
        m_rotation  = desc.rotation;

        // Set up the self-filter for sweep queries.
        m_filter.self = m_body;
    }

    /// @brief  Destroy the character — removes the kinematic body and capsule shape from the world.
    ~CharacterController() {
        if (m_world && m_body != kInvalidHandle) {
            m_world->removeBody(m_body);
        }
        // Shape removal is deferred: shapes live in the world's shape registry.
        // If the world is destroyed before the controller, the shape is cleaned up automatically.
        // For explicit shape removal, the user should manage shapes externally.
    }

    // ─── Input ──────────────────────────────────────────────────────────────────────────────────

    /// @brief  Set the desired horizontal movement direction.
    /// @param direction  Normalized world-space direction (XZ components used, Y ignored).
    void setMoveDirection(const Vec3f& direction) noexcept {
        m_moveDir = Vec3f(direction.x, 0.0f, direction.z).normalized();
    }

    /// @brief  Apply a jump impulse (only effective when grounded).
    void jump() noexcept {
        if (m_onGround) {
            m_velocity.y = m_config.jumpSpeed;
        }
    }

    // ─── Per-frame update ───────────────────────────────────────────────────────────────────────

    /// @brief  Advance the character by one fixed timestep.
    ///
    /// Applies gravity, processes input (acceleration / friction), attempts step-up movement,
    /// falls back to sliding along surfaces, detects ground contact, and updates the kinematic
    /// body's position in the world.
    ///
    /// @param dt  Fixed timestep (typically 1/60 s).
    void update(float dt) {
        constexpr float kEps = 1e-8f;

        // ── 1. Apply gravity ─────────────────────────────────────────────────────────────────
        if (!m_onGround) {
            m_velocity.y -= m_config.gravityScale * 9.81f * dt;
        }

        // ── 2. Horizontal input (acceleration / friction) ──────────────────────────────────
        float moveLenSq = m_moveDir.lengthSquared();
        if (moveLenSq > 0.01f) {
            // Accelerate toward target velocity.
            Vec3f targetVel = m_moveDir * m_config.maxSpeed;
            float factor = std::min(1.0f, m_config.acceleration * dt);
            m_velocity.x += (targetVel.x - m_velocity.x) * factor;
            m_velocity.z += (targetVel.z - m_velocity.z) * factor;
        } else if (m_onGround) {
            // Friction deceleration on ground.
            float speed = std::sqrt(m_velocity.x * m_velocity.x + m_velocity.z * m_velocity.z);
            if (speed > kEps) {
                float decel = std::min(speed, m_config.friction * dt);
                float scale = (speed - decel) / speed;
                m_velocity.x *= scale;
                m_velocity.z *= scale;
            }
        }

        // ── 3. Clip vertical velocity ─────────────────────────────────────────────────────
        constexpr float kMaxFallSpeed = 100.0f;
        m_velocity.y = std::max(m_velocity.y, -kMaxFallSpeed);

        // ── 4. Compute displacement ────────────────────────────────────────────────────────
        Vec3f displacement = m_velocity * dt;

        // ── 5. Attempt step-up movement (only when grounded and not jumping) ───────────────
        Vec3f newPos = m_position;
        bool stepped = false;
        bool isJumping = m_velocity.y > 0.1f;
        if (displacement.lengthSquared() >= 1e-8f && m_onGround && !isJumping) {
            stepped = tryStepUp(displacement, newPos);
        }

        if (!stepped) {
            // ── 6. Sliding movement ────────────────────────────────────────────────────────
            newPos = slideMovement(displacement);
        }

        // ── 7. Commit movement before ground detection ─────────────────────────────────────
        m_position = newPos;

        // ── 8. Update grounded state (snaps m_position in-place, zeroes vy) ────────────────
        // Skip ground detection when jumping upward — updateGroundedState would snap the
        // character back to the ground because its sweep distance (half-height + 0.2) is larger
        // than the upward displacement per frame (~0.08).
        if (isJumping) {
            m_onGround     = false;
            m_groundNormal = Vec3f::unitY();
        } else {
            updateGroundedState();
        }

        // Update the kinematic body's transform so the engine's solver handles character↔dynamic
        // collisions correctly.
        RigidBodyStorage& bodies = m_world->bodies();
        bodies.positions[m_body] = m_position;
        bodies.rotations[m_body] = m_rotation;
        bodies.linearVelocities[m_body]  = m_velocity;
        bodies.angularVelocities[m_body] = Vec3f::zero();
    }

    // ─── Accessors ──────────────────────────────────────────────────────────────────────────────

    /// @brief  Current world-space position (centre of the capsule).
    [[nodiscard]] Vec3f getPosition() const noexcept { return m_position; }

    /// @brief  Teleport the character to a new position.
    void setPosition(const Vec3f& pos) noexcept {
        m_position = pos;
        m_velocity = Vec3f::zero();
        // Immediately update the kinematic body.
        if (m_body != kInvalidHandle) {
            m_world->bodies().positions[m_body] = m_position;
        }
    }

    /// @brief  True if the character is standing on a walkable surface.
    [[nodiscard]] bool isOnGround() const noexcept { return m_onGround; }

    /// @brief  World-space normal of the surface the character is standing on.
    [[nodiscard]] Vec3f getGroundNormal() const noexcept { return m_groundNormal; }

    /// @brief  Current velocity (m/s).
    [[nodiscard]] Vec3f getVelocity() const noexcept { return m_velocity; }

    /// @brief  Handle of the character's kinematic body (for external queries / filtering).
    [[nodiscard]] RigidBodyHandle getBody() const noexcept { return m_body; }

    /// @brief  Read-only access to the configuration.
    [[nodiscard]] const CharacterControllerConfig& getConfig() const noexcept { return m_config; }

    /// @brief  Mutable access to the configuration (adjust parameters at runtime).
    [[nodiscard]] CharacterControllerConfig& getConfig() noexcept { return m_config; }

private:
    // ─── Self-filter for sweep / overlap queries ──────────────────────────────────────────────

    /// @brief  Collision filter that excludes the character's own kinematic body.
    struct SelfFilter : ICollisionFilter {
        RigidBodyHandle self = kInvalidHandle;

        bool shouldCollide(RigidBodyHandle a, RigidBodyHandle) const noexcept override {
            return a != self;
        }
    };

    // ─── Sliding movement ─────────────────────────────────────────────────────────────────────

    /// @brief  Sweep the capsule along `displacement`, sliding on surfaces up to 4 iterations.
    ///
    /// Algorithm:
    ///   For each iteration:
    ///     1. Sweep the capsule from current position to current + remaining displacement.
    ///     2. If no hit: move full distance → done.
    ///     3. If hit: advance to the point of impact (displacement × hit.t).
    ///     4. Project remaining displacement onto the hit plane (slide).
    ///     5. Reduce remaining displacement by the advance already consumed.
    ///     6. Stop when the remaining displacement is tiny or max iterations reached.
    [[nodiscard]] Vec3f slideMovement(const Vec3f& displacement) const {
        constexpr int kMaxIter = 4;
        constexpr float kMinDisplacement = 1e-6f;

        Vec3f pos = m_position;
        Vec3f remaining = displacement;

        for (int iter = 0; iter < kMaxIter; ++iter) {
            float remainingLen = remaining.length();
            if (remainingLen < kMinDisplacement) break;

            Vec3f targetPos = pos + remaining;
            PhysicsWorld::SweepHit hit;

            Shape capsule = makeCapsuleShape();
            bool swept = m_world->sweepShape(capsule, pos, m_rotation,
                                             targetPos, m_rotation,
                                             &m_filter, hit);

            if (!swept || hit.t >= 1.0f) {
                // No obstacle — move the full remaining distance.
                pos = targetPos;
                break;
            }

            // Advance to the point of impact (with a small nudging to prevent re-collision).
            float advance = std::max(hit.t - 1e-6f, 0.0f);
            pos = pos + remaining * advance;

            // Slide: project the remaining displacement onto the hit surface plane.
            remaining = remaining * (1.0f - advance);
            Vec3f normal = hit.normal;
            float dot = remaining.dot(normal);
            remaining = remaining - normal * dot;

            // If the character is moving upward and hit a steep slope, add downward push.
            if (dot < 0.0f && remaining.y > 0.01f) {
                // Partially cancel upward movement against the slope.
                remaining.y *= 0.3f;
            }
        }

        // Clamp position to prevent falling through the floor.
        if (pos.y < -1000.0f) pos.y = m_position.y - 10.0f;

        return pos;
    }

    // ─── Step-up movement ─────────────────────────────────────────────────────────────────────

    /// @brief  Attempt to step up onto an obstacle.
    ///
    /// Sequence:
    ///   a) Sweep up by `stepHeight`.  If blocked → return false.
    ///   b) Sweep forward by the horizontal displacement.  If blocked → return false.
    ///   c) Sweep down by `stepHeight + small margin` to find ground.
    ///   d) If ground found within stepHeight and the slope is walkable → accept the step.
    [[nodiscard]] bool tryStepUp(const Vec3f& displacement, Vec3f& newPos) const {
        if (!m_onGround) return false;
        if (m_config.stepHeight < 1e-4f) return false;
        // Only attempt step-up when moving horizontally; stationary step-up would lift the
        // character off the ground every frame, causing jitter.
        float horizLenSq = displacement.x * displacement.x + displacement.z * displacement.z;
        if (horizLenSq < 1e-12f) return false;

        float stepH = m_config.stepHeight;
        Shape capsule = makeCapsuleShape();

        // a) Sweep upward.
        Vec3f upTarget = m_position + Vec3f(0.0f, stepH, 0.0f);
        PhysicsWorld::SweepHit upHit;
        bool upBlocked = m_world->sweepShape(capsule, m_position, m_rotation,
                                              upTarget, m_rotation,
                                              &m_filter, upHit);
        if (upBlocked && upHit.t < 1.0f) return false;

        // The upward sweep was clear (or nearly clear).
        float upAdvance = upBlocked ? std::max(upHit.t - 1e-4f, 0.0f) : 1.0f;
        Vec3f topPos = m_position + Vec3f(0.0f, stepH * upAdvance, 0.0f);

        // b) Sweep forward horizontally.
        Vec3f horizDisp(displacement.x, 0.0f, displacement.z);
        float horizLen = horizDisp.length();
        if (horizLen < 1e-6f) {
            // No horizontal movement — just check if we can stand at the stepped-up position.
            newPos = topPos;
            return true;
        }

        Vec3f fwdTarget = topPos + horizDisp;
        PhysicsWorld::SweepHit fwdHit;
        bool fwdBlocked = m_world->sweepShape(capsule, topPos, m_rotation,
                                               fwdTarget, m_rotation,
                                               &m_filter, fwdHit);

        float fwdAdvance = 1.0f;
        if (!fwdBlocked) {
            // Forward sweep is clear — move full horizontal distance.
            fwdAdvance = 1.0f;
        } else if (fwdHit.t > 0.3f) {
            // Partial forward — at least 30 % of the step is clear.
            fwdAdvance = std::max(fwdHit.t - 1e-4f, 0.0f);
        } else {
            // Forward sweep is too obstructed — don't step.
            return false;
        }

        Vec3f fwdPos = topPos + horizDisp * fwdAdvance;

        // c) Sweep down to find ground.
        float downDist = stepH * 1.1f + m_config.skinWidth;
        Vec3f downTarget = fwdPos - Vec3f(0.0f, downDist, 0.0f);
        PhysicsWorld::SweepHit downHit;
        bool downHitSomething = m_world->sweepShape(capsule, fwdPos, m_rotation,
                                                     downTarget, m_rotation,
                                                     &m_filter, downHit);

        if (downHitSomething && downHit.t < 1.0f) {
            // d) Check that the ground is within step height and the slope is walkable.
            float groundDist = downHit.t * downDist;
            if (groundDist <= stepH + m_config.skinWidth) {
                // Check slope angle (negate contact normal to get surface normal).
                Vec3f surfNml = -downHit.normal;
                float slopeAngle = std::acos(std::clamp(surfNml.dot(Vec3f::unitY()),
                                                        -1.0f, 1.0f));
                float maxSlopeRad = m_config.maxSlopeAngle * (3.14159265358979f / 180.0f);

                if (slopeAngle <= maxSlopeRad) {
                    // Accept the step.
                    float downAdvance = std::max(downHit.t - 1e-4f, 0.0f);
                    newPos = fwdPos + Vec3f(0.0f, -downDist * downAdvance, 0.0f);
                    return true;
                }
            }
        }

        // Fall back to the simple stepped-up position if forward sweep was clear but
        // we couldn't find valid ground below.
        if (!fwdBlocked) {
            newPos = fwdPos;
            return true;
        }

        return false;
    }

    // ─── Ground detection ─────────────────────────────────────────────────────────────────────

    /// @brief  Sweep downward to detect walkable ground and snap the character to it.
    ///
    /// The sweep distance is the capsule half-height (centre to ground at rest) plus a small
    /// margin, ensuring the sweep reaches the ground from any airborne position within one
    /// fixed timestep of velocity.
    void updateGroundedState() {
        float capsuleHalf = m_config.height * 0.5f;
        // Sweep distance = half-height + margin (enough to catch a falling character).
        float kCheckDist = capsuleHalf + 0.2f;
        float maxSlopeRad = m_config.maxSlopeAngle * (3.14159265358979f / 180.0f);

        Vec3f downTarget = m_position - Vec3f(0.0f, kCheckDist, 0.0f);
        PhysicsWorld::SweepHit hit;
        Shape capsule = makeCapsuleShape();
        bool hitGround = m_world->sweepShape(capsule, m_position, m_rotation,
                                              downTarget, m_rotation,
                                              &m_filter, hit);

        if (hitGround && hit.t >= 0.0f && hit.t < 1.0f) {
            // The contact normal points from capsule (A) toward plane (B), i.e. downward.
            // For slope angle we need the surface normal (from the surface toward the
            // character), which is the negation of the contact normal.
            Vec3f surfaceNormal = -hit.normal;
            float slopeAngle = std::acos(std::clamp(surfaceNormal.dot(Vec3f::unitY()),
                                                    -1.0f, 1.0f));
            if (slopeAngle <= maxSlopeRad) {
                m_onGround = true;
                m_groundNormal = hit.normal;

                // Compute the new centre position — the capsule bottom touches the ground
                // surface at the point of first contact during the sweep.
                //   centre_at_contact = start_y - hit.t * kCheckDist
                //   bottom at contact = centre_at_contact - capsuleHalf = groundY
                //   new centre = groundY + capsuleHalf = start_y - hit.t * kCheckDist
                float newY = m_position.y - hit.t * kCheckDist;

                // Apply skinWidth offset to prevent exact surface contact, which causes
                // t=0 hits on subsequent sweeps that block horizontal movement.
                // On flat ground this matches the old restY clamp; on slopes it lifts the
                // capsule slightly off the surface along world Y.
                newY += m_config.skinWidth;

                // Clamp so we never push below the natural rest height.
                float restY = capsuleHalf + m_config.skinWidth;
                if (newY < restY) newY = restY;

                m_position.y = newY;

                // Cancel vertical velocity when grounded on a walkable slope.
                if (m_velocity.y < 0.0f) m_velocity.y = 0.0f;
                return;
            }
        }

        m_onGround = false;
        m_groundNormal = Vec3f::unitY();
    }

    // ─── Capsule shape helper ─────────────────────────────────────────────────────────────────

    /// @brief  Create a temporary Shape from the stored capsule parameters.
    /// @note   The capsule is reconstructed each time because the Shape union is small
    ///         (cheap to recreate) and avoids storing a mutable Shape member.
    [[nodiscard]] Shape makeCapsuleShape() const noexcept {
        float halfHeight = (m_config.height * 0.5f) - m_config.radius;
        if (halfHeight < 0.01f) halfHeight = 0.01f;
        return Shape(Capsule{m_config.radius, halfHeight});
    }

    // ─── Members ──────────────────────────────────────────────────────────────────────────────

    PhysicsWorld*             m_world = nullptr;
    CharacterControllerConfig m_config;

    ShapeHandle               m_capsuleShape  = kInvalidShapeHandle;
    RigidBodyHandle           m_body           = kInvalidHandle;

    Vec3f                     m_position      = Vec3f::zero();
    Quaternionf               m_rotation      = Quaternionf::identity();
    Vec3f                     m_velocity      = Vec3f::zero();
    Vec3f                     m_moveDir       = Vec3f::zero();

    bool                      m_onGround      = false;
    Vec3f                     m_groundNormal  = Vec3f::unitY();

    SelfFilter                m_filter;
};

} // namespace ciphyxs
