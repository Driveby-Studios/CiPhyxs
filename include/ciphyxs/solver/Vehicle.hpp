//==================================================================================================
/// @file  Vehicle.hpp
/// @brief  Ray-cast vehicle controller with suspension, tire friction, and drive train.
///
/// ## Overview
///
/// The Vehicle system provides a ray-cast based wheeled vehicle model.  Each wheel casts a ray
/// downward from its suspension attachment point to find the ground, then applies spring-damper
/// forces, lateral/longitudinal tire friction, drive torque, and braking forces to the chassis.
///
/// ## Usage
///
/// @code
///   // 1. Create a chassis body and wheels.
///   VehicleConfig vcfg;
///   vcfg.chassisBody = chassis;
///   vcfg.wheels.resize(4);
///   vcfg.wheels[0] = WheelConfig{ .attachmentPoint = Vec3f(-1, 0,  1.5f), .canSteer = true,  .canDrive = true };
///   vcfg.wheels[1] = WheelConfig{ .attachmentPoint = Vec3f( 1, 0,  1.5f), .canSteer = true,  .canDrive = true };
///   vcfg.wheels[2] = WheelConfig{ .attachmentPoint = Vec3f(-1, 0, -1.5f), .canSteer = false, .canDrive = false };
///   vcfg.wheels[3] = WheelConfig{ .attachmentPoint = Vec3f( 1, 0, -1.5f), .canSteer = false, .canDrive = false };
///
///   Vehicle vehicle(world, vcfg);
///
///   // 2. Drive with an ISolverHook.
///   struct DriveHook : ISolverHook {
///       Vehicle* veh;
///       void onApplyForces(float dt, RigidBodyStorage&) override { veh->update(dt); }
///   } hook{&vehicle};
///   world.addHook(&hook);
///
///   // 3. Control inputs each frame.
///   vehicle.setThrottle(0.5f);
///   vehicle.setSteerAngle(0.3f);
///   world.step(1.0f/60.0f);
/// @endcode
//==================================================================================================
#pragma once

#include "../core/PhysicsWorld.hpp"
#include "../collision/Ray.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// WheelConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning parameters for a single vehicle wheel.
struct WheelConfig {
    /// @brief  Local-space attachment point on the chassis body.
    Vec3f attachmentPoint = Vec3f::zero();

    // ─── Suspension ──────────────────────────────────────────────────────────────────────────

    /// @brief  Rest (uncompressed) length of the suspension spring.
    float suspensionRestLength = 0.5f;

    /// @brief  Spring constant (F = k · compression).
    float suspensionStiffness  = 200.0f;

    /// @brief  Damping coefficient (F = c · compressionVelocity).
    float suspensionDamping    = 20.0f;

    /// @brief  Maximum suspension force (clamping prevents instability at full compression).
    float suspensionMaxForce   = 10000.0f;

    // ─── Wheel geometry ─────────────────────────────────────────────────────────────────────

    /// @brief  Wheel radius (used for contact-point computation and angular velocity).
    float wheelRadius = 0.3f;

    // ─── Function ───────────────────────────────────────────────────────────────────────────

    /// @brief  If true, this wheel responds to steering input.
    bool canSteer = false;

    /// @brief  If true, engine torque is applied to this wheel.
    bool canDrive = false;

    /// @brief  If true, brake torque can be applied to this wheel.
    bool canBrake = true;

    // ─── Tire friction ──────────────────────────────────────────────────────────────────────

    /// @brief  Lateral friction coefficient (cornering grip).
    float lateralFriction = 0.8f;

    /// @brief  Longitudinal friction coefficient (acceleration/braking grip).
    float longitudinalFriction = 0.8f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// WheelState
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Per-wheel runtime state, updated each frame by Vehicle::update().
struct WheelState {
    /// @brief  Current suspension length (m).
    float suspensionLength = 0.0f;

    /// @brief  Previous frame's suspension length, used for damping velocity.
    float suspensionPrevLength = 0.0f;

    /// @brief  Wheel angular velocity (rad/s).  Positive = forward rotation.
    float angularVelocity = 0.0f;

    /// @brief  Current steering angle (rad).
    float steerAngle = 0.0f;

    /// @brief  True when the suspension ray-cast hits a surface.
    bool  isGrounded = false;

    /// @brief  World-space contact point on the ground surface.
    Vec3f contactPoint = Vec3f::zero();

    /// @brief  World-space contact normal (points from surface toward wheel).
    Vec3f contactNormal = Vec3f::zero();

    /// @brief  World-space attachment point (chassis position + rotated local attach).
    Vec3f worldAttachment = Vec3f::zero();
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// VehicleConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Complete vehicle configuration.
struct VehicleConfig {
    /// @brief  Handle of the chassis rigid body.
    RigidBodyHandle chassisBody = kInvalidHandle;

    /// @brief  Per-wheel configurations.
    std::vector<WheelConfig> wheels;

    // ─── Drive train ────────────────────────────────────────────────────────────────────────

    /// @brief  Maximum engine torque (N·m) applied to driven wheels.
    float engineMaxTorque = 500.0f;

    /// @brief  Maximum brake torque (N·m) per wheel.
    float brakeTorque = 3000.0f;

    /// @brief  Maximum steering angle (rad).  Default ~30 degrees.
    float maxSteerAngle = 0.52f;

    /// @brief  Ray length beyond the suspension for ground detection.
    ///         Should be large enough to catch ground even when the wheel is fully extended.
    float rayLengthScale = 1.5f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Vehicle
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A ray-cast wheeled vehicle controller.
///
/// The Vehicle is **not** tied to the simulation pipeline — the user must call `update(dt)`
/// from an `ISolverHook::onApplyForces()` callback or manually each fixed step.  This keeps
/// the vehicle logic fully decoupled from the engine.
///
/// @note  All forces are applied at the contact point and produce appropriate torques on the
///        chassis via the lever arm (r × F).
class Vehicle {
public:
    /// @brief  Construct a vehicle and attach it to a physics world.
    /// @param world  PhysicsWorld that owns the chassis body.
    /// @param config Vehicle configuration (chassis handle, wheels, drive params).
    Vehicle(PhysicsWorld& world, const VehicleConfig& config)
        : m_world(&world), m_config(config) {
        m_wheelStates.resize(config.wheels.size());
    }

    // ─── Control inputs ──────────────────────────────────────────────────────────────────────

    /// @brief  Set throttle input [0, 1].
    void setThrottle(float t) noexcept { m_throttle = std::clamp(t, 0.0f, 1.0f); }

    /// @brief  Set brake input [0, 1].
    void setBrake(float b) noexcept { m_brake = std::clamp(b, 0.0f, 1.0f); }

    /// @brief  Set steering angle (rad).  Clamped to ±maxSteerAngle.
    void setSteerAngle(float angle) noexcept { m_steerAngle = angle; }

    // ─── Accessors ───────────────────────────────────────────────────────────────────────────

    /// @brief  Read-only access to per-wheel runtime state (for rendering / debug).
    [[nodiscard]] const std::vector<WheelState>& wheelStates() const noexcept { return m_wheelStates; }

    /// @brief  Read-only access to the vehicle configuration.
    [[nodiscard]] const VehicleConfig& config() const noexcept { return m_config; }

    /// @brief  Mutable access to the vehicle configuration (adjust params at runtime).
    [[nodiscard]] VehicleConfig& config() noexcept { return m_config; }

    // ─── Per-step update ─────────────────────────────────────────────────────────────────────

    /// @brief  Update suspension, tire forces, and drive for all wheels.
    ///
    /// Call this **once per fixed step** from your `ISolverHook::onApplyForces()`.
    ///
    /// @param dt  Fixed timestep (usually 1/60).
    void update(float dt) {
        RigidBodyStorage& bodies = m_world->bodies();
        RigidBodyHandle h = m_config.chassisBody;

        // Compute chassis orientation vectors once per frame.
        Vec3f chassisFwd   = bodies.rotations[h].rotate(Vec3f::unitZ());
        Vec3f chassisUp    = bodies.rotations[h].rotate(Vec3f::unitY());

        // Accelerations and torques accumulate on the chassis.
        for (std::size_t i = 0; i < m_config.wheels.size(); ++i) {
            updateWheel(static_cast<int>(i), dt, bodies, chassisFwd, chassisUp);
        }
    }

private:
    /// @brief  Update a single wheel: ray-cast, suspension, tire forces, drive, brakes.
    void updateWheel(int idx, float dt,
                     RigidBodyStorage& bodies,
                     const Vec3f& chassisFwd,
                     const Vec3f& chassisUp) {

        const auto& cfg = m_config.wheels[idx];
        auto& state     = m_wheelStates[idx];
        RigidBodyHandle h = m_config.chassisBody;

        // ── 1. World-space attachment point ────────────────────────────────────────────────
        state.worldAttachment = bodies.positions[h]
                              + bodies.rotations[h].rotate(cfg.attachmentPoint);

        // ── 2. Steering rotation ───────────────────────────────────────────────────────────
        float steerAngle = cfg.canSteer ? m_steerAngle : 0.0f;
        // Clamp to max steer angle.
        steerAngle = std::clamp(steerAngle, -m_config.maxSteerAngle, m_config.maxSteerAngle);
        state.steerAngle = steerAngle;

        // Wheel forward direction: start with chassis forward, rotate around up by steer.
        Quaternionf steerRot = Quaternionf::fromAxisAngle(chassisUp, steerAngle);
        Vec3f wheelFwd = steerRot.rotate(chassisFwd);
        // ── 3. Suspension ray-cast ─────────────────────────────────────────────────────────
        float rayLen = (cfg.suspensionRestLength + cfg.wheelRadius) * m_config.rayLengthScale;
        Vec3f rayDir = -chassisUp; // cast downward in chassis local space

        Ray ray;
        ray.origin    = state.worldAttachment;
        ray.direction = rayDir;

        RayHit hit;
        bool hitGround = m_world->rayCast(ray, rayLen, hit);

        state.isGrounded = hitGround;

        if (hitGround && hit.t > 0.0f && hit.body != h) {
            // ── Contact geometry ───────────────────────────────────────────────────────────
            float contactDepth = hit.t; // distance from attachment to hit point
            state.suspensionLength = std::clamp(contactDepth - cfg.wheelRadius,
                                                0.0f,
                                                cfg.suspensionRestLength);
            state.contactPoint  = hit.point;
            state.contactNormal = hit.normal;

            // ── 4. Suspension spring-damper ────────────────────────────────────────────────
            float compression    = cfg.suspensionRestLength - state.suspensionLength;
            float prevCompression = cfg.suspensionRestLength - state.suspensionPrevLength;
            float compVel        = (compression - prevCompression) / std::max(dt, 1e-8f);

            float springForceMag = compression * cfg.suspensionStiffness
                                 + compVel * cfg.suspensionDamping;
            springForceMag = std::clamp(springForceMag, 0.0f, cfg.suspensionMaxForce);

            // Apply suspension force along the contact normal.
            Vec3f suspensionForce = state.contactNormal * springForceMag;
            Vec3f r = state.contactPoint - bodies.positions[h];
            bodies.forces[h]  += suspensionForce;
            bodies.torques[h] += r.cross(suspensionForce);

            // ── 5. Tire friction ───────────────────────────────────────────────────────────
            //
            //  Project wheel forward/right onto the contact plane to get the
            //  longitudinal (rolling) and lateral (cornering) directions.
            Vec3f longDir = (wheelFwd - state.contactNormal *
                             state.contactNormal.dot(wheelFwd)).normalized();
            Vec3f latDir  = state.contactNormal.cross(longDir).normalized();

            // Chassis velocity at the contact point.
            Vec3f chassisVel = bodies.linearVelocities[h]
                             + bodies.angularVelocities[h].cross(r);

            // Longitudinal slip ratio.
            float wheelSurfVel   = state.angularVelocity * cfg.wheelRadius;
            float chassisLongVel = chassisVel.dot(longDir);
            float speedMag       = std::max(std::abs(chassisLongVel), 0.1f);
            float slipLong       = (wheelSurfVel - chassisLongVel) / speedMag;
            slipLong = std::clamp(slipLong, -1.0f, 1.0f);

            // Lateral slip ratio.
            float chassisLatVel = chassisVel.dot(latDir);
            float slipLat       = chassisLatVel / speedMag;
            slipLat = std::clamp(slipLat, -1.0f, 1.0f);

            // Friction forces (simplified Pacejka-like: force ∝ slip, clamped at max).
            float normalForce = springForceMag;
            float longForce   = slipLong * cfg.longitudinalFriction * normalForce;
            float latForce    = -slipLat * cfg.lateralFriction * normalForce;

            // Combined friction circle limiting.
            float maxFriction = normalForce * std::sqrt(
                cfg.longitudinalFriction * cfg.longitudinalFriction +
                cfg.lateralFriction * cfg.lateralFriction);
            float totalForce = std::sqrt(longForce * longForce + latForce * latForce);
            if (totalForce > maxFriction && totalForce > 1e-8f) {
                float scale = maxFriction / totalForce;
                longForce *= scale;
                latForce  *= scale;
            }

            // ── Anti-squat tire force application ────────────────────────────────────────
            //
            // Longitudinal (drive/brake) force is applied at the center of mass with NO
            // torque.  In a real vehicle the suspension geometry creates an "anti-squat"
            // reaction that cancels most of the pitch torque from the drive force.
            // Without modelling the full linkage, applying the longitudinal force at
            // the COM prevents the drive force from creating an un-damped nose-up pitch
            // that loads the rear suspension and launches the chassis into the air.
            //
            // Lateral (cornering) force IS applied at the contact patch with full torque,
            // producing realistic body roll during turns.
            Vec3f tireForce = longDir * longForce + latDir * latForce;
            bodies.forces[h]  += tireForce;
            Vec3f latTorque   = r.cross(latDir * latForce);  // roll torque from cornering
            bodies.torques[h] += latTorque;

            // ── 6. Drive torque (engine) ───────────────────────────────────────────────────
            if (cfg.canDrive && m_throttle > 0.0f) {
                float driveTorque = m_throttle * m_config.engineMaxTorque;
                // Wheel rotational inertia is approximated as a solid disk: I = 0.5 * m * r².
                // We use a normalised mass of 20 kg per wheel.
                constexpr float kWheelMass = 20.0f;
                float wheelInertia = 0.5f * kWheelMass * cfg.wheelRadius * cfg.wheelRadius;
                float torqueAccel  = driveTorque / wheelInertia;
                state.angularVelocity += torqueAccel * dt;
            }

            // ── 7. Brake torque ────────────────────────────────────────────────────────────
            if (cfg.canBrake && m_brake > 0.0f) {
                float brakeDecel = m_brake * m_config.brakeTorque / (0.5f * 20.0f * cfg.wheelRadius * cfg.wheelRadius);
                if (state.angularVelocity > 0.0f) {
                    state.angularVelocity = std::max(0.0f, state.angularVelocity - brakeDecel * dt);
                } else {
                    state.angularVelocity = std::min(0.0f, state.angularVelocity + brakeDecel * dt);
                }
            }

        } else {
            // ── Wheel in air ───────────────────────────────────────────────────────────────
            state.suspensionLength = cfg.suspensionRestLength;

            // Free spin with air resistance.
            if (cfg.canDrive && m_throttle > 0.0f) {
                constexpr float kWheelMass = 20.0f;
                float wheelInertia = 0.5f * kWheelMass * cfg.wheelRadius * cfg.wheelRadius;
                state.angularVelocity += (m_throttle * m_config.engineMaxTorque / wheelInertia) * dt;
            }
            if (cfg.canBrake && m_brake > 0.0f) {
                state.angularVelocity *= (1.0f - m_brake * 0.05f);
            }
            state.angularVelocity *= 0.995f; // mild air drag
        }

        // ── 8. Store suspension length for next frame's damping velocity ──────────────────
        state.suspensionPrevLength = state.suspensionLength;
    }

    PhysicsWorld*              m_world;
    VehicleConfig              m_config;
    std::vector<WheelState>    m_wheelStates;
    float                      m_throttle    = 0.0f;
    float                      m_brake       = 0.0f;
    float                      m_steerAngle  = 0.0f;
};

} // namespace ciphyxs
