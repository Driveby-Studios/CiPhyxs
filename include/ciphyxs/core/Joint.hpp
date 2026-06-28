//==================================================================================================
/// @file  Joint.hpp
/// @brief  Joint constraint types, descriptors, and storage for CiPhyxs.
///
/// Joints connect two rigid bodies with kinematic constraints.  Supported types:
///   - **BallSocket**  – 3-DOF (position only).  Acts like a spherical pendulum.
///   - **Hinge**       – 1-DOF rotation.  Supports angular limits and a motor.
///   - **Fixed**       – 0-DOF.  Welds two bodies together.
///   - **Spring**      – Damped spring along the distance between two anchor points.
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include "RigidBody.hpp"
#include <cstdint>
#include <limits>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// JointType
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  The concrete type of a joint constraint.
enum class JointType : std::uint8_t {
    BallSocket, ///< 3 translational constraints – anchor points coincide.
    Hinge,      ///< 5 constraints – rotation about a single axis is free.
    Fixed,      ///< 6 constraints – fully locked relative transform.
    Spring      ///< Damped spring force between two anchors (soft constraint).
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// JointHandle
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Opaque handle referencing a joint inside the PhysicsWorld.
using JointHandle = std::uint32_t;

/// @brief  Sentinel value representing an invalid / null joint handle.
inline constexpr JointHandle kInvalidJointHandle = std::numeric_limits<JointHandle>::max();

// ────────────────────────────────────────────────────────────────────────────────────────────────
// JointDesc
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  User-facing descriptor used to create a joint in the PhysicsWorld.
///
/// Anchor points and axes are specified in **world space** at creation time;
/// the PhysicsWorld converts them to **local space** internally.
struct JointDesc {
    JointType type = JointType::BallSocket;

    RigidBodyHandle bodyA = kInvalidHandle;
    RigidBodyHandle bodyB = kInvalidHandle;

    // ─── Anchor points ────────────────────────────────────────────────────────────────────────
    Vec3f anchorA = Vec3f::zero();  ///< Attachment point on body A (world space).
    Vec3f anchorB = Vec3f::zero();  ///< Attachment point on body B (world space).

    // ─── Sub-shape attachment (for compound shapes) ─────────────────────────────────────────
    /// @brief  Optional sub-shape index on body A for constraint attachment.
    ///         Set to -1 (default) to attach at the body centre (current behaviour).
    ///         When set to a valid sub-shape index, the joint anchor is computed relative
    ///         to that sub-shape's local transform, allowing constraints to connect at
    ///         compound sub-shape positions.
    int subShapeA = -1;

    /// @brief  Optional sub-shape index on body B (same semantics as subShapeA).
    int subShapeB = -1;

    // ─── Hinge axis (world space; only used for JointType::Hinge) ────────────────────────────
    Vec3f axisA = Vec3f::unitX();   ///< Hinge axis on body A.
    Vec3f axisB = Vec3f::unitX();   ///< Hinge axis on body B (should align initially).

    // ─── Limits (hinge rotation angle in radians, for future slider use) ─────────────────────
    bool  enableLimits = false;
    float limitMin     = -3.14159265f;
    float limitMax     =  3.14159265f;

    // ─── Motor ───────────────────────────────────────────────────────────────────────────────
    bool  enableMotor          = false;
    float motorTargetVelocity  = 0.0f;  ///< Desired angular velocity (rad/s).
    float motorMaxForce        = 0.0f;  ///< Maximum impulse the motor can apply.

    // ─── Spring parameters (only used for JointType::Spring) ─────────────────────────────────
    float springStiffness = 0.0f;
    float springDamping   = 0.0f;
    float restLength      = 1.0f;

    // ─── Breakable constraints ────────────────────────────────────────────────────────────────
    /// @brief  Maximum total accumulated impulse the joint can withstand before breaking.
    ///         Set to 0 or a negative value for unbreakable joints.
    ///         When the sum of absolute impulses across all constraint rows exceeds this
    ///         threshold, the joint is marked as broken and will be removed.
    ///
    /// @note   Breakable joints are useful for destructible structures, force-limited
    ///         robotics, and gameplay mechanics where constraints should yield under load.
    float breakThreshold = 0.0f;  ///< 0 = unbreakable
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// JointStorage
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Internal per‑joint storage in the PhysicsWorld (SoA-compatible).
///
/// Anchor points and axes are stored in **local space** of each body so they
/// remain valid as the bodies move.  Created from a `JointDesc`.
struct JointStorage {
    JointType         type;
    RigidBodyHandle   bodyA;
    RigidBodyHandle   bodyB;

    Vec3f             localAnchorA;       ///< Anchor in body A's local frame.
    Vec3f             localAnchorB;       ///< Anchor in body B's local frame.

    int               subShapeA = -1;     ///< Sub-shape index on body A (-1 = body centre).
    int               subShapeB = -1;     ///< Sub-shape index on body B.

    Vec3f             localAxisA;         ///< Hinge axis in body A's local frame.
    Vec3f             localAxisB;         ///< Hinge axis in body B's local frame.

    bool              enableLimits;
    float             limitMin;
    float             limitMax;

    bool              enableMotor;
    float             motorTargetVelocity;
    float             motorMaxForce;

    float             springStiffness;
    float             springDamping;
    float             restLength;

    // ─── Breakable constraints ────────────────────────────────────────────────────────────────
    float breakThreshold = 0.0f;   ///< 0 = unbreakable.  Accumulated impulse sum > this = broken.
    bool  broken         = false;  ///< Set to true when the break threshold is exceeded.

    // ─── Warm‑start cache ───────────────────────────────────────────────────────────────────
    /// Accumulated impulses from the previous sub‑step, indexed by constraint row.
    /// For BallSocket: 3 rows (X, Y, Z).  For Hinge: 5 rows (XYZ + 2 angular).
    /// For Fixed: 6 rows (XYZ + 3 angular).  For Spring: 1 row.
    float impulseCache[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    /// Number of valid cache entries (derived from joint type).
    [[nodiscard]] int cacheSize() const noexcept {
        switch (type) {
        case JointType::BallSocket: return 3;
        case JointType::Hinge:      return 5;
        case JointType::Fixed:      return 6;
        case JointType::Spring:     return 1;
        }
        return 0;
    }
};

} // namespace ciphyxs
