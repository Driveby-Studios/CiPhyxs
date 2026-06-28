//==================================================================================================
/// @file  Contact.hpp
/// @brief  Contact-point and contact-manifold data structures for the constraint pipeline.
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../core/RigidBody.hpp"
#include "../collision/Shape.hpp"
#include <cstdint>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ContactPoint
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A single contact point between two rigid bodies.
///
/// Each point carries warm-starting impulse caches so that the solver can re-use impulses
/// from the previous frame (greatly improving convergence).
struct ContactPoint {
    Vec3f position    = Vec3f::zero();   ///< World-space contact location.
    Vec3f normal      = Vec3f::unitY();  ///< Contact normal pointing from body A toward body B.
    float penetration = 0.0f;            ///< Interpenetration depth (>= 0).

    // ─── Warm-starting cache (updated each frame by the solver) ──────────────────────────────
    float normalImpulse          = 0.0f;  ///< Accumulated impulse along the normal.
    float tangentImpulse[2]      = {0.0f, 0.0f}; ///< Accumulated impulse along two friction tangents.
    Vec3f tangent[2]             = {};    ///< Cached tangent directions (rebuilt each step).

    // ─── Feature tags (used to match contact points across frames) ───────────────────────────
    std::int32_t featureA = -1;  ///< Shape-feature ID on body A (e.g. face/edge index).
    std::int32_t featureB = -1;  ///< Shape-feature ID on body B.
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ContactManifold
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A manifold groups up to 4 contact points shared by the same body pair.
///
/// Manifolds are the fundamental unit of work for the sequential-impulse solver.
/// Storing 4 points per manifold is sufficient for face-face box contacts;
/// edge-edge or sphere contacts typically produce 1–2 points.
struct ContactManifold {
    RigidBodyHandle bodyA = kInvalidHandle;
    RigidBodyHandle bodyB = kInvalidHandle;

    ContactPoint points[4];
    int           pointCount = 0;

    /// @brief  Combined friction coefficient for this pair (set by the collision pipeline).
    float combinedFriction    = 0.5f;
    /// @brief  Combined restitution for this pair (set by the collision pipeline).
    float combinedRestitution = 0.35f;

    // ─── Helpers ────────────────────────────────────────────────────────────────────────────────

    /// @brief  Add a contact point (up to 4 – excess entries are ignored).
    void addPoint(const ContactPoint& pt) noexcept {
        if (pointCount < 4) points[pointCount++] = pt;
    }

    /// @brief  Remove the point at index `i` (swap-with-last).
    void removePoint(int i) noexcept {
        if (i < pointCount - 1) points[i] = points[pointCount - 1];
        --pointCount;
    }

    /// @brief  Clear all points but keep the body pair.
    void clearPoints() noexcept { pointCount = 0; }

    /// @brief  True if the manifold references the given body pair.
    [[nodiscard]] bool involves(RigidBodyHandle a, RigidBodyHandle b) const noexcept {
        return (bodyA == a && bodyB == b) || (bodyA == b && bodyB == a);
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// BroadphasePair
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A candidate collision pair produced by the broad-phase.
struct BroadphasePair {
    RigidBodyHandle bodyA;
    RigidBodyHandle bodyB;
};

} // namespace ciphyxs
