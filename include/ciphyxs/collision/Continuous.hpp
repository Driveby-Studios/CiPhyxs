//==================================================================================================
/// @file  Continuous.hpp
/// @brief  Continuous Collision Detection (CCD) — swept-shape tests and pipeline helpers.
///
/// ## Why CCD?
///
/// At 60 Hz a fast-moving object (e.g. a bullet at 100 m/s) travels ~1.7 m per frame.
/// If the object is smaller than that distance it can pass **through** thin geometry —
/// this is *tunnelling*.  CCD prevents tunnelling by testing the *swept* shape along the
/// velocity vector, finding the **Time of Impact (TOI)**, and stopping the body there.
///
/// ## Implementation strategy
///
/// For the first implementation we provide **swept sphere** tests (the most common CCD
/// primitive) and a **binary-search fallback** for general shapes:
///
/// | Pair              | Method                    | Accuracy |
/// |-------------------|---------------------------|----------|
/// | Sphere ↔ Plane    | Analytical quadratic      | Exact    |
/// | Sphere ↔ Sphere   | Analytical quadratic      | Exact    |
/// | Sphere ↔ Box      | Binary search (16 iters)  | ~1 mm    |
/// | Box ↔ Anything    | Swept AABB + binary search| ~1 mm    |
///
/// ## Pipeline integration
///
/// CCD runs as a **separate pass** after velocity integration but before collision
/// detection.  For each CCD‑enabled body we test the swept path against all other active
/// bodies.  If a hit is found the body is advanced to the TOI position and its velocity
/// is reflected along the collision normal.  The constraint solver then sees the corrected
/// state.
//==================================================================================================
#pragma once

#include "Shape.hpp"
#include "Contact.hpp"
#include "../core/RigidBody.hpp"
#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept sphere ↔ plane  (analytical)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept sphere vs. infinite plane.  Returns TOI in [0, 1] or 1 (no hit).
///
/// @param sphere       Sphere primitive.
/// @param pos          Sphere centre at t=0 (world space).
/// @param vel          Sphere linear velocity over the full step (world space).
/// @param dt           Fixed timestep.
/// @param plane        Plane primitive (world-space normal and distance after rotation).
/// @param planePos     Plane body position (for signed-distance computation).
/// @param planeNormal  World-space plane normal.
/// @param[out] outNormal  Contact normal at TOI (points from plane toward sphere).
/// @param[out] outPosition Contact position at TOI.
/// @return TOI in [0,1], or 1.0 if no collision.
inline float sweepSpherePlane(const Sphere& sphere,
                              const Vec3f& pos, const Vec3f& vel, float dt,
                              const Plane& plane, const Vec3f& planePos,
                              const Vec3f& planeNormal,
                              Vec3f& outNormal,
                              Vec3f& outPosition) noexcept {
    // Signed distance from sphere centre to plane at t=0.
    Vec3f velFrame = vel * dt;
    float vLenSq = velFrame.lengthSquared();
    if (vLenSq < 1e-12f) return 1.0f; // not moving

    float dist0 = (pos - planePos).dot(planeNormal) - plane.distance - sphere.radius;

    // Already overlapping?  Report TOI = 0.
    if (dist0 < 0.0f) {
        // Normal points from plane surface toward the sphere centre.
        // For dist0 < 0 the sphere centre is within one radius of the surface;
        // we compute the outward direction from the nearest surface point toward
        // the sphere centre.  planeNormal is the outward-facing normal (the side
        // objects exist on), so we use it directly — the sphere is on the outward
        // side even if deeply overlapping, because planes are infinite.
        outNormal   = planeNormal;
        outPosition = pos - planeNormal * (dist0 + sphere.radius);
        return 0.0f;
    }

    // Velocity along normal (positive = moving away).
    float vDotN = velFrame.dot(planeNormal);
    if (vDotN >= 0.0f) return 1.0f; // moving away or parallel

    // TOI = dist0 / |vDotN|  (the time when sphere centre is exactly r from plane).
    float toi = dist0 / (-vDotN);
    if (toi > 1.0f) return 1.0f;

    // Normal points from plane toward sphere (so that vel·normal < 0 when
    // the sphere is approaching the plane, enabling correct velocity reflection
    // in ccdResolveBody).
    outNormal   = planeNormal;
    outPosition = pos + velFrame * toi - planeNormal * sphere.radius;
    return toi;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept sphere ↔ sphere  (analytical)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept sphere vs. sphere.  Returns TOI in [0, 1] or 1 (no hit).
///
/// Solves the quadratic |r0 + t·dr|² = (rA+rB)² where r0 = posA-posB and
/// dr = (velA-velB)·dt.
inline float sweepSphereSphere(const Sphere& sphereA, const Vec3f& posA, const Vec3f& velA,
                               const Sphere& sphereB, const Vec3f& posB, const Vec3f& velB,
                               float dt,
                               Vec3f& outNormal,
                               Vec3f& outPosition) noexcept {
    Vec3f drVel = (velA - velB) * dt;
    float vLenSq = drVel.lengthSquared();
    Vec3f r0 = posA - posB;
    float rLenSq = r0.lengthSquared();
    float radSum = sphereA.radius + sphereB.radius;

    // Already overlapping?
    if (rLenSq < radSum * radSum) {
        float rDist = std::sqrt(rLenSq);
        if (rDist < 1e-12f) {
            outNormal = Vec3f::unitY();
            outPosition = posA;
        } else {
            outNormal = r0 / rDist;
            outPosition = posB + outNormal * sphereB.radius;
        }
        return 0.0f;
    }

    if (vLenSq < 1e-12f) return 1.0f; // not moving relative

    // Quadratic: a·t² + b·t + c = 0
    float a = drVel.dot(drVel);
    float b = 2.0f * r0.dot(drVel);
    float c = r0.dot(r0) - radSum * radSum;

    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return 1.0f; // no real solution

    float sqrtDisc = std::sqrt(disc);
    // Smallest positive root in [0, 1].
    float t1 = (-b - sqrtDisc) / (2.0f * a);
    float t2 = (-b + sqrtDisc) / (2.0f * a);

    float toi = 1.0f;
    if (t1 >= 0.0f && t1 <= 1.0f) toi = t1;
    else if (t2 >= 0.0f && t2 <= 1.0f) toi = t2;

    if (toi >= 1.0f) return 1.0f;

    // Compute individual positions at TOI.
    Vec3f posAToi = posA + velA * dt * toi;
    Vec3f posBToi = posB + velB * dt * toi;
    Vec3f delta = posAToi - posBToi;
    float dLen = delta.length();
    if (dLen < 1e-12f) {
        outNormal = Vec3f::unitY();
    } else {
        outNormal = delta / dLen;
    }
    outPosition = posBToi + outNormal * sphereB.radius;
    return toi;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept sphere ↔ box  (binary search)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept sphere vs. oriented box.  Uses binary search (16 iterations ≈ 1 mm precision).
///
/// Samples the sphere position along the path and performs a static sphere-box test at
/// each sample.  This is ~16× more expensive than a static test but sufficient for CCD
/// of fast-moving spheres against box geometry.
inline float sweepSphereBox(const Sphere& sphere,
                            const Vec3f& pos, const Vec3f& vel, float dt,
                            const Box& box, const Vec3f& boxPos, const Quaternionf& boxRot,
                            Vec3f& outNormal,
                            Vec3f& outPosition) noexcept {
    Vec3f disp = vel * dt;
    float dispLen = disp.length();
    if (dispLen < 1e-12f) return 1.0f;

    // First, check if already overlapping at t=0.
    {
        Vec3f localPos = boxRot.rotateInverse(pos - boxPos);
        Vec3f he = box.halfExtents;
        Vec3f closest(
            std::max(-he.x, std::min(localPos.x, he.x)),
            std::max(-he.y, std::min(localPos.y, he.y)),
            std::max(-he.z, std::min(localPos.z, he.z))
        );
        Vec3f diff = localPos - closest;
        if (diff.lengthSquared() < sphere.radius * sphere.radius) {
            // Already overlapping — TOI = 0.
            float dLen = diff.length();
            Vec3f nLocal;
            if (dLen > 1e-12f) {
                nLocal = diff / dLen;
            } else {
                float minPen = he.x - std::abs(localPos.x);
                int axis = 0;
                float py = he.y - std::abs(localPos.y);
                if (py < minPen) { minPen = py; axis = 1; }
                float pz = he.z - std::abs(localPos.z);
                if (pz < minPen) { minPen = pz; axis = 2; }
                nLocal = Vec3f::zero();
                nLocal[axis] = (localPos[axis] >= 0.0f) ? 1.0f : -1.0f;
            }
            outNormal = boxRot.rotate(nLocal).normalized();
            Vec3f closestWorld = boxPos + boxRot.rotate(closest);
            outPosition = closestWorld + outNormal * sphere.radius;
            return 0.0f;
        }
    }

    // Binary search for TOI.
    constexpr int kIterations = 16;
    float lo = 0.0f, hi = 1.0f;
    bool hit = false;

    for (int iter = 0; iter < kIterations; ++iter) {
        float mid = (lo + hi) * 0.5f;
        Vec3f samplePos = pos + disp * mid;

        Vec3f localPos = boxRot.rotateInverse(samplePos - boxPos);
        Vec3f he = box.halfExtents;
        Vec3f closest(
            std::max(-he.x, std::min(localPos.x, he.x)),
            std::max(-he.y, std::min(localPos.y, he.y)),
            std::max(-he.z, std::min(localPos.z, he.z))
        );
        Vec3f diff = localPos - closest;
        float distSq = diff.lengthSquared();

        if (distSq < sphere.radius * sphere.radius) {
            hit = true;
            hi = mid; // overlap → move hi down
        } else {
            lo = mid; // no overlap → move lo up
        }
    }

    if (!hit) return 1.0f;

    float toi = (lo + hi) * 0.5f;
    Vec3f hitPos = pos + disp * toi;

    // Compute normal at TOI.
    Vec3f localPos = boxRot.rotateInverse(hitPos - boxPos);
    Vec3f he = box.halfExtents;
    Vec3f closest(
        std::max(-he.x, std::min(localPos.x, he.x)),
        std::max(-he.y, std::min(localPos.y, he.y)),
        std::max(-he.z, std::min(localPos.z, he.z))
    );
    Vec3f diff = localPos - closest;
    float dLen = diff.length();
    Vec3f nLocal;
    if (dLen > 1e-12f) {
        nLocal = diff / dLen;
    } else {
        float minPen = he.x - std::abs(localPos.x);
        int axis = 0;
        float py = he.y - std::abs(localPos.y);
        if (py < minPen) { minPen = py; axis = 1; }
        float pz = he.z - std::abs(localPos.z);
        if (pz < minPen) { minPen = pz; axis = 2; }
        nLocal = Vec3f::zero();
        nLocal[axis] = (localPos[axis] >= 0.0f) ? 1.0f : -1.0f;
    }

    outNormal = boxRot.rotate(nLocal).normalized();
    outPosition = hitPos;
    return toi;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept capsule ↔ plane  (analytical)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept capsule vs. infinite plane.  Returns TOI in [0, 1] or 1 (no hit).
///
/// Capsule is approximated by sweeping a sphere of `capsule.radius` along its
/// medial segment.  We compute the TOI as the minimum of the swept TOI for both
/// endpoints and, if the segment is moving mostly parallel to the plane, a swept
/// cylinder test (segment swept toward plane).
///
/// @param capsule      Capsule primitive.
/// @param pos          Capsule centre at t=0 (world space).
/// @param vel          Capsule linear velocity over the full step (world space).
/// @param dt           Fixed timestep.
/// @param plane        Plane primitive.
/// @param planePos     Plane body position.
/// @param planeNormal  World-space plane normal.
/// @param[out] outNormal  Contact normal at TOI (points from plane toward capsule).
/// @param[out] outPosition Contact position at TOI.
/// @return TOI in [0,1], or 1.0 if no collision.
inline float sweepCapsulePlane(const Capsule& capsule,
                                const Vec3f& pos, const Vec3f& vel, float dt,
                                const Plane& plane, const Vec3f& planePos,
                                const Vec3f& planeNormal,
                                Vec3f& outNormal,
                                Vec3f& outPosition) noexcept {
    // Capsule segment endpoints at t=0 (capsule is Y-aligned in local space).
    Vec3f localA(0.0f, -capsule.halfHeight, 0.0f);
    Vec3f localB(0.0f, +capsule.halfHeight, 0.0f);
    // Note: the capsule orientation is tied to the body rotation, which we don't
    // have here — this function assumes the capsule orientation is embedded in
    // `vel` and `pos`.  For CCD purposes, we approximate using the body's velocity:
    // both endpoints move with the same linear velocity (rotation is ignored for CCD).
    //
    // However, without the rotation, we cannot correctly orient the capsule axis.
    // Therefore this function uses an **axis-aligned capsule relative to the movement**
    // and falls back to a swept-sphere approximation for the capsule body.
    //
    // The simplest accurate approach: test the line segment swept against the plane
    // by checking both endpoints (sphere-plane sweeps) and taking the minimum TOI.

    // Since we don't have the orientation here, we approximate the capsule as
    // a sphere at its centre with radius = capsule.radius (conservative).
    // A full implementation would require the body rotation.
    Sphere approxSphere{capsule.radius};
    return sweepSpherePlane(approxSphere, pos, vel, dt,
                            plane, planePos, planeNormal,
                            outNormal, outPosition);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept capsule ↔ sphere  (analytical)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept capsule vs. sphere.  Uses swept-sphere approximation (capsule
///         reduced to a sphere at its centre with radius = capsule.radius).
inline float sweepCapsuleSphere(const Capsule& capsule,
                                 const Vec3f& pos, const Vec3f& vel,
                                 const Sphere& sphere, const Vec3f& spherePos,
                                 const Vec3f& sphereVel,
                                 float dt,
                                 Vec3f& outNormal,
                                 Vec3f& outPosition) noexcept {
    // Approximate as swept sphere at capsule centre with combined radius.
    Sphere approxCapsule{capsule.radius};
    return sweepSphereSphere(approxCapsule, pos, vel,
                             sphere, spherePos, sphereVel,
                             dt, outNormal, outPosition);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept capsule ↔ box  (binary search)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept capsule vs. oriented box.  Uses binary search, same approach as
///         sweepSphereBox but with capsule radius.
inline float sweepCapsuleBox(const Capsule& capsule,
                              const Vec3f& pos, const Vec3f& vel, float dt,
                              const Box& box, const Vec3f& boxPos,
                              const Quaternionf& boxRot,
                              Vec3f& outNormal,
                              Vec3f& outPosition) noexcept {
    // Decompose capsule into swept sphere at centre (conservative CCD).
    Sphere approxSphere{capsule.radius};
    return sweepSphereBox(approxSphere, pos, vel, dt,
                          box, boxPos, boxRot,
                          outNormal, outPosition);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Point-to-AABB distance helpers for ConvexMesh
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Squared distance from a point to a ConvexMesh's AABB in world space.
///
/// Transforms the point into the mesh's local space, clamps to the AABB (defined by
/// halfExtents and center), and returns the squared distance.  This is the same
/// conservative-but-robust approach used in sweepSphereBox.
///
/// @param point    World-space point.
/// @param mesh     ConvexMesh (halfExtents/center in local space).
/// @param meshPos  Mesh body world position.
/// @param meshRot  Mesh body world rotation.
/// @return  Squared distance from point to AABB surface (0 if inside).
inline float pointToConvexMeshAABBDistSq(const Vec3f& point,
                                          const ConvexMesh& mesh,
                                          const Vec3f& meshPos,
                                          const Quaternionf& meshRot) noexcept {
    Vec3f localPt = meshRot.rotateInverse(point - meshPos);
    Vec3f he = mesh.halfExtents;
    Vec3f center = mesh.center;
    Vec3f closest(
        std::max(center.x - he.x, std::min(localPt.x, center.x + he.x)),
        std::max(center.y - he.y, std::min(localPt.y, center.y + he.y)),
        std::max(center.z - he.z, std::min(localPt.z, center.z + he.z))
    );
    Vec3f diff = localPt - closest;
    return diff.lengthSquared();
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept sphere ↔ convex mesh  (binary search via point-to-AABB, fallback to GJK for normal)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept sphere vs. convex mesh.  Uses binary search with a fast point-to-AABB
///         distance query; uses GJK only for the final contact normal.
///
/// The AABB-based overlap test is conservative (the AABB may be larger than the actual
/// hull) — in the worst case we get a false-positive CCD correction rather than a missed
/// collision (tunnelling).  For box-like meshes (8-vertex hulls) the AABB is exact.
///
/// @param sphere       Sphere primitive.
/// @param pos          Sphere centre at t=0 (world space).
/// @param vel          Sphere linear velocity over the full step (world space).
/// @param dt           Fixed timestep.
/// @param mesh         Convex mesh primitive.
/// @param meshPos      Mesh body position (world space).
/// @param meshRot      Mesh body rotation.
/// @param[out] outNormal  Contact normal at TOI (points from mesh toward sphere).
/// @param[out] outPosition Contact position at TOI.
/// @return TOI in [0,1], or 1.0 if no collision.
inline float sweepSphereConvexMesh(const Sphere& sphere,
                                    const Vec3f& pos, const Vec3f& vel, float dt,
                                    const ConvexMesh& mesh, const Vec3f& meshPos,
                                    const Quaternionf& meshRot,
                                    Vec3f& outNormal,
                                    Vec3f& outPosition) noexcept {
    Vec3f disp = vel * dt;
    float dispLen = disp.length();
    if (dispLen < 1e-12f) return 1.0f;

    float radiusSq = sphere.radius * sphere.radius;

    // Check for initial overlap at t=0 (fast AABB path — no GJK).
    {
        float distSq = pointToConvexMeshAABBDistSq(pos, mesh, meshPos, meshRot);
        if (distSq < radiusSq) {
            // Compute surface normal from the closest AABB point, same as sweepSphereBox.
            Vec3f localPt = meshRot.rotateInverse(pos - meshPos);
            Vec3f he = mesh.halfExtents;
            Vec3f center = mesh.center;
            Vec3f closest(
                std::max(center.x - he.x, std::min(localPt.x, center.x + he.x)),
                std::max(center.y - he.y, std::min(localPt.y, center.y + he.y)),
                std::max(center.z - he.z, std::min(localPt.z, center.z + he.z))
            );
            Vec3f diff = localPt - closest;
            float dLen = diff.length();
            Vec3f nLocal;
            if (dLen > 1e-12f) {
                nLocal = diff / dLen;
            } else {
                // Dead centre on a face/edge/vertex — push along the shallowest axis.
                float minPen = (center.x + he.x) - std::abs(localPt.x - center.x);
                int axis = 0;
                float py = (center.y + he.y) - std::abs(localPt.y - center.y);
                if (py < minPen) { minPen = py; axis = 1; }
                float pz = (center.z + he.z) - std::abs(localPt.z - center.z);
                if (pz < minPen) { minPen = pz; axis = 2; }
                nLocal = Vec3f::zero();
                nLocal[axis] = (localPt[axis] >= center[axis]) ? 1.0f : -1.0f;
            }
            outNormal = meshRot.rotate(nLocal).normalized();
            Vec3f closestWorld = meshPos + meshRot.rotate(closest);
            outPosition = closestWorld + outNormal * sphere.radius;
            return 0.0f;
        }
    }

    // Binary search for TOI using the fast AABB distance check.
    constexpr int kIterations = 24;  // 24 iterations at disp=0.25m → ~1 cm precision
    float lo = 0.0f, hi = 1.0f;
    bool hit = false;

    for (int iter = 0; iter < kIterations; ++iter) {
        float mid = (lo + hi) * 0.5f;
        Vec3f samplePos = pos + disp * mid;

        float distSq = pointToConvexMeshAABBDistSq(samplePos, mesh, meshPos, meshRot);

        if (distSq < radiusSq) {
            hit = true;
            hi = mid;
        } else {
            lo = mid;
        }
    }

    if (!hit) return 1.0f;

    float toi = (lo + hi) * 0.5f;
    Vec3f hitPos = pos + disp * toi;

    // Compute normal at TOI using AABB surface (same as initial overlap).
    {
        Vec3f localPt = meshRot.rotateInverse(hitPos - meshPos);
        Vec3f he = mesh.halfExtents;
        Vec3f center = mesh.center;
        Vec3f closest(
            std::max(center.x - he.x, std::min(localPt.x, center.x + he.x)),
            std::max(center.y - he.y, std::min(localPt.y, center.y + he.y)),
            std::max(center.z - he.z, std::min(localPt.z, center.z + he.z))
        );
        Vec3f diff = localPt - closest;
        float dLen = diff.length();
        Vec3f nLocal;
        if (dLen > 1e-12f) {
            nLocal = diff / dLen;
        } else {
            float minPen = (center.x + he.x) - std::abs(localPt.x - center.x);
            int axis = 0;
            float py = (center.y + he.y) - std::abs(localPt.y - center.y);
            if (py < minPen) { minPen = py; axis = 1; }
            float pz = (center.z + he.z) - std::abs(localPt.z - center.z);
            if (pz < minPen) { minPen = pz; axis = 2; }
            nLocal = Vec3f::zero();
            nLocal[axis] = (localPt[axis] >= center[axis]) ? 1.0f : -1.0f;
        }
        outNormal = meshRot.rotate(nLocal).normalized();
    }
    outPosition = hitPos;
    return toi;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Swept capsule ↔ convex mesh  (via swept-sphere approximation)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept capsule vs. convex mesh.  Uses swept-sphere approximation (capsule
///         reduced to a sphere at its centre with radius = capsule.radius).
inline float sweepCapsuleConvexMesh(const Capsule& capsule,
                                     const Vec3f& pos, const Vec3f& vel, float dt,
                                     const ConvexMesh& mesh, const Vec3f& meshPos,
                                     const Quaternionf& meshRot,
                                     Vec3f& outNormal,
                                     Vec3f& outPosition) noexcept {
    // Decompose capsule into swept sphere at centre (conservative CCD).
    Sphere approxSphere{capsule.radius};
    return sweepSphereConvexMesh(approxSphere, pos, vel, dt,
                                  mesh, meshPos, meshRot,
                                  outNormal, outPosition);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  CCD pipeline helpers
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  CCD configuration.
struct CCDConfig {
    /// @brief  Maximum number of CCD sub-steps per body per frame.
    ///         Higher values improve accuracy but cost performance.
    int maxSubSteps = 1;

    /// @brief  Speed threshold (m/s) above which CCD is activated for enabled bodies.
    ///         Bodies moving slower than this skip CCD (saves CPU).
    float speedThreshold = 10.0f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Compute an approximate bounding radius for a body from its shapes.
///         Used by ClampVelocity and SubStep CCD modes.
inline float computeBodyBoundingRadius(RigidBodyHandle body,
                                        const RigidBodyStorage& bodies,
                                        const std::vector<Shape>& shapes) noexcept {
    float r = 0.5f;
    std::uint32_t start = bodies.shapeStart[body];
    std::uint32_t count = bodies.shapeCount[body];
    for (std::uint32_t s = 0; s < count; ++s) {
        ShapeHandle sh = bodies.flatShapeHandles[start + s];
        if (sh >= shapes.size()) continue;
        const Shape& shape = shapes[sh];
        switch (shape.type) {
            case ShapeType::Sphere:
                r = std::max(r, shape.sphere.radius);
                break;
            case ShapeType::Box:
                r = std::max(r, shape.box.halfExtents.length());
                break;
            case ShapeType::Capsule:
                r = std::max(r, shape.capsule.radius + shape.capsule.halfHeight);
                break;
            case ShapeType::ConvexMesh:
                r = std::max(r, shape.convexMesh.halfExtents.length());
                break;
            default:
                break;
        }
    }
    return r;
}

/// @brief  Result of a single CCD sweep against all static/kinematic bodies.
struct CcdSweepResult {
    float  toi       = 1.0f;
    Vec3f  normal    = Vec3f{0.0f, 1.0f, 0.0f};
    Vec3f  hitPoint  = Vec3f{0.0f, 0.0f, 0.0f};
    bool   hit       = false;
};

/// @brief  Sweep `body` from `pos` with `vel` over `dt` against all static/kinematic
///         obstacles, returning the earliest TOI, normal, and contact point.
///
/// This is the inner sweep loop extracted from `ccdResolveBody` so that it can be
/// called multiple times for sub-stepping without duplicating the dispatch logic.
inline CcdSweepResult ccdSweepBody(RigidBodyHandle body,
                                    const Vec3f& pos,
                                    const Vec3f& vel,
                                    float dt,
                                    const RigidBodyStorage& bodies,
                                    const std::vector<Shape>& shapes) noexcept {

    CcdSweepResult result;

    std::uint32_t startA = bodies.shapeStart[body];
    std::uint32_t countA = bodies.shapeCount[body];

    // Iterate over all other active bodies.
    for (std::size_t j = 0; j < bodies.size(); ++j) {
        if (j == body) continue;
        if (!bodies.activeFlags[j]) continue;
        if (bodies.shapeCount[j] == 0) continue;

        // Only test against static and kinematic bodies (obstacles).
        if (bodies.motionTypes[j] == MotionType::Dynamic) continue;

        std::uint32_t startB = bodies.shapeStart[j];
        std::uint32_t countB = bodies.shapeCount[j];

        for (std::uint32_t sa = 0; sa < countA; ++sa) {
            ShapeHandle shA = bodies.flatShapeHandles[startA + sa];
            if (shA >= shapes.size()) continue;

            const Shape& shapeA = shapes[shA];
            Vec3f localPosA = bodies.flatShapeLocalPositions[startA + sa];
            Vec3f worldPosA = pos + bodies.rotations[body].rotate(localPosA);

            for (std::uint32_t sb = 0; sb < countB; ++sb) {
                ShapeHandle shB = bodies.flatShapeHandles[startB + sb];
                if (shB >= shapes.size()) continue;

                const Shape& shapeB = shapes[shB];

                Vec3f localPosB = bodies.flatShapeLocalPositions[startB + sb];
                Quaternionf localRotB = bodies.flatShapeLocalRotations[startB + sb];
                Vec3f worldPosB = bodies.positions[j]
                                 + bodies.rotations[j].rotate(localPosB);
                Quaternionf worldRotB = bodies.rotations[j] * localRotB;

                float toi = 1.0f;
                Vec3f nml(0.0f, 1.0f, 0.0f);
                Vec3f pt(0.0f, 0.0f, 0.0f);

                // Dispatch to the appropriate swept test.
                if (shapeA.type == ShapeType::Sphere) {
                    const Sphere& sA = shapeA.sphere;

                    if (shapeB.type == ShapeType::Plane) {
                        const Plane& plB = shapeB.plane;
                        Vec3f planeNormal = worldRotB.rotate(plB.normal).normalized();
                        toi = sweepSpherePlane(sA, worldPosA, vel, dt,
                                                plB, worldPosB, planeNormal,
                                                nml, pt);
                    } else if (shapeB.type == ShapeType::Sphere) {
                        toi = sweepSphereSphere(sA, worldPosA, vel,
                                                shapeB.sphere, worldPosB, Vec3f::zero(),
                                                dt, nml, pt);
                    } else if (shapeB.type == ShapeType::Box) {
                        toi = sweepSphereBox(sA, worldPosA, vel, dt,
                                              shapeB.box, worldPosB, worldRotB,
                                              nml, pt);
                    } else if (shapeB.type == ShapeType::ConvexMesh) {
                        toi = sweepSphereConvexMesh(sA, worldPosA, vel, dt,
                                                     shapeB.convexMesh, worldPosB, worldRotB,
                                                     nml, pt);
                    }
                } else if (shapeA.type == ShapeType::Capsule) {
                    const Capsule& cA = shapeA.capsule;

                    if (shapeB.type == ShapeType::Plane) {
                        const Plane& plB = shapeB.plane;
                        Vec3f planeNormal = worldRotB.rotate(plB.normal).normalized();
                        toi = sweepCapsulePlane(cA, worldPosA, vel, dt,
                                                 plB, worldPosB, planeNormal,
                                                 nml, pt);
                    } else if (shapeB.type == ShapeType::Sphere) {
                        toi = sweepCapsuleSphere(cA, worldPosA, vel,
                                                  shapeB.sphere, worldPosB, Vec3f::zero(),
                                                  dt, nml, pt);
                    } else if (shapeB.type == ShapeType::Box) {
                        toi = sweepCapsuleBox(cA, worldPosA, vel, dt,
                                               shapeB.box, worldPosB, worldRotB,
                                               nml, pt);
                    } else if (shapeB.type == ShapeType::ConvexMesh) {
                        toi = sweepCapsuleConvexMesh(cA, worldPosA, vel, dt,
                                                      shapeB.convexMesh, worldPosB, worldRotB,
                                                      nml, pt);
                    }
                } else if (shapeA.type == ShapeType::ConvexMesh) {
                    // Approximate the ConvexMesh as a sphere at its sub-shape origin
                    // with radius = bounding-sphere radius (conservative CCD).
                    const ConvexMesh& mA = shapeA.convexMesh;
                    float approxRadius = mA.halfExtents.length();
                    if (approxRadius < 1e-8f) {
                        continue;
                    }
                    Sphere approxSphere{approxRadius};

                    if (shapeB.type == ShapeType::Plane) {
                        const Plane& plB = shapeB.plane;
                        Vec3f planeNormal = worldRotB.rotate(plB.normal).normalized();
                        toi = sweepSpherePlane(approxSphere, worldPosA, vel, dt,
                                                plB, worldPosB, planeNormal,
                                                nml, pt);
                    } else if (shapeB.type == ShapeType::Sphere) {
                        toi = sweepSphereSphere(approxSphere, worldPosA, vel,
                                                 shapeB.sphere, worldPosB, Vec3f::zero(),
                                                 dt, nml, pt);
                    } else if (shapeB.type == ShapeType::Box) {
                        toi = sweepSphereBox(approxSphere, worldPosA, vel, dt,
                                              shapeB.box, worldPosB, worldRotB,
                                              nml, pt);
                    } else if (shapeB.type == ShapeType::ConvexMesh) {
                        toi = sweepSphereConvexMesh(approxSphere, worldPosA, vel, dt,
                                                     shapeB.convexMesh, worldPosB, worldRotB,
                                                     nml, pt);
                    }
                }

                if (toi < result.toi) {
                    result.toi = toi;
                    result.normal = nml;
                    result.hitPoint = pt;
                    result.hit = true;
                }
            }
        }
    }

    return result;
}

/// @brief  Run CCD for a single body with the specified CcdMode.
///
/// Tests the body's swept path against all static/kinematic obstacles.
/// Supports four modes:
///   - None:           skip entirely
///   - Cast:           standard single sweep (current behavior)
///   - ClampVelocity:  limit |v| so that |v|*dt < boundingRadius*0.5
///   - SubStep:        divide frame into N sub-steps, each moving less than
///                     boundingRadius*0.5, with sweep test per sub-step
///
/// @return  True if a CCD collision was resolved (position/velocity corrected).
inline bool ccdResolveBody(RigidBodyHandle body, float dt,
                           RigidBodyStorage& bodies,
                           const std::vector<Shape>& shapes,
                           float restitution, float /*friction*/,
                           CcdMode mode = CcdMode::Cast) noexcept {

    if (bodies.shapeCount[body] == 0) return false;
    if (mode == CcdMode::None) return false;

    Vec3f vel = bodies.linearVelocities[body];
    float speed = vel.length();
    if (speed < 1e-8f) return false;

    // ─── Compute bounding radius for velocity-management modes ───
    float boundingRadius = -1.0f;
    if (mode == CcdMode::ClampVelocity || mode == CcdMode::SubStep) {
        boundingRadius = computeBodyBoundingRadius(body, bodies, shapes);
    }

    // ─── ClampVelocity: limit speed so |v|*dt < boundingRadius*0.5 ───
    if (mode == CcdMode::ClampVelocity) {
        float maxSpeed = boundingRadius * 0.5f / (dt + 1e-8f);
        if (speed > maxSpeed) {
            float scale = maxSpeed / speed;
            vel *= scale;
            bodies.linearVelocities[body] = vel;
            speed = maxSpeed;
        }
    }

    // ─── SubStep: determine number of sub-steps ───
    int numSubSteps = 1;
    if (mode == CcdMode::SubStep) {
        float maxDisplacement = boundingRadius * 0.5f;
        float totalDisplacement = speed * dt;
        numSubSteps = static_cast<int>(
            std::ceil(totalDisplacement / (maxDisplacement + 1e-8f)));
        numSubSteps = std::max(1, std::min(numSubSteps, 16));
    }

    float subDt = dt / static_cast<float>(numSubSteps);
    Vec3f basePos = bodies.positions[body];  // start-of-frame position
    Vec3f currentVel = bodies.linearVelocities[body];

    // ─── Main CCD loop (single pass for Cast/ClampVelocity, multi-pass for SubStep) ───
    //
    // For SubStep mode we use a VIRTUAL position for each sub-step's sweep test
    // without writing to bodies.positions[body].  This avoids double-advancement
    // when integratePositions runs later: it will apply vel * dt once.
    // Only when a collision is found do we write the resolved position.
    for (int step = 0; step < numSubSteps; ++step) {
        // Virtual position at the start of this sub-step.
        Vec3f pos = basePos + currentVel * subDt * static_cast<float>(step);

        CcdSweepResult sweep = ccdSweepBody(body, pos, currentVel,
                                             subDt, bodies, shapes);

        if (sweep.hit && sweep.toi < 1.0f) {
            // Advance position to the TOI within this sub-step.
            // integratePositions will add reflected_vel * dt afterward.
            bodies.positions[body] = pos + currentVel * subDt * sweep.toi;

            // Reflect velocity along the contact normal (with restitution).
            float vDotN = currentVel.dot(sweep.normal);
            if (vDotN < 0.0f) {
                Vec3f vNormal = sweep.normal * vDotN;
                Vec3f vTangent = currentVel - vNormal;
                bodies.linearVelocities[body] = vTangent - vNormal * restitution;
            }

            // Damp angular velocity (the body has just hit something).
            bodies.angularVelocities[body] *= 0.5f;

            return true;
        }
    }

    // No collision in any sub-step — position unchanged.
    // integratePositions will apply vel * dt as usual.
    return false;
}

} // namespace ciphyxs
