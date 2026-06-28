//==================================================================================================
/// @file  Ray.hpp
/// @brief  Ray structure, hit result, and per-primitive ray-cast functions.
///
/// Each shape type has a corresponding `rayCast` function that tests a ray against
/// that primitive in **local space**.  The `PhysicsWorld::rayCast` method transforms
/// the ray into each body's local space, tests, and returns the closest hit.
//==================================================================================================
#pragma once

#include "Shape.hpp"
#include "Convex.hpp"
#include "Bvh.hpp"
#include "../core/RigidBody.hpp"
#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Ray
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A ray defined by an origin and a direction (does not need to be unit length).
struct Ray {
    Vec3f origin;     ///< World-space origin of the ray.
    Vec3f direction;  ///< Direction of the ray (does NOT need to be normalized).

    /// @brief  Point on the ray at parameter `t` (t >= 0).
    [[nodiscard]] Vec3f pointAt(float t) const noexcept {
        return origin + direction * t;
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RayHit
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Result of a successful ray cast.
struct RayHit {
    float           t        = std::numeric_limits<float>::max();  ///< Ray parameter at hit.
    Vec3f           point    = Vec3f::zero();                      ///< World-space hit position.
    Vec3f           normal   = Vec3f::zero();                      ///< World-space surface normal at hit.
    RigidBodyHandle body     = kInvalidHandle;                     ///< Body that was hit.
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Per-primitive ray-cast functions (local-space)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Ray vs sphere in local space.  Sphere is centered at origin with given radius.
/// @return True if the ray hits the sphere within [0, maxT].
inline bool rayCastSphere(const Vec3f& rayOrigin, const Vec3f& rayDir,
                          float radius, float maxT, float& outT, Vec3f& outNormal) noexcept {
    Vec3f oc  = rayOrigin;  // sphere at origin
    float a   = rayDir.dot(rayDir);
    float b   = 2.0f * oc.dot(rayDir);
    float c   = oc.dot(oc) - radius * radius;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return false;

    float sqrtDisc = std::sqrt(disc);
    float tMinus   = (-b - sqrtDisc) / (2.0f * a);
    float tPlus    = (-b + sqrtDisc) / (2.0f * a);

    // Find the smallest t >= 0 that is a valid hit.
    float t = (tMinus >= 0.0f) ? tMinus : tPlus;
    if (t < 0.0f || t > maxT) return false;

    outT         = t;
    Vec3f hitPt  = rayOrigin + rayDir * t;
    outNormal    = hitPt.normalized();
    return true;
}

/// @brief  Ray vs AABB (aligned box) in local space.  Box is centered at origin
///         with given half-extents.  Uses the slab method (Williams et al.).
/// @return True if the ray hits the box within [0, maxT].
inline bool rayCastBox(const Vec3f& rayOrigin, const Vec3f& rayDir,
                       const Vec3f& halfExtents, float maxT,
                       float& outT, Vec3f& outNormal) noexcept {
    float tMin = 0.0f;
    float tMax = maxT;

    int hitAxis  = 0;
    bool hitSide = false;  // true = +face, false = -face

    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / rayDir[i];
        float t0   = (-halfExtents[i] - rayOrigin[i]) * invD;
        float t1   = (+halfExtents[i] - rayOrigin[i]) * invD;
        if (invD < 0.0f) std::swap(t0, t1);

        if (t0 > tMin) { tMin = t0; hitAxis = i; hitSide = (rayDir[i] < 0.0f); }
        if (t1 < tMax) { tMax = t1; }

        if (tMin > tMax) return false;
    }

    if (tMin < 0.0f || tMin > maxT) return false;

    outT = tMin;
    outNormal = Vec3f::zero();
    outNormal[hitAxis] = hitSide ? 1.0f : -1.0f;
    return true;
}

/// @brief  Ray vs infinite plane in local space.
///         Plane local equation: `normal · p = distance`.
/// @return True if the ray hits the plane within [0, maxT].
inline bool rayCastPlane(const Vec3f& rayOrigin, const Vec3f& rayDir,
                         const Vec3f& planeNormal, float planeDistance,
                         float maxT, float& outT, Vec3f& outNormal) noexcept {
    float denom = rayDir.dot(planeNormal);
    if (std::abs(denom) < 1e-12f) return false;  // parallel

    float t = (planeDistance - rayOrigin.dot(planeNormal)) / denom;
    if (t < 0.0f || t > maxT) return false;

    outT      = t;
    outNormal = planeNormal;
    return true;
}

/// @brief  Ray vs capsule in local space (capsule aligned to Y axis).
///         Capsule is a line segment from (0, -halfHeight, 0) to (0, +halfHeight, 0)
///         with `radius` around it.
/// @return True if the ray hits the capsule within [0, maxT].
inline bool rayCastCapsule(const Vec3f& rayOrigin, const Vec3f& rayDir,
                           float radius, float halfHeight,
                           float maxT, float& outT, Vec3f& outNormal) noexcept {
    // Test ray against the infinite cylinder first, then clamp to segment.
    // Cylinder equation: (x² + z²) = radius², for y in [-halfHeight, halfHeight].
    //
    // For a ray p(t) = ro + rd * t, we need points where:
    //   (rox + rdx*t)² + (roz + rdz*t)² = radius²
    //
    // This is: a*t² + b*t + c = 0
    // where a = rdx² + rdz², b = 2*(rox*rdx + roz*rdz), c = rox² + roz² - radius²

    float a = rayDir.x * rayDir.x + rayDir.z * rayDir.z;
    float b = 2.0f * (rayOrigin.x * rayDir.x + rayOrigin.z * rayDir.z);
    float c = rayOrigin.x * rayOrigin.x + rayOrigin.z * rayOrigin.z - radius * radius;

    float tCyl = std::numeric_limits<float>::max();

    if (std::abs(a) > 1e-12f) {
        float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) {
            // No intersection with infinite cylinder — test the spherical caps.
            goto testCaps;
        }

        float sqrtDisc = std::sqrt(disc);
        float t1 = (-b - sqrtDisc) / (2.0f * a);
        float t2 = (-b + sqrtDisc) / (2.0f * a);
        if (t1 > t2) std::swap(t1, t2);

        // Check the first valid t where y is within [-halfHeight, halfHeight].
        if (t1 >= 0.0f) {
            float y1 = rayOrigin.y + rayDir.y * t1;
            if (y1 >= -halfHeight && y1 <= halfHeight) {
                tCyl = t1;
            }
        }
        if (tCyl > maxT && t2 >= 0.0f) {
            float y2 = rayOrigin.y + rayDir.y * t2;
            if (y2 >= -halfHeight && y2 <= halfHeight) {
                tCyl = t2;
            }
        }
    }

testCaps:
    // --- Test the spherical end caps ---
    // Top cap at (0, +halfHeight, 0), bottom cap at (0, -halfHeight, 0).
    float tSphere = std::numeric_limits<float>::max();
    Vec3f sphereNormal;

    // Bottom sphere cap.
    Vec3f bottom(0.0f, -halfHeight, 0.0f);
    {
        Vec3f oc = rayOrigin - bottom;
        float sa = rayDir.dot(rayDir);
        float sb = 2.0f * oc.dot(rayDir);
        float sc = oc.dot(oc) - radius * radius;
        float sdisc = sb * sb - 4.0f * sa * sc;
        if (sdisc >= 0.0f) {
            float sqrtSd = std::sqrt(sdisc);
            float st1 = (-sb - sqrtSd) / (2.0f * sa);
            float st2 = (-sb + sqrtSd) / (2.0f * sa);
            float st = (st1 >= 0.0f) ? st1 : st2;
            if (st >= 0.0f && st < maxT) {
                Vec3f hitP = rayOrigin + rayDir * st;
                // Only valid if hit is on the bottom hemisphere (y <= -halfHeight)
                // and not on the cylinder body.
                if (hitP.y <= -halfHeight && hitP.y >= -halfHeight - radius) {
                    // Check that this hit is on the hemisphere, not the cylinder.
                    Vec3f toCenter = hitP - bottom;
                    // The hemisphere is the half where the dot with the axial direction (0,-1,0) is positive.
                    // Actually for bottom cap, we want points where the normal points downward-ish.
                    // Simple check: the angle between (hit - center) and (0,-1,0) should be < 90°.
                    if (toCenter.dot(Vec3f(0.0f, -1.0f, 0.0f)) > 0.0f) {
                        // Check that this intersection is outside the cylinder body:
                        // the hit should be below -halfHeight.
                        if (st < tSphere) {
                            tSphere = st;
                            sphereNormal = toCenter.normalized();
                        }
                    }
                }
            }
        }
    }

    // Top sphere cap.
    Vec3f top(0.0f, halfHeight, 0.0f);
    {
        Vec3f oc = rayOrigin - top;
        float sa = rayDir.dot(rayDir);
        float sb = 2.0f * oc.dot(rayDir);
        float sc = oc.dot(oc) - radius * radius;
        float sdisc = sb * sb - 4.0f * sa * sc;
        if (sdisc >= 0.0f) {
            float sqrtSd = std::sqrt(sdisc);
            float st1 = (-sb - sqrtSd) / (2.0f * sa);
            float st2 = (-sb + sqrtSd) / (2.0f * sa);
            float st = (st1 >= 0.0f) ? st1 : st2;
            if (st >= 0.0f && st < maxT) {
                Vec3f hitP = rayOrigin + rayDir * st;
                if (hitP.y >= halfHeight && hitP.y <= halfHeight + radius) {
                    Vec3f toCenter = hitP - top;
                    if (toCenter.dot(Vec3f(0.0f, 1.0f, 0.0f)) > 0.0f) {
                        if (st < tSphere) {
                            tSphere = st;
                            sphereNormal = toCenter.normalized();
                        }
                    }
                }
            }
        }
    }

    // --- Choose the closest valid hit ---
    if (tCyl <= maxT && tCyl < tSphere) {
        outT = tCyl;
        // Normal at cylinder hit is the radial direction in the XZ plane.
        Vec3f hitPt = rayOrigin + rayDir * tCyl;
        outNormal = Vec3f(hitPt.x, 0.0f, hitPt.z).normalized();
        return true;
    }
    if (tSphere <= maxT) {
        outT = tSphere;
        outNormal = sphereNormal;
        return true;
    }

    return false;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  ConvexMesh ray cast  (GJK-based binary search)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Ray vs ConvexMesh in local space using conservative advancement.
///
/// Iteratively advances the ray toward the hull surface by computing the GJK
/// distance from the current ray point to the hull.  Converges linearly and is
/// robust for any convex hull regardless of ray orientation.
///
/// @param rayOrigin  Ray origin in local space.
/// @param rayDir     Ray direction in local space (normalized).
/// @param vertices   Convex mesh vertex array (local space).
/// @param vertexCount  Number of vertices.
/// @param maxT       Maximum ray parameter.
/// @param[out] outT      Ray parameter at hit.
/// @param[out] outNormal Surface normal in local space.
/// @return True if the ray hits the convex hull.
///
/// @note  Performance: O(iterations × GJK-iterations × vertices).  For heavy use,
///        precompute face planes and use analytic ray-triangle intersection instead.
inline bool rayCastConvexMesh(const Vec3f& rayOrigin, const Vec3f& rayDir,
                               const Vec3f* vertices, int vertexCount,
                               float maxT, float& outT, Vec3f& outNormal) noexcept {
    if (vertexCount < 1 || !vertices) return false;

    // Compute AABB from vertices (for fast initial distance estimation).
    Vec3f bbMin( 1e10f), bbMax(-1e10f);
    for (int i = 0; i < vertexCount; ++i) {
        bbMin.x = std::min(bbMin.x, vertices[i].x);
        bbMin.y = std::min(bbMin.y, vertices[i].y);
        bbMin.z = std::min(bbMin.z, vertices[i].z);
        bbMax.x = std::max(bbMax.x, vertices[i].x);
        bbMax.y = std::max(bbMax.y, vertices[i].y);
        bbMax.z = std::max(bbMax.z, vertices[i].z);
    }

    // Support function for the convex mesh.
    auto supportMesh = [&](const Vec3f& d) -> Vec3f {
        float bestDot = -std::numeric_limits<float>::max();
        const Vec3f* bestPt = vertices;
        for (int i = 0; i < vertexCount; ++i) {
            float d2 = vertices[i].dot(d);
            if (d2 > bestDot) { bestDot = d2; bestPt = vertices + i; }
        }
        return *bestPt;
    };

    // Helper: returns true if point is inside (or on) the convex hull AABB.
    auto pointInAABB = [&](const Vec3f& pt) -> bool {
        return pt.x >= bbMin.x && pt.x <= bbMax.x &&
               pt.y >= bbMin.y && pt.y <= bbMax.y &&
               pt.z >= bbMin.z && pt.z <= bbMax.z;
    };

    // Compute the closest point on AABB to a given point.
    auto closestAABBPoint = [&](const Vec3f& pt) -> Vec3f {
        Vec3f c;
        c.x = std::max(bbMin.x, std::min(pt.x, bbMax.x));
        c.y = std::max(bbMin.y, std::min(pt.y, bbMax.y));
        c.z = std::max(bbMin.z, std::min(pt.z, bbMax.z));
        return c;
    };

    // Check if origin is already inside the hull.
    // Use a two-stage test: AABB first (fast), then GJK (accurate).
    bool originInside = false;
    if (pointInAABB(rayOrigin)) {
        // Refine with GJK point-in-hull test.
        auto supportPt = [&](const Vec3f& /*d*/) -> Vec3f { return rayOrigin; };
        gjk_detail::Simplex simplex;
        Vec3f pA, pB;
        float distSq = gjk_detail::gjkDistance(supportPt, supportMesh,
                                                pA, pB, simplex);
        originInside = (distSq <= gjk_detail::kGJKTolerance * gjk_detail::kGJKTolerance);
    }

    if (originInside) {
        // Compute normal using the closest point on hull to the origin.
        gjk_detail::Simplex simplex;
        Vec3f pA, pB;
        gjk_detail::gjkDistance(
            [&](const Vec3f& /*d*/) -> Vec3f { return rayOrigin; },
            supportMesh, pA, pB, simplex);
        Vec3f diff = rayOrigin - pB;
        float len = diff.length();
        outT = 0.0f;
        outNormal = (len > 1e-12f) ? (diff / len) : Vec3f::unitY();
        return true;
    }

    // Conservative advancement with AABB-assisted GJK initialization.
    constexpr int kMaxIter = 64;
    constexpr float kTol = 1e-6f;
    float t = 0.0f;

    for (int iter = 0; iter < kMaxIter; ++iter) {
        Vec3f rayPt = rayOrigin + rayDir * t;

        // Compute direction from rayPt toward hull using AABB for initial guess.
        Vec3f aabbClosest = closestAABBPoint(rayPt);
        Vec3f initialDir = aabbClosest - rayPt;
        float initLenSq = initialDir.lengthSquared();

        Vec3f pA, pB;
        gjk_detail::Simplex simplex;
        float distSq;

        if (initLenSq < 1e-12f) {
            // Already at AABB surface — use standard GJK.
            distSq = gjk_detail::gjkDistance(
                [&](const Vec3f& /*d*/) -> Vec3f { return rayPt; },
                supportMesh, pA, pB, simplex);
        } else {
            // Use direction toward AABB as initial GJK direction.
            // We need a specialized GJK that accepts an initial direction.
            // For simplicity, fall back to standard GJK with more iterations.
            distSq = gjk_detail::gjkDistance(
                [&](const Vec3f& /*d*/) -> Vec3f { return rayPt; },
                supportMesh, pA, pB, simplex);
        }

        float dist = std::sqrt(distSq);
        if (dist < kTol) {
            Vec3f diff = rayPt - pB;
            float len = diff.length();
            outNormal = (len > 1e-12f) ? (diff / len) : Vec3f::unitY();
            outT = t;
            return true;
        }

        // Use the AABB-based distance as a fallback if GJK is unreliable.
        // The AABB distance is a conservative estimate.
        Vec3f toAABB = aabbClosest - rayPt;
        float aabbDist = toAABB.length();

        // Use AABB distance for advancement if GJK returned something unrealistic.
        // A reasonable GJK distance for a point outside the AABB should be >= AABB distance.
        // But GJK can return too-large values if it hasn't converged.
        // We use min(dist, aabbDist) for conservative advancement.
        float effectiveDist = std::min(dist, aabbDist + kTol);
        if (effectiveDist < kTol) effectiveDist = aabbDist;

        Vec3f toHullDir;
        float toHullLen;
        if (aabbDist < dist && aabbDist > kTol) {
            // AABB gave a better direction.
            toHullDir = toAABB / aabbDist;
            toHullLen = aabbDist;
        } else if (dist > kTol) {
            Vec3f toHull = pB - rayPt;
            toHullLen = toHull.length();
            toHullDir = toHull / toHullLen;
        } else {
            break;
        }

        float cosAngle = std::abs(rayDir.dot(toHullDir));
        if (cosAngle < 1e-12f) return false;

        float advance = std::max(effectiveDist / cosAngle, 1e-7f);
        t += advance;

        if (t > maxT) return false;
    }

    return false;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Ray vs TriangleMesh  (BVH-accelerated, with linear fallback)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Ray vs TriangleMesh intersection (BVH-accelerated).
/// @param origin   Ray origin (local space).
/// @param dir      Ray direction (local space).
/// @param mesh     The triangle mesh.
/// @param maxT     Maximum hit distance.
/// @param outT     Hit distance.
/// @param outNormal Hit normal (local space).
/// @return true if the ray hits any triangle.
inline bool rayCastTriangleMesh(const Vec3f& origin, const Vec3f& dir,
                                 const TriangleMesh& mesh, float maxT,
                                 float& outT, Vec3f& outNormal) noexcept {
    if (!mesh.bvh || mesh.triangleCount <= 0) {
        // Without BVH, fall back to a slow linear scan.
        float closestT = maxT;
        bool hit = false;
        for (int i = 0; i < mesh.triangleCount; ++i) {
            const Vec3f& v0 = mesh.vertices[mesh.indices[i * 3 + 0]];
            const Vec3f& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
            const Vec3f& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];
            // Inline Möller-Trumbore ray-triangle intersection.
            constexpr float kEps = 1e-12f;
            Vec3f edge1 = v1 - v0;
            Vec3f edge2 = v2 - v0;
            Vec3f h = dir.cross(edge2);
            float a = edge1.dot(h);
            if (std::abs(a) < kEps) continue;
            float f = 1.0f / a;
            Vec3f s = origin - v0;
            float u_ = f * s.dot(h);
            if (u_ < 0.0f || u_ > 1.0f) continue;
            Vec3f q = s.cross(edge1);
            float v_ = f * dir.dot(q);
            if (v_ < 0.0f || u_ + v_ > 1.0f) continue;
            float t_ = f * edge2.dot(q);
            if (t_ < 0.0f || t_ > closestT) continue;
            closestT = t_;
            outT = t_;
            outNormal = edge1.cross(edge2);
            if (outNormal.dot(dir) > 0.0f) outNormal = -outNormal;
            float nl = outNormal.length();
            if (nl > kEps) outNormal = outNormal / nl;
            hit = true;
        }
        return hit;
    }
    int triIdx;
    return mesh.bvh->rayCast(origin, dir, maxT,
                             mesh.vertices, mesh.indices,
                             outT, outNormal, triIdx);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Shape-dispatched ray cast
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Test a ray against a shape in world space.
///
/// @param shape      The shape to test.
/// @param bodyPos    World-space position of the body.
/// @param bodyRot    World-space rotation of the body.
/// @param rayOrigin  Ray origin in world space.
/// @param rayDir     Ray direction in world space (does not need to be normalized).
/// @param maxT       Maximum ray parameter (closest hit further than this is ignored).
/// @param[out] outT      Hit parameter (world-space ray).
/// @param[out] outNormal World-space surface normal at hit.
/// @return True if the ray hit the shape.
inline bool rayCastShape(const Shape& shape,
                         const Vec3f& bodyPos, const Quaternionf& bodyRot,
                         const Vec3f& rayOrigin, const Vec3f& rayDir,
                         float maxT,
                         float& outT, Vec3f& outNormal) noexcept {
    // Transform ray into local space of the body.
    Vec3f localOrigin = bodyRot.rotateInverse(rayOrigin - bodyPos);
    Vec3f localDir    = bodyRot.rotateInverse(rayDir);

    float localT = 0.0f;
    Vec3f localNormal;

    bool hit = false;
    switch (shape.type) {
    case ShapeType::Sphere:
        hit = rayCastSphere(localOrigin, localDir, shape.sphere.radius, maxT, localT, localNormal);
        break;
    case ShapeType::Box:
        hit = rayCastBox(localOrigin, localDir, shape.box.halfExtents, maxT, localT, localNormal);
        break;
    case ShapeType::Plane:
        hit = rayCastPlane(localOrigin, localDir, shape.plane.normal, shape.plane.distance,
                           maxT, localT, localNormal);
        break;
    case ShapeType::Capsule:
        hit = rayCastCapsule(localOrigin, localDir, shape.capsule.radius, shape.capsule.halfHeight,
                             maxT, localT, localNormal);
        break;
    case ShapeType::ConvexMesh:
        hit = rayCastConvexMesh(localOrigin, localDir,
                                shape.convexMesh.vertices, shape.convexMesh.vertexCount,
                                maxT, localT, localNormal);
        break;
    case ShapeType::TriangleMesh:
        hit = rayCastTriangleMesh(localOrigin, localDir, shape.triangleMesh,
                                  maxT, localT, localNormal);
        break;
    }

    if (!hit) return false;

    // Convert local t to world-space t (scale by direction length ratio).
    // localDir length = |R⁻¹ · worldDir| = |worldDir| (rotation preserves length).
    // So localT == worldT when directions are not normalized.
    outT      = localT;
    outNormal = bodyRot.rotate(localNormal).normalized();
    return true;
}

} // namespace ciphyxs
