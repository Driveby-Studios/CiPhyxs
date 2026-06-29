//==================================================================================================
/// @file  Narrowphase.hpp
/// @brief  Discrete collision detection between primitive shapes.
///
/// Implemented pair types:
///   - Sphere ↔ Sphere
///   - Sphere ↔ Plane
///   - Sphere ↔ Box
///   - Box    ↔ Plane
///   - Box    ↔ Box     (full SAT + face clipping)
///   - Capsule ↔ Sphere
///   - Capsule ↔ Capsule
///   - Capsule ↔ Box
///   - Capsule ↔ Plane
//==================================================================================================
#pragma once

#include "Contact.hpp"
#include "Shape.hpp"
#include "Convex.hpp"
#include "Bvh.hpp"
#include "../math/Mat3.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ciphyxs {

// ── Forward declarations for the narrowphase entry points ────────────────────────────────────
// (needed because collideShapes is defined before the per-primitive functions)

struct Sphere;
struct Box;
struct Plane;
struct Capsule;
struct ContactManifold;

inline bool collideSphereSphere(const Sphere& a, const Vec3f& posA, const Quaternionf& rotA,
                                const Sphere& b, const Vec3f& posB, const Quaternionf& rotB,
                                ContactManifold& manifold) noexcept;

inline bool collideSpherePlane(const Sphere& sphere, const Vec3f& posA,
                               const Plane& plane, const Vec3f& posB,
                               const Quaternionf& rotB,
                               ContactManifold& manifold) noexcept;

inline bool collideSphereBox(const Sphere& sphere, const Vec3f& posA,
                             const Box& box, const Vec3f& posB,
                             const Quaternionf& rotB,
                             ContactManifold& manifold) noexcept;

inline bool collideBoxPlane(const Box& box, const Vec3f& posA, const Quaternionf& rotA,
                            const Plane& plane, const Vec3f& posB,
                            const Quaternionf& rotB,
                            ContactManifold& manifold) noexcept;

// ────────────────────────────────────────────────────────────────────────────────────────────────
inline bool collideBoxBox(const Box& boxA, const Vec3f& posA, const Quaternionf& rotA,
                          const Box& boxB, const Vec3f& posB, const Quaternionf& rotB,
                          ContactManifold& manifold) noexcept;

// ── Capsule forward declarations ─────────────────────────────────────────────────────────────

inline bool collideCapsuleSphere(const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                                 const Sphere& sphere, const Vec3f& posSphere, const Quaternionf& rotSphere,
                                 ContactManifold& manifold) noexcept;

inline bool collideCapsuleCapsule(const Capsule& capA, const Vec3f& posA, const Quaternionf& rotA,
                                  const Capsule& capB, const Vec3f& posB, const Quaternionf& rotB,
                                  ContactManifold& manifold) noexcept;

inline bool collideCapsuleBox(const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                              const Box& box, const Vec3f& posBox, const Quaternionf& rotBox,
                              ContactManifold& manifold) noexcept;

inline bool collideCapsulePlane(const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                                const Plane& plane, const Vec3f& posPlane, const Quaternionf& rotPlane,
                                ContactManifold& manifold) noexcept;

// ── ConvexMesh forward declarations ──────────────────────────────────────────────────────────

inline bool collideConvexMeshPlane(const ConvexMesh& mesh, const Vec3f& posM, const Quaternionf& rotM,
                                    const Plane& plane, const Vec3f& posP, const Quaternionf& rotP,
                                    ContactManifold& manifold) noexcept;

// ── Triangle mesh collision prototypes ──────────────────────────────────────────────────
inline bool collideSphereTriangleMesh(const TriangleMesh& mesh,
                                       const Vec3f& posM, const Quaternionf& rotM,
                                       const Sphere& sphere, const Vec3f& posS, 
                                       ContactManifold& manifold) noexcept;
inline bool collidePlaneTriangleMesh(const TriangleMesh& mesh,
                                      const Vec3f& posM, const Quaternionf& rotM,
                                      const Plane& plane, const Vec3f& posP, const Quaternionf& rotP,
                                      ContactManifold& manifold) noexcept;

inline bool collideBoxTriangleMesh(const TriangleMesh& mesh,
                                   const Vec3f& posM, const Quaternionf& rotM,
                                   const Box& box, const Vec3f& posB, const Quaternionf& rotB,
                                   ContactManifold& manifold) noexcept;

inline bool collideCapsuleTriangleMesh(const TriangleMesh& mesh,
                                        const Vec3f& posM, const Quaternionf& rotM,
                                        const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                                        ContactManifold& manifold) noexcept;

inline bool collideConvexMeshTriangleMesh(const TriangleMesh& mesh,
                                           const Vec3f& posM, const Quaternionf& rotM,
                                           const ConvexMesh& convex, const Vec3f& posC, const Quaternionf& rotC,
                                           ContactManifold& manifold) noexcept;

// ── Sphere-vs-ConvexMesh forward declaration (AABB-based, before collideShapes) ──────────
inline bool collideSphereConvexMesh(const Sphere& sphere, const Vec3f& posA,
                                    const ConvexMesh& mesh, const Vec3f& posB,
                                    const Quaternionf& rotB,
                                    ContactManifold& manifold) noexcept;

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ────────────────────────────────────────────────────────────────────────────────────────────────

namespace detail {

/// @brief  Clamp `x` to [lo, hi].
constexpr float clamp(float x, float lo, float hi) noexcept {
    return std::max(lo, std::min(x, hi));
}

/// @brief  Get the world-space corners (8) of an OBB.
inline void getBoxCorners(const Vec3f& center, const Quaternionf& rot,
                          const Vec3f& he, Vec3f* out) noexcept {
    Mat3f m = rot.toMat3();
    // Three scaled axes.
    Vec3f ax = m.col(0) * he.x;
    Vec3f ay = m.col(1) * he.y;
    Vec3f az = m.col(2) * he.z;

    out[0] = center - ax - ay - az;
    out[1] = center + ax - ay - az;
    out[2] = center - ax + ay - az;
    out[3] = center + ax + ay - az;
    out[4] = center - ax - ay + az;
    out[5] = center + ax - ay + az;
    out[6] = center - ax + ay + az;
    out[7] = center + ax + ay + az;
}

/// @brief  Project the 8-box corners onto an axis. Returns min/max dot product.
inline void projectBox(const Vec3f* corners, const Vec3f& axis,
                       float& outMin, float& outMax) noexcept {
    outMin = outMax = corners[0].dot(axis);
    for (int i = 1; i < 8; ++i) {
        float d = corners[i].dot(axis);
        outMin = std::min(outMin, d);
        outMax = std::max(outMax, d);
    }
}

/// @brief  Compute overlap of two intervals [min1, max1] and [min2, max2].
///         Returns penetration (positive) or separation (negative).
inline float intervalOverlap(float min1, float max1, float min2, float max2) noexcept {
    return std::min(max1, max2) - std::max(min1, min2);
}

// ─── Face-clipping helpers for Box-Box ─────────────────────────────────────────────────────────

/// @brief  Get the face vertices (4 corners) of a box face.
///         faceIndex: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
inline void getBoxFace(const Vec3f& center, const Quaternionf& rot,
                       const Vec3f& he, int faceIndex,
                       Vec3f* faceVerts, Vec3f& faceNormal, Vec3f& faceCenter) noexcept {
    Mat3f m       = rot.toMat3();
    Vec3f axes[3] = {m.col(0), m.col(1), m.col(2)};
    int axisIdx   = faceIndex >> 1;           // 0,1,2
    bool positive = (faceIndex & 1) == 0;     // +x,+y,+z

    float sign = positive ? 1.0f : -1.0f;
    faceNormal = axes[axisIdx] * sign;
    Vec3f heSc = axes[axisIdx] * (he[axisIdx] * sign);
    faceCenter = center + heSc;

    // The two tangent axes.
    int tA = (axisIdx + 1) % 3;
    int tB = (axisIdx + 2) % 3;
    Vec3f t0 = axes[tA];
    Vec3f t1 = axes[tB];

    Vec3f e0 = t0 * he[tA];
    Vec3f e1 = t1 * he[tB];

    faceVerts[0] = faceCenter - e0 - e1;
    faceVerts[1] = faceCenter + e0 - e1;
    faceVerts[2] = faceCenter + e0 + e1;
    faceVerts[3] = faceCenter - e0 + e1;
}

/// @brief  Clip a polygon (given as vertices) against a plane: n·v = d.
///         Keeps vertices on the side where n·v < d.
inline int clipPolygonToPlane(const Vec3f* inVerts, int inCount,
                              const Vec3f& n, float d,
                              Vec3f* outVerts) noexcept {
    if (inCount < 1) return 0;

    int outCount = 0;
    Vec3f prev = inVerts[inCount - 1];
    float prevD = prev.dot(n) - d;

    for (int i = 0; i < inCount; ++i) {
        Vec3f cur  = inVerts[i];
        float curD = cur.dot(n) - d;

        if (prevD < 0.0f) {  // prev is inside
            if (curD < 0.0f) {
                outVerts[outCount++] = cur;           // both inside
            } else {
                // prev inside, cur outside – interpolate
                float t = prevD / (prevD - curD);
                outVerts[outCount++] = prev + (cur - prev) * t;
            }
        } else if (curD < 0.0f) {
            // prev outside, cur inside
            float t = prevD / (prevD - curD);
            outVerts[outCount++] = prev + (cur - prev) * t;
            outVerts[outCount++] = cur;
        }
        prev  = cur;
        prevD = curD;
    }
    return outCount;
}

} // namespace detail

// ────────────────────────────────────────────────────────────────────────────────────────────
//  Triangle collision helpers
// ────────────────────────────────────────────────────────────────────────────────────────────

namespace tri_detail {

/// @brief  Closest point on a triangle to a given point, with barycentric coordinates.
inline void closestPointOnTriangle(const Vec3f& p,
                                    const Vec3f& a, const Vec3f& b, const Vec3f& c,
                                    Vec3f& closest, Vec3f& bary) noexcept {
    // Compute barycentric coordinates via projection onto triangle plane.
    Vec3f ab = b - a;
    Vec3f ac = c - a;
    Vec3f ap = p - a;
    
    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        bary = Vec3f(1, 0, 0);
        return;
    }
    
    Vec3f bp = p - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        bary = Vec3f(0, 1, 0);
        return;
    }
    
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        closest = a + ab * v;
        bary = Vec3f(1 - v, v, 0);
        return;
    }
    
    Vec3f cp = p - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        bary = Vec3f(0, 0, 1);
        return;
    }
    
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        closest = a + ac * w;
        bary = Vec3f(1 - w, 0, w);
        return;
    }
    
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + (c - b) * w;
        bary = Vec3f(0, 1 - w, w);
        return;
    }
    
    // Inside triangle \u2014 use plane projection.
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float u = 1.0f - v - w;
    closest = a * u + b * v + c * w;
    bary = Vec3f(u, v, w);
}

/// @brief  Sphere vs single triangle contact generation.
inline bool collideSphereTriangle(const Vec3f& sphereCenter, float sphereRadius,
                                   const Vec3f& t0, const Vec3f& t1, const Vec3f& t2,
                                   ContactManifold& manifold) noexcept {
    Vec3f closest, bary;
    closestPointOnTriangle(sphereCenter, t0, t1, t2, closest, bary);
    
    Vec3f delta = sphereCenter - closest;
    float distSq = delta.lengthSquared();
    float radSq = sphereRadius * sphereRadius;
    
    if (distSq >= radSq) return false;
    
    float dist = std::sqrt(distSq);
    float pen = sphereRadius - dist;
    
    Vec3f normal;
    if (dist > 1e-8f) {
        normal = delta / dist;
    } else {
        // Sphere center is on the triangle surface \u2014 use triangle normal.
        Vec3f edge1 = t1 - t0;
        Vec3f edge2 = t2 - t0;
        normal = edge1.cross(edge2);
        float nl = normal.length();
        if (nl > 1e-12f) normal = normal / nl;
        else normal = Vec3f(0, 1, 0);
    }
    
    ContactPoint pt;
    pt.position    = closest;
    pt.normal      = normal;
    pt.penetration = pen;
    manifold.addPoint(pt);
    return true;
}

} // namespace tri_detail

/// @brief  Sphere vs TriangleMesh collision (BVH-accelerated).
inline bool collideSphereTriangleMesh(const TriangleMesh& mesh,
                                       const Vec3f& posM, const Quaternionf& rotM,
                                       const Sphere& sphere, const Vec3f& posS,
                                       ContactManifold& manifold) noexcept {
    // Transform sphere center into mesh\'s local space.
    Vec3f localCenter = rotM.rotateInverse(posS - posM);
    float radius = sphere.radius;
    
    // Compute mesh-space AABB of the sphere.
    Vec3f r(radius);
    AABB sphereAABB = {localCenter - r, localCenter + r};
    
    bool hit = false;
    
    if (mesh.bvh && mesh.triangleCount > 0) {
        // BVH-accelerated: query overlapping triangles.
        int triBuffer = 0;
        int outCount = 0;
        constexpr int kMaxTris = 256;
        int triIds[kMaxTris];
        
        if (mesh.bvh->queryOverlap(sphereAABB, mesh.vertices, mesh.indices,
                                    triBuffer, kMaxTris, triIds, outCount)) {
            manifold.clearPoints();
            for (int i = 0; i < outCount && i < kMaxTris; ++i) {
                int t = triIds[i];
                const Vec3f& v0 = mesh.vertices[mesh.indices[t * 3 + 0]];
                const Vec3f& v1 = mesh.vertices[mesh.indices[t * 3 + 1]];
                const Vec3f& v2 = mesh.vertices[mesh.indices[t * 3 + 2]];
                if (tri_detail::collideSphereTriangle(localCenter, radius, v0, v1, v2, manifold))
                    hit = true;
            }
        }
    } else {
        // Fallback: linear scan.
        manifold.clearPoints();
        for (int i = 0; i < mesh.triangleCount; ++i) {
            const Vec3f& v0 = mesh.vertices[mesh.indices[i * 3 + 0]];
            const Vec3f& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
            const Vec3f& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];
            if (tri_detail::collideSphereTriangle(localCenter, radius, v0, v1, v2, manifold))
                hit = true;
        }
    }
    
    return hit;
}

/// @brief  Plane vs TriangleMesh collision.
inline bool collidePlaneTriangleMesh(const TriangleMesh& mesh,
                                      const Vec3f& posM, const Quaternionf& rotM,
                                      const Plane& plane, const Vec3f& posP, const Quaternionf& rotP,
                                      ContactManifold& manifold) noexcept {
    // Compute world-space plane.
    Vec3f worldNormal = rotP.rotate(plane.normal);
    float worldD = plane.distance + worldNormal.dot(posP);
    
    // Transform plane into mesh local space.
    Vec3f localNormal = rotM.rotateInverse(worldNormal);
    float localD = worldD - localNormal.dot(posM);
    
    manifold.clearPoints();
    bool hit = false;
    
    for (int i = 0; i < mesh.triangleCount; ++i) {
        const Vec3f& v0 = mesh.vertices[mesh.indices[i * 3 + 0]];
        const Vec3f& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
        const Vec3f& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];
        
        float d0 = v0.dot(localNormal) - localD;
        float d1 = v1.dot(localNormal) - localD;
        float d2 = v2.dot(localNormal) - localD;
        
        // Check if any vertex is below the plane (d < 0).
        float minD = std::min({d0, d1, d2});
        if (minD >= 0.0f) continue;
        
        float pen = -minD;
        Vec3f deepestPt = (d0 <= d1 && d0 <= d2) ? v0 : ((d1 <= d2) ? v1 : v2);
        // Transform to world space.
        Vec3f worldPt = posM + rotM.rotate(deepestPt);
        
        ContactPoint pt;
        pt.position    = worldPt;
        pt.normal      = worldNormal;  // normal points from plane toward mesh
        pt.penetration = pen;
        manifold.addPoint(pt);
        hit = true;
        if (manifold.pointCount >= 4) break;
    }
    
    return hit;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Narrow-phase entry points
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Collide two shapes given their transforms.  Returns true if contacts were generated.
///
/// This is the **main dispatch function** – call it for every broad-phase pair.
///
/// @param shapeA   Shape of body A.
/// @param posA     World position of body A.
/// @param rotA     World rotation of body A.
/// @param shapeB   Shape of body B.
/// @param posB     World position of body B.
/// @param rotB     World rotation of body B.
/// @param restA    Restitution of body A (combined with B by the caller).
/// @param restB    Restitution of body B.
/// @param fricA    Friction of body A (combined with B by the caller).
/// @param fricB    Friction of body B.
/// @param[out] manifold  Populated with contact points if shapes intersect.
/// @return True if at least one contact point was generated.
inline bool collideShapes(
    const Shape& shapeA, const Vec3f& posA, const Quaternionf& rotA,
    const Shape& shapeB, const Vec3f& posB, const Quaternionf& rotB,
    float restA, float restB, float fricA, float fricB,
    ContactManifold& manifold) noexcept {

    // Combined material properties.
    manifold.combinedRestitution = std::sqrt(restA * restB);  // geometric mean
    manifold.combinedFriction    = std::sqrt(fricA * fricB);

    using ST = ShapeType;

    // Dispatch table – every combination.
    auto a = shapeA.type;
    auto b = shapeB.type;

    if (a == ST::Sphere && b == ST::Sphere) {
        return collideSphereSphere(shapeA.sphere, posA, rotA,
                                   shapeB.sphere, posB, rotB, manifold);
    }
    if (a == ST::Sphere && b == ST::Plane) {
        return collideSpherePlane(shapeA.sphere, posA,
                                  shapeB.plane, posB, rotB, manifold);
    }
    if (a == ST::Plane && b == ST::Sphere) {
        bool hit = collideSpherePlane(shapeB.sphere, posB,
                                      shapeA.plane, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    if (a == ST::Sphere && b == ST::Box) {
        return collideSphereBox(shapeA.sphere, posA,
                                shapeB.box, posB, rotB, manifold);
    }
    if (a == ST::Box && b == ST::Sphere) {
        bool hit = collideSphereBox(shapeB.sphere, posB,
                                    shapeA.box, posA, rotA, manifold);
        // Flip normals for swapped order.
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    if (a == ST::Box && b == ST::Plane) {
        return collideBoxPlane(shapeA.box, posA, rotA,
                               shapeB.plane, posB, rotB, manifold);
    }
    if (a == ST::Plane && b == ST::Box) {
        bool hit = collideBoxPlane(shapeB.box, posB, rotB,
                                   shapeA.plane, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    if (a == ST::Box && b == ST::Box) {
        return collideBoxBox(shapeA.box, posA, rotA,
                             shapeB.box, posB, rotB, manifold);
    }

    // ── Capsule pairs ──────────────────────────────────────────────────────────────────────────
    if (a == ST::Capsule && b == ST::Sphere) {
        return collideCapsuleSphere(shapeA.capsule, posA, rotA,
                                    shapeB.sphere, posB, rotB, manifold);
    }
    if (a == ST::Sphere && b == ST::Capsule) {
        bool hit = collideCapsuleSphere(shapeB.capsule, posB, rotB,
                                        shapeA.sphere, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    if (a == ST::Capsule && b == ST::Capsule) {
        return collideCapsuleCapsule(shapeA.capsule, posA, rotA,
                                     shapeB.capsule, posB, rotB, manifold);
    }
    if (a == ST::Capsule && b == ST::Box) {
        return collideCapsuleBox(shapeA.capsule, posA, rotA,
                                 shapeB.box, posB, rotB, manifold);
    }
    if (a == ST::Box && b == ST::Capsule) {
        bool hit = collideCapsuleBox(shapeB.capsule, posB, rotB,
                                     shapeA.box, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    if (a == ST::Capsule && b == ST::Plane) {
        return collideCapsulePlane(shapeA.capsule, posA, rotA,
                                   shapeB.plane, posB, rotB, manifold);
    }
    if (a == ST::Plane && b == ST::Capsule) {
        bool hit = collideCapsulePlane(shapeB.capsule, posB, rotB,
                                       shapeA.plane, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // ── TriangleMesh pairs ──────────────────────────────────────────────────────────────
    // TriangleMesh must be checked BEFORE ConvexMesh GJK dispatch, since ConvexMesh+GJK
    // would otherwise intercept ConvexMesh-vs-TriangleMesh pairs.

    if (a == ST::TriangleMesh && b == ST::Sphere) {
        return collideSphereTriangleMesh(shapeA.triangleMesh, posA, rotA,
                                          shapeB.sphere, posB,
                                          manifold);
    }
    if (a == ST::Sphere && b == ST::TriangleMesh) {
        bool hit = collideSphereTriangleMesh(shapeB.triangleMesh, posB, rotB,
                                              shapeA.sphere, posA,
                                              manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    if (a == ST::TriangleMesh && b == ST::Plane) {
        return collidePlaneTriangleMesh(shapeA.triangleMesh, posA, rotA,
                                         shapeB.plane, posB, rotB,
                                         manifold);
    }
    if (a == ST::Plane && b == ST::TriangleMesh) {
        bool hit = collidePlaneTriangleMesh(shapeB.triangleMesh, posB, rotB,
                                             shapeA.plane, posA, rotA,
                                             manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }
    // ── TriangleMesh ↔ Box
    if (a == ST::TriangleMesh && b == ST::Box) {
        return collideBoxTriangleMesh(shapeA.triangleMesh, posA, rotA,
                                       shapeB.box, posB, rotB, manifold);
    }
    if (a == ST::Box && b == ST::TriangleMesh) {
        bool hit = collideBoxTriangleMesh(shapeB.triangleMesh, posB, rotB,
                                           shapeA.box, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // ── TriangleMesh ↔ Capsule
    if (a == ST::TriangleMesh && b == ST::Capsule) {
        return collideCapsuleTriangleMesh(shapeA.triangleMesh, posA, rotA,
                                           shapeB.capsule, posB, rotB, manifold);
    }
    if (a == ST::Capsule && b == ST::TriangleMesh) {
        bool hit = collideCapsuleTriangleMesh(shapeB.triangleMesh, posB, rotB,
                                               shapeA.capsule, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // ── TriangleMesh ↔ ConvexMesh (GJK per triangle)
    if (a == ST::TriangleMesh && b == ST::ConvexMesh) {
        return collideConvexMeshTriangleMesh(shapeA.triangleMesh, posA, rotA,
                                              shapeB.convexMesh, posB, rotB, manifold);
    }
    if (a == ST::ConvexMesh && b == ST::TriangleMesh) {
        bool hit = collideConvexMeshTriangleMesh(shapeB.triangleMesh, posB, rotB,
                                                  shapeA.convexMesh, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // TriangleMesh ↔ TriangleMesh not yet implemented.
    if (a == ST::TriangleMesh || b == ST::TriangleMesh) {
        return false;
    }

    // ── Sphere ↔ ConvexMesh (AABB-based, same robust approach as collideSphereBox) ───────────
    if (a == ST::Sphere && b == ST::ConvexMesh) {
        return collideSphereConvexMesh(shapeA.sphere, posA,
                                       shapeB.convexMesh, posB, rotB,
                                       manifold);
    }
    if (a == ST::ConvexMesh && b == ST::Sphere) {
        bool hit = collideSphereConvexMesh(shapeB.sphere, posB,
                                           shapeA.convexMesh, posA, rotA,
                                           manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // ── ConvexMesh pairs (GJK/EPA) ─────────────────────────────────────────────────────────
    if ((a == ST::ConvexMesh || b == ST::ConvexMesh) && a != ST::Plane && b != ST::Plane) {
        // Any convex-convex pair except Plane (handled separately below).
        return gjk_detail::collideConvexGJK(shapeA, posA, rotA,
                                             shapeB, posB, rotB,
                                             manifold);
    }

    // ── ConvexMesh ↔ Plane (SAT-style vertex sweep) ───────────────────────────────────────
    if (a == ST::ConvexMesh && b == ST::Plane) {
        return collideConvexMeshPlane(shapeA.convexMesh, posA, rotA,
                                       shapeB.plane, posB, rotB, manifold);
    }
    if (a == ST::Plane && b == ST::ConvexMesh) {
        bool hit = collideConvexMeshPlane(shapeB.convexMesh, posB, rotB,
                                           shapeA.plane, posA, rotA, manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // Unsupported pair – no collision.
    return false;
}
// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Sphere ↔ ConvexMesh (AABB-based, same as collideSphereBox)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Sphere vs ConvexMesh collision using AABB-based point-to-mesh distance.
///         Uses the mesh's halfExtents and center for a fast conservative test.
inline bool collideSphereConvexMesh(const Sphere& sphere, const Vec3f& posA,
                                    const ConvexMesh& mesh, const Vec3f& posB,
                                    const Quaternionf& rotB,
                                    ContactManifold& manifold) noexcept {
    // Transform sphere centre into mesh local space.
    Vec3f localPos = rotB.rotateInverse(posA - posB);
    Vec3f he       = mesh.halfExtents;
    Vec3f center   = mesh.center;

    // Closest point on AABB (in local space).
    Vec3f closest = Vec3f(
        detail::clamp(localPos.x, center.x - he.x, center.x + he.x),
        detail::clamp(localPos.y, center.y - he.y, center.y + he.y),
        detail::clamp(localPos.z, center.z - he.z, center.z + he.z)
    );

    Vec3f diff    = localPos - closest;
    float distSq  = diff.lengthSquared();
    if (distSq >= sphere.radius * sphere.radius) return false;

    float dist = std::sqrt(distSq);

    Vec3f normalLocal;
    if (dist > 1e-8f) {
        normalLocal = diff / dist;
    } else {
        // Sphere centre is inside the AABB — push out along axis of least penetration.
        float minPen = (center.x + he.x) - std::abs(localPos.x - center.x);
        int axis     = 0;
        float penY   = (center.y + he.y) - std::abs(localPos.y - center.y);
        if (penY < minPen) { minPen = penY; axis = 1; }
        float penZ   = (center.z + he.z) - std::abs(localPos.z - center.z);
        if (penZ < minPen) { minPen = penZ; axis = 2; }
        normalLocal = Vec3f::zero();
        normalLocal[axis] = (localPos[axis] >= center[axis]) ? 1.0f : -1.0f;
        dist = 0.0f;
    }

    Vec3f normalWorld = rotB.rotate(normalLocal).normalized();
    float pen = sphere.radius - dist;

    ContactPoint pt;
    pt.position    = rotB.rotate(closest) + posB;  // world-space closest point
    pt.normal      = normalWorld;
    pt.penetration = std::max(pen, 0.001f);
    pt.featureA    = 0;
    pt.featureB    = -1;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Sphere ↔ Sphere
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Sphere-sphere collision.
inline bool collideSphereSphere(const Sphere& a, const Vec3f& posA, const Quaternionf& /*rotA*/,
                                const Sphere& b, const Vec3f& posB, const Quaternionf& /*rotB*/,
                                ContactManifold& manifold) noexcept {
    Vec3f delta  = posB - posA;
    float distSq = delta.lengthSquared();
    float radSum = a.radius + b.radius;
    if (distSq >= radSum * radSum) return false;

    float dist   = std::sqrt(distSq);
    float pen    = radSum - dist;

    Vec3f normal;
    if (dist > 1e-8f) {
        normal = delta / dist;
    } else {
        normal = Vec3f::unitY();
    }

    ContactPoint pt;
    pt.position    = posA + normal * (a.radius - pen * 0.5f);
    pt.normal      = normal;
    pt.penetration = pen;
    pt.featureA    = 0;
    pt.featureB    = 0;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Sphere ↔ Plane
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Sphere-plane collision.  Plane `normal · p = distance`.
inline bool collideSpherePlane(const Sphere& sphere, const Vec3f& posA,
                               const Plane& plane, const Vec3f& posB,
                               const Quaternionf& rotB,
                               ContactManifold& manifold) noexcept {
    // Transform plane normal by the plane body's rotation.
    Vec3f nWorld = rotB.rotate(plane.normal).normalized();
    // Signed distance from sphere center to plane.
    // Plane equation in world space: nWorld · (x - posB) = plane.distance
    float signedDist = (posA - posB).dot(nWorld) - plane.distance;

    float pen = sphere.radius - signedDist;
    if (pen <= 0.0f) return false;

    ContactPoint pt;
    // Contact point is the projection of the sphere centre onto the plane surface.
    pt.position    = posA - nWorld * signedDist;
    pt.normal      = -nWorld;   // points from plane toward sphere
    pt.penetration = pen;
    pt.featureA    = 0;
    pt.featureB    = 0;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Sphere ↔ Box
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Sphere-box collision using closest-point-on-box.
inline bool collideSphereBox(const Sphere& sphere, const Vec3f& posA,
                             const Box& box, const Vec3f& posB,
                             const Quaternionf& rotB,
                             ContactManifold& manifold) noexcept {
    // Transform sphere centre into box local space.
    Vec3f localPos = rotB.rotateInverse(posA - posB);
    Vec3f he       = box.halfExtents;

    // Closest point on box (in local space).
    Vec3f closest = Vec3f(
        detail::clamp(localPos.x, -he.x, he.x),
        detail::clamp(localPos.y, -he.y, he.y),
        detail::clamp(localPos.z, -he.z, he.z)
    );

    Vec3f diff    = localPos - closest;
    float distSq  = diff.lengthSquared();
    if (distSq >= sphere.radius * sphere.radius) return false;

    float dist = std::sqrt(distSq);

    Vec3f normalLocal;
    if (dist > 1e-8f) {
        normalLocal = diff / dist;
    } else {
        // Sphere centre is inside the box – push out along the axis of least penetration.
        float minPen = he.x - std::abs(localPos.x);
        int axis     = 0;
        float penY   = he.y - std::abs(localPos.y);
        if (penY < minPen) { minPen = penY; axis = 1; }
        float penZ   = he.z - std::abs(localPos.z);
        if (penZ < minPen) { minPen = penZ; axis = 2; }
        normalLocal = Vec3f::zero();
        normalLocal[axis] = (localPos[axis] >= 0.0f) ? 1.0f : -1.0f;
        dist = 0.0f;
    }

    Vec3f normalWorld = rotB.rotate(normalLocal).normalized();
    float pen = sphere.radius - dist;

    ContactPoint pt;
    pt.position    = rotB.rotate(closest) + posB;  // world-space closest point
    pt.normal      = normalWorld;
    pt.penetration = pen;
    pt.featureA    = 0;
    // Encode which box face (0-5) the closest point is on, if any.
    pt.featureB    = -1;
    for (int a = 0; a < 3; ++a) {
        if (std::abs(closest[a]) >= he[a] - 1e-6f) {
            pt.featureB = closest[a] >= 0 ? a * 2 : a * 2 + 1;
            break;
        }
    }
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Box ↔ Plane
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Box-plane collision test all 8 vertices against the plane.
inline bool collideBoxPlane(const Box& box, const Vec3f& posA, const Quaternionf& rotA,
                            const Plane& plane, const Vec3f& posB,
                            const Quaternionf& rotB,
                            ContactManifold& manifold) noexcept {
    // World-space plane normal.
    Vec3f nWorld = rotB.rotate(plane.normal).normalized();

    Vec3f corners[8];
    detail::getBoxCorners(posA, rotA, box.halfExtents, corners);

    bool hit = false;
    for (int i = 0; i < 8; ++i) {
        float signedDist = (corners[i] - posB).dot(nWorld) - plane.distance;
        if (signedDist < 0.0f) {
            ContactPoint pt;
            pt.position    = corners[i] - nWorld * signedDist;
            pt.normal      = -nWorld;
            pt.penetration = -signedDist;
            pt.featureA    = i;
            manifold.addPoint(pt);
            hit = true;
        }
    }
    return hit;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Box ↔ Box  (Separating Axis Theorem + face clipping)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Box-box collision using SAT.
///
/// Tests 15 axes (6 face normals + 9 edge cross products).  If overlap is found on every axis,
/// the axis with minimum penetration is used as the reference face for contact-point generation
/// via clipping.
inline bool collideBoxBox(const Box& boxA, const Vec3f& posA, const Quaternionf& rotA,
                          const Box& boxB, const Vec3f& posB, const Quaternionf& rotB,
                          ContactManifold& manifold) noexcept {
    // Corners of both boxes in world space.
    Vec3f cornersA[8], cornersB[8];
    detail::getBoxCorners(posA, rotA, boxA.halfExtents, cornersA);
    detail::getBoxCorners(posB, rotB, boxB.halfExtents, cornersB);

    // Axes (world-space face normals).
    Mat3f mA = rotA.toMat3();
    Mat3f mB = rotB.toMat3();
    Vec3f axesA[3] = {mA.col(0), mA.col(1), mA.col(2)};
    Vec3f axesB[3] = {mB.col(0), mB.col(1), mB.col(2)};

    // Candidate axes for SAT.
    struct AxisTest {
        Vec3f axis;
        float penetration; // positive = overlap
    };

    AxisTest best;
    best.penetration = std::numeric_limits<float>::max();
    int bestIsFaceOfB = 0;  // +1 = B face, 0 = edge, -1 = A face

    auto testAxis = [&](const Vec3f& ax, bool isEdge, int whichBox) -> bool {
        float lenSq = ax.lengthSquared();
        if (lenSq < 1e-10f) return true;      // degenerate, skip
        Vec3f n = ax / std::sqrt(lenSq);

        float minA, maxA, minB, maxB;
        detail::projectBox(cornersA, n, minA, maxA);
        detail::projectBox(cornersB, n, minB, maxB);

        float overlap = detail::intervalOverlap(minA, maxA, minB, maxB);
        if (overlap <= 0.0f) return false;     // separating axis – early out

        if (overlap < best.penetration) {
            best.penetration = overlap;
            best.axis        = n;
            bestIsFaceOfB    = isEdge ? 0 : (whichBox == 1 ? 1 : -1);
        }
        return true; // still overlapping on this axis
    };

    // Test 6 face normals (3 from A, 3 from B).
    for (int i = 0; i < 3; ++i) {
        if (!testAxis(axesA[i], false, -1)) return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!testAxis(axesB[i], false,  1)) return false;
    }

    // Test 9 edge cross products.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Vec3f edge = axesA[i].cross(axesB[j]);
            if (!testAxis(edge, true, 0)) return false;
        }
    }

    // ─── Contact-point generation via face clipping ───────────────────────────────────────────
    //
    // The reference face is on the box that owns the minimum-penetration axis.
    // If bestIsFaceOfB is 0 (edge-edge), we use a dedicated edge-edge closest-point
    // contact generator instead of face clipping (which produces poor results for
    // edge-edge configurations).

    if (bestIsFaceOfB == 0) {
        // ── Edge-edge contact ──────────────────────────────────────────────────────────────
        // Find the two edges (one from each box) that are most aligned with the contact
        // normal's perpendicular plane, compute closest points between them, and generate
        // a single contact point at the midpoint.
        //
        // The contact normal (best.axis) is from the edge cross product SAT test.
        // Each box's edge direction is one of its local axes.
        // We find which axis of each box is most perpendicular to best.axis (i.e.
        // the axis with the smallest |dot| — that axis lies in the contact plane).

        Mat3f mA = rotA.toMat3();
        Mat3f mB = rotB.toMat3();
        Vec3f worldAxesA[3] = {mA.col(0), mA.col(1), mA.col(2)};
        Vec3f worldAxesB[3] = {mB.col(0), mB.col(1), mB.col(2)};

        // Find the edge axis for box A (most perpendicular to best.axis).
        int edgeA = 0;
        float minADot = std::abs(worldAxesA[0].dot(best.axis));
        for (int i = 1; i < 3; ++i) {
            float d = std::abs(worldAxesA[i].dot(best.axis));
            if (d < minADot) { minADot = d; edgeA = i; }
        }
        // Find the edge axis for box B.
        int edgeB = 0;
        float minBDot = std::abs(worldAxesB[0].dot(best.axis));
        for (int i = 1; i < 3; ++i) {
            float d = std::abs(worldAxesB[i].dot(best.axis));
            if (d < minBDot) { minBDot = d; edgeB = i; }
        }

        // Edge centres (midpoints of the two relevant edges) and directions.
        // The edge direction is the selected local axis; the centre is the body position
        // offset along the two *other* axes.
        auto edgeCenterAndDir = [&](const Vec3f& pos, const Quaternionf& rot,
                                    const Vec3f& he, int edgeAxis,
                                    Vec3f& center, Vec3f& dir) {
            Mat3f m = rot.toMat3();
            Vec3f axes[3] = {m.col(0), m.col(1), m.col(2)};
            dir = axes[edgeAxis];
            // Centre = body pos + sum of offsets along non-edge axes.
            center = pos;
            int tA = (edgeAxis + 1) % 3;
            int tB = (edgeAxis + 2) % 3;
            // The edge is at the extremal positions along the non-edge axes.
            // For the centre of the edge, we use 0 offset on those (passes through body centre).
            // But we need the actual edge line: pass through the box face centre on the
            // positive side of both non-edge axes, or the negative side.
            // We compute the two endpoints:
            Vec3f offset = axes[tA] * he[tA] + axes[tB] * he[tB];
            center = pos + offset;
        };

        Vec3f centerA, dirA, centerB, dirB;
        edgeCenterAndDir(posA, rotA, boxA.halfExtents, edgeA, centerA, dirA);
        edgeCenterAndDir(posB, rotB, boxB.halfExtents, edgeB, centerB, dirB);

        // Also try the opposite sign offset for the other parallel edge.
        Vec3f centerA2 = posA - (centerA - posA);
        Vec3f centerB2 = posB - (centerB - posB);

        // We have 4 possible edge pairs (A edge 1 or 2) × (B edge 1 or 2).
        // Pick the pair with the closest distance.
        auto closestPtEdgeEdge = [](const Vec3f& p1, const Vec3f& d1,
                                    const Vec3f& p2, const Vec3f& d2,
                                    Vec3f& pt1, Vec3f& pt2) {
            // Project p2-p1 onto the plane of the two edge directions.
            Vec3f r = p1 - p2;
            float a = d1.dot(d1);
            float b = d1.dot(d2);
            float c = d2.dot(d2);
            float d = d1.dot(r);
            float e = d2.dot(r);
            float denom = a * c - b * b;
            float t1, t2;
            if (std::abs(denom) < 1e-12f) {
                t1 = 0.0f;
                t2 = e / c;
            } else {
                t1 = (b * e - c * d) / denom;
                t2 = (a * e - b * d) / denom;
            }
            pt1 = p1 + d1 * t1;
            pt2 = p2 + d2 * t2;
        };

        Vec3f bestPA, bestPB;
        float bestDistSq = std::numeric_limits<float>::max();

        Vec3f candidatesA[2] = {centerA, centerA2};
        Vec3f candidatesB[2] = {centerB, centerB2};

        for (int ca = 0; ca < 2; ++ca) {
            for (int cb = 0; cb < 2; ++cb) {
                Vec3f pa, pb;
                closestPtEdgeEdge(candidatesA[ca], dirA,
                                  candidatesB[cb], dirB, pa, pb);
                float dSq = (pa - pb).lengthSquared();
                if (dSq < bestDistSq) {
                    bestDistSq = dSq;
                    bestPA = pa;
                    bestPB = pb;
                }
            }
        }

        Vec3f midPt = (bestPA + bestPB) * 0.5f;
        float depth = std::sqrt(bestDistSq);
        // The penetration along the SAT normal.
        float pen = best.penetration;

        manifold.clearPoints();
        ContactPoint pt;
        pt.position    = midPt;
        pt.normal      = best.axis;  // pointing from A to B
        pt.penetration = std::max(pen, depth * 0.5f);
        // Encode which edges (box A: edgeAxis; box B: edgeAxis).
        pt.featureA    = 10 + edgeA;  // 10+ signals "edge" rather than "face"
        pt.featureB    = 10 + edgeB;
        manifold.addPoint(pt);

        // ── Also try the opposite face direction for a second contact point ────────────────
        // Edge-edge contacts can be unstable with a single point; add a second at the
        // opposite end of the contact region if the boxes are deeply overlapping.
        if (pen > 0.02f) {
            Vec3f contactDir = bestPA - bestPB;
            float cdLen = contactDir.length();
            if (cdLen > 1e-6f) {
                contactDir /= cdLen;
                Vec3f pt2Pos = midPt + contactDir * std::min(pen, 0.1f);
                // Check that pt2Pos is inside both boxes (rough check via SAT depth).
                ContactPoint pt2;
                pt2.position    = pt2Pos;
                pt2.normal      = best.axis;
                pt2.penetration = pen * 0.5f;
                pt2.featureA    = 10 + edgeA;
                pt2.featureB    = 10 + edgeB;
                manifold.addPoint(pt2);
            }
        }

        return true;
    }

    bool useBFace = (bestIsFaceOfB == 1);
    const Box& refBox    = useBFace ? boxB : boxA;
    const Vec3f& refPos  = useBFace ? posB  : posA;
    const Quaternionf& refRot = useBFace ? rotB : rotA;
    const Box& incBox    = useBFace ? boxA : boxB;
    const Vec3f& incPos  = useBFace ? posA  : posB;
    const Quaternionf& incRot = useBFace ? rotA : rotB;
    float signFlip       = useBFace ? -1.0f : 1.0f;

    // Find which face of the reference box is most anti-parallel to best.axis.
    Mat3f refM = refRot.toMat3();
    Vec3f refAxes[3] = {refM.col(0), refM.col(1), refM.col(2)};
    int refFace = 0;
    float maxDot = refAxes[0].dot(best.axis * signFlip);
    for (int i = 1; i < 3; ++i) {
        float d = refAxes[i].dot(best.axis * signFlip);
        if (d > maxDot) { maxDot = d; refFace = i; }
    }
    // Positive or negative face?
    int refFaceIndex = (best.axis.dot(refAxes[refFace] * signFlip) > 0.0f)
                       ? (refFace * 2)         // +face
                       : (refFace * 2 + 1);    // -face

    // Get reference face vertices and normal.
    Vec3f refVerts[4], refNormal, refCenter;
    detail::getBoxFace(refPos, refRot, refBox.halfExtents, refFaceIndex,
                       refVerts, refNormal, refCenter);

    // Get the incident box face whose outward normal is most anti-parallel
    // to the reference face normal.  We find the axis with the largest absolute
    // dot product, then select the face that opposes refNormal.
    Mat3f incM  = incRot.toMat3();
    Vec3f incAxes[3] = {incM.col(0), incM.col(1), incM.col(2)};
    int incFace = 0;
    float maxAbsDot = std::abs(incAxes[0].dot(refNormal));
    for (int i = 1; i < 3; ++i) {
        float d = std::abs(incAxes[i].dot(refNormal));
        if (d > maxAbsDot) { maxAbsDot = d; incFace = i; }
    }
    int incFaceIndex = (incAxes[incFace].dot(refNormal) > 0.0f)
                       ? (incFace * 2 + 1)    // +axis aligns → use -face
                       : (incFace * 2);        // +axis opposes → use +face

    Vec3f incVerts[4], incNormal, incCenter;
    detail::getBoxFace(incPos, incRot, incBox.halfExtents, incFaceIndex,
                       incVerts, incNormal, incCenter);

    // Clip the incident face against the side planes of the reference face.
    // We need the 4 side planes: for each edge of the reference face, compute the plane
    // that passes through the edge and has a normal perpendicular to refNormal.

    // Start with the incident face polygon.
    int clippedCount = 4;
    Vec3f clipped[8];
    std::copy(incVerts, incVerts + 4, clipped);

    // Build the 4 side planes: for edges (i → i+1).
    for (int i = 0; i < 4; ++i) {
        int next      = (i + 1) % 4;
        Vec3f edge    = refVerts[next] - refVerts[i];
        Vec3f planeN  = edge.cross(refNormal).normalized();
        float planeD  = planeN.dot(refVerts[i]);
        int newCount  = detail::clipPolygonToPlane(clipped, clippedCount,
                                                    planeN, planeD, clipped);
        clippedCount = newCount;
        if (clippedCount < 1) break;
    }

    // Clip against the reference face plane itself (keep vertices behind it).
    if (clippedCount > 0) {
        float refD = refNormal.dot(refVerts[0]);
        clippedCount = detail::clipPolygonToPlane(clipped, clippedCount,
                                                   -refNormal, -refD + 0.01f, clipped);
    }

    // ── Fallback: if clipping culled everything, reproject onto reference plane ──
    //
    // This happens when boxes overlap primarily along non-orthogonal axes and the
    // incident polygon lies entirely on the "outside" of the reference plane clip.
    // We project the incident face vertices onto the reference face plane and
    // re-clip against the 4 side planes.  If that also fails, we fall back to
    // the reference face center as a single contact point.
    //
    // Updated: also tries the **incident** face as reference if the first attempt
    // fails, before falling back to the reference center.  This handles cases where
    // the initial reference/incident face assignment was poor (common in near-edge
    // configurations resolved as face contacts).
    if (clippedCount < 1) {
        // Attempt 1: project incident onto reference plane and re-clip.
        {
            float refD = refNormal.dot(refVerts[0]);
            int tmpCount = 4;
            Vec3f tmp[8];
            std::copy(incVerts, incVerts + 4, tmp);
            for (int i = 0; i < tmpCount; ++i) {
                float d = tmp[i].dot(refNormal) - refD;
                tmp[i] = tmp[i] - refNormal * d;
            }
            for (int i = 0; i < 4; ++i) {
                int next      = (i + 1) % 4;
                Vec3f edge    = refVerts[next] - refVerts[i];
                Vec3f planeN  = edge.cross(refNormal).normalized();
                float planeD  = planeN.dot(refVerts[i]);
                int newCount  = detail::clipPolygonToPlane(tmp, tmpCount,
                                                            planeN, planeD, tmp);
                tmpCount = newCount;
                if (tmpCount < 1) break;
            }
            if (tmpCount >= 1) {
                clippedCount = tmpCount;
                std::copy(tmp, tmp + tmpCount, clipped);
            }
        }

        // Attempt 2: swap reference and incident and retry clipping.
        if (clippedCount < 1) {
            // Use the incident face as the new reference.
            auto tmpRefVerts = incVerts;
            auto tmpRefNormal = incNormal;

            int tmpCount = 4;
            Vec3f tmp[8];
            std::copy(refVerts, refVerts + 4, tmp);
            float refD = tmpRefNormal.dot(tmpRefVerts[0]);
            for (int i = 0; i < tmpCount; ++i) {
                float d = tmp[i].dot(tmpRefNormal) - refD;
                tmp[i] = tmp[i] - tmpRefNormal * d;
            }
            // Compute side planes from the (now incident) reference face vertices.
            for (int i = 0; i < 4; ++i) {
                int next      = (i + 1) % 4;
                Vec3f edge    = tmpRefVerts[next] - tmpRefVerts[i];
                Vec3f planeN  = edge.cross(tmpRefNormal).normalized();
                float planeD  = planeN.dot(tmpRefVerts[i]);
                int newCount  = detail::clipPolygonToPlane(tmp, tmpCount,
                                                            planeN, planeD, tmp);
                tmpCount = newCount;
                if (tmpCount < 1) break;
            }
            if (tmpCount >= 1) {
                clippedCount = tmpCount;
                std::copy(tmp, tmp + tmpCount, clipped);
            }
        }

        // Last resort: use the reference face center.
        if (clippedCount < 1) {
            clippedCount = 1;
            clipped[0] = refCenter;
            // Also clamp penetration to a reasonable value.
            best.penetration = std::min(best.penetration, 0.05f);
        }
    }

    // Remaining vertices are contact points.
    manifold.clearPoints();
    for (int i = 0; i < std::min(clippedCount, 4); ++i) {
        ContactPoint pt;
        pt.position    = clipped[i];
        pt.normal      = best.axis * signFlip;  // consistent: pointing from A to B
        pt.penetration = best.penetration;

        // Encode the reference face and the closest incident-face vertex.
        // This provides a stable feature pair for warm-start matching across frames.
        pt.featureA = refFaceIndex;
        int closestV = 0;
        float minDSq = (clipped[i] - incVerts[0]).lengthSquared();
        for (int v = 1; v < 4; ++v) {
            float dSq = (clipped[i] - incVerts[v]).lengthSquared();
            if (dSq < minDSq) { minDSq = dSq; closestV = v; }
        }
        pt.featureB = incFaceIndex * 4 + closestV;

        manifold.addPoint(pt);
    }

    return manifold.pointCount > 0;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Capsule ↔ Sphere
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Capsule-sphere collision using closest-point on capsule segment.
inline bool collideCapsuleSphere(const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                                 const Sphere& sphere, const Vec3f& posSphere, const Quaternionf& /*rotSphere*/,
                                 ContactManifold& manifold) noexcept {
    // Capsule segment endpoints in world space.
    Vec3f localA(0.0f, -capsule.halfHeight, 0.0f);
    Vec3f localB(0.0f, +capsule.halfHeight, 0.0f);
    Vec3f worldA = posCap + rotCap.rotate(localA);
    Vec3f worldB = posCap + rotCap.rotate(localB);

    // Closest point on segment to sphere center.
    Vec3f seg    = worldB - worldA;
    float lenSq  = seg.lengthSquared();
    float t      = 0.0f;
    if (lenSq > 1e-12f) {
        t = (posSphere - worldA).dot(seg) / lenSq;
        t = detail::clamp(t, 0.0f, 1.0f);
    }
    Vec3f closest = worldA + seg * t;

    Vec3f delta   = posSphere - closest;
    float distSq  = delta.lengthSquared();
    float radSum  = capsule.radius + sphere.radius;
    if (distSq >= radSum * radSum) return false;

    float dist   = std::sqrt(distSq);
    float pen    = radSum - dist;

    Vec3f normal;
    if (dist > 1e-8f) {
        normal = delta / dist;
    } else {
        // Degenerate: push along vector from capsule center to sphere center.
        Vec3f fallback = posSphere - (worldA + worldB) * 0.5f;
        normal = fallback.lengthSquared() > 1e-12f ? fallback.normalized() : Vec3f::unitY();
    }

    ContactPoint pt;
    pt.position    = closest + normal * capsule.radius;
    pt.normal      = normal;
    pt.penetration = pen;
    // Encode which part of the capsule (0=body, 1=top cap, 2=bottom cap).
    pt.featureA    = (t < 0.01f) ? 2 : ((t > 0.99f) ? 1 : 0);
    pt.featureB    = 0;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Capsule ↔ Capsule
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Capsule-capsule collision via closest-point between medial segments.
inline bool collideCapsuleCapsule(const Capsule& capA, const Vec3f& posA, const Quaternionf& rotA,
                                  const Capsule& capB, const Vec3f& posB, const Quaternionf& rotB,
                                  ContactManifold& manifold) noexcept {
    // Segment endpoints in world space.
    Vec3f localA0(0.0f, -capA.halfHeight, 0.0f);
    Vec3f localA1(0.0f, +capA.halfHeight, 0.0f);
    Vec3f localB0(0.0f, -capB.halfHeight, 0.0f);
    Vec3f localB1(0.0f, +capB.halfHeight, 0.0f);

    Vec3f A0 = posA + rotA.rotate(localA0);
    Vec3f A1 = posA + rotA.rotate(localA1);
    Vec3f B0 = posB + rotB.rotate(localB0);
    Vec3f B1 = posB + rotB.rotate(localB1);

    // Direction vectors.
    Vec3f dA = A1 - A0;
    Vec3f dB = B1 - B0;
    float lenSqA = dA.lengthSquared();
    float lenSqB = dB.lengthSquared();

    // For each degenerate segment, fall back to sphere-sphere at its midpoint.
    if (lenSqA < 1e-12f && lenSqB < 1e-12f) {
        Sphere sA{capA.radius};
        Sphere sB{capB.radius};
        Vec3f midA = (A0 + A1) * 0.5f;
        Vec3f midB = (B0 + B1) * 0.5f;
        return collideSphereSphere(sA, midA, Quaternionf::identity(),
                                   sB, midB, Quaternionf::identity(), manifold);
    }
    if (lenSqA < 1e-12f) {
        Sphere sA{capA.radius};
        Vec3f midA = (A0 + A1) * 0.5f;
        return collideCapsuleSphere(capB, posB, rotB, sA, midA, Quaternionf::identity(), manifold);
    }
    if (lenSqB < 1e-12f) {
        Sphere sB{capB.radius};
        Vec3f midB = (B0 + B1) * 0.5f;
        bool hit = collideCapsuleSphere(capA, posA, rotA, sB, midB, Quaternionf::identity(), manifold);
        for (int i = 0; i < manifold.pointCount; ++i)
            manifold.points[i].normal = -manifold.points[i].normal;
        return hit;
    }

    // Find closest points between two line segments.
    // Ref: Oren, "Distance Between Two Line Segments in 3D"
    Vec3f r    = A0 - B0;
    float a    = dA.dot(dA);
    float b    = dA.dot(dB);
    float c    = dB.dot(dB);
    float d    = dA.dot(r);
    float e    = dB.dot(r);
    float denom = a * c - b * b;

    float tA, tB;
    if (std::abs(denom) < 1e-12f) {
        // Parallel segments.
        tA = 0.0f;
        tB = detail::clamp(e / c, 0.0f, 1.0f);
    } else {
        tA = detail::clamp((b * e - c * d) / denom, 0.0f, 1.0f);
        tB = detail::clamp((a * e - b * d) / denom, 0.0f, 1.0f);
        // Snap to valid region and re-project.
        float tAold = tA;
        tA = detail::clamp((b * tB + d) / a, 0.0f, 1.0f);
        tB = tB + (tA - tAold) * b / c;
        tB = detail::clamp(tB, 0.0f, 1.0f);
    }

    Vec3f closestA = A0 + dA * tA;
    Vec3f closestB = B0 + dB * tB;

    Vec3f delta  = closestB - closestA;
    float distSq = delta.lengthSquared();
    float radSum = capA.radius + capB.radius;
    if (distSq >= radSum * radSum) return false;

    float dist = std::sqrt(distSq);
    float pen  = radSum - dist;

    Vec3f normal;
    if (dist > 1e-8f) {
        normal = delta / dist;
    } else {
        Vec3f fallback = closestB - closestA;
        normal = fallback.lengthSquared() > 1e-12f ? fallback.normalized() : Vec3f::unitY();
    }

    ContactPoint pt;
    pt.position    = closestA + normal * capA.radius;
    pt.normal      = normal;
    pt.penetration = pen;
    // Encode which part of each capsule (0=body, 1=top cap, 2=bottom cap).
    int featA = (tA < 0.01f) ? 2 : ((tA > 0.99f) ? 1 : 0);
    int featB = (tB < 0.01f) ? 2 : ((tB > 0.99f) ? 1 : 0);
    pt.featureA = featA;
    pt.featureB = featB;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Capsule ↔ Box
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Capsule-box collision via segment sampling + sphere-box tests.
///
/// Samples points along the capsule medial segment and tests each as a sphere-box
/// collision.  The deepest penetration is retained.  This correctly handles most
/// capsule-box orientations including edge and corner contacts.
inline bool collideCapsuleBox(const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                              const Box& box, const Vec3f& posBox, const Quaternionf& rotBox,
                              ContactManifold& manifold) noexcept {
    // Capsule segment endpoints in world space.
    Vec3f localA(0.0f, -capsule.halfHeight, 0.0f);
    Vec3f localB(0.0f, +capsule.halfHeight, 0.0f);
    Vec3f worldA = posCap + rotCap.rotate(localA);
    Vec3f worldB = posCap + rotCap.rotate(localB);

    // Sample N points along the segment and test each as a sphere-box.
    const int kSamples = 7;
    float bestPen = -1e10f;
    Vec3f bestNormal(0.0f, 1.0f, 0.0f);
    Vec3f bestPos(0.0f, 0.0f, 0.0f);
    bool hit = false;

    for (int i = 0; i < kSamples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kSamples - 1);
        Vec3f center = worldA + (worldB - worldA) * t;

        // Sphere-box test with capsule radius.
        // Transform centre into box local space.
        Vec3f localPos = rotBox.rotateInverse(center - posBox);
        Vec3f he = box.halfExtents;

        Vec3f closest = Vec3f(
            detail::clamp(localPos.x, -he.x, he.x),
            detail::clamp(localPos.y, -he.y, he.y),
            detail::clamp(localPos.z, -he.z, he.z)
        );

        Vec3f diff   = localPos - closest;
        float distSq = diff.lengthSquared();
        if (distSq >= capsule.radius * capsule.radius) continue;

        float dist = std::sqrt(distSq);
        float pen  = capsule.radius - dist;

        Vec3f normalLocal;
        if (dist > 1e-8f) {
            normalLocal = diff / dist;
        } else {
            // Inside box — push out along axis of least penetration.
            float minPen = he.x - std::abs(localPos.x);
            int axis     = 0;
            float py     = he.y - std::abs(localPos.y);
            if (py < minPen) { minPen = py; axis = 1; }
            float pz     = he.z - std::abs(localPos.z);
            if (pz < minPen) { minPen = pz; axis = 2; }
            normalLocal = Vec3f::zero();
            normalLocal[axis] = (localPos[axis] >= 0.0f) ? 1.0f : -1.0f;
            pen = capsule.radius + minPen; // depth of sphere center inside box + radius
        }

        if (pen > bestPen) {
            bestPen    = pen;
            bestNormal = rotBox.rotate(normalLocal).normalized();
            bestPos    = rotBox.rotate(closest) + posBox;
            hit = true;
        }
    }

    if (!hit) return false;

    manifold.clearPoints();
    ContactPoint pt;
    pt.position    = bestPos;
    pt.normal      = bestNormal;
    pt.penetration = bestPen;
    pt.featureA    = 0;  // capsule: single point from deepest sample
    pt.featureB    = -1; // will be set to box face index below
    // Encode which box face the closest point is on.
    // The closest point is in box local space (che computed during the loop).
    // We need to recompute the closest feature from bestPos.
    {
        Vec3f localP = rotBox.rotateInverse(bestPos - posBox);
        Vec3f heL    = box.halfExtents;
        for (int a = 0; a < 3; ++a) {
            if (std::abs(localP[a]) >= heL[a] - 1e-6f) {
                pt.featureB = localP[a] >= 0 ? a * 2 : a * 2 + 1;
                break;
            }
        }
        if (pt.featureB < 0) pt.featureB = 0;
    }
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Capsule ↔ Plane
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Capsule-plane collision.  Finds the deepest point of the capsule segment
///         relative to the plane and generates a contact if the radius penetrates.
inline bool collideCapsulePlane(const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                                const Plane& plane, const Vec3f& posPlane, const Quaternionf& rotPlane,
                                ContactManifold& manifold) noexcept {
    // World-space plane normal.
    Vec3f nWorld = rotPlane.rotate(plane.normal).normalized();

    // Capsule segment endpoints in world space.
    Vec3f localA(0.0f, -capsule.halfHeight, 0.0f);
    Vec3f localB(0.0f, +capsule.halfHeight, 0.0f);
    Vec3f worldA = posCap + rotCap.rotate(localA);
    Vec3f worldB = posCap + rotCap.rotate(localB);

    // The deepest point on the capsule into the plane is either one of the endpoints
    // or the point on the segment farthest in the -nWorld direction.
    float dA = (worldA - posPlane).dot(nWorld) - plane.distance;
    float dB = (worldB - posPlane).dot(nWorld) - plane.distance;

    // Signed distance of the deepest capsule point to the plane.
    float minD = std::min(dA, dB);
    float pen  = capsule.radius - minD;
    if (pen <= 0.0f) return false;

    // The contact point on the capsule is at the deepest point, projected onto the plane.
    Vec3f deepest = (dA < dB) ? worldA : worldB;
    Vec3f contactOnCapsule = deepest - nWorld * capsule.radius;
    Vec3f contactOnPlane   = deepest - nWorld * minD;

    ContactPoint pt;
    pt.position    = (contactOnCapsule + contactOnPlane) * 0.5f;
    pt.normal      = -nWorld;   // points from plane toward capsule
    pt.penetration = pen;
    // Encode which end of the capsule (0=bottom, 1=top, 2=body).
    pt.featureA    = (dA < dB) ? 2 : 1;  // deepest end: A=bottom(2), B=top(1)
    pt.featureB    = 0;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  ConvexMesh ↔ Plane  (SAT-style vertex sweep)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Collide a ConvexMesh against an infinite plane by sweeping all vertices.
///
/// Transforms each vertex of the convex mesh into world space and computes its signed
/// distance to the plane.  The most penetrating vertex (minimum signed distance) gives
/// the contact point and penetration depth.
///
/// This is O(n) in the vertex count of the mesh and produces a single contact at the
/// deepest vertex.
///
/// @return  True if the mesh penetrates the plane.
inline bool collideConvexMeshPlane(const ConvexMesh& mesh, const Vec3f& posM, const Quaternionf& rotM,
                                    const Plane& plane, const Vec3f& posP, const Quaternionf& rotP,
                                    ContactManifold& manifold) noexcept {
    // World-space plane normal and distance.
    Vec3f nWorld = rotP.rotate(plane.normal).normalized();
    float dWorld = plane.distance;  // plane equation in local space: normal · p = distance
    // Plane in world space: nWorld · (p - posP) = dWorld  =>  nWorld · p = nWorld · posP + dWorld
    float planeD = nWorld.dot(posP) + dWorld;

    if (mesh.vertexCount <= 0 || !mesh.vertices) return false;

    float minDist = std::numeric_limits<float>::max();
    int deepestIdx = 0;

    for (int i = 0; i < mesh.vertexCount; ++i) {
        Vec3f worldV = posM + rotM.rotate(mesh.vertices[i]);
        float signedDist = nWorld.dot(worldV) - planeD;
        if (signedDist < minDist) {
            minDist = signedDist;
            deepestIdx = i;
        }
    }

    // Penetration depth = -minDist (positive when the vertex is below the plane).
    float pen = -minDist;
    if (pen <= 0.0f) return false;

    Vec3f deepestWorld = posM + rotM.rotate(mesh.vertices[deepestIdx]);
    // Project the deepest vertex onto the plane surface.
    Vec3f contactOnPlane = deepestWorld - nWorld * minDist;  // minDist is negative
    // The contact point is on the plane surface (midpoint with the vertex).
    Vec3f contactPos = (deepestWorld + contactOnPlane) * 0.5f;

    ContactPoint pt;
    pt.position    = contactPos;
    pt.normal      = -nWorld;      // points from plane toward mesh
    pt.penetration = pen;
    pt.featureA    = deepestIdx;   // encodes which vertex is deepest
    pt.featureB    = 0;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Box ↔ TriangleMesh
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Box vs TriangleMesh collision using vertex penetration + triangle-vs-box tests.
///
/// For each triangle overlapping the box AABB:
///   1. Checks each box corner against the triangle plane (vertex-in-triangle test).
///   2. Checks each triangle vertex against the box (point-in-box test).
/// Generates up to 4 contacts from the deepest penetrations.
/// BVH-accelerated when available.
inline bool collideBoxTriangleMesh(const TriangleMesh& mesh,
                                   const Vec3f& posM, const Quaternionf& rotM,
                                   const Box& box, const Vec3f& posB, const Quaternionf& rotB,
                                   ContactManifold& manifold) noexcept {
    manifold.clearPoints();
    bool hit = false;

    // Compute box corners in world space.
    Vec3f corners[8];
    detail::getBoxCorners(posB, rotB, box.halfExtents, corners);

    // Compute box AABB in mesh local space (for BVH overlap query).
    Vec3f boxWorldHe;
    {
        Mat3f m = rotB.toMat3();
        Vec3f he = box.halfExtents;
        boxWorldHe = Vec3f(
            std::abs(m(0,0)) * he.x + std::abs(m(0,1)) * he.y + std::abs(m(0,2)) * he.z,
            std::abs(m(1,0)) * he.x + std::abs(m(1,1)) * he.y + std::abs(m(1,2)) * he.z,
            std::abs(m(2,0)) * he.x + std::abs(m(2,1)) * he.y + std::abs(m(2,2)) * he.z
        );
    }
    AABB boxWorldAABB = {posB - boxWorldHe, posB + boxWorldHe};

    // Transform box AABB to mesh local space.
    Vec3f localMin = rotM.rotateInverse(boxWorldAABB.min - posM);
    Vec3f localMax = rotM.rotateInverse(boxWorldAABB.max - posM);
    AABB meshSpaceBox;
    meshSpaceBox.min = Vec3f(std::min(localMin.x, localMax.x),
                             std::min(localMin.y, localMax.y),
                             std::min(localMin.z, localMax.z));
    meshSpaceBox.max = Vec3f(std::max(localMin.x, localMax.x),
                             std::max(localMin.y, localMax.y),
                             std::max(localMin.z, localMax.z));

    auto processTri = [&](const Vec3f& t0, const Vec3f& t1, const Vec3f& t2) {
        // World-space triangle vertices.
        Vec3f wt0 = posM + rotM.rotate(t0);
        Vec3f wt1 = posM + rotM.rotate(t1);
        Vec3f wt2 = posM + rotM.rotate(t2);

        // Triangle normal (winding-dependent — we handle both sides).
        Vec3f edge1 = wt1 - wt0;
        Vec3f edge2 = wt2 - wt0;
        Vec3f triNormal = edge1.cross(edge2);
        float nLen = triNormal.length();
        if (nLen < 1e-12f) return; // degenerate triangle
        triNormal = triNormal / nLen;

        // Triangle plane: normal · p = d
        float triD = triNormal.dot(wt0);

        // 1. Check box corners against triangle.
        // We use a winding-independent approach: any corner whose projection into the
        // triangle plane falls inside the triangle generates a contact. The contact
        // normal always points FROM the triangle surface AWAY from the mesh interior,
        // which is the direction of -triNormal for a well-formed mesh.
        //
        // NOTE: sd = triNormal·corner - triD.
        //   For a ground mesh with triNormal = (0,-1,0):
        //     sd < 0  → corner above the surface (no penetration, skip)
        //     sd > 0  → corner below the surface (penetrating, process)
        //   The contact normal points opposite to triNormal (i.e., outward from the
        //   mesh surface), regardless of which side the corner is on.
        Vec3f outwardNormal = (triNormal.lengthSquared() > 1e-12f)
                              ? -triNormal / triNormal.length()
                              : Vec3f(0.0f, 1.0f, 0.0f);
        for (int i = 0; i < 8; ++i) {
            float sd = triNormal.dot(corners[i]) - triD;
            if (sd < 1e-8f) continue; // at or above the surface — no penetration

            // Project corner onto triangle plane.
            Vec3f proj = corners[i] - triNormal * sd;

            // Check if projection is inside the triangle using barycentric closest-point.
            Vec3f closest, bary;
            tri_detail::closestPointOnTriangle(proj, wt0, wt1, wt2, closest, bary);
            Vec3f diff = proj - closest;
            if (diff.lengthSquared() > 1e-6f) continue; // projection outside triangle

            // Valid contact: corner penetrating through this triangle.
            // Use the outward-facing normal (from mesh surface toward exterior).
            // The penetration depth is the actual Euclidean distance from corner
            // to the closest point on the triangle surface.
            Vec3f contactDir = corners[i] - closest;
            float dirLen = contactDir.length();
            ContactPoint pt;
            pt.position    = closest;
            pt.normal      = outwardNormal;
            pt.penetration = dirLen > 1e-8f ? dirLen : sd;
            manifold.addPoint(pt);
            hit = true;
        }

        // 2. Check triangle vertices against box (if not too many contacts already).
        if (manifold.pointCount >= 4) return;

        Vec3f triVerts[3] = {wt0, wt1, wt2};
        for (int v = 0; v < 3; ++v) {
            // Transform vertex into box local space.
            Vec3f localV = rotB.rotateInverse(triVerts[v] - posB);
            Vec3f he = box.halfExtents;

            // Check if vertex is inside the box.
            if (std::abs(localV.x) > he.x + 1e-6f ||
                std::abs(localV.y) > he.y + 1e-6f ||
                std::abs(localV.z) > he.z + 1e-6f) continue;

            // Vertex is inside the box. Find closest box face.
            float minPen = he.x - std::abs(localV.x);
            int axis = 0;
            float penY = he.y - std::abs(localV.y);
            if (penY < minPen) { minPen = penY; axis = 1; }
            float penZ = he.z - std::abs(localV.z);
            if (penZ < minPen) { minPen = penZ; axis = 2; }

            Vec3f normalLocal = Vec3f::zero();
            normalLocal[axis] = (localV[axis] >= 0.0f) ? 1.0f : -1.0f;
            Vec3f worldNormal = rotB.rotate(normalLocal).normalized();

            ContactPoint pt;
            pt.position    = triVerts[v];
            pt.normal      = -worldNormal; // points from box toward mesh
            pt.penetration = minPen;
            manifold.addPoint(pt);
            hit = true;
            if (manifold.pointCount >= 4) return;
        }
    };

    // Process overlapping triangles via BVH or linear scan.
    if (mesh.bvh && mesh.triangleCount > 0) {
        int triBuffer = 0;
        int outCount = 0;
        constexpr int kMaxTris = 512;
        int triIds[kMaxTris];

        if (mesh.bvh->queryOverlap(meshSpaceBox, mesh.vertices, mesh.indices,
                                    triBuffer, kMaxTris, triIds, outCount)) {
            for (int i = 0; i < outCount && i < kMaxTris; ++i) {
                int t = triIds[i];
                const Vec3f& v0 = mesh.vertices[mesh.indices[t * 3 + 0]];
                const Vec3f& v1 = mesh.vertices[mesh.indices[t * 3 + 1]];
                const Vec3f& v2 = mesh.vertices[mesh.indices[t * 3 + 2]];
                processTri(v0, v1, v2);
                if (manifold.pointCount >= 4) break;
            }
        }
    } else {
        // Linear scan fallback (capped at 1024 triangles for performance).
        int maxTris = std::min(mesh.triangleCount, 1024);
        for (int i = 0; i < maxTris; ++i) {
            const Vec3f& v0 = mesh.vertices[mesh.indices[i * 3 + 0]];
            const Vec3f& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
            const Vec3f& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];
            processTri(v0, v1, v2);
            if (manifold.pointCount >= 4) break;
        }
    }

    return hit;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Capsule ↔ TriangleMesh
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Capsule vs TriangleMesh collision via segment sampling.
///
/// Samples points along the capsule's medial segment and tests each as a sphere-vs-triangle
/// collision.  Retains the deepest penetration.  BVH-accelerated when available.
inline bool collideCapsuleTriangleMesh(const TriangleMesh& mesh,
                                        const Vec3f& posM, const Quaternionf& rotM,
                                        const Capsule& capsule, const Vec3f& posCap, const Quaternionf& rotCap,
                                        ContactManifold& manifold) noexcept {
    manifold.clearPoints();

    // Capsule segment endpoints in world space.
    Vec3f localA(0.0f, -capsule.halfHeight, 0.0f);
    Vec3f localB(0.0f, +capsule.halfHeight, 0.0f);
    Vec3f worldA = posCap + rotCap.rotate(localA);
    Vec3f worldB = posCap + rotCap.rotate(localB);
    float radius = capsule.radius;

    // Compute capsule AABB in mesh local space.
    Vec3f capMin(
        std::min(worldA.x, worldB.x) - radius,
        std::min(worldA.y, worldB.y) - radius,
        std::min(worldA.z, worldB.z) - radius
    );
    Vec3f capMax(
        std::max(worldA.x, worldB.x) + radius,
        std::max(worldA.y, worldB.y) + radius,
        std::max(worldA.z, worldB.z) + radius
    );
    Vec3f localMin = rotM.rotateInverse(capMin - posM);
    Vec3f localMax = rotM.rotateInverse(capMax - posM);
    AABB capsuleAABB;
    capsuleAABB.min = Vec3f(std::min(localMin.x, localMax.x),
                             std::min(localMin.y, localMax.y),
                             std::min(localMin.z, localMax.z));
    capsuleAABB.max = Vec3f(std::max(localMin.x, localMax.x),
                             std::max(localMin.y, localMax.y),
                             std::max(localMin.z, localMax.z));

    const int kSamples = 5;
    float bestPen = 0.0f;
    Vec3f bestPos = Vec3f::zero();
    Vec3f bestNormal = Vec3f::unitY();
    bool hit = false;

    auto processTri = [&](const Vec3f& t0, const Vec3f& t1, const Vec3f& t2) {
        // World-space triangle vertices.
        Vec3f wt0 = posM + rotM.rotate(t0);
        Vec3f wt1 = posM + rotM.rotate(t1);
        Vec3f wt2 = posM + rotM.rotate(t2);

        for (int s = 0; s < kSamples; ++s) {
            float t = static_cast<float>(s) / static_cast<float>(kSamples - 1);
            Vec3f sphereCenter = worldA + (worldB - worldA) * t;

            ContactManifold tmpManifold;
            if (tri_detail::collideSphereTriangle(sphereCenter, radius, wt0, wt1, wt2, tmpManifold)) {
                for (int p = 0; p < tmpManifold.pointCount; ++p) {
                    if (tmpManifold.points[p].penetration > bestPen) {
                        bestPen = tmpManifold.points[p].penetration;
                        bestPos = tmpManifold.points[p].position;
                        bestNormal = tmpManifold.points[p].normal;
                        hit = true;
                    }
                }
            }
        }
    };

    // Process overlapping triangles via BVH or linear scan.
    if (mesh.bvh && mesh.triangleCount > 0) {
        int triBuffer = 0;
        int outCount = 0;
        constexpr int kMaxTris = 512;
        int triIds[kMaxTris];

        if (mesh.bvh->queryOverlap(capsuleAABB, mesh.vertices, mesh.indices,
                                    triBuffer, kMaxTris, triIds, outCount)) {
            for (int i = 0; i < outCount && i < kMaxTris; ++i) {
                int t = triIds[i];
                const Vec3f& v0 = mesh.vertices[mesh.indices[t * 3 + 0]];
                const Vec3f& v1 = mesh.vertices[mesh.indices[t * 3 + 1]];
                const Vec3f& v2 = mesh.vertices[mesh.indices[t * 3 + 2]];
                processTri(v0, v1, v2);
            }
        }
    } else {
        int maxTris = std::min(mesh.triangleCount, 1024);
        for (int i = 0; i < maxTris; ++i) {
            const Vec3f& v0 = mesh.vertices[mesh.indices[i * 3 + 0]];
            const Vec3f& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
            const Vec3f& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];
            processTri(v0, v1, v2);
        }
    }

    if (!hit) return false;

    // Ensure the normal points from mesh toward capsule body.
    Vec3f meshToCapDir = (posCap - posM).normalized();
    if (bestNormal.dot(meshToCapDir) < 0.0f) {
        bestNormal = -bestNormal;
    }

    manifold.clearPoints();
    ContactPoint pt;
    pt.position    = bestPos;
    pt.normal      = bestNormal;
    pt.penetration = bestPen;
    manifold.addPoint(pt);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  ConvexMesh ↔ TriangleMesh  (GJK per triangle)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Collide a ConvexMesh against a TriangleMesh using vertex-plane sweep.
///
/// For each triangle overlapping the convex mesh AABB, projects each convex hull vertex
/// onto the triangle plane.  If a vertex is on the negative side of the plane and its
/// projection falls inside the triangle, a contact is generated.  Also checks triangle
/// vertices against the convex hull (point-in-convex-hull test via vertex-plane dist).
///
/// @note  GJK cannot be used for triangle meshes because triangles have zero thickness
///        in one dimension, causing the Minkowski difference to miss interpenetration.
///        This vertex-sweep approach is correct for one-sided triangle surfaces.
///        BVH-accelerated when available.
inline bool collideConvexMeshTriangleMesh(const TriangleMesh& mesh,
                                           const Vec3f& posM, const Quaternionf& rotM,
                                           const ConvexMesh& convex, const Vec3f& posC, const Quaternionf& rotC,
                                           ContactManifold& manifold) noexcept {
    manifold.clearPoints();
    bool hit = false;

    if (convex.vertexCount <= 0 || !convex.vertices) return false;

    // Compute convex mesh world-space AABB.
    Vec3f worldHe;
    {
        Mat3f m = rotC.toMat3();
        Vec3f he = convex.halfExtents;
        worldHe = Vec3f(
            std::abs(m(0,0)) * he.x + std::abs(m(0,1)) * he.y + std::abs(m(0,2)) * he.z,
            std::abs(m(1,0)) * he.x + std::abs(m(1,1)) * he.y + std::abs(m(1,2)) * he.z,
            std::abs(m(2,0)) * he.x + std::abs(m(2,1)) * he.y + std::abs(m(2,2)) * he.z
        );
    }
    Vec3f worldCenter = posC + rotC.rotate(convex.center);
    AABB worldAABB = {worldCenter - worldHe, worldCenter + worldHe};

    // Transform world AABB to mesh local space.
    Vec3f lMin = rotM.rotateInverse(worldAABB.min - posM);
    Vec3f lMax = rotM.rotateInverse(worldAABB.max - posM);
    AABB meshSpaceAABB;
    meshSpaceAABB.min = Vec3f(std::min(lMin.x, lMax.x),
                               std::min(lMin.y, lMax.y),
                               std::min(lMin.z, lMax.z));
    meshSpaceAABB.max = Vec3f(std::max(lMin.x, lMax.x),
                               std::max(lMin.y, lMax.y),
                               std::max(lMin.z, lMax.z));

    float bestPen = 0.0f;
    Vec3f bestPos = Vec3f::zero();
    Vec3f bestNormal = Vec3f::unitY();

    // Pre-compute convex hull world-space vertices.
    // Use a small stack buffer for typical convex hulls (up to 128 verts).
    constexpr int kMaxStackVerts = 128;
    Vec3f worldVerts[kMaxStackVerts];
    int numWorldVerts = std::min(convex.vertexCount, kMaxStackVerts);
    for (int i = 0; i < numWorldVerts; ++i) {
        worldVerts[i] = posC + rotC.rotate(convex.vertices[i]);
    }

    auto processTri = [&](const Vec3f& t0, const Vec3f& t1, const Vec3f& t2) {
        // World-space triangle vertices.
        Vec3f wt0 = posM + rotM.rotate(t0);
        Vec3f wt1 = posM + rotM.rotate(t1);
        Vec3f wt2 = posM + rotM.rotate(t2);

        // Triangle normal (winding-dependent).
        Vec3f edge1 = wt1 - wt0;
        Vec3f edge2 = wt2 - wt0;
        Vec3f triNormal = edge1.cross(edge2);
        float nLen = triNormal.length();
        if (nLen < 1e-12f) return;
        triNormal = triNormal / nLen;

        // Triangle plane: normal · p = d
        float triD = triNormal.dot(wt0);

        // 1. Check each convex hull vertex against the triangle plane.
        // Uses winding-independent approach: any vertex whose projection onto the
        // triangle plane falls inside the triangle generates a contact.
        // The contact normal always points outward from the mesh surface (-
        // triNormal), which is the direction to push the other body away.
        // Vertices at or above the surface are skipped (sd < 0 = above).
        Vec3f outwardNormal = (triNormal.lengthSquared() > 1e-12f)
                              ? -triNormal / triNormal.length()
                              : Vec3f(0.0f, 1.0f, 0.0f);
        for (int i = 0; i < numWorldVerts; ++i) {
            float sd = triNormal.dot(worldVerts[i]) - triD;
            if (sd < 1e-8f) continue; // at or above the surface — no penetration

            // Project vertex onto triangle plane.
            Vec3f proj = worldVerts[i] - triNormal * sd;

            // Check if projection is inside the triangle.
            Vec3f closest, bary;
            tri_detail::closestPointOnTriangle(proj, wt0, wt1, wt2, closest, bary);
            Vec3f diff = proj - closest;
            if (diff.lengthSquared() > 1e-6f) continue; // projection outside triangle

            // Valid contact: vertex penetrating through this triangle.
            // Use outward-facing normal, penetration = Euclidean distance to surface.
            Vec3f contactDir = worldVerts[i] - closest;
            float dirLen = contactDir.length();
            if (dirLen > bestPen) {
                bestPen = dirLen > 1e-8f ? dirLen : sd;
                bestPos = closest;
                bestNormal = outwardNormal;
                hit = true;
            }
        }

        // 2. Check triangle vertices against convex hull.
        // (Uses a simple point-in-convex-hull test via signed distance to hull planes,
        // which is approximated by checking if the point is inside the convex AABB.)
        if (hit && bestPen > 0.01f) return; // skip if we already have a deep contact

        Vec3f triVerts[3] = {wt0, wt1, wt2};
        for (int v = 0; v < 3; ++v) {
            Vec3f localV = rotC.rotateInverse(triVerts[v] - posC);
            // Quick AABB check: is the vertex inside the convex hull's local AABB?
            if (std::abs(localV.x) > convex.halfExtents.x + 1e-4f ||
                std::abs(localV.y) > convex.halfExtents.y + 1e-4f ||
                std::abs(localV.z) > convex.halfExtents.z + 1e-4f) continue;

            // Vertex may be inside the convex hull. Use the hull center as reference
            // and find the support direction that most opposes this vertex.
            Vec3f dirToVert = triVerts[v] - worldCenter;
            float dLen = dirToVert.length();
            if (dLen < 1e-8f) continue;
            dirToVert = dirToVert / dLen;

            // Get support point in the direction opposite to the vertex.
            Vec3f support = gjk_detail::supportConvexMeshWorld(convex, posC, rotC, -dirToVert);

            // If the vertex is beyond the support point in this direction, it's outside.
            float supportDot = support.dot(dirToVert);
            float vertDot = triVerts[v].dot(dirToVert);
            float pen = supportDot - vertDot;
            if (pen > 0.0f && pen > bestPen) {
                bestPen = pen;
                bestPos = triVerts[v];
                bestNormal = dirToVert; // points from hull toward vertex
                hit = true;
            }
        }
    };

    // Process overlapping triangles via BVH or linear scan.
    if (mesh.bvh && mesh.triangleCount > 0) {
        int triBuffer = 0;
        int outCount = 0;
        constexpr int kMaxTris = 512;
        int triIds[kMaxTris];

        if (mesh.bvh->queryOverlap(meshSpaceAABB, mesh.vertices, mesh.indices,
                                    triBuffer, kMaxTris, triIds, outCount)) {
            for (int i = 0; i < outCount && i < kMaxTris; ++i) {
                int t = triIds[i];
                const Vec3f& v0 = mesh.vertices[mesh.indices[t * 3 + 0]];
                const Vec3f& v1 = mesh.vertices[mesh.indices[t * 3 + 1]];
                const Vec3f& v2 = mesh.vertices[mesh.indices[t * 3 + 2]];
                processTri(v0, v1, v2);
            }
        }
    } else {
        // Linear scan fallback — capped at 256 triangles.
        int maxTris = std::min(mesh.triangleCount, 256);
        for (int i = 0; i < maxTris; ++i) {
            const Vec3f& v0 = mesh.vertices[mesh.indices[i * 3 + 0]];
            const Vec3f& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
            const Vec3f& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];
            processTri(v0, v1, v2);
        }
    }

    if (!hit) return false;

    manifold.clearPoints();
    ContactPoint pt;
    pt.position    = bestPos;
    pt.normal      = bestNormal;
    pt.penetration = bestPen;
    manifold.addPoint(pt);
    return true;
}

} // namespace ciphyxs
