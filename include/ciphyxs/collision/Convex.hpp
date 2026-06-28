//==================================================================================================
/// @file  Convex.hpp
/// @brief  GJK (Gilbert-Johnson-Keerthi) + EPA (Expanding Polytope Algorithm) collision detection
///         for arbitrary convex shapes, including ConvexMesh.
///
/// GJK computes the distance between two convex shapes.  If the shapes intersect (distance ≤ 0),
/// EPA expands the GJK simplex into a polytope to extract the contact normal and penetration depth.
///
/// ##  Support Function Abstraction
///
/// GJK/EPA only require a **support function** per shape: `getFarthestPointInDirection(dir) → Vec3f`.
/// This file provides support functions for Sphere, Box, Capsule, and ConvexMesh, enabling
/// collision between **any** pair of these types (including e.g. ConvexMesh↔Sphere, ConvexMesh↔Box).
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include "../math/Mat3.hpp"
#include "Contact.hpp"
#include "Shape.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cfloat>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Support Functions
// ────────────────────────────────────────────────────────────────────────────────────────────────

namespace gjk_detail {

/// @brief  Get the farthest vertex of a ConvexMesh in a given direction (local space).
inline Vec3f supportConvexMesh(const ConvexMesh& mesh, const Vec3f& dir) noexcept {
    float bestDot = -std::numeric_limits<float>::max();
    Vec3f bestPt = Vec3f::zero();
    const float eps = 1e-12f;
    // Early out for degenerate direction.
    float lenSq = dir.lengthSquared();
    if (lenSq < eps) return mesh.center;
    for (int i = 0; i < mesh.vertexCount; ++i) {
        float d = mesh.vertices[i].dot(dir);
        if (d > bestDot) {
            bestDot = d;
            bestPt = mesh.vertices[i];
        }
    }
    return bestPt;
}

/// @brief  Get the farthest point of a Sphere in a given direction (world space).
inline Vec3f supportSphere(const Sphere& sphere, const Vec3f& center,
                           const Vec3f& dir) noexcept {
    float len = dir.length();
    if (len < 1e-12f) return center;
    return center + (dir / len) * sphere.radius;
}

/// @brief  Get the farthest point of a Box in a given direction (world frame).
inline Vec3f supportBox(const Box& box, const Vec3f& center, const Quaternionf& rot,
                        const Vec3f& dir) noexcept {
    // Transform direction into local space, find extremal corner, transform back.
    Vec3f localDir = rot.rotateInverse(dir);
    Vec3f localPt(
        (localDir.x >= 0) ? box.halfExtents.x : -box.halfExtents.x,
        (localDir.y >= 0) ? box.halfExtents.y : -box.halfExtents.y,
        (localDir.z >= 0) ? box.halfExtents.z : -box.halfExtents.z
    );
    return center + rot.rotate(localPt);
}

/// @brief  Get the farthest point of a Capsule in a given direction (world frame).
inline Vec3f supportCapsule(const Capsule& capsule, const Vec3f& center,
                            const Quaternionf& rot, const Vec3f& dir) noexcept {
    // Capsule is a line segment [0,0,-h] to [0,0,+h] in local Y, swept with radius.
    // Support = support of segment + support of sphere of radius along dir.
    Vec3f localDir = rot.rotateInverse(dir);
    // Support of the segment along localDir: pick the endpoint with larger dot product.
    Vec3f segEnd(0.0f, (localDir.y >= 0) ? capsule.halfHeight : -capsule.halfHeight, 0.0f);
    // Support of the sphere of radius capsule.radius.
    float len = localDir.length();
    Vec3f sphereOffset(0, 0, 0);
    if (len > 1e-12f) {
        sphereOffset = (localDir / len) * capsule.radius;
    }
    return center + rot.rotate(segEnd + sphereOffset);
}

/// @brief  Get the farthest point of a ConvexMesh in a given direction (world frame).
inline Vec3f supportConvexMeshWorld(const ConvexMesh& mesh, const Vec3f& center,
                                     const Quaternionf& rot, const Vec3f& dir) noexcept {
    // Transform direction to local space, find support, transform back.
    Vec3f localDir = rot.rotateInverse(dir);
    Vec3f localPt = supportConvexMesh(mesh, localDir);
    return center + rot.rotate(localPt);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  GJK  —  Gilbert-Johnson-Keerthi Distance Algorithm
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Maximum GJK iterations before falling back.
constexpr int kGJKMaxIterations = 64;

/// @brief  Tolerance for GJK convergence.
constexpr float kGJKTolerance = 1e-6f;

/// @brief  GJK simplex vertex.
struct SimplexVertex {
    Vec3f p;   ///< Point in Minkowski difference (A - B).
    Vec3f pA;  ///< Corresponding point on shape A.
    Vec3f pB;  ///< Corresponding point on shape B.
};

/// @brief  GJK simplex (up to 4 vertices: point, line, triangle, tetrahedron).
struct Simplex {
    SimplexVertex vertices[4];
    int count = 0;
};

/// @brief  Compute the closest point on a line segment (2-point simplex) to the origin.
///         Returns the barycentric coordinates.
inline float closestPtSegmentToOrigin(const Vec3f& a, const Vec3f& b,
                                       Vec3f& closest) noexcept {
    Vec3f ab = b - a;
    float abLenSq = ab.lengthSquared();
    if (abLenSq < 1e-30f) {
        closest = a;
        return 1.0f; // t = 1 => closest is a
    }
    float t = -a.dot(ab) / abLenSq;
    t = std::clamp(t, 0.0f, 1.0f);
    closest = a + ab * t;
    return t;
}

/// @brief  Compute the closest point on a triangle (3-point simplex) to the origin.
///         Uses barycentric coordinates.  Returns false if degenerate.
inline bool closestPtTriangleToOrigin(const Vec3f& a, const Vec3f& b, const Vec3f& c,
                                       Vec3f& closest) noexcept {
    // Edge vectors.
    Vec3f ab = b - a;
    Vec3f ac = c - a;
    Vec3f normal = ab.cross(ac);
    float nLenSq = normal.lengthSquared();
    if (nLenSq < 1e-30f) {
        // Degenerate triangle — fall back to closest on edges.
        Vec3f cl1, cl2, cl3;
        closestPtSegmentToOrigin(a, b, cl1);
        closestPtSegmentToOrigin(a, c, cl2);
        closestPtSegmentToOrigin(b, c, cl3);
        float d1 = cl1.lengthSquared();
        float d2 = cl2.lengthSquared();
        float d3 = cl3.lengthSquared();
        if (d1 <= d2 && d1 <= d3) { closest = cl1; return true; }
        if (d2 <= d3) { closest = cl2; return true; }
        closest = cl3;
        return true;
    }

    // Check if origin is inside the prism of the triangle (project onto normal).
    normal = normal / std::sqrt(nLenSq);

    // If the origin is on the negative side of the triangle plane, flip.
    // Actually, for GJK we care about distance to the triangle region.
    // Project origin onto the triangle plane.
    float dist = normal.dot(a); // signed distance from origin to plane
    Vec3f proj = a - normal * dist; // projection of origin onto plane

    // Barycentric coordinates of the projection.
    Vec3f v0 = c - a;
    Vec3f v1 = b - a;
    Vec3f v2 = proj - a;
    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);
    float denom = d00 * d11 - d01 * d01;

    if (std::abs(denom) < 1e-30f) {
        closest = proj;
        return true;
    }

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    // Clamp to triangle region.
    if (u >= 0 && v >= 0 && w >= 0) {
        // Inside triangle.
        closest = a * u + b * v + c * w;
        return true;
    }

    // Outside triangle — find closest on edges.
    Vec3f cl1, cl2, cl3;
    closestPtSegmentToOrigin(a, b, cl1);
    closestPtSegmentToOrigin(a, c, cl2);
    closestPtSegmentToOrigin(b, c, cl3);
    float d1 = cl1.lengthSquared();
    float d2 = cl2.lengthSquared();
    float d3 = cl3.lengthSquared();
    if (d1 <= d2 && d1 <= d3) { closest = cl1; return true; }
    if (d2 <= d3) { closest = cl2; return true; }
    closest = cl3;
    return true;
}

/// @brief  Check if the origin (0,0,0) is strictly inside the tetrahedron defined by
///         four Minkowski-difference points.  Uses a volumetric sign test on each face.
///
/// For a tetrahedron with vertices v0,v1,v2,v3 forming 4 faces, the origin is inside
/// iff it lies on the **opposite** side of each face from the face's opposite vertex.
/// @return  true if the origin is inside the tetrahedron.
inline bool originInsideTetrahedron(const Vec3f& v0, const Vec3f& v1,
                                     const Vec3f& v2, const Vec3f& v3) noexcept {
    constexpr float eps = 1e-12f;

    // Fast AABB rejection: if the origin is outside all vertices' bounding box,
    // it cannot be inside the tetrahedron.  This avoids false positives when GJK
    // produces a near-degenerate tetrahedron (all vertices clustered) for far-apart shapes.
    float minX = std::min({v0.x, v1.x, v2.x, v3.x});
    float maxX = std::max({v0.x, v1.x, v2.x, v3.x});
    if (0.0f < minX || 0.0f > maxX) return false;
    float minY = std::min({v0.y, v1.y, v2.y, v3.y});
    float maxY = std::max({v0.y, v1.y, v2.y, v3.y});
    if (0.0f < minY || 0.0f > maxY) return false;
    float minZ = std::min({v0.z, v1.z, v2.z, v3.z});
    float maxZ = std::max({v0.z, v1.z, v2.z, v3.z});
    if (0.0f < minZ || 0.0f > maxZ) return false;

    // Volumetric test: for each face, the origin must lie on the OPPOSITE side
    // of the face from the opposite vertex.
    // n·(oppositeV - faceRef) and n·(origin - faceRef) must have OPPOSITE signs
    // (product < 0).  If product > 0, origin is on same side → outside.

    // Face (1,2,3) — opposite v0
    Vec3f n = (v2 - v1).cross(v3 - v1);
    if (n.dot(v0 - v1) * n.dot(-v1) > eps) return false;

    // Face (0,2,3) — opposite v1
    n = (v2 - v0).cross(v3 - v0);
    if (n.dot(v1 - v0) * n.dot(-v0) > eps) return false;

    // Face (0,1,3) — opposite v2
    n = (v1 - v0).cross(v3 - v0);
    if (n.dot(v2 - v0) * n.dot(-v0) > eps) return false;

    // Face (0,1,2) — opposite v3
    n = (v1 - v0).cross(v2 - v0);
    if (n.dot(v3 - v0) * n.dot(-v0) > eps) return false;

    return true;
}

/// @brief  Reduce a 4-point simplex (tetrahedron) to the smallest subset whose
///         closest point to the origin is the same.  Returns the closest point.
///         This is the core of the GJK loop.
///
/// @note  When the tetrahedron contains the origin, sets closest = 0 to signal
///        the calling GJK loop that the shapes intersect.
inline void reduceSimplex(Simplex& s, Vec3f& closest, int& numDirs) noexcept {
    closest = Vec3f::zero();
    numDirs = 0;

    if (s.count == 1) {
        closest = s.vertices[0].p;
        return;
    }

    if (s.count == 2) {
        closestPtSegmentToOrigin(s.vertices[0].p, s.vertices[1].p, closest);
        return;
    }

    if (s.count == 3) {
        closestPtTriangleToOrigin(s.vertices[0].p, s.vertices[1].p, s.vertices[2].p, closest);
        return;
    }

    // Tetrahedron: first check if the origin is inside.
    if (originInsideTetrahedron(s.vertices[0].p, s.vertices[1].p,
                                s.vertices[2].p, s.vertices[3].p)) {
        // Origin is inside the tetrahedron → shapes intersect.
        closest = Vec3f::zero();
        return;
    }

    // Origin is outside — find which face is closest to the origin.
    constexpr int faceIndices[4][3] = {
        {0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {1, 3, 2}
    };

    float bestDistSq = std::numeric_limits<float>::max();
    int bestFace = -1;
    Vec3f bestClosest = Vec3f::zero();

    for (int f = 0; f < 4; ++f) {
        int i0 = faceIndices[f][0];
        int i1 = faceIndices[f][1];
        int i2 = faceIndices[f][2];
        Vec3f fc;
        closestPtTriangleToOrigin(s.vertices[i0].p, s.vertices[i1].p, s.vertices[i2].p, fc);
        float dsq = fc.lengthSquared();
        if (dsq < bestDistSq) {
            bestDistSq = dsq;
            bestFace = f;
            bestClosest = fc;
        }
    }

    if (bestFace < 0) {
        closest = s.vertices[0].p;
        return;
    }

    // Reduce simplex to the vertices of the best face.
    Simplex reduced;
    reduced.count = 3;
    for (int i = 0; i < 3; ++i) {
        int idx = faceIndices[bestFace][i];
        reduced.vertices[i] = s.vertices[idx];
    }
    s = reduced;
    closest = bestClosest;
}

/// @brief  Run GJK to find closest points between two convex shapes.
///
/// @param supportA  Function returning the farthest point on shape A in world-space direction.
/// @param supportB  Function returning the farthest point on shape B in world-space direction.
/// @param[out] pA   Closest point on shape A (world space).
/// @param[out] pB   Closest point on shape B (world space).
/// @param[out] simplex  Final simplex (for EPA if shapes intersect).
/// @return  Squared distance between shapes.  ≤ 0 means intersection.
template<typename SupportA, typename SupportB>
inline float gjkDistance(const SupportA& supportA, const SupportB& supportB,
                          Vec3f& pA, Vec3f& pB, Simplex& simplex) noexcept {
    simplex.count = 0;

    // Initial search direction: try (1,0,0) first, fall back to other axes
    // to avoid degenerate simplexes when shapes are coaxially aligned.
    Vec3f dir(1.0f, 0.0f, 0.0f);
    Vec3f supA = supportA(dir);
    Vec3f supB = supportB(-dir);
    SimplexVertex sv;
    sv.p  = supA - supB;
    sv.pA = supA;
    sv.pB = supB;

    // If first support is near-origin, try other axes.
    if (sv.p.lengthSquared() < 1e-12f) {
        dir = Vec3f(0.0f, 1.0f, 0.0f);
        supA = supportA(dir);
        supB = supportB(-dir);
        sv.p  = supA - supB;
        sv.pA = supA;
        sv.pB = supB;
    }
    if (sv.p.lengthSquared() < 1e-12f) {
        dir = Vec3f(0.0f, 0.0f, 1.0f);
        supA = supportA(dir);
        supB = supportB(-dir);
        sv.p  = supA - supB;
        sv.pA = supA;
        sv.pB = supB;
    }

    simplex.vertices[0] = sv;
    simplex.count = 1;

    // Use the vector from the first support to the origin as search direction.
    dir = -sv.p;
    float dirLenSq = dir.lengthSquared();
    if (dirLenSq < 1e-12f) {
        // Origin is at the first support point — shapes intersect.
        pA = supA;
        pB = supB;
        return 0.0f;
    }
    dir = dir / std::sqrt(dirLenSq);

    Vec3f closest = sv.p;

    for (int iter = 0; iter < kGJKMaxIterations; ++iter) {
        // Get support in current direction.
        supA = supportA(dir);
        supB = supportB(-dir);
        Vec3f newPt = supA - supB;

        // Check if new point is past the current closest point along dir.
        float dot = newPt.dot(dir);
        if (dot < closest.dot(dir) + kGJKTolerance) {
            // No progress — shapes are separated.
            break;
        }

        // Add to simplex.
        sv.p  = newPt;
        sv.pA = supA;
        sv.pB = supB;
        simplex.vertices[simplex.count] = sv;
        simplex.count++;

        // Reduce simplex and find new closest point.
        int numDirs;
        reduceSimplex(simplex, closest, numDirs);

        // Update search direction.
        float cLenSq = closest.lengthSquared();
        if (cLenSq < kGJKTolerance * kGJKTolerance) {
            // Closest point is at origin — shapes intersect.
            pA = supA;
            pB = supB;
            return 0.0f;
        }

        dir = -closest;
        float dirLen = std::sqrt(dir.lengthSquared());
        if (dirLen < 1e-12f) break;
        dir = dir / dirLen;
    }

    // Compute closest points from the simplex closest point.
    // The closest point in Minkowski space = closest.
    // We need to find the corresponding points on A and B.
    // Use the simplex vertices' weighted contribution.
    if (simplex.count == 1) {
        pA = simplex.vertices[0].pA;
        pB = simplex.vertices[0].pB;
    } else if (simplex.count == 2) {
        // Interpolate between two vertices.
        Vec3f a = simplex.vertices[0].p;
        Vec3f b = simplex.vertices[1].p;
        float t;
        Vec3f ab = b - a;
        float abLenSq = ab.lengthSquared();
        if (abLenSq > 1e-30f) {
            t = -a.dot(ab) / abLenSq;
            t = std::clamp(t, 0.0f, 1.0f);
        } else {
            t = 0.0f;
        }
        pA = simplex.vertices[0].pA + (simplex.vertices[1].pA - simplex.vertices[0].pA) * t;
        pB = simplex.vertices[0].pB + (simplex.vertices[1].pB - simplex.vertices[0].pB) * t;
    } else {
        // For 3+ vertices, use triangle closest point.
        Vec3f closestLocal;
        if (simplex.count >= 3) {
            closestPtTriangleToOrigin(simplex.vertices[0].p, simplex.vertices[1].p,
                                      simplex.vertices[2].p, closestLocal);
        } else {
            closestLocal = simplex.vertices[0].p;
        }
        // Barycentric-like combination on A and B points.
        float totalWeight = 0;
        Vec3f avgA = Vec3f::zero();
        Vec3f avgB = Vec3f::zero();
        for (int i = 0; i < simplex.count; ++i) {
            float w = 1.0f;
            avgA += simplex.vertices[i].pA * w;
            avgB += simplex.vertices[i].pB * w;
            totalWeight += w;
        }
        if (totalWeight > 1e-12f) {
            pA = avgA / totalWeight;
            pB = avgB / totalWeight;
        } else {
            pA = simplex.vertices[0].pA;
            pB = simplex.vertices[0].pB;
        }
    }

    return closest.lengthSquared();
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  EPA  —  Expanding Polytope Algorithm
// ────────────────────────────────────────────────────────────────────────────────────────────────

constexpr int kEPAMaxIterations = 64;
constexpr float kEPATolerance = 1e-4f;

/// @brief  EPA face (triangle).
struct EpaFace {
    int indices[3];  ///< Indices into the polytope vertex array.
    Vec3f normal;    ///< Outward-facing normal.
    float distance;  ///< Distance from origin to the face plane.
};

/// @brief  Compute the normal and distance of a face from 3 vertices.
inline void computeFace(const Vec3f* verts, int i0, int i1, int i2,
                         Vec3f& normal, float& dist) noexcept {
    Vec3f a = verts[i0];
    Vec3f b = verts[i1];
    Vec3f c = verts[i2];
    Vec3f ab = b - a;
    Vec3f ac = c - a;
    normal = ab.cross(ac);
    float len = normal.length();
    if (len > 1e-12f) {
        normal = normal / len;
        dist = normal.dot(a);
    } else {
        normal = Vec3f(0, 1, 0);
        dist = 0;
    }
}

/// @brief  Run EPA to compute penetration depth and contact normal for intersecting shapes.
///
/// @param supportA   Support function for shape A.
/// @param supportB   Support function for shape B.
/// @param simplex    Initial GJK simplex (must contain the origin).
/// @param[out] pA    Contact point on A.
/// @param[out] pB    Contact point on B.
/// @param[out] normal World-space contact normal (from A toward B).
/// @return  Penetration depth (> 0).
template<typename SupportA, typename SupportB>
inline float epaPenetration(const SupportA& supportA, const SupportB& supportB,
                             const Simplex& initialSimplex,
                             Vec3f& pA, Vec3f& pB, Vec3f& normal) noexcept {
    // Build initial polytope from GJK simplex.
    // We need at least a triangle (3 vertices) for EPA.  If the simplex has < 3 vertices,
    // expand it artificially.
    constexpr int kMaxVerts = 128;
    Vec3f polytope[kMaxVerts];
    int vertCount = 0;

    // Copy GJK simplex vertices.
    for (int i = 0; i < std::min(initialSimplex.count, 4); ++i) {
        polytope[vertCount++] = initialSimplex.vertices[i].p;
    }

    // If we have fewer than 3 vertices, expand along directions 
    // orthogonal to the current polytope to build a non-degenerate tetrahedron.
    while (vertCount < 3) {
        Vec3f dir;
        if (vertCount == 1) {
            // Single point: choose an axis not aligned with it.
            Vec3f p = polytope[0];
            float ax = std::abs(p.x), ay = std::abs(p.y), az = std::abs(p.z);
            if (ax <= ay && ax <= az) dir = Vec3f(1, 0, 0);
            else if (ay <= az)        dir = Vec3f(0, 1, 0);
            else                      dir = Vec3f(0, 0, 1);
        } else {
            // Two points: find two directions orthogonal to the segment.
            Vec3f seg = polytope[1] - polytope[0];
            float segLen = seg.length();
            if (segLen > 1e-12f) {
                seg = seg / segLen;
                // Find a reference axis not parallel to seg.
                Vec3f ref(
                    (std::abs(seg.x) < 0.9f) ? 1.0f : 0.0f,
                    (std::abs(seg.y) < 0.9f) ? 1.0f : 0.0f,
                    (std::abs(seg.z) < 0.9f) ? 1.0f : 0.0f
                );
                if (ref.lengthSquared() < 0.5f) ref = Vec3f(0, 0, 1);
                ref = ref / std::sqrt(ref.lengthSquared());
                dir = seg.cross(ref);
                float dLen = dir.length();
                if (dLen > 1e-12f) dir = dir / dLen;
                else              dir = Vec3f(1, 0, 0);
            } else {
                dir = Vec3f(1, 0, 0);
            }
        }
        Vec3f sA = supportA(dir);
        Vec3f sB = supportB(-dir);
        polytope[vertCount++] = sA - sB;
    }

    // Build initial faces (triangulation of a tetrahedron from 4 points, or triangle from 3).
    // For 3 vertices, we have 1 face (the triangle itself).
    // For 4 vertices, we have 4 faces.
    constexpr int kMaxFaces = 256;
    EpaFace faces[kMaxFaces];
    int faceCount = 0;

    if (vertCount == 3) {
        // Single triangle face — both sides.
        Vec3f nml; float dist;
        computeFace(polytope, 0, 1, 2, nml, dist);
        faces[faceCount++] = {{0, 1, 2}, nml, std::abs(dist)};
        // Add the opposite face.
        faces[faceCount++] = {{0, 2, 1}, -nml, std::abs(dist)};
    } else {
        // Tetrahedron: 4 faces.
        // Ensure the 4th vertex is not coplanar with the first 3.
        int tetra[4][3] = {{0,1,2}, {0,3,1}, {0,2,3}, {1,3,2}};
        for (int f = 0; f < 4; ++f) {
            Vec3f nml; float dist;
            computeFace(polytope, tetra[f][0], tetra[f][1], tetra[f][2], nml, dist);
            // Ensure normal faces outward (away from the 4th tetra vertex).
            int opposite = (f == 0) ? 3 : ((f == 1) ? 2 : ((f == 2) ? 1 : 0));
            Vec3f toOpposite = polytope[opposite] - polytope[tetra[f][0]];
            if (toOpposite.dot(nml) > 0) {
                nml = -nml;
            }
            faces[faceCount++] = {{tetra[f][0], tetra[f][1], tetra[f][2]}, nml, std::abs(dist)};
        }
    }

    if (faceCount == 0) {
        normal = Vec3f(0, 1, 0);
        pA = polytope[0];
        pB = Vec3f::zero();
        return 1.0f;
    }

    // EPA main loop.
    for (int iter = 0; iter < kEPAMaxIterations; ++iter) {
        // Find closest face to origin (smallest positive distance).
        int bestFace = -1;
        float bestDist = std::numeric_limits<float>::max();
        for (int f = 0; f < faceCount; ++f) {
            if (faces[f].distance < bestDist && faces[f].distance >= -kEPATolerance) {
                bestDist = faces[f].distance;
                bestFace = f;
            }
        }

        if (bestFace < 0) break;

        Vec3f searchDir = faces[bestFace].normal;

        // Get support point in search direction.
        Vec3f supA = supportA(searchDir);
        Vec3f supB = supportB(-searchDir);
        Vec3f newPt = supA - supB;

        float supportDist = searchDir.dot(newPt);

        // Check if the support point is significantly beyond the face plane.
        if (supportDist - bestDist < kEPATolerance) {
            // Converged: use current face as contact.
            normal = faces[bestFace].normal;

            // Compute contact points.
            int i0 = faces[bestFace].indices[0];
            int i1 = faces[bestFace].indices[1];
            int i2 = faces[bestFace].indices[2];

            // Closest point on the face to origin.
            Vec3f faceClosest;
            closestPtTriangleToOrigin(polytope[i0], polytope[i1], polytope[i2], faceClosest);

            // To find the world-space contact points, we need to know the A and B
            // contributions.  Since we only stored Minkowski points, we estimate using
            // the support function at the contact normal direction.
            pA = supportA(-normal);
            pB = supportB(normal);

            // Ensure normal points from A to B.
            float sign = normal.dot(pB - pA);
            if (sign < 0) normal = -normal;

            return bestDist;
        }

        // Add new vertex to polytope.
        if (vertCount >= kMaxVerts) break;
        int newIdx = vertCount++;
        polytope[newIdx] = newPt;

        // Find all faces visible from the new vertex (i.e., faces whose plane the new point
        // is on the positive side of, for outward normals).  Remove visible faces and
        // create new faces connecting the new vertex to each edge of the horizon.
        bool visible[kMaxFaces] = {false};
        int visibleCount = 0;
        for (int f = 0; f < faceCount; ++f) {
            Vec3f& fn = faces[f].normal;
            float side = fn.dot(newPt - polytope[faces[f].indices[0]]);
            // Use a small tolerance to avoid self-intersection.
            if (side > kEPATolerance) {
                visible[f] = true;
                visibleCount++;
            }
        }

        if (visibleCount == 0) {
            // New vertex is inside the polytope — should not happen in EPA.
            // Try a perturbation.
            continue;
        }

        if (visibleCount == faceCount) {
            // All faces visible — the new vertex is outside the entire polytope.
            // This can happen for very deep penetrations.  Keep the polytope simple.
            vertCount--; // Remove the vertex.
            continue;
        }

        // Build the horizon: edges that belong to exactly one visible face.
        struct Edge { int i0, i1; };
        constexpr int kMaxEdges = 256;
        Edge horizon[kMaxEdges];
        int horizonCount = 0;

        for (int f = 0; f < faceCount; ++f) {
            if (!visible[f]) continue;
            for (int e = 0; e < 3; ++e) {
                int ei0 = faces[f].indices[e];
                int ei1 = faces[f].indices[(e + 1) % 3];
                // Check if this edge is shared with another visible face.
                bool shared = false;
                for (int of = 0; of < faceCount; ++of) {
                    if (of == f || !visible[of]) continue;
                    for (int oe = 0; oe < 3; ++oe) {
                        int oei0 = faces[of].indices[oe];
                        int oei1 = faces[of].indices[(oe + 1) % 3];
                        if ((ei0 == oei0 && ei1 == oei1) || (ei0 == oei1 && ei1 == oei0)) {
                            shared = true;
                            break;
                        }
                    }
                    if (shared) break;
                }
                if (!shared) {
                    // Horizon edge.
                    if (horizonCount < kMaxEdges) {
                        horizon[horizonCount++] = {ei0, ei1};
                    }
                }
            }
        }

        // Remove visible faces.
        int writeIdx = 0;
        for (int f = 0; f < faceCount; ++f) {
            if (!visible[f]) {
                faces[writeIdx++] = faces[f];
            }
        }
        faceCount = writeIdx;

        // Create new faces connecting the new vertex to each horizon edge.
        for (int h = 0; h < horizonCount; ++h) {
            if (faceCount >= kMaxFaces) break;
            Vec3f nml; float dist;
            computeFace(polytope, newIdx, horizon[h].i0, horizon[h].i1, nml, dist);
            // Ensure normal points outward (away from the existing polytope center).
            faces[faceCount++] = {{newIdx, horizon[h].i0, horizon[h].i1}, nml, std::abs(dist)};
        }
    }

    // Fallback: return the best face found so far.
    float bestDist = std::numeric_limits<float>::max();
    int bestFace = 0;
    for (int f = 0; f < faceCount; ++f) {
        if (faces[f].distance < bestDist) {
            bestDist = faces[f].distance;
            bestFace = f;
        }
    }
    normal = faces[bestFace].normal;
    pA = supportA(-normal);
    pB = supportB(normal);
    float sign = normal.dot(pB - pA);
    if (sign < 0) normal = -normal;
    return bestDist;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  High-level GJK+EPA collision entry point
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Collide two convex shapes using GJK+EPA.
///
/// Supports any pair of convex primitives (Sphere, Box, Capsule, ConvexMesh) by routing
/// to the appropriate support functions.
///
/// @param shapeA   First shape.
/// @param posA     World position of shape A.
/// @param rotA     World rotation of shape A.
/// @param shapeB   Second shape.
/// @param posB     World position of shape B.
/// @param rotB     World rotation of shape B.
/// @param[out] manifold  Contact manifold populated if the shapes intersect.
/// @return  True if shapes intersect (manifold has at least one contact point).
inline bool collideConvexGJK(const Shape& shapeA, const Vec3f& posA, const Quaternionf& rotA,
                              const Shape& shapeB, const Vec3f& posB, const Quaternionf& rotB,
                              ContactManifold& manifold) noexcept {
    // Build support function lambdas for each shape type.
    // (Support functions are dispatched inline below in the `support` lambda.)

    // Need to use type-erased function wrappers since lambdas have different types.
    // We use a small function pointer approach (std::function would add a dependency).
    // Instead, encode the support function as a simple callback.

    // Use a simpler approach: dispatch directly per shape type pair.
    // This is less generic but avoids std::function / type erasure complexity.

    Vec3f pA, pB;
    Simplex simplex;

    // Run GJK.
    float distSq = 0;

    // Manual dispatch for GJK distance.
    // Since we know the shape types at compile time in each branch, but the function
    // is not templated, we use the indirect support functions above.

    // Actually, the simplest approach is to use `std::function`-like erasure.
    // We'll use a compact approach with function pointers and captured parameters.

    // For simplicity, implement GKL inline using direct switch-based support.
    // We'll just iterate the GJK loop manually with switch statements.

    // ── Inline GJK ──────────────────────────────────────────────────────────────────────────
    {
        Vec3f dir(1.0f, 0.0f, 0.0f);
        gjk_detail::Simplex localSimplex;
        localSimplex.count = 0;

        auto support = [&](const Vec3f& d, Vec3f& a, Vec3f& b) {
            switch (shapeA.type) {
            case ShapeType::Sphere:   a = gjk_detail::supportSphere(shapeA.sphere, posA, d); break;
            case ShapeType::Box:      a = gjk_detail::supportBox(shapeA.box, posA, rotA, d); break;
            case ShapeType::Capsule:  a = gjk_detail::supportCapsule(shapeA.capsule, posA, rotA, d); break;
            case ShapeType::ConvexMesh: a = gjk_detail::supportConvexMeshWorld(shapeA.convexMesh, posA, rotA, d); break;
            default: a = posA; break;
            }
            switch (shapeB.type) {
            case ShapeType::Sphere:   b = gjk_detail::supportSphere(shapeB.sphere, posB, -d); break;
            case ShapeType::Box:      b = gjk_detail::supportBox(shapeB.box, posB, rotB, -d); break;
            case ShapeType::Capsule:  b = gjk_detail::supportCapsule(shapeB.capsule, posB, rotB, -d); break;
            case ShapeType::ConvexMesh: b = gjk_detail::supportConvexMeshWorld(shapeB.convexMesh, posB, rotB, -d); break;
            default: b = posB; break;
            }
        };

        auto addPoint = [&](const Vec3f& pt, const Vec3f& ptA, const Vec3f& ptB) {
            gjk_detail::SimplexVertex sv;
            sv.p = pt; sv.pA = ptA; sv.pB = ptB;
            localSimplex.vertices[localSimplex.count] = sv;
            localSimplex.count++;
        };

        // First support — try multiple axes to avoid degeneracy.
        {
            Vec3f a, b;
            support(dir, a, b);
            Vec3f mink = a - b;
            // Try other axes if first support is near-origin.
            if (mink.lengthSquared() < 1e-12f) {
                dir = Vec3f(0.0f, 1.0f, 0.0f);
                support(dir, a, b);
                mink = a - b;
            }
            if (mink.lengthSquared() < 1e-12f) {
                dir = Vec3f(0.0f, 0.0f, 1.0f);
                support(dir, a, b);
                mink = a - b;
            }
            addPoint(mink, a, b);
            dir = -mink;
            float dl = dir.length();
            if (dl < 1e-12f) { distSq = 0; simplex = localSimplex; goto epaPhase; }
            dir = dir / dl;
        }

        Vec3f closest = localSimplex.vertices[0].p;

        for (int iter = 0; iter < gjk_detail::kGJKMaxIterations; ++iter) {
            Vec3f a, b;
            support(dir, a, b);
            Vec3f newPt = a - b;

            float dot = newPt.dot(dir);
            if (dot < closest.dot(dir) + gjk_detail::kGJKTolerance) break;

            addPoint(newPt, a, b);

            // Reduce simplex.
            Vec3f newClosest;
            int numDirs;
            gjk_detail::reduceSimplex(localSimplex, newClosest, numDirs);
            closest = newClosest;

            float cLenSq = closest.lengthSquared();
            if (cLenSq < gjk_detail::kGJKTolerance * gjk_detail::kGJKTolerance) {
                distSq = 0;
                simplex = localSimplex;
                goto epaPhase;
            }

            dir = -closest;
            float dirLen = std::sqrt(dir.lengthSquared());
            if (dirLen < 1e-12f) break;
            dir = dir / dirLen;
        }

        distSq = closest.lengthSquared();
        simplex = localSimplex;
    }

    if (distSq > 1e-6f) {
        // Shapes are separated — no collision.
        return false;
    }

epaPhase:
    // ── Shapes intersect — run EPA ──────────────────────────────────────────────────────
    {
        Vec3f localPA, localPB, normal;
        float depth = gjk_detail::epaPenetration(
            [&](const Vec3f& d) -> Vec3f {
                switch (shapeA.type) {
                case ShapeType::Sphere:   return gjk_detail::supportSphere(shapeA.sphere, posA, d);
                case ShapeType::Box:      return gjk_detail::supportBox(shapeA.box, posA, rotA, d);
                case ShapeType::Capsule:  return gjk_detail::supportCapsule(shapeA.capsule, posA, rotA, d);
                case ShapeType::ConvexMesh: return gjk_detail::supportConvexMeshWorld(shapeA.convexMesh, posA, rotA, d);
                default: return posA;
                }
            },
            [&](const Vec3f& d) -> Vec3f {
                switch (shapeB.type) {
                case ShapeType::Sphere:   return gjk_detail::supportSphere(shapeB.sphere, posB, d);
                case ShapeType::Box:      return gjk_detail::supportBox(shapeB.box, posB, rotB, d);
                case ShapeType::Capsule:  return gjk_detail::supportCapsule(shapeB.capsule, posB, rotB, d);
                case ShapeType::ConvexMesh: return gjk_detail::supportConvexMeshWorld(shapeB.convexMesh, posB, rotB, d);
                default: return posB;
                }
            },
            simplex, localPA, localPB, normal
        );

        // Build contact manifold.
        manifold.clearPoints();
        ContactPoint pt;
        pt.position    = (localPA + localPB) * 0.5f;
        pt.normal      = normal;
        pt.penetration = std::max(depth, 0.001f);
        pt.featureA    = -1;
        pt.featureB    = -1;
        manifold.addPoint(pt);
        return true;
    }
}

} // namespace gjk_detail

} // namespace ciphyxs
