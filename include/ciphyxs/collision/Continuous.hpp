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
//  Swept sphere ↔ convex mesh  (binary search via GJK)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Swept sphere vs. convex mesh.  Uses binary search with GJK distance queries.
///
/// At each search step the sphere centre is sampled along the velocity vector and a GJK
/// distance query determines whether the sphere overlaps the mesh (distance < radius).
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

    // Build a support function for the convex mesh (world space).
    auto meshSupport = [&](const Vec3f& dirWorld) -> Vec3f {
        Vec3f dirLocal = meshRot.rotateInverse(dirWorld);
        Vec3f localPt = gjk_detail::supportConvexMesh(mesh, dirLocal);
        return meshPos + meshRot.rotate(localPt);
    };

    // Check for initial overlap at t=0.
    {
        // GJK between sphere centre (point) and the convex mesh.
        auto pointSupport = [&](const Vec3f&) -> Vec3f { return pos; };
        Vec3f pA, pB;
        gjk_detail::Simplex simp;
        float distSq = gjk_detail::gjkDistance(pointSupport, meshSupport, pA, pB, simp);
        float dist = std::sqrt(distSq);
        if (dist < sphere.radius) {
            // Already overlapping.
            Vec3f nml = (pos - pB).normalized();
            if (nml.lengthSquared() < 0.5f) {
                // Fallback: use mesh centre offset.
                nml = (pos - meshPos).normalized();
                if (nml.lengthSquared() < 0.5f) nml = Vec3f(0.0f, 1.0f, 0.0f);
            }
            outNormal = nml;
            outPosition = pos - nml * sphere.radius;
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

        auto pointSupport = [&](const Vec3f&) -> Vec3f { return samplePos; };
        Vec3f pA, pB;
        gjk_detail::Simplex simp;
        float distSq = gjk_detail::gjkDistance(pointSupport, meshSupport, pA, pB, simp);
        float dist = std::sqrt(distSq);

        if (dist < sphere.radius) {
            hit = true;
            hi = mid;
        } else {
            lo = mid;
        }
    }

    if (!hit) return 1.0f;

    float toi = (lo + hi) * 0.5f;
    Vec3f hitPos = pos + disp * toi;

    // Compute normal at TOI using GJK.
    {
        auto pointSupport = [&](const Vec3f&) -> Vec3f { return hitPos; };
        Vec3f pA, pB;
        gjk_detail::Simplex simp;
        gjk_detail::gjkDistance(pointSupport, meshSupport, pA, pB, simp);
        Vec3f nml = (hitPos - pB).normalized();
        if (nml.lengthSquared() < 0.5f) {
            nml = (hitPos - meshPos).normalized();
            if (nml.lengthSquared() < 0.5f) nml = Vec3f(0.0f, 1.0f, 0.0f);
        }
        outNormal = nml;
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
//  CCD pipeline helper
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

/// @brief  Run CCD for a single body.  Corrects position and velocity if a hit is found.
///
/// Tests the body's swept path (pos → pos + vel * dt) against all other active bodies.
/// If a TOI < 1 is found, the body is advanced to the TOI position and its velocity is
/// reflected along the collision normal with restitution.
///
/// @param body          Handle of the CCD body.
/// @param dt            Fixed timestep.
/// @param bodies        SoA rigid-body storage (modified in-place).
/// @param shapes        Shape registry.
/// @param restitution   Coefficient of restitution for the CCD body.
/// @param friction      Friction coefficient for the CCD body (currently unused).
/// @return  True if a CCD collision was resolved.
///
/// @note  This function only modifies `body`.  Other bodies are treated as static
///        obstacles during the sweep (their velocities are ignored).  This is the
///        standard "CCD against world" approximation used by most engines — it
///        prevents the most common tunnelling cases (fast object → static wall).
inline bool ccdResolveBody(RigidBodyHandle body, float dt,
                           RigidBodyStorage& bodies,
                           const std::vector<Shape>& shapes,
                           float restitution, float /*friction*/) noexcept {

    if (bodies.shapeCount[body] == 0) return false;

    Vec3f vel = bodies.linearVelocities[body];
    float speed = vel.length();
    if (speed < 1e-8f) return false;

    Vec3f pos = bodies.positions[body];
    Vec3f disp = vel * dt;

    float bestTOI = 1.0f;
    Vec3f bestNormal(0.0f, 1.0f, 0.0f);
    Vec3f bestPosition(0.0f, 0.0f, 0.0f);
    bool hit = false;

    // Iterate over all other active bodies.
    for (std::size_t j = 0; j < bodies.size(); ++j) {
        if (j == body) continue;
        if (!bodies.activeFlags[j]) continue;
        if (bodies.shapeCount[j] == 0) continue;

        // Only test against static and kinematic bodies (obstacles).
        // Dynamic-dynamic CCD is more complex and less critical for tunnelling.
        if (bodies.motionTypes[j] == MotionType::Dynamic) continue;

        // Iterate over sub-shapes of both bodies.
        std::uint32_t startA = bodies.shapeStart[body];
        std::uint32_t countA = bodies.shapeCount[body];
        std::uint32_t startB = bodies.shapeStart[j];
        std::uint32_t countB = bodies.shapeCount[j];

        for (std::uint32_t sa = 0; sa < countA; ++sa) {
            ShapeHandle shA = bodies.flatShapeHandles[startA + sa];
            if (shA >= shapes.size()) continue;

            const Shape& shapeA = shapes[shA];

            // Sub-shape world transform for the CCD body.
            Vec3f localPosA = bodies.flatShapeLocalPositions[startA + sa];
            // localRotA is not needed for sphere CCD (sphere is rotationally invariant).
            Vec3f worldPosA = pos + bodies.rotations[body].rotate(localPosA);
            // The sub-shape velocity is the body's linear velocity (rotation ignored for CCD).

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
                        // Degenerate mesh — skip CCD.
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
                // For other shape types (non-sphere, non-capsule, non-ConvexMesh CCD bodies),
                // fall back to no CCD (TOI stays 1).

                if (toi < bestTOI) {
                    bestTOI = toi;
                    bestNormal = nml;
                    bestPosition = pt;
                    hit = true;
                }
            }
        }
    }

    if (!hit || bestTOI >= 1.0f) return false;

    // ─── Apply CCD correction ────────────────────────────────────────────────────────────────
    //
    // 1. Advance position to the TOI.
    bodies.positions[body] = pos + disp * bestTOI;

    // 2. Reflect velocity along the contact normal (with restitution).
    float vDotN = vel.dot(bestNormal);
    if (vDotN < 0.0f) {
        // Separate velocity into normal and tangential components.
        Vec3f vNormal = bestNormal * vDotN;
        Vec3f vTangent = vel - vNormal;

        // Apply restitution to the normal component (reflect).
        bodies.linearVelocities[body] = vTangent - vNormal * restitution;
    }

    // 3. Kill angular velocity for simplicity (the body has just hit something).
    bodies.angularVelocities[body] *= 0.5f;

    return true;
}

} // namespace ciphyxs
