//==================================================================================================
/// @file  Fracture.hpp
/// @brief  Destructible mesh system — Voronoi fracture + impulse-threshold shattering.
///
/// CiPhyxs destructibles use a **pre-fracture** model: at setup time a convex mesh is
/// divided into convex fragments via Voronoi seeds.  At runtime, when a contact impulse
/// or joint force exceeds the break threshold, the original body is replaced by its
/// fragments with matching velocities, conserv ing momentum.
///
/// ## Pipeline
///
/// ```
/// 1. Pre-fracture (setup)
///    VoronoiFracture::generateSeeds(mesh, count) → seed points
///    VoronoiFracture::fragment(mesh, seeds)      → Fragment[]
///
/// 2. Per-step monitoring  (integrated into PhysicsWorld)
///    After constraint solve:
///      for each contact:   check |impulse| > threshold → mark broken
///      for each joint:     check accumulated impulse  → mark broken
///
/// 3. Fracture execution
///    Remove original body, add fragment bodies with:
///      - position  = original position + localFragmentCenter
///      - velocity  = original linear  + angular × offset
///      - rotation  = original rotation
/// ```
///
/// ## Numerics
///
/// The Voronoi clipping uses half-space intersection against the original convex hull
/// vertices.  Each fragment is guaranteed convex because the intersection of convex
/// sets (original hull + half-space) is convex.  Fragment inertia is approximated using
/// the AABB-based formula from `Shape::computeInertia()`.
///
/// @see  ConvexMesh  — The primitive that can be fractured.
/// @see  PhysicsWorld — Orchestrates fracture monitoring and execution.
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Mat3.hpp"
#include "../core/RigidBody.hpp"
#include "Shape.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Fragment
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A single convex fragment produced by fracturing a ConvexMesh.
///
/// Each fragment stores its own geometry, centre of mass, mass, and local inertia
/// so it can be spawned as an independent rigid body at runtime.
struct Fragment {
    ConvexMesh mesh;     ///< Convex hull geometry (local space).
    Vec3f      center;   ///< Centre of mass relative to the original body's origin.
    float      mass;     ///< Mass of this fragment (> 0).
    Vec3f      inertia;  ///< Diagonal of the local inertia tensor.
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// DestructibleConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Per-body configuration for destructible behaviour.
struct DestructibleConfig {
    /// @brief  Minimum contact impulse (N·s) that triggers fracture.
    ///         Typical range: 10 (glass) … 1000 (concrete).
    float breakImpulseThreshold = 100.0f;

    /// @brief  Number of Voronoi fragments to generate on fracture.
    ///         More fragments = finer debris but higher runtime cost.
    int numFragments = 6;

    /// @brief  Speed scale applied to fragments on separation [0, ∞).
    ///         1.0 = inherit parent velocity; >1 = explosive.
    float fragmentSpeedScale = 1.0f;

    /// @brief  If true, fragments can themselves fracture again on further impacts.
    bool enableChainFracture = false;

    /// @brief  If true, the body is replaced by fragments when broken.
    ///         If false, the body is simply destroyed.
    bool spawnFragmentsOnBreak = true;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// DestructibleData
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Runtime state for a single destructible body.
///
/// Stored in a parallel vector inside PhysicsWorld, keyed by RigidBodyHandle.
struct DestructibleData {
    DestructibleConfig            config;
    std::vector<Fragment>         fragments;   ///< Pre-computed convex fragments.
    bool                          isBroken = false;
    float                         maxImpulseThisStep = 0.0f; ///< Reset each fixed step.

    /// @brief  Reset the per-step impulse tracker.
    void resetImpulse() noexcept { maxImpulseThisStep = 0.0f; }

    /// @brief  Record an impulse magnitude observed this step.
    void recordImpulse(float impulse) noexcept {
        if (impulse > maxImpulseThisStep) maxImpulseThisStep = impulse;
    }

    /// @brief  Check whether the break threshold has been exceeded this step.
    [[nodiscard]] bool shouldBreak() const noexcept {
        return !isBroken && maxImpulseThisStep >= config.breakImpulseThreshold;
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// VoronoiFracture
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Static utility for generating Voronoi fracture patterns on ConvexMesh objects.
///
/// ## Fracture Algorithm
///
/// 1. Generate `numSeeds` seed points within the convex hull's bounding box.
/// 2. For each seed, the Voronoi cell is the intersection of half-spaces defined by
///    the perpendicular bisectors between that seed and all other seeds.
/// 3. Each cell is intersected with the original convex hull via plane clipping
///    (Sutherland–Hodgman in 3D) to produce a convex fragment.
/// 4. Mass properties are computed for each fragment based on its bounding box.
///
/// @note  The vertex data for the generated Fragment meshes is heap-allocated and
///        stored inside `Fragment::mesh`.  The caller (PhysicsWorld) is responsible
///        for keeping it alive as long as the fragments exist.
struct VoronoiFracture {
    /// @brief  Generate random seed points within the bounding volume of a ConvexMesh.
    ///
    /// Seeds are placed within the AABB of the mesh, slightly perturbed to avoid
    /// degenerate (coincident) seeds.
    ///
    /// @param mesh    The convex mesh to fracture.
    /// @param count   Number of seed points to generate.
    /// @return  A vector of local-space seed points.
    static std::vector<Vec3f> generateSeeds(const ConvexMesh& mesh, int count) noexcept {
        std::vector<Vec3f> seeds;
        seeds.reserve(static_cast<std::size_t>(count));

        Vec3f he = mesh.halfExtents;
        Vec3f c  = mesh.center;

        // Simple deterministic LCG for reproducibility.
        std::uint32_t state = 42;

        for (int i = 0; i < count; ++i) {
            // Random point in [-1, 1]³, scaled to bounding box.
            float rx = (static_cast<float>(randFraction(state)) * 2.0f - 1.0f) * he.x * 0.8f;
            float ry = (static_cast<float>(randFraction(state)) * 2.0f - 1.0f) * he.y * 0.8f;
            float rz = (static_cast<float>(randFraction(state)) * 2.0f - 1.0f) * he.z * 0.8f;
            seeds.emplace_back(c.x + rx, c.y + ry, c.z + rz);
        }

        return seeds;
    }

    /// @brief  Fracture a ConvexMesh into fragments using Voronoi seeds.
    ///
    /// Each fragment is the intersection of a Voronoi cell with the original convex hull.
    /// The fragments are guaranteed convex and their collective volume approximates the
    /// original hull's volume (some material is lost at the fracture boundaries, which
    /// is acceptable for visual destruction).
    ///
    /// @param mesh    The convex mesh to fracture.
    /// @param seeds   Voronoi seed points (e.g. from `generateSeeds()`).
    /// @param totalMass  Total mass of the original body, distributed among fragments.
    /// @return  A vector of Fragments, each with mesh, center, mass, and inertia.
    static std::vector<Fragment> fragment(const ConvexMesh& mesh,
                                          const std::vector<Vec3f>& seeds,
                                          float totalMass) noexcept {
        std::size_t numSeeds = seeds.size();
        if (numSeeds == 0) return {};
        if (numSeeds == 1) {
            // Single fragment = the original mesh itself.
            Fragment f;
            f.mesh    = mesh;
            f.center  = mesh.center;
            f.mass    = totalMass;
            f.inertia = computeBoxInertia(mesh.halfExtents, totalMass);
            return {f};
        }

        // Pre-compute clipping planes for each Voronoi cell.
        // For seed i, the cell is the half-space on i's side of each
        // plane passing through midpoint(i,j) with normal (seeds[j] - seeds[i]).
        struct ClipPlane {
            Vec3f normal;     ///< Normal points toward the cell interior.
            float distance;   ///< Signed distance from origin (n·p = d).
        };

        std::vector<std::vector<ClipPlane>> cellPlanes(numSeeds);
        for (std::size_t i = 0; i < numSeeds; ++i) {
            for (std::size_t j = 0; j < numSeeds; ++j) {
                if (i == j) continue;
                Vec3f mid = (seeds[i] + seeds[j]) * 0.5f;
                Vec3f normal = seeds[j] - seeds[i];
                float len = normal.length();
                if (len < 1e-12f) continue;
                normal = normal / len;
                // Plane: n · (p - mid) = 0  →  n·p = n·mid
                float d = normal.dot(mid);
                cellPlanes[i].push_back({normal, d});
            }
        }

        // First pass: clip all Voronoi cells, collect non-degenerate fragments.
        // We store bounding-box info per fragment to compute mass fractions.
        struct CellResult {
            std::vector<Vec3f> verts;
            Vec3f halfExtents;
            Vec3f center;
        };
        std::vector<CellResult> cells;
        cells.reserve(numSeeds);

        for (std::size_t i = 0; i < numSeeds; ++i) {
            // Start with all vertices of the original mesh.
            std::vector<Vec3f> verts(mesh.vertices, mesh.vertices + mesh.vertexCount);

            // Clip against each Voronoi plane.
            bool empty = false;
            for (const auto& plane : cellPlanes[i]) {
                if (verts.empty()) { empty = true; break; }
                verts = clipPolyhedron(verts, plane.normal, plane.distance);
            }

            if (empty || verts.size() < 4) {
                // Degenerate fragment — skip.
                continue;
            }

            // Compute bounding box and centre.
            Vec3f minV = verts[0], maxV = verts[0];
            Vec3f sumV = Vec3f::zero();
            for (const auto& v : verts) {
                minV.x = std::min(minV.x, v.x);
                minV.y = std::min(minV.y, v.y);
                minV.z = std::min(minV.z, v.z);
                maxV.x = std::max(maxV.x, v.x);
                maxV.y = std::max(maxV.y, v.y);
                maxV.z = std::max(maxV.z, v.z);
                sumV   = sumV + v;
            }

            Vec3f fragCenter = sumV / static_cast<float>(verts.size());
            Vec3f halfExtents = (maxV - minV) * 0.5f;
            cells.push_back({std::move(verts), halfExtents, fragCenter});
        }

        if (cells.empty()) return {};

        // Compute mass fractions from bounding-box volume, then normalize
        // so the fractions sum to 1.0.  This avoids double-counting from
        // overlapping bounding boxes (which is inherent in the box approximation).
        Vec3f origHe = mesh.halfExtents;
        float origVol = (origHe.x * 2.0f) * (origHe.y * 2.0f) * (origHe.z * 2.0f);

        std::vector<float> massFracs(cells.size(), 1.0f / static_cast<float>(cells.size()));
        float totalFrac = 0.0f;

        if (origVol > 0.0f) {
            for (std::size_t i = 0; i < cells.size(); ++i) {
                const auto& c = cells[i];
                float fragVol = (c.halfExtents.x * 2.0f)
                              * (c.halfExtents.y * 2.0f)
                              * (c.halfExtents.z * 2.0f);
                massFracs[i] = fragVol / origVol;
                totalFrac   += massFracs[i];
            }
        }

        // Build final Fragment objects with proper ownership.
        std::vector<Fragment> result;
        result.reserve(cells.size());

        for (std::size_t i = 0; i < cells.size(); ++i) {
            auto& c = cells[i];

            float fragMass = (totalFrac > 0.0f)
                ? totalMass * massFracs[i] / totalFrac
                : totalMass / static_cast<float>(cells.size());
            Vec3f fragInertia = computeBoxInertia(c.halfExtents, fragMass);

            // Allocate and copy the fragment vertices (owned by the Fragment).
            auto* vertCopy = new Vec3f[c.verts.size()];
            for (std::size_t vi = 0; vi < c.verts.size(); ++vi) {
                vertCopy[vi] = c.verts[vi];
            }

            ConvexMesh fragMesh;
            fragMesh.vertices    = vertCopy;
            fragMesh.vertexCount = static_cast<int>(c.verts.size());
            fragMesh.halfExtents = c.halfExtents;
            fragMesh.center      = c.center;

            result.push_back({fragMesh, c.center, fragMass, fragInertia});
        }

        return result;
    }

private:
    // ─── Simple LCG for reproducible randomness ───────────────────────────────────────────

    /// @brief  Generate a pseudo-random float in [0, 1) using a simple LCG.
    static float randFraction(std::uint32_t& state) noexcept {
        state = state * 1103515245u + 12345u;
        return static_cast<float>(state & 0x7FFFFFFFu) / 2147483648.0f;
    }

    // ─── 3D half-space clipping (Sutherland–Hodgman) ──────────────────────────────────────

    /// @brief  Clip a convex polyhedron against a half-space defined by n·p <= d.
    ///
    /// The normal `n` points toward the kept half-space.  `d` is the signed distance
    /// from the origin to the plane along `n` (i.e. n·p = d on the plane).
    ///
    /// @param verts    Input convex vertices (modified in-place).
    /// @param n        Plane normal (kept side: n·p <= d).
    /// @param d        Plane distance.
    /// @return The clipped vertex set (may be empty if the polyhedron is entirely clipped).
    static std::vector<Vec3f> clipPolyhedron(const std::vector<Vec3f>& verts,
                                              const Vec3f& n, float d) noexcept {
        if (verts.empty()) return {};

        std::vector<Vec3f> output;
        output.reserve(verts.size() * 2);

        std::size_t count = verts.size();
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t j = (i + 1) % count;

            const Vec3f& p = verts[i];
            const Vec3f& q = verts[j];

            float distP = n.dot(p) - d;
            float distQ = n.dot(q) - d;

            bool insideP = distP <= 0.0f;
            bool insideQ = distQ <= 0.0f;

            if (insideP) {
                output.push_back(p);
            }

            if (insideP != insideQ) {
                // Edge crosses the clipping plane — compute intersection.
                float t = distP / (distP - distQ);
                Vec3f intersection = p + (q - p) * t;
                output.push_back(intersection);
            }
        }

        return output;
    }

    // ─── Box-approximated inertia ─────────────────────────────────────────────────────────

    /// @brief  Compute the diagonal inertia of a solid box with given half-extents and mass.
    static Vec3f computeBoxInertia(const Vec3f& halfExtents, float mass) noexcept {
        float ix = (1.0f / 3.0f) * mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z);
        float iy = (1.0f / 3.0f) * mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z);
        float iz = (1.0f / 3.0f) * mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y);
        float eps = 1e-8f;
        return Vec3f(std::max(ix, eps), std::max(iy, eps), std::max(iz, eps));
    }
};

} // namespace ciphyxs
