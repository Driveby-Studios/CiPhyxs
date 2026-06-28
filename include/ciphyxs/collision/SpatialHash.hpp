//==================================================================================================
/// @file  SpatialHash.hpp
/// @brief  O(n) 3D uniform-grid broadphase using a spatial hash.
///
/// Replaces the brute-force O(n²) broadphase for scenes with >500 bodies.
/// Each frame the grid is rebuilt from scratch, then each body queries its
/// own cell + the 26 neighboring cells for potential collision pairs.
//==================================================================================================
#pragma once

#include "Contact.hpp"
#include "Shape.hpp"
#include "../core/RigidBody.hpp"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SpatialHashConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning parameters for the spatial-hash broadphase.
struct SpatialHashConfig {
    /// @brief  Size of each grid cell in world units.
    ///
    /// A good default is 2–4× the average body bounding radius.
    /// Smaller cells reduce pair count but increase hash-table overhead.
    float cellSize = 4.0f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────────────────────

namespace detail {

/// @brief  Integer cell coordinate in the 3D grid.
struct Cell {
    int x{}, y{}, z{};
    bool operator==(const Cell&) const noexcept = default;
};

struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
        // Large primes for good bit-mixing.
        std::size_t h = static_cast<std::size_t>(c.x) * 73856093;
        h ^= static_cast<std::size_t>(c.y) * 19349663;
        h ^= static_cast<std::size_t>(c.z) * 83492791;
        return h;
    }
};

/// @brief  World position → cell coordinate.
[[nodiscard]] inline Cell posToCell(const Vec3f& pos, float cellSize) noexcept {
    float inv = 1.0f / cellSize;
    return Cell{
        static_cast<int>(std::floor(pos.x * inv)),
        static_cast<int>(std::floor(pos.y * inv)),
        static_cast<int>(std::floor(pos.z * inv))
    };
}

/// @brief  Grid type: cell coordinate → list of body handles in that cell.
using Grid = std::unordered_map<Cell, std::vector<RigidBodyHandle>, CellHash>;

inline void buildGrid(Grid& grid,
                      const RigidBodyStorage& bodies,
                      float cellSize) noexcept {
    grid.clear();   // preserves capacity

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (!bodies.activeFlags[i]) continue;
        if (bodies.shapeCount[i] == 0) continue;  // no shapes

        Cell c = posToCell(bodies.positions[i], cellSize);
        grid[c].push_back(static_cast<RigidBodyHandle>(i));
    }
}

} // namespace detail

// ────────────────────────────────────────────────────────────────────────────────────────────────
// generateSpatialHashPairs
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Generate candidate collision pairs using a 3D spatial hash.
///
/// @param bodies        SoA rigid-body storage (position, active flags, motion type, compound shapes).
/// @param shapes        Flat shape registry (used for bounding-radius computation).
/// @param config        Cell-size and culling parameters.
/// @param[out] pairs    Cleared and filled with overlapping body pairs (no duplicates).
///
/// @note  This function is **stateless** — it rehashes the grid every call.
///        For determinism, resultant pairs are ordered by (bodyA, bodyB).
///
/// ## Compound shape support
///
/// The bounding radius is the **maximum** over all sub-shapes of
/// `(boundingRadius(subShape) + localOffset.length())`, ensuring the combined AABB
/// encloses every sub-shape.
inline void generateSpatialHashPairs(
    const RigidBodyStorage&   bodies,
    const std::vector<Shape>&       shapes,
    const SpatialHashConfig&  config,
    std::vector<BroadphasePair>&    pairs) noexcept {

    pairs.clear();
    std::size_t n = bodies.size();
    if (n < 2) return;

    // ── 1. Pre‑compute bounding radii for AABB culling (compound-aware) ────────────────
    struct Bounds { Vec3f center; float radius; };
    std::vector<Bounds> bounds(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!bodies.activeFlags[i]) continue;
        bounds[i].center = bodies.positions[i];
        bounds[i].radius = 0.0f;

        std::uint32_t start = bodies.shapeStart[i];
        std::uint32_t count = bodies.shapeCount[i];
        for (std::uint32_t s = 0; s < count; ++s) {
            ShapeHandle sh = bodies.flatShapeHandles[start + s];
            if (sh < shapes.size()) {
                float r = shapes[sh].boundingRadius();
                Vec3f localPos = bodies.flatShapeLocalPositions[start + s];
                float offset = localPos.length();
                bounds[i].radius = std::max(bounds[i].radius, r + offset);
            }
        }
    }

    // ── 2. Build the hash grid ──────────────────────────────────────────────────────────
    detail::Grid grid;                   // local (not static) for thread safety
    detail::buildGrid(grid, bodies, config.cellSize);

    // ── 3. Query neighbours for each active body ────────────────────────────────────────
    for (std::size_t i = 0; i < n; ++i) {
        if (!bodies.activeFlags[i]) continue;
        if (bodies.shapeCount[i] == 0) continue;  // no shapes

        bool isAStatic = (bodies.motionTypes[i] != MotionType::Dynamic);
        detail::Cell base = detail::posToCell(bodies.positions[i], config.cellSize);

        // Check 27-cell neighbourhood (self + 26 neighbours).
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    detail::Cell nbr{base.x + dx, base.y + dy, base.z + dz};
                    auto it = grid.find(nbr);
                    if (it == grid.end()) continue;

                    for (RigidBodyHandle j : it->second) {
                        // Process each unordered pair only once (i < j).
                        if (j <= i) continue;
                        if (!bodies.activeFlags[j]) continue;
                        if (bodies.shapeCount[j] == 0) continue;

                        // Skip static–static pairs.
                        if (isAStatic && bodies.motionTypes[j] != MotionType::Dynamic) continue;

                        // ── Bounding-sphere AABB cull ───────────────────────────────
                        Vec3f toCenter = bounds[j].center - bounds[i].center;
                        float distSq   = toCenter.lengthSquared();
                        float radSum   = bounds[i].radius + bounds[j].radius;
                        if (distSq > radSum * radSum) continue;

                        pairs.push_back({static_cast<RigidBodyHandle>(i), j});
                    }
                }
            }
        }
    }
}

} // namespace ciphyxs
