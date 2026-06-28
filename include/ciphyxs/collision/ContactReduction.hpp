//==================================================================================================
/// @file  ContactReduction.hpp
/// @brief  Coplanar contact normal reduction — merges nearly-identical normals from adjacent
///         triangles into a single plane manifold.
///
/// ## Problem
///
/// When a box rests on a flat triangulated surface, each triangle that the box touches produces
/// a separate `ContactManifold` via the narrowphase.  Even though the triangles are perfectly
/// coplanar, floating-point rounding and per-triangle basis vectors produce slightly different
/// contact normals (deviations of 0.5–2° are common).  The sequential-impulse solver sees these
/// as conflicting constraint directions and applies opposing impulses, causing **persistent
/// micro-bouncing** or "jitter".
///
/// ## Solution
///
/// After narrowphase, collect all manifolds for each (bodyA, bodyB) pair, cluster contact
/// points whose normals are within a tolerance (configurable cosine threshold), and replace
/// each cluster's points with a **single unified normal** (the unweighted average of the
/// cluster's normals).  This transforms, e.g., 4 manifolds × 4 points with slightly different
/// normals into 1 manifold × 4 points all sharing a single, geometrically correct normal.
///
/// ## Determinism
///
/// The algorithm is fully deterministic:
///   - Points are grouped by a (bodyA, bodyB) key and processed in sorted order.
///   - Clustering is greedy but always starts from the first unprocessed point.
///   - No hash maps, no random seeds, no platform-dependent floating-point.
//==================================================================================================
#pragma once

#include "Contact.hpp"
#include "../math/Vec3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ContactReductionConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning parameters for coplanar contact normal reduction.
///
/// The default threshold of 0.999 (≈2.5°) works well for triangle meshes with typical
/// floating-point variation.  Increase to 0.9999 for very coarse meshes; decrease for
/// intentionally faceted surfaces.
struct ContactReductionConfig {
    /// @brief  Cosine of the maximum angular deviation between normals considered coplanar.
    ///
    /// | Value  | Max angle | Use case                     |
    /// |--------|-----------|------------------------------|
    /// | 0.999  | ~2.5°     | Flat triangulated surfaces   |
    /// | 0.9999 | ~0.8°     | Coarse / low-poly meshes     |
    /// | 0.99   | ~8.1°     | Gentle curvature (risky)     |
    float normalCosineThreshold = 0.999f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Merge coplanar contact points across all manifolds sharing the same body pair.
///
/// For each (bodyA, bodyB) pair, contact points are clustered by normal direction similarity.
/// Each cluster emits one `ContactManifold` whose contact points all share the average normal
/// of the cluster.  The deepest-penetration points are retained (up to 4 per cluster).
///
/// Call this **once per frame**, after `detectCollisions()`, before the solver runs.
///
/// @param manifolds  In/out.  On return, may contain fewer manifolds with unified normals.
/// @param config     Cosine threshold for normal similarity.
///
/// @note  Warm-start impulse caches for merged points are **not preserved** across the merge.
///        This is acceptable because:
///          1. The unified normals allow the solver to converge in 2–3 iterations anyway.
///          2. The previous frame's warm-start cache (stored in `oldManifolds` during the
///             next frame's `detectCollisions()`) is matched by feature ID, which is preserved
///             on the individual contact points within each cluster.
inline void reduceContactNormals(
    std::vector<ContactManifold>& manifolds,
    const ContactReductionConfig& config = {}) noexcept {

    // Need at least 2 manifolds to merge anything.
    if (manifolds.size() < 2) return;

    // ─── Step 1: flatten all contact points into a sorted entry list ──────────────────────────
    //
    // Each entry carries a 64-bit key = (bodyA << 32) | bodyB, so all points for the same
    // body pair are contiguous after sorting.

    struct Entry {
        std::uint64_t key;
        ContactPoint  pt;
        float         combinedFriction;
        float         combinedRestitution;
    };

    std::vector<Entry> entries;
    entries.reserve(manifolds.size() * 4); // typical upper bound

    for (const auto& m : manifolds) {
        std::uint64_t key = (static_cast<std::uint64_t>(m.bodyA) << 32)
                            | static_cast<std::uint64_t>(m.bodyB);
        for (int pi = 0; pi < m.pointCount; ++pi) {
            entries.push_back({key, m.points[pi],
                               m.combinedFriction, m.combinedRestitution});
        }
    }

    // Stable sort keeps equal-key entries in insertion order (deterministic).
    std::stable_sort(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) noexcept { return a.key < b.key; });

    // ─── Step 2: greedy clustering by body-pair group ────────────────────────────────────────

    std::vector<ContactManifold> reduced;
    reduced.reserve(manifolds.size()); // upper bound

    std::size_t start = 0;
    while (start < entries.size()) {
        std::uint64_t key   = entries[start].key;
        std::size_t   end   = start;
        while (end < entries.size() && entries[end].key == key) ++end;

        RigidBodyHandle bodyA = static_cast<RigidBodyHandle>(key >> 32);
        RigidBodyHandle bodyB = static_cast<RigidBodyHandle>(key & 0xFFFFFFFF);

        // Borrow friction/restitution from the first entry of this group.
        float friction    = entries[start].combinedFriction;
        float restitution = entries[start].combinedRestitution;

        std::size_t groupSize = end - start;
        std::vector<bool> used(groupSize, false);

        // Greedy cluster formation.
        for (std::size_t i = 0; i < groupSize; ++i) {
            if (used[i]) continue;

            ContactManifold cluster;
            cluster.bodyA              = bodyA;
            cluster.bodyB              = bodyB;
            cluster.combinedFriction    = friction;
            cluster.combinedRestitution = restitution;
            cluster.pointCount         = 0;

            // Start a new cluster with point i.
            cluster.addPoint(entries[start + i].pt);
            used[i] = true;

            Vec3f normalSum = entries[start + i].pt.normal;
            int   normalCnt = 1;

            // Collect all remaining points with a similar normal.
            for (std::size_t j = i + 1; j < groupSize; ++j) {
                if (used[j]) continue;

                float dot = entries[start + i].pt.normal.dot(
                            entries[start + j].pt.normal);
                if (dot < config.normalCosineThreshold) continue;

                cluster.addPoint(entries[start + j].pt);
                normalSum += entries[start + j].pt.normal;
                ++normalCnt;
                used[j] = true;
            }

            // Compute the average normal for the cluster.
            Vec3f avgNormal = normalSum / static_cast<float>(normalCnt);
            float len = avgNormal.length();
            if (len > 1e-8f) {
                avgNormal = avgNormal / len;
            } else {
                avgNormal = entries[start + i].pt.normal; // fallback
            }

            // Apply the unified normal to all points in the cluster.
            for (int pi = 0; pi < cluster.pointCount; ++pi) {
                cluster.points[pi].normal = avgNormal;
            }

            reduced.push_back(std::move(cluster));
        }

        start = end;
    }

    // ─── Step 3: replace the original set if reduction was effective ─────────────────────────

    if (reduced.size() < manifolds.size()) {
        manifolds.swap(reduced);
    }
}

} // namespace ciphyxs
