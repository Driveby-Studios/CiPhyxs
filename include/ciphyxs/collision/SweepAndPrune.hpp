//==================================================================================================
/// @file  SweepAndPrune.hpp
/// @brief  O(n log n) sweep-and-prune broadphase with AABB overlap culling.
///
/// ## Algorithm
///
///   1. Compute the world-space AABB for every active body (compound-aware).
///   2. Create two events per body: **start** (min.X) and **end** (max.X).
///   3. Sort by coordinate using a **stable sort** (deterministic).
///   4. Sweep left to right: maintain an "active" set of bodies whose start has
///      been seen but whose end has not.  When a start event is encountered, test
///      the new body against all currently active bodies using full 3D AABB overlap.
///      When an end event is encountered, remove the body from the active set.
///
/// ## Why Sweep-and-Prune?
///
/// | Algorithm     | Typical complexity | Best for          | Deterministic |
/// |---------------|-------------------|--------------------|---------------|
/// | Brute-force   | O(n²)             | < 500 bodies       | Yes           |
/// | Sweep-and-Prune | O(n log n + k)  | 100–2000 bodies    | Yes           |
/// | Spatial hash  | O(n) average      | > 1000 bodies      | Depends on hash |
///
/// SAP is **fully deterministic** (stable sort + fixed sweep order), making it ideal
/// for scenes where broadphase determinism is required without the overhead of a
/// spatial hash.
///
/// ## Temporal Coherence
///
/// When bodies move slowly between frames, the sorted event list changes very little.
/// A future optimisation could reuse the previous frame's sorted list and re-sort
/// only a small fraction (insertion-sort style).  Currently we re-sort every frame
/// for simplicity.
//==================================================================================================
#pragma once

#include "Contact.hpp"
#include "Shape.hpp"
#include "../core/RigidBody.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SweepAndPruneConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning for the sweep-and-prune broadphase.
///
/// Currently empty — SAP works well out of the box.  Future tuning options may include
/// the sweep axis, temporal-coherence reuse, or multi-sweep (Y/Z) fallback.
struct SweepAndPruneConfig {
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────────────────────

namespace detail {

/// @brief  Compute the world-space AABB for a body, accounting for all compound sub-shapes.
///         Returns an empty (inverted) AABB if the body has no valid shapes.
[[nodiscard]] inline AABB computeBodyAABB(
    RigidBodyHandle handle,
    const RigidBodyStorage& bodies,
    const std::vector<Shape>& shapes) noexcept {

    std::uint32_t start = bodies.shapeStart[handle];
    std::uint32_t count = bodies.shapeCount[handle];

    AABB bodyAABB;
    bool first = true;

    for (std::uint32_t s = 0; s < count; ++s) {
        ShapeHandle sh = bodies.flatShapeHandles[start + s];
        if (sh >= shapes.size()) continue;

        // World-space sub-shape transform.
        Vec3f localPos = bodies.flatShapeLocalPositions[start + s];
        Quaternionf localRot = bodies.flatShapeLocalRotations[start + s];
        Vec3f worldPos = bodies.positions[handle]
                        + bodies.rotations[handle].rotate(localPos);
        Quaternionf worldRot = bodies.rotations[handle] * localRot;

        AABB subAABB = shapes[sh].getAABB(worldPos, worldRot);
        if (first) {
            bodyAABB = subAABB;
            first = false;
        } else {
            bodyAABB = bodyAABB.united(subAABB);
        }
    }

    return bodyAABB; // inverted sentinel if no valid shapes
}

} // namespace detail

/// @brief  Generate candidate broad-phase pairs using sweep-and-prune on the X axis.
///
/// @param bodies   SoA rigid-body storage.
/// @param shapes   Flat shape registry.
/// @param config   (unused) Reserved for future tuning.
/// @param[out] pairs  Filled with overlapping body pairs (deterministic order).
///
/// ## Determinism
///
/// This function produces **bit-identical results** across runs and platforms
/// because:
///   - Stable sort orders equal-coordinate events by insertion order.
///   - Active-set iteration order is insertion order (ascending body handle).
///   - Pairs are emitted as `(minHandle, maxHandle)`.
///   - Static-static pairs are skipped.
///
/// ## Compound shape support
///
/// Each body's AABB is the union of all its sub-shape AABBs, correctly accounting
/// for local transforms.  See `detail::computeBodyAABB`.
inline void generateSweepAndPrunePairs(
    const RigidBodyStorage& bodies,
    const std::vector<Shape>& shapes,
    const SweepAndPruneConfig& /*config*/,
    std::vector<BroadphasePair>& pairs) noexcept {

    pairs.clear();
    std::size_t n = bodies.size();
    if (n < 2) return;

    // ─── 1. Compute AABBs and create sweep events ───────────────────────────────────────────
    //
    // Each active body generates two events:
    //   (min.X, body, isStart=true)
    //   (max.X, body, isStart=false)

    struct Event {
        float           coord;
        RigidBodyHandle body;
        bool            isStart; // true = entering, false = leaving
    };

    // Pre-allocate worst-case: every body active.
    std::vector<Event> events;
    events.reserve(n * 2);
    std::vector<AABB> aabbs(n);

    for (std::size_t i = 0; i < n; ++i) {
        if (!bodies.activeFlags[i]) continue;
        if (bodies.shapeCount[i] == 0) continue;

        RigidBodyHandle h = static_cast<RigidBodyHandle>(i);
        AABB box = detail::computeBodyAABB(h, bodies, shapes);

        // Skip bodies with no valid sub-shapes (inverted sentinel AABB).
        if (box.min.x > box.max.x) continue;

        aabbs[h] = box;
        events.push_back({box.min.x, h, true});
        events.push_back({box.max.x, h, false});
    }

    // ─── 2. Stable sort — fully deterministic ────────────────────────────────────────────────
    //
    // Stable sort guarantees that events with equal coordinates retain their
    // insertion order.  For start events with the same X, the body with the
    // smaller handle enters first.  For end events with the same X, the body
    // with the larger handle leaves first (start events for a given body are
    // always pushed before its end event, so the relative order among different
    // bodies at the same coordinate is: starts ascending, ends descending).

    // Stable sort — for equal coordinates, start events MUST sort before end events.
    // Otherwise, touching AABBs would have their end processed before the neighbour's
    // start, causing those pairs to be silently dropped.
    std::stable_sort(events.begin(), events.end(),
        [](const Event& a, const Event& b) noexcept {
            if (a.coord != b.coord) return a.coord < b.coord;
            // Start events (isStart=true) before end events at the same coordinate.
            return a.isStart > b.isStart;
        });

    // ─── 3. Sweep ────────────────────────────────────────────────────────────────────────────
    //
    // Active set: bodies whose start has been seen but end has not.
    // Using a simple vector; the active set stays small (average overlap count).

    std::vector<RigidBodyHandle> active;
    active.reserve(64);

    for (const auto& ev : events) {
        RigidBodyHandle h = ev.body;

        if (ev.isStart) {
            // Test the new body against all currently active bodies.
            MotionType mtH = bodies.motionTypes[h];

            for (RigidBodyHandle a : active) {
                // Skip static-static pairs (neither is dynamic).
                if (mtH != MotionType::Dynamic &&
                    bodies.motionTypes[a] != MotionType::Dynamic) continue;

                // Full 3D AABB overlap test.
                if (aabbs[h].overlaps(aabbs[a])) {
                    // Emit pair in canonical order: smaller handle first.
                    if (h < a) {
                        pairs.push_back({h, a});
                    } else {
                        pairs.push_back({a, h});
                    }
                }
            }

            active.push_back(h);
        } else {
            // Remove the body from the active set.
            // Linear search is fine — active set size is typically < 20.
            auto it = std::find(active.begin(), active.end(), h);
            if (it != active.end()) {
                // Swap with last (fast) — order doesn't matter for the active set.
                *it = active.back();
                active.pop_back();
            }
        }
    }
}

} // namespace ciphyxs
