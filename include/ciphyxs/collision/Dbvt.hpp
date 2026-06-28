//==================================================================================================
/// @file  Dbvt.hpp
/// @brief  Dynamic Bounding Volume Tree broadphase — incremental AABB tree with refit.
///
/// ## Overview
///
/// The Dbvt (Dynamic Bounding Volume Tree) is a **binary AABB tree** that supports
/// incremental insertion, removal, and refit.  Unlike the brute-force O(n²) or
/// sweep-and-prune O(n log n) broadphases, the Dbvt maintains a hierarchical tree
/// across frames and only updates AABBs (refit) for bodies that have moved.
///
/// ## Algorithm
///
///   1. Each active body gets a **leaf node** holding its fat AABB (padded by a
///      configurable margin).  Internal nodes store the union of their children's AABBs.
///   2. When a body moves, its leaf AABB is checked against the body's new world‑space
///      AABB.  If the new AABB fits inside the fat AABB, the tree is **not** modified
///      (temporal coherence).  Otherwise the leaf is removed and re‑inserted.
///   3. Insertion uses a **surface‑area heuristic** (SAH) descent to find the best
///      sibling, minimising the cost of future queries.
///   4. Pair generation walks the tree depth‑first, testing internal‑node AABBs for
///      overlap and recursing into children.  Output pairs are sorted for determinism.
///   5. Bodies that become inactive or are removed are cleaned up lazily — their leaves
///      stay in the tree until `sync()` is called, at which point they are evicted.
///
/// ## Complexity
///
/// | Operation      | Complexity     | Notes                              |
/// |----------------|----------------|------------------------------------|
/// | Insert         | O(log n)       | SAH descent to find sibling        |
/// | Remove         | O(log n)       | Collapse parent, refit up the tree |
/// | Update (refit) | O(log n)       | Only when fat AABB is exceeded     |
/// | Pair query     | O(n log n + k) | n = bodies, k = overlapping pairs  |
/// | Overlap query  | O(log n + m)   | m = overlapping bodies             |
///
/// ## Temporal Coherence
///
/// The fat‑AABB margin (default 5 % of each extent) means that a slowly‑moving body
/// may not trigger a tree modification for many frames.  This makes the Dbvt ideal
/// for scenes with high frame‑to‑frame coherence (most mechanical/industrial simulations).
///
/// ## Determinism
///
/// The tree walk order is deterministic (fixed stack discipline), and the output
/// pairs are sorted by (bodyA, bodyB) and deduplicated, guaranteeing **bit‑identical**
/// results across runs on the same platform.
//==================================================================================================
#pragma once

#include "Contact.hpp"
#include "Shape.hpp"
#include "../core/RigidBody.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// DbvtConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning parameters for the Dynamic BVH broadphase.
///
/// The fat‑AABB margin is the primary control for temporal coherence vs. tree quality.
/// A larger margin reduces tree updates but increases the number of false‑positive
/// overlap tests during pair generation.
struct DbvtConfig {
    /// @brief  Padding applied to each axis of the body AABB, as a fraction of the
    ///         axis extent.
    ///
    /// A value of 0.05 = 5 % means a 1 m wide box gets ±0.05 m fatness on each side.
    /// Minimum padding of 0.01 m is enforced to keep degenerately thin AABBs stable.
    float fatAABBScale = 0.05f;

    /// @brief  Number of body removals before the tree is fully rebuilt from scratch.
    ///
    /// A full rebuild produces an optimal tree and prunes all stale nodes.
    int rebuildThreshold = 64;

    /// @brief  Number of new body insertions before a full rebuild is forced.
    ///
    /// When many bodies are added incrementally, the tree becomes pathologically
    /// unbalanced because insertLeaf() attaches new leaves at the SAH-optimal
    /// location but never restructures the tree globally.  A threshold of
    /// ~10-20% of current body count prevents degenerate trees in scenes
    /// with many spawns (fracture, debris, crowds).
    ///
    /// Set to 0 to disable insertion-triggered rebuild (default: 10% growth).
    /// The value is interpreted as a fraction of current body count when < 1,
    /// or as an absolute count when >= 1.
    float insertionGrowthThreshold = 0.10f;

    /// @brief  Enable frame‑to‑frame pair caching.
    ///
    /// When enabled, overlapping pairs from the previous frame are reused (their AABBs
    /// are rechecked for overlap via AABB test — no tree walk).  Only newly overlapping
    /// pairs require a tree descent.  This can reduce pair‑generation cost by 50–80 %
    /// in scenes with high temporal coherence (most industrial simulations).
    ///
    /// The cache is updated deterministically and produces bit‑identical output to
    /// the uncached walk for the same input.
    bool enablePairCaching = false;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// DbvtNode  (internal)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Node in the dynamic BVH tree.
///
/// @note  This is an implementation detail and should not be accessed directly.
///        Exposed only for testing and debugging.
struct DbvtNode {
    AABB bounds;                    ///< AABB of this node (union of children for internal nodes).
    int   parent   = -1;            ///< Parent node index, or -1 for root.
    int   child0   = -1;            ///< Left child index, or -1 for leaf.
    int   child1   = -1;            ///< Right child index, or -1 for leaf.
    int   body     = -1;            ///< RigidBodyHandle for leaves; -1 for internal / invalid.
    int   height   = 0;             ///< Subtree height: 0 for leaf, 1+ for internal.
    int   nextFree = -1;            ///< Free‑list link (only used when node is in the free pool).

    /// @brief  True if this node is a leaf (no children).
    [[nodiscard]] bool isLeaf() const noexcept { return child0 < 0; }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Dbvt  — Dynamic Bounding Volume Tree
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A dynamic axis-aligned bounding volume tree for broad‑phase collision culling.
///
/// The Dbvt maintains a binary tree of AABBs across frames.  Bodies are inserted as
/// leaf nodes and internal nodes store the union of their children's AABBs.  The tree
/// supports incremental insertion, removal, and refit — it is ***not*** rebuilt from
/// scratch each frame.
///
/// ## Usage
///
/// @code
///   Dbvt dbvt;
///
///   // Each frame, after integrating velocities:
///   dbvt.sync(bodies, shapes, config);
///
///   // Then generate pairs:
///   std::vector<BroadphasePair> pairs;
///   dbvt.collidePairs(bodies, pairs);
/// @endcode
///
/// @note  Not thread‑safe for concurrent tree modification.  Pair generation may run
///        in parallel with other reads (read‑only).
class Dbvt {
public:
    /// @brief  Default constructor.
    Dbvt() = default;

    // ─── Tree mutation ─────────────────────────────────────────────────────────────────────────

    /// @brief  Insert a new leaf for `body` with the given fat AABB.
    /// @param body   Rigid-body handle.
    /// @param box    Fat AABB (padded) for the body.
    void insert(RigidBodyHandle body, const AABB& box);

    /// @brief  Remove the leaf for `body` from the tree immediately.
    /// @param body  Rigid-body handle.  Safe to call for bodies that have no leaf.
    void remove(RigidBodyHandle body);

    /// @brief  Update the leaf for `body`.  If `newBox` fits inside the current fat AABB
    ///         the call is a no‑op (temporal coherence).  Otherwise the leaf is removed
    ///         and re‑inserted into the tree.
    /// @param body   Rigid-body handle.
    /// @param newBox New (non‑fat) world‑space AABB for the body.
    void update(RigidBodyHandle body, const AABB& newBox);

    /// @brief  Remove all nodes and reset to an empty tree.
    void clear() noexcept;

    // ─── Synchronisation ───────────────────────────────────────────────────────────────────────

    /// @brief  Synchronise the tree with the current body state.
    ///
    /// Call this **once per frame**, after velocity integration, before pair generation.
    /// This method:
    ///   - Inserts leaves for bodies that have no leaf yet (newly created).
    ///   - Updates leaves for bodies whose AABB has moved outside the fat AABB.
    ///   - Removes leaves for bodies that are inactive, have no shapes, or no longer exist.
    ///   - Optionally triggers a full tree rebuild after many removals.
    ///
    /// @param bodies  The SoA rigid-body storage.
    /// @param shapes  The shape registry (indexed by ShapeHandle).
    /// @param config  Tuning parameters (fat‑AABB scale, rebuild threshold).
    void sync(const RigidBodyStorage& bodies,
              const std::vector<Shape>& shapes,
              const DbvtConfig& config);

    // ─── Queries ────────────────────────────────────────────────────────────────────────────────

    /// @brief  Generate all overlapping leaf‑leaf pairs (broad‑phase output).
    ///
    /// When `enablePairCaching` is true in the config passed to `sync()`, the
    /// method uses a frame‑to‑frame pair cache to avoid re‑walking the tree for
    /// pairs that persisted from the previous frame.  Output is deterministic in
    /// both modes.
    ///
    /// Static‑static pairs are skipped.
    ///
    /// @param bodies  Used to query motion type for static‑static filtering.
    /// @param[out] out  Filled with overlapping body pairs.
    void collidePairs(const RigidBodyStorage& bodies,
                      std::vector<BroadphasePair>& out) const;

    /// @brief  Query all leaf bodies whose AABB overlaps `queryBox`.
    ///
    /// Results are sorted and deduplicated.
    ///
    /// @param queryBox  The query AABB in world space.
    /// @param[out] out  Filled with handles of bodies whose AABB overlaps.
    void overlapQuery(const AABB& queryBox,
                      std::vector<RigidBodyHandle>& out) const;

    // ─── Introspection ─────────────────────────────────────────────────────────────────────────

    /// @brief  Current number of nodes in the pool (including free nodes).
    [[nodiscard]] int nodeCount() const noexcept { return static_cast<int>(m_nodes.size()); }

    /// @brief  Current root node index, or -1 for an empty tree.
    [[nodiscard]] int root() const noexcept { return m_root; }

    /// @brief  Read‑only access to the node array (for debugging / testing).
    [[nodiscard]] const std::vector<DbvtNode>& nodes() const noexcept { return m_nodes; }

private:
    // ─── Internal helpers ───────────────────────────────────────────────────────────────────

    /// @brief  Compute the world‑space AABB for a body, accounting for all compound sub‑shapes.
    ///         Returns an empty (inverted) AABB when the body has no valid sub‑shapes.
    [[nodiscard]] static AABB computeBodyAABB(RigidBodyHandle h,
                                               const RigidBodyStorage& bodies,
                                               const std::vector<Shape>& shapes) noexcept;

    /// @brief  Pad an AABB by a fraction of its extents (with a minimum absolute pad).
    [[nodiscard]] static AABB fattenAABB(const AABB& box, float scale) noexcept;

    /// @brief  Surface area of an AABB (used for SAH insertion cost).
    [[nodiscard]] static float surfaceArea(const AABB& box) noexcept;

    // ─── Node pool management ─────────────────────────────────────────────────────────────

    /// @brief  Allocate a node from the free‑list, or grow the pool.
    int allocNode();

    /// @brief  Return a node to the free‑list.
    void freeNode(int idx);

    // ─── Tree operations ───────────────────────────────────────────────────────────────────

    /// @brief  Insert a leaf (already in m_nodes) into the tree.  Returns the new root index.
    int insertLeaf(int leafIdx);

    /// @brief  Remove a leaf from the tree structure without freeing the node.
    void removeLeafFromTree(int leafIdx);

    /// @brief  Walk up from `nodeIdx` recomputing AABBs and heights.
    void refit(int nodeIdx);

    /// @brief  Single AVL-style right rotation at nodeIdx (left child too deep).
    /// @return The new subtree root index.
    int rotateRightInternal(int nodeIdx);

    /// @brief  Single AVL-style left rotation at nodeIdx (right child too deep).
    /// @return The new subtree root index.
    int rotateLeftInternal(int nodeIdx);

    /// @brief  Check height balance at nodeIdx and perform AVL-style rotation(s)
    ///         to restore balance.  Handles single and double rotations.
    /// @return The new subtree root index (same as nodeIdx if no rotation).
    int rotateNode(int nodeIdx);

    /// @brief  Temporary leaf data used during a full tree rebuild.
    struct RebuildLeaf {
        RigidBodyHandle body;
        AABB bounds;
    };

    /// @brief  Recursively build a balanced BVH from a range of RebuildLeaf entries.
    ///         Partitions at the median along @p axis using nth_element.
    int buildLeafRange(std::vector<RebuildLeaf>& leaves,
                       std::size_t begin, std::size_t end, int axis);

    /// @brief  Rebuild the entire tree from scratch using current body AABBs.
    ///         Produces a height-balanced tree via median-split, eliminating
    ///         all accumulated structural degradation.
    void rebuild(const RigidBodyStorage& bodies,
                 const std::vector<Shape>& shapes,
                 const DbvtConfig& config);

    // ─── Pair cache ────────────────────────────────────────────────────────────────────────

    /// @brief  A cached overlapping pair with its last‑seen frame counter.
    struct CachedPair {
        std::uint64_t key;          ///< (bodyA << 32) | bodyB.
        std::uint64_t lastSeen;     ///< Frame counter when this pair was last verified.
    };

    /// @brief  Walk the tree without caching (original algorithm).
    void collidePairsUncached(const RigidBodyStorage& bodies,
                              std::vector<BroadphasePair>& out) const;

    /// @brief  Walk only new pairs; reuse cached pairs from the previous frame.
    /// @return  false if the cache was invalidated and a full walk was done instead.
    bool collidePairsCached(const RigidBodyStorage& bodies,
                            std::vector<BroadphasePair>& out) const;

    // ─── Members ───────────────────────────────────────────────────────────────────────────

    std::vector<DbvtNode> m_nodes;         ///< Node pool (indices are stable).
    int                   m_root     = -1; ///< Root node index, or -1 if empty.
    int                   m_freeHead = -1; ///< Head of the free‑list.

    /// @brief  Maps RigidBodyHandle → leaf node index, or -1 if no leaf exists.
    std::vector<int>      m_bodyToNode;

    /// @brief  Frame counter for pair‑cache temporal coherence.
    mutable std::uint64_t m_frameCount = 0;

    /// @brief  Cached pairs from the previous frame.
    mutable std::vector<CachedPair> m_pairCache;

    /// @brief  Copy of `enablePairCaching` from the last `sync()` call.
    bool m_enablePairCaching = false;

    /// @brief  Count of removals since the last full rebuild.
    ///         When this reaches DbvtConfig::rebuildThreshold the tree is
    ///         rebuilt from scratch to restore near-optimal balance.
    int m_removalCount = 0;

    /// @brief  Count of insertions since the last full rebuild.
    ///         When this exceeds the growth threshold the tree is rebuilt.
    int m_insertionCount = 0;

    /// @brief  Body count at the time of the last full rebuild.
    ///         Used to calculate the insertion growth trigger.
    std::size_t m_bodyCountAtLastRebuild = 0;
};

//==================================================================================================
//  Implementation
//==================================================================================================

// ─── Node pool management ─────────────────────────────────────────────────────────────────────

inline int Dbvt::allocNode() {
    if (m_freeHead >= 0) {
        int idx = m_freeHead;
        m_freeHead = m_nodes[idx].nextFree;
        m_nodes[idx] = DbvtNode{}; // zero-init
        return idx;
    }
    int idx = static_cast<int>(m_nodes.size());
    m_nodes.emplace_back();
    return idx;
}

inline void Dbvt::freeNode(int idx) {
    m_nodes[idx] = DbvtNode{};
    m_nodes[idx].nextFree = m_freeHead;
    m_freeHead = idx;
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────────────────────

inline float Dbvt::surfaceArea(const AABB& box) noexcept {
    Vec3f ext = box.extents();
    return 2.0f * (ext.x * ext.y + ext.x * ext.z + ext.y * ext.z);
}

inline AABB Dbvt::fattenAABB(const AABB& box, float scale) noexcept {
    Vec3f ext = box.extents();
    // Enforce a minimum absolute pad so degenerate AABBs don't collapse.
    float minPad = 0.01f;
    Vec3f margin(
        std::max(ext.x * scale, minPad),
        std::max(ext.y * scale, minPad),
        std::max(ext.z * scale, minPad)
    );
    AABB fat;
    fat.min = box.min - margin;
    fat.max = box.max + margin;
    return fat;
}

inline AABB Dbvt::computeBodyAABB(RigidBodyHandle h,
                                   const RigidBodyStorage& bodies,
                                   const std::vector<Shape>& shapes) noexcept {
    std::uint32_t start = bodies.shapeStart[h];
    std::uint32_t count = bodies.shapeCount[h];

    AABB bodyAABB;
    bool first = true;

    for (std::uint32_t s = 0; s < count; ++s) {
        ShapeHandle sh = bodies.flatShapeHandles[start + s];
        if (sh >= shapes.size()) continue;

        // World-space sub-shape transform.
        Vec3f localPos = bodies.flatShapeLocalPositions[start + s];
        Quaternionf localRot = bodies.flatShapeLocalRotations[start + s];
        Vec3f worldPos = bodies.positions[h]
                        + bodies.rotations[h].rotate(localPos);
        Quaternionf worldRot = bodies.rotations[h] * localRot;

        AABB subAABB = shapes[sh].getAABB(worldPos, worldRot);
        if (first) {
            bodyAABB = subAABB;
            first = false;
        } else {
            bodyAABB = bodyAABB.united(subAABB);
        }
    }

    return bodyAABB; // inverted sentinel if no valid sub-shapes
}

// ─── Tree operations ───────────────────────────────────────────────────────────────────────────

inline int Dbvt::insertLeaf(int leafIdx) {
    auto& leaf = m_nodes[leafIdx];
    leaf.parent = -1;

    if (m_root < 0) {
        m_root = leafIdx;
        return m_root;
    }

    // ── Find the best sibling using the surface‑area heuristic ──────────────────────────────
    //
    //  Descend from the root, always choosing the child whose AABB increases the
    //  least in surface area when combined with the new leaf.

    int sibling = m_root;
    while (!m_nodes[sibling].isLeaf()) {
        int c0 = m_nodes[sibling].child0;
        int c1 = m_nodes[sibling].child1;

        // Cost of inserting into each child subtree.
        float costC0 = surfaceArea(m_nodes[c0].bounds.united(leaf.bounds))
                     - surfaceArea(m_nodes[c0].bounds);
        float costC1 = surfaceArea(m_nodes[c1].bounds.united(leaf.bounds))
                     - surfaceArea(m_nodes[c1].bounds);

        sibling = (costC0 < costC1) ? c0 : c1;
    }

    // ── Create a new internal node to parent the leaf and sibling ──────────────────────────

    int internal = allocNode();
    auto& inode = m_nodes[internal];
    inode.child0  = sibling;
    inode.child1  = leafIdx;
    inode.bounds  = m_nodes[sibling].bounds.united(leaf.bounds);
    inode.height  = 1 + std::max(m_nodes[sibling].height, 0);
    inode.parent  = m_nodes[sibling].parent;

    // Link leaf and sibling to the new parent.
    m_nodes[leafIdx].parent  = internal;
    m_nodes[sibling].parent  = internal;

    // Wire the new internal node into the tree at the sibling's former position.
    if (inode.parent >= 0) {
        auto& parent = m_nodes[inode.parent];
        if (parent.child0 == sibling) parent.child0 = internal;
        else                          parent.child1 = internal;
    } else {
        m_root = internal;
    }

    // ── Refit AABBs upward ─────────────────────────────────────────────────────────────────
    refit(internal);

    // ── Restore AVL height balance via tree rotations ─────────────────────────────────────
    //
    //  Walk up from the new parent internal node checking balance at each
    //  ancestor.  Rotations maintain near-optimal tree shape across many
    //  insert/delete cycles, preventing degenerative behaviour.
    {
        int checkIdx = m_nodes[leafIdx].parent;
        while (checkIdx >= 0) {
            checkIdx = rotateNode(checkIdx);
            checkIdx = m_nodes[checkIdx].parent;
        }
    }

    return m_root;
}

inline void Dbvt::removeLeafFromTree(int leafIdx) {
    auto& leaf = m_nodes[leafIdx];

    if (leafIdx == m_root) {
        m_root = -1;
        return;
    }

    int parent  = leaf.parent;
    int sibling = (m_nodes[parent].child0 == leafIdx)
                    ? m_nodes[parent].child1
                    : m_nodes[parent].child0;
    int grandPa = m_nodes[parent].parent;

    // Sibling takes the parent's place.
    m_nodes[sibling].parent = grandPa;
    if (grandPa >= 0) {
        auto& gp = m_nodes[grandPa];
        if (gp.child0 == parent) gp.child0 = sibling;
        else                     gp.child1 = sibling;
        refit(grandPa);
    } else {
        m_root = sibling;
    }

    // Free the (now disconnected) parent internal node.
    freeNode(parent);
}

inline void Dbvt::refit(int nodeIdx) {
    while (nodeIdx >= 0) {
        auto& node = m_nodes[nodeIdx];
        if (!node.isLeaf()) {
            node.bounds = m_nodes[node.child0].bounds.united(m_nodes[node.child1].bounds);
            node.height = 1 + std::max(m_nodes[node.child0].height,
                                       m_nodes[node.child1].height);
        }
        nodeIdx = node.parent;
    }
}

// ─── Tree rotations (AVL-style) ────────────────────────────────────────────────────────────────

inline int Dbvt::rotateRightInternal(int nodeIdx) {
    // Single right rotation on node A (left child B is too deep):
    //
    //       A              B
    //      / |            / |
    //     B  Ar    ->    Bl  A
    //    / |                / |
    //   Bl Br             Br  Ar

    auto& A       = m_nodes[nodeIdx];
    int   parentA = A.parent;
    int   B       = A.child0;

    if (B < 0) return nodeIdx; // safety — should not happen

    auto& nodeB = m_nodes[B];
    int   Br    = nodeB.child1; // B's right child becomes A's left

    // A adopts Br as its left child.
    A.child0 = Br;
    if (Br >= 0) m_nodes[Br].parent = nodeIdx;

    // B adopts A as its right child.
    nodeB.child1 = nodeIdx;
    A.parent     = B;

    // B takes A's former position in the tree.
    nodeB.parent = parentA;
    if (parentA >= 0) {
        auto& pA = m_nodes[parentA];
        if (pA.child0 == nodeIdx) pA.child0 = B;
        else                      pA.child1 = B;
    } else {
        m_root = B;
    }

    // Refit upwards — A is now a child of B, so refit(A) propagates through B.
    refit(nodeIdx);
    return B;
}

inline int Dbvt::rotateLeftInternal(int nodeIdx) {
    // Single left rotation on node A (right child B is too deep):
    //
    //     A                 B
    //    / |               / |
    //   Al  B      ->     A  Br
    //      / |           / |
    //     Bl Br         Al  Bl

    auto& A       = m_nodes[nodeIdx];
    int   parentA = A.parent;
    int   B       = A.child1;

    if (B < 0) return nodeIdx;

    auto& nodeB = m_nodes[B];
    int   Bl    = nodeB.child0; // B's left child becomes A's right

    // A adopts Bl as its right child.
    A.child1 = Bl;
    if (Bl >= 0) m_nodes[Bl].parent = nodeIdx;

    // B adopts A as its left child.
    nodeB.child0 = nodeIdx;
    A.parent     = B;

    // B takes A's former position in the tree.
    nodeB.parent = parentA;
    if (parentA >= 0) {
        auto& pA = m_nodes[parentA];
        if (pA.child0 == nodeIdx) pA.child0 = B;
        else                      pA.child1 = B;
    } else {
        m_root = B;
    }

    refit(nodeIdx);
    return B;
}

inline int Dbvt::rotateNode(int nodeIdx) {
    // AVL-style balance check.  If the height difference between children
    // exceeds 1, perform single or double rotations to restore balance.

    if (nodeIdx < 0) return nodeIdx;
    auto& node = m_nodes[nodeIdx];
    if (node.isLeaf()) return nodeIdx;

    int c0 = node.child0;
    int c1 = node.child1;
    if (c0 < 0 || c1 < 0) return nodeIdx;

    auto& ch0 = m_nodes[c0];
    auto& ch1 = m_nodes[c1];
    int bal = ch0.height - ch1.height;

    if (bal >= -1 && bal <= 1)
        return nodeIdx; // already balanced

    if (bal > 1) {
        // Left-heavy — need a right rotation (possibly left-right double).
        if (!ch0.isLeaf()) {
            auto& ch0c0 = m_nodes[ch0.child0];
            auto& ch0c1 = m_nodes[ch0.child1];
            if (ch0c1.height > ch0c0.height) {
                // Left-Right case: first rotate left at child, then right here.
                node.child0 = rotateLeftInternal(c0);
            }
        }
        return rotateRightInternal(nodeIdx);
    }

    // Right-heavy — need a left rotation (possibly right-left double).
    if (!ch1.isLeaf()) {
        auto& ch1c0 = m_nodes[ch1.child0];
        auto& ch1c1 = m_nodes[ch1.child1];
        if (ch1c0.height > ch1c1.height) {
            // Right-Left case: first rotate right at child, then left here.
            node.child1 = rotateRightInternal(c1);
        }
    }
    return rotateLeftInternal(nodeIdx);
}

// ─── Public API ────────────────────────────────────────────────────────────────────────────────

inline void Dbvt::insert(RigidBodyHandle body, const AABB& box) {
    int leafIdx = allocNode();
    auto& leaf = m_nodes[leafIdx];
    leaf.bounds = box;
    leaf.body   = static_cast<int>(body);
    leaf.height = 0;

    // Ensure the body-to-node mapping is large enough.
    if (static_cast<std::size_t>(body) >= m_bodyToNode.size()) {
        m_bodyToNode.resize(static_cast<std::size_t>(body) + 1, -1);
    }
    m_bodyToNode[body] = leafIdx;

    m_root = insertLeaf(leafIdx);
}

inline void Dbvt::remove(RigidBodyHandle body) {
    if (static_cast<std::size_t>(body) >= m_bodyToNode.size()) return;
    int leafIdx = m_bodyToNode[body];
    if (leafIdx < 0) return;

    removeLeafFromTree(leafIdx);
    freeNode(leafIdx);
    m_bodyToNode[body] = -1;
    ++m_removalCount;
}

inline void Dbvt::update(RigidBodyHandle body, const AABB& newBox) {
    if (static_cast<std::size_t>(body) >= m_bodyToNode.size()) return;
    int leafIdx = m_bodyToNode[body];
    if (leafIdx < 0) return;

    auto& leaf = m_nodes[leafIdx];

    // Temporal coherence: skip the update if the new AABB still fits inside the fat AABB.
    if (newBox.min.x >= leaf.bounds.min.x &&
        newBox.min.y >= leaf.bounds.min.y &&
        newBox.min.z >= leaf.bounds.min.z &&
        newBox.max.x <= leaf.bounds.max.x &&
        newBox.max.y <= leaf.bounds.max.y &&
        newBox.max.z <= leaf.bounds.max.z) {
        return;
    }

    // If this is the only node, just update the bounds in-place.
    if (leafIdx == m_root) {
        leaf.bounds = newBox;
        return;
    }

    // Otherwise remove the leaf from the tree and re-insert it.
    removeLeafFromTree(leafIdx);
    leaf.bounds = newBox;
    m_root = insertLeaf(leafIdx);
}

inline void Dbvt::clear() noexcept {
    m_nodes.clear();
    m_root     = -1;
    m_freeHead = -1;
    m_bodyToNode.clear();
    m_pairCache.clear();
    m_frameCount = 0;
    m_enablePairCaching = false;
    m_removalCount = 0;
}

inline void Dbvt::sync(const RigidBodyStorage& bodies,
                        const std::vector<Shape>& shapes,
                        const DbvtConfig& config) {
    m_enablePairCaching = config.enablePairCaching;
    std::size_t n       = bodies.size();
    std::size_t oldSize = m_bodyToNode.size();
    // Resize the body-to-node mapping for newly created bodies.
    if (n > oldSize) {
        m_bodyToNode.resize(n, -1);
    }

    // ── Phase 0: full rebuild on first sync ───────────────────────────────────────────────
    //
    //  When the tree is brand new (oldSize == 0) we skip incremental insertion entirely
    //  and build a balanced tree directly.  Incremental insertion of hundreds
    //  or thousands of bodies via SAH can produce a degenerate tree (depth O(n)), which
    //  causes crashes in insertLeaf() at >= 1024 bodies.
    //
    //  After the first sync, incremental insertion/update is safe because the tree is
    //  already balanced and the body count delta between frames is small.

    if (oldSize == 0 && n > 0) {
        // Collect all active bodies directly (don't check m_bodyToNode since it's all -1).
        std::vector<RebuildLeaf> leaves;
        leaves.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!bodies.activeFlags[i]) continue;
            if (bodies.shapeCount[i] == 0) continue;
            RigidBodyHandle h = static_cast<RigidBodyHandle>(i);
            AABB aabb = computeBodyAABB(h, bodies, shapes);
            if (aabb.min.x > aabb.max.x) continue;
            AABB fatBox = fattenAABB(aabb, config.fatAABBScale);
            leaves.push_back({h, fatBox});
        }

        // Clear existing state and build balanced tree.
        m_nodes.clear();
        m_freeHead = -1;
        m_bodyToNode.assign(n, -1);
        m_root = -1;

        if (!leaves.empty()) {
            // Sort along longest axis for spatial coherence.
            AABB totalBounds;
            for (const auto& l : leaves) totalBounds = totalBounds.united(l.bounds);
            Vec3f ext = totalBounds.extents();
            int splitAxis = 0;
            if (ext.y > ext.x) splitAxis = 1;
            if (ext.z > ext.x && ext.z > ext.y) splitAxis = 2;
            std::sort(leaves.begin(), leaves.end(),
                [splitAxis](const RebuildLeaf& a, const RebuildLeaf& b) noexcept {
                    float ca = a.bounds.min[splitAxis] + a.bounds.max[splitAxis];
                    float cb = b.bounds.min[splitAxis] + b.bounds.max[splitAxis];
                    return ca < cb;
                });
            m_root = buildLeafRange(leaves, 0, leaves.size(), splitAxis);
        }

        m_insertionCount = 0;
        m_bodyCountAtLastRebuild = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (bodies.activeFlags[j] && bodies.shapeCount[j] > 0) ++m_bodyCountAtLastRebuild;
        }
        return;
    }

    // ── Phase 1: insert new bodies and update moved ones ──────────────────────────────────

    for (std::size_t i = 0; i < n; ++i) {
        if (!bodies.activeFlags[i]) continue;
        if (bodies.shapeCount[i] == 0) continue;

        RigidBodyHandle h = static_cast<RigidBodyHandle>(i);
        AABB aabb = computeBodyAABB(h, bodies, shapes);

        // Skip bodies with no valid sub-shapes (inverted sentinel AABB).
        if (aabb.min.x > aabb.max.x) continue;

        AABB fatBox = fattenAABB(aabb, config.fatAABBScale);

        if (static_cast<std::size_t>(h) >= m_bodyToNode.size() ||
            m_bodyToNode[h] < 0) {
            insert(h, fatBox);
            ++m_insertionCount;
        } else {
            update(h, fatBox);
        }
    }

    // ── Phase 2: remove leaves for bodies that are gone, inactive, or shapeless ──────────

    for (std::size_t i = 0; i < oldSize; ++i) {
        if (m_bodyToNode[i] < 0) continue;
        bool keep = (i < n) && bodies.activeFlags[i] && bodies.shapeCount[i] > 0;
        if (!keep) {
            remove(static_cast<RigidBodyHandle>(i));
        }
    }

    // ── Phase 3: trim the body-to-node mapping if the body count shrunk ──────────────────

    if (n < m_bodyToNode.size()) {
        m_bodyToNode.resize(n);
    }

    // ── Phase 4: full rebuild if removal or insertion threshold exceeded ────────────────
    //
    //  A full rebuild produces a perfectly balanced tree and eliminates all
    //  stale internal nodes that accumulated during incremental updates.
    //  We check both removal count AND insertion growth to prevent degenerate
    //  trees in spawn-heavy scenes (fracture, crowds, debris).

    bool removalTrigger = (m_removalCount >= config.rebuildThreshold);

    bool insertionTrigger = false;
    if (config.insertionGrowthThreshold > 0.0f) {
        std::size_t currentActive = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (bodies.activeFlags[j] && bodies.shapeCount[j] > 0) ++currentActive;
        }
        std::size_t delta = (currentActive > m_bodyCountAtLastRebuild)
                            ? (currentActive - m_bodyCountAtLastRebuild)
                            : 0;
        if (config.insertionGrowthThreshold >= 1.0f) {
            // Absolute threshold.
            insertionTrigger = (delta >= static_cast<std::size_t>(config.insertionGrowthThreshold));
        } else {
                // Fractional threshold of the body count at last rebuild.
                std::size_t threshold = static_cast<std::size_t>(
                    static_cast<float>(m_bodyCountAtLastRebuild) * config.insertionGrowthThreshold);
                if (threshold < 1) threshold = 1;
                insertionTrigger = (delta >= threshold);
            }
        }

        if (removalTrigger || insertionTrigger) {
            rebuild(bodies, shapes, config);
        m_removalCount = 0;
        m_insertionCount = 0;
        // Count active bodies for next growth trigger.
        m_bodyCountAtLastRebuild = 0;
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            if (bodies.activeFlags[j] && bodies.shapeCount[j] > 0) ++m_bodyCountAtLastRebuild;
        }
    }
}

inline void Dbvt::collidePairs(const RigidBodyStorage& bodies,
                                std::vector<BroadphasePair>& out) const {
    out.clear();
    if (m_root < 0) return;

    if (m_enablePairCaching) {
        collidePairsCached(bodies, out);
    } else {
        collidePairsUncached(bodies, out);
    }
}

inline void Dbvt::collidePairsUncached(const RigidBodyStorage& bodies,
                                        std::vector<BroadphasePair>& out) const {
    // ── Stack-based tree walk ─────────────────────────────────────────────────────────────
    //
    //  We use a fixed-discipline descent: when both nodes are internal, we expand the one
    //  with the greater height (more balanced tree).  When one is a leaf, we always expand
    //  the internal node.

    struct StackPair { int a; int b; };
    std::vector<StackPair> stack;
    stack.reserve(128);
    stack.push_back({m_root, m_root});

    while (!stack.empty()) {
        StackPair p = stack.back();
        stack.pop_back();

        const auto& nodeA = m_nodes[p.a];
        const auto& nodeB = m_nodes[p.b];

        if (!nodeA.bounds.overlaps(nodeB.bounds)) continue;

        if (nodeA.isLeaf() && nodeB.isLeaf()) {
            if (p.a == p.b) continue;

            RigidBodyHandle ba = static_cast<RigidBodyHandle>(nodeA.body);
            RigidBodyHandle bb = static_cast<RigidBodyHandle>(nodeB.body);

            // Skip static-static pairs.
            if (bodies.motionTypes[ba] != MotionType::Dynamic &&
                bodies.motionTypes[bb] != MotionType::Dynamic) continue;

            // Emit in canonical order: smaller handle first.
            if (ba < bb) {
                out.push_back({ba, bb});
            } else {
                out.push_back({bb, ba});
            }
            continue;
        }

        // Decide which node to expand.
        if (nodeB.isLeaf() || (!nodeA.isLeaf() && nodeA.height >= nodeB.height)) {
            stack.push_back({nodeA.child0, p.b});
            stack.push_back({nodeA.child1, p.b});
        } else {
            stack.push_back({p.a, nodeB.child0});
            stack.push_back({p.a, nodeB.child1});
        }
    }

    // ── Sort and deduplicate for deterministic output ─────────────────────────────────────
    std::sort(out.begin(), out.end(),
        [](const BroadphasePair& a, const BroadphasePair& b) noexcept {
            return a.bodyA < b.bodyA || (a.bodyA == b.bodyA && a.bodyB < b.bodyB);
        });
    out.erase(std::unique(out.begin(), out.end(),
        [](const BroadphasePair& a, const BroadphasePair& b) noexcept {
            return a.bodyA == b.bodyA && a.bodyB == b.bodyB;
        }), out.end());
}

inline bool Dbvt::collidePairsCached(const RigidBodyStorage& bodies,
                                      std::vector<BroadphasePair>& out) const {
    // ── Frame counter ─────────────────────────────────────────────────────────────────────
    ++m_frameCount;

    // ── Phase 1: validate the existing cache ──────────────────────────────────────────────
    //
    //  For each cached pair, verify that both bodies still exist and their AABBs still
    //  overlap.  Remove stale entries and update the lastSeen counter for validated ones.
    //  This is done in-place using a read/write index.

    std::size_t writeIdx = 0;
    for (std::size_t i = 0; i < m_pairCache.size(); ++i) {
        std::uint64_t key       = m_pairCache[i].key;
        RigidBodyHandle ba = static_cast<RigidBodyHandle>(key >> 32);
        RigidBodyHandle bb = static_cast<RigidBodyHandle>(key & 0xFFFFFFFF);

        // Skip if either body handle is out of range or inactive/shapeless.
        std::size_t idxA = static_cast<std::size_t>(ba);
        std::size_t idxB = static_cast<std::size_t>(bb);
        if (idxA >= bodies.size() || idxB >= bodies.size()) continue;
        if (!bodies.activeFlags[ba] || !bodies.activeFlags[bb]) continue;
        if (bodies.shapeCount[ba] == 0 || bodies.shapeCount[bb] == 0) continue;

        // Skip static-static pairs.
        if (bodies.motionTypes[ba] != MotionType::Dynamic &&
            bodies.motionTypes[bb] != MotionType::Dynamic) continue;

        // Re-check AABB overlap for the cached pair.
        // We need the leaf node indices to get the fat AABBs.
        if (static_cast<std::size_t>(ba) >= m_bodyToNode.size() ||
            static_cast<std::size_t>(bb) >= m_bodyToNode.size()) continue;
        int leafA = m_bodyToNode[ba];
        int leafB = m_bodyToNode[bb];
        if (leafA < 0 || leafB < 0) continue;
        if (!m_nodes[leafA].bounds.overlaps(m_nodes[leafB].bounds)) continue;

        // Pair is still valid — keep it.
        m_pairCache[writeIdx] = m_pairCache[i];
        m_pairCache[writeIdx].lastSeen = m_frameCount;
        ++writeIdx;
    }
    m_pairCache.resize(writeIdx);

    // ── Phase 2: walk the tree for NEW pairs ──────────────────────────────────────────────
    //
    //  Walk the full tree but skip leaf-leaf pairs that are already in the cache.
    //  Build a set of cached keys for O(log cache) lookup via binary search.
    //
    //  Sort the cache now so binary search works during the walk.
    std::sort(m_pairCache.begin(), m_pairCache.end(),
        [](const CachedPair& a, const CachedPair& b) noexcept {
            return a.key < b.key;
        });

    // Helper: binary search the cache for a given key.
    auto isCached = [&](std::uint64_t k) noexcept -> bool {
        auto it = std::lower_bound(m_pairCache.begin(), m_pairCache.end(), k,
            [](const CachedPair& cp, std::uint64_t kk) noexcept { return cp.key < kk; });
        return (it != m_pairCache.end() && it->key == k);
    };

    struct StackPair { int a; int b; };
    std::vector<StackPair> stack;
    stack.reserve(128);
    stack.push_back({m_root, m_root});

    while (!stack.empty()) {
        StackPair p = stack.back();
        stack.pop_back();

        const auto& nodeA = m_nodes[p.a];
        const auto& nodeB = m_nodes[p.b];

        if (!nodeA.bounds.overlaps(nodeB.bounds)) continue;

        if (nodeA.isLeaf() && nodeB.isLeaf()) {
            if (p.a == p.b) continue;

            RigidBodyHandle ba = static_cast<RigidBodyHandle>(nodeA.body);
            RigidBodyHandle bb = static_cast<RigidBodyHandle>(nodeB.body);

            // Skip static-static pairs.
            if (bodies.motionTypes[ba] != MotionType::Dynamic &&
                bodies.motionTypes[bb] != MotionType::Dynamic) continue;

            // Canonical key.
            std::uint64_t key = (ba < bb)
                ? ((static_cast<std::uint64_t>(ba) << 32) | bb)
                : ((static_cast<std::uint64_t>(bb) << 32) | ba);

            if (!isCached(key)) {
                // New pair — add to cache.
                m_pairCache.push_back({key, m_frameCount});
            }
            continue;
        }

        if (nodeB.isLeaf() || (!nodeA.isLeaf() && nodeA.height >= nodeB.height)) {
            stack.push_back({nodeA.child0, p.b});
            stack.push_back({nodeA.child1, p.b});
        } else {
            stack.push_back({p.a, nodeB.child0});
            stack.push_back({p.a, nodeB.child1});
        }
    }

    // ── Phase 3: deduplicate the cache and emit pairs ──────────────────────────────────────
    //
    //  The tree walk can visit the same leaf pair via different traversal
    //  paths when internal-node AABBs overlap at multiple tree levels.
    //  Sorting + unique removes these duplicates.
    std::sort(m_pairCache.begin(), m_pairCache.end(),
        [](const CachedPair& a, const CachedPair& b) noexcept {
            return a.key < b.key;
        });
    m_pairCache.erase(
        std::unique(m_pairCache.begin(), m_pairCache.end(),
            [](const CachedPair& a, const CachedPair& b) noexcept {
                return a.key == b.key;
            }),
        m_pairCache.end());

    out.reserve(m_pairCache.size());
    for (const auto& cp : m_pairCache) {
        RigidBodyHandle ba = static_cast<RigidBodyHandle>(cp.key >> 32);
        RigidBodyHandle bb = static_cast<RigidBodyHandle>(cp.key & 0xFFFFFFFF);
        out.push_back({ba, bb});
    }

    return true;
}

inline void Dbvt::overlapQuery(const AABB& queryBox,
                                std::vector<RigidBodyHandle>& out) const {
    out.clear();
    if (m_root < 0) return;

    std::vector<int> stack;
    stack.reserve(64);
    stack.push_back(m_root);

    while (!stack.empty()) {
        int idx = stack.back();
        stack.pop_back();

        const auto& node = m_nodes[idx];
        if (!node.bounds.overlaps(queryBox)) continue;

        if (node.isLeaf()) {
            if (node.body >= 0) {
                out.push_back(static_cast<RigidBodyHandle>(node.body));
            }
        } else {
            stack.push_back(node.child0);
            stack.push_back(node.child1);
        }
    }

    // Sort and deduplicate.
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

// ─── Rebuild (full tree from scratch) ───────────────────────────────────────────────────────────

inline void Dbvt::rebuild(const RigidBodyStorage& bodies,
                          const std::vector<Shape>& shapes,
                          const DbvtConfig& config) {
    // ── Collect all active leaves ───────────────────────────────────────────────────────────
    std::vector<RebuildLeaf> leaves;
    leaves.reserve(m_bodyToNode.size());

    for (std::size_t i = 0; i < m_bodyToNode.size(); ++i) {
        if (m_bodyToNode[i] < 0) continue;

        RigidBodyHandle h = static_cast<RigidBodyHandle>(i);
        AABB aabb = computeBodyAABB(h, bodies, shapes);
        if (aabb.min.x > aabb.max.x) continue; // invalid / inverted sentinel
        AABB fatBox = fattenAABB(aabb, config.fatAABBScale);
        leaves.push_back({h, fatBox});
    }

    // ── Clear the existing tree ─────────────────────────────────────────────────────────────
    m_nodes.clear();
    m_freeHead = -1;
    m_bodyToNode.assign(m_bodyToNode.size(), -1);
    m_root = -1;

    if (leaves.empty()) return;

    // ── Sort once along the longest global axis for spatial coherence ───────────────────────
    AABB totalBounds;
    for (const auto& l : leaves) totalBounds = totalBounds.united(l.bounds);
    Vec3f ext = totalBounds.extents();
    int splitAxis = 0;
    if (ext.y > ext.x) splitAxis = 1;
    if (ext.z > ext.x && ext.z > ext.y) splitAxis = 2;

    std::sort(leaves.begin(), leaves.end(),
        [splitAxis](const RebuildLeaf& a, const RebuildLeaf& b) noexcept {
            float ca = a.bounds.min[splitAxis] + a.bounds.max[splitAxis];
            float cb = b.bounds.min[splitAxis] + b.bounds.max[splitAxis];
            return ca < cb;
        });

    // ── Recursively build a balanced tree ───────────────────────────────────────────────────
    m_root = buildLeafRange(leaves, 0, leaves.size(), splitAxis);
}

inline int Dbvt::buildLeafRange(std::vector<RebuildLeaf>& leaves,
                                 std::size_t begin, std::size_t end,
                                 int axis) {
    std::size_t count = end - begin;

    // Base case: single leaf.
    if (count == 1) {
        int idx = allocNode();
        auto& node = m_nodes[idx];
        node.bounds = leaves[begin].bounds;
        node.body   = static_cast<int>(leaves[begin].body);
        node.height = 0;
        m_bodyToNode[leaves[begin].body] = idx;
        return idx;
    }

    // Partition at the median along the current axis using nth_element.
    std::size_t mid = begin + count / 2;
    std::nth_element(
        leaves.begin() + static_cast<std::ptrdiff_t>(begin),
        leaves.begin() + static_cast<std::ptrdiff_t>(mid),
        leaves.begin() + static_cast<std::ptrdiff_t>(end),
        [axis](const RebuildLeaf& a, const RebuildLeaf& b) noexcept {
            float ca = a.bounds.min[axis] + a.bounds.max[axis];
            float cb = b.bounds.min[axis] + b.bounds.max[axis];
            return ca < cb;
        });

    // Compute the best split axis for each child from their bounds.
    auto computeSplitAxis = [](const std::vector<RebuildLeaf>& lvs,
                                std::size_t b, std::size_t e) -> int {
        AABB bbox;
        for (std::size_t i = b; i < e; ++i) bbox = bbox.united(lvs[i].bounds);
        Vec3f ext = bbox.extents();
        int a = 0;
        if (ext.y > ext.x) a = 1;
        if (ext.z > ext.x && ext.z > ext.y) a = 2;
        return a;
    };

    int leftAxis  = computeSplitAxis(leaves, begin, mid);
    int rightAxis = computeSplitAxis(leaves, mid, end);

    int left  = buildLeafRange(leaves, begin, mid, leftAxis);
    int right = buildLeafRange(leaves, mid, end, rightAxis);

    // Create the internal parent node.
    int parent = allocNode();
    auto& node    = m_nodes[parent];
    node.child0   = left;
    node.child1   = right;
    node.bounds   = m_nodes[left].bounds.united(m_nodes[right].bounds);
    node.height   = 1 + std::max(m_nodes[left].height, m_nodes[right].height);
    node.body     = -1;
    node.parent   = -1;
    m_nodes[left].parent  = parent;
    m_nodes[right].parent = parent;
    return parent;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Free function (dispatch compatibility)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Generate candidate broad‑phase pairs using a synchronised Dbvt.
///
/// @note  Unlike stateless broadphase algorithms (brute‑force, SAP), the Dbvt is
///        **stateful** — it maintains the AABB tree across frames.  Call `Dbvt::sync()`
///        before this function to ensure the tree reflects the current body state.
///
/// @param bodies   SoA rigid-body storage.
/// @param config   Dbvt tuning parameters (fat‑AABB scale, rebuild threshold).
/// @param dbvt     The Dbvt instance that has already been `sync()`'d this frame.
/// @param[out] pairs  Filled with overlapping body pairs (deterministic order).
inline void generateDbvtPairs(
    const RigidBodyStorage& bodies,
    const DbvtConfig& /*config*/,
    const Dbvt& dbvt,
    std::vector<BroadphasePair>& pairs) noexcept {

    dbvt.collidePairs(bodies, pairs);
}

} // namespace ciphyxs
