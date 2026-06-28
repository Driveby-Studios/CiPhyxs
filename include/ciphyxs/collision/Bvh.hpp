//==================================================================================================
/// @file  Bvh.hpp
/// @brief  Bounding Volume Hierarchy for triangle mesh acceleration.
///
/// Provides a median-split BVH for ray casting and AABB overlap queries against triangle
/// meshes.  The builder uses a simple median-split along the longest axis of the centroid
/// bounds (a pragmatic SAH approximation that works well for static geometry at load time).
///
/// ## Performance
///
///   - Build: O(N log N) median-split (no SAH binning — fine for ~100k triangles).
///   - Traversal: iterative stack with ray-AABB pruning.
///   - Ray–triangle test: Möller–Trumbore (barycentric).
///
/// ## Usage
/// @code
///   Bvh bvh;
///   bvh.build(vertices, indices, triCount);
///
///   float t;
///   Vec3f normal;
///   int triIdx;
///   if (bvh.rayCast(origin, dir, maxT, vertices, indices, t, normal, triIdx)) {
///       // hit at distance t on triangle triIdx
///   }
///
///   int cursor = 0;
///   int buffer[256];
///   int count  = 0;
///   if (bvh.queryOverlap(aabb, vertices, indices, cursor, 256, buffer, count)) {
///       // buffer[0 .. count-1] are overlapping triangle indices
///   }
/// @endcode
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../core/AlignedAllocator.hpp"
#include "Shape.hpp"                   // AABB

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Maximum triangles in a BVH leaf node.
///         Smaller values produce deeper trees with tighter bounds; 2–4 is typical.
inline constexpr int kBvhLeafSize = 2;

/// @brief  Maximum stack depth for iterative traversal.
///         With median-split and up to 100k triangles, depth never exceeds ~30.
inline constexpr int kBvhMaxStack = 64;

// ────────────────────────────────────────────────────────────────────────────────────────────────
// BvhNode
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A single node in the BVH tree.
///
/// Internal nodes (`isLeaf() == false`) have valid `left` / `right` child indices and
/// `triCount == -1`.  Leaf nodes have `triCount >= 0` and index a contiguous block of
/// triangles in the BVH's internal primitive index array via `triStart` / `triCount`.
struct BvhNode {
    AABB bounds;       ///< Axis-aligned bounding box of all primitives in this subtree.
    int   left     = -1;  ///< Left child node index (internal nodes only).
    int   right    = -1;  ///< Right child node index (internal nodes only).
    int   triStart =  0;  ///< First triangle index into the internal index array (leaf only).
    int   triCount = -1;  ///< Number of triangles (-1 for internal nodes).

    /// @brief  True when this node is a leaf (contains actual triangles).
    [[nodiscard]] bool isLeaf() const noexcept { return triCount >= 0; }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Bvh
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Bounding Volume Hierarchy for triangle mesh ray-casting and overlap queries.
///
/// The BVH is built via median-split along the longest axis of the triangle centroids
/// (a simple SAH approximation that avoids binning overhead).  It supports up to ~100k
/// triangles efficiently and is suitable for static geometry built at load time.
///
/// @note  The builder rearranges an internal primitive-index array so that leaf triangles
///        form contiguous ranges for cache-friendly traversal.
struct Bvh {
    /// @brief  Tree nodes stored in depth-first order (parent before children).
    AlignedVector<BvhNode, 16> nodes;

    // ─── Construction ───────────────────────────────────────────────────────────────────────

    /// @brief  Build the BVH from a triangle mesh.
    ///
    /// This is a one-shot builder that partitions the triangles using median-split.
    /// It stores a rearranged copy of the triangle index data internally so that leaf
    /// triangles are contiguous in memory for cache-friendly traversal.
    ///
    /// @param verts     Vertex array (must remain valid for subsequent queries).
    /// @param indices   Triangle index array, 3 indices per triangle
    ///                  (`verts[indices[i*3+0]]` etc.).
    /// @param triCount  Number of triangles in the mesh.
    void build(const Vec3f* verts, const int* indices, int triCount) noexcept {
        nodes.clear();
        primitiveIndices_.clear();

        if (triCount <= 0)
            return;

        // --- initialise primitive permutation ------------------------------------------------
        primitiveIndices_.resize(static_cast<std::size_t>(triCount));
        for (int i = 0; i < triCount; ++i)
            primitiveIndices_[static_cast<std::size_t>(i)] = i;

        // --- pre-compute centroids -----------------------------------------------------------
        centroids_.resize(static_cast<std::size_t>(triCount));
        for (int i = 0; i < triCount; ++i) {
            const Vec3f v0 = verts[indices[i * 3 + 0]];
            const Vec3f v1 = verts[indices[i * 3 + 1]];
            const Vec3f v2 = verts[indices[i * 3 + 2]];
            centroids_[static_cast<std::size_t>(i)] = (v0 + v1 + v2) * (1.0f / 3.0f);
        }

        // --- pre-allocate worst-case node count (all leaves) ---------------------------------
        nodes.reserve(static_cast<std::size_t>(triCount * 2));

        // --- recursive build ----------------------------------------------------------------
        buildRec(0, triCount, verts, indices);

        // --- free temporary centroid storage ------------------------------------------------
        centroids_.clear();
        centroids_.shrink_to_fit();
    }

    // ─── Ray cast ──────────────────────────────────────────────────────────────────────────

    /// @brief  Ray-cast traversal.  Finds the closest triangle intersected by the ray.
    ///
    /// Uses an iterative stack-based traversal with ray-AABB pruning and Möller–Trumbore
    /// ray-triangle intersection.
    ///
    /// @param origin    Ray origin in world / mesh space.
    /// @param dir       Ray direction (need not be normalised; the `t` returned is
    ///                  parametric along this direction).
    /// @param maxT      Maximum ray distance (clamp).  Only hits with `t ∈ [0, maxT]`
    ///                  are considered.
    /// @param verts     Vertex array (same pointer passed to `build()`).
    /// @param indices   Triangle index array (same pointer passed to `build()`).
    /// @param outT      Receives the parametric distance of the closest hit.
    /// @param outNormal Receives the geometric normal at the hit point, oriented to face
    ///                  the ray origin.
    /// @param outTriIdx Receives the index (0-based) of the hit triangle.
    /// @return true if any triangle was hit within [0, maxT].
    [[nodiscard]] bool rayCast(const Vec3f& origin, const Vec3f& dir, float maxT,
                               const Vec3f* verts, const int* indices,
                               float& outT, Vec3f& outNormal, int& outTriIdx) const noexcept {
        if (nodes.empty())
            return false;

        float closestT = maxT;
        bool  hit      = false;

        // Iterative traversal stack.
        int stack[kBvhMaxStack];
        int stackPtr = 0;
        stack[stackPtr++] = 0; // root

        while (stackPtr > 0) {
            const int nodeIdx    = stack[--stackPtr];
            const BvhNode& node  = nodes[static_cast<std::size_t>(nodeIdx)];

            // Ray–AABB test (slab method).
            if (!rayAABB(origin, dir, closestT, node.bounds))
                continue;

            if (node.isLeaf()) {
                // Test each triangle in the leaf.
                for (int i = 0; i < node.triCount; ++i) {
                    const int triIdx = primitiveIndices_[
                        static_cast<std::size_t>(node.triStart + i)];

                    const int i0 = indices[triIdx * 3 + 0];
                    const int i1 = indices[triIdx * 3 + 1];
                    const int i2 = indices[triIdx * 3 + 2];

                    float t, u, v;
                    if (rayTriangle(origin, dir, closestT,
                                    verts[i0], verts[i1], verts[i2],
                                    t, u, v)) {
                        closestT   = t;
                        outTriIdx  = triIdx;
                        hit        = true;

                        // Geometric normal (counter-clockwise winding).
                        const Vec3f edge1 = verts[i1] - verts[i0];
                        const Vec3f edge2 = verts[i2] - verts[i0];
                        Vec3f n = edge1.cross(edge2);
                        // Orient normal to face the ray origin.
                        if (n.dot(dir) > 0.0f)
                            n = -n;
                        outNormal = n.normalized();
                    }
                }
            } else {
                // Internal node — push children (order does not matter because the
                // ray–AABB test already prunes misses; a production BVH might push
                // the nearer child last for slightly better early-out).
                // Bounds check the stack.
                if (stackPtr + 2 > kBvhMaxStack)
                    continue;
                stack[stackPtr++] = node.left;
                stack[stackPtr++] = node.right;
            }
        }

        if (hit)
            outT = closestT;
        return hit;
    }

    // ─── Overlap query ─────────────────────────────────────────────────────────────────────

    /// @brief  Find all triangles whose AABB overlaps `queryAABB`.
    ///
    /// Traverses the BVH and collects overlapping triangle indices into the provided
    /// output buffer.  The search is conservative at the node level (node AABB overlap);
    /// individual triangle AABBs are **not** tested, making this suitable for broad-phase
    /// collision pair discovery.
    ///
    /// @param queryAABB  The query volume.
    /// @param verts      Vertex array (unused in the current implementation but kept
    ///                   for API symmetry).
    /// @param indices    Triangle index array (unused in the current implementation but
    ///                   kept for API symmetry).
    /// @param triBuffer  Write cursor into `outTriIds` (in/out).  On entry it specifies
    ///                   the starting index in the buffer; on return it is advanced by
    ///                   `outCount` so callers can chain multiple queries.
    /// @param maxTris    Capacity of the `outTriIds` buffer.
    /// @param outTriIds  Output buffer filled with overlapping triangle indices.
    /// @param outCount   Number of triangle indices written to the buffer (≤ maxTris).
    /// @return true if at least one triangle overlap was found.
    [[nodiscard]] bool queryOverlap(const AABB& queryAABB,
                                    const Vec3f* /*verts*/,
                                    const int*   /*indices*/,
                                    int& triBuffer,
                                    int  maxTris,
                                    int* outTriIds,
                                    int& outCount) const noexcept {
        outCount = 0;
        if (nodes.empty())
            return false;

        int stack[kBvhMaxStack];
        int stackPtr = 0;
        stack[stackPtr++] = 0;

        while (stackPtr > 0) {
            const int nodeIdx    = stack[--stackPtr];
            const BvhNode& node  = nodes[static_cast<std::size_t>(nodeIdx)];

            if (!queryAABB.overlaps(node.bounds))
                continue;

            if (node.isLeaf()) {
                const int room = maxTris - (triBuffer + outCount);
                const int n    = (std::min)(node.triCount, room);
                for (int i = 0; i < n; ++i) {
                    outTriIds[triBuffer + outCount] =
                        primitiveIndices_[static_cast<std::size_t>(node.triStart + i)];
                    ++outCount;
                }
                if (outCount >= maxTris - triBuffer)
                    break; // buffer full
            } else {
                if (stackPtr + 2 > kBvhMaxStack)
                    continue;
                stack[stackPtr++] = node.left;
                stack[stackPtr++] = node.right;
            }
        }

        triBuffer += outCount;
        return outCount > 0;
    }

private:
    // ─── Internal state ────────────────────────────────────────────────────────────────────

    /// @brief  Re-ordered triangle permutation (leaf-contiguous after `build()`).
    ///         `primitiveIndices_[i]` = original triangle index for the i-th slot in
    ///         the BVH's leaf-contiguous layout.
    AlignedVector<int, 16> primitiveIndices_;

    /// @brief  Temporary centroid storage (alive only during `build()`).
    std::vector<Vec3f> centroids_;

    // ─── Internal helpers ──────────────────────────────────────────────────────────────────

    /// @brief  Slab method ray–AABB intersection test.
    /// @return true if the ray segment [0, maxT] intersects the box.
    static bool rayAABB(const Vec3f& origin, const Vec3f& dir,
                        float maxT, const AABB& bounds) noexcept {
        float tMin = 0.0f;
        float tMax = maxT;

        for (int axis = 0; axis < 3; ++axis) {
            const float invD = 1.0f / dir[axis];
            float t0 = (bounds.min[axis] - origin[axis]) * invD;
            float t1 = (bounds.max[axis] - origin[axis]) * invD;
            if (invD < 0.0f)
                std::swap(t0, t1);
            tMin = (std::max)(tMin, t0);
            tMax = (std::min)(tMax, t1);
            if (tMin > tMax)
                return false;
        }
        return true;
    }

    /// @brief  Möller–Trumbore ray–triangle intersection.
    ///
    /// @param origin  Ray origin.
    /// @param dir     Ray direction.
    /// @param maxT    Maximum valid hit distance (closest found so far).
    /// @param v0,v1,v2  Triangle vertices.
    /// @param[out] t  Hit distance along ray.
    /// @param[out] u  Barycentric u coordinate.
    /// @param[out] v  Barycentric v coordinate.
    /// @return true if the ray hits the triangle within [0, maxT].
    static bool rayTriangle(const Vec3f& origin, const Vec3f& dir, float maxT,
                            const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
                            float& t, float& u, float& v) noexcept {
        constexpr float kEpsilon = 1e-12f;

        const Vec3f edge1 = v1 - v0;
        const Vec3f edge2 = v2 - v0;

        const Vec3f h = dir.cross(edge2);
        const float a = edge1.dot(h);

        // Ray parallel to triangle.
        if (std::abs(a) < kEpsilon)
            return false;

        const float f = 1.0f / a;
        const Vec3f s = origin - v0;

        const float u_ = f * s.dot(h);
        if (u_ < 0.0f || u_ > 1.0f)
            return false;

        const Vec3f q = s.cross(edge1);
        const float v_ = f * dir.dot(q);
        if (v_ < 0.0f || u_ + v_ > 1.0f)
            return false;

        const float t_ = f * edge2.dot(q);
        if (t_ < 0.0f || t_ > maxT)
            return false;

        t = t_;
        u = u_;
        v = v_;
        return true;
    }

    /// @brief  Compute the AABB enclosing a single triangle.
    static AABB triangleAABB(const Vec3f& v0, const Vec3f& v1,
                             const Vec3f& v2) noexcept {
        AABB bb;
        bb.expand(v0);
        bb.expand(v1);
        bb.expand(v2);
        return bb;
    }

    /// @brief  Recursive BVH builder (median-split).
    ///
    /// Operates on the sub-range `[start, start + count)` of `primitiveIndices_`.
    /// The array is partitioned in-place so that after building, leaf triangles appear
    /// in contiguous blocks.
    ///
    /// @param start    Start index into `primitiveIndices_`.
    /// @param count    Number of triangles in this range.
    /// @param verts    Vertex array.
    /// @param indices  Triangle index array (3 per triangle).
    void buildRec(int start, int count,
                  const Vec3f* verts, const int* indices) noexcept {
        // ── compute node AABB ───────────────────────────────────────────────────────────────
        AABB bounds;
        for (int i = 0; i < count; ++i) {
            const int t = primitiveIndices_[static_cast<std::size_t>(start + i)];
            bounds.expand(verts[indices[t * 3 + 0]]);
            bounds.expand(verts[indices[t * 3 + 1]]);
            bounds.expand(verts[indices[t * 3 + 2]]);
        }

        const int nodeIdx = static_cast<int>(nodes.size());
        nodes.push_back(BvhNode{}); // placeholder

        // ── leaf ────────────────────────────────────────────────────────────────────────────
        if (count <= kBvhLeafSize) {
            nodes[static_cast<std::size_t>(nodeIdx)] =
                BvhNode{bounds, -1, -1, start, count};
            return;
        }

        // ── centroid bounds & split axis ────────────────────────────────────────────────────
        AABB centroidBounds;
        for (int i = 0; i < count; ++i) {
            centroidBounds.expand(
                centroids_[static_cast<std::size_t>(
                    primitiveIndices_[static_cast<std::size_t>(start + i)])]);
        }

        const Vec3f centroidExtents = centroidBounds.extents();
        int axis = 0;
        if (centroidExtents.y > centroidExtents.x) axis = 1;
        if (centroidExtents.z > centroidExtents[axis]) axis = 2;

        // Degenerate: all centroids coincident — make a leaf.
        if (centroidExtents[axis] < 1e-12f) {
            nodes[static_cast<std::size_t>(nodeIdx)] =
                BvhNode{bounds, -1, -1, start, count};
            return;
        }

        // ── median split (nth_element) ──────────────────────────────────────────────────────
        const int mid = count / 2;
        auto begin = primitiveIndices_.begin() + start;
        std::nth_element(begin, begin + mid, begin + count,
            [this, axis](int a, int b) noexcept {
                return centroids_[static_cast<std::size_t>(a)][axis]
                     < centroids_[static_cast<std::size_t>(b)][axis];
            });

        // ── recurse ─────────────────────────────────────────────────────────────────────────
        const int leftIdx = static_cast<int>(nodes.size());
        buildRec(start,       mid,            verts, indices);
        const int rightIdx = static_cast<int>(nodes.size());
        buildRec(start + mid, count - mid,    verts, indices);

        // ── internal node ───────────────────────────────────────────────────────────────────
        nodes[static_cast<std::size_t>(nodeIdx)] =
            BvhNode{bounds, leftIdx, rightIdx, -1, -1};
    }
};

} // namespace ciphyxs
