//==================================================================================================
/// @file  IslandSolver.hpp
/// @brief  Island formation (union‑find) + per‑island parallel constraint solving.
///
/// ## Why islands?
///
/// A physics scene of N bodies typically partitions into **independent groups** connected by
/// contacts or joints.  These groups are *islands*.  Bodies in different islands cannot affect
/// each other within a single sub‑step, so each island can be solved concurrently.
///
/// ## Determinism
///
/// Islands are sorted by their smallest body handle.  Within each island, manifolds are
/// processed in the order they appear (which is deterministic given a deterministic
/// broadphase).  This yields identical results between single‑threaded and parallel execution.
//==================================================================================================
#pragma once

#include "ConstraintSolver.hpp"
#include "Joint.hpp"
#include "ThreadPool.hpp"
#include "../collision/Contact.hpp"
#include "../collision/Broadphase.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Island
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A group of rigid bodies that are coupled through contacts (or, in the future, joints).
///
/// Every body appears in at most one island.  Bodies with zero contacts are *singletons*
/// and are not stored in any island (they are trivially integrated).
struct Island {
    /// @brief  Handles of all bodies in this island (sorted ascending for determinism).
    std::vector<RigidBodyHandle> bodies;

    /// @brief  Indices into the global manifold array that belong to this island.
    std::vector<std::size_t> manifoldIndices;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// IslandSolver
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Builds a partition of bodies into solver islands and dispatches constraint solving.
///
/// ## Pipeline integration
///
/// ```
/// detectCollisions()          ← produces global manifold array
/// islandSolver.build(...)     ← union‑find over body–body contacts
/// islandSolver.solve(...)     ← sequential or parallel per‑island solving
/// integratePositions()        ← per‑body (always single‑threaded or over all bodies)
/// ```
class IslandSolver {
public:
    // ─── Island construction ────────────────────────────────────────────────────────────────────

    /// @brief  Partition the body graph into islands using union‑find.
    ///
    /// Both contacts and joints contribute to connectivity.  Bodies that are not connected
    /// to any other body are trivially excluded from the island set.
    ///
    /// @param manifolds   The current frame's contact manifolds.
    /// @param joints      All active joints (connect their two bodies).
    /// @param bodyCount   Total number of rigid bodies in the world (for DSU sizing).
    ///
    /// @note  O(N · α(N)) where N = number of connectivity edges.
    ///        The resultant islands are sorted deterministically.
    void build(const std::vector<ContactManifold>& manifolds,
               const std::vector<JointStorage>& joints,
               std::size_t bodyCount) {
        m_islands.clear();
        if (manifolds.empty() && joints.empty()) return;

        // ── 1. Union‑find ────────────────────────────────────────────────────────────
        DSU dsu(bodyCount);

        // Add edges from contact manifolds.
        for (const auto& m : manifolds) {
            if (m.bodyA < bodyCount && m.bodyB < bodyCount) {
                dsu.unite(m.bodyA, m.bodyB);
            }
        }

        // Add edges from joints.
        for (const auto& j : joints) {
            if (j.bodyA < bodyCount && j.bodyB < bodyCount) {
                dsu.unite(j.bodyA, j.bodyB);
            }
        }

        // ── 2. Group manifold indices by connected‑component root ─────────────────────
        //     root → index in m_islands
        std::vector<std::size_t> rootToIsland(bodyCount, std::size_t(-1));

        for (std::size_t i = 0; i < manifolds.size(); ++i) {
            std::size_t root = dsu.find(manifolds[i].bodyA);
            if (rootToIsland[root] == std::size_t(-1)) {
                rootToIsland[root] = m_islands.size();
                m_islands.emplace_back();
            }
            m_islands[rootToIsland[root]].manifoldIndices.push_back(i);
        }

        // ── 3. Collect unique body handles per island ─────────────────────────────────
        std::vector<bool> visited(bodyCount, false);

        for (auto& island : m_islands) {
            // Gather unique bodies from all manifolds in this island.
            for (auto idx : island.manifoldIndices) {
                const auto& m = manifolds[idx];
                if (!visited[m.bodyA]) {
                    visited[m.bodyA] = true;
                    island.bodies.push_back(m.bodyA);
                }
                if (!visited[m.bodyB]) {
                    visited[m.bodyB] = true;
                    island.bodies.push_back(m.bodyB);
                }
            }
            // Sort for deterministic processing order.
            std::sort(island.bodies.begin(), island.bodies.end());
            // Clear visited flags for the next island.
            for (auto b : island.bodies) visited[b] = false;
        }

        // ── 4. Sort islands by first body handle for determinism ──────────────────────
        std::sort(m_islands.begin(), m_islands.end(),
                  [](const Island& a, const Island& b) {
                      return a.bodies[0] < b.bodies[0];
                  });
    }

    /// @brief  Build islands from broadphase **pairs** and joints (before narrowphase).
    ///
    /// This variant forms islands from the raw broadphase pair list rather than from
    /// contact manifolds, enabling the per-island narrowphase → solve pipeline.
    /// Each pair connects two bodies that form an edge in the connectivity graph.
    ///
    /// @param pairs      Broadphase body-body candidate pairs.
    /// @param joints     All active joints.
    /// @param bodyCount  Total number of rigid bodies.
    void buildFromPairs(const std::vector<BroadphasePair>& pairs,
                        const std::vector<JointStorage>& joints,
                        std::size_t bodyCount) {
        m_islands.clear();
        if (bodyCount == 0) return;

        // ── 1. Union‑find ────────────────────────────────────────────────────────────
        DSU dsu(bodyCount);

        for (const auto& p : pairs) {
            if (p.bodyA < bodyCount && p.bodyB < bodyCount) {
                dsu.unite(p.bodyA, p.bodyB);
            }
        }

        for (const auto& j : joints) {
            if (j.bodyA < bodyCount && j.bodyB < bodyCount) {
                dsu.unite(j.bodyA, j.bodyB);
            }
        }

        // ── 2. Group ALL bodies by connected‑component root ──────────────────────────
        //     Every body (including isolated ones) is placed into an island.  Isolated
        //     bodies become size‑1 islands, which allows per‑island CCD processing.
        std::unordered_map<std::size_t, std::size_t> rootToIsland;

        for (std::size_t b = 0; b < bodyCount; ++b) {
            std::size_t root = dsu.find(b);
            auto it = rootToIsland.find(root);
            if (it == rootToIsland.end()) {
                it = rootToIsland.emplace(root, m_islands.size()).first;
                m_islands.emplace_back();
            }
            m_islands[it->second].bodies.push_back(b);
        }

        // ── 3. Sort bodies within each island for determinism ────────────────────────
        for (auto& island : m_islands) {
            std::sort(island.bodies.begin(), island.bodies.end());
        }

        // ── 4. Sort islands by first body handle ─────────────────────────────────────
        if (!m_islands.empty()) {
            std::sort(m_islands.begin(), m_islands.end(),
                      [](const Island& a, const Island& b) {
                          return a.bodies[0] < b.bodies[0];
                      });
        }
    }

    // ─── Solving ────────────────────────────────────────────────────────────────────────────────

    /// @brief  Solve all islands **sequentially** on the calling thread.
    ///
    /// @param dt         Fixed sub‑step duration.
    /// @param manifolds  Global manifold array (mutated for warm‑start cache).
    /// @param bodies     SoA body storage (velocities are modified).
    /// @param config     Solver parameters.
    void solve(float dt,
               std::vector<ContactManifold>& manifolds,
               RigidBodyStorage& bodies,
               const SolverConfig& config) const {

        for (const auto& island : m_islands) {
            solveIsland(dt, island, manifolds, bodies, config);
        }
    }

    /// @brief  Solve all islands **in parallel** via the thread pool.
    ///
    /// Each island is solved as an independent task.  This is safe because islands
    /// reference disjoint sets of bodies.
    ///
    /// @param dt         Fixed sub‑step duration.
    /// @param manifolds  Global manifold array (mutated for warm‑start cache).
    /// @param bodies     SoA body storage (velocities are modified).
    /// @param config     Solver parameters.
    /// @param pool       Thread pool for task dispatch.
    void solveParallel(float dt,
                       std::vector<ContactManifold>& manifolds,
                       RigidBodyStorage& bodies,
                       const SolverConfig& config,
                       ThreadPool& pool) const {

        if (m_islands.size() <= 1) {
            // No benefit from parallelism — run inline.
            solve(dt, manifolds, bodies, config);
            return;
        }

        // Capture everything by reference; each island is independent.
        pool.parallelFor(static_cast<std::size_t>(0), m_islands.size(),
                         [&, dt](std::size_t idx) {
                             solveIsland(dt, m_islands[idx], manifolds, bodies, config);
                         });
    }

    /// @brief  Solve a **single island** by index.
    ///
    /// This is the core per-island solve primitive used by both the sequential/parallel
    /// solve methods and the task-graph per-island sub-DAG pipeline.
    ///
    /// Safe to call from multiple threads concurrently (each island references
    /// disjoint body handles and disjoint manifold indices).
    ///
    /// @param dt         Fixed sub‑step duration.
    /// @param islandIdx  Index into the `islands()` array.
    /// @param manifolds  Global manifold array (mutated for warm‑start cache).
    /// @param bodies     SoA body storage (velocities are modified).
    /// @param config     Solver parameters.
    void solveOne(float dt,
                  std::size_t islandIdx,
                  std::vector<ContactManifold>& manifolds,
                  RigidBodyStorage& bodies,
                  const SolverConfig& config) const {
        if (islandIdx >= m_islands.size()) return;
        solveIsland(dt, m_islands[islandIdx], manifolds, bodies, config);
    }

    // ─── Access ─────────────────────────────────────────────────────────────────────────────────

    /// @brief  Islands built by the most recent `build()` call.
    [[nodiscard]] const std::vector<Island>& islands() const noexcept { return m_islands; }

    /// @brief  True after `build()` found at least one island.
    [[nodiscard]] bool hasIslands() const noexcept { return !m_islands.empty(); }

    /// @brief  Clear all island data.
    void clear() noexcept { m_islands.clear(); }

private:
    // ─── Disjoint Set Union (union‑find) ─────────────────────────────────────────────────────────
    struct DSU {
        std::vector<std::size_t> parent;
        std::vector<std::uint8_t> rank;

        explicit DSU(std::size_t n) {
            parent.resize(n);
            rank.resize(n, 0);
            for (std::size_t i = 0; i < n; ++i) parent[i] = i;
        }

        std::size_t find(std::size_t x) noexcept {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];  // path halving
                x = parent[x];
            }
            return x;
        }

        void unite(std::size_t a, std::size_t b) noexcept {
            a = find(a);
            b = find(b);
            if (a == b) return;
            if (rank[a] < rank[b]) std::swap(a, b);
            parent[b] = a;
            if (rank[a] == rank[b]) ++rank[a];
        }
    };

    // ─── Per‑island solve helper ────────────────────────────────────────────────────────────────

    /// @brief  Solve a single island by extracting its manifolds into a temporary vector
    ///         and delegating to `ConstraintSolver::solve()`.
    ///
    /// We copy manifold data into a local vector rather than reordering the global array
    /// to preserve the original manifold order when the island count is small (common case).
    /// After solving, warm‑start impulses are written back to the global array.
    static void solveIsland(float dt,
                            const Island& island,
                            std::vector<ContactManifold>& allManifolds,
                            RigidBodyStorage& bodies,
                            const SolverConfig& config) {

        if (island.manifoldIndices.empty()) return;

        // Build a contiguous working vector for this island.
        // This also warms up the cache for the solver.
        std::vector<ContactManifold> local;
        local.reserve(island.manifoldIndices.size());
        for (auto idx : island.manifoldIndices) {
            local.push_back(allManifolds[idx]);
        }

        // Run the sequential‑impulse solver on this island's manifolds.
        ConstraintSolver solver;
        solver.solve(dt, local, bodies, config);

        // Write warm‑start impulses back to the global manifolds.
        for (std::size_t i = 0; i < island.manifoldIndices.size(); ++i) {
            allManifolds[island.manifoldIndices[i]] = std::move(local[i]);
        }
    }

    std::vector<Island> m_islands;
};

} // namespace ciphyxs
