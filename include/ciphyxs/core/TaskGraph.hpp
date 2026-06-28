//==================================================================================================
/// @file  TaskGraph.hpp
/// @brief  Lightweight Directed-Acyclic-Graph task scheduler for pipeline parallelism.
///
/// ## Design
///
/// Build a DAG of named task nodes with explicit dependencies, then execute the entire graph
/// on a ThreadPool.  A task becomes eligible as soon as **all** its dependencies complete.
/// The scheduler launches ready tasks in **construction order** for determinism.
///
/// ## Why a task graph instead of parallelFor?
///
/// A simple `parallelFor` works when work is uniform and independent.  Physics pipelines have
/// complex dependencies:
///
///   - Broadphase must finish before narrowphase.
///   - Narrowphase must finish before solving.
///   - **But** narrowphase for island A and island B can run concurrently.
///   - Solving for island A has no dependency on solving for island B.
///
/// A task graph captures these relationships naturally, enabling **pipeline parallelism**
/// (like PhysX's TaskGraph or Chaos's task scheduler) without global barriers between stages.
///
/// ## Determinism
///
/// Ready tasks are launched in ascending TaskId order (construction order).  Since the graph
/// is built deterministically every frame, execution order is reproducible across runs.
///
/// ## Usage
///
/// @code
///   TaskGraph graph;
///
///   auto readInput = graph.add("ReadInput", {}, [&]{ ... });
///   auto physics   = graph.add("PhysicsStep", {readInput}, [&]{ ... });
///   auto render    = graph.add("Render", {physics}, [&]{ ... });
///
///   graph.execute(pool);   // blocks until all tasks complete
///   graph.clear();         // ready for next frame
/// @endcode
//==================================================================================================
#pragma once

#include "ThreadPool.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// TaskId
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Opaque handle returned by TaskGraph::add().  Use as a dependency argument.
using TaskId = std::uint32_t;

/// @brief  Sentinel representing "no task" or an invalid dependency.
inline constexpr TaskId kInvalidTaskId = ~TaskId{0};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// TaskGraph
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Directed-acyclic-graph task scheduler.
///
/// Build the graph by adding task nodes with explicit dependency lists, then call execute()
/// to run the entire DAG on a ThreadPool.  The graph is reusable — call clear() then rebuild
/// for the next frame.
///
/// @note  Not thread-safe for modification (add/clear must be called from one thread).
///        execute() is thread-safe (called from one thread, tasks run on the pool).
class TaskGraph {
public:
    TaskGraph() = default;

    /// @brief  Not copyable, but movable (for use as a member of PhysicsWorld).
    TaskGraph(const TaskGraph&) = delete;
    TaskGraph& operator=(const TaskGraph&) = delete;

    TaskGraph(TaskGraph&& other) noexcept
        : m_nodes(std::move(other.m_nodes))
        , m_remaining(std::move(other.m_remaining))
        , m_remainingSize(std::exchange(other.m_remainingSize, 0))
        , m_execRemaining(std::move(other.m_execRemaining))
        , m_profileMutex()  // mutex is not movable — re-initialised
        , m_profilingEnabled(std::exchange(other.m_profilingEnabled, false))
        , m_profileEvents(std::move(other.m_profileEvents))
    {}

    TaskGraph& operator=(TaskGraph&& other) noexcept {
        if (this != &other) {
            m_nodes              = std::move(other.m_nodes);
            m_remaining          = std::move(other.m_remaining);
            m_remainingSize      = std::exchange(other.m_remainingSize, 0);
            m_execRemaining      = std::move(other.m_execRemaining);
            m_profilingEnabled   = std::exchange(other.m_profilingEnabled, false);
            m_profileEvents      = std::move(other.m_profileEvents);
            // m_profileMutex is intentionally NOT moved (mutex is not movable)
        }
        return *this;
    }

    // ─── Graph construction ──────────────────────────────────────────────────────────────────────

    /// @brief  Add a task node to the graph.
    ///
    /// @tparam F  Invocable `void()`.  The callable is moved into the node.
    /// @param  name   Human-readable label (for debugging / profiling).
    /// @param  deps   Span of TaskIds that must complete before this task starts.
    ///                May be empty (root task).
    /// @param  callable  The work to perform.
    /// @return TaskId that can be passed as a dependency to other tasks.
    template <typename F>
        requires std::invocable<F&>
    TaskId add(std::string_view name, std::span<const TaskId> deps, F&& callable) {
        TaskId id = static_cast<TaskId>(m_nodes.size());
        auto& node = m_nodes.emplace_back();
        node.name = std::string(name);
        node.work = std::move(callable);
        node.originalCount = static_cast<int>(deps.size());
        node.dependents.reserve(4); // small vector optimization hint

        // Record this node as a dependent of each dependency.
        for (TaskId dep : deps) {
            if (dep < id) {
                m_nodes[static_cast<std::size_t>(dep)].dependents.push_back(id);
            }
        }

        return id;
    }

    /// @brief  Reset the graph for reuse.  Invalidates all previously-returned TaskIds.
    void clear() {
        m_nodes.clear();
        m_remaining.reset(); // force reallocation on next execute()
    }

    /// @brief  Add a barrier (join) node that waits for all given tasks to complete.
    ///
    /// The barrier has no work body — it simply blocks its dependents until every task
    /// in `deps` has finished.  This is useful for synchronising parallel fan-out
    /// before a serial phase (e.g., wait for all per-island solves before post-solve).
    ///
    /// @param  name   Human-readable label (for debugging / profiling).
    /// @param  deps   Span of TaskIds that must complete before the barrier fires.
    /// @return TaskId of the barrier (pass as a dependency to downstream tasks).
    TaskId addBarrier(std::string_view name, std::span<const TaskId> deps) {
        TaskId id = static_cast<TaskId>(m_nodes.size());
        auto& node = m_nodes.emplace_back();
        node.name = std::string(name);
        node.work = nullptr; // no-op; completeNode fires immediately when deps are met
        node.originalCount = static_cast<int>(deps.size());
        node.dependents.reserve(deps.size());

        for (TaskId dep : deps) {
            if (dep < id) {
                m_nodes[static_cast<std::size_t>(dep)].dependents.push_back(id);
            }
        }

        return id;
    }

    /// @brief  Number of registered task nodes.
    [[nodiscard]] std::size_t size() const noexcept { return m_nodes.size(); }

    /// @brief  True when no tasks have been registered.
    [[nodiscard]] bool empty() const noexcept { return m_nodes.empty(); }

    // ─── Execution ───────────────────────────────────────────────────────────────────────────────

    // ─── Profiling ──────────────────────────────────────────────────────────────────────────────

    /// @brief  A single task execution record produced when profiling is enabled.
    struct ProfileEvent {
        std::string       name;        ///< Task node name.
        double            elapsedMs;   ///< Wall-clock execution time in milliseconds.
        std::thread::id   threadId;    ///< Thread that executed this task.
    };

    /// @brief  Enable or disable task-level profiling.
    ///         When enabled, every call to executeNode() records timing data.
    ///         Profiling is disabled by default (zero overhead when off).
    void setProfilingEnabled(bool enabled) noexcept { m_profilingEnabled = enabled; }

    /// @brief  True when task profiling is active.
    [[nodiscard]] bool isProfilingEnabled() const noexcept { return m_profilingEnabled; }

    /// @brief  Profile events from the most recent execute() call.
    ///         Each event corresponds to one executed task node.
    ///         Events are in completion order (not deterministic).
    [[nodiscard]] const std::vector<ProfileEvent>& profileEvents() const noexcept {
        return m_profileEvents;
    }

    /// @brief  Clear all recorded profile events.
    void clearProfileEvents() noexcept { m_profileEvents.clear(); }

    // ─── Visualization ────────────────────────────────────────────────────────────────────────────

    /// @brief  Export the task graph as a Graphviz DOT string for offline visualization.
    ///
    /// The output uses a top-to-bottom hierarchical layout with labelled nodes and directed edges.
    /// Root tasks (zero dependencies) are highlighted with a light blue fill; all other nodes use
    /// light gray.  Special characters in node names (quotes, backslashes) are sanitised.
    ///
    /// ## Usage
    ///
    /// @code
    ///   std::ofstream file("taskgraph.dot");
    ///   file << graph.toDot();
    ///   // Render:  dot -Tsvg taskgraph.dot -o taskgraph.svg
    /// @endcode
    ///
    /// @return  A string containing the full DOT digraph definition.
    ///
    /// @note  The graph must be fully built (all add()/addBarrier() calls made) before calling
    ///        this method.  Calling it mid-construction produces a partial graph.
    [[nodiscard]] std::string toDot() const {
        static constexpr std::string_view kNodeShape =
            "shape=box, style=rounded, fontname=courier";

        std::string dot;
        dot.reserve(512 + m_nodes.size() * 160);

        // ── Header ────────────────────────────────────────────────────────────────────────────
        dot += "digraph TaskGraph {\n";
        dot += "    rankdir=TB;\n";
        dot += "    node [" + std::string(kNodeShape) + "];\n";
        dot += "    graph [dpi=150, bgcolor=transparent, splines=polyline];\n";
        dot += "    edge [arrowhead=vee, arrowsize=0.8];\n\n";

        // ── Node declarations ─────────────────────────────────────────────────────────────────
        for (std::size_t i = 0; i < m_nodes.size(); ++i) {
            const auto& node = m_nodes[i];
            std::string label = sanitiseDot(node.name);

            // Root tasks (zero deps) are coloured differently.
            if (node.originalCount == 0) {
                dot += "    \"" + label + "\" [style=filled, fillcolor=lightblue];\n";
            } else {
                dot += "    \"" + label + "\" [style=filled, fillcolor=lightgray];\n";
            }
        }

        dot += '\n';

        // ── Edge declarations ─────────────────────────────────────────────────────────────────
        for (std::size_t i = 0; i < m_nodes.size(); ++i) {
            const auto& node = m_nodes[i];
            std::string src = sanitiseDot(node.name);

            for (TaskId depId : node.dependents) {
                std::string dst = sanitiseDot(m_nodes[static_cast<std::size_t>(depId)].name);
                dot += "    \"" + src + "\" -> \"" + dst + "\";\n";
            }
        }

        dot += "}\n";
        return dot;
    }

    /// @brief  Execute the graph on the given thread pool.
    ///
    /// Blocks the calling thread until **all** tasks complete.  The calling thread does **not**
    /// participate in task execution — it spins briefly checking completion.  This is by design:
    /// the calling thread is typically the main simulation thread and should not be saturated
    /// with physics work.
    ///
    /// Safe to call multiple times (after clear() + rebuild).
    ///
    /// @param pool  ThreadPool to dispatch tasks onto.
    void execute(ThreadPool& pool) {
        std::size_t n = m_nodes.size();
        if (n == 0) return;

        // ── Allocate runtime atomic counters ────────────────────────────────────────────────
        if (!m_remaining || m_remainingSize < n) {
            m_remaining = std::make_unique<std::atomic<int>[]>(n);
            m_remainingSize = n;
        }

        // Reset counters: each node's remaining = its dependency count.
        for (std::size_t i = 0; i < n; ++i) {
            m_remaining[i].store(m_nodes[i].originalCount, std::memory_order_relaxed);
        }
        m_execRemaining->store(static_cast<int>(n), std::memory_order_release);

        // Clear profile events from the previous execution.
        if (m_profilingEnabled) {
            m_profileEvents.clear();
        }

        // ── Enqueue root tasks (zero dependencies) ─────────────────────────────────────────
        // Roots are enqueued in construction order for determinism.
        for (TaskId i = 0; i < static_cast<TaskId>(n); ++i) {
            if (m_nodes[static_cast<std::size_t>(i)].originalCount == 0) {
                if (m_nodes[static_cast<std::size_t>(i)].work) {
                    pool.enqueue([this, i, &pool]() { executeNode(i, pool); });
                } else {
                    // No-op root — complete inline.
                    completeNode(i, pool);
                }
            }
        }

        // ── Wait for completion ────────────────────────────────────────────────────────────
        while (m_execRemaining->load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

private:
    // ─── Internal node ───────────────────────────────────────────────────────────────────────────

    /// @brief  Persistent graph node (remains between frames, rebuilt via clear()+add()).
    struct Node {
        std::string                         name;            ///< Debug label.
        std::function<void()>               work;            ///< The task body (empty = sync-only).
        std::vector<TaskId>                 dependents;      ///< Nodes that depend on this one.
        int                                 originalCount{0};///< Dependency count (fixed at add()).
    };

    // ─── Internal helpers ────────────────────────────────────────────────────────────────────────

    /// @brief  Sanitise a node name for safe embedding in a DOT string.
    ///         Replaces double-quotes and backslashes with underscores.
    [[nodiscard]] static std::string sanitiseDot(const std::string& name) {
        std::string result = name;
        for (auto& ch : result) {
            if (ch == '"' || ch == '\\') {
                ch = '_';
            }
        }
        return result;
    }

    /// @brief  Run one task node, then signal dependents.
    ///         Optionally records profiling data when enabled.
    void executeNode(TaskId id, ThreadPool& pool) {
        if (m_profilingEnabled) {
            auto start = std::chrono::high_resolution_clock::now();
            m_nodes[static_cast<std::size_t>(id)].work();
            auto end = std::chrono::high_resolution_clock::now();

            double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
            {
                std::lock_guard<std::mutex> lock(m_profileMutex);
                m_profileEvents.push_back({
                    m_nodes[static_cast<std::size_t>(id)].name,
                    elapsedMs,
                    std::this_thread::get_id()
                });
            }
        } else {
            m_nodes[static_cast<std::size_t>(id)].work();
        }
        completeNode(id, pool);
    }

    /// @brief  Signal dependents that this node is done, then decrement the completion counter.
    ///
    /// Call this instead of executeNode when the node has no work body, or after work completes.
    void completeNode(TaskId id, ThreadPool& pool) {
        const Node& node = m_nodes[static_cast<std::size_t>(id)];

        // Enqueue each dependent whose dependency counter just reached zero.
        // Iteration order follows the dependent list (built in add() order) for determinism.
        for (TaskId depId : node.dependents) {
            int prev = m_remaining[static_cast<std::size_t>(depId)]
                           .fetch_sub(1, std::memory_order_acq_rel);
            if (prev == 1) {
                // All dependencies satisfied — this dependent is ready to run.
                auto& depNode = m_nodes[static_cast<std::size_t>(depId)];
                if (depNode.work) {
                    pool.enqueue([this, depId, &pool]() { executeNode(depId, pool); });
                } else {
                    // Empty work function — signal completion directly.
                    completeNode(depId, pool);
                }
            }
        }

        // Decrement the global completion counter.
        // This MUST happen after enqueuing dependents, so the counter never reaches
        // zero before all work has been submitted to the thread pool.
        m_execRemaining->fetch_sub(1, std::memory_order_release);
    }

    // ─── Member variables ────────────────────────────────────────────────────────────────────────

    std::vector<Node>                       m_nodes;           ///< Persistent node storage.
    std::unique_ptr<std::atomic<int>[]>     m_remaining;       ///< Per-node dep counters (heap).
    std::size_t                             m_remainingSize = 0;///< Allocated size of m_remaining.
    std::unique_ptr<std::atomic<int>>       m_execRemaining{std::make_unique<std::atomic<int>>(0)}; ///< Global completion counter.

    // ─── Profiling state ─────────────────────────────────────────────────────────────────────────
    mutable std::mutex                      m_profileMutex;       ///< Must be declared before m_profilingEnabled
                                                                   ///< (initialised first in move ctor).
    bool                                    m_profilingEnabled = false;
    std::vector<ProfileEvent>               m_profileEvents;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// TaskGraphPipeline  —  convenience builder for linear pipeline stages
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Convenience helper for building a simple linear pipeline with TaskGraph.
///
/// Each stage depends on the previous stage, and parallel fan-out/fan-in is supported
/// within each stage.
///
/// Usage:
/// @code
///   TaskGraph graph;
///   auto pipe = TaskGraphPipeline(graph);
///
///   auto stage1 = pipe.stage("Stage1", [&]{ ... });
///   auto stage2 = pipe.stage("Stage2", [&]{ ... });
///
///   // Parallel work that all depend on stage2 and feed into stage3
///   auto t1 = graph.add("Task1", {stage2}, [&]{ ... });
///   auto t2 = graph.add("Task2", {stage2}, [&]{ ... });
///
///   auto stage3 = pipe.stage("Stage3", [&]{ ... });
/// @endcode
class TaskGraphPipeline {
public:
    /// @brief  Construct a pipeline builder backed by the given graph.
    explicit TaskGraphPipeline(TaskGraph& graph) noexcept : m_graph(&graph) {}

    /// @brief  Add a linear pipeline stage that depends on the previous stage (if any).
    /// @return TaskId of the new stage (usable as a dependency for other tasks).
    template <typename F>
        requires std::invocable<F&>
    TaskId stage(std::string_view name, F&& callable) {
        TaskId id;
        if (m_prevId == kInvalidTaskId) {
            id = m_graph->add(name, std::span<const TaskId>{}, std::forward<F>(callable));
        } else {
            TaskId deps[] = {m_prevId};
            id = m_graph->add(name, deps, std::forward<F>(callable));
        }
        m_prevId = id;
        return id;
    }

    /// @brief  The TaskId of the most recently added stage.
    [[nodiscard]] TaskId lastStageId() const noexcept { return m_prevId; }

private:
    TaskGraph* m_graph;
    TaskId     m_prevId = kInvalidTaskId;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// TaskBarrier  —  simple fork-join synchronization primitive
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A reusable barrier (count-down latch) for coordinating parallel work within a task.
///
/// Unlike `std::barrier`, this has no completion phase — it simply blocks the calling thread
/// until N participants have called `arrive()`.  Useful for manual fork-join within a single
/// task (e.g., distributing sub-work across threads inside a task graph node).
///
/// Usage:
/// @code
///   // Inside a task graph node:
///   TaskBarrier barrier(4);   // main thread + 3 workers
///   for (int i = 0; i < 3; ++i)
///       pool.enqueue([&] { doWork(i); barrier.arrive(); });
///   doWork(3);
///   barrier.wait();           // all 4 arrived
/// @endcode
class TaskBarrier {
public:
    /// @brief  Construct with the participant count.
    explicit TaskBarrier(int count) noexcept : m_count(count) {}

    /// @brief  Reset for reuse with a new count.
    void reset(int count) noexcept {
        m_count.store(count, std::memory_order_release);
    }

    /// @brief  Signal arrival at the barrier.  Decrements the internal counter.
    void arrive() noexcept {
        m_count.fetch_sub(1, std::memory_order_acq_rel);
    }

    /// @brief  Block until all participants have arrived.
    void wait() noexcept {
        while (m_count.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

    /// @brief  Convenience: arrive and wait in one call (for the last participant).
    void arriveAndWait() noexcept {
        arrive();
        wait();
    }

private:
    std::atomic<int>   m_count;
};

} // namespace ciphyxs
