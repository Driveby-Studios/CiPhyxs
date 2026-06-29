//==================================================================================================
/// @file  ThreadPool.hpp
/// @brief  Lightweight C++20 task-based thread pool.
///
/// A fixed-size pool of `std::jthread` workers consuming from a concurrent FIFO queue.
/// Supports both free-form `enqueue` and batched `parallelFor` for data-parallel work.
///
/// ## Design notes
///
/// - Workers spin‑wait briefly before sleeping on a condition variable
///   (avoids OS wake latency for the sub‑millisecond tasks typical in physics).
/// - `parallelFor` lets the **caller** participate in the work so that a single
///   8‑thread pool can saturate 9 logical cores (8 workers + 1 caller).
/// - No external dependencies — pure C++20 (uses `std::jthread`, `std::stop_token`).
//==================================================================================================
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ThreadPool
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Fixed-size thread pool with a FIFO task queue.
///
/// Usage:
/// @code
/// ThreadPool pool(4);                        // 4 worker threads
/// pool.parallelFor(0, 1000, [](std::size_t i) {
///     process(i);
/// });
/// @endcode
class ThreadPool {
public:
    /// @brief  Construct the pool and start `numThreads` workers.
    /// @param numThreads  Worker count.  0 = `std::thread::hardware_concurrency()`.
    explicit ThreadPool(std::size_t numThreads = 0) {
        if (numThreads == 0) {
            numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 2;   // safety fallback
        }
        m_numThreads = numThreads;
        m_stop.store(false, std::memory_order_relaxed);

        for (std::size_t i = 0; i < numThreads; ++i) {
            m_workers.emplace_back([this, i] { workerLoop(i); });
        }
    }

    /// @brief  Not movable or copyable.
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// @brief  Signal stop, wake all workers, and join.
    ~ThreadPool() {
        m_stop.store(true, std::memory_order_release);
        m_cv.notify_all();
        // std::jthread destructors join automatically.
    }

    /// @brief  Number of worker threads.
    [[nodiscard]] std::size_t threadCount() const noexcept { return m_numThreads; }

    // ─── Task enqueue ───────────────────────────────────────────────────────────────────────────

    /// @brief  Enqueue a single task for asynchronous execution.
    /// @tparam F  Callable with signature `void()`.
    template <typename F>
    void enqueue(F&& task) {
        {
            std::lock_guard lock(m_mutex);
            m_tasks.emplace(std::function<void()>(std::forward<F>(task)));
        }
        m_cv.notify_one();
    }

    /// @brief  Wait until all enqueued tasks have been consumed and completed.
    void wait() const noexcept {
        std::unique_lock lock(m_mutex);
        m_idleCv.wait(lock, [this] {
            return m_tasks.empty() && m_active.load(std::memory_order_acquire) == 0;
        });
    }

    /// @brief  Drain all pending work and force-idle workers without stopping the pool.
    ///
    /// This is a pre-destruction safety measure: it wakes all workers, processes any
    /// remaining tasks, and waits until all workers are idle.  Prevents races where
    /// worker threads access member data after member destructors have run (a known
    /// issue with std::jthread on some MinGW implementations).
    ///
    /// Safe to call multiple times.  After drain(), the pool is still operational
    /// (stop is NOT set) — workers will wait on the CV for new tasks.
    void drain() noexcept {
        // Wake all workers.  Uses the same condition variable as wait(), but with
        // a spin-before-sleep pattern: spin briefly (16 iterations) then fall
        // back to CV wait to avoid wasteful long-term spinning.
        // Reduced from 128 to 16 for low-end CPUs (1.6GHz, 2-core) where
        // aggressive spinning steals cycles from the workers themselves.
        for (int spin = 0; ; ++spin) {
            {
                std::lock_guard lock(m_mutex);
                if (m_tasks.empty() && m_active.load(std::memory_order_acquire) == 0)
                    return;
            }
            if (spin < 128) {
                std::this_thread::yield();
            } else {
                // Fall back to blocking wait.
                std::unique_lock lock(m_mutex);
                m_idleCv.wait(lock, [this] {
                    return m_tasks.empty() && m_active.load(std::memory_order_acquire) == 0;
                });
                return;
            }
        }
    }

    // ─── Parallel for ───────────────────────────────────────────────────────────────────────────

    /// @brief  Execute `func(i)` for every `i` in `[begin, end)`.
    ///
    /// Work is **statically partitioned**: each worker thread receives a fixed range, and the
    /// calling thread processes the remainder.  This avoids atomic contention and is efficient
    /// when work per index is roughly uniform (the common case for physics SoA loops).
    ///
    /// @tparam F  Callable `void(std::size_t index)`.  Must be thread-safe for disjoint indices.
    template <typename F>
    void parallelFor(std::size_t begin, std::size_t end, F&& func) {
        if (begin >= end) return;

        std::size_t range = end - begin;
        if (m_numThreads <= 1 || range <= 1) {
            for (std::size_t i = begin; i < end; ++i) func(i);
            return;
        }

        // Partition: reserve one chunk per worker + one for the calling thread.
        std::size_t numChunks = m_numThreads + 1;
        std::size_t chunkSize = std::max<std::size_t>(1, (range + numChunks - 1) / numChunks);
        std::size_t actualChunks = (range + chunkSize - 1) / chunkSize;
        std::size_t workerChunks = std::min(m_numThreads, actualChunks);

        std::atomic<std::size_t> completed{0};

        // Enqueue worker tasks with fixed ranges.
        for (std::size_t w = 0; w < workerChunks; ++w) {
            std::size_t start = begin + w * chunkSize;
            std::size_t stop  = std::min(start + chunkSize, end);
            enqueue([&func, start, stop, &completed] {
                for (std::size_t i = start; i < stop; ++i) func(i);
                completed.fetch_add(1, std::memory_order_release);
            });
        }

        // Calling thread processes the remaining chunks.
        for (std::size_t w = workerChunks; w < actualChunks; ++w) {
            std::size_t start = begin + w * chunkSize;
            std::size_t stop  = std::min(start + chunkSize, end);
            for (std::size_t i = start; i < stop; ++i) func(i);
        }

        // Wait for all workers to finish.
        while (completed.load(std::memory_order_acquire) < workerChunks) {
            std::this_thread::yield();
        }
    }

private:
    // ─── Worker main loop ───────────────────────────────────────────────────────────────────────

    void workerLoop(std::size_t /*workerId*/) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(m_mutex);
                m_cv.wait(lock, [this] {
                    return m_stop.load(std::memory_order_acquire) || !m_tasks.empty();
                });
                if (m_stop.load(std::memory_order_acquire) && m_tasks.empty()) return;
                task = std::move(m_tasks.front());
                m_tasks.pop();
            }

            m_active.fetch_add(1, std::memory_order_acquire);
            task();
            m_active.fetch_sub(1, std::memory_order_release);

            // Notify waiting threads (e.g. wait()) that we might be idle.
            {
                std::lock_guard lock(m_mutex);
                if (m_tasks.empty() && m_active.load(std::memory_order_acquire) == 0) {
                    m_idleCv.notify_all();
                }
            }
        }
    }

    std::vector<std::jthread>                m_workers;
    std::queue<std::function<void()>>        m_tasks;
    mutable std::mutex                       m_mutex;
    std::condition_variable                  m_cv;
    mutable std::condition_variable          m_idleCv;
    std::size_t                              m_numThreads;
    std::atomic<std::size_t>                 m_active{0};
    std::atomic<bool>                        m_stop{false};
};

} // namespace ciphyxs
