//==================================================================================================
/// @file  StressTestBase.hpp
/// @brief  Common interface, utilities, and null debug renderer for CiPhyxs stress tests.
///
/// All stress tests derive from StressTest and implement:
///   - setup()    — build the world, bodies, shapes
///   - run()      — execute the simulation loop
///   - verify()   — check results and compute a deterministic hash
///   - name()     — human-readable test name
///
/// Coding standards:
///   - alignas(16) types throughout
///   - AlignedAllocator for heap storage (no std::vector for sim data)
///   - SoA (RigidBodyStorage) for simulation data
///   - Fixed seeds for all RNG
///   - IDebugRenderer used for visualization (null impl by default)
//==================================================================================================
#pragma once

#include "../include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

namespace ciphyxs {
namespace stresstest {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// Fixed seed for all RNG in stress tests (determinism).
inline constexpr std::uint32_t kStressFixedSeed = 42u;

/// Fixed timestep used across all stress tests.
inline constexpr float kFixedDt = 1.0f / 60.0f;

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Printf-style helpers for consistent output
// ────────────────────────────────────────────────────────────────────────────────────────────────

inline void printHeader(const char* name) noexcept {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║  Stress Test: %-30s ║\n", name);
    printf("╚══════════════════════════════════════════════╝\n\n");
}

inline void printPass() noexcept {
    printf("   ✓ PASS\n");
}

inline void printFail(const char* reason) noexcept {
    printf("   ✗ FAIL: %s\n", reason);
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Simple hash function for deterministic state verification
// Uses a fixed-seed FNV-1a variant
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Compute a deterministic 64-bit hash of a buffer.
inline std::uint64_t computeHash(const void* data, std::size_t bytes) noexcept {
    constexpr std::uint64_t kFNVOffsetBasis = 14695981039346656037ULL;
    constexpr std::uint64_t kFNVPrime       = 1099511628211ULL;
    std::uint64_t h = kFNVOffsetBasis;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= kFNVPrime;
    }
    return h;
}

/// @brief  Hash the positions and velocities from a RigidBodyStorage.
inline std::uint64_t hashBodyState(const RigidBodyStorage& bodies) noexcept {
    std::uint64_t h = 0;
    // Hash positions (each Vec3f = 12 bytes, but struct has padding to 16)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        h ^= computeHash(&bodies.positions[i], sizeof(Vec3f));
        h *= 1099511628211ULL;
        h ^= computeHash(&bodies.linearVelocities[i], sizeof(Vec3f));
        h *= 1099511628211ULL;
        h ^= computeHash(&bodies.angularVelocities[i], sizeof(Vec3f));
        h *= 1099511628211ULL;
    }
    return h;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Null debug renderer (satisfies IDebugRenderer, produces no output)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A no-op IDebugRenderer that discards all draw calls.
/// Used to satisfy the stress-tests' requirement to use IDebugRenderer without
/// requiring an actual graphics context.
struct NullDebugRenderer final : public IDebugRenderer {
    void drawLine(const Vec3f&, const Vec3f&, const Color&) noexcept override {}
    void drawTriangle(const Vec3f&, const Vec3f&, const Vec3f&,
                      const Color&) noexcept override {}
    void drawPoint(const Vec3f&, const Color&) noexcept override {}
    void drawArrow(const Vec3f&, const Vec3f&, float,
                   const Color&) noexcept override {}
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Minimal deterministic pseudo-random number generator (xorshift32)
// Uses a fixed seed; no global state.
// ────────────────────────────────────────────────────────────────────────────────────────────────

struct alignas(16) FixedRng {
    std::uint32_t state;

    explicit constexpr FixedRng(std::uint32_t seed = kStressFixedSeed) noexcept
        : state(seed)
    {
        if (state == 0) state = 1;  // xorshift cannot have zero state
    }

    /// @brief  Return a pseudo-random uint32.
    std::uint32_t next() noexcept {
        std::uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    /// @brief  Return a float in [0, 1).
    float nextFloat() noexcept {
        return static_cast<float>(next() & 0x00FFFFFF) / 16777216.0f;
    }

    /// @brief  Return a float in [lo, hi).
    float range(float lo, float hi) noexcept {
        return lo + nextFloat() * (hi - lo);
    }
};
static_assert(alignof(FixedRng) >= 16, "FixedRng must be 16-byte aligned");

// ────────────────────────────────────────────────────────────────────────────────────────────────
// StressTest — abstract base for all stress tests
// ────────────────────────────────────────────────────────────────────────────────────────────────

struct StressTest {
    virtual ~StressTest() = default;

    /// @brief  Human-readable test name (for console output).
    virtual const char* name() const noexcept = 0;

    /// @brief  Build the world and populate bodies/shapes.
    virtual void setup() = 0;

    /// @brief  Advance the simulation.
    virtual void run() = 0;

    /// @brief  Validate results and return true on pass.
    virtual bool verify() = 0;

    /// @brief  Return a deterministic hash of the final state.
    virtual std::uint64_t resultHash() const noexcept = 0;

    /// @brief  Helper: run setup + run + verify and report.
    bool execute() noexcept {
        printHeader(name());
        setup();
        run();
        bool ok = verify();
        if (ok) {
            printPass();
        }
        printf("   Result hash: 0x%016llX\n",
               static_cast<unsigned long long>(resultHash()));
        printf("   Elapsed: %lld ms\n\n",
               static_cast<long long>(elapsedMs()));
        return ok;
    }

    /// @brief  Elapsed wall-clock time for the run phase (ms).
    virtual long long elapsedMs() const noexcept = 0;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Timer helper
// ────────────────────────────────────────────────────────────────────────────────────────────────

struct Stopwatch {
    using Clock = std::chrono::steady_clock;

    void start() noexcept { m_start = Clock::now(); }
    void stop()  noexcept { m_end   = Clock::now(); }

    [[nodiscard]] long long ms() const noexcept {
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count();
    }

private:
    Clock::time_point m_start;
    Clock::time_point m_end;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SoA-compatible scratch vector (uses AlignedAllocator)
// ────────────────────────────────────────────────────────────────────────────────────────────────

template <typename T, std::size_t Align = 16>
using ScratchVec = std::vector<T, AlignedAllocator<T, Align>>;

} // namespace stresstest
} // namespace ciphyxs
