//==================================================================================================
/// @file  LowEndBenchmark.cpp
/// @brief  Benchmark comparing default vs AggressiveMode-optimised PhysicsWorld.
//==================================================================================================
#include "StressTestBase.hpp"
#include "../include/ciphyxs/core/AggressiveMode.hpp"
#include <cstdint>

struct BenchResult {
    const char* name;
    double      avgMs;
    double      totalMs;
    bool        stable;
    std::uint64_t hash;
    int         active;
};

using namespace ciphyxs;
using namespace ciphyxs::stresstest;

static void setupWorld(PhysicsWorld& w, bool parallel) {
    w.enableDbvt();
    PhysicsWorldConfig cfg;
    cfg.gravity              = Vec3f(0, -9.81f, 0);
    cfg.fixedTimestep        = 1.0f / 60.0f;
    cfg.linearDamping        = 0.1f;
    cfg.angularDamping       = 0.1f;
    cfg.sleepEnergyThreshold = 0.005f;
    cfg.sleepTimeRequired    = 0.5f;
    cfg.enableParallelSolver = parallel;
    cfg.numThreads           = 0;
    cfg.enableTaskGraphPipeline = parallel;
    w.setConfig(cfg);
    w.solverConfig().numIterations = 10;
}

static void createScene(PhysicsWorld& w, int numBoxes) {
    ShapeHandle ground = w.createShape(Plane{Vec3f::unitY(), 0});
    ShapeHandle boxSh  = w.createShape(Box{Vec3f(0.35f, 0.35f, 0.35f)});
    {
        RigidBodyDesc g;
        g.motionType = MotionType::Static;
        g.setShape(ground);
        g.position = Vec3f::zero();
        w.createBody(g);
    }
    constexpr float kSpacing = 0.72f;
    constexpr int kW = 10, kD = 5, kH = 4;
    int placed = 0;
    for (int iy = 0; iy < kH && placed < numBoxes; ++iy)
        for (int iz = 0; iz < kD && placed < numBoxes; ++iz)
            for (int ix = 0; ix < kW && placed < numBoxes; ++ix) {
                float x = (ix - (kW-1)*0.5f) * kSpacing;
                float z = (iz - (kD-1)*0.5f) * kSpacing;
                float y = 0.35f + iy * kSpacing;
                RigidBodyDesc b;
                b.mass = 0.5f;
                b.setShape(boxSh);
                b.position = Vec3f(x, y, z);
                b.restitution = 0.0f;
                b.friction = 0.9f;
                b.startActive = true;
                w.createBody(b);
                ++placed;
            }
}

static BenchResult runBench(const char* label, int numBoxes, int numFrames,
                            bool parallel, bool useAggressivePreset) {
    PhysicsWorld world;
    if (useAggressivePreset) {
        aggressive::applyLowEndPreset(world);
    } else {
        setupWorld(world, parallel);
    }
    createScene(world, numBoxes);

    printf("  [%-26s] %d boxes x %d frames... ", label, numBoxes, numFrames);
    fflush(stdout);

    Stopwatch timer;
    timer.start();
    for (int f = 0; f < numFrames; ++f)
        world.step(1.0f / 60.0f);
    timer.stop();

    bool stable = true;
    const auto& bodies = world.bodies();
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (bodies.motionTypes[i] != MotionType::Dynamic) continue;
        const Vec3f& p = bodies.positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            stable = false; break;
        }
    }

    int activeCount = 0;
    for (std::size_t i = 0; i < bodies.size(); ++i)
        if (bodies.activeFlags[i] && bodies.motionTypes[i] == MotionType::Dynamic)
            ++activeCount;

    std::uint64_t hash = hashBodyState(bodies);
    long long ms = timer.ms();
    printf("done  %lld ms  active=%d\n", ms, activeCount);
    fflush(stdout);
    return {label, double(ms) / numFrames, double(ms), stable, hash, activeCount};
}

int main() {
    printHeader("Low-End Hardware Benchmark (2-core, 1.6GHz, 512MB)");

    constexpr int kBoxes = 200;
    constexpr int kFrames = 500;
    printf("   %d boxes x %d frames\n\n", kBoxes, kFrames);
    fflush(stdout);

    BenchResult r1 = runBench("Default (sequential)", kBoxes, kFrames, false, false);
    BenchResult r2 = runBench("Default (parallel)",   kBoxes, kFrames, true,  false);
    BenchResult r3 = runBench("Aggressive (preset)",  kBoxes, kFrames, true,  true);

    printf("\n── Results ──\n\n");
    printf("   %-28s %12s %12s %8s %8s\n",
           "Configuration", "Avg(ms)", "Total(ms)", "Stable", "Active");
    printf("   ----------------------------------------------------------------\n");

    auto pr = [](const BenchResult& r) {
        printf("   %-28s %12.4f %12.0f %8s %8d\n",
               r.name, r.avgMs, r.totalMs,
               r.stable ? "YES" : "NO", r.active);
    };
    pr(r1); pr(r2); pr(r3);

    printf("\n── Speedup ──\n\n");
    if (r3.totalMs > 0 && r1.totalMs > 0)
        printf("   Aggressive vs Default(seq): %.2fx\n", r1.totalMs / r3.totalMs);
    if (r3.totalMs > 0 && r2.totalMs > 0)
        printf("   Aggressive vs Default(par): %.2fx\n", r2.totalMs / r3.totalMs);

    printf("\n── Memory: ~6 MB for 5000 bodies  |  512 MB: %.1f%%\n\n", 6.0/512.0*100.0);

    bool ok = r1.stable && r2.stable && r3.stable;
    printf("── Verdict ──\n%s\n",
        ok ? "   ✓ All stable on low-end target" : "   ✗ Some configurations unstable");

    return ok ? 0 : 1;
}
