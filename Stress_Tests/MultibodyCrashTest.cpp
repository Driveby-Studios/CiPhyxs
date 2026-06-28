//==================================================================================================
/// @file  MultibodyCrashTest.cpp
/// @brief  Diagnostic test that probes the 4096+ dynamic body boundary with multiple broadphase
///         and pipeline configurations to isolate the pre-existing crash.
///
/// ## Background
///
/// The engine has a pre-existing crash at >= 4096 dynamic bodies when `enableTaskGraphPipeline`
/// is `true`.  This test runs several sub-tests with different configurations to narrow down
/// the root cause.  It reports which configurations pass/fail.
///
/// ## Suspects
///
///   - `std::vector<BroadphasePair> m_pairs` overflow or allocation issue
///   - `m_islandManifolds` resize issue
///   - `m_bodyToNode` in Dbvt indexed by `RigidBodyHandle` (uint32_t, should be fine up to 4B)
///   - Stack overflow from deeply-nested or large-stack allocations in the task graph
///   - The `islandFinalIds` vector or task graph node count hitting an internal limit
///   - Dbvt-specific crash vs. general crash (isolate by testing SpatialHash broadphase)
///
/// ## Usage
///
/// Compile and run in Release mode (Debug is very slow for 5000+ bodies).
/// ```
///   cmake -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build --target MultibodyCrashTest
///   ./build/MultibodyCrashTest
/// ```
//==================================================================================================
#include "StressTestBase.hpp"
#include <cstring>

int main() {
    using namespace ciphyxs;
    using namespace ciphyxs::stresstest;

    constexpr int kBodyCount4096 = 4096;
    constexpr int kBodyCount5000 = 5000;
    constexpr int kFramesShort   = 10;    // Short run to detect instant crash
    constexpr float kBoxHalf     = 0.3f;

    struct SubTestResult {
        const char* name;
        bool        passed;
        const char* failure;
    };

    // We run multiple sub-tests, each with its own PhysicsWorld.
    // Each sub-test creates `kBodyCount` dynamic boxes in a tight cluster + a ground plane.
    auto runSubTest = [](const char* label,
                         std::size_t numBodies,
                         bool useTaskGraph,
                         BroadphaseType bpType,
                         int numFrames) -> SubTestResult {

        PhysicsWorld world;
        NullDebugRenderer debugRenderer;

        try {
            world.setBroadphaseType(bpType);
            {
                PhysicsWorldConfig cfg;
                cfg.gravity                = Vec3f(0.0f, -9.81f, 0.0f);
                cfg.fixedTimestep          = kFixedDt;
                cfg.linearDamping          = 0.1f;
                cfg.angularDamping         = 0.1f;
                cfg.sleepEnergyThreshold   = 0.001f;
                cfg.sleepTimeRequired      = 5.0f;
                cfg.enableParallelSolver   = useTaskGraph;
                cfg.numThreads             = 4;
                cfg.enableTaskGraphPipeline = useTaskGraph;
                cfg.ccdSpeedThreshold      = 0.0f;
                cfg.ccdMaxSubSteps         = 0;
                world.setConfig(cfg);
            }

            world.enableTaskGraphProfiling(useTaskGraph);

            // Shapes.
            ShapeHandle groundShape = world.createShape(Plane{Vec3f::unitY(), 0.0f});
            ShapeHandle boxShape    = world.createShape(Box{Vec3f(kBoxHalf, kBoxHalf, kBoxHalf)});

            // Ground plane.
            {
                RigidBodyDesc ground;
                ground.motionType = MotionType::Static;
                ground.setShape(groundShape);
                ground.position = Vec3f(0.0f, -0.5f, 0.0f);
                world.createBody(ground);
            }

            // Spawn `numBodies` dynamic boxes in a wide area to avoid O(n²) pair explosion
            // in the broadphase.  A 50 m spread ensures most bodies do not overlap in the
            // broadphase, keeping pair generation O(n) rather than O(n²).
            constexpr float kSpread = 50.0f;
            FixedRng rng(kStressFixedSeed);
            for (std::size_t i = 0; i < numBodies; ++i) {
                float rx = rng.range(-kSpread, kSpread);
                float ry = rng.range(0.5f, kSpread);
                float rz = rng.range(-kSpread, kSpread);

                RigidBodyDesc box;
                box.mass        = 0.5f;
                box.setShape(boxShape);
                box.position    = Vec3f(rx, ry, rz);
                box.restitution = 0.2f;
                box.friction    = 0.6f;
                box.startActive = true;
                world.createBody(box);
            }

            // Run simulation.
            for (int frame = 0; frame < numFrames; ++frame) {
                world.step(kFixedDt);

                // Check for NaN every few frames.
                if ((frame % 5) == 0) {
                    const auto& bodies = world.bodies();
                    for (std::size_t i = 0; i < bodies.size(); ++i) {
                        const Vec3f& p = bodies.positions[i];
                        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                            return {label, false, "NaN/Inf detected during simulation"};
                        }
                    }
                }
            }

            return {label, true, nullptr};

        } catch (const std::exception& e) {
            return {label, false, e.what()};
        } catch (...) {
            return {label, false, "Unknown exception (likely segfault / access violation)"};
        }
    };

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Run sub-tests
    // ════════════════════════════════════════════════════════════════════════════════════════════
    printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  4096+ Body Crash Diagnostic Tests                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    // To avoid crashing the test harness itself, we wrap each sub-test in a
    // separate process or try-catch.  Since we can't fork, we use try-catch
    // and report any crash/exception.

    SubTestResult results[] = {
        runSubTest("Dbvt + TaskGraph pipeline, 4096 bodies",  kBodyCount4096, true,
                   BroadphaseType::Dbvt, kFramesShort),
        runSubTest("SpatialHash + TaskGraph pipeline, 4096 bodies", kBodyCount4096, true,
                   BroadphaseType::SpatialHash, kFramesShort),
        runSubTest("Dbvt + Sequential pipeline, 4096 bodies", kBodyCount4096, false,
                   BroadphaseType::Dbvt, kFramesShort),
        runSubTest("SpatialHash + Sequential pipeline, 4096 bodies", kBodyCount4096, false,
                   BroadphaseType::SpatialHash, kFramesShort),
        runSubTest("SIMD-BruteForce + Sequential pipeline, 4096 bodies", kBodyCount4096, false,
                   BroadphaseType::SimdBruteForce, kFramesShort),
        runSubTest("Dbvt + TaskGraph pipeline, 5000 bodies", kBodyCount5000, true,
                   BroadphaseType::Dbvt, kFramesShort),
        runSubTest("SpatialHash + TaskGraph pipeline, 5000 bodies", kBodyCount5000, true,
                   BroadphaseType::SpatialHash, kFramesShort),
    };

    // ── Report ──────────────────────────────────────────────────────────────────────────────────
    printf("\n── Diagnostic Results ──\n\n");

    int passed = 0;
    int failed = 0;
    for (const auto& r : results) {
        printf("   [%s]  %s\n", r.passed ? "PASS" : "FAIL", r.name);
        if (!r.passed && r.failure) {
            printf("          Reason: %s\n", r.failure);
        }
        if (r.passed) ++passed; else ++failed;
    }

    printf("\n   Passed: %d / %d  Failed: %d / %d\n",
           passed, static_cast<int>(sizeof(results)/sizeof(results[0])),
           failed, static_cast<int>(sizeof(results)/sizeof(results[0])));

    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    if (failed == 0) {
        printf("   All configurations passed — crash may be platform-specific\n");
        printf("   or may require > 4096 bodies or longer simulation.\n");
    } else {
        printf("   FAILURES DETECTED — examine the failing configuration(s) above.\n");
        printf("   The crash pattern will help narrow down the root cause:\n\n");
        printf("   • If Dbvt-TaskGraph fails but SpatialHash-TaskGraph passes:\n");
        printf("     → The issue is Dbvt-specific (m_bodyToNode, tree rebuild,\n");
        printf("       or pair generation at high body counts).\n\n");
        printf("   • If both TaskGraph variants fail but Sequential passes:\n");
        printf("     → The issue is in fixedStepTaskGraph (island formation,\n");
        printf("       per-island lambda captures, or task graph limits).\n\n");
        printf("   • If all configurations fail:\n");
        printf("     → General issue at high body counts (SoA storage resize,\n");
        printf("       memory allocation, or SIMD alignment).\n\n");
        printf("   • If only 5000-TaskGraph fails but 4096-TaskGraph passes:\n");
        printf("     → Scaling limit beyond 4096 (pair explosion, memory pressure).\n");
    }
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    return failed > 0 ? 1 : 0;
}
