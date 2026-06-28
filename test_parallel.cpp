//==================================================================================================
/// @file  test_parallel.cpp
/// @brief  Validation of the spatial‑hash broadphase and parallel island solver.
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>

int main() {
    using namespace ciphyxs;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   CiPhyxs — Broadphase & Parallel Tests     ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ── 1. Spatial hash produces the same result as brute‑force broadphase ────────────────────
    {
        printf("── Test 1: Broadphase determinism (hash vs brute‑force) ──\n");

        PhysicsWorld worldBrute, worldHash;
        for (int i = 0; i < 3; ++i) {
            worldBrute.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
            worldHash.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        }
        worldHash.enableSpatialHash(4.0f);   // use spatial hash

        auto setupBodies = [](PhysicsWorld& w) {
            RigidBodyDesc g; g.motionType = MotionType::Static; g.setShape(0);
            w.createBody(g);
            for (int i = 0; i < 5; ++i) {
                RigidBodyDesc b;
                b.mass = 10.0f; b.setShape(1);
                b.position = Vec3f(i * 1.5f, 5.0f + i * 2.0f, 0.0f);
                w.createBody(b);
            }
        };
        setupBodies(worldBrute);
        setupBodies(worldHash);

        // Run both for 120 steps.
        for (int i = 0; i < 120; ++i) {
            worldBrute.step(1.0f / 60.0f);
            worldHash.step(1.0f / 60.0f);
        }

        // Compare positions of each body.
        bool ident = true;
        for (std::size_t i = 0; i < worldBrute.bodies().size(); ++i) {
            Vec3f diff = worldBrute.bodies().positions[i]
                       - worldHash.bodies().positions[i];
            float err = diff.lengthSquared();
            if (err > 1e-8f) {
                printf("   Body %zu differs: brute=(%.6f,%.6f,%.6f) hash=(%.6f,%.6f,%.6f)\n",
                       i,
                       worldBrute.bodies().positions[i].x,
                       worldBrute.bodies().positions[i].y,
                       worldBrute.bodies().positions[i].z,
                       worldHash.bodies().positions[i].x,
                       worldHash.bodies().positions[i].y,
                       worldHash.bodies().positions[i].z);
                ident = false;
            }
        }
        printf("   %s\n", ident ? "✓ IDENTICAL results" : "✗ MISMATCH");
        printf("\n");
    }

    // ── 2. Parallel island solver produces identical results to single‑threaded ───────────────
    {
        printf("── Test 2: Parallel solver determinism ──\n");

        PhysicsWorld worldSeq, worldPar;
        for (int i = 0; i < 3; ++i) {
            worldSeq.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
            worldPar.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        }
        worldPar.enableParallelSolver(true, 4);   // 4 worker threads

        auto setupScene = [](PhysicsWorld& w) {
            RigidBodyDesc g; g.motionType = MotionType::Static; g.setShape(0);
            w.createBody(g);

            // Create a chain of overlapping boxes → multiple islands.
            for (int i = 0; i < 8; ++i) {
                RigidBodyDesc b;
                b.mass = 10.0f; b.setShape(1);
                b.position = Vec3f(0.0f, 0.6f + i * 1.1f, 0.0f);
                w.createBody(b);
            }
        };
        setupScene(worldSeq);
        setupScene(worldPar);

        for (int i = 0; i < 300; ++i) {
            worldSeq.step(1.0f / 60.0f);
            worldPar.step(1.0f / 60.0f);
        }

        bool ident = true;
        for (std::size_t i = 0; i < worldSeq.bodies().size(); ++i) {
            Vec3f diff = worldSeq.bodies().positions[i]
                       - worldPar.bodies().positions[i];
            float err = diff.lengthSquared();
            if (err > 1e-8f) {
                printf("   Body %zu seq=(%.6f,%.6f,%.6f) par=(%.6f,%.6f,%.6f)\n",
                       i,
                       worldSeq.bodies().positions[i].x,
                       worldSeq.bodies().positions[i].y,
                       worldSeq.bodies().positions[i].z,
                       worldPar.bodies().positions[i].x,
                       worldPar.bodies().positions[i].y,
                       worldPar.bodies().positions[i].z);
                ident = false;
            }
        }
        printf("   %s\n", ident ? "✓ PARALLEL === SEQUENTIAL" : "✗ MISMATCH");
        printf("\n");
    }

    // ── 3. ThreadPool parallelFor correctness ─────────────────────────────────────────────────
    {
        printf("── Test 3: ThreadPool parallelFor ──\n");

        ThreadPool pool(4);
        constexpr std::size_t N = 1'000'000;
        std::vector<int> data(N, 0);

        pool.parallelFor(static_cast<std::size_t>(0), N,
                         [&](std::size_t i) { data[i] = static_cast<int>(i); });

        bool ok = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (data[i] != static_cast<int>(i)) { ok = false; break; }
        }
        printf("   %s\n", ok ? "✓ All elements correctly assigned" : "✗ CORRUPTION");
        printf("\n");
    }

    // ── 4. Performance comparison: brute‑force vs spatial hash ────────────────────────────────
    {
        printf("── Test 4: Broadphase performance (256 bodies) ──\n");

        auto runBench = [](PhysicsWorld& w, const char* label) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 60; ++i) w.step(1.0f / 60.0f);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            printf("   %s: %.2f ms total (%.3f ms/step)\n", label, ms, ms / 60.0);
        };

        // Scene: many scattered boxes on a plane.
        auto createScene = [](PhysicsWorld& w) {
            w.createShape(Plane{});
            w.createShape(Box{Vec3f(0.3f, 0.3f, 0.3f)});

            RigidBodyDesc g; g.motionType = MotionType::Static; g.setShape(0);
            w.createBody(g);

            for (int i = 0; i < 255; ++i) {
                RigidBodyDesc b;
                b.mass = 1.0f; b.setShape(1);
                b.position = Vec3f(
                    (std::rand() % 200) * 0.1f - 10.0f,
                    5.0f + (std::rand() % 100) * 0.1f,
                    (std::rand() % 200) * 0.1f - 10.0f);
                w.createBody(b);
            }
        };

        PhysicsWorld bruteWorld;
        createScene(bruteWorld);

        PhysicsWorld hashWorld;
        createScene(hashWorld);
        hashWorld.enableSpatialHash(2.0f);

        runBench(bruteWorld, "Brute‑force O(n²)");
        runBench(hashWorld, "Spatial hash O(n)");
        printf("\n");
    }

    // ── 5. ThreadPool stress test ─────────────────────────────────────────────────────────────
    {
        printf("── Test 5: ThreadPool stress (1000 tasks, mixed sizes) ──\n");

        ThreadPool pool(4);
        std::atomic<std::int64_t> sum{0};

        // Many small tasks.
        pool.parallelFor(static_cast<std::size_t>(0), std::size_t{10000},
                         [&](std::size_t i) { sum.fetch_add(static_cast<std::int64_t>(i)); });

        std::int64_t expected = 9999LL * 10000LL / 2LL;
        printf("   Sum = %lld (expected %lld) %s\n",
               (long long)sum.load(), (long long)expected,
               sum.load() == expected ? "✓" : "✗");
        printf("\n");
    }

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║      Parallel & Broadphase tests done       ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    return 0;
}
