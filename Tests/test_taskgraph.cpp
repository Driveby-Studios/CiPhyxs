//==================================================================================================
/// @file  test_taskgraph.cpp
/// @brief  Validation tests for TaskGraph scheduler + task-graph pipeline mode.
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <atomic>
#include <thread>

int main() {
    using namespace ciphyxs;

    // Unbuffer stdout/stderr for ctest compatibility.
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║    CiPhyxs \x97 TaskGraph Validation           ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    int failures = 0;

    // ── 1. Basic DAG execution ───────────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 1: Basic linear pipeline (A \x86 B \x86 C) \x94\x94\n");

        ThreadPool pool(2);
        TaskGraph graph;

        std::atomic<int> order{0};
        int resultA = 0, resultB = 0, resultC = 0;

        auto a = graph.add("A", {}, [&] { resultA = 1; order.fetch_add(1); });
        TaskId depsB[] = {a};
        auto b = graph.add("B", depsB, [&] { resultB = resultA + 1; order.fetch_add(1); });
        TaskId depsC[] = {b};
        graph.add("C", depsC, [&] { resultC = resultB + 1; order.fetch_add(1); });

        graph.execute(pool);

        bool ok = (resultA == 1 && resultB == 2 && resultC == 3 && order.load() == 3);
        printf("   Results: A=%d B=%d C=%d  order=3  %s\n",
               resultA, resultB, resultC, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 2. Fan-out / fan-in ──────────────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 2: Fan-out (root \x86 A, B \x86 join) \x94\x94\n");

        ThreadPool pool(4);
        TaskGraph graph;

        std::atomic<int> counter{0};
        int shared = 0;

        auto root = graph.add("Root", {}, [&] { shared = 10; });

        TaskId depsA[] = {root};
        auto a = graph.add("A", depsA, [&] { counter.fetch_add(1); });

        TaskId depsB[] = {root};
        auto b = graph.add("B", depsB, [&] { counter.fetch_add(1); });

        TaskId joinDeps[] = {a, b};
        int joinResult = 0;
        graph.add("Join", joinDeps, [&] { joinResult = shared + counter.load(); });

        graph.execute(pool);

        bool ok = (shared == 10 && counter.load() == 2 && joinResult == 12);
        printf("   shared=%d counter=%d joinResult=%d  %s\n",
               shared, counter.load(), joinResult, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 3. Deterministic ordering ────────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 3: Parallel root task completion \x94\x94\n");

        ThreadPool pool(4);
        TaskGraph graph;

        std::atomic<int> step{0};
        int execOrder[2] = {0, 0};

        graph.add("Root0", {}, [&] {
            while (step.load(std::memory_order_acquire) != 0)
                std::this_thread::yield();
            execOrder[0] = 0;
            step.store(1, std::memory_order_release);
        });

        graph.add("Root1", {}, [&] {
            while (step.load(std::memory_order_acquire) != 1)
                std::this_thread::yield();
            execOrder[1] = 1;
            step.store(2, std::memory_order_release);
        });

        graph.execute(pool);

        bool ok = (execOrder[0] == 0 && execOrder[1] == 1 && step.load() == 2);
        printf("   Order: [%d, %d]  %s\n",
               execOrder[0], execOrder[1], ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 4. No-op (empty) tasks ───────────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 4: No-op (sync-only) nodes \x94\x94\n");

        ThreadPool pool(2);
        TaskGraph graph;

        std::atomic<int> done{0};

        auto nop = graph.add("Nop", {}, std::function<void()>{});

        TaskId deps[] = {nop};
        graph.add("AfterNop", deps, [&] { done.store(1, std::memory_order_release); });

        graph.execute(pool);

        bool ok = (done.load() == 1);
        printf("   Nop task completed:  %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 5. TaskGraphPipeline convenience ─────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 5: TaskGraphPipeline builder \x94\x94\n");

        TaskGraph graph;
        auto pipe = TaskGraphPipeline(graph);

        int val = 0;
        pipe.stage("S1", [&] { val = 1; });
        pipe.stage("S2", [&] { val = val * 2 + 1; });
        pipe.stage("S3", [&] { val = val * 3 + 1; });

        ThreadPool pool(2);
        graph.execute(pool);

        bool ok = (val == 10);
        printf("   Pipeline result: %d (expected 10)  %s\n", val, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 6. PhysicsWorld task-graph pipeline mode ─────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 6: PhysicsWorld task-graph pipeline \x94\x94\n");

        auto buildWorld = [](PhysicsWorld& w) {
            ShapeHandle hPlane = w.createShape(Plane{});
            ShapeHandle hBox   = w.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
            ShapeHandle hSphere = w.createShape(Sphere{0.4f});

            RigidBodyDesc ground;
            ground.motionType = MotionType::Static;
            ground.setShape(hPlane);
            w.createBody(ground);

            RigidBodyDesc b1; b1.mass = 10.0f; b1.setShape(hBox);
            b1.position = Vec3f(0, 5, 0); w.createBody(b1);

            RigidBodyDesc b2; b2.mass = 1.0f; b2.setShape(hSphere);
            b2.position = Vec3f(0.5f, 8, 0.3f); w.createBody(b2);
        };

        // 6a: TaskGraph mode WITHOUT parallel solver
        printf("   6a: TaskGraph, sequential solver...\n");
        {
            PhysicsWorld tgWorld;
            tgWorld.enableTaskGraphPipeline(true);
            buildWorld(tgWorld);

            for (int i = 0; i < 10; ++i) {
                tgWorld.step(1.0f / 60.0f);
            }
            printf("      10 steps OK\n");
        }

        // 6b: TaskGraph mode WITH parallel solver
        printf("   6b: TaskGraph, parallel solver...\n");
        {
            PhysicsWorld tgWorld;
            tgWorld.enableParallelSolver(true, 2);
            tgWorld.enableTaskGraphPipeline(true);
            buildWorld(tgWorld);

            for (int i = 0; i < 10; ++i) {
                tgWorld.step(1.0f / 60.0f);
            }
            printf("      10 steps OK\n");
        }

        // 6c: Determinism check (no parallel solver)
        printf("   6c: Determinism check...\n");
        {
            PhysicsWorld seqWorld;
            buildWorld(seqWorld);

            PhysicsWorld tgWorld;
            tgWorld.enableTaskGraphPipeline(true);
            buildWorld(tgWorld);

            for (int i = 0; i < 120; ++i) {
                seqWorld.step(1.0f / 60.0f);
                tgWorld.step(1.0f / 60.0f);
            }

            auto& seqPos = seqWorld.bodies().positions;
            auto& tgPos  = tgWorld.bodies().positions;

            bool match = true;
            for (std::size_t i = 0; i < seqWorld.bodies().size(); ++i) {
                Vec3f dP = seqPos[i] - tgPos[i];
                if (dP.lengthSquared() > 1e-8f) {
                    match = false;
                    printf("      Mismatch at body %zu: (%.8f,%.8f,%.8f)\n",
                           i, dP.x, dP.y, dP.z);
                }
            }

            printf("      Sequential vs TaskGraph:  %s\n", match ? "IDENTICAL" : "DIFFERENT");
            if (!match) ++failures;
        }

        // 6d: Determinism check WITH parallel solver
        printf("   6d: Determinism (parallel solver)...\n");
        {
            PhysicsWorld seqWorld;
            seqWorld.enableParallelSolver(true, 2);
            buildWorld(seqWorld);

            PhysicsWorld tgWorld;
            tgWorld.enableParallelSolver(true, 2);
            tgWorld.enableTaskGraphPipeline(true);
            buildWorld(tgWorld);

            for (int i = 0; i < 120; ++i) {
                seqWorld.step(1.0f / 60.0f);
                tgWorld.step(1.0f / 60.0f);
            }

            auto& seqPos = seqWorld.bodies().positions;
            auto& tgPos  = tgWorld.bodies().positions;

            bool match = true;
            for (std::size_t i = 0; i < seqWorld.bodies().size(); ++i) {
                Vec3f dP = seqPos[i] - tgPos[i];
                if (dP.lengthSquared() > 1e-8f) {
                    match = false;
                    printf("      Mismatch at body %zu: (%.8f,%.8f,%.8f)\n",
                           i, dP.x, dP.y, dP.z);
                }
            }

            printf("      Sequential vs TaskGraph:  %s\n", match ? "IDENTICAL" : "DIFFERENT");
            if (!match) ++failures;
        }

        printf("   Test 6 complete\n");
    }

    // ── 7. TaskBarrier ───────────────────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 7: TaskBarrier fork-join \x94\x94\n");

        ThreadPool pool(4);
        TaskBarrier barrier(5);
        std::atomic<int> sum{0};

        for (int i = 0; i < 4; ++i) {
            pool.enqueue([i, &sum, &barrier] {
                sum.fetch_add(i + 1, std::memory_order_release);
                barrier.arrive();
            });
        }
        barrier.arrive();
        barrier.wait();

        bool ok = (sum.load() == 10);
        printf("   Sum: %d (expected 10)  %s\n", sum.load(), ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 8. TaskGraph::addBarrier() ──────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 8: addBarrier() join node \x94\x94\n");

        ThreadPool pool(4);
        TaskGraph graph;

        std::atomic<int> counter{0};
        int afterBarrier = 0;

        // Three independent root tasks.
        auto a = graph.add("A", {}, [&] { counter.fetch_add(1, std::memory_order_release); });
        auto b = graph.add("B", {}, [&] { counter.fetch_add(1, std::memory_order_release); });
        auto c = graph.add("C", {}, [&] { counter.fetch_add(1, std::memory_order_release); });

        // Barrier that waits for all three.
        TaskId barrierDeps[] = {a, b, c};
        auto barrier = graph.addBarrier("Barrier", barrierDeps);

        // Dependent on barrier — must see counter == 3.
        TaskId postDeps[] = {barrier};
        graph.add("Post", postDeps, [&] {
            afterBarrier = counter.load(std::memory_order_acquire);
        });

        graph.execute(pool);

        bool ok = (afterBarrier == 3);
        printf("   Barrier join: afterBarrier=%d (expected 3)  %s\n",
               afterBarrier, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── 9. Per-island task sub-graphs in PhysicsWorld pipeline ─────────────────────────────────
    {
        printf("\x94\x94 Test 9: Per-island task sub-graph pipeline \x94\x94\n");

        // Create a world with multiple islands: several pairs of bodies that
        // are only in contact with their partner, forming isolated islands.
        auto buildMultiIsland = [](PhysicsWorld& w) {
            ShapeHandle hPlane = w.createShape(Plane{});
            ShapeHandle hBox   = w.createShape(Box{Vec3f(0.3f, 0.3f, 0.3f)});

            // Ground.
            RigidBodyDesc ground;
            ground.motionType = MotionType::Static;
            ground.setShape(hPlane);
            w.createBody(ground);

            // Island 1: box pair at x = -3
            {
                RigidBodyDesc b1; b1.mass = 5.0f; b1.setShape(hBox);
                b1.position = Vec3f(-3, 3, 0); w.createBody(b1);
                RigidBodyDesc b2; b2.mass = 5.0f; b2.setShape(hBox);
                b2.position = Vec3f(-3, 5, 0); w.createBody(b2);
            }

            // Island 2: box pair at x = 0
            {
                RigidBodyDesc b1; b1.mass = 5.0f; b1.setShape(hBox);
                b1.position = Vec3f(0, 3, 0); w.createBody(b1);
                RigidBodyDesc b2; b2.mass = 5.0f; b2.setShape(hBox);
                b2.position = Vec3f(0, 5, 0); w.createBody(b2);
            }

            // Island 3: box pair at x = 3
            {
                RigidBodyDesc b1; b1.mass = 5.0f; b1.setShape(hBox);
                b1.position = Vec3f(3, 3, 0); w.createBody(b1);
                RigidBodyDesc b2; b2.mass = 5.0f; b2.setShape(hBox);
                b2.position = Vec3f(3, 5, 0); w.createBody(b2);
            }
        };

        // 9a: Sequential pipeline baseline.
        printf("   9a: Sequential pipeline...\n");
        PhysicsWorld seqWorld;
        buildMultiIsland(seqWorld);
        for (int i = 0; i < 120; ++i) {
            seqWorld.step(1.0f / 60.0f);
        }
        printf("      120 steps OK\n");

        // 9b: TaskGraph pipeline with per-island solve tasks.
        printf("   9b: TaskGraph per-island pipeline...\n");
        PhysicsWorld tgWorld;
        tgWorld.enableParallelSolver(true, 4);
        tgWorld.enableTaskGraphPipeline(true);
        buildMultiIsland(tgWorld);
        for (int i = 0; i < 120; ++i) {
            tgWorld.step(1.0f / 60.0f);
        }
        printf("      120 steps OK\n");

        // 9c: Determinism check — results must match.
        printf("   9c: Determinism (sequential vs per-island task graph)...\n");
        {
            auto& seqPos = seqWorld.bodies().positions;
            auto& seqVel = seqWorld.bodies().linearVelocities;
            auto& tgPos  = tgWorld.bodies().positions;
            auto& tgVel  = tgWorld.bodies().linearVelocities;

            bool match = true;
            for (std::size_t i = 0; i < seqWorld.bodies().size(); ++i) {
                Vec3f dP = seqPos[i] - tgPos[i];
                Vec3f dV = seqVel[i] - tgVel[i];
                if (dP.lengthSquared() > 1e-6f || dV.lengthSquared() > 1e-6f) {
                    match = false;
                    printf("      Mismatch at body %zu: dP=(%.6f,%.6f,%.6f) dV=(%.6f,%.6f,%.6f)\n",
                           i, dP.x, dP.y, dP.z, dV.x, dV.y, dV.z);
                }
            }
            printf("      Sequential vs TaskGraph:  %s\n", match ? "IDENTICAL" : "DIFFERENT");
            if (!match) ++failures;
        }

        // 9d: Bodies should have fallen and settled.
        printf("   9d: Bodies have settled...\n");
        {
            bool settled = true;
            for (std::size_t i = 0; i < seqWorld.bodies().size(); ++i) {
                if (seqWorld.bodies().motionTypes[i] != MotionType::Dynamic) continue;
                if (seqWorld.bodies().activeFlags[i]) {
                    float speed = seqWorld.bodies().linearVelocities[i].length();
                    if (speed > 0.1f) {
                        settled = false;
                        break;
                    }
                }
            }
            printf("      All dynamic bodies at rest: %s\n", settled ? "YES" : "NO");
        }
    }

    // ── 10. Task-graph profiling ────────────────────────────────────────────────────────────
    {
        printf("\x94\x94 Test 10: Task-graph profiling validation \x94\x94\n");

        // Create a simple scene with a few dynamic bodies.
        PhysicsWorld world;
        world.enableTaskGraphPipeline(true);
        world.enableParallelSolver(true, 2);

        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        RigidBodyDesc b1; b1.mass = 10.0f; b1.setShape(hBox);
        b1.position = Vec3f(0, 5, 0); world.createBody(b1);

        RigidBodyDesc b2; b2.mass = 1.0f; b2.setShape(hBox);
        b2.position = Vec3f(0.5f, 8, 0.3f); world.createBody(b2);

        // 10a: Profiling disabled by default — should produce no events.
        printf("   10a: Profiling disabled by default...\n");
        {
            for (int i = 0; i < 5; ++i)
                world.step(1.0f / 60.0f);

            bool empty = world.taskGraphProfileEvents().empty();
            printf("      Events empty (default): %s\n", empty ? "PASS" : "FAIL");
            if (!empty) ++failures;
        }

        // 10b: Enable profiling, run steps, verify events are recorded.
        printf("   10b: Enable profiling and capture events...\n");
        {
            world.enableTaskGraphProfiling(true);

            for (int i = 0; i < 5; ++i)
                world.step(1.0f / 60.0f);

            // After the last step, events should be from that single step execution.
            const auto& events = world.taskGraphProfileEvents();

            bool hasEvents   = !events.empty();
            bool validNames  = true;
            bool validTimes  = true;

            for (const auto& ev : events) {
                if (ev.name.empty())              validNames = false;
                if (ev.elapsedMs < 0.0)           validTimes = false;
                // Allow zero for no-op/barrier nodes.
            }

            printf("      Events recorded: %zu  %s\n", events.size(), hasEvents ? "PASS" : "FAIL");
            printf("      All names non-empty:        %s\n", validNames ? "PASS" : "FAIL");
            printf("      All durations valid:        %s\n", validTimes ? "PASS" : "FAIL");

            if (!hasEvents)  ++failures;
            if (!validNames) ++failures;
            if (!validTimes) ++failures;
        }

        // 10c: Verify profile summary aggregation.
        printf("   10c: Profile summary aggregation...\n");
        {
            // Run more steps so we have fresh events.
            for (int i = 0; i < 3; ++i)
                world.step(1.0f / 60.0f);

            auto summary = world.taskGraphProfileSummary();

            // Should have at least one summary entry.
            bool hasSummary = !summary.empty();

            // Verify each summary entry.
            bool validCounts = true;
            bool validAverages = true;
            int  totalCount = 0;

            for (const auto& s : summary) {
                if (s.count < 1) validCounts = false;
                if (s.count > 0) {
                    // avg = total / count within tolerance.
                    double expectedAvg = s.totalMs / static_cast<double>(s.count);
                    double diff = std::abs(s.avgMs - expectedAvg);
                    if (diff > 1e-6) validAverages = false;
                }
                totalCount += s.count;

                // min <= avg <= max
                if (s.minMs > s.avgMs + 1e-9 || s.avgMs > s.maxMs + 1e-9)
                    validAverages = false;
            }

            // Total count should match the number of events from the last step.
            bool countMatch = (totalCount == static_cast<int>(world.taskGraphProfileEvents().size()));

            printf("      Summary entries: %zu  %s\n", summary.size(), hasSummary ? "PASS" : "FAIL");
            printf("      All counts >= 1:            %s\n", validCounts ? "PASS" : "FAIL");
            printf("      All averages consistent:    %s\n", validAverages ? "PASS" : "FAIL");
            printf("      Total count matches events: %s\n", countMatch ? "PASS" : "FAIL");

            if (!hasSummary)      ++failures;
            if (!validCounts)     ++failures;
            if (!validAverages)   ++failures;
            if (!countMatch)      ++failures;
        }

        // 10d: Disable profiling — no new events should be recorded after a step.
        //      Events are cleared only at the start of execute() when profiling is
        //      enabled. After disabling, old events remain in the vector; we verify
        //      the count does NOT grow (no new events appended).
        printf("   10d: Disable profiling...\n");
        {
            world.enableTaskGraphProfiling(false);

            std::size_t beforeCount = world.taskGraphProfileEvents().size();
            for (int i = 0; i < 3; ++i)
                world.step(1.0f / 60.0f);
            std::size_t afterCount = world.taskGraphProfileEvents().size();

            // When disabled, no new events are added, so count stays the same.
            bool noNewEvents = (afterCount == beforeCount);
            printf("      Events before: %zu, after: %zu (no growth)  %s\n",
                   beforeCount, afterCount, noNewEvents ? "PASS" : "FAIL");
            if (!noNewEvents) ++failures;
        }

        // 10e: Re-enable profiling — events should appear again.
        printf("   10e: Re-enable profiling (no CCD on bodies)...\n");
        {
            world.enableTaskGraphProfiling(true);
            world.step(1.0f / 60.0f);

            bool hasEvents = !world.taskGraphProfileEvents().empty();
            printf("      Events after re-enable:     %s\n", hasEvents ? "PASS" : "FAIL");
            if (!hasEvents) ++failures;

            // Verify event names include known pipeline stages.
            // Note: CCD is now per-island and only emitted for islands with
            // CCD-eligible bodies. Since this test doesn't enable CCD on any
            // body, the global fallback "GlobalCollideAndSolve" task subsumes CCD.
            const auto& events = world.taskGraphProfileEvents();
            bool hasBroadphase = false;
            bool hasForces     = false;
            bool hasVelocities = false;
            bool hasPositions  = false;
            for (const auto& ev : events) {
                if (ev.name == "Broadphase")           hasBroadphase = true;
                if (ev.name == "IntegrateForces")      hasForces     = true;
                if (ev.name == "IntegrateVelocities")  hasVelocities = true;
                if (ev.name == "IntegratePositions")   hasPositions  = true;
            }
            printf("      Contains Broadphase:          %s\n", hasBroadphase ? "PASS" : "FAIL");
            printf("      Contains IntegrateForces:     %s\n", hasForces ? "PASS" : "FAIL");
            printf("      Contains IntegrateVelocities: %s\n", hasVelocities ? "PASS" : "FAIL");
            printf("      Contains IntegratePositions:  %s\n", hasPositions ? "PASS" : "FAIL");
            if (!hasBroadphase) ++failures;
            if (!hasForces)     ++failures;
            if (!hasVelocities) ++failures;
            if (!hasPositions)  ++failures;
        }

        // 10f: Per-island CCD task emitted when CCD is enabled on bodies in
        //      a multi-island scene.  Two independent pairs of boxes, each with a
        //      fast-moving CCD-enabled body, forming two disjoint islands.
        printf("   10f: Per-island CCD task (multi-island scene)...\n");
        {
            // Use zero speed threshold so bodies always qualify for CCD.
            PhysicsWorldConfig cfg;
            cfg.ccdSpeedThreshold = 0.0f;
            cfg.enableParallelSolver = true;
            cfg.enableTaskGraphPipeline = true;
            PhysicsWorld ccdWorld(cfg);
            ccdWorld.enableTaskGraphPipeline(true);
            ccdWorld.enableParallelSolver(true, 2);
            ccdWorld.enableTaskGraphProfiling(true);

            ShapeHandle hGroundBox = ccdWorld.createShape(Box{Vec3f(2.0f, 0.2f, 2.0f)});
            ShapeHandle hFastBox  = ccdWorld.createShape(Box{Vec3f(0.3f, 0.3f, 0.3f)});

            // Island 1: static ground box at x=-6 + fast dynamic box just above it.
            // The fast box starts with velocity toward the ground so CCD triggers.
            {
                RigidBodyDesc ground;
                ground.motionType = MotionType::Static;
                ground.setShape(hGroundBox);
                ground.position = Vec3f(-6, 0, 0);
                ccdWorld.createBody(ground);
            }
            {
                RigidBodyDesc fast;
                fast.mass = 1.0f;
                fast.setShape(hFastBox);
                fast.position = Vec3f(-6, 0.4f, 0);  // overlapping AABB with ground
                fast.linearVelocity = Vec3f(0, -50, 0);
                fast.ccdEnabled = true;
                ccdWorld.createBody(fast);
            }

            // Island 2: static ground box at x=6 + fast dynamic box just above it.
            {
                RigidBodyDesc ground;
                ground.motionType = MotionType::Static;
                ground.setShape(hGroundBox);
                ground.position = Vec3f(6, 0, 0);
                ccdWorld.createBody(ground);
            }
            {
                RigidBodyDesc fast;
                fast.mass = 1.0f;
                fast.setShape(hFastBox);
                fast.position = Vec3f(6, 0.4f, 0);  // overlapping AABB with ground
                fast.linearVelocity = Vec3f(0, -50, 0);
                fast.ccdEnabled = true;
                ccdWorld.createBody(fast);
            }

            // The per-island path uses lagging island data from the previous frame:
            // usePerIsland is checked at graph-build time, before FormIslands executes.
            // So we need one warm-up frame to seed island data for the next frame.
            ccdWorld.step(1.0f / 60.0f);  // frame 1: global path, populates islands

            // Clear profiling events from the warm-up frame.
            ccdWorld.enableTaskGraphProfiling(true);
            ccdWorld.step(1.0f / 60.0f);  // frame 2+: per-island path with CCD

            const auto& events = ccdWorld.taskGraphProfileEvents();
            int ccdCount = 0;
            for (const auto& ev : events) {
                if (ev.name.find("CCD_") == 0) {
                    ++ccdCount;
                }
            }
            // Should have 2 CCD tasks (one per island with CCD bodies).
            bool correctCount = (ccdCount == 2);
            printf("      Found %d per-island CCD_* tasks (expected 2)  %s\n",
                   ccdCount, correctCount ? "PASS" : "FAIL");
            if (!correctCount) {
                printf("      Events (%zu):", events.size());
                for (const auto& ev : events) {
                    printf(" %s", ev.name.c_str());
                }
                printf("\n");
                ++failures;
            }
        }

        printf("   Test 10 complete\n");
    }

    // ─── Test 11: Task-graph DOT export ──────────────────────────────────────────
    {
        printf("\nTest 11: TaskGraph::toDot() DOT export\n");

        ThreadPool pool(4);
        TaskGraph graph;

        auto rootA = graph.add("RootA", std::span<const TaskId>{}, []{});
        auto rootB = graph.add("RootB", std::span<const TaskId>{}, []{});
        TaskId depsMid[] = {rootA, rootB};
        auto mid = graph.add("Mid", depsMid, []{});
        TaskId depsLeaf[] = {mid};
        auto leaf = graph.add("Leaf", depsLeaf, []{});

        // Also add a barrier node to test barrier export.
        TaskId barrierDeps[] = {leaf};
        graph.addBarrier("Join", barrierDeps);

        std::string dot = graph.toDot();

        // ── Validate header ──────────────────────────────────────────────────────
        bool hasGraphHeader = (dot.find("digraph TaskGraph") != std::string::npos);
        bool hasRankDir    = (dot.find("rankdir=TB") != std::string::npos);
        printf("      DOT header present                      %s\n",
               (hasGraphHeader && hasRankDir) ? "PASS" : "FAIL");
        if (!hasGraphHeader || !hasRankDir) {
            printf("      DOT output:\n%s\n", dot.c_str());
            ++failures;
        }

        // ── Validate nodes ───────────────────────────────────────────────────────
        bool hasRootA = (dot.find("RootA") != std::string::npos);
        bool hasRootB = (dot.find("RootB") != std::string::npos);
        bool hasMid   = (dot.find("Mid") != std::string::npos);
        bool hasLeaf  = (dot.find("Leaf") != std::string::npos);
        bool hasJoin  = (dot.find("Join") != std::string::npos);
        bool allNodes = hasRootA && hasRootB && hasMid && hasLeaf && hasJoin;
        printf("      All 5 task nodes present                %s\n",
               allNodes ? "PASS" : "FAIL");
        if (!allNodes) {
            printf("      rootA=%d rootB=%d mid=%d leaf=%d join=%d\n",
                   hasRootA, hasRootB, hasMid, hasLeaf, hasJoin);
            ++failures;
        }

        // ── Validate edges ───────────────────────────────────────────────────────
        bool edgeRootAtoMid = (dot.find("RootA -> Mid") != std::string::npos ||
                               dot.find("RootA\" -> \"Mid") != std::string::npos);
        bool edgeRootBtoMid = (dot.find("RootB -> Mid") != std::string::npos ||
                               dot.find("RootB\" -> \"Mid") != std::string::npos);
        bool edgeMidToLeaf  = (dot.find("Mid -> Leaf") != std::string::npos ||
                               dot.find("Mid\" -> \"Leaf") != std::string::npos);
        bool edgeLeafToJoin = (dot.find("Leaf -> Join") != std::string::npos ||
                               dot.find("Leaf\" -> \"Join") != std::string::npos);
        bool allEdges = edgeRootAtoMid && edgeRootBtoMid && edgeMidToLeaf && edgeLeafToJoin;
        printf("      All 4 dependency edges present          %s\n",
               allEdges ? "PASS" : "FAIL");
        if (!allEdges) {
            printf("      RootA->Mid=%d RootB->Mid=%d Mid->Leaf=%d Leaf->Join=%d\n",
                   edgeRootAtoMid, edgeRootBtoMid, edgeMidToLeaf, edgeLeafToJoin);
            printf("      DOT output:\n%s\n", dot.c_str());
            ++failures;
        }

        // ── Validate root node styling ────────────────────────────────────────────
        bool rootStyle = (dot.find("lightblue") != std::string::npos);
        printf("      Root node styling (lightblue)           %s\n",
               rootStyle ? "PASS" : "FAIL");
        if (!rootStyle) {
            ++failures;
        }

        // ── Validate DOT closing brace ────────────────────────────────────────────
        bool hasClosing = (dot.find("}") != std::string::npos && dot.back() == '\n');
        printf("      Valid DOT closing                       %s\n",
               hasClosing ? "PASS" : "FAIL");
        if (!hasClosing) {
            ++failures;
        }

        // ── Execute after export to ensure no side effects ─────────────────────────
        graph.execute(pool);  // should still work after toDot()
        printf("      Graph executes after toDot()            PASS\n");

        printf("   Test 11 complete\n");
    }

    // ── Summary ──────────────────────────────────────────────────────────────────
    int totalTests = 11;
    printf("\n\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\n");
    printf("  Results: %d / %d passed%s\n",
           totalTests - failures, totalTests, failures == 0 ? "" : " (failures)");
    printf("\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\n");

    return failures;
}
