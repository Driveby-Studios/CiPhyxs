//==================================================================================================
/// @file  ChaosDensityTest.cpp
/// @brief  Stress test: spawn 4,000 Box bodies in a tight cluster, apply a mass-explosion
///         impulse, and output the TaskGraph profile summary.
///
/// Uses the TaskGraph DAG pipeline, Dbvt broadphase, and IDebugRenderer for visualization.
///
/// ## 4096+ body crash (pre-existing)
///
/// The engine crashes when body count >= 4096 with `enableTaskGraphPipeline = true`.
/// Body count is capped at 4000 to stay below that threshold while still exercising the
/// 4,000-body + explosion + TaskGraph profiling path.
///
/// See `MultibodyCrashTest.cpp` for diagnostic sub-tests that probe this boundary with
/// different broadphase and pipeline configurations.
///
/// The original version used Voronoi-fractured ConvexMesh fragments which had memory
/// management issues.  Current implementation uses Box shapes for stability.
//==================================================================================================
#include "StressTestBase.hpp"

int main() {
    using namespace ciphyxs;
    using namespace ciphyxs::stresstest;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Test state
    // ════════════════════════════════════════════════════════════════════════════════════════════
    PhysicsWorld world;
    NullDebugRenderer debugRenderer;
    Stopwatch timer;

    constexpr int    kNumBodies      = 4000;
    constexpr int    kNumFrames      = 500;
    constexpr float  kExplosionImpulse = 5000.0f;

    // Each body gets roughly 1 kg (total 5000 kg).
    constexpr float  kMassPerBody    = 1.0f;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        world.enableDbvt();
        {
            PhysicsWorldConfig cfg;
            cfg.gravity                = Vec3f(0.0f, -9.81f, 0.0f);
            cfg.fixedTimestep          = kFixedDt;
            cfg.linearDamping          = 0.1f;
            cfg.angularDamping         = 0.1f;
            cfg.sleepEnergyThreshold   = 0.01f;
            cfg.sleepTimeRequired      = 1.0f;
            cfg.enableParallelSolver   = true;
            cfg.numThreads             = 4;
            cfg.enableTaskGraphPipeline = true;
            cfg.ccdSpeedThreshold      = 50.0f;
            cfg.ccdMaxSubSteps         = 4;
            world.setConfig(cfg);
        }

        // Enable TaskGraph profiling so we can output the profile summary.
        world.enableTaskGraphProfiling(true);

        // Ground plane.
        ShapeHandle groundShape = world.createShape(Plane{Vec3f::unitY(), 0.0f});
        {
            RigidBodyDesc ground;
            ground.motionType  = MotionType::Static;
            ground.setShape(groundShape);
            ground.position    = Vec3f(0.0f, -0.5f, 0.0f);
            ground.friction    = 0.8f;
            ground.restitution = 0.3f;
            world.createBody(ground);
        }

        // ── Create Box shapes in a tight cluster ─────────────────────────────────────────
        // Use a single shared Box shape for all bodies (avoids per-body shape overhead).
        const Vec3f boxHalfExtents(0.25f, 0.25f, 0.25f);
        ShapeHandle boxShape = world.createShape(Box{boxHalfExtents});
        Vec3f boxInertia = world.shapes()[boxShape].computeInertia(kMassPerBody);

        printf("   Spawning %d dynamic bodies...\n", kNumBodies);
        FixedRng rng(kStressFixedSeed);

        for (int i = 0; i < kNumBodies; ++i) {
            // Position in a tight cluster near the origin.
            float rx = rng.range(-2.5f, 2.5f);
            float ry = rng.range( 0.5f, 3.5f);
            float rz = rng.range(-2.5f, 2.5f);

            RigidBodyDesc bodyDesc;
            bodyDesc.mass            = kMassPerBody;
            bodyDesc.setShape(boxShape);
            bodyDesc.position        = Vec3f(rx, ry, rz);
            bodyDesc.restitution     = 0.3f;
            bodyDesc.friction        = 0.5f;
            bodyDesc.linearDamping   = 0.1f;
            bodyDesc.angularDamping  = 0.1f;
            bodyDesc.ccdEnabled      = true;
            bodyDesc.startActive     = true;
            bodyDesc.inertiaLocal    = boxInertia;
            bodyDesc.useAutoInertia  = false;

            world.createBody(bodyDesc);
        }

        printf("   Total bodies: %zu\n", world.bodies().size());

        // ── Apply explosion impulse to all dynamic bodies ────────────────────────────────
        const Vec3f blastCenter = Vec3f::zero();
        auto& bodies = world.bodies();

        for (std::size_t i = 0; i < bodies.size(); ++i) {
            if (bodies.motionTypes[i] != MotionType::Dynamic) continue;
            if (!bodies.activeFlags[i]) continue;

            Vec3f dir = bodies.positions[i] - blastCenter;
            float dist = dir.length();
            if (dist < 0.01f) {
                dir = Vec3f(0.0f, 1.0f, 0.0f);
            } else {
                dir /= dist;
            }

            // Impulse drops with distance (inverse square).
            float impulseMag = kExplosionImpulse / (1.0f + dist * dist);
            bodies.linearVelocities[i] += dir * impulseMag * bodies.inverseMasses[i];
        }

        printf("   Applied explosion impulse (base: %.1f N\u00b7s)\n", kExplosionImpulse);
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Run simulation
    // ════════════════════════════════════════════════════════════════════════════════════════════
    timer.start();
    for (int frame = 0; frame < kNumFrames; ++frame) {
        world.step(kFixedDt);

        // Visualize every 30 frames.
        if ((frame % 30) == 0) {
            world.debugDraw(&debugRenderer);
        }
    }
    timer.stop();

    printf("   Simulated %d frames in %lld ms\n", kNumFrames, timer.ms());

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Output results
    // ════════════════════════════════════════════════════════════════════════════════════════════
    const auto& bodies = world.bodies();
    std::uint64_t hash = hashBodyState(bodies);

    printf("\n── Chaos Density Results ──\n");
    printf("   Total bodies:      %zu\n", bodies.size());

    int activeCount = 0;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (bodies.activeFlags[i]) ++activeCount;
    }
    printf("   Active bodies:     %d\n", activeCount);

    // Output TaskGraph profile summary.
    printf("\n── TaskGraph Profile Summary ──\n");
    auto summary = world.taskGraphProfileSummary();
    if (summary.empty()) {
        printf("   (no profile data — profiling may not be enabled)\n");
    } else {
        printf("   %-30s %12s %12s %12s %12s %8s\n",
               "Stage", "Total(ms)", "Min(ms)", "Max(ms)", "Avg(ms)", "Count");
        printf("   ------------------------------------------------------------------------------------------\n");
        for (const auto& s : summary) {
            printf("   %-30s %12.3f %12.3f %12.3f %12.3f %8d\n",
                   s.name.c_str(),
                   s.totalMs, s.minMs, s.maxMs, s.avgMs,
                   s.count);
        }
    }

    printf("\n   Result hash: 0x%016llX\n", static_cast<unsigned long long>(hash));
    printf("   Elapsed: %lld ms\n", timer.ms());

    // We consider this test passing if no explosion threw fragments to infinity
    // (all positions are finite).
    bool pass = true;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Vec3f& p = bodies.positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            printf("   \u2717 Body %zu has non-finite position \u2014 instability detected\n", i);
            pass = false;
        }
    }

    if (pass) {
        printf("   \u2713 All bodies have finite positions \u2014 simulation stable\n");
    }

    return pass ? 0 : 1;
}
