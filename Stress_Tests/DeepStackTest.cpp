//==================================================================================================
/// @file  DeepStackTest.cpp
/// @brief  Stress test: spawn 200 boxes arranged in a tall wall-like stack, simulate 2000 frames,
///         and verify the simulation remains stable (no NaNs, no explosions, no wild divergence).
///
/// Uses aligned storage, SoA bodies (RigidBodyStorage), Dbvt broadphase, and IDebugRenderer.
/// The test does not require a perfect stack — it validates that the solver does not diverge
/// and that no body reaches an invalid state after sustained simulation.
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

    constexpr int    kNumBoxes      = 200;
    constexpr int    kNumFrames     = 2000;
    constexpr float  kBoxHalf       = 0.35f;
    constexpr float  kStackSpacing  = 0.72f;

    // Build a dense block: 10 wide x 5 deep x 4 high = 200 boxes
    constexpr int kWallW = 10;
    constexpr int kWallD = 5;
    constexpr int kWallH = 4;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        world.enableDbvt();
        {
            PhysicsWorldConfig cfg;
            cfg.gravity              = Vec3f(0.0f, -9.81f, 0.0f);
            cfg.fixedTimestep        = kFixedDt;
            cfg.linearDamping        = 0.1f;
            cfg.angularDamping       = 0.1f;
            cfg.sleepEnergyThreshold = 0.005f;
            cfg.sleepTimeRequired    = 1.0f;
            cfg.enableParallelSolver = true;
            cfg.numThreads = 4;
            cfg.enableTaskGraphPipeline = true;
            cfg.ccdSpeedThreshold    = 0.0f;
            cfg.ccdMaxSubSteps       = 0;
            world.setConfig(cfg);
        }
        {
            auto& sc = world.solverConfig();
            sc.numIterations = 20;
            sc.baumgarte = 0.25f;
            sc.maxPenetrationCorrection = 0.15f;
            sc.enableWarmStart = true;
            sc.warmStartFactor = 0.85f;
        }

        // Create shapes.
        ShapeHandle groundShape = world.createShape(Plane{Vec3f::unitY(), 0.0f});
        ShapeHandle boxShape    = world.createShape(Box{Vec3f(kBoxHalf, kBoxHalf, kBoxHalf)});

        // Ground plane at y=0.
        {
            RigidBodyDesc ground;
            ground.motionType  = MotionType::Static;
            ground.setShape(groundShape);
            ground.position    = Vec3f::zero();
            ground.friction    = 0.9f;
            ground.restitution = 0.0f;
            world.createBody(ground);
        }

        // Build a dense wall of boxes.
        int placed = 0;
        for (int iy = 0; iy < kWallH && placed < kNumBoxes; ++iy) {
            for (int iz = 0; iz < kWallD && placed < kNumBoxes; ++iz) {
                for (int ix = 0; ix < kWallW && placed < kNumBoxes; ++ix) {
                    float x = (static_cast<float>(ix) - static_cast<float>(kWallW - 1) * 0.5f) * kStackSpacing;
                    float z = (static_cast<float>(iz) - static_cast<float>(kWallD - 1) * 0.5f) * kStackSpacing;
                    float y = kBoxHalf + static_cast<float>(iy) * kStackSpacing;

                    RigidBodyDesc box;
                    box.mass         = 0.5f;
                    box.setShape(boxShape);
                    box.position     = Vec3f(x, y, z);
                    box.restitution  = 0.0f;
                    box.friction     = 0.9f;
                    box.linearDamping  = 0.1f;
                    box.angularDamping = 0.1f;
                    box.startActive    = true;

                    world.createBody(box);
                    ++placed;
                }
            }
        }

        printf("   Created %d boxes (wall: %dx%dx%d)\n", placed, kWallW, kWallD, kWallH);
        printf("   Total bodies: %zu\n", world.bodies().size());
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Run simulation
    // ════════════════════════════════════════════════════════════════════════════════════════════
    timer.start();
    for (int frame = 0; frame < kNumFrames; ++frame) {
        world.step(kFixedDt);

        // Visualize every 120 frames.
        if ((frame % 120) == 0) {
            world.debugDraw(&debugRenderer);
        }

        // Periodically check for NaN / inf to detect early instability.
        if ((frame % 200) == 0 && frame > 0) {
            const auto& bodies = world.bodies();
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                const Vec3f& p = bodies.positions[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    printf("   ✗ NaN/Inf detected in body %zu at frame %d — simulation unstable\n",
                           i, frame);
                    printf("   Result hash: 0x%016llX\n",
                           static_cast<unsigned long long>(hashBodyState(bodies)));
                    return 1;
                }
            }
        }
    }
    timer.stop();

    printf("   Simulated %d frames in %lld ms\n", kNumFrames, timer.ms());

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Verify stability
    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Stability criteria:
    //   1. All positions are finite (no NaN/Inf)
    //   2. No body has escaped to infinity (positions within reasonable bounds)
    //   3. All inverse masses are non-negative
    //   4. Report any ground penetrations (informational, not fatal)
    constexpr float kMaxReasonablePos = 10000.0f;
    constexpr float kPenetrationThreshold = -kBoxHalf - 0.2f;

    bool allFinite  = true;
    bool allInBounds = true;
    bool massValid  = true;
    int  penetrators = 0;

    const auto& bodies = world.bodies();
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (bodies.motionTypes[i] != MotionType::Dynamic) continue;

        const Vec3f& p = bodies.positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            allFinite = false;
        }
        if (std::abs(p.x) > kMaxReasonablePos ||
            std::abs(p.y) > kMaxReasonablePos ||
            std::abs(p.z) > kMaxReasonablePos) {
            allInBounds = false;
        }
        if (bodies.inverseMasses[i] < 0.0f || !std::isfinite(bodies.inverseMasses[i])) {
            massValid = false;
        }
        if (bodies.positions[i].y < kPenetrationThreshold) {
            ++penetrators;
        }
    }

    // Compute result hash for verification.
    std::uint64_t hash = hashBodyState(bodies);

    // ── Report ──────────────────────────────────────────────────────────────────────────────────
    printf("\n── Deep Stack Results ──\n");
    printf("   All finite positions:  %s\n", allFinite ? "YES" : "NO");
    printf("   All within bounds:     %s\n", allInBounds ? "YES" : "NO");
    printf("   Valid inverse masses:  %s\n", massValid ? "YES" : "NO");
    printf("   Ground penetrations:  %d / %d\n", penetrators, kNumBoxes);

    bool stable = allFinite && allInBounds && massValid;
    if (stable) {
        printf("   ✓ Simulation stable after %d frames with %d boxes\n",
               kNumFrames, kNumBoxes);
    } else {
        printf("   ✗ Simulation instability detected\n");
    }

    printf("   Result hash: 0x%016llX\n", static_cast<unsigned long long>(hash));
    printf("   Elapsed: %lld ms\n\n", timer.ms());

    return stable ? 0 : 1;
}
