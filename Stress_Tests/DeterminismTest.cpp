//==================================================================================================
/// @file  DeterminismTest.cpp
/// @brief  Stress test: run two parallel World instances for 500 frames with identical
///         initial conditions and compare state hashes for bit-exact determinism.
///
/// Uses the TaskGraph DAG pipeline, Dbvt broadphase, and aligned SoA storage.
///
/// @note  5000 frames is the full spec.  In Debug builds, reduce to 500 with:
///        `-DkNumFrames=500` or edit the line below.  Release mode runs 5000 easily.
//==================================================================================================
#include "StressTestBase.hpp"

int main() {
    using namespace ciphyxs;
    using namespace ciphyxs::stresstest;

    constexpr int kNumFrames     = 5000;
    constexpr int kNumBodies     = 100;

    NullDebugRenderer debugRenderer;
    Stopwatch timer;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Helper: configure a world with consistent settings
    // ════════════════════════════════════════════════════════════════════════════════════════════
    auto configureWorld = [](PhysicsWorld& w) {
        w.enableDbvt();
        PhysicsWorldConfig cfg;
        cfg.gravity                = Vec3f(0.0f, -9.81f, 0.0f);
        cfg.fixedTimestep          = kFixedDt;
        cfg.linearDamping          = 0.05f;
        cfg.angularDamping         = 0.05f;
        cfg.sleepEnergyThreshold   = 0.001f;
        cfg.sleepTimeRequired      = 0.5f;
        cfg.enableParallelSolver   = true;
        cfg.numThreads             = 4;
        cfg.enableTaskGraphPipeline = true;
        cfg.ccdSpeedThreshold      = 20.0f;
        cfg.ccdMaxSubSteps         = 4;
        w.setConfig(cfg);
    };

    auto populateWorld = [](PhysicsWorld& w, FixedRng& rng) {
        // Shapes.
        ShapeHandle groundShape = w.createShape(Plane{Vec3f::unitY(), 0.0f});
        ShapeHandle boxShape    = w.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        ShapeHandle sphereShape = w.createShape(Sphere{0.5f});
        ShapeHandle capsuleShape = w.createShape(Capsule{0.4f, 0.6f});

        // Static ground.
        {
            RigidBodyDesc ground;
            ground.motionType  = MotionType::Static;
            ground.setShape(groundShape);
            ground.position    = Vec3f(0.0f, -0.5f, 0.0f);
            ground.friction    = 0.8f;
            w.createBody(ground);
        }

        // Spawn a mix of dynamic bodies with deterministic random positions.
        for (int i = 0; i < kNumBodies; ++i) {
            // Cycle through shape types deterministically.
            ShapeHandle sh;
            switch (i % 3) {
            case 0: sh = boxShape;    break;
            case 1: sh = sphereShape; break;
            case 2: sh = capsuleShape; break;
            }

            float x = rng.range(-5.0f, 5.0f);
            float z = rng.range(-5.0f, 5.0f);
            float y = rng.range(1.0f, 8.0f);

            RigidBodyDesc body;
            body.mass           = rng.range(0.5f, 5.0f);
            body.setShape(sh);
            body.position       = Vec3f(x, y, z);
            body.restitution    = rng.range(0.0f, 0.5f);
            body.friction       = rng.range(0.2f, 0.9f);
            body.linearVelocity  = Vec3f(
                rng.range(-2.0f, 2.0f),
                rng.range(-2.0f, 2.0f),
                rng.range(-2.0f, 2.0f)
            );
            body.angularVelocity = Vec3f(
                rng.range(-1.0f, 1.0f),
                rng.range(-1.0f, 1.0f),
                rng.range(-1.0f, 1.0f)
            );
            body.linearDamping   = 0.05f;
            body.angularDamping  = 0.05f;
            body.startActive     = true;
            body.ccdEnabled      = (i % 5 == 0);  // every 5th body uses CCD

            w.createBody(body);
        }
    };

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup world A
    // ════════════════════════════════════════════════════════════════════════════════════════════
    PhysicsWorld worldA;
    configureWorld(worldA);

    {
        FixedRng rngA(kStressFixedSeed);
        populateWorld(worldA, rngA);
    }

    printf("   World A: %zu bodies\n", worldA.bodies().size());

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup world B (identical — same config, same RNG seed)
    // ════════════════════════════════════════════════════════════════════════════════════════════
    PhysicsWorld worldB;
    configureWorld(worldB);

    {
        FixedRng rngB(kStressFixedSeed);
        populateWorld(worldB, rngB);
    }

    printf("   World B: %zu bodies\n", worldB.bodies().size());

    // ── Verify identical initial state ───────────────────────────────────────────────────────────
    {
        std::uint64_t hashA = hashBodyState(worldA.bodies());
        std::uint64_t hashB = hashBodyState(worldB.bodies());

        printf("   Initial hash A: 0x%016llX\n", static_cast<unsigned long long>(hashA));
        printf("   Initial hash B: 0x%016llX\n", static_cast<unsigned long long>(hashB));

        if (hashA != hashB) {
            printf("   ✗ INITIAL STATE MISMATCH — worlds are not identical before simulation\n");
            return 1;
        }
        printf("   ✓ Initial states match\n");
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Run both worlds in lockstep for kNumFrames
    // ════════════════════════════════════════════════════════════════════════════════════════════
    timer.start();
    for (int frame = 0; frame < kNumFrames; ++frame) {
        worldA.step(kFixedDt);
        worldB.step(kFixedDt);

        // Visualize from world A every 200 frames.
        if ((frame % 200) == 0) {
            worldA.debugDraw(&debugRenderer);
        }
    }
    timer.stop();

    printf("   Simulated %d frames in %lld ms\n", kNumFrames, timer.ms());

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Final state comparison
    // ════════════════════════════════════════════════════════════════════════════════════════════
    std::uint64_t finalHashA = hashBodyState(worldA.bodies());
    std::uint64_t finalHashB = hashBodyState(worldB.bodies());

    printf("\n── Determinism Test Results ──\n");
    printf("   Final hash A: 0x%016llX\n", static_cast<unsigned long long>(finalHashA));
    printf("   Final hash B: 0x%016llX\n", static_cast<unsigned long long>(finalHashB));

    // Full per-body comparison for detailed reporting.
    int mismatchCount = 0;
    const auto& bodiesA = worldA.bodies();
    const auto& bodiesB = worldB.bodies();

    std::size_t maxBodies = std::min(bodiesA.size(), bodiesB.size());
    for (std::size_t i = 0; i < maxBodies; ++i) {
        Vec3f dPos = bodiesA.positions[i] - bodiesB.positions[i];
        Vec3f dLin = bodiesA.linearVelocities[i] - bodiesB.linearVelocities[i];
        Vec3f dAng = bodiesA.angularVelocities[i] - bodiesB.angularVelocities[i];

        if (dPos.lengthSquared() > 1e-8f ||
            dLin.lengthSquared() > 1e-8f ||
            dAng.lengthSquared() > 1e-8f) {
            ++mismatchCount;
            if (mismatchCount <= 5) {
                printf("   Body %zu: pos diff=(%.2e, %.2e, %.2e)\n",
                       i, dPos.x, dPos.y, dPos.z);
            }
        }
    }

    bool pass = (finalHashA == finalHashB) && (mismatchCount == 0);

    if (pass) {
        printf("   ✓ PERFECT DETERMINISM — all %zu bodies match after %d frames\n",
               maxBodies, kNumFrames);
    } else {
        printf("   ✗ DETERMINISM FAILURE — %d bodies differ after %d frames\n",
               mismatchCount, kNumFrames);
    }

    printf("   Result hash: 0x%016llX\n", static_cast<unsigned long long>(finalHashA));
    printf("   Elapsed: %lld ms\n", timer.ms());

    return pass ? 0 : 1;
}
