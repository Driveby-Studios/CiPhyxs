//==================================================================================================
/// @file  ChaosDensityDiag.cpp
/// @brief  Diagnostic variant of ChaosDensityTest: 5 frames only, sequential pipeline,
///         per-frame timing to identify bottleneck.
///
/// Same setup as ChaosDensityTest (5,000 Voronoi fragments × explosion impulse) but:
///   - Runs only 5 frames
///   - Forces enableTaskGraphPipeline = false (sequential path)
///   - Prints setup time and per-frame times for frames 0–4
///   - Does NOT set minPrecomputationPoints (removes any threshold override)
///
/// ## Voronoi fragmentation
///
/// A single "mother" ConvexMesh (a 2×2×2 box hull) is fractured via `VoronoiFracture::fragment()`
/// into ~100 convex shards.  Each shard's mesh is registered as a Shape, and the shape is
/// instantiated ~50 times to reach 5,000 dynamic bodies — each carrying a unique ConvexMesh
/// shape from a Voronoi fracture pattern.
//==================================================================================================
#include "StressTestBase.hpp"
#include "../include/ciphyxs/core/AggressiveMode.hpp"
#include <cstdio>
#include <cmath>

int main() {
    using namespace ciphyxs;
    using namespace ciphyxs::stresstest;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Test state
    // ════════════════════════════════════════════════════════════════════════════════════════════
    PhysicsWorld world(PhysicsWorld::Preset::LowEnd);
    NullDebugRenderer debugRenderer;
    Stopwatch timer;

    constexpr int    kTargetBodies       = 5000;
    constexpr int    kNumFragments       = 100;     // Voronoi fragments from "mother" mesh
    constexpr int    kNumFrames          = 5;       // DIAG: only 5 frames
    constexpr float  kExplosionImpulse   = 8000.0f;
    constexpr float  kMotherHalf         = 1.0f;    // 2.0 m box (mother shape)

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup (timed)
    // ════════════════════════════════════════════════════════════════════════════════════════════
    timer.start();
    {
        // SpatialHash broadphase with a small cell size (1.0m vs default 4.0m) since 5000 bodies
        // are clustered in ~5m × 5m × 5m space.  With cellSize=4.0, most bodies fall into 1-2
        // cells, making the hash degenerate to O(n²).  Cell size of 1.0 distributes them across
        // ~125 cells for efficient culling.
        world.setBroadphaseType(BroadphaseType::SpatialHash);
        world.broadphaseConfig().spatialHashCellSize = 1.0f;
        {
            PhysicsWorldConfig cfg = world.config();
            cfg.gravity                = Vec3f(0.0f, -9.81f, 0.0f);
            cfg.fixedTimestep          = kFixedDt;
            cfg.linearDamping          = 0.1f;
            cfg.angularDamping         = 0.1f;
            cfg.sleepEnergyThreshold   = 0.01f;
            cfg.sleepTimeRequired      = 0.5f;
            cfg.enableParallelSolver   = true;
            cfg.numThreads             = 0; // auto = hardware_concurrency
            cfg.enableTaskGraphPipeline = true; // DIAG: task-graph pipeline
            cfg.ccdSpeedThreshold      = 50.0f;
            cfg.ccdMaxSubSteps         = 4;
            world.setConfig(cfg);
        }

        // DIAG: no pre-computation threshold override (minPrecomputationPoints not set)

        // ── Ground plane ────────────────────────────────────────────────────────────────────
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

        // ── Build the "mother" ConvexMesh (a box hull to fracture) ──────────────────────────
        ScratchVec<Vec3f, 16> motherVerts(8);
        {
            float h = kMotherHalf;
            motherVerts[0] = Vec3f(-h, -h, -h);
            motherVerts[1] = Vec3f(+h, -h, -h);
            motherVerts[2] = Vec3f(-h, +h, -h);
            motherVerts[3] = Vec3f(+h, +h, -h);
            motherVerts[4] = Vec3f(-h, -h, +h);
            motherVerts[5] = Vec3f(+h, -h, +h);
            motherVerts[6] = Vec3f(-h, +h, +h);
            motherVerts[7] = Vec3f(+h, +h, +h);
        }

        ConvexMesh motherMesh;
        motherMesh.vertices    = motherVerts.data();
        motherMesh.vertexCount = 8;
        motherMesh.halfExtents = Vec3f(kMotherHalf, kMotherHalf, kMotherHalf);
        motherMesh.center      = Vec3f::zero();

        // ── Fracture the mother mesh into Voronoi fragments ─────────────────────────────────
        //     Total mass: kTargetBodies kg (each spawned body gets ~1 kg average).
        //     Fragment masses are proportional to their bounding-box volume.
        float totalMass = static_cast<float>(kTargetBodies);
        auto seeds = VoronoiFracture::generateSeeds(motherMesh, kNumFragments);

        printf("   Fracturing mother mesh into %d Voronoi fragments...\n", kNumFragments);
        auto fragments = VoronoiFracture::fragment(motherMesh, seeds, totalMass);
        std::size_t actualFragments = fragments.size();
        printf("   Generated %zu valid fragments\n", actualFragments);

        if (actualFragments == 0) {
            printf("   ✗ Fragment generation failed — no valid cells\n");
            return 1;
        }

        // ── Register each fragment as a Shape ───────────────────────────────────────────────
        ScratchVec<ShapeHandle, 16> fragmentShapes(actualFragments);
        for (std::size_t fi = 0; fi < actualFragments; ++fi) {
            fragmentShapes[fi] = world.createShape(Shape(fragments[fi].mesh));
        }

        // ── Spawn kTargetBodies bodies using the fragment shapes ────────────────────────────
        printf("   Spawning %d dynamic Voronoi-fragment bodies...\n", kTargetBodies);
        FixedRng rng(kStressFixedSeed);

        for (int i = 0; i < kTargetBodies; ++i) {
            std::size_t fi = static_cast<std::size_t>(i) % actualFragments;
            const auto& frag = fragments[fi];

            // Position in a tight cluster near the origin.
            float rx = rng.range(-2.5f, 2.5f);
            float ry = rng.range(0.5f, 3.5f);
            float rz = rng.range(-2.5f, 2.5f);

            RigidBodyDesc body;
            body.mass            = frag.mass;
            body.setShape(fragmentShapes[fi]);
            body.position        = Vec3f(rx, ry, rz);
            body.restitution     = 0.3f;
            body.friction        = 0.5f;
            body.linearDamping   = 0.1f;
            body.angularDamping  = 0.1f;
            body.ccdMode         = CcdMode::Cast;
            body.startActive     = true;
            body.useAutoInertia  = true;

            world.createBody(body);
        }

        printf("   Total bodies: %zu\n", world.bodies().size());

        // ── Apply explosion impulse to all dynamic bodies ───────────────────────────────────
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
            Vec3f deltaV = dir * impulseMag * bodies.inverseMasses[i];

            // Clamp velocity delta to prevent NaN/Inf from extreme speeds.
            constexpr float kMaxDeltaV = 100.0f;
            float dvLen = deltaV.length();
            if (dvLen > kMaxDeltaV) {
                deltaV *= kMaxDeltaV / dvLen;
            }
            bodies.linearVelocities[i] += deltaV;
        }

        printf("   Applied explosion impulse (base: %.1f N·s)\n", kExplosionImpulse);
    }
    timer.stop();
    long long setupMs = timer.ms();
    printf("\n── Setup time: %lld ms ──\n\n", setupMs);

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Run simulation — timed per frame
    // ════════════════════════════════════════════════════════════════════════════════════════════
    long long frameTimes[5] = {0, 0, 0, 0, 0};

    for (int frame = 0; frame < kNumFrames; ++frame) {
        timer.start();
        world.step(kFixedDt);
        timer.stop();
        frameTimes[frame] = timer.ms();
        printf("   Frame %d: %lld ms\n", frame, frameTimes[frame]);

        // DIAG: No NaN checks or debug draw needed for 5 frames.
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Output results
    // ════════════════════════════════════════════════════════════════════════════════════════════
    const auto& bodies = world.bodies();
    std::uint64_t hash = hashBodyState(bodies);

    printf("\n── Chaos Density DIAG Results ──\n");
    printf("   Total bodies:      %zu\n", bodies.size());

    int activeCount = 0;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (bodies.activeFlags[i]) ++activeCount;
    }
    printf("   Active bodies:     %d\n", activeCount);

    printf("\n── Frame-by-frame timing ──\n");
    printf("   Setup:             %5lld ms\n", setupMs);
    printf("   Frame 0 (post-explosion): %5lld ms\n", frameTimes[0]);
    printf("   Frame 1:           %5lld ms\n", frameTimes[1]);
    printf("   Frame 2:           %5lld ms\n", frameTimes[2]);
    printf("   Frame 3:           %5lld ms\n", frameTimes[3]);
    printf("   Frame 4:           %5lld ms\n", frameTimes[4]);

    long long totalSim = 0;
    for (int i = 0; i < 5; ++i) totalSim += frameTimes[i];
    printf("\n   Total sim (5 frames): %lld ms\n", totalSim);
    printf("   Total (setup + sim):  %lld ms\n", setupMs + totalSim);

    printf("\n   Result hash: 0x%016llX\n", static_cast<unsigned long long>(hash));

    // Verify all positions are finite.
    bool pass = true;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Vec3f& p = bodies.positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            printf("   ✗ Body %zu has non-finite position — instability detected\n", i);
            pass = false;
        }
    }

    if (pass) {
        printf("   ✓ All bodies have finite positions — simulation stable\n");
    }

    return pass ? 0 : 1;
}
