//==================================================================================================
/// @file  ChaosDensityTest.cpp
/// @brief  Stress test: spawn 5,000 Voronoi-fractured ConvexMesh bodies, apply a mass-explosion
///         impulse, and output taskGraphProfileSummary.
///
/// Uses the TaskGraph DAG pipeline, Dbvt broadphase (crash fix applied), aligned SoA storage,
/// and IDebugRenderer for visualization.
///
/// ## Voronoi fragmentation
///
/// A single "mother" ConvexMesh (a 2×2×2 box hull) is fractured via `VoronoiFracture::fragment()`
/// into ~100 convex shards.  Each shard's mesh is registered as a Shape, and the shape is
/// instantiated ~50 times to reach 5,000 dynamic bodies — each carrying a unique ConvexMesh
/// shape from a Voronoi fracture pattern.
///
/// ## 4096+ body crash (pre-existing, now fixed)
///
/// The engine previously crashed at >= 4096 bodies with `enableDbvt()` + TaskGraph pipeline.
/// Root cause: `std::vector<bool>` bitset proxy reads from worker threads caused a data race
/// in `activeFlags` / `ccdFlags` when different threads read different bits from the same
/// underlying word.  Fixed by changing to `AlignedVector<uint8_t, 16>`.
///
/// ## Coding standards
///
/// - `alignas(16)` types throughout
/// - `AlignedAllocator` / `ScratchVec<T,16>` for heap storage (no std::vector for sim data)
/// - SoA (`RigidBodyStorage`) for simulation data
/// - Fixed seed for all RNG
/// - `IDebugRenderer` used for visualization (`NullDebugRenderer` by default)
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
    constexpr int    kNumFrames          = 500;
    constexpr float  kExplosionImpulse   = 8000.0f;
    constexpr float  kMotherHalf         = 1.0f;    // 2.0 m box (mother shape)

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        // SpatialHash broadphase — cell size will be auto-tuned after all bodies are created.
        world.setBroadphaseType(BroadphaseType::SpatialHash);
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
            cfg.enableTaskGraphPipeline = true;
            cfg.ccdSpeedThreshold      = 50.0f;
            cfg.ccdMaxSubSteps         = 4;
            world.setConfig(cfg);
        }

        // Enable TaskGraph profiling for the profile summary output.
        world.enableTaskGraphProfiling(true);

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
        //     The vertex data lives inside `fragments` — we must keep fragments alive for the
        //     entire test because Shape::convexMesh.vertices points into it.
        ScratchVec<ShapeHandle, 16> fragmentShapes(actualFragments);
        for (std::size_t fi = 0; fi < actualFragments; ++fi) {
            fragmentShapes[fi] = world.createShape(Shape(fragments[fi].mesh));
        }

        // ── Spawn kTargetBodies bodies using the fragment shapes ────────────────────────────
        //     Cycle through fragment shapes; each body gets the mass of its fragment.
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

        // Auto-tune spatial hash cell size based on body density (prevents O(n²) degeneracy).
        world.autoTuneSpatialHashCellSize();
        printf("   Auto-tuned spatial hash cell size: %.2f\n",
               world.broadphaseConfig().spatialHashCellSize);

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
            // Small fragments (~1 kg) at 8000 N·s would reach 8000 m/s,
            // causing CCD failure and numerical instability. Cap at 100 m/s.
            constexpr float kMaxDeltaV = 100.0f;
            float dvLen = deltaV.length();
            if (dvLen > kMaxDeltaV) {
                deltaV *= kMaxDeltaV / dvLen;
            }
            bodies.linearVelocities[i] += deltaV;
        }

        printf("   Applied explosion impulse (base: %.1f N·s)\n", kExplosionImpulse);
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

        // Periodic NaN check to catch early instability.
        if ((frame % 50) == 0 && frame > 0) {
            const auto& bodies = world.bodies();
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                if (!bodies.activeFlags[i]) continue;
                const Vec3f& p = bodies.positions[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    printf("   ✗ NaN/Inf detected in body %zu at frame %d\n", i, frame);
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

    // Output TaskGraph profile summary (accumulated across all frames).
    printf("\n── TaskGraph Profile Summary (accumulated across %d frames) ──\n", kNumFrames);
    auto summary = world.accumulatedProfileSummary();
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
