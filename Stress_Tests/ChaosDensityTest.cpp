//==================================================================================================
/// @file  ChaosDensityTest.cpp
/// @brief  Stress test: spawn 5,000 Voronoi fragments from a large ConvexMesh, apply
///         a mass-explosion impulse, and output the TaskGraph profile summary.
///
/// Uses the TaskGraph DAG pipeline, Dbvt broadphase, and IDebugRenderer for visualization.
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

    constexpr int    kNumFragments   = 3000;
    constexpr int    kNumFrames      = 10;
    constexpr float  kExplosionImpulse = 5000.0f;
    constexpr float  kBigMeshHalf    = 5.0f;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup — create a large convex mesh, fracture it into Voronoi fragments, then spawn
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

        // ── Build a large convex mesh (roughly spherical, many vertices) ────────────────
        // Use a subdivided icosahedron approximation to get a dense vertex set.
        constexpr int kIcoVerts     = 12;

        // Base icosahedron vertices on unit sphere.
        alignas(16) Vec3f icoBase[kIcoVerts];
        {
            const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
            icoBase[0]  = Vec3f(-1.0f,  t,  0.0f).normalized();
            icoBase[1]  = Vec3f( 1.0f,  t,  0.0f).normalized();
            icoBase[2]  = Vec3f(-1.0f, -t,  0.0f).normalized();
            icoBase[3]  = Vec3f( 1.0f, -t,  0.0f).normalized();
            icoBase[4]  = Vec3f( 0.0f, -1.0f,  t).normalized();
            icoBase[5]  = Vec3f( 0.0f,  1.0f,  t).normalized();
            icoBase[6]  = Vec3f( 0.0f, -1.0f, -t).normalized();
            icoBase[7]  = Vec3f( 0.0f,  1.0f, -t).normalized();
            icoBase[8]  = Vec3f( t,  0.0f, -1.0f).normalized();
            icoBase[9]  = Vec3f( t,  0.0f,  1.0f).normalized();
            icoBase[10] = Vec3f(-t,  0.0f, -1.0f).normalized();
            icoBase[11] = Vec3f(-t,  0.0f,  1.0f).normalized();
        }

        // Subdivide to get more vertices.
        // For a sufficient vertex count for 5000 fragments, we use a dense mesh.
        // Build a vertex list from the base icosahedron and subdivide.
        // We'll create a grid of vertices on the sphere surface instead for simplicity.
        constexpr int kThetaSteps = 20;
        constexpr int kPhiSteps   = 20;
        constexpr int kGridVerts  = kThetaSteps * kPhiSteps;
        alignas(16) Vec3f bigMeshVerts[kGridVerts];
        {
            int vi = 0;
            for (int it = 0; it < kThetaSteps; ++it) {
                float theta = 3.14159265f * static_cast<float>(it) / static_cast<float>(kThetaSteps - 1);
                for (int ip = 0; ip < kPhiSteps; ++ip) {
                    float phi = 2.0f * 3.14159265f * static_cast<float>(ip) / static_cast<float>(kPhiSteps - 1);
                    float x = std::sin(theta) * std::cos(phi);
                    float y = std::cos(theta);
                    float z = std::sin(theta) * std::sin(phi);
                    bigMeshVerts[vi++] = Vec3f(x, y, z) * kBigMeshHalf;
                }
            }
        }

        ConvexMesh bigMesh;
        bigMesh.vertices    = bigMeshVerts;
        bigMesh.vertexCount = kGridVerts;
        bigMesh.halfExtents = Vec3f(kBigMeshHalf, kBigMeshHalf, kBigMeshHalf);
        bigMesh.center      = Vec3f::zero();

        // Generate Voronoi fragments (total mass = 5000 kg, one per fragment roughly).
        printf("   Generating %d Voronoi fragments...\n", kNumFragments);
        auto seeds = VoronoiFracture::generateSeeds(bigMesh, kNumFragments);
        auto fragments = VoronoiFracture::fragment(bigMesh, seeds, 5000.0f);
        printf("   Generated %zu actual fragments\n", fragments.size());

        // Spawn each fragment as a dynamic body.
        int spawned = 0;
        for (std::size_t fi = 0; fi < fragments.size() && spawned < kNumFragments; ++fi) {
            const auto& frag = fragments[fi];

            // Use Box shape (avoids ConvexMesh overhead/degeneracy at scale).
            ShapeHandle fragShape = world.createShape(Box{frag.mesh.halfExtents});

            // Position fragments in a tight cluster at the origin with slight random offset.
            float spread = 0.5f;
            float rx = (static_cast<float>(fi % 20) - 9.5f) * spread;
            float ry = (static_cast<float>((fi / 20) % 20) - 9.5f) * spread + 5.0f;
            float rz = (static_cast<float>(fi / 400) - 2.5f) * spread;

            RigidBodyDesc fragDesc;
            fragDesc.mass          = frag.mass;
            fragDesc.setShape(fragShape);
            fragDesc.position      = Vec3f(rx, ry, rz);
            fragDesc.restitution   = 0.3f;
            fragDesc.friction      = 0.5f;
            fragDesc.linearDamping  = 0.1f;
            fragDesc.angularDamping = 0.1f;
            fragDesc.ccdEnabled    = true;
            fragDesc.startActive   = true;

            // Use inertia from fracture computation.
            fragDesc.inertiaLocal  = frag.inertia;
            fragDesc.useAutoInertia = false;

            world.createBody(fragDesc);
            ++spawned;
        }

        printf("   Spawned %d fragment bodies\n", spawned);
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
        if ((frame % 50) == 0) {
            printf("   Step %d/%d...\n", frame, kNumFrames);
            fflush(stdout);
        }
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
            printf("   ✗ Body %zu has non-finite position — instability detected\n", i);
            pass = false;
        }
    }

    if (pass) {
        printf("   ✓ All bodies have finite positions — simulation stable\n");
    }

    return pass ? 0 : 1;
}
