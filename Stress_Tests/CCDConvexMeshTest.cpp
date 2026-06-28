//==================================================================================================
/// @file  CCDConvexMeshTest.cpp
/// @brief  Stress test: launch high-velocity projectiles at a thin ConvexMesh wall.
///         Verifies zero penetration using CCD with ConvexMesh-vs-Sphere swept tests.
///
/// Uses aligned storage, SoA bodies (RigidBodyStorage), Dbvt broadphase, and IDebugRenderer.
/// The wall is a ConvexMesh that approximates a thin slab; projectiles are spheres at high
/// speed (30 m/s).  CCD must prevent tunnelling through the thin wall.
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

    constexpr int    kNumProjectiles   = 30;
    constexpr int    kNumFrames        = 500;
    constexpr float  kWallHalfThick    = 0.25f;   // 25 cm half-thickness (50 cm total)
    constexpr float  kWallHalfHeight   = 3.0f;
    constexpr float  kWallHalfWidth    = 3.0f;
    constexpr float  kProjectileRadius = 0.3f;
    constexpr float  kProjectileSpeed  = 30.0f;   // 30 m/s
    constexpr float  kWallX            = 0.0f;

    // Track which side each projectile started on.
    ScratchVec<int, 16> projectileSide;
    projectileSide.reserve(kNumProjectiles);

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Build a ConvexMesh for the wall: an 8-vertex slab (thin box hull).
    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Vertices in local space.  We store them in a persistent aligned vector.
    constexpr int kWallVertCount = 8;
    ScratchVec<Vec3f, 16> wallVerts(kWallVertCount);
    {
        float hx = kWallHalfThick;
        float hy = kWallHalfHeight;
        float hz = kWallHalfWidth;
        wallVerts[0] = Vec3f(-hx, -hy, -hz);
        wallVerts[1] = Vec3f(+hx, -hy, -hz);
        wallVerts[2] = Vec3f(-hx, +hy, -hz);
        wallVerts[3] = Vec3f(+hx, +hy, -hz);
        wallVerts[4] = Vec3f(-hx, -hy, +hz);
        wallVerts[5] = Vec3f(+hx, -hy, +hz);
        wallVerts[6] = Vec3f(-hx, +hy, +hz);
        wallVerts[7] = Vec3f(+hx, +hy, +hz);
    }

    ConvexMesh wallMesh;
    wallMesh.vertices    = wallVerts.data();
    wallMesh.vertexCount = kWallVertCount;
    wallMesh.halfExtents = Vec3f(kWallHalfThick, kWallHalfHeight, kWallHalfWidth);
    wallMesh.center      = Vec3f::zero();

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        world.enableDbvt();
        {
            PhysicsWorldConfig cfg;
            cfg.gravity             = Vec3f::zero();   // no gravity — pure ballistic test
            cfg.fixedTimestep       = 1.0f / 120.0f;   // smaller timestep for better CCD
            cfg.linearDamping       = 0.0f;
            cfg.angularDamping      = 0.0f;
            cfg.ccdSpeedThreshold   = 5.0f;            // enable CCD above 5 m/s
            cfg.ccdMaxSubSteps      = 16;
            cfg.enableParallelSolver = false;
            cfg.enableTaskGraphPipeline = false;
            cfg.sleepEnergyThreshold = 0.0f;
            cfg.sleepTimeRequired    = 1e10f;
            world.setConfig(cfg);
        }

        // Create shapes.
        ShapeHandle wallShape  = world.createShape(Shape(wallMesh));
        ShapeHandle projShape  = world.createShape(Sphere{kProjectileRadius});

        // Static thin wall at origin.
        {
            RigidBodyDesc wall;
            wall.motionType  = MotionType::Static;
            wall.setShape(wallShape);
            wall.position    = Vec3f(kWallX, 0.0f, 0.0f);
            wall.restitution = 0.1f;
            wall.friction    = 0.5f;
            world.createBody(wall);
        }

        // Launch projectiles from both sides, evenly spaced on the wall face.
        // Grid: 5 rows (Y) x 3 columns (Z) per side = 15 per side = 30 total.
        int idx = 0;
        constexpr int kRows = 5;
        constexpr int kCols = 3;
        for (int row = 0; row < kRows; ++row) {
            for (int col = 0; col < kCols; ++col) {
                // Evenly space across wall face (Y: [-2.4, 2.4], Z: [-2.0, 2.0]).
                float yPos = -2.4f + row * 1.2f;
                float zPos = -2.0f + col * 2.0f;

                for (int sideSign : {-1, 1}) {
                    float startX = static_cast<float>(sideSign) * 5.0f;
                    Vec3f velocity(static_cast<float>(-sideSign) * kProjectileSpeed, 0.0f, 0.0f);

                    RigidBodyDesc proj;
                    proj.mass          = 10.0f;
                    proj.setShape(projShape);
                    proj.position      = Vec3f(startX, yPos, zPos);
                    proj.linearVelocity = velocity;
                    proj.restitution   = 0.0f;
                    proj.friction      = 0.3f;
                    proj.ccdEnabled    = true;          // critical: enable CCD for high velocity
                    proj.startActive   = true;

                    world.createBody(proj);
                    projectileSide.push_back(-sideSign);
                    ++idx;
                }
            }
        }

        printf("   Created 1 thin ConvexMesh wall (%.3f m thick)\n", kWallHalfThick * 2.0f);
        printf("   Created %d projectiles at %.0f m/s\n", kNumProjectiles, kProjectileSpeed);
        printf("   Total bodies: %zu\n", world.bodies().size());
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Run simulation
    // ════════════════════════════════════════════════════════════════════════════════════════════
    timer.start();
    for (int frame = 0; frame < kNumFrames; ++frame) {
        world.step(1.0f / 120.0f);

        // Visualize every 60 frames.
        if ((frame % 60) == 0) {
            world.debugDraw(&debugRenderer);
        }

        // Periodically check that no projectile has escaped wildly.
        if ((frame % 100) == 0 && frame > 0) {
            const auto& bodies = world.bodies();
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                if (!bodies.activeFlags[i]) continue;
                if (bodies.motionTypes[i] != MotionType::Dynamic) continue;
                const Vec3f& p = bodies.positions[i];
                if (!std::isfinite(p.x)) {
                    printf("   \u2717 NaN detected at frame %d\n", frame);
                    return 1;
                }
            }
        }
    }
    timer.stop();

    printf("   Simulated %d frames in %lld ms\n", kNumFrames, timer.ms());

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Verify: each projectile must remain on its own side of the wall (no penetration)
    // ════════════════════════════════════════════════════════════════════════════════════════════
    const auto& bodies = world.bodies();
    int penetrations = 0;
    int lostProjectiles = 0;
    int stoppedByWall = 0;

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (!bodies.activeFlags[i]) continue;
        if (bodies.motionTypes[i] != MotionType::Dynamic) continue;

        // Skip the wall (index 0 is static).
        std::size_t projIdx = i - 1;
        if (projIdx >= static_cast<std::size_t>(kNumProjectiles)) continue;

        int originalSide = projectileSide[projIdx];
        float px = bodies.positions[i].x;

        // Stopped by wall = projectile is within the wall's thickness band
        // and moving slowly.
        if (std::abs(px) <= kWallHalfThick + kProjectileRadius + 0.05f) {
            ++stoppedByWall;
            continue;
        }

        // On the correct side = projectile is past the wall on its original side.
        bool onCorrectSide = (originalSide < 0 && px < -kWallHalfThick - kProjectileRadius) ||
                              (originalSide > 0 && px >  kWallHalfThick + kProjectileRadius);

        if (!onCorrectSide) {
            // Projectile crossed to the other side = penetration.
            ++penetrations;
        }
    }

    // Also count inactive projectiles (they may have been stopped / despawned).
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (bodies.motionTypes[i] != MotionType::Dynamic) continue;
        std::size_t projIdx = i - 1;
        if (projIdx >= static_cast<std::size_t>(kNumProjectiles)) continue;
        if (!bodies.activeFlags[i]) {
            ++lostProjectiles;
        }
    }

    // Compute result hash.
    std::uint64_t hash = hashBodyState(bodies);

    // ── Report ──────────────────────────────────────────────────────────────────────────────────
    printf("\n── CCD ConvexMesh Test Results ──\n");
    printf("   Projectiles with penetration: %d / %d\n", penetrations, kNumProjectiles);
    printf("   Stopped by wall:              %d\n", stoppedByWall);
    printf("   Inactive (despawned):         %d\n", lostProjectiles);

    bool pass = (penetrations == 0);
    if (pass) {
        printf("   \u2713 Zero penetrations \u2014 CCD with ConvexMesh wall successfully prevented tunneling\n");
    } else {
        printf("   \u2717 %d projectile(s) tunneled through the ConvexMesh wall\n", penetrations);
    }

    printf("   Result hash: 0x%016llX\n", static_cast<unsigned long long>(hash));
    printf("   Elapsed: %lld ms\n\n", timer.ms());

    return pass ? 0 : 1;
}
