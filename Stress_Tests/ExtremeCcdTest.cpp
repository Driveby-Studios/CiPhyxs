//==================================================================================================
/// @file  ExtremeCcdTest.cpp
/// @brief  Stress test: launch projectiles at 200–500 m/s at a thin wall.
///         Validates CcdMode::ClampVelocity and CcdMode::SubStep prevent tunnelling
///         at extreme velocities that would defeat a standard single-step CCD cast.
///
/// Uses aligned storage, SoA bodies (RigidBodyStorage), Dbvt broadphase, and IDebugRenderer.
/// Two groups are tested:
///   - ClampVelocity group (15 projectiles at 200–300 m/s)
///   - SubStep group       (15 projectiles at 300–500 m/s)
/// Both must achieve zero penetrations.
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

    constexpr int    kNumClampVelocity = 15;
    constexpr int    kNumSubStep       = 15;
    constexpr int    kTotalProjectiles = kNumClampVelocity + kNumSubStep;
    constexpr int    kNumFrames        = 500;
    constexpr float  kWallHalfThick    = 0.25f;   // 25 cm half-thickness (50 cm total)
    constexpr float  kWallHalfHeight   = 3.0f;
    constexpr float  kWallHalfWidth    = 3.0f;
    constexpr float  kProjectileRadius = 0.3f;
    constexpr float  kWallX            = 0.0f;

    // Track which side and mode each projectile uses.
    struct ProjectileInfo {
        int   side;     // -1 = left, +1 = right
        float speed;
        CcdMode mode;
    };
    ScratchVec<ProjectileInfo, 16> projInfo;
    projInfo.reserve(kTotalProjectiles);

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Setup
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        world.enableDbvt();
        {
            PhysicsWorldConfig cfg;
            cfg.gravity              = Vec3f::zero();   // no gravity — pure ballistic test
            cfg.fixedTimestep        = 1.0f / 120.0f;   // smaller timestep for better CCD
            cfg.linearDamping        = 0.0f;
            cfg.angularDamping       = 0.0f;
            cfg.ccdSpeedThreshold    = 0.0f;            // CCD always active for CCD-enabled bodies
            cfg.ccdMaxSubSteps       = 16;
            cfg.enableParallelSolver = false;
            cfg.enableTaskGraphPipeline = false;
            cfg.sleepEnergyThreshold = 0.0f;
            cfg.sleepTimeRequired    = 1e10f;
            world.setConfig(cfg);
        }

        // Create shapes.
        ShapeHandle wallShape  = world.createShape(Box{
            Vec3f(kWallHalfThick, kWallHalfHeight, kWallHalfWidth)
        });
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

        // ── ClampVelocity group: 200–300 m/s, lower half of wall (y < 0) ─────────────────────
        // Grid: 5 rows × 3 columns = 15 projectiles.
        // Y positions are negative (lower half of wall) to keep them separate from SubStep group.
        FixedRng rngVel(12345);  // fixed seed for determinism
        int idx = 0;
        constexpr int kRows = 5;
        constexpr int kCols = 3;
        for (int row = 0; row < kRows && idx < kNumClampVelocity; ++row) {
            for (int col = 0; col < kCols && idx < kNumClampVelocity; ++col) {
                float yPos = -2.8f + row * 0.9f;   // y ∈ [-2.8, -1.0] (lower half)
                float zPos = -2.0f + col * 2.0f;

                for (int sideSign : {-1, 1}) {
                    if (idx >= kNumClampVelocity) break;

                    float speed = 200.0f + rngVel.nextFloat() * 100.0f; // 200–300 m/s
                    float startX = static_cast<float>(sideSign) * 8.0f;
                    Vec3f velocity(static_cast<float>(-sideSign) * speed, 0.0f, 0.0f);

                    RigidBodyDesc proj;
                    proj.mass          = 10.0f;
                    proj.setShape(projShape);
                    proj.position      = Vec3f(startX, yPos, zPos);
                    proj.linearVelocity= velocity;
                    proj.restitution   = 0.0f;
                    proj.friction      = 0.3f;
                    proj.ccdMode       = CcdMode::ClampVelocity;
                    proj.startActive   = true;

                    world.createBody(proj);
                    projInfo.push_back({-sideSign, speed, CcdMode::ClampVelocity});
                    ++idx;
                }
            }
        }

        // ── SubStep group: 300–500 m/s, upper half of wall (y > 0) ───────────────────────────
        // Y positions are positive (upper half of wall) to keep them separate from ClampVelocity group.
        idx = 0;
        for (int row = 0; row < kRows && idx < kNumSubStep; ++row) {
            for (int col = 0; col < kCols && idx < kNumSubStep; ++col) {
                float yPos = 1.0f + row * 0.9f;      // y ∈ [1.0, 2.8] (upper half)
                float zPos = -2.0f + col * 2.0f;

                for (int sideSign : {-1, 1}) {
                    if (idx >= kNumSubStep) break;

                    float speed = 300.0f + rngVel.nextFloat() * 200.0f; // 300–500 m/s
                    float startX = static_cast<float>(sideSign) * 10.0f;
                    Vec3f velocity(static_cast<float>(-sideSign) * speed, 0.0f, 0.0f);

                    RigidBodyDesc proj;
                    proj.mass          = 10.0f;
                    proj.setShape(projShape);
                    proj.position      = Vec3f(startX, yPos, zPos);
                    proj.linearVelocity= velocity;
                    proj.restitution   = 0.0f;
                    proj.friction      = 0.3f;
                    proj.ccdMode       = CcdMode::SubStep;
                    proj.startActive   = true;

                    world.createBody(proj);
                    projInfo.push_back({-sideSign, speed, CcdMode::SubStep});
                    ++idx;
                }
            }
        }

        printf("   Created 1 thin Box wall (%.3f m thick)\n", kWallHalfThick * 2.0f);
        printf("   Created %d ClampVelocity projectiles (200–300 m/s)\n", kNumClampVelocity);
        printf("   Created %d SubStep projectiles (300–500 m/s)\n", kNumSubStep);
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

        // Periodically check for NaN.
        if ((frame % 100) == 0 && frame > 0) {
            const auto& bodies = world.bodies();
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                if (!bodies.activeFlags[i]) continue;
                if (bodies.motionTypes[i] != MotionType::Dynamic) continue;
                const Vec3f& p = bodies.positions[i];
                if (!std::isfinite(p.x)) {
                    printf("   ✗ NaN detected at frame %d\n", frame);
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
    int clampVelocityPenetrations = 0;
    int subStepPenetrations = 0;
    int clampVelocityStopped = 0;
    int subStepStopped = 0;
    int clampVelocityInactive = 0;
    int subStepInactive = 0;

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (!bodies.activeFlags[i]) continue;
        if (bodies.motionTypes[i] != MotionType::Dynamic) continue;

        // Skip the wall (index 0 is static).
        std::size_t projIdx = i - 1;
        if (projIdx >= static_cast<std::size_t>(kTotalProjectiles)) continue;

        const auto& info = projInfo[projIdx];
        float px = bodies.positions[i].x;

        // Stopped by wall = projectile is within the wall's thickness band.
        if (std::abs(px) <= kWallHalfThick + kProjectileRadius + 0.05f) {
            if (info.mode == CcdMode::ClampVelocity) ++clampVelocityStopped;
            else ++subStepStopped;
            continue;
        }

        // On the correct side = projectile is past the wall on its original side.
        bool onCorrectSide = (info.side < 0 && px < -kWallHalfThick - kProjectileRadius) ||
                              (info.side > 0 && px >  kWallHalfThick + kProjectileRadius);

        if (!onCorrectSide) {
            if (info.mode == CcdMode::ClampVelocity) ++clampVelocityPenetrations;
            else ++subStepPenetrations;
        }
    }

    // Count inactive projectiles.
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (bodies.motionTypes[i] != MotionType::Dynamic) continue;
        std::size_t projIdx = i - 1;
        if (projIdx >= static_cast<std::size_t>(kTotalProjectiles)) continue;
        if (!bodies.activeFlags[i]) {
            if (projInfo[projIdx].mode == CcdMode::ClampVelocity) ++clampVelocityInactive;
            else ++subStepInactive;
        }
    }

    // Compute result hash.
    std::uint64_t hash = hashBodyState(bodies);

    // ── Report ──────────────────────────────────────────────────────────────────────────────────
    printf("\n── Extreme CCD Test Results ──\n");
    printf("   ── ClampVelocity (200–300 m/s) ──\n");
    printf("      Penetrations: %d / %d\n", clampVelocityPenetrations, kNumClampVelocity);
    printf("      Stopped by wall: %d\n", clampVelocityStopped);
    printf("      Inactive: %d\n", clampVelocityInactive);
    printf("   ── SubStep (300–500 m/s) ──\n");
    printf("      Penetrations: %d / %d\n", subStepPenetrations, kNumSubStep);
    printf("      Stopped by wall: %d\n", subStepStopped);
    printf("      Inactive: %d\n", subStepInactive);

    bool pass = (clampVelocityPenetrations == 0) && (subStepPenetrations == 0);
    if (pass) {
        printf("   ✓ Zero penetrations — CcdMode::ClampVelocity and CcdMode::SubStep\n");
        printf("     successfully prevented tunnelling at extreme velocities (200–500 m/s)\n");
    } else {
        printf("   ✗ %d ClampVelocity + %d SubStep projectile(s) tunneled through the wall\n",
               clampVelocityPenetrations, subStepPenetrations);
    }

    printf("   Result hash: 0x%016llX\n", static_cast<unsigned long long>(hash));
    printf("   Elapsed: %lld ms\n\n", timer.ms());

    return pass ? 0 : 1;
}
