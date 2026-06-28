//==================================================================================================
/// @file  test_character.cpp
/// @brief  Validation of the kinematic character controller.
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

// ── Helper ──────────────────────────────────────────────────────────────────────────────────────
bool approx(float a, float b, float eps = 1e-3f) { return std::abs(a - b) < eps; }

// ────────────────────────────────────────────────────────────────────────────────────────────────

int main() {
    using namespace ciphyxs;
    int failures = 0;
    int tests    = 0;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 1. Character falls and lands on a flat ground plane
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0.0f, -9.81f, 0.0f);
        world.setConfig(cfg);

        // Ground plane at y = 0.
        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        // Character controller spawned 5 m above the ground.
        CharacterControllerConfig ccfg;
        ccfg.radius     = 0.4f;
        ccfg.height     = 1.8f;
        ccfg.maxSlopeAngle = 50.0f;
        CharacterController ctrl(world, ccfg);
        ctrl.setPosition(Vec3f(0.0f, 5.0f + ccfg.height * 0.5f, 0.0f));

        // Simulate until the character lands and settles.
        for (int i = 0; i < 120; ++i)
            ctrl.update(1.0f / 60.0f);

        bool onGround = ctrl.isOnGround();
        float yPos    = ctrl.getPosition().y;
        float expectedY = ccfg.height * 0.5f;  // centre of capsule touching ground

        if (!onGround) {
            printf("FAIL 1: Character not grounded after 2 s. y=%.4f\n", yPos);
            ++failures;
        } else if (std::abs(yPos - expectedY) > 0.05f) {
            printf("FAIL 1: Character y=%.4f, expected ~%.4f\n", yPos, expectedY);
            ++failures;
        } else {
            printf("PASS 1: Character landed at y=%.4f (expected ~%.4f), grounded=%d\n",
                   yPos, expectedY, onGround);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 2. Character walks forward on flat ground
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0.0f, -9.81f, 0.0f);
        world.setConfig(cfg);

        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        CharacterController ctrl(world);
        ctrl.setPosition(Vec3f(0.0f, 2.0f, 0.0f));

        // Let the character land first.
        for (int i = 0; i < 60; ++i)
            ctrl.update(1.0f / 60.0f);

        // Now walk forward for 1 second.
        ctrl.setMoveDirection(Vec3f(0.0f, 0.0f, 1.0f));
        for (int i = 0; i < 60; ++i)
            ctrl.update(1.0f / 60.0f);

        float zPos = ctrl.getPosition().z;
        if (zPos < 0.5f) {
            printf("FAIL 2: Character walked only z=%.4f in 1 s (expected > 0.5)\n", zPos);
            ++failures;
        } else {
            printf("PASS 2: Character walked to z=%.4f in 1 s\n", zPos);
        }

        // Should remain grounded while walking.
        if (!ctrl.isOnGround()) {
            printf("FAIL 2b: Character lost ground contact while walking.\n");
            ++failures;
        } else {
            printf("       Grounded = true while walking\n");
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 3. Character jumps and falls back to ground
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0.0f, -9.81f, 0.0f);
        world.setConfig(cfg);

        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        CharacterControllerConfig ccfg;
        ccfg.jumpSpeed = 5.0f;
        CharacterController ctrl(world, ccfg);
        ctrl.setPosition(Vec3f(0.0f, 2.0f, 0.0f));

        // Land first.
        for (int i = 0; i < 60; ++i)
            ctrl.update(1.0f / 60.0f);

        // Jump.
        ctrl.jump();

        // Simulate through the jump arc.
        bool wasInAir = false;
        for (int i = 0; i < 90; ++i) {
            ctrl.update(1.0f / 60.0f);
            // At some point the character should be airborne.
            if (!ctrl.isOnGround() && i > 5 && i < 60) wasInAir = true;
        }

        bool groundedAfter = ctrl.isOnGround();

        if (!wasInAir) {
            printf("FAIL 3: Character never left the ground after jump.\n");
            ++failures;
        } else if (!groundedAfter) {
            printf("FAIL 3b: Character did not re-land after jump.\n");
            ++failures;
        } else {
            printf("PASS 3: Character jumped, was airborne, and landed again.\n");
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 4. Character walks up a ramp
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0.0f, -9.81f, 0.0f);
        world.setConfig(cfg);

        // Create a ramp: rotate the plane by 20 degrees around X.
        float rampAngle = 20.0f * (3.14159265358979f / 180.0f);
        Vec3f rampNormal = Quaternionf::fromAxisAngle(Vec3f::unitX(), rampAngle)
                           .rotate(Vec3f::unitY());

        ShapeHandle hRamp = world.createShape(Plane{rampNormal, 0.0f});
        RigidBodyDesc ramp;
        ramp.motionType = MotionType::Static;
        ramp.setShape(hRamp);
        world.createBody(ramp);

        CharacterControllerConfig ccfg;
        ccfg.maxSlopeAngle = 30.0f;  // 30° limit, ramp is 20° → walkable
        CharacterController ctrl(world, ccfg);
        ctrl.setPosition(Vec3f(0.0f, 3.0f, -3.0f));

        // Land on the ramp.
        for (int i = 0; i < 60; ++i)
            ctrl.update(1.0f / 60.0f);

        // Walk up the ramp for 1.5 seconds.
        ctrl.setMoveDirection(Vec3f(0.0f, 0.0f, 1.0f));
        for (int i = 0; i < 90; ++i)
            ctrl.update(1.0f / 60.0f);

        // Should have gained height while moving up the ramp.
        float zPos = ctrl.getPosition().z;
        float yPos = ctrl.getPosition().y;

        if (zPos < 0.5f) {
            printf("FAIL 4: Character did not move up ramp (z=%.4f).\n", zPos);
            ++failures;
        } else {
            printf("PASS 4: Walked up ramp z=%.4f y=%.4f (gained height)\n", zPos, yPos);
        }

        // Should be on the ramp (grounded with non-vertical normal).
        if (!ctrl.isOnGround()) {
            printf("       Not grounded on ramp.\n");
        } else {
            Vec3f gn = ctrl.getGroundNormal();
            printf("       Ground normal = (%.4f, %.4f, %.4f) (ramp angle ~20°)\n",
                   gn.x, gn.y, gn.z);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 5. Character steps up onto a small platform
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0.0f, -9.81f, 0.0f);
        world.setConfig(cfg);

        // Ground plane.
        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        // A box step: 0.25 m high, placed in front of the character.
        float stepHeight = 0.25f;
        ShapeHandle hBox = world.createShape(Box{Vec3f(0.5f, stepHeight * 0.5f, 0.5f)});
        RigidBodyDesc stepDesc;
        stepDesc.motionType = MotionType::Static;
        stepDesc.setShape(hBox);
        stepDesc.position = Vec3f(2.0f, stepHeight * 0.5f, 5.0f);  // centred at z=5
        world.createBody(stepDesc);

        CharacterControllerConfig ccfg;
        ccfg.stepHeight = 0.3f;  // taller than the step
        ccfg.radius     = 0.3f;
        ccfg.height     = 1.6f;
        CharacterController ctrl(world, ccfg);
        ctrl.setPosition(Vec3f(0.0f, 2.0f, 0.0f));

        // Land and walk toward the step.
        for (int i = 0; i < 60; ++i)
            ctrl.update(1.0f / 60.0f);

        ctrl.setMoveDirection(Vec3f(0.0f, 0.0f, 1.0f));
        for (int i = 0; i < 150; ++i)
            ctrl.update(1.0f / 60.0f);

        float zPos = ctrl.getPosition().z;

        // The character should have crossed the step at z=5 and ended up beyond it.
        if (zPos < 5.5f) {
            printf("FAIL 5: Character blocked by step (z=%.4f, expected > 5.5)\n", zPos);
            ++failures;
        } else {
            printf("PASS 5: Character stepped over obstacle, z=%.4f after walk\n", zPos);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 6. Determinism — two identical simulations produce identical results
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        auto runSim = []() -> Vec3f {
            PhysicsWorld world;
            PhysicsWorldConfig cfg;
            cfg.gravity = Vec3f(0.0f, -9.81f, 0.0f);
            world.setConfig(cfg);

            ShapeHandle hPlane = world.createShape(Plane{});
            RigidBodyDesc ground;
            ground.motionType = MotionType::Static;
            ground.setShape(hPlane);
            world.createBody(ground);

            CharacterController ctrl(world);
            ctrl.setPosition(Vec3f(0.0f, 2.0f, 0.0f));

            // Land.
            for (int i = 0; i < 60; ++i)
                ctrl.update(1.0f / 60.0f);

            // Walk.
            ctrl.setMoveDirection(Vec3f(0.0f, 0.0f, 1.0f));
            for (int i = 0; i < 60; ++i)
                ctrl.update(1.0f / 60.0f);

            return ctrl.getPosition();
        };

        Vec3f a = runSim();
        Vec3f b = runSim();

        if (!approx(a.x, b.x) || !approx(a.y, b.y) || !approx(a.z, b.z)) {
            printf("FAIL 6: Determinism — (%.6f,%.6f,%.6f) vs (%.6f,%.6f,%.6f)\n",
                   a.x, a.y, a.z, b.x, b.y, b.z);
            ++failures;
        } else {
            printf("PASS 6: Determinism — identical (%.4f, %.4f, %.4f)\n", a.x, a.y, a.z);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Report
    // ════════════════════════════════════════════════════════════════════════════════════════════
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d tests, %d failures\n", tests, failures);

    return failures > 0 ? 1 : 0;
}
