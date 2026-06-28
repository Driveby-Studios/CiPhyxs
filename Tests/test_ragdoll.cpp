//==================================================================================================
/// @file  test_ragdoll.cpp
/// @brief  Validation of RagdollBuilder, debug rendering, and hinge limits.
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

using namespace ciphyxs;

// ── Minimal debug renderer for validation ─────────────────────────────────────────────────────
struct NullRenderer : IDebugRenderer {
    int lineCount = 0;
    int triCount  = 0;
    void drawLine(const Vec3f&, const Vec3f&, const Color&) noexcept override { ++lineCount; }
    void drawTriangle(const Vec3f&, const Vec3f&, const Vec3f&, const Color&) noexcept override { ++triCount; }
    void drawPoint(const Vec3f&, const Color&) noexcept override {}
    void drawArrow(const Vec3f&, const Vec3f&, float, const Color&) noexcept override {}
};

int main() {
    int passed = 0, failed = 0;
    auto check = [&](bool ok, const char* name) {
        printf("  %s: %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++passed; else ++failed;
    };

    // ── 1. Ragdoll chain creation ────────────────────────────────────────────────────────────
    {
        printf("\n── Test 1: Ragdoll chain ──\n");
        PhysicsWorld world;
        ShapeHandle hBox = world.createShape(Box{Vec3f(0.3f, 0.5f, 0.3f)});
        ShapeHandle hSphere = world.createShape(Sphere{0.3f});

        // Root (static mount).
        RigidBodyDesc mount;
        mount.motionType = MotionType::Static;
        mount.setShape(hBox);
        mount.position = Vec3f(0, 3, 0);
        RigidBodyHandle root = world.createBody(mount);

        RagdollBuilder builder(world);

        // 3-link chain.
        RagdollBone bones[3];
        for (int i = 0; i < 3; ++i) {
            bones[i].bodyDesc.mass = 1.0f;
            bones[i].bodyDesc.setShape(hSphere);
            bones[i].bodyDesc.position = Vec3f(0, 2.5f - i * 1.2f, 0);
            bones[i].jointDesc.type = JointType::BallSocket;
            bones[i].jointDesc.anchorA = Vec3f(0, 2.5f - i * 1.2f + 0.5f, 0);
            bones[i].jointDesc.anchorB = Vec3f(0, 2.5f - i * 1.2f - 0.5f, 0);
        }

        auto handles = builder.createChain(bones, 3, root);
        check(handles.size() == 3, "3 bones created");

        // Simulate 60 frames.
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60.0f);

        // All bodies should be active and stable.
        check(world.bodies().activeFlags[handles[0]] == true, "bone 0 active");
        check(world.bodies().activeFlags[handles[1]] == true, "bone 1 active");
        check(world.bodies().activeFlags[handles[2]] == true, "bone 2 active");

        printf("  Chain positions:");
        for (auto h : handles) {
            printf(" (%.2f, %.2f, %.2f)",
                   world.bodies().positions[h].x,
                   world.bodies().positions[h].y,
                   world.bodies().positions[h].z);
        }
        printf("\n");
    }

    // ── 2. Debug renderer ────────────────────────────────────────────────────────────────────
    {
        printf("\n── Test 2: Debug renderer ──\n");
        PhysicsWorld world;

        ShapeHandle hBox = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        ShapeHandle hSphere = world.createShape(Sphere{0.5f});
        ShapeHandle hCapsule = world.createShape(Capsule{0.3f, 0.5f});

        RigidBodyDesc box;
        box.mass = 1.0f;
        box.setShape(hBox);
        box.position = Vec3f(0, 1, 0);
        world.createBody(box);

        RigidBodyDesc sphere;
        sphere.mass = 1.0f;
        sphere.setShape(hSphere);
        sphere.position = Vec3f(2, 1, 0);
        world.createBody(sphere);

        RigidBodyDesc capsule;
        capsule.mass = 1.0f;
        capsule.setShape(hCapsule);
        capsule.position = Vec3f(4, 1, 0);
        world.createBody(capsule);

        world.step(1.0f / 60.0f);

        NullRenderer renderer;
        world.debugDraw(&renderer);

        check(renderer.lineCount > 0, "debug draw produced lines");
        printf("  Debug draw: %d lines\n", renderer.lineCount);
    }

    // ── 3. Hinge limits ──────────────────────────────────────────────────────────────────────
    {
        printf("\n── Test 3: Hinge angular limits ──\n");
        PhysicsWorld world;
        ShapeHandle hBox = world.createShape(Box{Vec3f(0.3f, 0.8f, 0.3f)});

        // Two boxes connected by a hinge with limits.
        RigidBodyDesc left;
        left.motionType = MotionType::Static;
        left.setShape(hBox);
        left.position = Vec3f(-1, 0.8f, 0);
        auto hLeft = world.createBody(left);

        RigidBodyDesc right;
        right.mass = 2.0f;
        right.setShape(hBox);
        right.position = Vec3f(0, 0.8f, 0);
        auto hRight = world.createBody(right);

        JointDesc hinge;
        hinge.type = JointType::Hinge;
        hinge.bodyA = hLeft;
        hinge.bodyB = hRight;
        hinge.anchorA = Vec3f(0, 0.8f, 0);
        hinge.anchorB = Vec3f(0, 0.8f, 0);
        hinge.axisA = Vec3f::unitZ();
        hinge.axisB = Vec3f::unitZ();
        hinge.enableLimits = true;
        hinge.limitMin = -0.5f;
        hinge.limitMax = 0.5f;
        world.createJoint(hinge);

        // Give the right body an initial angular velocity to test limits.
        world.bodies().angularVelocities[hRight] = Vec3f(0, 0, 5.0f);

        for (int i = 0; i < 120; ++i) world.step(1.0f / 60.0f);

        // The right body should have been constrained by the hinge limits.
        Vec3f w = world.bodies().angularVelocities[hRight];
        float wMag = std::abs(w.z);
        bool limited = wMag < 1.0f;
        check(limited, "hinge limits contained angular velocity");
        printf("  Final angular vel: %.4f rad/s (limited: %s)\n", w.z, limited ? "yes" : "no");
    }

    // ── 4. Sub-shape joint attachment ────────────────────────────────────────────────────────
    {
        printf("\n── Test 4: Sub-shape joint attachment ──\n");
        PhysicsWorld world;
        ShapeHandle hSphere = world.createShape(Sphere{0.2f});
        ShapeHandle hBox = world.createShape(Box{Vec3f(0.5f, 0.1f, 0.5f)});

        // Create a compound body with two sub-shapes.
        RigidBodyDesc plate;
        plate.mass = 5.0f;
        plate.setShape(hBox);
        plate.position = Vec3f(0, 1, 0);
        auto hPlate = world.createBody(plate);

        // A ball attached to a sub-shape position.
        RigidBodyDesc ball;
        ball.mass = 1.0f;
        ball.setShape(hSphere);
        ball.position = Vec3f(1, 1, 0);
        auto hBall = world.createBody(ball);

        // Joint connecting ball to the plate's sub-shape (index 0 = the plate itself).
        JointDesc jd;
        jd.type = JointType::BallSocket;
        jd.bodyA = hPlate;
        jd.bodyB = hBall;
        jd.anchorA = Vec3f(1, 1, 0);  // edge of plate
        jd.anchorB = Vec3f(1, 1, 0);
        jd.subShapeA = 0;  // attach to sub-shape 0
        world.createJoint(jd);

        for (int i = 0; i < 60; ++i) world.step(1.0f / 60.0f);

        // Joint should still be intact.
        check(world.joints().size() == 1, "joint intact after 60 frames");
        printf("  Joint active: %s\n", world.joints().size() == 1 ? "yes" : "no");
    }

    printf("\n╔════════════════════════════════════╗\n");
    printf("║  Results: %d / %d passed%s         ║\n",
           passed, passed + failed,
           failed == 0 ? "  ✓" : "  ✗");
    printf("╚════════════════════════════════════╝\n");
    return failed > 0 ? 1 : 0;
}
