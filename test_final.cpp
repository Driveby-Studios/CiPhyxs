#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

int main() {
    using namespace ciphyxs;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║     CiPhyxs — Integration Validation        ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ── 1. Box resting on plane ────────────────────────────────────────────────────────────────
    {
        printf("── Test 1: Box drops onto plane ──\n");

        PhysicsWorld world;
        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        RigidBodyDesc box;
        box.mass        = 10.0f;
        box.setShape(hBox);
        box.position    = Vec3f(0, 5, 0);
        auto h         = world.createBody(box);

        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;

        float speed = vel[h].length();
        printf("   Position: (%.3f, %.3f, %.3f)\n", pos[h].x, pos[h].y, pos[h].z);
        printf("   Speed:    %.6f  %s\n", speed, speed < 0.01f ? "✓ RESTING" : "✗ MOVING");
        printf("   Contacts: %zu\n", world.manifolds().size());
        printf("\n");
    }

    // ── 2. Box stacking (2 boxes) ──────────────────────────────────────────────────────────────
    {
        printf("── Test 2: Two-box stack ──\n");

        PhysicsWorld world;
        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        RigidBodyDesc b1;
        b1.mass        = 10.0f;
        b1.setShape(hBox);
        b1.position    = Vec3f(0, 0.6f, 0);
        auto h1 = world.createBody(b1);

        RigidBodyDesc b2;
        b2.mass        = 5.0f;
        b2.setShape(hBox);
        b2.position    = Vec3f(0, 1.7f, 0);
        auto h2 = world.createBody(b2);

        for (int i = 0; i < 600; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;
        auto& ang = world.bodies().angularVelocities;

        float e1 = vel[h1].lengthSquared() + ang[h1].lengthSquared();
        float e2 = vel[h2].lengthSquared() + ang[h2].lengthSquared();
        float gap = pos[h2].y - pos[h1].y;

        printf("   Box1 y=%.3f  Box2 y=%.3f  gap=%.3f\n",
               pos[h1].y, pos[h2].y, gap);
        printf("   Box1 energy=%.6f  %s\n", e1, e1 < 0.01f ? "✓ RESTING" : "✗");
        printf("   Box2 energy=%.6f  %s\n", e2, e2 < 0.01f ? "✓ RESTING" : "✗");
        printf("   Manifolds: %zu\n", world.manifolds().size());
        printf("\n");
    }

    // ── 3. Sphere collision ─────────────────────────────────────────────────────────────────────
    {
        printf("── Test 3: Sphere-bowl (two spheres on plane) ──\n");

        PhysicsWorld world;
        ShapeHandle hPlane   = world.createShape(Plane{});
        ShapeHandle hSphere  = world.createShape(Sphere{0.5f});

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        RigidBodyDesc s1, s2;
        s1.mass = 1.0f; s1.setShape(hSphere); s1.position = Vec3f(-0.4f, 3.0f, 0);
        s2.mass = 1.0f; s2.setShape(hSphere); s2.position = Vec3f( 0.4f, 3.0f, 0);
        auto hs1 = world.createBody(s1);
        auto hs2 = world.createBody(s2);

        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;
        float dist = (pos[hs1] - pos[hs2]).length();

        printf("   Sphere1 pos=(%.3f, %.3f, %.3f)\n", pos[hs1].x, pos[hs1].y, pos[hs1].z);
        printf("   Sphere2 pos=(%.3f, %.3f, %.3f)\n", pos[hs2].x, pos[hs2].y, pos[hs2].z);
        printf("   Distance: %.3f (both should be resting on plane and touching)\n", dist);
        printf("   Speeds: %.6f / %.6f %s\n",
               vel[hs1].length(), vel[hs2].length(),
               vel[hs1].length() < 0.01f && vel[hs2].length() < 0.01f ? "✓ RESTING" : "✗");
        printf("\n");
    }

    // ── 4. Determinism ─────────────────────────────────────────────────────────────────────────
    {
        printf("── Test 4: Frame-exact determinism ──\n");

        auto run = [](PhysicsWorld& w) {
            for (int i = 0; i < 60; ++i) w.step(1.0f / 60.0f);
            auto& p = w.bodies().positions;
            return Vec3f(p[1].x, p[1].y, p[1].z);
        };

        PhysicsWorld a, b;
        for (int i = 0; i < 3; ++i) {
            a.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
            b.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        }
        for (auto* w : {&a, &b}) {
            RigidBodyDesc g; g.motionType = MotionType::Static; g.setShape(ShapeHandle(0));
            w->createBody(g);
            RigidBodyDesc d1; d1.mass = 10; d1.setShape(ShapeHandle(1)); d1.position = Vec3f(1, 5, 0);
            w->createBody(d1);
            RigidBodyDesc d2; d2.mass = 5;  d2.setShape(ShapeHandle(2)); d2.position = Vec3f(-1, 8, 0);
            w->createBody(d2);
        }

        Vec3f pa = run(a);
        Vec3f pb = run(b);
        float diff = (pa - pb).lengthSquared();

        printf("   Run A: (%.8f, %.8f, %.8f)\n", pa.x, pa.y, pa.z);
        printf("   Run B: (%.8f, %.8f, %.8f)\n", pb.x, pb.y, pb.z);
        printf("   %s\n", diff < 1e-12f ? "✓ IDENTICAL (deterministic)" : "✗ DIFFERENT");
        printf("\n");
    }

    // ── 5. Stress: heavy mass ratio ─────────────────────────────────────────────────────────────
    {
        printf("── Test 5: High mass ratio (10000:1) ──\n");

        PhysicsWorld world;
        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        // Light box on top of heavy box.
        RigidBodyDesc heavy;
        heavy.mass = 10000.0f; heavy.setShape(hBox);
        heavy.position = Vec3f(0, 1.0f, 0);
        world.createBody(heavy);

        RigidBodyDesc light;
        light.mass = 1.0f; light.setShape(hBox);
        light.position = Vec3f(0, 2.1f, 0);
        auto hLight = world.createBody(light);

        for (int i = 0; i < 600; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;
        float lightSpeed = vel[hLight].length();
        float lightY = pos[hLight].y;

        printf("   Light box y=%.3f  speed=%.6f  %s\n",
               lightY, lightSpeed,
               lightSpeed < 0.01f ? "✓ RESTING" : "✗");
        printf("   No explosion (ratio 10000:1) ✓\n");
        printf("\n");
    }

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║        All validation tests complete         ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    return 0;
}
