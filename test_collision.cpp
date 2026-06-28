#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

// ── Helper: approximate float equality ─────────────────────────────────────────────────────────
bool approx(float a, float b, float eps = 1e-4f) { return std::abs(a - b) < eps; }

int main() {
    using namespace ciphyxs;

    // ── 1. Shape registry + body creation ──────────────────────────────────────────────────────
    {
        PhysicsWorld world;

        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        world.createShape(Sphere{0.5f});

        printf("Shapes created: %zu\n", world.shapes().size()); // expect 3

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        ground.position    = Vec3f(0, -0.5f, 0);
        world.createBody(ground);

        RigidBodyDesc box;
        box.mass         = 10.0f;
        box.setShape(hBox);
        box.position     = Vec3f(0, 5, 0);
        box.restitution  = 0.0f;
        auto b = world.createBody(box);

        printf("Bodies: %zu\n", world.bodies().size()); // expect 2

        // Simulate.
        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);  // 5 seconds

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;

        printf("Box after 5s: pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)\n",
               pos[b].x, pos[b].y, pos[b].z,
               vel[b].x, vel[b].y, vel[b].z);
        printf("Contact manifolds: %zu\n", world.manifolds().size());

        // The box should have landed on the plane and be at rest.
        bool resting = vel[b].lengthSquared() < 0.01f;
        printf("Box resting: %s\n", resting ? "YES" : "NO");

        // Box centre should be ~0.5 above plane (half-extent on y).
        printf("Box y=%.3f (expected ~0.0 to 0.5)\n", pos[b].y);
    }

    // ── 2. Sphere-sphere collision ─────────────────────────────────────────────────────────────
    {
        PhysicsWorld world;

        ShapeHandle hSphere = world.createShape(Sphere{1.0f});

        RigidBodyDesc sphereA;
        sphereA.mass        = 5.0f;
        sphereA.setShape(hSphere);
        sphereA.position    = Vec3f(-1.5f, 5.0f, 0);
        sphereA.restitution = 0.0f;
        auto a = world.createBody(sphereA);

        RigidBodyDesc sphereB;
        sphereB.mass        = 5.0f;
        sphereB.setShape(hSphere);
        sphereB.position    = Vec3f(1.5f, 5.0f, 0);
        sphereB.restitution = 0.0f;
        auto b = world.createBody(sphereB);

        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;

        printf("\nSphere-sphere:\n");
        printf("  A pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)\n",
               pos[a].x, pos[a].y, pos[a].z, vel[a].x, vel[a].y, vel[a].z);
        printf("  B pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)\n",
               pos[b].x, pos[b].y, pos[b].z, vel[b].x, vel[b].y, vel[b].z);

        // Spheres should have collided (they attract via gravity), be separated by ~2.0 radii.
        float dist = (pos[a] - pos[b]).length();
        printf("  Distance: %.3f (expected ~2.0)\n", dist);

        bool resting = (vel[a].lengthSquared() + vel[b].lengthSquared()) < 0.01f;
        printf("  Resting: %s\n", resting ? "YES" : "NO");
    }

    // ── 3. Box-box SAT + stacking ──────────────────────────────────────────────────────────────
    {
        PhysicsWorld world;

        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        ground.position    = Vec3f(0, -0.5f, 0);
        world.createBody(ground);

        // Stack two boxes.
        RigidBodyDesc box1;
        box1.mass        = 10.0f;
        box1.setShape(hBox);
        box1.position    = Vec3f(0, 0.6f, 0);
        box1.restitution = 0.0f;
        auto b1 = world.createBody(box1);

        RigidBodyDesc box2;
        box2.mass        = 5.0f;
        box2.setShape(hBox);
        box2.position    = Vec3f(0, 1.7f, 0);
        box2.restitution = 0.0f;
        auto b2 = world.createBody(box2);

        for (int i = 0; i < 600; ++i) world.step(1.0f / 60.0f);  // 10 seconds

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;

        printf("\nBox-box stacking:\n");
        printf("  Box1 pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)\n",
               pos[b1].x, pos[b1].y, pos[b1].z, vel[b1].x, vel[b1].y, vel[b1].z);
        printf("  Box2 pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)\n",
               pos[b2].x, pos[b2].y, pos[b2].z, vel[b2].x, vel[b2].y, vel[b2].z);
        printf("  Manifolds: %zu\n", world.manifolds().size());

        // Both should be resting.
        auto& angVel = world.bodies().angularVelocities;
        float e1 = vel[b1].lengthSquared() + angVel[b1].lengthSquared();
        float e2 = vel[b2].lengthSquared() + angVel[b2].lengthSquared();
        printf("  Box1 energy: %.6f%s\n", e1, e1 < 0.01f ? " (resting)" : "");
        printf("  Box2 energy: %.6f%s\n", e2, e2 < 0.01f ? " (resting)" : "");
    }

    // ── 4. Determinism check ──────────────────────────────────────────────────────────────────
    {
        PhysicsWorld world;
        ShapeHandle hPlane = world.createShape(Plane{});
        ShapeHandle hBox   = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(hPlane);
        ground.position    = Vec3f(0, -0.5f, 0);
        world.createBody(ground);

        RigidBodyDesc box;
        box.mass        = 10.0f;
        box.setShape(hBox);
        box.position    = Vec3f(0, 5, 0);
        auto b = world.createBody(box);

        // Run simulation once.
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60.0f);
        Vec3f posA = world.bodies().positions[b];

        // Run again in a fresh world with identical inputs.
        PhysicsWorld world2;
        world2.createShape(Plane{});
        world2.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        RigidBodyDesc ground2;
        ground2.motionType  = MotionType::Static;
        ground2.setShape(ShapeHandle(0));
        ground2.position    = Vec3f(0, -0.5f, 0);
        world2.createBody(ground2);
        RigidBodyDesc box2;
        box2.mass        = 10.0f;
        box2.setShape(ShapeHandle(1));
        box2.position    = Vec3f(0, 5, 0);
        auto b2 = world2.createBody(box2);
        for (int i = 0; i < 60; ++i) world2.step(1.0f / 60.0f);
        Vec3f posB = world2.bodies().positions[b2];

        printf("\nDeterminism: pos=(%.6f, %.6f, %.6f) vs (%.6f, %.6f, %.6f) %s\n",
               posA.x, posA.y, posA.z, posB.x, posB.y, posB.z,
               (posA - posB).lengthSquared() < 1e-10f ? "IDENTICAL" : "DIFFERENT");
    }

    // ── 5. Custom solver hook ──────────────────────────────────────────────────────────────────
    {
        struct GravityHook : ISolverHook {
            Vec3f extraGravity;
            GravityHook(Vec3f g) : extraGravity(g) {}
            void onApplyForces(float, RigidBodyStorage& bodies) override {
                for (std::size_t i = 0; i < bodies.size(); ++i) {
                    if (bodies.activeFlags[i] && bodies.motionTypes[i] == MotionType::Dynamic) {
                        float invM = bodies.inverseMasses[i];
                        if (invM > 0.0f) bodies.forces[i] += extraGravity / invM;
                    }
                }
            }
            void onPreSolve(float, RigidBodyStorage&) override {}
            void onPostSolve(float, RigidBodyStorage&) override {}
        };

        PhysicsWorld world;
        GravityHook hook(Vec3f(0, -5.0f, 0));  // extra downward acceleration
        world.addHook(&hook);

        ShapeHandle hSphere = world.createShape(Sphere{0.5f});
        RigidBodyDesc sphere;
        sphere.mass = 1.0f;
        sphere.setShape(hSphere);
        sphere.position = Vec3f(0, 5, 0);
        world.createBody(sphere);

        for (int i = 0; i < 60; ++i) world.step(1.0f / 60.0f);

        printf("\nHook test: step count = %llu\n", (unsigned long long)world.stepCount());
        world.removeHook(&hook);
        printf("Hook removed successfully.\n");
    }

    printf("\n── All tests complete ──\n");
    return 0;
}
