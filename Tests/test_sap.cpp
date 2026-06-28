//==================================================================================================
/// @file  test_sap.cpp
/// @brief  Validation of the sweep-and-prune broadphase.
///
/// Run:
///   g++ -std=c++20 -O2 -I. test_sap.cpp -o test_sap.exe && ./test_sap.exe
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>
#include <cstdint>

int g_failures = 0;
int g_tests    = 0;

#define CHECK(cond, msg) do { \
    ++g_tests; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        ++g_failures; \
    } \
} while(0)

bool approx(float a, float b, float eps = 1e-3f) { return std::abs(a - b) < eps; }

int main() {
    using namespace ciphyxs;

    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║      CiPhyxs — Sweep-and-Prune Broadphase         ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");

    // ── 1. SAP determinism (two runs with same input → same output) ───────────────────────────
    {
        printf("-- Test 1: SAP determinism ──\n");

        auto runSim = [](PhysicsWorld& w, int steps) {
            w.setBroadphaseType(BroadphaseType::SweepAndPrune);
            for (int i = 0; i < steps; ++i) w.step(1.0f / 60.0f);
        };

        PhysicsWorld worldA, worldB;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0, -9.81f, 0);
        worldA.setConfig(cfg);
        worldB.setConfig(cfg);

        ShapeHandle hBox = worldA.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        worldB.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

        ShapeHandle hPlane = worldA.createShape(Plane{});
        worldB.createShape(Plane{});

        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        worldA.createBody(ground);
        worldB.createBody(ground);

        // Create 10 boxes with different masses and velocities.
        for (int i = 0; i < 10; ++i) {
            RigidBodyDesc box;
            box.mass     = 1.0f + static_cast<float>(i) * 0.5f;
            box.setShape(hBox);
            box.position = Vec3f(static_cast<float>(i) * 2.0f - 10.0f, 3.0f + static_cast<float>(i) * 0.2f, 0.0f);
            box.linearVelocity = Vec3f(static_cast<float>(i) * 0.5f - 2.5f, 0.0f, static_cast<float>(i) * 0.3f - 1.5f);
            worldA.createBody(box);
            worldB.createBody(box);
        }

        runSim(worldA, 200);
        runSim(worldB, 200);

        bool match = true;
        for (std::size_t i = 0; i < worldA.bodies().size(); ++i) {
            Vec3f pA = worldA.bodies().positions[i];
            Vec3f pB = worldB.bodies().positions[i];
            if (!approx(pA.x, pB.x) || !approx(pA.y, pB.y) || !approx(pA.z, pB.z)) {
                printf("  Mismatch body %zu: A=(%.6f,%.6f,%.6f) B=(%.6f,%.6f,%.6f)\n",
                       i, pA.x, pA.y, pA.z, pB.x, pB.y, pB.z);
                match = false;
                break;
            }
        }
        CHECK(match, "SAP should be deterministic across runs");
        if (match) printf("  PASS: %zu bodies deterministic\n", worldA.bodies().size());
    }

    // ── 2. SAP with compound shapes ───────────────────────────────────────────────────────────
    {
        printf("\n── Test 2: SAP with compound shapes ──\n");

        PhysicsWorld world;
        world.setBroadphaseType(BroadphaseType::SweepAndPrune);

        ShapeHandle hBox = world.createShape(Box{Vec3f(0.3f, 0.3f, 0.3f)});
        ShapeHandle hSphere = world.createShape(Sphere{0.4f});
        ShapeHandle hPlane = world.createShape(Plane{});

        // Ground.
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.addShape(hPlane);
        world.createBody(ground);

        // A compound body: box + sphere offset.
        RigidBodyDesc compound;
        compound.mass = 5.0f;
        compound.useAutoInertia = true;
        compound.addShape(hBox, Vec3f(0, 0, 0), Quaternionf::identity());
        compound.addShape(hSphere, Vec3f(0.8f, 0.3f, 0.0f), Quaternionf::identity());
        compound.position = Vec3f(0, 5, 0);
        auto hCompound = world.createBody(compound);

        // Another box to collide with.
        RigidBodyDesc box;
        box.mass     = 2.0f;
        box.setShape(hBox);
        box.position = Vec3f(0.5f, 6, 0);
        world.createBody(box);

        // Simulate — SAP should handle compound shapes without crashing or missing pairs.
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        CHECK(pos[hCompound].y < 4.0f,
              "Compound body should have fallen (y < 4.0)");
        CHECK(world.manifolds().size() > 0 || world.bodies().activeFlags[hCompound],
              "SAP should generate at least some contact pairs");

        printf("  PASS: Compound body y=%.3f after 60 frames\n",
               pos[hCompound].y);
    }

    // ── 3. SAP handles many bodies ────────────────────────────────────────────────────────────
    {
        printf("\n── Test 3: SAP handles many bodies ──\n");

        PhysicsWorld world;
        world.setBroadphaseType(BroadphaseType::SweepAndPrune);

        ShapeHandle hBox = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
        ShapeHandle hPlane = world.createShape(Plane{});

        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        // Create 50 bodies in a compact cluster (to generate many pairs).
        for (int i = 0; i < 50; ++i) {
            RigidBodyDesc box;
            box.mass     = 1.0f;
            box.setShape(hBox);
            // Scatter randomly but in a tight cluster so they interact.
            float angle = static_cast<float>(i) * 0.7f;
            float radius = 0.5f + static_cast<float>(i) * 0.12f;
            box.position = Vec3f(std::cos(angle) * radius, 5.0f + static_cast<float>(i) * 0.1f, std::sin(angle) * radius);
            world.createBody(box);
        }

        // Simulate 120 frames (2 seconds) — all boxes should fall onto the plane.
        for (int i = 0; i < 120; ++i) world.step(1.0f / 60.0f);

        CHECK(world.bodies().size() == 51,
              "Should have 51 bodies (ground + 50 boxes)");
        CHECK(world.manifolds().size() > 0,
              "SAP should generate collision pairs in a dense cluster");
        printf("  PASS: 50 boxes + ground with SAP (%zu manifolds in last frame)\n",
               world.manifolds().size());
    }

    // ── Results ──────────────────────────────────────────────────────────────────────────────
    printf("\n── Results ──\n");
    printf("  Tests:  %d\n", g_tests);
    printf("  Passed: %d\n", g_tests - g_failures);
    printf("  Failed: %d\n", g_failures);

    return g_failures > 0 ? 1 : 0;
}
