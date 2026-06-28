//==================================================================================================
/// @file  test_ccd.cpp
/// @brief  Validation of Continuous Collision Detection (CCD).
///
/// Run:
///   g++ -std=c++20 -O2 -I. test_ccd.cpp -o test_ccd.exe && ./test_ccd.exe
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

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

using namespace ciphyxs;

// ── Helper: create a world with zero gravity and zero damping for clean CCD tests ─────────────
PhysicsWorld makeCCDWorld() {
    PhysicsWorldConfig cfg;
    cfg.gravity         = Vec3f::zero();      // No gravity — pure CCD test
    cfg.linearDamping   = 0.0f;               // No damping — avoid CCD-obscuring forces
    cfg.angularDamping  = 0.0f;
    cfg.ccdSpeedThreshold = 0.0f;             // Always CCD (even slow bodies)
    PhysicsWorld w(cfg);
    return w;
}

int main() {
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║       CiPhyxs — Continuous Collision Detection    ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");

    // ── 1. Without CCD: fast sphere with restitution does NOT bounce ──────────────────────────
    //
    // The plane has an infinite AABB, so the broadphase always finds pairs and the
    // positional solver prevents geometric tunneling.  However, without CCD the velocity
    // is NOT reflected at the point of impact — the sphere keeps its downward velocity
    // and the positional solver fights it every frame, preventing a proper bounce.
    {
        printf("── Test 1: Without CCD, fast sphere does NOT reflect velocity ──\n");

        PhysicsWorld world = makeCCDWorld();
        ShapeHandle hPlane  = world.createShape(Plane{});
        ShapeHandle hSphere = world.createShape(Sphere{0.1f});

        RigidBodyDesc plane;
        plane.motionType = MotionType::Static;
        plane.setShape(hPlane);
        world.createBody(plane);

        // Fast sphere with restitution enabled.
        RigidBodyDesc s;
        s.mass = 1.0f;
        s.setShape(hSphere);
        s.position = Vec3f(0, 10, 0);
        s.linearVelocity = Vec3f(0, -200, 0);
        s.ccdEnabled = false;
        s.restitution = 0.8f;          // Would bounce if CCD handled it
        auto sphere = world.createBody(s);

        for (int i = 0; i < 6; ++i) world.step(1.0f / 60.0f);

        auto& vel = world.bodies().linearVelocities;
        auto& pos = world.bodies().positions;
        float sv = vel[sphere].y;
        float sy = pos[sphere].y;
        printf("  Sphere y=%.4f  vy=%.4f after 6 frames\n", sy, sv);
        printf("  Reference: CCD sphere with restitution=0.8 achieves vy=160 (Test 3)\n");

        // Without CCD, the velocity is NOT properly reflected — it should be far
        // below the CCD bounce velocity of ~160 m/s.  The positional solver keeps
        // the sphere near the surface but cannot reflect its velocity.
        CHECK(std::abs(sv) < 50.0f,
              "Without CCD, sphere velocity should be well below the CCD bounce velocity");
    }

    // ── 2. With CCD: same sphere stops at plane ───────────────────────────────────────────────
    {
        printf("\n── Test 2: With CCD, sphere stops at plane ──\n");

        PhysicsWorld world = makeCCDWorld();
        ShapeHandle hPlane  = world.createShape(Plane{});
        ShapeHandle hSphere = world.createShape(Sphere{0.1f});

        RigidBodyDesc plane;
        plane.motionType = MotionType::Static;
        plane.setShape(hPlane);
        world.createBody(plane);

        // Same setup but CCD enabled.
        RigidBodyDesc s;
        s.mass = 1.0f;
        s.setShape(hSphere);
        s.position = Vec3f(0, 10, 0);
        s.linearVelocity = Vec3f(0, -200, 0);
        s.ccdEnabled = true;
        s.restitution = 0.0f;
        auto sphere = world.createBody(s);

        for (int i = 0; i < 3; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        float sy = pos[sphere].y;
        printf("  Sphere y after 3 frames: %.4f (plane at 0)\n", sy);

        // With CCD, the sphere stops ON the plane (y ≈ 0.1, the sphere radius).
        CHECK(sy > 0.05f && sy < 0.2f,
              "With CCD, sphere should stop on the plane (y ≈ sphere radius = 0.1)");
    }

    // ── 3. CCD sphere reflects off plane with restitution ────────────────────────────────────
    {
        printf("\n── Test 3: CCD sphere reflects off plane ──\n");

        PhysicsWorld world = makeCCDWorld();
        ShapeHandle hPlane  = world.createShape(Plane{});
        ShapeHandle hSphere = world.createShape(Sphere{0.1f});

        RigidBodyDesc plane;
        plane.motionType = MotionType::Static;
        plane.setShape(hPlane);
        world.createBody(plane);

        // Sphere with restitution — should bounce back up.
        RigidBodyDesc s;
        s.mass = 1.0f;
        s.setShape(hSphere);
        s.position = Vec3f(0, 10, 0);
        s.linearVelocity = Vec3f(0, -200, 0);
        s.ccdEnabled = true;
        s.restitution = 0.8f;
        auto sphere = world.createBody(s);

        // Simulate 6 frames so the sphere has time to reflect and move back up.
        for (int i = 0; i < 6; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vel = world.bodies().linearVelocities;
        float sy = pos[sphere].y;
        float sv = vel[sphere].y;
        printf("  Sphere y=%.4f  vy=%.4f after 6 frames\n", sy, sv);

        // With restitution=0.8 and the formula v_new = v_normal_component * (-restitution),
        // after first frame: velocity reflects upward at 0.8× the incoming speed.
        // The reflection normal points UP (+Y), so after bounce: vy = 200 * 0.8 = 160.
        printf("  %s — sphere reflected up (vy ≈ 160 is ideal for restitution=0.8)\n",
               sv > 50.0f ? "REFLECTED" : "NOT REFLECTED");

        CHECK(sv > 50.0f, "Sphere should reflect upward with positive velocity");
        CHECK(sy > 0.5f, "Sphere should be well above the plane after 6 frames");
    }

    // ── 4. CCD determinism ────────────────────────────────────────────────────────────────────
    {
        printf("\n── Test 4: CCD determinism ──\n");

        auto run = [](PhysicsWorld& w, int steps) {
            for (int i = 0; i < steps; ++i) w.step(1.0f / 60.0f);
        };

        PhysicsWorld worldA = makeCCDWorld();
        PhysicsWorld worldB = makeCCDWorld();

        ShapeHandle hPlaneA  = worldA.createShape(Plane{});
        ShapeHandle hSphereA = worldA.createShape(Sphere{0.1f});
        ShapeHandle hPlaneB  = worldB.createShape(Plane{});
        ShapeHandle hSphereB = worldB.createShape(Sphere{0.1f});

        RigidBodyDesc plane;
        plane.motionType = MotionType::Static;
        plane.setShape(hPlaneA);
        worldA.createBody(plane);
        plane.setShape(hPlaneB);
        worldB.createBody(plane);

        RigidBodyDesc s;
        s.mass = 1.0f;
        s.setShape(hSphereA);
        s.position = Vec3f(0, 10, 0);
        s.linearVelocity = Vec3f(0, -200, 0);
        s.ccdEnabled = true;
        s.restitution = 0.5f;
        worldA.createBody(s);
        s.setShape(hSphereB);
        worldB.createBody(s);

        run(worldA, 10);
        run(worldB, 10);

        Vec3f pA = worldA.bodies().positions[1];
        Vec3f pB = worldB.bodies().positions[1];
        bool match = approx(pA.x, pB.x) && approx(pA.y, pB.y) && approx(pA.z, pB.z);

        CHECK(match, "CCD should be deterministic across runs");
        if (match) {
            printf("  PASS: pos=(%.6f, %.6f, %.6f) deterministic\n", pA.x, pA.y, pA.z);
        }
    }

    // ── 5. CCD combined with normal physics (gravity + damping) ──────────────────────────────
    //
    // Both CCD and non-CCD spheres end up resting on the plane because the positional
    // solver prevents tunneling against infinite-AABB planes.  The difference is that
    // CCD places the body at the exact TOI position (velocity zeroed at contact),
    // while the non-CCD body relies on oscillatory positional correction.
    //
    // We verify that CCD gives a clean rest position, and that non-CCD sphere still
    // ends up near the surface (oscillating slightly).
    {
        printf("\n── Test 5: CCD with gravity (both stop on plane, CCD cleaner) ──\n");

        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity      = Vec3f(0, -9.81f, 0);
        cfg.linearDamping = 0.5f;
        cfg.ccdSpeedThreshold = 5.0f; // CCD only for bodies faster than 5 m/s
        world.setConfig(cfg);

        ShapeHandle hPlane  = world.createShape(Plane{});
        ShapeHandle hSphere = world.createShape(Sphere{0.2f});

        RigidBodyDesc plane;
        plane.motionType = MotionType::Static;
        plane.setShape(hPlane);
        world.createBody(plane);

        // Two spheres: one with CCD, one without.
        RigidBodyDesc s1;
        s1.mass = 1.0f;
        s1.setShape(hSphere);
        s1.position = Vec3f(-2.0f, 5.0f, 0.0f);
        s1.linearVelocity = Vec3f(0, -50, 0);
        s1.ccdEnabled = true;
        s1.restitution = 0.0f;
        auto ccdSphere = world.createBody(s1);

        RigidBodyDesc s2;
        s2.mass = 1.0f;
        s2.setShape(hSphere);
        s2.position = Vec3f(2.0f, 5.0f, 0.0f);
        s2.linearVelocity = Vec3f(0, -50, 0);
        s2.ccdEnabled = false;
        s2.restitution = 0.0f;
        auto noCcdSphere = world.createBody(s2);

        for (int i = 0; i < 30; ++i) world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        float yCcd   = pos[ccdSphere].y;
        float yNoCcd = pos[noCcdSphere].y;

        printf("  CCD sphere y:    %.4f (should be on plane: y \u2248 0.2)\n", yCcd);
        printf("  No-CCD sphere y: %.4f (positionally corrected but still near surface)\n", yNoCcd);

        // Both spheres should be on or near the plane (the positional solver is robust).
        CHECK(yCcd > 0.0f && yCcd < 0.5f,
              "CCD sphere should be resting on the plane (y \u2248 sphere radius = 0.2)");
        CHECK(yNoCcd > 0.0f && yNoCcd < 1.0f,
              "Non-CCD sphere should also be near the plane (positional solver catches it)");
    }

    // ── Results ──────────────────────────────────────────────────────────────────────────────
    printf("\n── Results ──\n");
    printf("  Tests:  %d\n", g_tests);
    printf("  Passed: %d\n", g_tests - g_failures);
    printf("  Failed: %d\n", g_failures);

    return g_failures > 0 ? 1 : 0;
}
