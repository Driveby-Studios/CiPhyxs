//==================================================================================================
/// @file  test_joints.cpp
/// @brief  Validation of joint constraints (BallSocket, Hinge, Fixed).
///
/// IMPORTANT: For BallSocket (and by extension Hinge/Fixed), the world-space anchor points
/// on both bodies MUST be initially COINCIDENT.  The anchor defines "the point where the
/// two bodies are connected."  If the anchors don't coincide, the solver applies large
/// corrective impulses that can destabilise the simulation.
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

int main() {
    using namespace ciphyxs;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║     CiPhyxs — Joint Constraint Tests        ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ── 0. Zero-error trivial joint (sanity check) ─────────────────────────────────────────────
    {
        printf("── Test 0: Zero-error ball-socket (sanity) ──\n");
        PhysicsWorld world;
        world.createShape(Sphere{0.25f});

        RigidBodyDesc fixed;
        fixed.motionType = MotionType::Static;
        fixed.position   = Vec3f(0, 0, 0);
        world.createBody(fixed);

        RigidBodyDesc bob;
        bob.mass        = 1.0f;
        bob.setShape(0);
                bob.position    = Vec3f(0, 0, 0);  // same as anchor → zero error
        auto hBob = world.createBody(bob);

        JointDesc jd;
        jd.type    = JointType::BallSocket;
        jd.bodyA   = 0;
        jd.bodyB   = hBob;
        jd.anchorA = Vec3f(0, 0, 0);
        jd.anchorB = Vec3f(0, 0, 0);       // coincides with anchorA
        world.createJoint(jd);

        for (int i = 0; i < 60; ++i) world.step(1.0f / 60.0f);
        auto& p = world.bodies().positions;
        printf("   Bob pos: (%.3f, %.3f, %.3f)  speed: %.6f  %s\n",
               p[hBob].x, p[hBob].y, p[hBob].z,
               world.bodies().linearVelocities[hBob].length(),
               std::abs(p[hBob].y) < 1.0f ? "✓ STABLE" : "✗");
        printf("\n");
    }

    // ── 1. Pendulum (ball-socket joint) ────────────────────────────────────────────────────────
    {
        printf("── Test 1: BallSocket pendulum ──\n");

        PhysicsWorld world;
        world.createShape(Sphere{0.25f});

        // Static pivot at (0, 5, 0).
        RigidBodyDesc pivotDesc;
        pivotDesc.motionType = MotionType::Static;
        pivotDesc.position   = Vec3f(0, 5, 0);
        world.createBody(pivotDesc);  // handle 0

        // Bob at (0, 2, 0) = 3m below pivot.
        RigidBodyDesc bob;
        bob.mass        = 1.0f;
        bob.setShape(0);
                bob.position    = Vec3f(0, 2, 0);
        auto hBob = world.createBody(bob);

        // Ball-socket: the anchor point on BOTH bodies is the pivot (0, 5, 0).
        // For the bob, this means the local anchor is offset 3m above its center.
        JointDesc jd;
        jd.type    = JointType::BallSocket;
        jd.bodyA   = 0;
        jd.bodyB   = hBob;
        jd.anchorA = Vec3f(0, 5, 0);   // pivot (coincident with bodyA center)
        jd.anchorB = Vec3f(0, 5, 0);   // same world point → anchors coincide
        world.createJoint(jd);

        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);

        auto& p = world.bodies().positions;
        auto& v = world.bodies().linearVelocities;
        Vec3f pivotWorld(0, 5, 0);
        float dist = (p[hBob] - pivotWorld).length();
        float speed = v[hBob].length();

        printf("   Bob pos: (%.3f, %.3f, %.3f)\n", p[hBob].x, p[hBob].y, p[hBob].z);
        printf("   Bob speed: %.6f  %s\n", speed,
               speed < 5.0f ? "✓ SWINGING" : "✗ EXPLODED");
        printf("   Distance from pivot: %.3f (should be ~3.0)  %s\n", dist,
               std::abs(dist - 3.0f) < 0.15f ? "✓" : "✗");
        printf("\n");
    }

    // ── 2. Hinge joint (revolute door) ─────────────────────────────────────────────────────────
    {
        printf("── Test 2: Hinge joint (door) ──\n");

        PhysicsWorld world;
        world.createShape(Box{Vec3f(0.5f, 1.0f, 0.05f)});  // door panel

        // Static hinge post at origin.
        RigidBodyDesc post;
        post.motionType = MotionType::Static;
        post.position   = Vec3f(0, 0, 0);
        world.createBody(post);  // handle 0

        // Door panel, 1m wide, centered at (0.5, 0, 0) so the left edge is at the hinge.
        RigidBodyDesc door;
        door.mass        = 5.0f;
        door.setShape(0);
        door.position    = Vec3f(0.5f, 0, 0);
        auto hDoor = world.createBody(door);

        // Hinge axis along Z, at world origin.
        JointDesc jd;
        jd.type    = JointType::Hinge;
        jd.bodyA   = 0;                 // static post
        jd.bodyB   = hDoor;             // door panel
        jd.anchorA = Vec3f(0, 0, 0);    // hinge point at origin
        jd.anchorB = Vec3f(0, 0, 0);    // door's hinge edge at origin too
        jd.axisA   = Vec3f(0, 0, 1);
        jd.axisB   = Vec3f(0, 0, 1);
        world.createJoint(jd);

        // Give the door a gentle push.
        world.bodies().angularVelocities[hDoor] = Vec3f(0, 0, 1.0f);

        for (int i = 0; i < 120; ++i) world.step(1.0f / 60.0f);

        auto& p = world.bodies().positions;
        auto& w = world.bodies().angularVelocities;
        printf("   Door pos: (%.3f, %.3f, %.3f)\n", p[hDoor].x, p[hDoor].y, p[hDoor].z);
        printf("   Angular vel (Z): %.4f rad/s  %s\n", w[hDoor].z,
               std::isfinite(w[hDoor].z) ? "✓ FINITE" : "✗ EXPLODED");
        printf("\n");
    }

    // ── 3. Fixed joint (bridge segment) ────────────────────────────────────────────────────────
    {
        printf("── Test 3: Fixed joint (bridge segment) ──\n");

        PhysicsWorld world;
        world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});  // handle 0

        // Two boxes positioned 1m apart (touching), welded at the contact point.
        RigidBodyDesc left, right;
        left.mass  = 5.0f; left.setShape(0); left.position  = Vec3f(-0.5f, 3, 0);
        right.mass = 5.0f; right.setShape(0); right.position = Vec3f( 0.5f, 3, 0);
        auto hL = world.createBody(left);
        auto hR = world.createBody(right);

        // Fixed joint at the contact interface.
        JointDesc jd;
        jd.type    = JointType::Fixed;
        jd.bodyA   = hL;
        jd.bodyB   = hR;
        jd.anchorA = Vec3f(0, 3, 0);    // right face of left box
        jd.anchorB = Vec3f(0, 3, 0);    // left face of right box (coincident)
        world.createJoint(jd);

        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);

        auto& p = world.bodies().positions;
        Vec3f sep = p[hL] - p[hR];
        printf("   Box L: (%.3f, %.3f, %.3f)\n", p[hL].x, p[hL].y, p[hL].z);
        printf("   Box R: (%.3f, %.3f, %.3f)\n", p[hR].x, p[hR].y, p[hR].z);
        printf("   Separation: %.4f (expected ~1.0)  %s\n", sep.length(),
               std::abs(sep.length() - 1.0f) < 0.2f ? "✓ BONDED" : "✗ DRIFTED");
        printf("\n");
    }

    // ── 4. 3-link pendulum chain ───────────────────────────────────────────────────────────────
    {
        printf("── Test 4: 3-link pendulum chain ──\n");

        PhysicsWorld world;
        world.createShape(Sphere{0.3f});   // handle 0

        RigidBodyDesc ceiling;
        ceiling.motionType = MotionType::Static;
        ceiling.position   = Vec3f(0, 6, 0);
        world.createBody(ceiling);  // handle 0

        // Three links spaced 1m apart vertically, connected by ball-socket joints.
        RigidBodyDesc l1, l2, l3;
        l1.mass = 1.0f; l1.setShape(0); l1.position = Vec3f(0, 5, 0);
        l2.mass = 1.0f; l2.setShape(0); l2.position = Vec3f(0, 4, 0);
        l3.mass = 1.0f; l3.setShape(0); l3.position = Vec3f(0, 3, 0);
        auto h1 = world.createBody(l1);
        auto h2 = world.createBody(l2);
        auto h3 = world.createBody(l3);

        // Joints at the connection points between consecutive links.
        // Each joint's world-space anchor is at the connection interface.
        auto link = [&](RigidBodyHandle parent, RigidBodyHandle child,
                        const Vec3f& anchorWorld) {
            JointDesc jd;
            jd.type    = JointType::BallSocket;
            jd.bodyA   = parent;
            jd.bodyB   = child;
            jd.anchorA = anchorWorld;
            jd.anchorB = anchorWorld;   // same world point
            world.createJoint(jd);
        };

        link(0,  h1, Vec3f(0, 5.5f, 0));   // ceiling → link1
        link(h1, h2, Vec3f(0, 4.5f, 0));   // link1 → link2
        link(h2, h3, Vec3f(0, 3.5f, 0));   // link2 → link3

        // Nudge the bottom link sideways.
        world.bodies().linearVelocities[h3] = Vec3f(3.0f, 0, 0);

        for (int i = 0; i < 300; ++i) world.step(1.0f / 60.0f);

        auto& p = world.bodies().positions;
        printf("   Link1: (%.3f, %.3f, %.3f)\n", p[h1].x, p[h1].y, p[h1].z);
        printf("   Link2: (%.3f, %.3f, %.3f)\n", p[h2].x, p[h2].y, p[h2].z);
        printf("   Link3: (%.3f, %.3f, %.3f)\n", p[h3].x, p[h3].y, p[h3].z);

        bool stable = true;
        for (auto h : {h1, h2, h3}) {
            if (std::abs(p[h].x) > 100.0f || std::abs(p[h].y) > 100.0f) { stable = false; break; }
        }
        printf("   %s\n", stable ? "✓ CHAIN STABLE" : "✗ CHAIN EXPLODED");
        printf("\n");
    }

    // ── 5. Determinism with joints ─────────────────────────────────────────────────────────────
    {
        printf("── Test 5: Determinism with joints ──\n");

        auto run = []() -> Vec3f {
            PhysicsWorld w;
            w.createShape(Sphere{0.2f});

            RigidBodyDesc f; f.motionType = MotionType::Static; f.position = Vec3f(0, 5, 0);
            w.createBody(f);

            RigidBodyDesc b; b.mass = 1.0f; b.setShape(0); b.position = Vec3f(0, 3, 0);
            auto hB = w.createBody(b);

            JointDesc jd;
            jd.type = JointType::BallSocket;
            jd.bodyA = 0; jd.bodyB = hB;
            jd.anchorA = Vec3f(0, 5, 0);
            jd.anchorB = Vec3f(0, 5, 0);  // coincides with anchorA
            w.createJoint(jd);

            for (int i = 0; i < 60; ++i) w.step(1.0f / 60.0f);
            return w.bodies().positions[hB];
        };

        Vec3f a = run();
        Vec3f b = run();
        float diff = (a - b).lengthSquared();

        printf("   Run A: (%.8f, %.8f, %.8f)\n", a.x, a.y, a.z);
        printf("   Run B: (%.8f, %.8f, %.8f)\n", b.x, b.y, b.z);
        printf("   %s\n", diff < 1e-12f ? "✓ DETERMINISTIC" : "✗ DIFFERENT");
        printf("\n");
    }

    // ── 6. Parallel vs sequential with joints ──────────────────────────────────────────────────
    {
        printf("── Test 6: Parallel solver with joints ──\n");

        auto runWorld = [](bool parallel) {
            PhysicsWorld w;
            if (parallel) {
                w.enableSpatialHash(4.0f);
                w.enableParallelSolver(true, 4);
            }
            w.createShape(Sphere{0.2f});

            RigidBodyDesc f; f.motionType = MotionType::Static; f.position = Vec3f(0, 5, 0);
            w.createBody(f);

            RigidBodyDesc b1, b2;
            b1.mass = 1.0f; b1.setShape(0); b1.position = Vec3f(0, 3, 0);
            b2.mass = 2.0f; b2.setShape(0); b2.position = Vec3f(2, 5, 0);
            auto hB1 = w.createBody(b1);
            auto hB2 = w.createBody(b2);

            JointDesc jd1, jd2;
            jd1.type = JointType::BallSocket; jd1.bodyA = 0; jd1.bodyB = hB1;
            jd1.anchorA = Vec3f(0, 5, 0); jd1.anchorB = Vec3f(0, 5, 0);
            w.createJoint(jd1);

            jd2.type = JointType::BallSocket; jd2.bodyA = 0; jd2.bodyB = hB2;
            jd2.anchorA = Vec3f(0, 5, 0); jd2.anchorB = Vec3f(2, 5, 0);
            w.createJoint(jd2);

            for (int i = 0; i < 60; ++i) w.step(1.0f / 60.0f);
            return w.bodies().positions;
        };

        auto posSeq = runWorld(false);
        auto posPar = runWorld(true);

        bool ident = true;
        for (std::size_t i = 0; i < posSeq.size(); ++i) {
            float err = (posSeq[i] - posPar[i]).lengthSquared();
            if (err > 1e-8f) { ident = false; break; }
        }
        printf("   %s\n", ident ? "✓ SEQ == PARALLEL with joints" : "✗ MISMATCH");
        printf("\n");
    }

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║         Joint constraint tests done          ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    return 0;
}
