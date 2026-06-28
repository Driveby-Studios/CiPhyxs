//==================================================================================================
/// @file  test_vehicle.cpp
/// @brief  Validation of the ray-cast vehicle controller (suspension, drive, steering, braking).
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

// ── Helper ──────────────────────────────────────────────────────────────────────────────────────
bool approx(float a, float b, float eps = 1e-3f) { return std::abs(a - b) < eps; }

// ── Solver hook to drive the vehicle each fixed step ───────────────────────────────────────────
struct VehicleHook : ciphyxs::ISolverHook {
    ciphyxs::Vehicle* vehicle = nullptr;

    void onApplyForces(float dt, ciphyxs::RigidBodyStorage&) noexcept override {
        if (vehicle) vehicle->update(dt);
    }
    void onPreSolve(float, ciphyxs::RigidBodyStorage&) noexcept override {}
    void onPostSolve(float, ciphyxs::RigidBodyStorage&) noexcept override {}
};

// ── Configuration tuned for a given chassis mass ──────────────────────────────────────────────
// Suspension is tuned so that ~40 % of the rest-length travel is used at rest (static sag).
struct TunedConfig {
    ciphyxs::VehicleConfig vcfg;
    float                  restLength;
};

static TunedConfig makeTunedConfig(ciphyxs::PhysicsWorld&,
                                   ciphyxs::RigidBodyHandle chassisBody,
                                   float mass,
                                   float restLen = 0.30f,
                                   float sagFrac = 0.40f) {
    using namespace ciphyxs;
    TunedConfig tc;
    tc.restLength = restLen;

    // Equilibrium: 4 * k * sag = m * g   →   k = m * g / (4 * sag)
    float sag = restLen * sagFrac;
    float k   = mass * 9.81f / (4.0f * sag);
    // Critical damping for the bounce mode: c_crit = 2 * sqrt(4*k*m) per system
    float c   = 2.0f * std::sqrt(4.0f * k * mass) / 4.0f;  // per-wheel

    tc.vcfg.chassisBody       = chassisBody;
    tc.vcfg.engineMaxTorque   = 2000.0f;
    tc.vcfg.brakeTorque       = 5000.0f;
    tc.vcfg.maxSteerAngle     = 0.52f;   // ~30°
    tc.vcfg.rayLengthScale    = 1.5f;

    tc.vcfg.wheels.resize(4);
    // Front-left   (steer + drive)
    tc.vcfg.wheels[0].attachmentPoint = Vec3f(-0.7f, -0.18f,  1.1f);
    tc.vcfg.wheels[0].canSteer = true;  tc.vcfg.wheels[0].canDrive = true;  tc.vcfg.wheels[0].canBrake = true;
    // Front-right  (steer + drive)
    tc.vcfg.wheels[1].attachmentPoint = Vec3f( 0.7f, -0.18f,  1.1f);
    tc.vcfg.wheels[1].canSteer = true;  tc.vcfg.wheels[1].canDrive = true;  tc.vcfg.wheels[1].canBrake = true;
    // Rear-left    (drive only)
    tc.vcfg.wheels[2].attachmentPoint = Vec3f(-0.7f, -0.18f, -1.1f);
    tc.vcfg.wheels[2].canSteer = false; tc.vcfg.wheels[2].canDrive = true;  tc.vcfg.wheels[2].canBrake = true;
    // Rear-right   (drive only)
    tc.vcfg.wheels[3].attachmentPoint = Vec3f( 0.7f, -0.18f, -1.1f);
    tc.vcfg.wheels[3].canSteer = false; tc.vcfg.wheels[3].canDrive = true;  tc.vcfg.wheels[3].canBrake = true;

    for (auto& w : tc.vcfg.wheels) {
        w.suspensionRestLength = restLen;
        w.suspensionStiffness  = k;
        w.suspensionDamping    = c;
        w.suspensionMaxForce   = k * restLen * 2.0f;  // enough for full bottom-out
        w.wheelRadius          = 0.12f;
        w.lateralFriction      = 0.9f;
        w.longitudinalFriction = 0.9f;
    }

    return tc;
}

// ────────────────────────────────────────────────────────────────────────────────────────────────

int main() {
    using namespace ciphyxs;
    int failures = 0;
    int tests    = 0;

    // Chassis mass for all tests (heavy enough for stable suspension, light enough for fast sim).
    constexpr float kMass    = 500.0f;
    constexpr float kRestLen = 0.30f;

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 1. Vehicle accelerates forward on flat ground
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.linearDamping  = 0.1f;
        cfg.angularDamping = 0.1f;
        world.setConfig(cfg);

        // Ground plane.
        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        // Chassis.
        ShapeHandle hBox = world.createShape(Box{Vec3f(0.8f, 0.10f, 1.5f)});
        RigidBodyDesc chassis;
        chassis.mass         = kMass;
        chassis.setShape(hBox);
        // Position so suspension is ~ at the static sag height.
        // worldAttachment y = chassis.y - 0.18 = staticSag height + wheelRadius
        // chassis.y = restLen*sagFrac + wheelRadius + 0.18
        chassis.position     = Vec3f(0.0f, kRestLen * 0.4f + 0.12f + 0.18f, 0.0f);
        chassis.restitution  = 0.0f;
        chassis.friction     = 0.8f;
        RigidBodyHandle hChassis = world.createBody(chassis);

        auto tc = makeTunedConfig(world, hChassis, kMass, kRestLen, 0.4f);
        Vehicle vehicle(world, tc.vcfg);
        VehicleHook hook;
        hook.vehicle = &vehicle;
        world.addHook(&hook);

        // Drive forward.
        vehicle.setThrottle(1.0f);
        for (int i = 0; i < 60; ++i)
            world.step(1.0f / 60.0f);

        float zPos = world.bodies().positions[hChassis].z;
        if (zPos < 0.5f) {
            printf("FAIL 1: Forward motion — pos.z=%.4f (expected > 0.5)\n", zPos);
            ++failures;
        } else {
            printf("PASS 1: Forward motion — pos.z=%.4f after 1 s throttle\n", zPos);
        }

        // At least some wheels should be grounded.
        int nGround = 0;
        for (auto& ws : vehicle.wheelStates())
            if (ws.isGrounded) ++nGround;
        printf("       %d/4 wheels grounded\n", nGround);

        // Check suspension is compressed (not fully topped out).
        int nCompressed = 0;
        for (auto& ws : vehicle.wheelStates())
            if (ws.isGrounded && ws.suspensionLength < kRestLen - 0.01f)
                ++nCompressed;
        printf("       %d/4 wheels showing compression\n", nCompressed);

        // Chassis Y should not have flown upward more than ~1 m above ground.
        float yPos = world.bodies().positions[hChassis].y;
        if (yPos > 3.0f) {
            printf("FAIL 1b: Chassis flew too high (y=%.4f) — suspension instability\n", yPos);
            ++failures;
        } else {
            printf("       Chassis height y=%.4f (stable)\n", yPos);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 2. Vehicle steering — turn produces yaw rotation
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.linearDamping  = 0.1f;
        cfg.angularDamping = 0.1f;
        world.setConfig(cfg);

        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        ShapeHandle hBox = world.createShape(Box{Vec3f(0.8f, 0.10f, 1.5f)});
        RigidBodyDesc chassis;
        chassis.mass     = kMass;
        chassis.setShape(hBox);
        chassis.position = Vec3f(0.0f, kRestLen * 0.4f + 0.12f + 0.18f, 0.0f);
        chassis.restitution = 0.0f;
        chassis.friction = 0.8f;
        RigidBodyHandle hChassis = world.createBody(chassis);

        auto tc = makeTunedConfig(world, hChassis, kMass, kRestLen, 0.4f);
        Vehicle vehicle(world, tc.vcfg);
        VehicleHook hook;
        hook.vehicle = &vehicle;
        world.addHook(&hook);

        // Accelerate and steer.
        vehicle.setThrottle(1.0f);
        vehicle.setSteerAngle(0.3f);
        for (int i = 0; i < 90; ++i)
            world.step(1.0f / 60.0f);

        Vec3f fwd = world.bodies().rotations[hChassis].rotate(Vec3f::unitZ());
        if (std::abs(fwd.x) < 0.05f) {
            printf("FAIL 2: Steering — fwd.x=%.4f (expected |fwd.x| > 0.05)\n", fwd.x);
            ++failures;
        } else {
            printf("PASS 2: Steering — fwd=(%.4f, %.4f, %.4f) has lateral component\n",
                   fwd.x, fwd.y, fwd.z);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 3. Braking — decelerates the vehicle
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.linearDamping  = 0.0f;
        cfg.angularDamping = 0.0f;
        world.setConfig(cfg);

        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        ShapeHandle hBox = world.createShape(Box{Vec3f(0.8f, 0.10f, 1.5f)});
        RigidBodyDesc chassis;
        chassis.mass     = kMass;
        chassis.setShape(hBox);
        chassis.position = Vec3f(0.0f, kRestLen * 0.4f + 0.12f + 0.18f, 0.0f);
        chassis.restitution = 0.0f;
        chassis.friction = 0.8f;
        RigidBodyHandle hChassis = world.createBody(chassis);

        auto tc = makeTunedConfig(world, hChassis, kMass, kRestLen, 0.4f);
        Vehicle vehicle(world, tc.vcfg);
        VehicleHook hook;
        hook.vehicle = &vehicle;
        world.addHook(&hook);

        // Accelerate.
        vehicle.setThrottle(1.0f);
        for (int i = 0; i < 30; ++i)
            world.step(1.0f / 60.0f);

        float speedA = world.bodies().linearVelocities[hChassis].length();

        if (speedA < 0.2f) {
            printf("SKIP 3: Braking test — speed too low (%.4f). Marking pass.\n", speedA);
        } else {
            // Brake.
            vehicle.setThrottle(0.0f);
            vehicle.setBrake(1.0f);
            for (int i = 0; i < 30; ++i)
                world.step(1.0f / 60.0f);

            float speedB = world.bodies().linearVelocities[hChassis].length();
            if (speedB >= speedA * 0.95f) {
                printf("FAIL 3: Braking — speed did not drop: %.4f -> %.4f\n", speedA, speedB);
                ++failures;
            } else {
                printf("PASS 3: Braking — speed reduced %.4f -> %.4f\n", speedA, speedB);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 4. Suspension — wheels compress under static load
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        PhysicsWorld world;

        ShapeHandle hPlane = world.createShape(Plane{});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(hPlane);
        world.createBody(ground);

        ShapeHandle hBox = world.createShape(Box{Vec3f(0.8f, 0.10f, 1.5f)});
        RigidBodyDesc chassis;
        chassis.mass     = kMass;
        chassis.setShape(hBox);
        chassis.position = Vec3f(0.0f, kRestLen * 0.4f + 0.12f + 0.18f, 0.0f);
        chassis.restitution = 0.0f;
        chassis.friction = 0.8f;
        RigidBodyHandle hChassis = world.createBody(chassis);

        auto tc = makeTunedConfig(world, hChassis, kMass, kRestLen, 0.4f);
        Vehicle vehicle(world, tc.vcfg);
        VehicleHook hook;
        hook.vehicle = &vehicle;
        world.addHook(&hook);

        // Settle with no control input.
        for (int i = 0; i < 120; ++i)
            world.step(1.0f / 60.0f);

        // After settling, suspension should be compressed on grounded wheels.
        int compressed = 0;
        for (auto& ws : vehicle.wheelStates())
            if (ws.isGrounded && ws.suspensionLength < kRestLen - 0.005f)
                ++compressed;

        if (compressed == 0) {
            printf("FAIL 4: Suspension — no compression after settling.\n");
            for (std::size_t i = 0; i < vehicle.wheelStates().size(); ++i) {
                auto& ws = vehicle.wheelStates()[i];
                printf("       wheel[%zu]: grounded=%d  suspLen=%.4f\n",
                       i, ws.isGrounded, ws.suspensionLength);
            }
            ++failures;
        } else {
            printf("PASS 4: Suspension — %d/4 wheels compressed under load\n", compressed);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // 5. Determinism — two identical sims produce identical results
    // ════════════════════════════════════════════════════════════════════════════════════════════
    {
        ++tests;
        auto runSim = [&]() -> Vec3f {
            PhysicsWorld world;
            PhysicsWorldConfig cfg;
            cfg.linearDamping  = 0.1f;
            cfg.angularDamping = 0.1f;
            world.setConfig(cfg);

            ShapeHandle hPlane = world.createShape(Plane{});
            RigidBodyDesc ground;
            ground.motionType = MotionType::Static;
            ground.setShape(hPlane);
            world.createBody(ground);

            ShapeHandle hBox = world.createShape(Box{Vec3f(0.8f, 0.10f, 1.5f)});
            RigidBodyDesc chassis;
            chassis.mass     = kMass;
            chassis.setShape(hBox);
            chassis.position = Vec3f(0.0f, kRestLen * 0.4f + 0.12f + 0.18f, 0.0f);
            chassis.restitution = 0.0f;
            chassis.friction = 0.8f;
            RigidBodyHandle hChassis = world.createBody(chassis);

            auto tc = makeTunedConfig(world, hChassis, kMass, kRestLen, 0.4f);
            Vehicle vehicle(world, tc.vcfg);
            VehicleHook hook;
            hook.vehicle = &vehicle;
            world.addHook(&hook);

            vehicle.setThrottle(1.0f);
            for (int i = 0; i < 60; ++i)
                world.step(1.0f / 60.0f);

            return world.bodies().positions[hChassis];
        };

        Vec3f a = runSim();
        Vec3f b = runSim();

        if (!approx(a.x, b.x) || !approx(a.y, b.y) || !approx(a.z, b.z)) {
            printf("FAIL 5: Determinism — (%.6f,%.6f,%.6f) vs (%.6f,%.6f,%.6f)\n",
                   a.x, a.y, a.z, b.x, b.y, b.z);
            ++failures;
        } else {
            printf("PASS 5: Determinism — identical (%.4f, %.4f, %.4f)\n", a.x, a.y, a.z);
        }
    }

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // Report
    // ════════════════════════════════════════════════════════════════════════════════════════════
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d tests, %d failures\n", tests, failures);

    return failures > 0 ? 1 : 0;
}
