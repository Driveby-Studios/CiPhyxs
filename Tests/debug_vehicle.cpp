//==================================================================================================
/// @file  debug_vehicle.cpp
/// @brief  Debug trace with angular velocity and force tracking.
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>
using namespace ciphyxs;

struct DebugHook : ISolverHook {
    Vehicle* vehicle;
    RigidBodyHandle hChassis;
    float traceForces[60][4] = {}; // [step][wheel] -> springForceMag
    int stepNum = 0;
    
    void onApplyForces(float dt, RigidBodyStorage&) noexcept override {
        if (vehicle) vehicle->update(dt);
    }
    void onPreSolve(float, RigidBodyStorage&) noexcept override {}
    void onPostSolve(float, RigidBodyStorage&) noexcept override {}
};

struct TunedConfig {
    VehicleConfig vcfg;
    float restLength;
};

static TunedConfig makeTunedConfig(PhysicsWorld& world,
                                   RigidBodyHandle chassisBody,
                                   float mass,
                                   float restLen = 0.30f,
                                   float sagFrac = 0.40f) {
    TunedConfig tc;
    tc.restLength = restLen;
    float sag = restLen * sagFrac;
    float k   = mass * 9.81f / (4.0f * sag);
    float c   = 2.0f * std::sqrt(4.0f * k * mass) / 4.0f;
    tc.vcfg.chassisBody = chassisBody;
    tc.vcfg.engineMaxTorque = 2000.0f;
    tc.vcfg.brakeTorque = 5000.0f;
    tc.vcfg.maxSteerAngle = 0.52f;
    tc.vcfg.rayLengthScale = 1.5f;
    tc.vcfg.wheels.resize(4);
    tc.vcfg.wheels[0].attachmentPoint = Vec3f(-0.7f, -0.18f,  1.1f);
    tc.vcfg.wheels[0].canSteer = true;  tc.vcfg.wheels[0].canDrive = true;  tc.vcfg.wheels[0].canBrake = true;
    tc.vcfg.wheels[1].attachmentPoint = Vec3f( 0.7f, -0.18f,  1.1f);
    tc.vcfg.wheels[1].canSteer = true;  tc.vcfg.wheels[1].canDrive = true;  tc.vcfg.wheels[1].canBrake = true;
    tc.vcfg.wheels[2].attachmentPoint = Vec3f(-0.7f, -0.18f, -1.1f);
    tc.vcfg.wheels[2].canSteer = false; tc.vcfg.wheels[2].canDrive = true;  tc.vcfg.wheels[2].canBrake = true;
    tc.vcfg.wheels[3].attachmentPoint = Vec3f( 0.7f, -0.18f, -1.1f);
    tc.vcfg.wheels[3].canSteer = false; tc.vcfg.wheels[3].canDrive = true;  tc.vcfg.wheels[3].canBrake = true;
    for (auto& w : tc.vcfg.wheels) {
        w.suspensionRestLength = restLen;
        w.suspensionStiffness  = k;
        w.suspensionDamping    = c;
        w.suspensionMaxForce   = k * restLen * 2.0f;
        w.wheelRadius          = 0.12f;
        w.lateralFriction      = 0.9f;
        w.longitudinalFriction = 0.9f;
    }
    return tc;
}

int main() {
    constexpr float kMass = 500.0f;
    constexpr float kRestLen = 0.30f;

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
    chassis.mass = kMass;
    chassis.setShape(hBox);
    chassis.position = Vec3f(0.0f, kRestLen * 0.4f + 0.12f + 0.18f, 0.0f);
    chassis.restitution = 0.0f;
    chassis.friction = 0.8f;
    RigidBodyHandle hChassis = world.createBody(chassis);

    auto tc = makeTunedConfig(world, hChassis, kMass, kRestLen, 0.4f);
    Vehicle vehicle(world, tc.vcfg);
    DebugHook hook;
    hook.vehicle = &vehicle;
    hook.hChassis = hChassis;
    world.addHook(&hook);

    vehicle.setThrottle(1.0f);

    for (int i = 0; i < 60; ++i) {
        auto& bodies = world.bodies();
        hook.stepNum = i;
        
        world.step(1.0f/60.0f);
        
        if (i < 5 || i % 10 == 0 || i == 59) {
            Vec3f pos = bodies.positions[hChassis];
            Vec3f vel = bodies.linearVelocities[hChassis];
            Vec3f w   = bodies.angularVelocities[hChassis];
            Quaternionf q = bodies.rotations[hChassis];
            
            // Extract pitch from quaternion
            float sinPitch = -2.0f*(q.y*q.z - q.w*q.x);
            float pitch = std::asin(std::clamp(sinPitch, -1.0f, 1.0f));
            
            printf("Step %d: pos=(%.4f, %.4f, %.4f) vel=(%.4f, %.4f, %.4f) w=(%.4f, %.4f, %.4f) pitch=%.4f\n",
                   i, pos.x, pos.y, pos.z,
                   vel.x, vel.y, vel.z,
                   w.x, w.y, w.z, pitch);
            
            int nGround = 0;
            for (size_t w = 0; w < vehicle.wheelStates().size(); ++w) {
                auto& ws = vehicle.wheelStates()[w];
                if (ws.isGrounded) ++nGround;
                printf("  wheel[%zu]: grounded=%d suspLen=%.4f prevLen=%.4f angVel=%.1f contact=(%.4f, %.4f, %.4f)\n",
                       w, ws.isGrounded, ws.suspensionLength, ws.suspensionPrevLength, ws.angularVelocity,
                       ws.contactPoint.x, ws.contactPoint.y, ws.contactPoint.z);
            }
            printf("  wheels grounded: %d/4\n", nGround);
        }
    }
    
    printf("\nFinal: pos=(%.4f, %.4f, %.4f)\n",
           world.bodies().positions[hChassis].x,
           world.bodies().positions[hChassis].y,
           world.bodies().positions[hChassis].z);
    
    // Check contact
    auto& manifolds = world.manifolds();
    printf("Manifolds: %zu\n", manifolds.size());

    return 0;
}
