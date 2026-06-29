// Check array sizes
#include "../include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>

int main() {
    using namespace ciphyxs;
    
    PhysicsWorld world(PhysicsWorld::Preset::LowEnd);
    
    ShapeHandle gs = world.createShape(Plane{Vec3f::unitY(), 0.0f});
    ShapeHandle bs = world.createShape(Box{Vec3f(0.35f, 0.35f, 0.35f)});
    
    {
        RigidBodyDesc g;
        g.motionType = MotionType::Static;
        g.setShape(gs); g.position = Vec3f::zero();
        world.createBody(g);
    }
    
    for (int i = 0; i < 10; ++i) {
        RigidBodyDesc b;
        b.mass = 0.5f; b.setShape(bs);
        b.position = Vec3f(0, 1.0f+i*0.72f, 0);
        b.startActive = true;
        world.createBody(b);
    }
    
    const auto& bodies = world.bodies();
    printf("size: %zu\n", bodies.size());
    printf("positions: %zu\n", bodies.positions.size());
    printf("collisionGroups: %zu\n", bodies.collisionGroups.size());
    printf("collisionMasks: %zu\n", bodies.collisionMasks.size());
    
    // Check all arrays for size mismatch
    #define CHECK(name) printf("%-25s: %zu %s\n", #name, bodies.name.size(), \
        bodies.name.size() == bodies.size() ? "OK" : "MISMATCH")
    
    CHECK(positions);
    CHECK(rotations);
    CHECK(linearVelocities);
    CHECK(angularVelocities);
    CHECK(forces);
    CHECK(torques);
    CHECK(inverseMasses);
    CHECK(inverseInertiaDiag);
    CHECK(inertiaRotations);
    CHECK(restitutions);
    CHECK(frictions);
    CHECK(linearDamping);
    CHECK(angularDamping);
    CHECK(motionTypes);
    CHECK(activeFlags);
    CHECK(ccdModes);
    CHECK(sleepTimers);
    CHECK(collisionGroups);
    CHECK(collisionMasks);
    CHECK(shapeStart);
    CHECK(shapeCount);
    
    return 0;
}
