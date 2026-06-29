#include "StressTestBase.hpp"
#include "../include/ciphyxs/core/AggressiveMode.hpp"

int main() {
    using namespace ciphyxs;
    using namespace ciphyxs::stresstest;
    
    printf("Testing applyLowEndPreset...\n");
    PhysicsWorld world;
    aggressive::applyLowEndPreset(world);
    
    printf("Config: parallel=%d taskgraph=%d threads=%u broadphase=%d\n",
           world.config().enableParallelSolver,
           world.config().enableTaskGraphPipeline,
           world.config().numThreads,
           (int)world.config().enableTaskGraphPipeline);
    
    printf("Creating shapes...\n");
    ShapeHandle ground = world.createShape(Plane{Vec3f::unitY(), 0});
    ShapeHandle boxSh = world.createShape(Box{Vec3f(0.35f, 0.35f, 0.35f)});
    
    printf("Creating ground...\n");
    RigidBodyDesc g;
    g.motionType = MotionType::Static;
    g.setShape(ground);
    g.position = Vec3f::zero();
    world.createBody(g);
    
    printf("Creating boxes...\n");
    constexpr float kSpacing = 0.72f;
    constexpr int kW = 10, kD = 5, kH = 4;
    int placed = 0;
    for (int iy = 0; iy < kH; ++iy)
        for (int iz = 0; iz < kD; ++iz)
            for (int ix = 0; ix < kW; ++ix) {
                float x = (ix - (kW-1)*0.5f) * kSpacing;
                float z = (iz - (kD-1)*0.5f) * kSpacing;
                float y = 0.35f + iy * kSpacing;
                RigidBodyDesc b;
                b.mass = 0.5f;
                b.setShape(boxSh);
                b.position = Vec3f(x, y, z);
                b.restitution = 0.0f;
                b.friction = 0.9f;
                b.startActive = true;
                world.createBody(b);
                ++placed;
            }
    
    printf("Stepping 500 frames with SimdBruteForce broadphase...\n");
    world.setBroadphaseType(BroadphaseType::SimdBruteForce);
    
    for (int f = 0; f < 500; ++f) {
        world.step(1.0f/60.0f);
        if ((f % 100) == 0) printf("  frame %d\n", f);
    }
    
    printf("Done! Hash: 0x%016llX\n", (unsigned long long)hashBodyState(world.bodies()));
    return 0;
}
