#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <chrono>
int main() {
    using namespace ciphyxs;
    
    // Test A: Sequential (no task graph, no parallel solver)
    {
        PhysicsWorldConfig cfg;
        cfg.enableParallelSolver = false;
        cfg.enableTaskGraphPipeline = false;
        PhysicsWorld world(cfg);
        
        ShapeHandle boxShape = world.createShape(Box{Vec3f(0.35f, 0.35f, 0.35f)});
        ShapeHandle groundShape = world.createShape(Plane{Vec3f::unitY(), 0.0f});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(groundShape);
        ground.friction = 0.9f;
        world.createBody(ground);
        for (int i = 0; i < 50; ++i) {
            RigidBodyDesc box;
            box.mass = 0.5f;
            box.setShape(boxShape);
            box.position = Vec3f(0.0f, 0.7f + i * 0.72f, 0.0f);
            box.friction = 0.9f;
            world.createBody(box);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < 200; ++f) world.step(1.0f/60.0f);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("SEQUENTIAL: 50 boxes x 200 frames: %.1f ms (%.1f ms/frame)\n", ms, ms/200.0);
    }
    
    // Test B: Task graph pipeline
    {
        PhysicsWorldConfig cfg;
        cfg.enableParallelSolver = true;
        cfg.enableTaskGraphPipeline = true;
        cfg.numThreads = 0;
        PhysicsWorld world(cfg);
        
        ShapeHandle boxShape = world.createShape(Box{Vec3f(0.35f, 0.35f, 0.35f)});
        ShapeHandle groundShape = world.createShape(Plane{Vec3f::unitY(), 0.0f});
        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(groundShape);
        ground.friction = 0.9f;
        world.createBody(ground);
        for (int i = 0; i < 50; ++i) {
            RigidBodyDesc box;
            box.mass = 0.5f;
            box.setShape(boxShape);
            box.position = Vec3f(0.0f, 0.7f + i * 0.72f, 0.0f);
            box.friction = 0.9f;
            world.createBody(box);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < 200; ++f) world.step(1.0f/60.0f);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("TASKGRAPH:  50 boxes x 200 frames: %.1f ms (%.1f ms/frame)\n", ms, ms/200.0);
    }
    return 0;
}
