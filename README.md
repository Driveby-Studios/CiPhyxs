# CiPhyxs — High-Performance Physics Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Header-Only](https://img.shields.io/badge/header--only-✓-brightgreen)
![CI](https://github.com/Driveby-Studios/CiPhyxs/actions/workflows/ci.yml/badge.svg) 
![GCC](https://img.shields.io/badge/GCC-11%2B-blue)
![Clang](https://img.shields.io/badge/Clang-14%2B-blue)
![MSVC](https://img.shields.io/badge/MSVC-2022%2B-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

**CiPhyxs** is a high-performance, **header-only**, C++20 physics engine designed for **heavy, mechanical, and industrial simulations** where numerical stability and extreme mass ratios (10⁶:1+) are critical. Its architecture draws inspiration from Nvidia PhysX and Unreal Engine's Chaos physics, with a focus on determinism, data-oriented design, and a clean, framework-agnostic API.

---

## Features

- **Symplectic (Semi-Implicit Euler) Integration** — Energy-conserving integration that remains stable at extreme mass ratios (>10⁶:1), essential for crane + steel beam, heavy machinery, and industrial robotics.

- **Warm-Started Sequential-Impulse Solver** — Industry-standard constraint resolution with friction cones, split-impulse positional correction, and feature-ID based warm-start matching for robust stacking and persistent contacts.

- **Rayleigh Damping** — Physically correct viscous damping (`F_damp = -c·v`) integrated into the force pipeline, making it frame-rate independent and mass-aware. No more ad-hoc post-solve velocity scaling.

- **Collision Detection** — Full SAT-based narrowphase with Sphere, Box, Plane, and Capsule primitives. **GJK + EPA** for arbitrary convex-convex pairs (ConvexMesh, Sphere, Box, Capsule). Three broadphase options: brute-force O(n²), sweep-and-prune O(n log n), and spatial hash O(n). Feature-ID encoding on all contact points for reliable warm-start matching across frames.

- **GJK/EPA Robustness** — Tetrahedron containment detection with fast AABB rejection prevents false intersections on far-apart shapes. Multi-axis initial direction tries (1,0,0), (0,1,0), (0,0,1) to avoid degenerate simplexes. Segment-orthogonal EPA polytope expansion instead of hardcoded axes.

- **ConvexMesh / Convex Hull** — Deformable convex hulls defined by vertex arrays with GJK (Gilbert-Johnson-Keerthi) distance computation and EPA (Expanding Polytope Algorithm) penetration depth extraction. Supports collision against all other primitive types via the GJK dispatch.

- **TriangleMesh / Static Geometry** — Concave triangle mesh support with BVH (Bounding Volume Hierarchy) acceleration for ray casting and collision queries. Median-split BVH builder with O(log N) traversal. Sphere-vs-mesh and plane-vs-mesh narrowphase with BVH overlap culling, linear-scan fallback for unaccelerated meshes.

- **Continuous Collision Detection (CCD)** — Swept-sphere, swept-capsule, and swept-ConvexMesh (sphere-approximated) tests against planes, spheres, and boxes prevent tunnelling at high velocities. Binary-search and analytical TOI computation for precision and performance.

- **Ray Casting** — Full ray-primitive intersection for Sphere, Box, Plane, Capsule, ConvexMesh, and TriangleMesh (BVH-accelerated for meshes). World-space dispatch integrated into PhysicsWorld.

- **Compound Shapes** — Each rigid body can be composed of multiple sub-shapes with local transforms, using the parallel-axis theorem for automatic inertia tensor computation.

- **Joint System** — BallSocket, Hinge, Fixed, and Spring constraints with optional **angular limits** (hinge), **velocity-servo motors** (hinge), Baumgarte stabilisation, **breakable constraints** with configurable impulse thresholds, and **sub-shape attachment** for compound body constraint placement.

- **Ragdoll Builder** — `RagdollBuilder` utility for constructing articulated chains (arms, legs, spines, tentacles) from body descriptors and hinge joints with automatic anchor placement.

- **Data-Oriented Design** — Structure-of-Arrays (SoA) body storage with 16-byte aligned memory for SIMD-friendly iteration. All hot vectors use `AlignedVector<T, 16>` with `std::assume_aligned` hints.

- **Debug Visualization** — `IDebugRenderer` interface with wireframe box/sphere/capsule/plane drawing, contact point rendering, and joint visualisation. Integrate with any rendering system via a single interface.

- **Island-Based Parallel Solver** — Automatic union-find island formation dispatches independent body groups to a thread pool for concurrent solving. Results are bit-identical between single-threaded and parallel execution.

- **Task-Graph Pipeline Scheduler** — DAG-based dependency-driven execution of the solver pipeline. Independent stages (CCD + collision detection) run concurrently, and the task graph naturally scales to per-island parallelism. Deterministic ready-task ordering guarantees bit-identical results with the sequential pipeline.

- **Deterministic** — Same input guarantees same output across runs, platforms, and solver configurations. No random seeds, no non-deterministic hash maps, fixed iteration orders.

- **Dynamic BVH (Dbvt) Broadphase** — Incremental AABB tree with SAH insertion, fat-AABB temporal coherence for minimal tree updates, frame-counter pair caching for 50–80% pair-gen cost reduction, and automatic AVL-style rebalancing with median-split full rebuild. Ideal for scenes with high frame-to-frame coherence.

- **Shape Queries** — `world.overlapShape()` for discrete overlap testing and `world.sweepShape()` for swept time-of-impact queries. Both use Dbvt acceleration and work with any shape type via binary-search or analytical swept collision.

- **Contact Normal Reduction** — Merges nearly-identical normals from adjacent triangle faces into a unified plane normal, eliminating the persistent micro-bouncing that occurs when a box rests on a flat triangulated surface.

- **Soft Body Simulation (PBD)** — Position-Based Dynamics for cloth, deformable volumes, and ropes. Verlet integration, distance/bend/volume constraints with Gauss-Seidel projection, particle-vs-rigid-body collision (sphere/box/plane/capsule). Factory methods: `SoftBody::createCloth()`, `SoftBody::createVolume()`, `SoftBody::createRope()`. 7 dedicated validation tests.

- **Kinematic Character Controller** — Sweep-based capsule controller with sliding collision resolution, step-up/down, slope handling (configurable max angle), grounded-state detection with skin-width gap, and jump support. Full integration test suite (6 tests, 100% pass).

- **Framework-Agnostic** — Zero external dependencies. Drop `include/ciphyxs/` into any C++20 project — game engine, robotics simulator, CAD tool, or industrial digital twin.

- **Extensible Hook System** — `ISolverHook` interface lets you inject custom forces, constraints, or debug drawing into the solver pipeline at any phase.

---

## Quick Start

```cpp
#include <ciphyxs/CiPhyxs.hpp>

int main() {
    using namespace ciphyxs;

    PhysicsWorld world;

    // 1. Create shapes.
    ShapeHandle groundShape = world.createShape(Plane{});
    ShapeHandle boxShape    = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});

    // 2. Create a static ground plane.
    RigidBodyDesc ground;
    ground.motionType = MotionType::Static;
    ground.setShape(groundShape);
    world.createBody(ground);

    // 3. Create a falling box.
    RigidBodyDesc box;
    box.mass        = 10.0f;
    box.position    = Vec3f(0, 10, 0);
    box.setShape(boxShape);
    world.createBody(box);

    // 4. Simulate 120 frames at 60 FPS.
    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
        auto& pos = world.bodies().positions;
        // ... render at pos[0], pos[1], etc.
    }
}
```

---

## Build & Integration

CiPhyxs is **header-only** — no compilation needed. Just drop the `include/` directory into your project and `#include <ciphyxs/CiPhyxs.hpp>`.

### CMake (recommended)

```bash
# Add as a subdirectory in your CMakeLists.txt
add_subdirectory(path/to/ciphyxs)
target_link_libraries(your_target PRIVATE ciphyxs)
```

Or install system-wide:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local
```

Then in your project:

```cmake
find_package(CiPhyxs REQUIRED)
target_link_libraries(your_target PRIVATE CiPhyxs::CiPhyxs)
```

### Options

| CMake Option              | Default | Description                            |
|---------------------------|---------|----------------------------------------|
| `CIPHYXS_BUILD_TESTS`     | `ON`    | Build the test suite                   |
| `CIPHYXS_ENABLE_SIMD`     | `OFF`   | Add compiler flags for SSE2/NEON       |

### Build Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### Manual Compilation (no CMake)

```bash
g++ -std=c++20 -O2 -I/path/to/ciphyxs/include your_program.cpp -o your_program
```

---

## Module Overview

| Module              | Headers                          | Description                                           |
|---------------------|----------------------------------|-------------------------------------------------------|
| **Math**            | `math/Vec3.hpp`                  | 3D vector (`alignas(16)`, concept-safe)               |
|                     | `math/Mat3.hpp`                  | 3×3 column-major matrix                               |
|                     | `math/Quaternion.hpp`            | Hamilton quaternion with slerp                        |
| **Core**            | `core/RigidBody.hpp`             | SoA storage, descriptors, body handles                |
|                     | `core/PhysicsWorld.hpp`          | World orchestrator, pipeline, hooks                   |
|                     | `core/ConstraintSolver.hpp`      | Sequential-impulse solver (warm start)                |
|                     | `core/Joint.hpp`                 | Joint descriptors and storage                         |
|                     | `core/JointSolver.hpp`           | Joint constraint resolution                           |
|                     | `core/IslandSolver.hpp`          | Union-find island formation + parallel solving        |
|                     | `core/ThreadPool.hpp`            | Lightweight C++20 task parallelism                    |
|                     | `core/TaskGraph.hpp`             | DAG-based task scheduler for pipeline parallelism     |
|                     | `core/AlignedAllocator.hpp`      | STL-compatible over-aligned allocator                 |
| **Collision**       | `collision/Shape.hpp`            | Sphere, Box, Plane, Capsule, ConvexMesh + AABB + inertia |
|                     | `collision/Contact.hpp`          | Contact point / manifold data structures              |
|                     | `collision/Broadphase.hpp`       | Brute-force + configuration                           |
|                     | `collision/Narrowphase.hpp`      | SAT + GJK/EPA primitive collision tests               |
|                     | `collision/Convex.hpp`           | GJK distance + EPA penetration for arbitrary convex shapes (tetrahedron containment, multi-axis init) |
|                     | `collision/SpatialHash.hpp`      | O(n) 3D uniform-grid broadphase                       |
|                     | `collision/SweepAndPrune.hpp`    | O(n log n) sweep-and-prune broadphase                 |
|                     | `collision/Bvh.hpp`              | BVH acceleration for triangle mesh ray cast + collision |
|                     | `collision/Ray.hpp`              | Ray casting API (all primitives including ConvexMesh) |
|                     | `collision/Dbvt.hpp`             | Dynamic BVH — incremental AABB tree with SAH insertion, fat-AABB temporal coherence, AVL rotations, pair cache, overlap queries |
|                     | `collision/ContactReduction.hpp`  | Coplanar contact normal reduction for stable box-on-triangle-mesh resting |
| **Solver**          | `solver/Vehicle.hpp`              | Ray-cast vehicle controller — suspension, tire friction, drive + brakes   |
|                     | `core/CharacterController.hpp`     | Kinematic sweep-based character controller — sliding, step-up/down, slope handling, jump |

---

## Architecture

### Solver Pipeline

Each fixed-timestep sub-step runs the following deterministic pipeline.  Two execution modes
are available:

#### Sequential Mode (default)

```
1. Apply Forces      → Clear accumulators, apply gravity, compute Rayleigh damping forces,
                       invoke hooks (custom force generators).
2. Integrate Velocities → Semi-implicit Euler: v += F/m · dt
3. Collision Detection  → Broadphase (AABB culling) + Narrowphase (SAT primitive tests)
4. Pre-Solve            → Hook call (warm-start init, debug state)
5. Solve Constraints    → Sequential-impulse with friction + Baumgarte (contacts → joints)
6. Post-Solve           → Hook call (debug drawing, statistics)
7. Integrate Positions  → x += v' · dt,  q += ½ ω' · q · dt
8. Update Sleep         → Put inert bodies to sleep
```

#### Task-Graph Mode (`enableTaskGraphPipeline = true`)

When enabled, the pipeline is expressed as a directed-acyclic graph of named stages.
The `TaskGraph` scheduler dispatches stages to the thread pool as their dependencies are
satisfied, enabling **pipeline parallelism**:

```
               ┌─────────────────────┐
               │   Integrate Forces  │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │ Integrate Velocities │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │     Broadphase       │  (generates body-body candidate pairs)
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │    Pre-Solve         │  (user hooks)
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Form Islands       │  (union-find over broadphase pairs + joints;
               │   Assign Pairs       │   every body is assigned to an island)
               └──────────┬──────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
     ┌────────▼──┐ ┌────▼────┐ ┌───▼───┐
     │ Isl #0    │ │ Isl #1  │ │ ...   │  (per-island sub-DAGs)
     │           │ │         │ │       │
  ┌──▼──┐ ┌──▼───┐  ┌──▼──┐ ┌──▼───┐  ┌──▼───┐
  │CCD │ │Narrow│  │CCD │ │Narrow│  │ ...  │
  │    │ │phase │  │    │ │phase │  │      │
  └──┬─┘ └──┬───┘  └──┬─┘ └──┬───┘  └──┬───┘
     │      │          │      │          │
     └──┬───┘          └──┬───┘          │
        │                 │              │
  ┌─────▼─────┐    ┌─────▼─────┐   ┌────▼────┐
  │ Reduce    │    │ Reduce    │   │ ...     │
  │ + Solve   │    │ + Solve   │   │         │
  └─────┬─────┘    └─────┬─────┘   └────┬────┘
        │                 │              │
        └─────────────────┼──────────────┘
                          │
               ┌──────────▼──────────┐
               │    Join Barrier      │  (all islands complete)
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Merge Manifolds    │  (per-island → global)
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Joint + Fracture    │  (joint solving, breakable
               └──────────┬──────────┘    constraints, fracture)
                          │
               ┌──────────▼──────────┐
               │    Post-Solve        │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Integrate Pos     │  + sleep management
               └─────────────────────┘
```

After velocity integration, broadphase generates body-body candidate pairs.
Islands are formed directly from pairs + joints **before** any contact generation.
Each island then runs a complete sub-DAG: **CCD (per-island) → narrowphase →
contact reduction → sequential-impulse solve**.  CCD is per-island, meaning
fast-moving bodies in one island don't delay CCD processing for other islands.
Every body (including isolated ones) belongs to exactly one island — isolated
bodies form size-1 islands with no narrowphase work but still get CCD.

This enables true island-level pipeline parallelism: island A's CCD + narrowphase
+ solve runs independently of island B's, and the solver for island A starts as
soon as narrowphase for island A finishes, without waiting for other islands.

After all islands complete, a join barrier synchronises, per-island contact
manifolds are merged into the global array, and joint solving + fracture checks
run globally.

##### Task-Graph Profiling

The task-graph pipeline supports per-task timing via `enableTaskGraphProfiling()`.
When enabled, each task node records its name, wall-clock duration, and executing
thread ID.  Profile events can be retrieved with `taskGraphProfileEvents()` (raw)
or `taskGraphProfileSummary()` (aggregated by task name).  Profiling is disabled
by default (zero overhead when off).

### Data-Oriented Storage

All rigid-body properties are stored in **Structure-of-Arrays (SoA)** layout:

```
RigidBodyStorage:
  positions[16]    // AlignedVector<Vec3f, 16>
  linearVelocities // AlignedVector<Vec3f, 16>
  forces           // AlignedVector<Vec3f, 16>
  inverseMasses    // AlignedVector<float, 16>
  ...
```

This layout minimises cache misses during solver iteration: the solver touches only velocity and inverse-mass fields for most bodies, and the compact, aligned storage enables the compiler to emit SIMD load/store instructions.

### Determinism

Determinism is guaranteed by:

- **Fixed-timestep accumulator** — wall-clock time is accumulated and consumed in discrete sub-steps.
- **Deterministic broadphase** — both brute-force and spatial hash produce pairs in a fixed order.
- **Sorted island processing** — islands are sorted by smallest body handle.
- **Feature-ID warm-start matching** — contact points are matched by shape-feature IDs, not by index.

---

## Configuration

### PhysicsWorld

```cpp
PhysicsWorldConfig cfg;
cfg.gravity              = Vec3f(0, -9.81f, 0);   // Earth gravity
cfg.fixedTimestep        = 1.0f / 60.0f;          // 60 Hz
cfg.linearDamping        = 0.5f;                  // Rayleigh damping coefficient
cfg.angularDamping       = 0.5f;                  // Rayleigh damping coefficient
cfg.sleepEnergyThreshold = 0.005f;
cfg.sleepTimeRequired    = 0.5f;

cfg.enableTaskGraphPipeline = true;  // Enable DAG-based pipeline parallelism

PhysicsWorld world(cfg);
```

Or toggle at runtime:

```cpp
world.enableTaskGraphPipeline(true);
```

### Solver Parameters

```cpp
SolverConfig scfg = world.solverConfig();
scfg.numIterations          = 10;   // 6–12 typical
scfg.baumgarte              = 0.15f;
scfg.restitutionThreshold   = 1.0f;
scfg.maxPenetrationCorrection = 0.15f;
scfg.enableWarmStart        = true;
scfg.warmStartFactor        = 0.8f;
```

---

## API Documentation

Generate Doxygen documentation:

```bash
cmake -B build
cmake --build build --target doc
# Open build/html/index.html
```

---

## Roadmap

### Completed (Core Features)

- [x] Math foundation (Vec3, Mat3, Quaternion) — C++20 constexpr, `alignas(16)`
- [x] SoA rigid-body storage with aligned allocator
- [x] SAT-based narrowphase (Sphere, Box, Plane, Capsule)
- [x] Warm-started sequential-impulse solver
- [x] Split-impulse positional correction
- [x] Fixed-timestep deterministic pipeline
- [x] Joint constraints (BallSocket, Hinge, Fixed, Spring)
- [x] Island-based parallel solver
- [x] Compound shapes with parallel-axis inertia
- [x] Feature-ID based warm-start matching
- [x] Rayleigh damping
- [x] Sweep-and-prune broadphase
- [x] Continuous Collision Detection (CCD) — sphere ↔ sphere/plane/box
- [x] CCD capsule sweeps — swept tests for capsule vs. sphere/plane/box
- [x] Box-box incident face reprojection (fallback clipping) + edge-edge contact generation
- [x] Motorised joints — velocity-servo hinge motor with force limits
- [x] Breakable constraints — impulse-threshold based joint breaking
- [x] Constraints on compound sub-shapes — joint attachment at sub-shape local positions
- [x] Ragdoll support — RagdollBuilder for articulated chains with angular limits
- [x] Debug visualization API — IDebugRenderer interface with wireframe drawing utilities
- [x] Hinge angular limit enforcement — Baumgarte-stabilised angle clamping
- [x] Convex hull (ConvexMesh) collision via GJK+EPA — full dispatch against all primitive types
- [x] Robust tetrahedron containment detection in GJK — volumetric origin-in-tetrahedron test
- [x] Multi-axis initial GJK direction — handles coaxially-aligned shapes
- [x] Segment-orthogonal EPA polytope expansion — geometry-aware expansion directions
- [x] Correct EPA contact point extraction — world-space contact points from Minkowski support
- [x] Triangle mesh collision with BVH acceleration — median-split BVH, sphere/plane/box/capsule/convex vs mesh
- [x] Dynamic BVH (Dbvt) broadphase — incremental AABB tree with SAH insertion, fat-AABB temporal coherence, overlap queries, deterministic pair generation, lazy cleanup
- [x] Dbvt pair cache — frame-counter-based temporal coherence; reuse cached overlapping pairs across frames, walk tree only for new overlaps
- [x] Contact normal reduction — merge coplanar triangle contacts into single-plane manifolds for stable box-on-mesh resting
- [x] Dbvt AVL-style tree balancing — height-balanced rotations on insertion to prevent degenerate trees; full median-split rebuild when removal threshold exceeded
- [x] `world.overlapShape(shape, transform, filter)` → `std::vector<RigidBodyHandle>` — brute-force narrowphase with optional Dbvt acceleration (synced on query)
- [x] `world.sweepShape(shape, from, to, filter)` → `SweepHit` — binary-search swept collision for all shape types with Dbvt candidate culling
- [x] Vehicle (ray-cast) controller — suspension spring-damper, lateral/longitudinal tire friction with combined-slip friction-circle limiting, drive torque, braking torque, Ackermann-style steering, ISolverHook integration
- [x] Kinematic character controller — sweep-based capsule controller with step-up/down, slope handling (max angle), grounded-state snapping with skin-width gap, sliding collision resolution, jump support
- [x] Character controller collision filtering — SelfFilter excludes the character's own kinematic body from sweep/overlap queries
- [x] 17-test suite (100% pass, zero warnings) — alignment, collision, final, joints, new features, parallel, SAP, convex mesh, triangle mesh, CCD, ragdoll, DBVT, contact reduction, vehicle, character controller, task graph, **soft body**, **fracture**
- [x] **Task-graph pipeline scheduler** — DAG-based dependency-driven execution of the solver pipeline (CCD || collision detection, deterministic ready-task ordering). 7 dedicated validation tests.
- [x] **Soft body simulation (PBD)** — Position-Based Dynamics cloth, deformable volumes, and ropes with Verlet integration, distance/bend/volume constraints, Gauss-Seidel projection, and particle-vs-rigid-body collision (sphere/box/plane/capsule). Factory methods: `SoftBody::createCloth()`, `SoftBody::createVolume()`, `SoftBody::createRope()`. Integrated into the fixed-step pipeline and task-graph. 7 dedicated validation tests.
- [x] **Destructible meshes (Voronoi fracture)** — Pre-computed Voronoi fragment meshes with impulse-threshold breaking, momentum-conserving fragment spawn, and optional chain fracture. Integrated into fixed-step and task-graph pipelines. 6 dedicated validation tests.
- [x] **Per-island task sub-graphs** — The solve stage in the task-graph pipeline is decomposed into per-island DAG nodes, enabling the scheduler to dispatch island solves as independent parallel tasks with a join barrier before post-solve. Joint solving, breakable joints, and fracture checks run after the island join.
- [x] **Per-island broadphase → narrowphase → solve sub-DAGs** — Islands are formed directly from broadphase pairs (before narrowphase). Each island runs an independent sub-DAG: narrowphase → contact reduction → sequential-impulse solve. A join barrier synchronises before merging per-island manifolds for global joint solving and fracture checks.
- [x] **Task-graph profiling** — Per-task wall-clock timing via `enableTaskGraphProfiling()`, exposed as `taskGraphProfileEvents()` and `taskGraphProfileSummary()`. Zero overhead when disabled.
- [x] **Per-island CCD** — CCD runs within each island's sub-DAG (CCD → narrowphase → reduce → solve), enabling CCD to overlap with other island work. Isolated bodies form size-1 islands and still receive CCD processing. `runCCDForBodies()` operates on arbitrary body subsets.
- [x] **All-bodies-in-islands** — `buildFromPairs` now assigns every body (including isolated ones) to an island, enabling per-island CCD and simplifying the pipeline structure.

### P1 — Short-Term

- [x] GitHub Actions CI — auto-build + test on push for GCC, Clang, MSVC (5 configurations: GCC/Clang on Linux, AppleClang on macOS, MSVC/Clang-CL on Windows, both Release/Debug)
- [x] MSVC / Clang-CL compatibility pass — cross-compiler flags, member initialisation-order fix, `_aligned_malloc` path
- [x] Task-graph DOT export — `toDot()` method for offline DAG visualization with Graphviz
- [x] SIMD-accelerated broadphase (`SimdBruteForce`) — SSE2 batch AABB overlap tests for 2–3× pair-generation throughput

### P2 — Short-Term (next)

- [ ] Benchmark suite — broadphase throughput, solver convergence, CCD performance
- [ ] Soft body self-collision — particle-particle repulsion within PBD soft bodies
- [ ] Continuous soft body-rigid body CCD — swept collision for fast-moving particles
- [ ] Task-graph JSON export — structured format for tool integration
- [ ] SIMD narrowphase dispatch — vectorised SAT axis projection for box-box collision

### P3 — Medium-Term

- [ ] Soft body self-collision — particle-particle repulsion within PBD soft bodies
- [ ] Continuous soft body-rigid body CCD — swept collision for fast-moving particles

---

## License

CiPhyxs is open source under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## References

- **Erwin Coumans**, *Bullet Physics SDK* — Sequential-impulse constraint solver
- **Nvidia**, *PhysX SDK 5.x* — Task graph, island formation, CCD
- **Unreal Engine**, *Chaos Physics* — Deterministic parallelism, MPR collision
- **David Baraff**, *Rigid Body Simulation* (SIGGRAPH 97 course notes)
- **Brian Mirtich**, *Impulse-based Dynamic Simulation of Rigid Body Systems*
