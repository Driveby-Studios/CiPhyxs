//==================================================================================================
/// @file  CiPhyxs.hpp
/// @brief  Umbrella header – include this to use the entire CiPhyxs physics engine.
///
/// @mainpage CiPhyxs — High-Performance Physics Engine
///
/// CiPhyxs is an open-source (MIT), header-only physics engine written in C++20.
/// It is designed for **heavy, mechanical, and industrial simulations** where numerical
/// stability and high mass ratios are critical.
///
/// ## Design Pillars
///
/// - **High-Impact Dynamics** — Symplectic (semi-implicit Euler) integration, sequential-impulse
///   constraint solver with Baumgarte stabilisation, and careful handling of inverse mass / inertia
///   to avoid singularities at extreme mass ratios (10⁶:1+).
///
/// - **Modular & Decoupled** — Framework-agnostic with zero external dependencies.
///   Integrate into any C++ project (game engine, robotics simulator, CAD tool).
///
/// - **Performance-First** — Structure-of-Arrays (SoA) body storage for cache-friendly
///   iteration.  Types aligned to 16 bytes for SIMD load/store.  Brute-force broadphase
///   (swap for spatial hash / BVH at scale).
///
/// - **Professional API** — Doxygen-documented public interfaces mirroring the clarity
///   of enterprise physics SDKs (PhysX, Bullet, Chaos).
///
/// ## Module Overview
///
/// | Module              | Headers                          | Description                              |
/// |---------------------|----------------------------------|------------------------------------------|
/// | **Math**            | `math/Vec3.hpp`                  | 3D vector (alignas(16), concept-safe)    |
/// |                     | `math/Mat3.hpp`                  | 3×3 column-major matrix                  |
/// |                     | `math/Quaternion.hpp`            | Hamilton quaternion with slerp           |
/// | **Core**            | `core/RigidBody.hpp`             | SoA storage, descriptors, body handles   |
/// |                     | `core/PhysicsWorld.hpp`          | World orchestrator, pipeline, hooks      |
/// |                     | `core/ConstraintSolver.hpp`      | Sequential-impulse solver (warm start)   |
/// |                     | `core/IslandSolver.hpp`          | Union‑find island formation + solving    |
/// |                     | `core/ThreadPool.hpp`            | Lightweight C++20 task parallelism       |
/// | **Collision**       | `collision/Shape.hpp`            | Sphere, Box, Plane, Capsule primitives + AABB |
/// |                     | `collision/Contact.hpp`          | Contact point / manifold data structures      |
/// |                     | `collision/Broadphase.hpp`       | Brute-force + spatial hash pair gen.         |
/// |                     | `collision/Narrowphase.hpp`      | Primitive collision tests (SAT-based)        |
/// |                     | `collision/SpatialHash.hpp`      | O(n) 3D uniform-grid broadphase              |
/// |                     | `collision/Bvh.hpp`              | BVH acceleration for triangle mesh collision |
/// |                     | `collision/Ray.hpp`              | Ray casting API (all primitives)             |
/// |                     | `collision/Convex.hpp`            | GJK + EPA for convex-convex collision        |
///
/// ## Quick Start
///
/// @code{.cpp}
/// #include <ciphyxs/CiPhyxs.hpp>
///
/// int main() {
///     using namespace ciphyxs;
///
///     PhysicsWorld world;
///
///     // 1. Create shapes.
///     ShapeHandle groundShape = world.createShape(Plane{});
///     ShapeHandle boxShape    = world.createShape(Box{Vec3f(0.5f, 0.5f, 0.5f)});
///
///     // 2. Create a static ground plane.
///     RigidBodyDesc ground;
///     ground.motionType  = MotionType::Static;
///     ground.shapeHandle = groundShape;
///     world.createBody(ground);
///
///     // 3. Create a falling box.
///     RigidBodyDesc box;
///     box.mass         = 10.0f;
///     box.position     = Vec3f(0, 10, 0);
///     box.shapeHandle  = boxShape;
///     world.createBody(box);
///
///     // 4. Simulate 120 frames at 60 FPS.
///     for (int i = 0; i < 120; ++i) {
///         world.step(1.0f / 60.0f);
///         auto& pos = world.bodies().positions;
///         // ... render
///     }
/// }
/// @endcode
///
/// @see  PhysicsWorld   — Main simulation orchestrator
/// @see  RigidBodyDesc  — Body creation descriptor
/// @see  ConstraintSolver — Warm-started sequential impulse
/// @see  Narrowphase    — SAT-based collision detection
//==================================================================================================
#pragma once

#include "math/Vec3.hpp"
#include "math/Mat3.hpp"
#include "math/Quaternion.hpp"

#include "core/Assert.hpp"
#include "core/RigidBody.hpp"
#include "core/ConstraintSolver.hpp"
#include "core/PhysicsWorld.hpp"
#include "core/ThreadPool.hpp"
#include "core/TaskGraph.hpp"
#include "core/IslandSolver.hpp"
#include "core/Joint.hpp"
#include "core/JointSolver.hpp"
#include "core/Ragdoll.hpp"
#include "core/DebugRenderer.hpp"
#include "core/CharacterController.hpp"
#include "core/SoftBody.hpp"

#include "collision/Shape.hpp"
#include "collision/Contact.hpp"
#include "collision/Broadphase.hpp"
#include "collision/Narrowphase.hpp"
#include "collision/SpatialHash.hpp"
#include "collision/SweepAndPrune.hpp"
#include "collision/Continuous.hpp"
#include "collision/Convex.hpp"
#include "collision/Bvh.hpp"
#include "collision/Ray.hpp"
#include "collision/Fracture.hpp"

#include "solver/Vehicle.hpp"
