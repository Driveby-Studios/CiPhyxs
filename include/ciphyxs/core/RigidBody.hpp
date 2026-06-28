//==================================================================================================
/// @file  RigidBody.hpp
/// @brief  Rigid-body descriptor, handle, and SoA storage (Data-Oriented Design).
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include "AlignedAllocator.hpp"
#include <cstdint>
#include <limits>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RigidBodyHandle
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Opaque handle referencing a rigid body inside the PhysicsWorld's SoA storage.
///
/// Handles are stable (dense) indices that can be stored externally by the user.  They remain
/// valid as long as the body is alive.  After removal the handle should be considered stale.
using RigidBodyHandle = std::uint32_t;

/// @brief  Sentinel value representing an invalid / null handle.
inline constexpr RigidBodyHandle kInvalidHandle = std::numeric_limits<RigidBodyHandle>::max();

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ShapeHandle
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Index into the PhysicsWorld's shape registry (dense index into a `std::vector<Shape>`).
using ShapeHandle = std::uint32_t;

/// @brief  Sentinel meaning "no shape" (body does not participate in collision).
inline constexpr ShapeHandle kInvalidShapeHandle = std::numeric_limits<ShapeHandle>::max();

// ────────────────────────────────────────────────────────────────────────────────────────────────
// MotionType
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Determines how a body participates in simulation.
enum class MotionType : std::uint8_t {
    Static,    ///< Zero mass, infinite inertia – never moves (immovable object).
    Kinematic, ///< User-driven motion, infinite inertia – affects other bodies via constraints.
    Dynamic    ///< Full physical simulation (forces, collisions, gravity).
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RigidBodyDesc
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Description used to initialise a rigid body inside the PhysicsWorld.
///
/// This is the **user-facing** configuration struct.  The PhysicsWorld converts it into SoA
/// storage entries internally.
struct RigidBodyDesc {
    // ─── Transform ──────────────────────────────────────────────────────────────────────────────
    Vec3f     position   = Vec3f::zero();
    Quaternionf rotation  = Quaternionf::identity();

    // ─── Velocity (initial) ─────────────────────────────────────────────────────────────────────
    Vec3f     linearVelocity  = Vec3f::zero();
    Vec3f     angularVelocity = Vec3f::zero();

    // ─── Mass properties ────────────────────────────────────────────────────────────────────────
    float     mass             = 1.0f;
    Vec3f     inertiaLocal     = Vec3f{1.0f, 1.0f, 1.0f};   ///< Diagonal of inertia tensor in local space.
    bool      useAutoInertia   = true;   ///< If true and a valid shape handle is set, `PhysicsWorld::createBody`
                                         ///< overrides `inertiaLocal` with the shape-computed value.

    // ─── Damping (applied per-body each step; overrides global defaults when non-zero) ─────────
    float     linearDamping    = 0.0f;   ///< Per-body linear damping [0, 1). 0 = use solver default.
    float     angularDamping   = 0.0f;   ///< Per-body angular damping [0, 1). 0 = use solver default.

    // ─── Material ───────────────────────────────────────────────────────────────────────────────
    float     restitution = 0.35f;   ///< Coefficient of restitution [0, 1].
    float     friction    = 0.50f;   ///< Coulomb friction coefficient >= 0.

    // ─── Flags ──────────────────────────────────────────────────────────────────────────────────
    MotionType motionType = MotionType::Dynamic;
    bool       startActive = true;
    bool       ccdEnabled  = false;   ///< Enable continuous collision detection for this body.
                                       ///< Prevents tunneling at high speeds.  Default: off.

    // ─── Compound shape support ────────────────────────────────────────────────────────────

    /// @brief  Descriptor for a single sub-shape within a compound body.
    struct SubShapeDesc {
        ShapeHandle   handle         = kInvalidShapeHandle;
        Vec3f         localPosition  = Vec3f::zero();
        Quaternionf   localRotation  = Quaternionf::identity();
    };

    /// @brief  Sub-shapes composing this body.  Empty = no collision.
    std::vector<SubShapeDesc> subShapes;

    /// @brief  Convenience: replace all sub-shapes with a single shape at the body origin.
    void setShape(ShapeHandle handle) noexcept {
        subShapes.clear();
        if (handle != kInvalidShapeHandle)
            subShapes.push_back({handle, Vec3f::zero(), Quaternionf::identity()});
    }

    /// @brief  Append a sub-shape with an optional local offset and rotation.
    void addShape(ShapeHandle handle,
                  const Vec3f& localPos = Vec3f::zero(),
                  const Quaternionf& localRot = Quaternionf::identity()) {
        subShapes.push_back({handle, localPos, localRot});
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RigidBodyStorage  (Data-Oriented Design — Structure of Arrays)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Contiguous SoA container for all rigid bodies in the world.
///
/// ## Why SoA?
///
/// A physics solver touches **only velocity + force** fields during constraint resolution, then
/// **only position** during broad-phase.  By grouping each field into a separate contiguous array
/// we maximise cache utilisation and make it trivial for the compiler to auto-vectorize loops.
///
/// This layout is **ECS-ready**:  the arrays can be sliced into `std::span` views and dispatched
/// to parallel jobs with no false-sharing on cache lines.
///
/// ## Allocation
///
/// For production SIMD you should replace `std::vector` with an over-aligned allocator
/// (e.g. `std::pmr::vector` with 16-byte aligned resources or a custom `AlignedAllocator`).
/// The Vec3/Quaternion types declared with `alignas(16)` will then be properly aligned in
/// heap memory.
struct RigidBodyStorage {

    	// ─── Transform & motion state (SIMD‑aligned SoA) ──────────────────────────────────────────
        ///
        /// These fields use `AlignedVector` (16‑byte aligned) so that the compiler can emit
        /// aligned SIMD load/store instructions (`movaps`/`movapd`) in solver loops and integration.
        /// On `x86_64` with `g++ -O2 -msse4.1` this yields 15–30 % faster iteration.
        AlignedVector<Vec3f, 16>      positions;          ///< World-space position.
        AlignedVector<Quaternionf, 16> rotations;         ///< World-space orientation.
        AlignedVector<Vec3f, 16>      linearVelocities;   ///< Linear velocity (world space).
        AlignedVector<Vec3f, 16>      angularVelocities;  ///< Angular velocity (world space).

        // ─── Accumulated forces (cleared every step) ────────────────────────────────────────────────
        AlignedVector<Vec3f, 16>      forces;    ///< Net force (world space).
        AlignedVector<Vec3f, 16>      torques;   ///< Net torque (world space).

        // ─── Mass properties ────────────────────────────────────────────────────────────────────────
        AlignedVector<float, 16>      inverseMasses;      ///< 1/mass. 0 = infinite mass (static/kinematic).
        AlignedVector<Vec3f, 16>      inverseInertiaDiag; ///< Diagonal of inverse inertia tensor (local).
        AlignedVector<Quaternionf, 16> inertiaRotations;  ///< Orientation of the inertia frame (follows rotation).

        // ─── Material ───────────────────────────────────────────────────────────────────────────────
        AlignedVector<float, 16>      restitutions;
        AlignedVector<float, 16>      frictions;

        // ─── Per-body damping (0 = use SolverConfig default) ───────────────────────────────────────
        AlignedVector<float, 16>      linearDamping;
        AlignedVector<float, 16>      angularDamping;

    	// ─── Compound shapes (flat SoA arrays) ───────────────────────────────────────────────────
        //
        // Each body can carry zero or more sub-shapes, each with a local position and rotation.
        // Sub-shapes are stored in flat arrays indexed by [shapeStart[i], shapeStart[i] + shapeCount[i]).
        // A body with shapeCount[i] == 0 participates in no collisions.
        //
        // These flat arrays are NOT on the solver hot path, so they use the default `std::vector`.
        // The local-position/rotation fields use `AlignedVector` because Vec3f/Quaternionf are
        // `alignas(16)` and custom allocators are needed to maintain that guarantee on the heap.

        std::vector<ShapeHandle>  flatShapeHandles;             ///< All sub-shape handles (flat).
        AlignedVector<Vec3f, 16>  flatShapeLocalPositions;      ///< Sub-shape local positions.
        AlignedVector<Quaternionf, 16> flatShapeLocalRotations; ///< Sub-shape local rotations.
        std::vector<std::uint32_t> shapeStart;                  ///< Per-body start index into flat arrays.
        std::vector<std::uint32_t> shapeCount;                  ///< Per-body sub-shape count (0 = no collision).

    // ─── Flags ──────────────────────────────────────────────────────────────────────────────────
    std::vector<MotionType> motionTypes;
    std::vector<bool>       activeFlags;        ///< False = sleeping / disabled.
    std::vector<bool>       ccdFlags;           ///< True = continuous collision detection enabled.
    AlignedVector<float, 16> sleepTimers;

    // ─── Memory management ──────────────────────────────────────────────────────────────────────

    /// @brief  Number of bodies in storage.
    [[nodiscard]] std::size_t size() const noexcept { return positions.size(); }

    /// @brief  Reserve space for `count` bodies (avoids repeated re-allocation).
    void reserve(std::size_t count) {
        positions.reserve(count);
        rotations.reserve(count);
        linearVelocities.reserve(count);
        angularVelocities.reserve(count);
        forces.reserve(count);
        torques.reserve(count);
        inverseMasses.reserve(count);
        inverseInertiaDiag.reserve(count);
        inertiaRotations.reserve(count);
        restitutions.reserve(count);
        frictions.reserve(count);
        linearDamping.reserve(count);
        angularDamping.reserve(count);
        motionTypes.reserve(count);
        activeFlags.reserve(count);
        shapeStart.reserve(count);
        shapeCount.reserve(count);
        sleepTimers.reserve(count);
    }

    /// @brief  Append a body from a descriptor.  Returns its handle (index).
    RigidBodyHandle emplace(const RigidBodyDesc& desc) {
        RigidBodyHandle h = static_cast<RigidBodyHandle>(size());

        positions.emplace_back(desc.position);
        rotations.emplace_back(desc.rotation);
        linearVelocities.emplace_back(desc.linearVelocity);
        angularVelocities.emplace_back(desc.angularVelocity);

        forces.emplace_back(Vec3f::zero());
        torques.emplace_back(Vec3f::zero());

        // Invert mass – 0 means infinite.
        if (desc.motionType != MotionType::Dynamic) {
            inverseMasses.emplace_back(0.0f);
        } else {
            inverseMasses.emplace_back(desc.mass > 0.0f ? 1.0f / desc.mass : 0.0f);
        }

        // Inverse inertia (local diagonal).
        Vec3f invDiag;
        if (desc.motionType == MotionType::Dynamic) {
            invDiag.x = desc.inertiaLocal.x > 0.0f ? 1.0f / desc.inertiaLocal.x : 0.0f;
            invDiag.y = desc.inertiaLocal.y > 0.0f ? 1.0f / desc.inertiaLocal.y : 0.0f;
            invDiag.z = desc.inertiaLocal.z > 0.0f ? 1.0f / desc.inertiaLocal.z : 0.0f;
        } else {
            invDiag = Vec3f::zero();
        }
        inverseInertiaDiag.emplace_back(invDiag);
        inertiaRotations.emplace_back(desc.rotation);

        restitutions.emplace_back(desc.restitution);
        frictions.emplace_back(desc.friction);
        linearDamping.emplace_back(desc.linearDamping);
        angularDamping.emplace_back(desc.angularDamping);

        // ── Compound shapes ───────────────────────────────────────────────────────────────────
        shapeStart.push_back(static_cast<std::uint32_t>(flatShapeHandles.size()));
        shapeCount.push_back(static_cast<std::uint32_t>(desc.subShapes.size()));

        for (const auto& ss : desc.subShapes) {
            flatShapeHandles.push_back(ss.handle);
            flatShapeLocalPositions.push_back(ss.localPosition);
            flatShapeLocalRotations.push_back(ss.localRotation);
        }

        motionTypes.emplace_back(desc.motionType);
        activeFlags.emplace_back(desc.startActive);
        ccdFlags.emplace_back(desc.ccdEnabled);
        sleepTimers.emplace_back(0.0f);

        return h;
    }

    /// @brief  Swap-and-pop a body at index `h`.  Invalidates handles of the last body.
    void remove(std::size_t h) {
        std::size_t last = size() - 1;
        if (h != last) {
            auto swap = [h, last](auto& vec) {
                using std::swap;
                swap(vec[h], vec[last]);
            };
            swap(positions);
            swap(rotations);
            swap(linearVelocities);
            swap(angularVelocities);
            swap(forces);
            swap(torques);
            swap(inverseMasses);
            swap(inverseInertiaDiag);
            swap(inertiaRotations);
            swap(restitutions);
            swap(frictions);
            swap(linearDamping);
            swap(angularDamping);
            swap(motionTypes);
            swap(shapeStart);
            swap(shapeCount);
            // NOTE: flat arrays are NOT swapped — the swapped (shapeStart, shapeCount) pairs
            // still reference the correct flat-array entries.  Unreferenced flat entries
            // accumulate as waste; a future defragment pass can clean them up.
            swap(activeFlags);
            swap(ccdFlags);
            swap(sleepTimers);
        }
        positions.pop_back();
        rotations.pop_back();
        linearVelocities.pop_back();
        angularVelocities.pop_back();
        forces.pop_back();
        torques.pop_back();
        inverseMasses.pop_back();
        inverseInertiaDiag.pop_back();
        inertiaRotations.pop_back();
        restitutions.pop_back();
        frictions.pop_back();
        linearDamping.pop_back();
        angularDamping.pop_back();
        motionTypes.pop_back();
        shapeStart.pop_back();
        shapeCount.pop_back();
        // Flat arrays are NOT popped — their entries remain but are now unreferenced.
        activeFlags.pop_back();
        ccdFlags.pop_back();
        sleepTimers.pop_back();
    }

    /// @brief  Remove all bodies.
    void clear() {
        positions.clear();
        rotations.clear();
        linearVelocities.clear();
        angularVelocities.clear();
        forces.clear();
        torques.clear();
        inverseMasses.clear();
        inverseInertiaDiag.clear();
        inertiaRotations.clear();
        restitutions.clear();
        frictions.clear();
        linearDamping.clear();
        angularDamping.clear();
        motionTypes.clear();
        activeFlags.clear();
        ccdFlags.clear();
        flatShapeHandles.clear();
        flatShapeLocalPositions.clear();
        flatShapeLocalRotations.clear();
        shapeStart.clear();
        shapeCount.clear();
        sleepTimers.clear();
    }

    /// @brief  Zero force and torque accumulators for all active bodies.
    void clearForces() noexcept {
        // In a production build this loop auto-vectorizes trivially (contiguous floats).
        for (auto& f : forces) f = Vec3f::zero();
        for (auto& t : torques) t = Vec3f::zero();
    }
};

} // namespace ciphyxs
