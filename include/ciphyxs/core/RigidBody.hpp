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
#include <span>
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
// CcdMode
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Continuous Collision Detection mode for a rigid body.
///
/// Controls how the CCD system prevents tunnelling for this body.
/// Higher modes provide stronger guarantees at increased CPU cost.
enum class CcdMode : std::uint8_t {
    None,            ///< No CCD for this body.
    Cast,            ///< Standard CCD sweep: cast the shape along its velocity and
                     ///< resolve the earliest TOI.
    ClampVelocity,   ///< Clamp the body's velocity so that `|v| * dt < boundingRadius * 0.5`.
                     ///< Prevents tunnelling at extreme speeds without sub-stepping.
    SubStep          ///< Automatic sub-stepping: divide the frame into N steps such that
                     ///< each sub-step moves less than `boundingRadius * 0.5`. Most robust.
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

    // ─── Collision filtering ───────────────────────────────────────────────────────
    std::uint32_t collisionGroup = 0xFFFFFFFF;   ///< Bitmask: which groups this body belongs to.
    std::uint32_t collisionMask  = 0xFFFFFFFF;   ///< Bitmask: which groups this body collides with.

    // ─── Flags ──────────────────────────────────────────────────────────────────
    MotionType motionType = MotionType::Dynamic;
    bool       startActive = true;
    CcdMode    ccdMode     = CcdMode::Cast;   ///< CCD mode for this body.
                                              ///< Default: Cast (standard swept CCD).

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
// RigidBodyHotSpan  — grouped view of solver-hot fields
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A grouped span view of the solver-hot subset of RigidBodyStorage.
///
/// These fields are touched **multiple times per frame** by the constraint solver,
/// integration loops, and CCD.  Grouping them together documents performance intent
/// and enables future physical repacking (e.g. into a single AoS array) without
/// changing call sites.
///
/// Cold fields (material, shape geometry, flags, sleep) are accessed directly on
/// `RigidBodyStorage` and are NOT included here.
///
/// @tparam Vec3T   `Vec3f` or `const Vec3f`
/// @tparam QuatT   `Quaternionf` or `const Quaternionf`
/// @tparam FloatT  `float` or `const float`
template <typename Vec3T, typename QuatT, typename FloatT>
struct RigidBodyHotSpanT {
    std::span<Vec3T>   positions;
    std::span<QuatT>   rotations;
    std::span<Vec3T>   linearVelocities;
    std::span<Vec3T>   angularVelocities;
    std::span<Vec3T>   forces;
    std::span<Vec3T>   torques;
    std::span<FloatT>  inverseMasses;
    std::span<Vec3T>   inverseInertiaDiag;
    std::span<QuatT>   inertiaRotations;

    /// @brief  Number of bodies in the view.
    [[nodiscard]] std::size_t size() const noexcept { return positions.size(); }

    /// @brief  True when the view is empty.
    [[nodiscard]] bool empty() const noexcept { return positions.empty(); }
};

/// @brief  Mutable hot-data span.
using RigidBodyHotSpan = RigidBodyHotSpanT<Vec3f, Quaternionf, float>;

/// @brief  Const hot-data span.
using RigidBodyHotConstSpan = RigidBodyHotSpanT<const Vec3f, const Quaternionf, const float>;

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RigidBodyStorage  (Data-Oriented Design — Structure of Arrays)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Contiguous SoA container for all rigid bodies in the world.
///
/// ## Hot / cold split
///
/// Fields are grouped into two categories:
///   - **HOT**  — solver-touched per-iteration (positions, velocities, masses,
///                forces, torques).  Accessed via `hot()` which returns a
///                `RigidBodyHotSpan` view.
///   - **COLD** — touched once per frame or less (material, shapes, flags, sleep).
///                Accessed directly on `RigidBodyStorage`.
///
/// This split improves cache utilization because solver loops only pull in the
/// hot cache lines, leaving cold data to be evicted.
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

    //
    // ─── HOT fields (touched per-iteration by solver and integration) ────────────
    //
    // These fields are accessed multiple times per frame by:
    //   - ConstraintSolver::solveManifold / positionalSolve / applyImpulse
    //   - JointSolver::solveRow / applyRow
    //   - PhysicsWorld::integrateForces / integrateVelocities / integratePositions
    //   - CCD sweep in Continuous.hpp
    //
    // Use the hot() accessor to signal that only this subset is needed in hot loops.
    //

    AlignedVector<Vec3f, 16>      positions;              ///< World-space position.
    AlignedVector<Quaternionf, 16> rotations;             ///< World-space orientation.
    AlignedVector<Vec3f, 16>      linearVelocities;       ///< Linear velocity (world space).
    AlignedVector<Vec3f, 16>      angularVelocities;      ///< Angular velocity (world space).
    AlignedVector<Vec3f, 16>      forces;                 ///< Net force (world space).
    AlignedVector<Vec3f, 16>      torques;                ///< Net torque (world space).
    AlignedVector<float, 16>      inverseMasses;          ///< 1/mass. 0 = infinite mass.
    AlignedVector<Vec3f, 16>      inverseInertiaDiag;     ///< Diagonal of inverse inertia tensor (local).
    AlignedVector<Quaternionf, 16> inertiaRotations;      ///< Orientation of the inertia frame.

    //
    // ─── COLD fields (touched once per frame or less) ──────────────────────────
    //
    // Material, shape geometry, flags, and sleep state.  Stored separately so that
    // solver hot loops do not pull these into cache.
    //

    AlignedVector<float, 16>      restitutions;
    AlignedVector<float, 16>      frictions;
    AlignedVector<float, 16>      linearDamping;
    AlignedVector<float, 16>      angularDamping;

    // Compound shapes (flat SoA arrays).
    std::vector<ShapeHandle>       flatShapeHandles;
    AlignedVector<Vec3f, 16>       flatShapeLocalPositions;
    AlignedVector<Quaternionf, 16> flatShapeLocalRotations;
    std::vector<std::uint32_t>     shapeStart;
    std::vector<std::uint32_t>     shapeCount;

    // Flags.
    std::vector<MotionType>        motionTypes;
    AlignedVector<uint8_t, 16>     activeFlags;
    AlignedVector<uint8_t, 16>     ccdModes;
    AlignedVector<float, 16>       sleepTimers;
    AlignedVector<std::uint32_t, 16> collisionGroups;
    AlignedVector<std::uint32_t, 16> collisionMasks;

    // ─── Hot / cold accessors ───────────────────────────────────────────────────────────────

    /// @brief  Return a grouped span of solver-hot fields (mutable).
    ///
    /// Use this in solver and integration hot loops to signal that only the hot subset
    /// is needed, enabling future physical repacking without changing call sites.
    [[nodiscard]] RigidBodyHotSpan hot() noexcept {
        return {
            .positions        = std::span(positions),
            .rotations        = std::span(rotations),
            .linearVelocities = std::span(linearVelocities),
            .angularVelocities= std::span(angularVelocities),
            .forces           = std::span(forces),
            .torques          = std::span(torques),
            .inverseMasses    = std::span(inverseMasses),
            .inverseInertiaDiag = std::span(inverseInertiaDiag),
            .inertiaRotations = std::span(inertiaRotations),
        };
    }

    /// @brief  Return a grouped span of solver-hot fields (const).
    [[nodiscard]] RigidBodyHotConstSpan hot() const noexcept {
        return {
            .positions        = std::span<const Vec3f>(positions),
            .rotations        = std::span<const Quaternionf>(rotations),
            .linearVelocities = std::span<const Vec3f>(linearVelocities),
            .angularVelocities= std::span<const Vec3f>(angularVelocities),
            .forces           = std::span<const Vec3f>(forces),
            .torques          = std::span<const Vec3f>(torques),
            .inverseMasses    = std::span<const float>(inverseMasses),
            .inverseInertiaDiag = std::span<const Vec3f>(inverseInertiaDiag),
            .inertiaRotations = std::span<const Quaternionf>(inertiaRotations),
        };
    }

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
        ccdModes.reserve(count);
        sleepTimers.reserve(count);
        collisionGroups.reserve(count);
        collisionMasks.reserve(count);
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
        activeFlags.emplace_back(desc.startActive ? uint8_t(1) : uint8_t(0));
        ccdModes.emplace_back(static_cast<uint8_t>(desc.ccdMode));
        sleepTimers.emplace_back(0.0f);
        collisionGroups.emplace_back(desc.collisionGroup);
        collisionMasks.emplace_back(desc.collisionMask);

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
            swap(ccdModes);
            swap(sleepTimers);
            swap(collisionGroups);
            swap(collisionMasks);
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
        ccdModes.pop_back();
        sleepTimers.pop_back();
        collisionGroups.pop_back();
        collisionMasks.pop_back();
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
        ccdModes.clear();
        collisionGroups.clear();
        collisionMasks.clear();
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
