//==================================================================================================
/// @file  PhysicsWorld.hpp
/// @brief  Physics world orchestrator – fixed‑timestep pipeline with full collision detection
///         and a sequential‑impulse constraint solver.
//==================================================================================================
#pragma once

#include "RigidBody.hpp"
#include "SoftBody.hpp"
#include "../math/SimdIntegration.hpp"
#include "ConstraintSolver.hpp"
#include "IslandSolver.hpp"
#include "Joint.hpp"
#include "JointSolver.hpp"
#include "ThreadPool.hpp"
#include "TaskGraph.hpp"
#include "DebugRenderer.hpp"
#include "../collision/Shape.hpp"
#include "../collision/Contact.hpp"
#include "../collision/Broadphase.hpp"
#include "../collision/Narrowphase.hpp"
#include "../collision/SpatialHash.hpp"
#include "../collision/SweepAndPrune.hpp"
#include "../collision/Dbvt.hpp"
#include "../collision/Continuous.hpp"
#include "../collision/Ray.hpp"
#include "../collision/Fracture.hpp"
#include "../collision/ContactReduction.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <span>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

// ─── Validation guard ─────────────────────────────────────────────────────────────────────────
// Define CIPHYXS_ENABLE_VALIDATION to enable NaN/Inf guards in hot loops.
// When defined, each integrated body is checked for NaN/Inf positions and velocities.
// When undefined (default for optimal performance on constrained hardware),
// these checks are skipped entirely, eliminating branches in the hot loop and
// enabling better compiler auto-vectorization.
//
// Define this during development/debugging; undefine for release.
#ifndef CIPHYXS_ENABLE_VALIDATION
#define CIPHYXS_ENABLE_VALIDATION 0
#endif

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ISolverHook
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Abstract interface for extending the solver pipeline.
///
/// Register your hook with `PhysicsWorld::addHook()`.  Hooks are called in registration order
/// during each fixed step.
///
/// Typical uses:
///   - Custom force generators (springs, dampers, buoyancy, aerodynamics).
///   - Custom constraint types (hinges, sliders, revolute joints).
///   - Per-step debug drawing or statistics gathering.
struct ISolverHook {
    virtual ~ISolverHook() = default;

    /// @brief  Apply forces / torques to bodies.  Called at the start of every fixed step.
    ///         Store impulses in `bodies.forces[i]` / `bodies.torques[i]`.
    /// @param dt  Fixed timestep for this sub-step.
    /// @param bodies  The world's SoA rigid-body storage (read/write).
    virtual void onApplyForces(float dt, RigidBodyStorage& bodies) = 0;

    /// @brief  Called after velocity integration, before constraint solving.
    virtual void onPreSolve(float dt, RigidBodyStorage& bodies) = 0;

    /// @brief  Post-solve hook — called after constraint solving.  Post-stabilisation or debug output.
    virtual void onPostSolve(float dt, RigidBodyStorage& bodies) = 0;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ICollisionFilter
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Abstract interface for filtering collision pairs.
///
/// Register a filter with `PhysicsWorld::setCollisionFilter()`.  The filter is called
/// once per broad-phase pair **before** the narrowphase test.  If the filter returns
/// false the pair is skipped entirely.
///
/// @note  The filter is also queried for pairs within the same body (compound shapes)
///        to allow or reject self-collisions between sub-shapes.
struct ICollisionFilter {
    virtual ~ICollisionFilter() = default;

    /// @brief  Should two bodies collide?
    /// @param bodyA  Handle of the first body.
    /// @param bodyB  Handle of the second body.
    /// @return  true to allow the narrowphase test, false to skip.
    ///
    /// The filter is called once per broad-phase pair during `detectCollisions()`.
    /// Bodies A and B are guaranteed to be distinct.
    virtual bool shouldCollide(RigidBodyHandle bodyA, RigidBodyHandle bodyB) const noexcept = 0;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// PhysicsWorldConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Configuration parameters for a PhysicsWorld instance.
struct PhysicsWorldConfig {
    /// Gravity vector (world space).  Earth: `{0, -9.81, 0}`.
    Vec3f gravity = Vec3f(0.0f, -9.81f, 0.0f);

    /// Duration of each physics sub-step in seconds.  1/60 ≈ 16.67 ms.
    float fixedTimestep = 1.0f / 60.0f;

    /// @brief  Maximum accumulated wall-clock time before clamping (prevents "spiral of death").
    float maxAccumulator = 1.0f / 10.0f;

    // ─── Damping (Rayleigh model) ────────────────────────────────────────────────────────────

    /// @brief  Linear Rayleigh damping coefficient (N·s/m).
    ///         Applied as `F_damp = -c * v` in the force integration phase.
    ///         A value of 0.5 gives moderate damping for a unit-mass body.
    ///         Set to 0 for undamped motion.
    float linearDamping = 0.5f;

    /// @brief  Angular Rayleigh damping coefficient (N·m·s/rad).
    ///         Applied as `tau_damp = -c * omega` in the force integration phase.
    ///         A value of 0.5 gives moderate damping for a typical body.
    float angularDamping = 0.5f;

    // ─── Sleeping ────────────────────────────────────────────────────────────────────────────
    /// Bodies with kinetic energy below this threshold are candidates for sleep.
    float sleepEnergyThreshold = 0.005f;

    /// Seconds a body must remain under the sleep threshold before being put to sleep.
    float sleepTimeRequired = 0.5f;

    // ─── Parallel solver ─────────────────────────────────────────────────────────────────────
    /// @brief  Enable multi‑threaded island‑based constraint solving.
    ///
    /// When enabled, the contact graph is partitioned into independent groups (islands)
    /// using union‑find, and each island is solved concurrently via the thread pool.
    /// Results are bit‑identical to single‑threaded execution.
    bool enableParallelSolver = false;

    /// @brief  Number of worker threads for the solver thread pool.
    /// 0 = `std::thread::hardware_concurrency()`.  Only meaningful when
    /// `enableParallelSolver == true`.
    std::uint32_t numThreads = 0;

    // ─── Continuous Collision Detection (CCD) ────────────────────────────────────────────────

    /// @brief  Speed threshold (m/s) for CCD.  Bodies with CCD enabled moving slower than
    ///         this skip the CCD pass to save CPU.  Set to 0 to always CCD.
    float ccdSpeedThreshold = 10.0f;

    /// @brief  Maximum CCD sub-steps per body per frame.
    ///         Most cases need only 1 sub-step.  Increase for fast tumblers.
    int ccdMaxSubSteps = 1;

    // ─── Task-graph pipeline ────────────────────────────────────────────────────────────────

    /// @brief  Enable the DAG-based task-graph solver pipeline.
    ///
    /// When enabled, `fixedStep()` builds a `TaskGraph` of the solver pipeline stages and
    /// executes them on the thread pool.  This allows **pipeline parallelism** — independent
    /// stages such as CCD and collision detection run concurrently, and per-island solving
    /// overlaps with other pipeline work.
    ///
    /// When disabled (default), the pipeline runs sequentially as a simple loop.
    /// Results are bit-identical between sequential and task-graph modes because the task
    /// graph preserves the same dependency order.
    bool enableTaskGraphPipeline = false;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// PhysicsWorld
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Core physics world that owns the rigid-body SoA storage and drives the full
///         fixed-timestep solver pipeline including collision detection.
///
/// ## Pipeline (each fixed sub-step)
///
///  1. **Apply forces**        – clear accumulators, apply gravity, invoke hooks.
///  2. **Integrate velocities** – semi-implicit Euler:  `v += a * dt`.
///  3. **Collision detection**  – broadphase (AABB culling) + narrowphase (primitive tests).
///  4. **Pre-solve**           – hook call (warm starting, cache init).
///  5. **Solve constraints**   – sequential impulse with friction and Baumgarte stabilisation.
///  6. **Post-solve**          – hook call (debug, stats).
///  7. **Integrate positions** – `x += v' * dt`; `q += ½ ω' * q * dt`.
///  8. **Sleep management**    – put inert bodies to sleep.
///
/// ## Shape management
///
/// Shapes live in a registry owned by the world.  Attach a shape handle to a body descriptor
/// at creation time, or call `setBodyShape()` later.
///
/// @code
/// auto hSphere = world.createShape(Sphere{1.0f});
/// RigidBodyDesc desc;
/// desc.shapeHandle = hSphere;
/// auto body = world.createBody(desc);
/// @endcode
///
/// @note  This class is **not** thread‑safe.  Advance the simulation from a single thread.
class PhysicsWorld {
public:
    // ─── Preset flags ─────────────────────────────────────────────────────────────────────────

    /// @brief  Quick-preset for common hardware targets.
    ///
    /// - `Default`:  Balanced accuracy/performance. 10 solver iterations, Dbvt broadphase,
    ///               CCD at 10 m/s.  Suitable for desktop/server-class hardware.
    /// - `LowEnd`:    Optimised for constrained hardware (2-core 1.6 GHz, 512 MB RAM).
    ///               Uses SimdBruteForce broadphase, 6 solver iterations, immediate freeze,
    ///               and reduced/eliminated CCD for up to 3× simulation throughput.
    enum class Preset : std::uint8_t {
        Default,
        LowEnd
    };

    // ─── Lifecycle ──────────────────────────────────────────────────────────────────────────────
    PhysicsWorld() = default;

    /// @brief  Destroy the world, ensuring the thread pool is drained before
    ///         member objects are destroyed (prevents MinGW/std::jthread races).
    ~PhysicsWorld() {
        if (m_threadPool) {
            // Ensure all worker threads are idle before any member destructors run.
            m_threadPool->drain();
        }
    }

    /// @brief  Move constructor.  Explicitly defaulted because the user-defined
    ///         destructor above suppresses the implicit move constructor.
    PhysicsWorld(PhysicsWorld&&) = default;

    /// @brief  Construct with explicit configuration.
    explicit PhysicsWorld(const PhysicsWorldConfig& config) : m_config(config) {}

    /// @brief  Construct with a quick-preset.
    ///
    /// Applies the preset immediately.  Individual settings can still be overridden
    /// via `setConfig()` / `solverConfig()` after construction.
    explicit PhysicsWorld(Preset preset) noexcept {
        if (preset == Preset::LowEnd) {
            // ── Low-end preset (optimised for 2-core 1.6GHz, 512MB RAM) ─────────────────
            //     Uses only 1 worker thread to avoid oversubscription on 2-core CPUs.
            //     The calling thread participates in the work, giving 2 total threads
            //     for the solver pipeline.  More threads would cause excessive context
            //     switching and cache pollution on constrained hardware.
            m_config.linearDamping          = 0.3f;
            m_config.angularDamping         = 0.3f;
            m_config.sleepEnergyThreshold   = 0.01f;
            m_config.sleepTimeRequired      = 0.1f;
            m_config.enableParallelSolver   = true;
            m_config.numThreads             = 1;       // 1 worker + 1 caller = 2 total
            m_config.enableTaskGraphPipeline = true;
            m_config.ccdSpeedThreshold      = 20.0f;
            m_config.ccdMaxSubSteps         = 2;
            m_config.fixedTimestep          = 1.0f / 60.0f;

            m_solverConfig.numIterations           = 6;
            m_solverConfig.baumgarte               = 0.10f;
            m_solverConfig.maxPenetrationCorrection = 0.10f;
            m_solverConfig.enableWarmStart         = true;
            m_solverConfig.warmStartFactor         = 0.6f;
            m_solverConfig.restitutionThreshold    = 2.0f;
            m_solverConfig.earlyExitThreshold      = 0.005f;  // more aggressive early-exit

            m_broadphaseConfig.type = BroadphaseType::SimdBruteForce;

            m_useParallelSolver    = true;
            m_useTaskGraphPipeline = true;
        }
    }

    // ─── Configuration ──────────────────────────────────────────────────────────────────────────

    /// @brief  Access the current configuration.
    [[nodiscard]] const PhysicsWorldConfig& config() const noexcept { return m_config; }

    /// @brief  Update configuration (takes effect at the next fixed step).
    void setConfig(const PhysicsWorldConfig& cfg) noexcept {
        m_config = cfg;
        m_useParallelSolver    = cfg.enableParallelSolver;
        m_useTaskGraphPipeline = cfg.enableTaskGraphPipeline;
    }

    /// @brief  Access the solver parameters.
    [[nodiscard]]       SolverConfig& solverConfig()       noexcept { return m_solverConfig; }
    [[nodiscard]] const SolverConfig& solverConfig() const noexcept { return m_solverConfig; }

    /// @brief  Access the contact normal reduction configuration.
    [[nodiscard]]       ContactReductionConfig& contactReductionConfig()       noexcept { return m_contactReductionConfig; }
    [[nodiscard]] const ContactReductionConfig& contactReductionConfig() const noexcept { return m_contactReductionConfig; }

    /// @brief  Access the broadphase parameters.
    /// @brief  Read/write access to the broadphase configuration.
    [[nodiscard]]       BroadphaseConfig& broadphaseConfig()       noexcept { return m_broadphaseConfig; }
    [[nodiscard]] const BroadphaseConfig& broadphaseConfig() const noexcept { return m_broadphaseConfig; }

    // ─── Broadphase ───────────────────────────────────────────────────────────────────────────

    /// @brief  Set the broadphase algorithm (takes effect next step).
    ///         If switching **away** from Dbvt, the internal tree is cleared.
    void setBroadphaseType(BroadphaseType type) noexcept {
        if (m_broadphaseConfig.type == BroadphaseType::Dbvt && type != BroadphaseType::Dbvt) {
            m_dbvt.clear();
        }
        m_broadphaseConfig.type = type;
    }

    /// @brief  Convenience: enable the spatial‑hash broadphase with a given cell size.
    void enableSpatialHash(float cellSize = 4.0f) noexcept {
        m_broadphaseConfig.type           = BroadphaseType::SpatialHash;
        m_broadphaseConfig.spatialHashCellSize = cellSize;
    }

    /// @brief  Convenience: enable the SIMD-accelerated brute-force broadphase.
    ///         Uses SSE2 batch AABB overlap tests for 2–3× faster pair generation
    ///         compared to the scalar bounding-sphere variant.
    void enableSimdBruteForce() noexcept {
        m_broadphaseConfig.type = BroadphaseType::SimdBruteForce;
    }

    /// @brief  Convenience: enable the DBVT broadphase (dynamic BVH).
    ///         Tune the fat‑AABB margin via `dbvt()` access if needed.
    void enableDbvt() noexcept {
        m_broadphaseConfig.type = BroadphaseType::Dbvt;
    }

    /// @brief  Access the internal Dbvt (for advanced queries like `overlapQuery` / `sweepShape`).
    [[nodiscard]] const Dbvt& dbvt() const noexcept { return m_dbvt; }

    // ─── Shape overlap query ──────────────────────────────────────────────────────────────

    /// @brief  Find all active bodies whose shape(s) overlap the given query shape.
    ///
    /// If the active broadphase is Dbvt, the tree is synced before the query to ensure
    /// accurate results.  Otherwise a brute-force O(n) iteration over all active bodies
    /// is performed.
    ///
    /// @param shape  The query shape at its world-space transform.
    /// @param pos    World-space position.
    /// @param rot    World-space rotation.
    /// @param filter Optional collision filter (nullptr = no filter).  The filter's
    ///               `shouldCollide(a, b)` is called with the query body as both
    ///               arguments since there is no "other" body.
    /// @param[out] out  Receives handles of all bodies whose shape(s) overlap.
    void overlapShape(const Shape& shape, const Vec3f& pos, const Quaternionf& rot,
                      const ICollisionFilter* filter,
                      std::vector<RigidBodyHandle>& out) noexcept {
        out.clear();
        if (m_bodies.size() == 0) return;

        // Ensure the Dbvt is up to date if it is the active broadphase.
        if (m_broadphaseConfig.type == BroadphaseType::Dbvt) {
            DbvtConfig dbvtc;
            m_dbvt.sync(m_bodies, m_shapes, dbvtc);
        }

        AABB queryAABB = shape.getAABB(pos, rot);

        // Collect broadphase candidates.
        std::vector<RigidBodyHandle> candidates;
        if (m_broadphaseConfig.type == BroadphaseType::Dbvt) {
            m_dbvt.overlapQuery(queryAABB, candidates);
        } else {
            candidates.reserve(m_bodies.size());
            for (std::size_t i = 0; i < m_bodies.size(); ++i) {
                if (m_bodies.activeFlags[i] && m_bodies.shapeCount[i] > 0) {
                    candidates.push_back(static_cast<RigidBodyHandle>(i));
                }
            }
        }

        // Narrowphase test: iterate sub-shapes of each candidate body.
        for (RigidBodyHandle h : candidates) {
            if (h >= m_bodies.size()) continue;
            if (!m_bodies.activeFlags[h]) continue;

            // Collision filter (if provided).
            if (filter && !filter->shouldCollide(h, h)) continue;

            std::uint32_t start = m_bodies.shapeStart[h];
            std::uint32_t count = m_bodies.shapeCount[h];

            for (std::uint32_t s = 0; s < count; ++s) {
                ShapeHandle sh = m_bodies.flatShapeHandles[start + s];
                if (sh >= m_shapes.size()) continue;

                // World-space sub-shape transform.
                Vec3f localPos  = m_bodies.flatShapeLocalPositions[start + s];
                Quaternionf localRot = m_bodies.flatShapeLocalRotations[start + s];
                Vec3f worldPos  = m_bodies.positions[h]
                                  + m_bodies.rotations[h].rotate(localPos);
                Quaternionf worldRot = m_bodies.rotations[h] * localRot;

                // Perform narrowphase overlap test.
                ContactManifold manifold;
                if (collideShapes(shape, pos, rot,
                                  m_shapes[sh], worldPos, worldRot,
                                  0.0f, 0.0f, 0.0f, 0.0f, manifold)) {
                    out.push_back(h);
                    break; // One overlap per body is sufficient.
                }
            }
        }

        // Sort and deduplicate for deterministic output.
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    // ─── Shape sweep query ───────────────────────────────────────────────────────────────

    /// @brief  Result of a shape sweep (time-of-impact) query.
    struct SweepHit {
        float           t        = 1.0f;  ///< TOI in [0,1]; 1 = no hit.
        Vec3f           point    = Vec3f::zero();  ///< World-space contact position.
        Vec3f           normal   = Vec3f::zero();  ///< World-space contact normal.
        RigidBodyHandle body     = kInvalidHandle; ///< The body that was hit.
    };

    /// @brief  Sweep a shape along a linear path and return the first hit.
    ///
    /// Uses a robust binary-search sweep for all shape types.  The maximum sweep
    /// distance is `|toPos - fromPos|`.  Rotation interpolation uses spherical
    /// linear interpolation (slerp).
    ///
    /// If the active broadphase is Dbvt, the tree is synced and queried for
    /// candidate culling.  Otherwise a brute-force O(n) iteration is performed.
    ///
    /// @param shape  The query shape.
    /// @param fromPos  Start position.
    /// @param fromRot  Start rotation.
    /// @param toPos    End position.
    /// @param toRot    End rotation.
    /// @param filter   Optional collision filter.
    /// @param[out] hit Filled with hit details if a collision is found.
    /// @return True if any body was hit (hit.t < 1.0f).
    bool sweepShape(const Shape& shape,
                    const Vec3f& fromPos, const Quaternionf& fromRot,
                    const Vec3f& toPos, const Quaternionf& toRot,
                    const ICollisionFilter* filter,
                    SweepHit& hit) noexcept {

        hit = SweepHit{};

        Vec3f disp = toPos - fromPos;
        float totalLen = disp.length();

        // Compute the swept AABB (union of start and end AABBs).
        AABB sweptAABB = shape.getAABB(fromPos, fromRot)
                            .united(shape.getAABB(toPos, toRot));

        // Ensure the Dbvt is up to date if it is the active broadphase.
        if (m_broadphaseConfig.type == BroadphaseType::Dbvt) {
            DbvtConfig dbvtc;
            m_dbvt.sync(m_bodies, m_shapes, dbvtc);
        }

        // Collect broadphase candidates.
        std::vector<RigidBodyHandle> candidates;
        if (m_broadphaseConfig.type == BroadphaseType::Dbvt) {
            m_dbvt.overlapQuery(sweptAABB, candidates);
        } else {
            candidates.reserve(m_bodies.size());
            for (std::size_t i = 0; i < m_bodies.size(); ++i) {
                if (m_bodies.activeFlags[i] && m_bodies.shapeCount[i] > 0) {
                    candidates.push_back(static_cast<RigidBodyHandle>(i));
                }
            }
        }

        enum { kSweepIterations = 16 }; // ~1 mm precision for a 1 m sweep
        float bestT = 1.0f;

        // Test each candidate.
        for (RigidBodyHandle h : candidates) {
            if (h >= m_bodies.size()) continue;
            if (!m_bodies.activeFlags[h]) continue;

            // Collision filter (if provided).
            if (filter && !filter->shouldCollide(h, h)) continue;

            std::uint32_t start = m_bodies.shapeStart[h];
            std::uint32_t count = m_bodies.shapeCount[h];

            for (std::uint32_t s = 0; s < count; ++s) {
                ShapeHandle sh = m_bodies.flatShapeHandles[start + s];
                if (sh >= m_shapes.size()) continue;

                // World-space sub-shape transform.
                Vec3f localPos  = m_bodies.flatShapeLocalPositions[start + s];
                Quaternionf localRot = m_bodies.flatShapeLocalRotations[start + s];
                Vec3f worldPos  = m_bodies.positions[h]
                                  + m_bodies.rotations[h].rotate(localPos);
                Quaternionf worldRot = m_bodies.rotations[h] * localRot;

                const Shape& worldShape = m_shapes[sh];

                // ── Binary-search sweep ─────────────────────────────────────────────────

                // Check overlap at t = 0 first.
                {
                    ContactManifold manifold;
                    if (collideShapes(shape, fromPos, fromRot,
                                      worldShape, worldPos, worldRot,
                                      0.0f, 0.0f, 0.0f, 0.0f, manifold)) {
                        if (0.0f < bestT) {
                            bestT = 0.0f;
                            hit.t      = 0.0f;
                            hit.point  = manifold.points[0].position;
                            hit.normal = manifold.points[0].normal;
                            hit.body   = h;
                        }
                        break; // earliest possible hit for this body
                    }
                }

                if (totalLen < 1e-12f) continue; // no movement, already checked t=0

                // Binary search for TOI.
                float lo = 0.0f, hi = 1.0f;
                bool localHit = false;
                Vec3f hitNml(0.0f, 1.0f, 0.0f), hitPos(0.0f, 0.0f, 0.0f);

                for (int iter = 0; iter < kSweepIterations; ++iter) {
                    float mid = (lo + hi) * 0.5f;
                    Vec3f samplePos    = fromPos + disp * mid;
                    Quaternionf sampleRot = Quaternionf::slerp(fromRot, toRot, mid);

                    ContactManifold manifold;
                    if (collideShapes(shape, samplePos, sampleRot,
                                      worldShape, worldPos, worldRot,
                                      0.0f, 0.0f, 0.0f, 0.0f, manifold)) {
                        localHit = true;
                        hi       = mid;
                        hitNml   = manifold.points[0].normal;
                        hitPos   = manifold.points[0].position;
                    } else {
                        lo = mid;
                    }
                }

                if (localHit) {
                    float toi = (lo + hi) * 0.5f;
                    if (toi < bestT) {
                        bestT    = toi;
                        hit.t      = toi;
                        hit.point  = hitPos;
                        hit.normal = hitNml;
                        hit.body   = h;
                    }
                    break; // per-body: one sub-shape hit is sufficient
                }
            }
        }

        return bestT < 1.0f;
    }

    // ─── Parallel solver ─────────────────────────────────────────────────────────────────────

    /// @brief  Enable or disable the parallel island‑based solver.
    /// @param enable  When true, islands are solved concurrently.
    /// @param numThreads  Worker thread count (0 = hardware concurrency).
    void enableParallelSolver(bool enable, std::uint32_t numThreads = 0) noexcept {
        m_useParallelSolver = enable;
        m_config.enableParallelSolver = enable;
        m_config.numThreads = numThreads;
    }

    /// @brief  Query whether the parallel solver is active.
    [[nodiscard]] bool isParallelSolverEnabled() const noexcept { return m_useParallelSolver; }

    // ─── Task-graph pipeline ─────────────────────────────────────────────────────────────────

    /// @brief  Enable or disable the DAG-based task-graph pipeline.
    ///
    /// When enabled, `fixedStep()` builds a `TaskGraph` of the solver pipeline and executes
    /// it on the thread pool.  Independent stages (e.g., CCD + collision detection) run
    /// concurrently.  Requires `enableParallelSolver` for best results.
    ///
    /// @param enable  When true, the task-graph pipeline is used instead of the sequential one.
    void enableTaskGraphPipeline(bool enable) noexcept {
        m_useTaskGraphPipeline = enable;
        m_config.enableTaskGraphPipeline = enable;
    }

    /// @brief  Query whether the task-graph pipeline is active.
    [[nodiscard]] bool isTaskGraphPipelineEnabled() const noexcept { return m_useTaskGraphPipeline; }

    // ─── Task-graph profiling ───────────────────────────────────────────────────────────────────

    /// @brief  Enable or disable per-task profiling in the task-graph pipeline.
    ///
    /// When enabled, each task node records its wall-clock execution time.
    /// Profile data is accessible via `taskGraphProfileEvents()` after `step()`.
    ///
    /// @param enable  When true, task execution timing is collected.
    void enableTaskGraphProfiling(bool enable) noexcept {
        m_taskGraph.setProfilingEnabled(enable);
    }

    /// @brief  Query whether task-graph profiling is active.
    [[nodiscard]] bool isTaskGraphProfilingEnabled() const noexcept {
        return m_taskGraph.isProfilingEnabled();
    }

    /// @brief  Profile events from the most recent task-graph pipeline execution.
    ///
    /// Each event contains the task name, elapsed wall-clock time in milliseconds,
    /// and the thread ID that executed the task.
    ///
    /// @return  A reference to the internal vector of profile events (read-only).
    ///          Events are in completion order (not deterministic).
    [[nodiscard]] const std::vector<TaskGraph::ProfileEvent>& taskGraphProfileEvents() const noexcept {
        return m_taskGraph.profileEvents();
    }

    /// @brief  Compute per-task aggregate statistics from the current profile events.
    ///
    /// Returns a map of task name → {total time, call count, min/max/avg ms}.
    struct ProfileSummary {
        std::string name;      ///< Task name.
        double      totalMs;   ///< Total wall-clock time.
        double      minMs;     ///< Fastest execution.
        double      maxMs;     ///< Slowest execution.
        double      avgMs;     ///< Average execution.
        int         count;     ///< Number of calls.
    };

    /// @brief  Aggregate the current profile events into per-task statistics.
    [[nodiscard]] std::vector<ProfileSummary> taskGraphProfileSummary() const {
        std::unordered_map<std::string, std::pair<double, std::pair<double, double>>> agg;
        // ^ name → {total, {min, max}}
        for (const auto& ev : m_taskGraph.profileEvents()) {
            auto it = agg.find(ev.name);
            if (it == agg.end()) {
                agg[ev.name] = {ev.elapsedMs, {ev.elapsedMs, ev.elapsedMs}};
            } else {
                it->second.first += ev.elapsedMs;
                it->second.second.first  = std::min(it->second.second.first, ev.elapsedMs);
                it->second.second.second = std::max(it->second.second.second, ev.elapsedMs);
            }
        }
        std::vector<ProfileSummary> result;
        result.reserve(agg.size());
        for (const auto& [name, data] : agg) {
            int count = 0;
            for (const auto& ev : m_taskGraph.profileEvents()) {
                if (ev.name == name) ++count;
            }
            result.push_back({
                name,
                data.first,
                data.second.first,
                data.second.second,
                data.first / static_cast<double>(count),
                count
            });
        }
        // Sort by total time descending.
        std::sort(result.begin(), result.end(),
                  [](const ProfileSummary& a, const ProfileSummary& b) {
                      return a.totalMs > b.totalMs;
                  });
        return result;
    }

    // ─── Shape registry ─────────────────────────────────────────────────────────────────────────

    /// @brief  Register a shape.  Returns a handle that can be attached to rigid bodies.
    ShapeHandle createShape(const Shape& shape) {
        ShapeHandle h = static_cast<ShapeHandle>(m_shapes.size());
        m_shapes.push_back(shape);
        return h;
    }

    /// @brief  Attach (or replace) the shape on an existing body.
    ///
    /// Replaces all existing sub-shapes with a single shape at the body origin.
    void setBodyShape(RigidBodyHandle body, ShapeHandle shape) {
        if (body < m_bodies.size()) {
            // Append the new sub-shape to the flat arrays.
            std::uint32_t start = static_cast<std::uint32_t>(m_bodies.flatShapeHandles.size());
            m_bodies.flatShapeHandles.push_back(shape);
            m_bodies.flatShapeLocalPositions.emplace_back(Vec3f::zero());
            m_bodies.flatShapeLocalRotations.emplace_back(Quaternionf::identity());
            m_bodies.shapeStart[body] = start;
            m_bodies.shapeCount[body] = 1;
        }
    }

    /// @brief  Read-only access to the shape registry.
    [[nodiscard]] const std::vector<Shape>& shapes() const noexcept { return m_shapes; }

    // ─── Body management ────────────────────────────────────────────────────────────────────────

    /// @brief  Create a new rigid body from a descriptor.  Returns its handle.
    ///
    /// If `desc.useAutoInertia` is true and `desc.subShapes` is not empty, the inertia tensor
    /// is automatically computed from all sub-shapes using closed-form formulas combined via
    /// the **parallel-axis theorem**.  The user can disable this by setting
    /// `useAutoInertia = false` to supply a custom tensor.
    ///
    /// ## Inertia aggregation (parallel-axis theorem)
    ///
    /// For each sub-shape with mass `m_i` and local offset `r_i`:
    /// \f[
    ///   I_{\text{total}} = \sum_i \left( I_{\text{local},i} + m_i \cdot (|r_i|^2 \cdot I - r_i \otimes r_i) \right)
    /// \f]
    /// Only the diagonal is stored; off-diagonal terms are discarded.
    RigidBodyHandle createBody(const RigidBodyDesc& desc) {
    RigidBodyHandle h = m_bodies.emplace(desc);

    // Body added — mark index array dirty for next rebuild.
    m_activeIndicesDirty = true;

    // Ensure destructible tracking arrays match body count.
    if (h >= m_destructibleFlags.size()) {
        m_destructibleFlags.resize(h + 1, 0);
        m_destructibleData.resize(h + 1);
    }

        // Auto-compute inertia from sub-shapes when requested.
        if (desc.useAutoInertia && !desc.subShapes.empty()
            && desc.motionType == MotionType::Dynamic && desc.mass > 0.0f) {

            Vec3f totalInertia = Vec3f::zero();
            float massPerShape = desc.mass / static_cast<float>(desc.subShapes.size());

            for (const auto& ss : desc.subShapes) {
                if (ss.handle < m_shapes.size()) {
                    const Shape& shape = m_shapes[ss.handle];
                    Vec3f localInertia = shape.computeInertia(massPerShape);

                    // Parallel-axis contribution from local offset.
                    Vec3f r = ss.localPosition;
                    float m = massPerShape;
                    totalInertia.x += localInertia.x + m * (r.y * r.y + r.z * r.z);
                    totalInertia.y += localInertia.y + m * (r.x * r.x + r.z * r.z);
                    totalInertia.z += localInertia.z + m * (r.x * r.x + r.y * r.y);
                }
            }

            // Guard against zero inertia (all sub-shape handles invalid).
            float eps = 1e-12f;
            m_bodies.inverseInertiaDiag[h] = Vec3f(
                totalInertia.x > eps ? 1.0f / totalInertia.x : 0.0f,
                totalInertia.y > eps ? 1.0f / totalInertia.y : 0.0f,
                totalInertia.z > eps ? 1.0f / totalInertia.z : 0.0f);
        }

        return h;
    }

    /// @brief  Remove a body by handle.  Invalidates the handles of the last body (swap+pop).
    void removeBody(RigidBodyHandle handle) {
        // Body removed — mark index array dirty for next rebuild.
        m_activeIndicesDirty = true;
        m_bodies.remove(handle);
        // Keep destructible tracking in sync with the swap-pop.
        std::size_t last = m_destructibleFlags.size() - 1;
        if (handle < m_destructibleFlags.size()) {
            if (handle != last) {
                std::swap(m_destructibleFlags[handle], m_destructibleFlags[last]);
                std::swap(m_destructibleData[handle], m_destructibleData[last]);
            }
            if (!m_destructibleFlags.empty()) {
                m_destructibleFlags.pop_back();
                m_destructibleData.pop_back();
            }
        }
    }

    /// @brief  Remove all bodies and reset state.
    void clear() {
        m_activeIndicesDirty = true;
        m_bodies.clear();
        m_shapes.clear();
        m_manifolds.clear();
        m_pairs.clear();
        m_joints.clear();
        m_softBodies.clear();
        m_islandSolver.clear();
        m_dbvt.clear();
        m_destructibleFlags.clear();
        m_destructibleData.clear();
        m_accumulator = 0.0f;
        m_stepCount   = 0;
    }

    /// @brief  Direct read/write access to the SoA storage.
    [[nodiscard]]       RigidBodyStorage& bodies()       noexcept { return m_bodies; }
    [[nodiscard]] const RigidBodyStorage& bodies() const noexcept { return m_bodies; }

    /// @brief  Pre-allocate storage for `count` bodies to avoid reallocations during simulation.
    ///
    /// Call this BEFORE creating bodies to reserve all internal SoA arrays to the required
    /// capacity.  After this call, body creation via `createBody()` will not trigger any
    /// memory allocations until the count exceeds the reserved capacity.
    ///
    /// For constrained hardware (512 MB RAM, 2-core), use with the `LowEnd` preset:
    /// @code
    ///   PhysicsWorld world(PhysicsWorld::Preset::LowEnd);
    ///   world.setMaxBodies(5000);
    ///   // ... create bodies ...
    /// @endcode
    void setMaxBodies(std::size_t count) noexcept { m_bodies.reserve(count); }

    // ────────────────────────────────────────────────────────────────────────────────────────────────
    // Soft body API
    // ────────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Create a new soft body from a descriptor.
    ///
    /// The descriptor defines particle positions, masses, and simulation parameters.
    /// For complex topologies (cloth, volume, rope), use the factory methods on
    /// `SoftBody` and pass the result through `addSoftBody()`.
    ///
    /// @code
    ///   SoftBodyDesc desc;
    ///   desc.positions = {...};
    ///   desc.masses    = {...};
    ///   SoftBodyHandle h = world.createSoftBody(desc);
    /// @endcode
    ///
    /// @param desc  Soft body descriptor.
    /// @return Handle (index) to the newly created soft body.
    SoftBodyHandle createSoftBody(const SoftBodyDesc& desc) {
        SoftBody body;
        std::size_t n = desc.positions.size();
        body.positions.reserve(n);
        body.positionsPrev.reserve(n);
        body.inverseMasses.reserve(n);
        body.pinned.reserve(n);

        for (std::size_t i = 0; i < n; ++i) {
            Vec3f pos = i < desc.positions.size() ? desc.positions[i] : Vec3f::zero();
            float mass = i < desc.masses.size() ? desc.masses[i] : 1.0f;
            body.positions.emplace_back(pos);
            body.positionsPrev.emplace_back(pos);
            body.inverseMasses.emplace_back(mass > 0.0f ? 1.0f / mass : 0.0f);
            body.pinned.emplace_back(mass <= 0.0f ? 1 : 0);
        }

        body.gravity = desc.gravity;
        body.damping = desc.damping;
        body.friction = desc.friction;
        body.collisionMargin = desc.collisionMargin;
        body.constraintIterations = desc.constraintIterations;
        body.constraintStiffness = desc.constraintStiffness;
        body.isCloth = desc.isCloth;

        m_softBodies.push_back(std::move(body));
        return m_softBodies.size() - 1;
    }

    /// @brief  Add a pre-configured SoftBody to the world (e.g. from a factory method).
    ///
    /// @code
    ///   auto cloth = SoftBody::createCloth(10, 10, 0.5f, 1.0f);
    ///   cloth.pinParticle(0);
    ///   cloth.pinParticle(9);
    ///   SoftBodyHandle h = world.addSoftBody(std::move(cloth));
    /// @endcode
    ///
    /// @param body  The soft body to add (moved into the world).
    /// @return Handle (index) to the added soft body.
    SoftBodyHandle addSoftBody(SoftBody&& body) {
        m_softBodies.push_back(std::move(body));
        return m_softBodies.size() - 1;
    }

    /// @brief  Remove a soft body by handle.  Invalidates back handle (swap+pop).
    /// @param handle  Handle of the soft body to remove.
    void removeSoftBody(SoftBodyHandle handle) {
        if (handle < m_softBodies.size()) {
            if (handle != m_softBodies.size() - 1) {
                std::swap(m_softBodies[handle], m_softBodies.back());
            }
            m_softBodies.pop_back();
        }
    }

    /// @brief  Mutable access to a soft body by handle.
    /// @param handle  Soft body handle.
    /// @return Reference to the soft body.
    [[nodiscard]] SoftBody& softBody(SoftBodyHandle handle) noexcept {
        return m_softBodies[handle];
    }

    /// @brief  Const access to a soft body by handle.
    [[nodiscard]] const SoftBody& softBody(SoftBodyHandle handle) const noexcept {
        return m_softBodies[handle];
    }

    /// @brief  Number of soft bodies in the world.
    [[nodiscard]] std::size_t softBodyCount() const noexcept { return m_softBodies.size(); }

    /// @brief  Mutable access to all soft bodies (vector).
    [[nodiscard]] std::vector<SoftBody>& softBodies() noexcept { return m_softBodies; }

    /// @brief  Const access to all soft bodies.
    [[nodiscard]] const std::vector<SoftBody>& softBodies() const noexcept { return m_softBodies; }

    // ────────────────────────────────────────────────────────────────────────────────────────────────
    // Destructible body API
    // ────────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Mark a rigid body as destructible, pre-computing its Voronoi fragments.
    ///
    /// The body must have at least one ConvexMesh sub-shape.  The fracture pattern is
    /// generated at call time (not at break time) so there is no hitch when fracture occurs.
    ///
    /// @param handle  Handle of an existing rigid body.
    /// @param config  Fracture configuration (threshold, fragment count, etc.).
    ///
    /// @code
    ///   RigidBodyHandle h = world.createBody(desc);
    ///   DestructibleConfig cfg;
    ///   cfg.breakImpulseThreshold = 200.0f;
    ///   cfg.numFragments = 8;
    ///   world.makeDestructible(h, cfg);
    /// @endcode
    void makeDestructible(RigidBodyHandle handle, const DestructibleConfig& config = {}) {
        // Ensure tracking arrays are large enough.
        if (handle >= m_destructibleFlags.size()) {
            m_destructibleFlags.resize(handle + 1, 0);
            m_destructibleData.resize(handle + 1);
        }

        if (m_destructibleFlags[handle]) return;  // Already destructible.

        // Find the first ConvexMesh sub-shape on this body.
        std::uint32_t start = m_bodies.shapeStart[handle];
        std::uint32_t count = m_bodies.shapeCount[handle];

        for (std::uint32_t s = 0; s < count; ++s) {
            ShapeHandle sh = m_bodies.flatShapeHandles[start + s];
            if (sh >= m_shapes.size()) continue;

            if (m_shapes[sh].type == ShapeType::ConvexMesh) {
                const ConvexMesh& mesh = m_shapes[sh].convexMesh;

                // Generate seeds and pre-compute fragments.
                float totalMass = (m_bodies.inverseMasses[handle] > 0.0f)
                    ? 1.0f / m_bodies.inverseMasses[handle]
                    : 1.0f;

                auto seeds = VoronoiFracture::generateSeeds(mesh, config.numFragments);
                auto fragments = VoronoiFracture::fragment(mesh, seeds, totalMass);

                m_destructibleFlags[handle] = 1;
                m_destructibleData[handle].config = config;
                m_destructibleData[handle].fragments = std::move(fragments);
                m_destructibleData[handle].isBroken = false;
                return;
            }
        }

        // No ConvexMesh shape found — destructible not enabled.
    }

    /// @brief  Remove destructible status from a body (no longer breakable).
    void removeDestructible(RigidBodyHandle handle) noexcept {
        if (handle < m_destructibleFlags.size()) {
            m_destructibleFlags[handle] = 0;
            m_destructibleData[handle] = DestructibleData{};
        }
    }

    /// @brief  Check whether a body is destructible.
    [[nodiscard]] bool isDestructible(RigidBodyHandle handle) const noexcept {
        return handle < m_destructibleFlags.size() && m_destructibleFlags[handle] != 0;
    }

    /// @brief  Access the destructible data for a body (const).
    [[nodiscard]] const DestructibleData& destructibleData(RigidBodyHandle handle) const noexcept {
        static const DestructibleData kEmpty{};
        if (handle < m_destructibleData.size())
            return m_destructibleData[handle];
        return kEmpty;
    }

    // ─── Collision filter ─────────────────────────────────────────────────────────────────────

    /// @brief  Set a collision filter callback.  Pass nullptr to remove.
    ///         The filter is called once per broad-phase pair before narrowphase.
    void setCollisionFilter(ICollisionFilter* filter) noexcept { m_collisionFilter = filter; }

    /// @brief  Access the current collision filter (may be nullptr).
    [[nodiscard]] const ICollisionFilter* collisionFilter() const noexcept { return m_collisionFilter; }

        /// @brief  Access the current frame's contact manifolds (read-only, for debug drawing).
        [[nodiscard]] const std::vector<ContactManifold>& manifolds() const noexcept { return m_manifolds; }

        // ─── Joint management ───────────────────────────────────────────────────────────────────────

        /// @brief  Create a joint from a descriptor.
        ///
        /// The anchors and axes are converted from world space to each body's local space
        /// at creation time, so the joint remains valid as bodies rotate.
        ///
        /// When `subShapeA` / `subShapeB` are set to a valid sub-shape index on the
        /// corresponding body, the anchor is computed relative to that sub-shape's local
        /// offset and rotation, enabling constraint attachment directly at compound-shape
        /// attachment points.
        JointHandle createJoint(const JointDesc& desc) {
            JointStorage js;
            js.type = desc.type;
            js.bodyA = desc.bodyA;
            js.bodyB = desc.bodyB;
            js.subShapeA = desc.subShapeA;
            js.subShapeB = desc.subShapeB;

            // Helper to compute the effective anchor position for a body, optionally
            // offset by a sub-shape's local transform.
            auto computeLocalAnchor = [&](RigidBodyHandle h, const Vec3f& worldAnchor,
                                           int subShape) -> Vec3f {
                if (subShape >= 0 && m_bodies.shapeCount[h] > 0
                    && static_cast<std::uint32_t>(subShape) < m_bodies.shapeCount[h]) {
                    // The anchor is relative to the sub-shape's world-space transform.
                    std::uint32_t idx = m_bodies.shapeStart[h] + static_cast<std::uint32_t>(subShape);
                    Vec3f subLocalPos = m_bodies.flatShapeLocalPositions[idx];
                    Quaternionf subLocalRot = m_bodies.flatShapeLocalRotations[idx];
                    // Sub-shape world transform.
                    Vec3f subWorldPos = m_bodies.positions[h]
                                       + m_bodies.rotations[h].rotate(subLocalPos);
                    Quaternionf subWorldRot = m_bodies.rotations[h] * subLocalRot;
                    // Convert world anchor to body-local space (accounting for sub-shape offset).
                    // The anchor was specified in world space; we want it in body-local space.
                    // First, express it relative to the sub-shape origin, then transform to
                    // body-local via the sub-shape's inverse rotation.
                    Vec3f relToSub = worldAnchor - subWorldPos;
                    Vec3f localRel = subWorldRot.rotateInverse(relToSub);
                    // Now convert from sub-shape-local to body-local by applying the
                    // sub-shape's local rotation and position.
                    return subLocalRot.rotate(localRel) + subLocalPos;
                }
                // Default: standard body-local space conversion.
                return m_bodies.rotations[h].rotateInverse(
                    worldAnchor - m_bodies.positions[h]);
            };

            // Convert world-space anchor to body-local space (accounting for sub-shape).
            js.localAnchorA = computeLocalAnchor(desc.bodyA, desc.anchorA, desc.subShapeA);
            js.localAnchorB = computeLocalAnchor(desc.bodyB, desc.anchorB, desc.subShapeB);

            // Convert world-space axes to local space.
            js.localAxisA = m_bodies.rotations[desc.bodyA].rotateInverse(desc.axisA).normalized();
            js.localAxisB = m_bodies.rotations[desc.bodyB].rotateInverse(desc.axisB).normalized();

            js.enableLimits = desc.enableLimits;
            js.limitMin     = desc.limitMin;
            js.limitMax     = desc.limitMax;
            js.enableMotor          = desc.enableMotor;
            js.motorTargetVelocity  = desc.motorTargetVelocity;
            js.motorMaxForce        = desc.motorMaxForce;
            js.springStiffness = desc.springStiffness;
            js.springDamping   = desc.springDamping;
            js.restLength      = desc.restLength;

            js.breakThreshold = desc.breakThreshold;
            js.broken         = false;

            // Clear warm-start cache.
            for (int i = 0; i < 6; ++i) js.impulseCache[i] = 0.0f;

            JointHandle h = static_cast<JointHandle>(m_joints.size());
            m_joints.push_back(js);
            return h;
        }

        /// @brief  Remove a joint by handle (swap-pop).
        void removeJoint(JointHandle handle) {
            if (handle < m_joints.size()) {
                if (handle != m_joints.size() - 1) {
                    m_joints[handle] = std::move(m_joints.back());
                }
                m_joints.pop_back();
            }
        }

        /// @brief  Access the joint array (read-only).
        [[nodiscard]] const std::vector<JointStorage>& joints() const noexcept { return m_joints; }

        /// @brief  Access the joint array (read/write).
        [[nodiscard]] std::vector<JointStorage>& joints() noexcept { return m_joints; }

    // ─── Simulation ─────────────────────────────────────────────────────────────────────────────

    /// @brief  Advance the simulation by `deltaTime` seconds.
    ///
    /// Internally this accumulates time and runs one or more fixed-timestep sub-steps.
    /// The number of sub-steps is deterministic given a fixed `deltaTime`.
    void step(float deltaTime) {
        m_accumulator += deltaTime;
        if (m_accumulator > m_config.maxAccumulator) {
            m_accumulator = m_config.maxAccumulator;
        }

        while (m_accumulator >= m_config.fixedTimestep) {
            fixedStep(m_config.fixedTimestep);
            m_accumulator -= m_config.fixedTimestep;
        }

        // TODO: Optional interpolation alpha = m_accumulator / m_config.fixedTimestep
    }

    // ─── Ray casting ────────────────────────────────────────────────────────────────────────────

    /// @brief  Cast a ray against all active bodies and return the closest hit.
    ///
    /// Iterates all bodies with valid sub-shapes, transforms the ray into each sub-shape's
    /// local space, tests the primitive, and keeps the closest hit (smallest t >= 0).
    ///
    /// @param ray      The ray in world space.
    /// @param maxDist  Maximum ray distance (t parameter).  Hits beyond this are ignored.
    /// @param[out] hit Filled with the closest hit information if any body was struck.
    /// @return True if any body was hit within maxDist.
    ///
    /// @note  This is a brute-force O(n) test over all active bodies.  For scenes with
    ///        many bodies, use a spatial acceleration structure (e.g. BVH) for ray casting.
    bool rayCast(const Ray& ray, float maxDist, RayHit& hit) const noexcept {
        bool anyHit = false;
        hit.t = maxDist;

        Vec3f dir = ray.direction;
        float dirLen = dir.length();
        if (dirLen < 1e-12f) return false;
        // Normalize direction for consistent t-parameterization.
        Vec3f dirN = dir / dirLen;
        float maxT = maxDist / dirLen;  // convert maxDist to t-parameter along non-normalized ray

        for (std::size_t i = 0; i < m_bodies.size(); ++i) {
            if (!m_bodies.activeFlags[i]) continue;

            std::uint32_t start = m_bodies.shapeStart[i];
            std::uint32_t count = m_bodies.shapeCount[i];
            if (count == 0) continue;

            for (std::uint32_t s = 0; s < count; ++s) {
                ShapeHandle sh = m_bodies.flatShapeHandles[start + s];
                if (sh >= m_shapes.size()) continue;

                // World-space sub-shape transform.
                Vec3f localPos = m_bodies.flatShapeLocalPositions[start + s];
                Quaternionf localRot = m_bodies.flatShapeLocalRotations[start + s];
                Vec3f worldPos = m_bodies.positions[i]
                                 + m_bodies.rotations[i].rotate(localPos);
                Quaternionf worldRot = m_bodies.rotations[i] * localRot;

                const Shape& shape = m_shapes[sh];
                float t;
                Vec3f normal;

                if (rayCastShape(shape, worldPos, worldRot,
                                 ray.origin, dirN,
                                 maxT, t, normal)) {
                    if (t < hit.t && t >= 0.0f) {
                        hit.t      = t;
                        hit.point  = ray.origin + dirN * t;
                        hit.normal = normal;
                        hit.body   = static_cast<RigidBodyHandle>(i);
                        anyHit = true;
                    }
                }
            }
        }

        // Convert t back to world-space distance.
        if (anyHit) hit.t *= dirLen;
        return anyHit;
    }

    // ─── Debug visualization ──────────────────────────────────────────────────────────────────

    /// @brief  Draw the current physics state using the provided debug renderer.
    ///
    /// Renders all active bodies as wireframe shapes (box, sphere, capsule, plane grid),
    /// contact points as small crosses with normal arrows, and joint anchor points.
    /// Colours:
    ///   - Dynamic bodies:  blue
    ///   - Static bodies:   grey
    ///   - Kinematic:       cyan
    ///   - Contact points:  red cross + green normal arrow
    ///   - Joint anchors:   yellow cross
    ///
    /// @param renderer  Debug renderer implementation (must outlive the call).
    /// @param opts      Bitfield of DebugDrawOption flags.
    void debugDraw(IDebugRenderer* renderer, std::uint32_t opts = 0xFFFF) const noexcept {
        if (!renderer) return;
        renderer->beginFrame();

                constexpr std::uint32_t kDrawBodies   = 1 << 0;
        constexpr std::uint32_t kDrawContacts = 1 << 1;
        constexpr std::uint32_t kDrawJoints   = 1 << 2;
        // AABB drawing (kDrawAABBs) reserved for future use

        // ── Bodies ────────────────────────────────────────────────────────────────────────
        if (opts & kDrawBodies) {
            for (std::size_t i = 0; i < m_bodies.size(); ++i) {
                if (!m_bodies.activeFlags[i]) continue;

                Color bodyColor = Color::blue();
                if (m_bodies.motionTypes[i] == MotionType::Static)
                    bodyColor = Color::grey();
                else if (m_bodies.motionTypes[i] == MotionType::Kinematic)
                    bodyColor = Color::cyan();

                // Draw each sub-shape.
                std::uint32_t start = m_bodies.shapeStart[i];
                std::uint32_t count = m_bodies.shapeCount[i];
                for (std::uint32_t s = 0; s < count; ++s) {
                    ShapeHandle sh = m_bodies.flatShapeHandles[start + s];
                    if (sh >= m_shapes.size()) continue;

                    Vec3f localPos = m_bodies.flatShapeLocalPositions[start + s];
                    Quaternionf localRot = m_bodies.flatShapeLocalRotations[start + s];
                    Vec3f worldPos = m_bodies.positions[i]
                                     + m_bodies.rotations[i].rotate(localPos);
                    Quaternionf worldRot = m_bodies.rotations[i] * localRot;

                    const Shape& shape = m_shapes[sh];
                    switch (shape.type) {
                    case ShapeType::Box:
                        debug_detail::drawBox(*renderer, worldPos, worldRot,
                                               shape.box.halfExtents, bodyColor);
                        break;
                    case ShapeType::Sphere:
                        debug_detail::drawSphere(*renderer, worldPos,
                                                  shape.sphere.radius, bodyColor);
                        break;
                    case ShapeType::Capsule:
                        debug_detail::drawCapsule(*renderer, worldPos, worldRot,
                                                   shape.capsule.radius,
                                                   shape.capsule.halfHeight, bodyColor);
                        break;
                    case ShapeType::Plane:
                        debug_detail::drawPlane(*renderer, worldPos, worldRot,
                                                 5.0f, bodyColor);
                        break;
                    case ShapeType::ConvexMesh:
                        debug_detail::drawBox(*renderer, worldPos, worldRot,
                                               shape.convexMesh.halfExtents, bodyColor);
                        break;
                    case ShapeType::TriangleMesh:
                        // TriangleMesh: draw AABB as a wireframe bounding box.
                        if (shape.triangleMesh.bvh) {
                            debug_detail::drawBox(*renderer,
                                worldPos + worldRot.rotate(shape.triangleMesh.center),
                                worldRot, shape.triangleMesh.halfExtents, bodyColor);
                        }
                        break;
                    }
                }

                // Velocity arrow.
                float vLen = m_bodies.linearVelocities[i].length();
                if (vLen > 0.01f) {
                    renderer->drawArrow(m_bodies.positions[i],
                                        m_bodies.linearVelocities[i] / vLen,
                                        std::min(vLen * 0.5f, 2.0f),
                                        Color::green());
                }
            }
        }

        // ── Contact points ───────────────────────────────────────────────────────────────
        if (opts & kDrawContacts) {
            for (const auto& manifold : m_manifolds) {
                for (int cp = 0; cp < manifold.pointCount; ++cp) {
                    const auto& pt = manifold.points[cp];
                    // Cross at contact position.
                    float sz = 0.05f;
                    renderer->drawLine(pt.position - Vec3f(sz, 0, 0),
                                       pt.position + Vec3f(sz, 0, 0), Color::red());
                    renderer->drawLine(pt.position - Vec3f(0, sz, 0),
                                       pt.position + Vec3f(0, sz, 0), Color::red());
                    renderer->drawLine(pt.position - Vec3f(0, 0, sz),
                                       pt.position + Vec3f(0, 0, sz), Color::red());
                    // Normal arrow.
                    renderer->drawArrow(pt.position, pt.normal, 0.2f, Color::green());
                }
            }
        }

        // ── Joint anchors ────────────────────────────────────────────────────────────────
        if (opts & kDrawJoints) {
            for (const auto& joint : m_joints) {
                // Compute world-space anchor positions.
                RigidBodyHandle hA = joint.bodyA;
                RigidBodyHandle hB = joint.bodyB;

                Vec3f worldAnchorA = m_bodies.positions[hA]
                                   + m_bodies.rotations[hA].rotate(joint.localAnchorA);
                Vec3f worldAnchorB = m_bodies.positions[hB]
                                   + m_bodies.rotations[hB].rotate(joint.localAnchorB);

                // Anchor A (yellow cross).
                float sz = 0.08f;
                renderer->drawLine(worldAnchorA - Vec3f(sz, 0, 0),
                                   worldAnchorA + Vec3f(sz, 0, 0), Color::yellow());
                renderer->drawLine(worldAnchorA - Vec3f(0, sz, 0),
                                   worldAnchorA + Vec3f(0, sz, 0), Color::yellow());
                renderer->drawLine(worldAnchorA - Vec3f(0, 0, sz),
                                   worldAnchorA + Vec3f(0, 0, sz), Color::yellow());

                // Connect anchors.
                renderer->drawLine(worldAnchorA, worldAnchorB, Color::yellow());

                // Hinge axis (if applicable).
                if (joint.type == JointType::Hinge) {
                    Vec3f axisWorld = m_bodies.rotations[hA].rotate(joint.localAxisA).normalized();
                    renderer->drawArrow(worldAnchorA, axisWorld, 0.3f, Color::magenta());
                }
            }
        }

        renderer->endFrame();
    }

    /// @brief  Flags for debugDraw() — OR together desired visualisations.
    enum DebugDrawOption : std::uint32_t {
        DrawBodies   = 1 << 0, ///< Wireframe body shapes
        DrawContacts = 1 << 1, ///< Contact points and normals
        DrawJoints   = 1 << 2, ///< Joint anchors and axes
        DrawAABBs    = 1 << 3  ///< Bounding boxes (not yet implemented)
    };

    // ─── Hooks ──────────────────────────────────────────────────────────────────────────────────

    void addHook(ISolverHook* hook) {
        if (hook) m_hooks.push_back(hook);
    }

    void removeHook(ISolverHook* hook) {
        auto it = std::find(m_hooks.begin(), m_hooks.end(), hook);
        if (it != m_hooks.end()) m_hooks.erase(it);
    }

    // ─── Statistics ─────────────────────────────────────────────────────────────────────────────

    [[nodiscard]] std::uint64_t stepCount() const noexcept { return m_stepCount; }
    [[nodiscard]] float        accumulator() const noexcept { return m_accumulator; }

private:
    // ─── Internal pipeline ──────────────────────────────────────────────────────────────────────

    void fixedStep(float dt) {
        if (m_useTaskGraphPipeline) {
            fixedStepTaskGraph(dt);
            return;
        }

        // Rebuild active-dynamic index array for this frame's hot loops.
        rebuildActiveDynamicIndices();

        // Phase 0 – soft body simulation (PBD step before rigid bodies)
        if (!m_softBodies.empty()) {
            stepSoftBodies(dt);
        }

        // Phase 1 – forces
        integrateForces(dt);

        // Phase 2 – semi-implicit velocity integration
        integrateVelocities(dt);

        // Phase 2b – Continuous Collision Detection (CCD)
        runCCD(dt);

        // Phase 3 – collision detection
        detectCollisions();

        // Phase 3b – coplanar contact normal reduction
        // Merges nearly-identical normals from adjacent triangles into a single plane
        // normal, eliminating the micro-bouncing/jitter that occurs when the solver sees
        // slightly-different normals on flat triangulated surfaces.
        reduceContactNormals(m_manifolds, m_contactReductionConfig);

        // Phase 4 – pre-solve hooks
        for (auto* hook : m_hooks) hook->onPreSolve(dt, m_bodies);

        // Phase 5 – constraint solving (contacts + joints)
        if (m_useParallelSolver && !m_manifolds.empty()) {
            m_islandSolver.build(m_manifolds, m_joints, m_bodies.size());
            if (m_islandSolver.hasIslands() && m_islandSolver.islands().size() > 1) {
                m_islandSolver.solveParallel(dt, m_manifolds, m_bodies,
                                             m_solverConfig, ensureThreadPool());
            } else {
                m_solver.solve(dt, m_manifolds, m_bodies, m_solverConfig);
            }
        } else {
            m_solver.solve(dt, m_manifolds, m_bodies, m_solverConfig);
        }

        // Joint constraints (always single-threaded; run after contacts).
        if (!m_joints.empty()) {
            m_jointSolver.solve(dt, m_joints, m_bodies, m_solverConfig);
            // Check for breakable joints (those exceeding their impulse threshold).
            checkBreakableJoints();
        }

        // Phase 5b – check destructible body fractures (impulse threshold exceeded).
        checkFractures();

        // Phase 6 – post-solve hooks
        for (auto* hook : m_hooks) hook->onPostSolve(dt, m_bodies);

        // Phase 7 – position integration
        integratePositions(dt);

        // Phase 8 – sleep management
        updateSleep(dt);

        // Rebuild active-dynamic index array for next frame's hot loops.
        rebuildActiveDynamicIndices();

        ++m_stepCount;
    }

    // ─── Task-graph pipeline ────────────────────────────────────────────────────────────────────
    // ─── Task-graph pipeline ────────────────────────────────────────────────────
    //
    //  Builds a DAG of the fixed-step pipeline stages and executes them on the thread pool.
    //  Independent stages run concurrently, enabling pipeline parallelism without global
    //  barriers between every phase.
    //
    //  Pipeline architecture:
    //    1. Forces → Velocities
    //    2. Broadphase
    //    3. PreSolve                   (user hooks, after broadphase)
    //    4. FormIslands + AssignPairs  (union-find over broadphase pairs + joints;
    //       every body is assigned to exactly one island)
    //    5. Per-island sub-DAGs:       (each island runs independently)
    //       [CCD → Narrowphase → Reduce → Solve]
    //       - CCD: swept collision for fast-moving bodies in this island
    //       - Narrowphase: SAT/GJK-EPA on the island's assigned broadphase pairs
    //       - Contact reduction merges coplanar normals
    //       - Sequential-impulse solver resolves constraints
    //       Islands with no CCD-eligible bodies skip the CCD task.
    //       Islands with no broadphase pairs skip narrowphase/reduce/solve.
    //    6. JoinBarrier                (wait for all islands)
    //    7. MergeManifolds             (per-island → m_manifolds)
    //    8. Joint + Fracture           (joint solving, breakable constraints, impulse check)
    //    9. PostSolve                  (user hooks)
    //   10. IntegratePositions + Sleep
    //
    //  This structure enables true island-level pipeline parallelism: island A's CCD +
    //  narrowphase + solve runs independently of island B's, allowing work to overlap
    //  across islands.  CCD is also parallelised per-island rather than as a global pass.
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void fixedStepTaskGraph(float dt) {
        TaskGraph& graph = m_taskGraph;
        graph.clear();

        // Rebuild active-dynamic index array for this frame's hot loops.
        rebuildActiveDynamicIndices();

        // ── Stage 0 – Soft body simulation (parallel to all other stages via early launch) ─
        if (!m_softBodies.empty()) {
            graph.add("SoftBodies", {}, [this, dt]() {
                stepSoftBodies(dt);
            });
        }

        // ── Stage 1 – Forces ───────────────────────────────────────────────────────────────
        auto stageForces = graph.add("IntegrateForces", {}, [this, dt]() {
            integrateForces(dt);
        });

        // ── Stage 2 – Velocity integration ─────────────────────────────────────────────────
        auto stageVelocities = graph.add("IntegrateVelocities", {&stageForces, 1}, [this, dt]() {
            integrateVelocities(dt);
        });

        // ── Stage 3 – Broadphase (depends on velocities) ───────────────────────────────────
        //     Generates all body-body candidate pairs.  Uses estimated positions from
        //     velocity integration (pre-CCD).  This is the same tradeoff as the original
        //     parallel [CCD || Broadphase] design — CCD-corrected positions are consumed
        //     by the per-island narrowphase, not by broadphase.
        auto stageBroadphase = graph.add("Broadphase", {&stageVelocities, 1}, [this]() {
            runBroadphase();
        });

        // ── Stage 4 – Pre-solve hooks (depends on broadphase) ──────────────────────────────
        //     Runs before island formation so hooks can initialise per-body state.
        auto stagePreSolve = graph.add("PreSolve", {&stageBroadphase, 1}, [this, dt]() {
            for (auto* hook : m_hooks) hook->onPreSolve(dt, m_bodies);
        });

        // ── Stage 5 – Island formation + pair assignment ───────────────────────────────────
        //     Partitions the body graph into independent islands using broadphase pairs
        //     and joints.  Every body is assigned to exactly one island (isolated bodies
        //     become size-1 islands).  Then assigns each broadphase pair to its island so
        //     that per-island narrowphase tasks have disjoint work sets.
        auto stageFormIslands = graph.add("FormIslands", {&stagePreSolve, 1}, [this]() {
            m_islandSolver.buildFromPairs(m_pairs, m_joints, m_bodies.size());
            assignPairsToIslands();
        });

        // ── Stage 6 – Per-island sub-DAGs: [CCD → Narrowphase → Reduce → Solve] ───────────
        //     Each island runs an independent pipeline. CCD (swept collision for fast-moving
        //     bodies) runs first to correct positions, then narrowphase detects contacts,
        //     contact reduction merges coplanar normals, and the solver resolves constraints.
        //
        //     Islands with no CCD-eligible bodies skip the CCD task.
        //     Islands with no broadphase pairs skip narrowphase/reduce/solve entirely.
        //     If there's only one island (or parallel solver is disabled), we fall back
        //     to a single global task that does everything.
        // ── Island-count threshold ──────────────────────────────────────────────────────
        // When there are too many tiny islands, the per-island task-creation overhead
        // (CCD + Narrowphase + Reduce + Solve tasks per island × N islands) dominates.
        // Falling back to the single global pipeline avoids this overhead entirely.
        // 200 islands × 4 tasks = 800 task nodes per frame — the DAG building and
        // dispatch cost is already significant at this point. For ChaosDensityTest's
        // 5000 Voronoi fragments, post-explosion we can see 2000+ islands.
        // Reduced to 32 for 2-core systems where per-island task overhead exceeds
        // parallelisation benefit for more than ~32 small islands.
        static constexpr std::size_t kMaxIslandsForPerIsland = 32;

        std::vector<TaskId> islandFinalIds;
        bool usePerIsland = (m_useParallelSolver && m_islandSolver.hasIslands()
                             && m_islandSolver.islands().size() > 1
                             && m_islandSolver.islands().size() <= kMaxIslandsForPerIsland);

        if (usePerIsland) {
            std::size_t numIslands = m_islandSolver.islands().size();
            islandFinalIds.reserve(numIslands);

            for (std::size_t i = 0; i < numIslands; ++i) {
                const auto& island = m_islandSolver.islands()[i];
                bool hasPairs = (i < m_islandPairIndices.size()
                                 && !m_islandPairIndices[i].empty());

                // Determine if this island has any CCD-eligible bodies.
                bool hasCCD = false;
                float speedThresh = m_config.ccdSpeedThreshold;
                for (auto h : island.bodies) {
                    if (m_bodies.activeFlags[h]
                        && m_bodies.motionTypes[h] == MotionType::Dynamic
                        && static_cast<CcdMode>(m_bodies.ccdModes[h]) != CcdMode::None
                        && m_bodies.linearVelocities[h].length() >= speedThresh
                        && m_bodies.shapeCount[h] > 0) {
                        hasCCD = true;
                        break;
                    }
                }

                if (!hasCCD && !hasPairs) continue; // nothing to do for this island

                char bufC[64] = {}, bufN[64] = {}, bufR[64] = {}, bufS[64] = {};
                if (hasCCD)    std::snprintf(bufC, sizeof(bufC), "CCD_%zu", i);
                if (hasPairs) {
                    std::snprintf(bufN, sizeof(bufN), "Narrowphase_%zu", i);
                    std::snprintf(bufR, sizeof(bufR), "Reduce_%zu", i);
                    std::snprintf(bufS, sizeof(bufS), "Solve_%zu", i);
                }

                // Build the per-island body list for CCD (avoids copying in the lambda).
                // We copy the body handles into a small vector for the CCD lambda.
                std::vector<RigidBodyHandle> ccdBodies;
                if (hasCCD) {
                    ccdBodies = island.bodies; // copy — needed by lambda capture
                }

                // ── CCD task ──────────────────────────────────────────────────────────
                TaskId stageCCD = kInvalidTaskId;
                if (hasCCD) {
                    stageCCD = graph.add(std::string_view(bufC), {&stageFormIslands, 1},
                        [this, dt, ccdBodies = std::move(ccdBodies)]() {
                            runCCDForBodies(ccdBodies, dt);
                        });
                }

                if (!hasPairs) {
                    // Island with CCD bodies but no pairs — final task is the CCD task.
                    islandFinalIds.push_back(stageCCD);
                    continue;
                }

                // ── Narrowphase task (depends on CCD if present, else FormIslands) ────
                TaskId narrowDep = (hasCCD) ? stageCCD : stageFormIslands;
                auto stageNarrow = graph.add(std::string_view(bufN), {&narrowDep, 1},
                    [this, dt, i]() {
                        auto pairs = getIslandPairs(i);
                        runNarrowphase(pairs, m_islandManifolds[i]);
                    });

                // ── Contact reduction task ────────────────────────────────────────────
                TaskId reduceDeps[] = {stageNarrow};
                auto stageReduce = graph.add(std::string_view(bufR), reduceDeps,
                    [this, i]() {
                        reduceContactNormals(m_islandManifolds[i], m_contactReductionConfig);
                    });

                // ── Solve task ────────────────────────────────────────────────────────
                TaskId solveDeps[] = {stageReduce};
                auto stageSolve = graph.add(std::string_view(bufS), solveDeps,
                    [this, dt, i]() {
                        solveManifolds(dt, m_islandManifolds[i], m_bodies, m_solverConfig);
                    });

                islandFinalIds.push_back(stageSolve);
            }
        }

        if (!usePerIsland || islandFinalIds.empty()) {
            // ── Parallel fallback ──────────────────────────────────────────────────
            // Split CCD bodies and broadphase pairs across the thread pool workers
            // for better CPU utilization even with a single large island.
            std::size_t numWorkers = ensureThreadPool().threadCount();
            // Only parallelize for large enough workloads — the chunk overhead
            // (8-16 extra DAG nodes + thread synchronization) is not worth it for
            // small scenes like the DeepStackTest (200 bodies).
            bool canParallel = (numWorkers > 1
                                && m_activeDynamicIndices.size() >= 256
                                && m_pairs.size() >= 512);

            if (!canParallel) {
                // Sequential path: everything in one task (small scenes).
                auto stageGlobal = graph.add("GlobalCollideAndSolve", {&stageFormIslands, 1},
                    [this, dt]() {
                        runCCD(dt);
                        runNarrowphase(m_pairs, m_manifolds);
                        reduceContactNormals(m_manifolds, m_contactReductionConfig);
                        m_solver.solve(dt, m_manifolds, m_bodies, m_solverConfig);
                    });
                islandFinalIds.push_back(stageGlobal);
            } else {
                // ── Parallel CCD: split active-dynamic bodies into chunks ─────────
                // Cap chunks at 4 to limit DAG node overhead.
                std::size_t numCCDChunks = std::min(numWorkers,
                    (m_activeDynamicIndices.size() + 127) / 128);
                numCCDChunks = std::min(numCCDChunks, std::size_t{4});
                std::size_t bodiesPerChunk = (m_activeDynamicIndices.size()
                                              + numCCDChunks - 1) / numCCDChunks;

                std::vector<TaskId> ccdTaskIds;
                ccdTaskIds.reserve(numCCDChunks);
                for (std::size_t ci = 0; ci < numCCDChunks; ++ci) {
                    std::size_t start = ci * bodiesPerChunk;
                    std::size_t end = std::min(start + bodiesPerChunk,
                                               m_activeDynamicIndices.size());
                    if (start >= end) break;

                    std::vector<RigidBodyHandle> chunkBodies;
                    chunkBodies.reserve(end - start);
                    for (std::size_t ai = start; ai < end; ++ai)
                        chunkBodies.push_back(m_activeDynamicIndices[ai]);

                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "CCD_C%zu", ci);
                    auto tid = graph.add(buf, {&stageFormIslands, 1},
                        [this, dt, bodies = std::move(chunkBodies)]() {
                            runCCDForBodies(bodies, dt);
                        });
                    ccdTaskIds.push_back(tid);
                }

                // ── Parallel narrowphase: split broadphase pairs into chunks ────
                // Cap chunks at 4 to limit DAG node overhead.
                std::size_t numNarrowChunks = std::min(numWorkers,
                    (m_pairs.size() + 255) / 256);
                numNarrowChunks = std::min(numNarrowChunks, std::size_t{4});
                std::size_t pairsPerChunk = (m_pairs.size()
                                             + numNarrowChunks - 1) / numNarrowChunks;

                // Resize thread-local manifold buffers.
                m_islandManifolds.resize(numNarrowChunks);
                for (auto& buf : m_islandManifolds) buf.clear();

                std::vector<TaskId> narrowTaskIds;
                narrowTaskIds.reserve(numNarrowChunks);
                for (std::size_t ci = 0; ci < numNarrowChunks; ++ci) {
                    std::size_t start = ci * pairsPerChunk;
                    std::size_t end = std::min(start + pairsPerChunk, m_pairs.size());
                    if (start >= end) break;

                    // Build a span of pairs for this chunk.
                    BroadphasePair* chunkStart = m_pairs.data() + start;
                    std::size_t chunkCount = end - start;
                    auto pairSpan = std::span<const BroadphasePair>(chunkStart, chunkCount);

                    // We need to copy the span into a vector for the lambda capture.
                    std::vector<BroadphasePair> chunkPairs(pairSpan.begin(), pairSpan.end());

                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "Narrow_C%zu", ci);
                    auto tid = graph.add(buf, {&stageFormIslands, 1},
                        [this, ci, pairs = std::move(chunkPairs)]() {
                            runNarrowphase(pairs, m_islandManifolds[ci]);
                        });
                    narrowTaskIds.push_back(tid);
                }

                // ── Barrier + merge + reduce + solve ────────────────────────────
                // Combine all CCD and narrowphase task IDs for the barrier.
                std::vector<TaskId> allWorkIds;
                allWorkIds.reserve(ccdTaskIds.size() + narrowTaskIds.size());
                allWorkIds.insert(allWorkIds.end(), ccdTaskIds.begin(), ccdTaskIds.end());
                allWorkIds.insert(allWorkIds.end(), narrowTaskIds.begin(), narrowTaskIds.end());

                auto workBarrier = graph.addBarrier("CollideBarrier", allWorkIds);

                // Merge thread-local manifolds + reduce + solve.
                auto stagePostCollide = graph.add("PostCollide", {&workBarrier, 1},
                    [this, dt]() {
                        // Merge per-chunk manifolds into m_manifolds.
                        m_manifolds.clear();
                        for (auto& buf : m_islandManifolds) {
                            for (auto& mf : buf) {
                                m_manifolds.push_back(std::move(mf));
                            }
                            buf.clear();
                        }
                        reduceContactNormals(m_manifolds, m_contactReductionConfig);
                        m_solver.solve(dt, m_manifolds, m_bodies, m_solverConfig);
                    });

                islandFinalIds.push_back(stagePostCollide);
            }
        }

        // ── Stage 7 – Barrier: wait for all island pipelines to complete ──────────────────
        auto joinBarrier = graph.addBarrier("JoinIslandPipelines", islandFinalIds);

        // ── Stage 8 – Merge manifolds + Joint solving + Fracture ──────────────────────────
        //     Merge per-island manifold buffers (if per-island path was used)
        //     into the global m_manifolds, then run joint solving and fracture checks.
        TaskId stageMergeDeps[] = {joinBarrier};
        auto stageMergeAndJoint = graph.add("MergeAndJoint", stageMergeDeps, [this, dt]() {
            if (m_islandSolver.hasIslands() && !m_islandPairIndices.empty()
                && m_islandSolver.islands().size() > 1) {
                mergeIslandManifolds();
            }

            if (!m_joints.empty()) {
                m_jointSolver.solve(dt, m_joints, m_bodies, m_solverConfig);
                checkBreakableJoints();
            }
            checkFractures();
        });

        // ── Stage 9 – Post-solve hooks ────────────────────────────────────────────────────
        TaskId stagePostSolveDeps[] = {stageMergeAndJoint};
        auto stagePostSolve = graph.add("PostSolve", stagePostSolveDeps, [this, dt]() {
            for (auto* hook : m_hooks) hook->onPostSolve(dt, m_bodies);
        });

        // ── Stage 10 – Position integration + sleep ───────────────────────────────────────
        TaskId stageIntegrateDeps[] = {stagePostSolve};
        graph.add("IntegratePositions", stageIntegrateDeps, [this, dt]() {
            integratePositions(dt);
            updateSleep(dt);
        });

        // ── Execute the graph ──────────────────────────────────────────────────────────────
        graph.execute(ensureThreadPool());

        // Drain the thread pool after each frame to flush any residual work items
        // and guarantee all worker threads are idle before the next frame builds
        // its DAG.  This prevents races on MinGW where worker threads may still
        // hold references to task graph data during the next frame's clear()+add().
        if (m_threadPool) {
            m_threadPool->drain();
        }

        // Accumulate profile events across frames if profiling is enabled.
        if (graph.isProfilingEnabled()) {
            const auto& events = graph.profileEvents();
            m_accumulatedProfileEvents.insert(
                m_accumulatedProfileEvents.end(),
                events.begin(), events.end());
        }

        ++m_stepCount;
    }

    // ─── Phase 1: Forces ────────────────────────────────────────────────────────────────────────
    //
    //  Rayleigh damping model:
    //    F_damp = -c · v         (linear)
    //    τ_damp = -c · ω         (angular)
    //
    //  This is physically correct viscous damping, applied as a force before velocity
    //  integration.  Unlike the old multiplicative model (v *= 1-damp), Rayleigh damping
    //  is frame-rate independent, mass-aware, and participates in the symplectic integration
    //  scheme naturally (damped energy leaves the system through the force channel rather than
    //  being an ad-hoc post-solve velocity scale).
    //
    //  Per-body damping overrides the global default when set to a value > 0.
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void integrateForces(float dt) {
        m_bodies.clearForces();
        auto h = m_bodies.hot();  // solver-hot view

        // Aligned pointers for SIMD-friendly access.
        float*      invMasses    = std::assume_aligned<16>(h.inverseMasses.data());
        Vec3f*      forces       = std::assume_aligned<16>(h.forces.data());
        Vec3f*      linVels      = std::assume_aligned<16>(h.linearVelocities.data());
        Vec3f*      angVels      = std::assume_aligned<16>(h.angularVelocities.data());
        Vec3f*      torques      = std::assume_aligned<16>(h.torques.data());
        float*      linearDamping = std::assume_aligned<16>(m_bodies.linearDamping.data());
        float*      angularDamping = std::assume_aligned<16>(m_bodies.angularDamping.data());

        // Active-dynamic index array: all indices in m_activeDynamicIndices are
        // guaranteed to be active && dynamic. Rebuilt at the start of each frame.
        //
        // SIMD batch-4 loop with scalar remainder. All dynamic bodies have invM > 0
        // so gravity is always applied. Damping terms are always computed — when the
        // coefficient is zero, the term is zero and has no effect.
        const Vec3f g = m_config.gravity;
        const float globalLinDamp = m_config.linearDamping;
        const float globalAngDamp = m_config.angularDamping;

        std::size_t ai = 0;
        std::size_t count = m_activeDynamicIndices.size();

        // SIMD batch: process 4 bodies at once using SSE2.
        for (; ai + 4 <= count; ai += 4) {
            std::size_t idx[4] = {
                m_activeDynamicIndices[ai + 0],
                m_activeDynamicIndices[ai + 1],
                m_activeDynamicIndices[ai + 2],
                m_activeDynamicIndices[ai + 3]
            };
            integrateForcesBatch4(idx, invMasses, forces, linVels, torques, angVels,
                                 linearDamping, angularDamping, g,
                                 globalLinDamp, globalAngDamp);
        }

        // Scalar remainder for the last 0-3 bodies.
        for (; ai < count; ++ai) {
            std::size_t i = m_activeDynamicIndices[ai];

            // Gravity: all dynamic bodies have invM > 0.
            forces[i] += g / invMasses[i];

            // Linear damping: branchless — term is zero when dampLin == 0.
            float dampLin = (linearDamping[i] > 0.0f) ? linearDamping[i] : globalLinDamp;
            forces[i] += -dampLin * linVels[i];

            // Angular damping: branchless.
            float dampAng = (angularDamping[i] > 0.0f) ? angularDamping[i] : globalAngDamp;
            torques[i] += -dampAng * angVels[i];
        }

        for (auto* hook : m_hooks) hook->onApplyForces(dt, m_bodies);
    }

    // ─── Phase 2: Velocity integration (Semi-Implicit / Symplectic Euler) ───────────────────────
    //
    //  Why semi-implicit Euler for high mass ratios?
    //    Standard explicit Euler systematically injects energy, causing blow-up with mass
    //    ratios > 10³:1.  Semi-implicit Euler is symplectic – it conserves energy on average
    //    and remains stable for industrial mass ratios > 10⁶:1 (e.g. crane + steel beam).
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void integrateVelocities(float dt) {
        auto h = m_bodies.hot();  // solver-hot view

        // Aligned pointers for SIMD-friendly access.
        float*      invMasses    = std::assume_aligned<16>(h.inverseMasses.data());
        Vec3f*      forces       = std::assume_aligned<16>(h.forces.data());
        Vec3f*      linVels      = std::assume_aligned<16>(h.linearVelocities.data());
        Vec3f*      angVels      = std::assume_aligned<16>(h.angularVelocities.data());
        Vec3f*      invInertia   = std::assume_aligned<16>(h.inverseInertiaDiag.data());
        Quaternionf* rots        = std::assume_aligned<16>(h.rotations.data());
        Vec3f*      torques      = std::assume_aligned<16>(h.torques.data());

        // Active-dynamic index array: all indices are guaranteed active && dynamic.
        // All bodies here have invM > 0, so the linear acceleration is always applied.
        // SIMD batch-4 for linear velocity, per-body scalar for angular.
        std::size_t ai = 0;
        std::size_t count = m_activeDynamicIndices.size();

        // SIMD batch: process 4 bodies' linear velocity at once.
        for (; ai + 4 <= count; ai += 4) {
            std::size_t idx[4] = {
                m_activeDynamicIndices[ai + 0],
                m_activeDynamicIndices[ai + 1],
                m_activeDynamicIndices[ai + 2],
                m_activeDynamicIndices[ai + 3]
            };
            integrateLinearVelocitiesBatch4(idx, invMasses, forces, linVels, dt);

            // Angular per-body (quaternion ops not easily SIMD-vectorized).
            integrateAngularVelocity(idx[0], torques, invInertia, rots, angVels, dt);
            integrateAngularVelocity(idx[1], torques, invInertia, rots, angVels, dt);
            integrateAngularVelocity(idx[2], torques, invInertia, rots, angVels, dt);
            integrateAngularVelocity(idx[3], torques, invInertia, rots, angVels, dt);
        }

        // Scalar remainder for the last 0-3 bodies.
        for (; ai < count; ++ai) {
            std::size_t i = m_activeDynamicIndices[ai];

            // Linear: accel = force * invMass (unconditional — all dynamic bodies have invM > 0).
            Vec3f accel = forces[i] * invMasses[i];
            linVels[i] += accel * dt;

            // Angular: a = I_w^-1 * t = R * (I_local^-1 * (R^T * t))
            integrateAngularVelocity(i, torques, invInertia, rots, angVels, dt);
        }

                // --- NaN/Inf guard + speed cap (always-on — prevents NaN propagation
                //     that causes extreme FPU slowdown and solver instability) ---
                // Uses batch-4 SIMD paths for velocity clamping and NaN detection.
                constexpr float kMaxSpeed = 500.0f;
                {
                    std::size_t ai = 0;
                    std::size_t count = m_activeDynamicIndices.size();
                    for (; ai + 4 <= count; ai += 4) {
                        std::size_t idx[4] = {
                            m_activeDynamicIndices[ai + 0],
                            m_activeDynamicIndices[ai + 1],
                            m_activeDynamicIndices[ai + 2],
                            m_activeDynamicIndices[ai + 3]
                        };
                        clampVelocityBatch4(idx,
                            h.linearVelocities.data(),
                            h.angularVelocities.data(),
                            kMaxSpeed);
                    }
                    // Scalar remainder for the last 0-3 bodies.
                    for (; ai < count; ++ai) {
                        std::size_t i = m_activeDynamicIndices[ai];
                        Vec3f& v = h.linearVelocities[i];
                        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                            v = Vec3f::zero();
                        } else {
                            float speedSq = v.lengthSquared();
                            if (speedSq > kMaxSpeed * kMaxSpeed) {
                                v *= kMaxSpeed / std::sqrt(speedSq);
                            }
                        }
                        Vec3f& w = h.angularVelocities[i];
                        if (!std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z)) {
                            w = Vec3f::zero();
                        } else {
                            float angSpeedSq = w.lengthSquared();
                            if (angSpeedSq > kMaxSpeed * kMaxSpeed) {
                                w *= kMaxSpeed / std::sqrt(angSpeedSq);
                            }
                        }
                    }
                }
    }

    // ─── Phase 2b: Continuous Collision Detection (CCD) ─────────────────────────────────────
    //
    //  Runs after velocity integration, before collision detection.  For each body with a
    //  non-None CcdMode, we test the swept shape against all static/kinematic obstacles
    //  and correct position/velocity if a TOI < 1 is found.
    //
    //  Only bodies moving faster than `m_config.ccdSpeedThreshold` are processed.
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void runCCD(float dt) {
        float speedThresh = m_config.ccdSpeedThreshold;
        auto h = m_bodies.hot();  // hot view for velocity checks

        // Use active-dynamic index array instead of scanning all bodies.
        // Use batch-4 SIMD speed threshold check for early-out on slow batches.
        std::size_t ai = 0;
        std::size_t count = m_activeDynamicIndices.size();
        for (; ai + 4 <= count; ai += 4) {
            std::size_t idx[4] = {
                m_activeDynamicIndices[ai + 0],
                m_activeDynamicIndices[ai + 1],
                m_activeDynamicIndices[ai + 2],
                m_activeDynamicIndices[ai + 3]
            };

            // Batch speed threshold check: returns 4-bit mask.
            int bits = ccdSpeedThresholdBatch4(idx,
                h.linearVelocities.data(), speedThresh);
            if (bits == 0) continue;  // All 4 bodies below threshold — skip batch.

            for (int b = 0; b < 4; ++b) {
                if (!(bits & (1 << b))) continue;  // This body below threshold.
                std::size_t i = idx[b];
                CcdMode mode = static_cast<CcdMode>(m_bodies.ccdModes[i]);
                if (mode == CcdMode::None) continue;
                if (m_bodies.shapeCount[i] == 0) continue;

                RigidBodyHandle hh = static_cast<RigidBodyHandle>(i);
                ccdResolveBody(hh, dt, m_bodies, m_shapes,
                               m_bodies.restitutions[hh],
                               m_bodies.frictions[hh],
                               mode);
            }
        }

        // Scalar remainder for the last 0-3 bodies.
        for (; ai < count; ++ai) {
            std::size_t i = m_activeDynamicIndices[ai];
            CcdMode mode = static_cast<CcdMode>(m_bodies.ccdModes[i]);
            if (mode == CcdMode::None) continue;
            if (h.linearVelocities[i].length() < speedThresh) continue;
            if (m_bodies.shapeCount[i] == 0) continue;

            RigidBodyHandle hh = static_cast<RigidBodyHandle>(i);
            ccdResolveBody(hh, dt, m_bodies, m_shapes,
                           m_bodies.restitutions[hh],
                           m_bodies.frictions[hh],
                           mode);
        }
    }

    /// @brief  Run CCD on a specific subset of bodies (for per-island CCD tasks).
    ///
    /// The caller is responsible for filtering: this method processes every handle
    /// in the span, applying the standard speed/active/CCD-mode checks per body.
    ///
    /// @param bodies  Span of RigidBodyHandles to test for CCD.
    /// @param dt      Fixed sub-step duration.
    void runCCDForBodies(std::span<const RigidBodyHandle> bodies, float dt) {
        float speedThresh = m_config.ccdSpeedThreshold;
        auto h = m_bodies.hot();  // hot view for velocity checks
        for (auto bh : bodies) {
            if (!m_bodies.activeFlags[bh]) continue;
            if (m_bodies.motionTypes[bh] != MotionType::Dynamic) continue;
            CcdMode mode = static_cast<CcdMode>(m_bodies.ccdModes[bh]);
            if (mode == CcdMode::None) continue;
            if (h.linearVelocities[bh].length() < speedThresh) continue;
            if (m_bodies.shapeCount[bh] == 0) continue;

            ccdResolveBody(bh, dt, m_bodies, m_shapes,
                           m_bodies.restitutions[bh],
                           m_bodies.frictions[bh],
                           mode);
        }
    }

    // ─── Phase 3a: Broadphase (global pair generation) ────────────────────────────────────────
    //
    //  Runs globally over all active bodies to find potential overlapping pairs.
    //  The broadphase output (m_pairs) is later partitioned by island for per-island
    //  narrowphase and solving.
    //
    //  Side-effect: saves the current m_manifolds into m_oldManifolds for warm-start
    //  cache matching during the subsequent narrowphase.
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void runBroadphase() {
        // Retain old manifolds for warm-start cache matching.
        // We swap rather than copy to avoid an allocation; the old array is
        // recycled as the new output array during narrowphase.
        m_oldManifolds.swap(m_manifolds);

        // Broadphase — dispatch to the selected algorithm.
        switch (m_broadphaseConfig.type) {
        case BroadphaseType::SpatialHash: {
            SpatialHashConfig shc;
            shc.cellSize = m_broadphaseConfig.spatialHashCellSize;
            generateSpatialHashPairs(m_bodies, m_shapes, shc, m_pairs);
            break;
        }
        case BroadphaseType::SweepAndPrune: {
            SweepAndPruneConfig sapc;
            generateSweepAndPrunePairs(m_bodies, m_shapes, sapc, m_pairs);
            break;
        }
        case BroadphaseType::Dbvt: {
            DbvtConfig dbvtc;
            m_dbvt.sync(m_bodies, m_shapes, dbvtc);
            generateDbvtPairs(m_bodies, dbvtc, m_dbvt, m_pairs);
            break;
        }
        case BroadphaseType::SimdBruteForce: {
            generateBroadphasePairsSimd(m_bodies, m_shapes, m_pairs);
            break;
        }
        default: {
            generateBroadphasePairs(m_bodies, m_shapes, m_broadphaseConfig, m_pairs);
            break;
        }
        }
    }

    // ─── Phase 3b: Narrowphase (per-pair contact generation) ───────────────────────────────────
    //
    //  Iterates a set of broadphase body-pairs and runs the narrowphase collision
    //  detection (SAT, GJK+EPA) to produce contact manifolds.
    //
    //  Thread-safe when `outManifolds` is local to the calling island task.
    //  Warm-start caches are transferred from `m_oldManifolds` for fast convergence.
    //
    //  @param pairs        Broadphase pairs to test (subset of m_pairs).
    //  @param outManifolds Output manifold buffer (appended to).
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void runNarrowphase(const std::vector<BroadphasePair>& pairs,
                        std::vector<ContactManifold>& outManifolds) const {

        for (const auto& pair : pairs) {
            RigidBodyHandle hA = pair.bodyA;
            RigidBodyHandle hB = pair.bodyB;

            // Collision filter check.
            if (m_collisionFilter && !m_collisionFilter->shouldCollide(hA, hB)) continue;

            std::uint32_t startA = m_bodies.shapeStart[hA];
            std::uint32_t countA = m_bodies.shapeCount[hA];
            std::uint32_t startB = m_bodies.shapeStart[hB];
            std::uint32_t countB = m_bodies.shapeCount[hB];

            if (countA == 0 || countB == 0) continue;

            for (std::uint32_t sa = 0; sa < countA; ++sa) {
                ShapeHandle shA = m_bodies.flatShapeHandles[startA + sa];
                if (shA >= m_shapes.size()) continue;

                Vec3f localPosA = m_bodies.flatShapeLocalPositions[startA + sa];
                Quaternionf localRotA = m_bodies.flatShapeLocalRotations[startA + sa];
                Vec3f worldPosA = m_bodies.positions[hA]
                                 + m_bodies.rotations[hA].rotate(localPosA);
                Quaternionf worldRotA = m_bodies.rotations[hA] * localRotA;

                const Shape& shapeA = m_shapes[shA];

                for (std::uint32_t sb = 0; sb < countB; ++sb) {
                    if (hA == hB && sa == sb) continue;

                    ShapeHandle shB = m_bodies.flatShapeHandles[startB + sb];
                    if (shB >= m_shapes.size()) continue;

                    Vec3f localPosB = m_bodies.flatShapeLocalPositions[startB + sb];
                    Quaternionf localRotB = m_bodies.flatShapeLocalRotations[startB + sb];
                    Vec3f worldPosB = m_bodies.positions[hB]
                                     + m_bodies.rotations[hB].rotate(localPosB);
                    Quaternionf worldRotB = m_bodies.rotations[hB] * localRotB;

                    const Shape& shapeB = m_shapes[shB];

                    ContactManifold manifold;
                    manifold.bodyA = hA;
                    manifold.bodyB = hB;

                    if (collideShapes(shapeA, worldPosA, worldRotA,
                                      shapeB, worldPosB, worldRotB,
                                      m_bodies.restitutions[hA],
                                      m_bodies.restitutions[hB],
                                      m_bodies.frictions[hA],
                                      m_bodies.frictions[hB],
                                      manifold)) {

                        // Warm-start cache transfer from previous frame.
                        transferWarmStart(manifold, m_oldManifolds);
                        outManifolds.push_back(std::move(manifold));
                    }
                }
            }
        }
    }

    // ─── Phase 3c: Sequential detectCollisions (combines broadphase + narrowphase) ────────────
    //
    //  Used by the sequential fixedStep() path.  The task-graph path uses runBroadphase()
    //  followed by per-island narrowphase tasks instead.
    // ────────────────────────────────────────────────────────────────────────────────────────────

    void detectCollisions() {
        runBroadphase();
        runNarrowphase(m_pairs, m_manifolds);
    }

    // ─── Island pair assignment ────────────────────────────────────────────────────────────────

    /// @brief  After island formation, assign each broadphase pair to its island.
    ///
    /// Builds a body→island lookup table, then scans m_pairs to produce
    /// per-island lists of pair indices.  Bodies and their pairs always
    /// belong to exactly one island (by construction of the union-find).
    void assignPairsToIslands() {
        m_islandPairIndices.clear();
        const auto& islands = m_islandSolver.islands();
        if (islands.empty()) return;

        // Body → island index lookup.
        std::vector<int> bodyToIsland(m_bodies.size(), -1);
        for (std::size_t i = 0; i < islands.size(); ++i) {
            for (auto b : islands[i].bodies) {
                bodyToIsland[b] = static_cast<int>(i);
            }
        }

        // Assign each pair to its island (check bodyA; bodyB is guaranteed
        // to be in the same island by union-find connectivity).
        m_islandPairIndices.resize(islands.size());
        for (std::size_t pi = 0; pi < m_pairs.size(); ++pi) {
            int islandIdx = bodyToIsland[m_pairs[pi].bodyA];
            if (islandIdx >= 0) {
                m_islandPairIndices[islandIdx].push_back(pi);
            }
        }

        // Resize per-island manifold buffers to match.
        m_islandManifolds.resize(islands.size());
        for (auto& buf : m_islandManifolds) {
            buf.clear();
        }
    }

    /// @brief  Extract the broadphase pairs for a given island as a contiguous vector.
    std::vector<BroadphasePair> getIslandPairs(std::size_t islandIdx) const {
        std::vector<BroadphasePair> pairs;
        if (islandIdx >= m_islandPairIndices.size()) return pairs;
        const auto& indices = m_islandPairIndices[islandIdx];
        pairs.reserve(indices.size());
        for (auto idx : indices) {
            pairs.push_back(m_pairs[idx]);
        }
        return pairs;
    }

    /// @brief  Solve a local manifold buffer directly (no island indexing).
    ///         Used by per-island solve tasks in the task-graph pipeline.
    void solveManifolds(float dt,
                        std::vector<ContactManifold>& manifolds,
                        RigidBodyStorage& bodies,
                        const SolverConfig& config) const {
        if (manifolds.empty()) return;
        ConstraintSolver solver;
        solver.solve(dt, manifolds, bodies, config);
    }

    /// @brief  Merge per-island manifold buffers into m_manifolds.
    ///         Called after the per-island sub-DAG join barrier.
    void mergeIslandManifolds() {
        m_manifolds.clear();
        for (auto& buf : m_islandManifolds) {
            for (auto& mf : buf) {
                m_manifolds.push_back(std::move(mf));
            }
            buf.clear();
        }
    }

    /// @brief  Match a new manifold against the previous frame's manifolds and copy impulse caches.
    ///
    /// Uses **feature-ID matching**: for each new contact point, we search the old manifold
    /// for a point with the same (featureA, featureB) pair.  This is robust against
    /// reordering or replacement of contact points (common in box-box face clipping).
    ///
    /// If no matching feature pair is found, the impulse starts at zero (cold start),
    /// which is safe and converges within 2-3 solver iterations.
    static void transferWarmStart(ContactManifold& manifold,
                                   const std::vector<ContactManifold>& oldManifolds) noexcept {
        for (const auto& old : oldManifolds) {
            if (old.bodyA == manifold.bodyA && old.bodyB == manifold.bodyB) {
                // Match each new point to an old point by (featureA, featureB).
                for (int i = 0; i < manifold.pointCount; ++i) {
                    for (int j = 0; j < old.pointCount; ++j) {
                        if (manifold.points[i].featureA == old.points[j].featureA &&
                            manifold.points[i].featureB == old.points[j].featureB) {
                            manifold.points[i].normalImpulse     = old.points[j].normalImpulse;
                            manifold.points[i].tangentImpulse[0] = old.points[j].tangentImpulse[0];
                            manifold.points[i].tangentImpulse[1] = old.points[j].tangentImpulse[1];
                            manifold.points[i].tangent[0]        = old.points[j].tangent[0];
                            manifold.points[i].tangent[1]        = old.points[j].tangent[1];
                            break;
                        }
                    }
                }
                break;
            }
        }
    }

    // ─── Phase 7: Position integration ──────────────────────────────────────────────────────────

    void integratePositions(float dt) {
        auto h = m_bodies.hot();  // solver-hot view

        // Aligned pointers for SIMD-friendly access.
        Vec3f*      positions    = std::assume_aligned<16>(h.positions.data());
        Vec3f*      linVels      = std::assume_aligned<16>(h.linearVelocities.data());
        Vec3f*      angVels      = std::assume_aligned<16>(h.angularVelocities.data());
        Quaternionf* rots        = std::assume_aligned<16>(h.rotations.data());
        Quaternionf* inertiaRots = std::assume_aligned<16>(h.inertiaRotations.data());

        // Active dynamic bodies — SIMD batch-4 for linear, per-body for angular.
        std::size_t ai = 0;
        std::size_t count = m_activeDynamicIndices.size();

        // SIMD batch: process 4 bodies' linear positions at once.
        for (; ai + 4 <= count; ai += 4) {
            std::size_t idx[4] = {
                m_activeDynamicIndices[ai + 0],
                m_activeDynamicIndices[ai + 1],
                m_activeDynamicIndices[ai + 2],
                m_activeDynamicIndices[ai + 3]
            };
            integrateLinearPositionsBatch4(idx, positions, linVels, dt);

            // Angular per-body (quaternion normalize not SIMD-friendly).
            integrateAngularPosition(idx[0], angVels, rots, inertiaRots);
            integrateAngularPosition(idx[1], angVels, rots, inertiaRots);
            integrateAngularPosition(idx[2], angVels, rots, inertiaRots);
            integrateAngularPosition(idx[3], angVels, rots, inertiaRots);
        }

        // Scalar remainder for the last 0-3 bodies.
        for (; ai < count; ++ai) {
            std::size_t i = m_activeDynamicIndices[ai];

            // Linear.
            positions[i] += linVels[i] * dt;

            // Angular: q' = q + 1/2 * w * q * dt
            integrateAngularPosition(i, angVels, rots, inertiaRots);
        }

        // ── Pass 2: Active kinematic bodies (not in dynamic index array) ────────────
        // These only get linear position updates (no angular integration).
        for (std::size_t i = 0; i < m_bodies.size(); ++i) {
            if (!m_bodies.activeFlags[i]) continue;
            if (m_bodies.motionTypes[i] != MotionType::Kinematic) continue;
            positions[i] += linVels[i] * dt;
        }

        // --- NaN/Inf guard (always-on, SIMD batch-4) ---
        {
            std::size_t ai = 0;
            std::size_t count = m_activeDynamicIndices.size();
            for (; ai + 4 <= count; ai += 4) {
                std::size_t idx[4] = {
                    m_activeDynamicIndices[ai + 0],
                    m_activeDynamicIndices[ai + 1],
                    m_activeDynamicIndices[ai + 2],
                    m_activeDynamicIndices[ai + 3]
                };
                checkPositionsFiniteBatch4(idx, positions);
            }
            // Scalar remainder for the last 0-3 bodies.
            for (; ai < count; ++ai) {
                std::size_t i = m_activeDynamicIndices[ai];
                Vec3f& p = positions[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    p = Vec3f::zero();
                }
            }
        }
    }

    // ─── Breakable constraints ────────────────────────────────────────────────────────────────

    /// @brief  Check all joints for break threshold violations.
    ///
    /// For each joint with breakThreshold > 0, compute the total accumulated impulse
    /// (sum of absolute cached impulses).  If it exceeds the threshold, mark the joint
    /// as broken.  Broken joints are removed in a second pass.
    ///
    /// The check is O(n) in the number of joints and happens once per fixed step,
    /// after the joint solver has updated the impulse caches.
    void checkBreakableJoints() noexcept {
        if (m_joints.empty()) return;

        // First pass: mark broken joints.
        for (auto& joint : m_joints) {
            if (joint.broken) continue;
            if (joint.breakThreshold <= 0.0f) continue;  // unbreakable

            float totalImpulse = 0.0f;
            int n = joint.cacheSize();
            for (int i = 0; i < n; ++i) {
                totalImpulse += std::abs(joint.impulseCache[i]);
            }

            if (totalImpulse > joint.breakThreshold) {
                joint.broken = true;
            }
        }

        // Second pass: remove broken joints (reverse order for stable indices).
        for (std::size_t i = m_joints.size(); i > 0; --i) {
            std::size_t idx = i - 1;
            if (m_joints[idx].broken) {
                if (idx != m_joints.size() - 1) {
                    m_joints[idx] = std::move(m_joints.back());
                }
                m_joints.pop_back();
            }
        }
    }

    // ─── Phase 8: Sleep management ──────────────────────────────────────────────────────────────

    void updateSleep(float dt) {
        auto h = m_bodies.hot();  // hot view for velocity reads

        // Aligned pointers for SIMD-friendly access.
        float*   sleepTimers = std::assume_aligned<16>(m_bodies.sleepTimers.data());
        uint8_t* activeFlags = std::assume_aligned<16>(m_bodies.activeFlags.data());

        // SIMD batch-4 loop with scalar remainder.
        // The batch function computes |v|² + |w|² for 4 bodies at once, then
        // falls back to per-body branches for the timer/flag writes (uint8_t
        // writes and conditional branches are not SIMD-friendly).
        std::size_t ai = 0;
        std::size_t count = m_activeDynamicIndices.size();

        for (; ai + 4 <= count; ai += 4) {
            std::size_t idx[4] = {
                m_activeDynamicIndices[ai + 0],
                m_activeDynamicIndices[ai + 1],
                m_activeDynamicIndices[ai + 2],
                m_activeDynamicIndices[ai + 3]
            };
            // Save original flags for change detection (the SIMD batch function
            // writes directly to activeFlags).
            uint8_t flagBefore[4] = {
                activeFlags[idx[0]], activeFlags[idx[1]],
                activeFlags[idx[2]], activeFlags[idx[3]]
            };
            updateSleepBatch4(idx,
                h.linearVelocities.data(),
                h.angularVelocities.data(),
                sleepTimers,
                activeFlags,
                dt,
                m_config.sleepEnergyThreshold,
                m_config.sleepTimeRequired);
            // Check if any flag changed in this batch.
            for (int b = 0; b < 4; ++b) {
                if (activeFlags[idx[b]] != flagBefore[b]) {
                    m_activeIndicesDirty = true;
                    break;
                }
            }
        }

        // Scalar remainder for the last 0-3 bodies.
        for (; ai < count; ++ai) {
            std::size_t i = m_activeDynamicIndices[ai];

            float eKin = h.linearVelocities[i].lengthSquared()
                       + h.angularVelocities[i].lengthSquared();

            uint8_t flagBefore = m_bodies.activeFlags[i];

            if (eKin < m_config.sleepEnergyThreshold) {
                m_bodies.sleepTimers[i] += dt;
                if (m_bodies.sleepTimers[i] >= m_config.sleepTimeRequired) {
                    m_bodies.activeFlags[i] = false;
                }
            } else {
                m_bodies.sleepTimers[i] = 0.0f;
                m_bodies.activeFlags[i] = true;
            }

            if (m_bodies.activeFlags[i] != flagBefore) {
                m_activeIndicesDirty = true;
            }
        }
    }

private:
    // ─── Internal helpers ───────────────────────────────────────────────────────────────────

    /// @brief  Run the PBD soft body simulation for all soft bodies.
    ///
    /// Steps each soft body through one fixed timestep using the
    /// Position-Based Dynamics solver.  Particle-rigid body collision
    /// is one-way (particles are projected out of shapes).
    ///
    /// Called from fixedStep() and fixedStepTaskGraph() before rigid body forces.
    void stepSoftBodies(float dt) noexcept {
        for (auto& body : m_softBodies) {
            if (body.particleCount() == 0) continue;
            m_softBodySolver.step(
                dt, body,
                m_bodies.size(),
                m_bodies.positions.data(),
                m_bodies.rotations.data(),
                m_bodies.shapeStart.data(),
                m_bodies.shapeCount.data(),
                m_bodies.flatShapeHandles.data(),
                m_bodies.flatShapeLocalPositions.data(),
                m_bodies.flatShapeLocalRotations.data(),
                m_shapes.data(),
                m_bodies.activeFlags.data()
            );
        }
    }

    // ─── Fracture system ────────────────────────────────────────────────────────────────────────

    /// @brief  After the constraint solve, check contact impulses on destructible bodies.
    ///
    /// Iterates all contact manifolds and joint caches, accumulating the maximum impulse
    /// seen by each destructible body.  Bodies exceeding their threshold are fractured.
    void checkFractures() noexcept {
        if (m_destructibleFlags.empty()) return;

        // Reset per-step impulse trackers.
        for (auto& dd : m_destructibleData) {
            dd.resetImpulse();
        }

        // Check contact impulses.
        for (const auto& manifold : m_manifolds) {
            RigidBodyHandle hA = manifold.bodyA;
            RigidBodyHandle hB = manifold.bodyB;

            for (int p = 0; p < manifold.pointCount; ++p) {
                float impulse = std::abs(manifold.points[p].normalImpulse);

                if (hA < m_destructibleFlags.size() && m_destructibleFlags[hA]) {
                    m_destructibleData[hA].recordImpulse(impulse);
                }
                if (hB < m_destructibleFlags.size() && m_destructibleFlags[hB]) {
                    m_destructibleData[hB].recordImpulse(impulse);
                }
            }
        }

        // Check joint impulses.
        for (const auto& joint : m_joints) {
            RigidBodyHandle hA = joint.bodyA;
            RigidBodyHandle hB = joint.bodyB;

            float totalImpulse = 0.0f;
            int n = joint.cacheSize();
            for (int i = 0; i < n; ++i) {
                totalImpulse += std::abs(joint.impulseCache[i]);
            }

            if (hA < m_destructibleFlags.size() && m_destructibleFlags[hA]) {
                m_destructibleData[hA].recordImpulse(totalImpulse);
            }
            if (hB < m_destructibleFlags.size() && m_destructibleFlags[hB]) {
                m_destructibleData[hB].recordImpulse(totalImpulse);
            }
        }

        // Execute fractures for bodies that exceeded their threshold.
        // Iterate in reverse order so removals don't invalidate indices.
        for (std::size_t h = m_destructibleData.size(); h > 0; --h) {
            std::size_t idx = h - 1;
            if (m_destructibleFlags[idx] && m_destructibleData[idx].shouldBreak()) {
                executeFracture(static_cast<RigidBodyHandle>(idx));
            }
        }
    }

    /// @brief  Execute fracture for a single body: replace it with fragments.
    ///
    /// The original body is removed and its pre-computed fragments are spawned as
    /// independent rigid bodies with matching velocities.  Fragment velocities include
    /// the contribution from the original body's angular velocity at the fragment's
    /// offset position, conserving linear and angular momentum.
    void executeFracture(RigidBodyHandle handle) noexcept {
        if (handle >= m_destructibleData.size()) return;

        auto& data = m_destructibleData[handle];
        if (data.isBroken || data.fragments.empty()) return;
        data.isBroken = true;

        // Save original body state before removal.
        Vec3f origPos = m_bodies.positions[handle];
        Quaternionf origRot = m_bodies.rotations[handle];
        Vec3f origLinVel = m_bodies.linearVelocities[handle];
        Vec3f origAngVel = m_bodies.angularVelocities[handle];
        float restitution = m_bodies.restitutions[handle];
        float friction = m_bodies.frictions[handle];
        RigidBodyHandle originalHandle = handle;

        if (!data.config.spawnFragmentsOnBreak) {
            // Just destroy the body.
            removeBody(handle);
            return;
        }

        // Spawn fragments.
        for (const auto& frag : data.fragments) {
            // Fragment world-space position = body position + rotated local center.
            Vec3f worldCenter = origPos + origRot.rotate(frag.center);

            // Fragment velocity = body linear vel + angular × offset.
            Vec3f offset = frag.center;
            Vec3f velFromAng = origAngVel.cross(origRot.rotate(offset));
            Vec3f fragVel = origLinVel + velFromAng * data.config.fragmentSpeedScale;

            // Build the fragment descriptor.
            ShapeHandle shapeH = createShape(ConvexMesh{
                frag.mesh.vertices,
                frag.mesh.vertexCount,
                frag.mesh.halfExtents,
                frag.mesh.center
            });

            RigidBodyDesc desc;
            desc.position        = worldCenter;
            desc.rotation        = origRot;
            desc.linearVelocity  = fragVel;
            desc.mass            = frag.mass;
            desc.inertiaLocal    = frag.inertia;
            desc.useAutoInertia  = false;
            desc.restitution     = restitution;
            desc.friction        = friction;
            desc.motionType      = MotionType::Dynamic;
            desc.setShape(shapeH);

            RigidBodyHandle newHandle = createBody(desc);

            // If chain fracture is enabled, make the fragment destructible too.
            if (data.config.enableChainFracture) {
                makeDestructible(newHandle, data.config);
            }
        }

        // Remove the original body.
        removeBody(originalHandle);
    }

    /// @brief  Lazily create the thread pool when parallel solving is first used.
    ThreadPool& ensureThreadPool() {
        if (!m_threadPool) {
            std::uint32_t n = m_config.numThreads;
            m_threadPool = std::make_unique<ThreadPool>(n ? static_cast<std::size_t>(n)
                                                          : std::size_t{0});
        }
        return *m_threadPool;
    }

    PhysicsWorldConfig                m_config;
    SolverConfig                      m_solverConfig;
    BroadphaseConfig                  m_broadphaseConfig;
    ContactReductionConfig            m_contactReductionConfig; ///< Contact normal reduction config.
    Dbvt                              m_dbvt;          ///< Dynamic BVH broadphase (used when type == Dbvt).
    RigidBodyStorage                  m_bodies;
    std::vector<Shape>                m_shapes;
    std::vector<ContactManifold>      m_manifolds;
    std::vector<ContactManifold>      m_oldManifolds;       ///< Previous-frame manifolds for warm-start.
    std::vector<BroadphasePair>       m_pairs;
    std::vector<ISolverHook*>         m_hooks;
    std::vector<std::vector<ContactManifold>>  m_islandManifolds;    ///< Per-island narrowphase outputs.
    std::vector<std::vector<std::size_t>>     m_islandPairIndices;  ///< Per-island pair index lists.
    ConstraintSolver                  m_solver;
    JointSolver                       m_jointSolver;
    IslandSolver                      m_islandSolver;
    std::vector<JointStorage>         m_joints;
    std::vector<SoftBody>             m_softBodies;      ///< Deformable bodies (PBD).
    SoftBodySolver                    m_softBodySolver;  ///< PBD soft body solver.
    std::vector<char>                 m_destructibleFlags; ///< 1 = destructible, keyed by RigidBodyHandle.
    std::vector<DestructibleData>     m_destructibleData;  ///< Per-body fracture data.
    std::unique_ptr<ThreadPool>       m_threadPool;      ///< Lazily created.
    ICollisionFilter*                 m_collisionFilter = nullptr; ///< Optional pair filter.
    bool                              m_useParallelSolver = false;
    bool                              m_useTaskGraphPipeline = false; ///< Enable DAG pipeline.
    float                             m_accumulator = 0.0f;
    std::uint64_t                     m_stepCount   = 0;

    /// @brief  Reusable task graph for the pipeline (rebuilt every fixed step).
    TaskGraph                         m_taskGraph;

    /// @brief  Accumulated profile events across all frames (for multi-frame summary).
    mutable std::vector<TaskGraph::ProfileEvent> m_accumulatedProfileEvents;

    // ─── Active body index array (avoids branching on activeFlags[] in hot loops) ─────────
    /// @brief  Compact array of handles for all active dynamic bodies, rebuilt every frame after
    /// the sleep phase. Hot loops iterate this array instead of all bodies with
    /// `if (!activeFlags[i]) continue;`, eliminating branch mispredictions and
    /// improving cache efficiency when many bodies are asleep.
    std::vector<RigidBodyHandle>      m_activeDynamicIndices;

    /// @brief  Dirty flag: when true, rebuildActiveDynamicIndices() must re-scan all bodies.
    /// Set to true when any body's activeFlags or motionTypes change.
    bool m_activeIndicesDirty = true;

    /// @brief  Rebuild the active-dynamic index array after sleep/awake/freeze changes.
    /// Only performs a full scan when m_activeIndicesDirty is true (i.e., when
    /// sleep/awake/freeze modified the active set since the last rebuild).
    /// This saves an O(N) scan every frame when no bodies change sleep state.
    void rebuildActiveDynamicIndices() noexcept {
        if (!m_activeIndicesDirty) return;
        m_activeDynamicIndices.clear();
        m_activeDynamicIndices.reserve(m_bodies.size());
        for (std::size_t i = 0; i < m_bodies.size(); ++i) {
            if (m_bodies.activeFlags[i] && m_bodies.motionTypes[i] == MotionType::Dynamic) {
                m_activeDynamicIndices.push_back(static_cast<RigidBodyHandle>(i));
            }
        }
        m_activeIndicesDirty = false;
    }

    /// @brief  Mark the active-dynamic indices as dirty, forcing a rebuild on the next frame.
    /// Must be called whenever a body's active flag or motion type changes.
    void markActiveIndicesDirty() noexcept { m_activeIndicesDirty = true; }

public:
    /// @brief  Auto-tune the spatial hash cell size based on body density.
    ///
    /// Uses a simple spacing heuristic: compute the diagonal of all bodies'
    /// bounding box, then estimate average spacing as diagonal / cbrt(N).
    /// Cell size = max(1.0f, avgSpacing * 2.0f).  This prevents degenerate
    /// O(n²) broadphase behaviour when many bodies are densely packed
    /// (e.g., 5000 Voronoi fragments in 5×5×5m with default 4.0m cells).
    ///
    /// Call this once after creating all bodies, before the first step().
    /// Only effective when broadphase type is SpatialHash.
    void autoTuneSpatialHashCellSize() noexcept {
        if (m_bodies.size() < 2) {
            m_broadphaseConfig.spatialHashCellSize = 4.0f;  // safe default
            return;
        }

        // Compute bounding box of all body positions.
        Vec3f bmin = Vec3f(+1e10f, +1e10f, +1e10f);
        Vec3f bmax = Vec3f(-1e10f, -1e10f, -1e10f);
        for (std::size_t i = 0; i < m_bodies.size(); ++i) {
            const Vec3f& p = m_bodies.positions[i];
            bmin.x = std::min(bmin.x, p.x);
            bmin.y = std::min(bmin.y, p.y);
            bmin.z = std::min(bmin.z, p.z);
            bmax.x = std::max(bmax.x, p.x);
            bmax.y = std::max(bmax.y, p.y);
            bmax.z = std::max(bmax.z, p.z);
        }

        // Diagonal of bounding box.
        Vec3f diag = bmax - bmin;
        float bboxDiagonal = diag.length();
        if (bboxDiagonal < 0.001f) {
            m_broadphaseConfig.spatialHashCellSize = 1.0f;
            return;
        }

        // Estimate average spacing: diagonal / cube_root(N)
        float nFloat = static_cast<float>(m_bodies.size());
        float avgSpacing = bboxDiagonal / std::cbrt(nFloat);

        // Cell size should be a few times the average spacing.
        float cellSize = std::max(1.0f, avgSpacing * 2.0f);
        // Cap at a reasonable maximum to prevent degenerate cases.
        cellSize = std::min(cellSize, bboxDiagonal * 0.25f);

        m_broadphaseConfig.spatialHashCellSize = cellSize;
    }

    // ─── Profile event accumulation (across frames) ─────────────────────────────────

    /// @brief  Accumulated profile events from all fixed-step executions.
    ///         Cleared by clearAccumulatedProfileEvents().
    [[nodiscard]] const std::vector<TaskGraph::ProfileEvent>& accumulatedProfileEvents() const noexcept {
        return m_accumulatedProfileEvents;
    }

    /// @brief  Clear the accumulated profile event buffer.
    void clearAccumulatedProfileEvents() noexcept {
        m_accumulatedProfileEvents.clear();
    }

    /// @brief  Compute per-task aggregate statistics from accumulated profile events.
    ///
    /// Unlike taskGraphProfileSummary() which only reflects the last frame's data,
    /// this method aggregates across all frames since the last clearAccumulatedProfileEvents().
    /// This gives meaningful multi-frame averages and counts.
    [[nodiscard]] std::vector<ProfileSummary> accumulatedProfileSummary() const {
        std::unordered_map<std::string, std::pair<double, std::pair<double, double>>> agg;
        for (const auto& ev : m_accumulatedProfileEvents) {
            auto it = agg.find(ev.name);
            if (it == agg.end()) {
                agg[ev.name] = {ev.elapsedMs, {ev.elapsedMs, ev.elapsedMs}};
            } else {
                it->second.first += ev.elapsedMs;
                it->second.second.first  = std::min(it->second.second.first, ev.elapsedMs);
                it->second.second.second = std::max(it->second.second.second, ev.elapsedMs);
            }
        }
        std::vector<ProfileSummary> result;
        result.reserve(agg.size());
        for (const auto& [name, data] : agg) {
            int count = 0;
            for (const auto& ev : m_accumulatedProfileEvents) {
                if (ev.name == name) ++count;
            }
            result.push_back({
                name,
                data.first,
                data.second.first,
                data.second.second,
                data.first / static_cast<double>(count),
                count
            });
        }
        std::sort(result.begin(), result.end(),
                  [](const ProfileSummary& a, const ProfileSummary& b) {
                      return a.totalMs > b.totalMs;
                  });
        return result;
    }
};

//==================================================================================================
// Multi‑threaded Task‑Based Stepping
//==================================================================================================
// ── Technical Note ───────────────────────────────────────────────────────────────────────────────
//
// The single-threaded pipeline above is deliberately simple.  Scaling to 10 000+ bodies with
// contacts and joints requires a task‑graph approach similar to PhysX or Chaos.  Below is the
// architectural blueprint.
//
// ## 1.  Island Formation
//
//     Disconnected groups of interacting bodies (via contacts/joints) are **islands**.
//     Islands are independent and can be solved in parallel.
//
//     ```
//     Island 0:  [A, B, C]  — sharing contacts
//     Island 1:  [D, E]     — sharing a joint
//     Island 2:  [F]        — solitary, trivial
//     ```
//
//     Use a **union-find (Disjoint Set Union)** pass over all active contact pairs and
//     joints to build islands in O(N · α(N)).
//
// ## 2.  Task Graph Structure
//
//     Each island yields a small DAG of jobs:
//
//     ```
//     ┌──────────────────────────────────────────────────────┐
//     │                   Frame Start                        │
//     └─────────────┬─────────────────────┬──────────────────┘
//                   │                     │
//           ┌───────▼───────┐     ┌───────▼───────┐
//           │  Broadphase   │     │  Broadphase   │     … (parallel
//           │  (Island 0)   │     │  (Island 1)   │       per island)
//           └───────┬───────┘     └───────┬───────┘
//                   │                     │
//           ┌───────▼───────┐     ┌───────▼───────┐
//           │  Narrowphase  │     │  Narrowphase  │     … (contacts)
//           └───────┬───────┘     └───────┬───────┘
//                   │                     │
//           ┌───────▼───────┐     ┌───────▼───────┐
//           │  Solver       │     │  Solver       │     … (seq. impulses)
//           │  (Island 0)   │     │  (Island 1)   │
//           └───────┬───────┘     └───────┬───────┘
//                   │                     │
//                   └──────────┬──────────┘
//                              │
//                      ┌───────▼───────┐
//                      │  Integrate    │   (can also be parallel by island)
//                      │  Positions   │
//                      └───────┬───────┘
//                              │
//                      ┌───────▼───────┐
//                      │  Frame End    │
//                      └───────────────┘
//     ```
//
// ## 3.  Determinism Guarantees
//
//     A parallel task graph can be deterministic if:
//       * Island assignment is deterministic (spatial hash vs sweep‑and‑prune).
//       * Islands are solved in a globally fixed order (e.g. increasing island ID).
//       * Within an island, contact/joint iteration order is fixed.
//     Both PhysX and Chaos rely on a **deterministic island ordering** + fixed solver
//     iteration patterns.
//
// ## 4.  Integration into This Codebase
//
//     ```cpp
//     // Step 1 – Split the SoA arrays into per-island spans.
//     //          The handles are already dense indices.
//
//     struct Island {
//         std::span<Vec3f>      positions;
//         std::span<Vec3f>      linearVelocities;
//         // … other SoA slices
//         std::vector<Contact>  contacts;
//         bool                  isActive;
//     };
//
//     // Step 2 – Enqueue island jobs to a task scheduler (e.g. Intel TBB, EnkiTS,
//     //          or a custom thread pool with work stealing).
//
//     // Step 3 – Merge results (trivial – the spans point into the SoA arrays,
//     //          so modifications are already reflected in the main storage).
//     ```
//
// ## 5.  Recommended Approach
//
//     | Phase             | Parallelism                | Job Type       |
//     |-------------------|----------------------------|----------------|
//     | Broadphase        | Partition space (grid/ BVH)| Parallel for   |
//     | Narrowphase       | Per pair                   | Parallel for   |
//     | Island building   | Union‑find (atomic)        | Parallel for   |
//     | Solver            | Per island                 | Task per island|
//     | Integration       | Per body                   | Parallel for   |
//
//     A simple thread‑pool with `std::atomic<int>` counters (as a fork‑join barrier) is
//     sufficient for the first parallel pass.  Later, a full DAG scheduler unlocks
//     inter‑island parallelism without global barriers.
//
// For a complete reference, see the PhysX "Scene Query" and "Task Programming" SDK docs,
// or the Unreal Engine "Chaos Physics" technical documentation.
//==================================================================================================

} // namespace ciphyxs
