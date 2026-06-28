//==================================================================================================
/// @file  SoftBody.hpp
/// @brief  Deformable body simulation using Position-Based Dynamics (PBD).
///
/// CiPhyxs implements soft bodies through a particle-based **Position-Based Dynamics**
/// (PBD) framework — the same approach used by Nvidia Flex, Unreal Chaos, and Bullet
/// for cloth, deformables, and volumetric soft bodies.
///
/// ## Why PBD?
///
/// Traditional force-based (spring-mass) soft bodies require tiny timesteps to remain
/// stable.  PBD operates directly on **positions**, projecting constraint violations
/// via Gauss-Seidel iterations.  This produces visually plausible, stable deformation
/// even at 60 Hz fixed timesteps with high stiffness values.
///
/// ## Architecture
///
/// ```
/// // 1. Verlet integrate positions (gravity, damping implicit in velocity)
/// // 2. Repeat N times:
/// //      a. Project distance constraints (stretch/shear springs)
/// //      b. Project bend constraints (cloth folding stiffness)
/// //      c. Project volume constraints (tetrahedral incompressibility)
/// //      d. Collide particles against rigid body shapes (sphere/box/plane/capsule)
/// // 3. Recover velocities from position deltas
/// ```
///
/// ## Integration into PhysicsWorld
///
/// Soft bodies step **before** rigid bodies each fixed timestep.  Particle-rigid
/// body collision is one-way (particles are pushed out; rigid bodies receive no
/// impulse).  Two-way coupling can be added by generating ContactManifold entries
/// from particle penetration and feeding them into the main sequential-impulse solver.
///
/// @see  PhysicsWorld  — Orchestrates the full simulation pipeline (rigid + soft).
/// @see  RigidBody     — SoA storage that soft body particles collide against.
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Mat3.hpp"
#include "../math/Quaternion.hpp"
#include "AlignedAllocator.hpp"
#include "RigidBody.hpp"
#include "../collision/Shape.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Soft body handle
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Index into the PhysicsWorld's soft-body array.
using SoftBodyHandle = std::size_t;

/// @brief  Sentinel for an invalid soft-body handle.
inline constexpr SoftBodyHandle kInvalidSoftBody = std::numeric_limits<std::size_t>::max();

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SoftBodyDesc
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Description used to initialise a SoftBody inside the PhysicsWorld.
///
/// @code
/// SoftBodyDesc desc;
/// // Option A: manual particle setup
/// desc.positions = { Vec3f(0,0,0), Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1) };
/// desc.masses    = { 1.0f, 1.0f, 1.0f, 1.0f };
/// // Option B: use factory methods on the returned SoftBody directly:
/// //   auto idx = world.createSoftBody(desc);
/// //   world.softBody(idx) = SoftBody::createCloth(10, 10, 0.5f, 1.0f);
/// @endcode
struct SoftBodyDesc {
    /// @brief  Initial particle positions (world space).
    std::vector<Vec3f> positions;

    /// @brief  Per-particle mass.  0 = pinned (immovable).
    std::vector<float> masses;

    /// @brief  Gravity applied to all particles (default: -9.81 on Y).
    Vec3f gravity = Vec3f(0.0f, -9.81f, 0.0f);

    /// @brief  Damping factor applied to velocity each step [0, 1).
    ///         Higher values dissipate energy faster.
    float damping = 0.01f;

    /// @brief  Friction coefficient for particle-rigid body contact [0, ∞).
    float friction = 0.5f;

    /// @brief  Skin margin around each particle for collision detection (metres).
    ///         Smaller values allow closer contact but may cause tunnelling.
    float collisionMargin = 0.005f;

    /// @brief  Number of Gauss-Seidel constraint-iteration passes per step.
    ///         Typical range: 3 (cheap cloth) to 10 (stiff volumetric).
    ///         Higher = stiffer but more expensive.
    int constraintIterations = 5;

    /// @brief  PBD constraint stiffness [0, 1].  1 = perfectly rigid constraint.
    ///         Lower values make the body softer.
    float constraintStiffness = 0.8f;

    /// @brief  Flag indicating if the body is a cloth (renders as triangle mesh).
    bool isCloth = false;

    /// @brief  Enable particle-particle self-collision repulsion for this body.
    ///         Prevents cloth from passing through itself and volumes from
    ///         self-intersecting.  Disabled by default for performance.
    bool enableSelfCollision = false;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SelfCollisionConfig
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Tuning parameters for soft body self-collision.
struct SelfCollisionConfig {
    /// @brief  Repulsion stiffness [0, 1].  Fraction of the overlap distance
    ///         corrected per constraint iteration.  Higher = faster separation
    ///         but can introduce jitter in dense particle clusters.
    float stiffness = 0.5f;

    /// @brief  Multiplier applied to `collisionMargin` to determine the
    ///         repulsion radius.  Two particles closer than
    ///         `repulsionRadiusMult * collisionMargin` are pushed apart.
    ///         Typical range: 1.5–3.0.  Lower = tighter cloth, higher = safer.
    float repulsionRadiusMult = 2.0f;

    /// @brief  Cell size for the spatial hash (world units).
    ///         Should be approximately equal to the repulsion radius.
    ///         Computed automatically from `repulsionRadiusMult * collisionMargin`
    ///         if set to 0.
    float spatialHashCellSize = 0.0f;
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SoftBody
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Deformable soft body composed of particles and constraints.
///
/// ## Storage Layout (SoA)
///
/// Particles are stored in Structure-of-Arrays fashion for cache efficiency.
/// The first two arrays (`positions`, `positionsPrev`) are SIMD-aligned for
/// fast vectorised Verlet integration.
///
/// ## Constraints
///
/// Three constraint types form a hierarchy:
///   - **Distance**  — pair springs (stretch/shear in cloth, edges in volume)
///   - **Bend**      — angular springs over triangle pairs (cloth folding resistance)
///   - **Volume**    — tetrahedral pressure preservation (incompressible solids)
///
/// ## Factory Methods
///
/// Use the static factory functions to construct common configurations:
/// @code
/// auto cloth  = SoftBody::createCloth(10, 10, 0.5f, 1.0f);
/// auto volume = SoftBody::createVolume(center, 2.0f, 1.0f);
/// auto rope   = SoftBody::createRope(start, end, 20, 0.5f);
/// @endcode
struct SoftBody {
    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Particle data (SoA)
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Current world-space particle positions.
    AlignedVector<Vec3f, 16> positions;

    /// @brief  Previous timestep positions (for Verlet integration).
    AlignedVector<Vec3f, 16> positionsPrev;

    /// @brief  Inverse mass (1/mass).  0 = pinned (infinite mass).
    AlignedVector<float, 16> inverseMasses;

    /// @brief  Per-particle pinned flag (1 = pinned, 0 = dynamic).
    ///         Stored as `char` for vector<bool>-free iteration.
    std::vector<char> pinned;

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Distance constraints  (spring edges)
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  A distance (spring) constraint between two particles.
    ///
    /// Projects particles `i` and `j` so that `||pos[i] - pos[j]|| = restLength`.
    /// This is the fundamental constraint for cloth, rope, and volumetric meshes.
    struct DistanceConstraint {
        std::uint32_t i;          ///< First particle index.
        std::uint32_t j;          ///< Second particle index.
        float         restLength; ///< Resting distance (set at construction).
    };
    std::vector<DistanceConstraint> distanceConstraints;

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Bend constraints  (angular preservation for cloth)
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  A bending constraint between two adjacent triangles (i-j-k and i-j-l).
    ///
    /// Preserves the dihedral angle along the shared edge (i, j).  Normally the
    /// angle should be near π (flat) for cloth; making `restAngle` smaller yields
    /// a naturally folded crease.
    struct BendConstraint {
        std::uint32_t i;         ///< Shared edge vertex A.
        std::uint32_t j;         ///< Shared edge vertex B.
        std::uint32_t k;         ///< Opposite vertex in triangle 1.
        std::uint32_t l;         ///< Opposite vertex in triangle 2.
        float         restAngle; ///< Resting dihedral angle (radians).
        float         stiffness; ///< Bend stiffness multiplier [0, 1].
    };
    std::vector<BendConstraint> bendConstraints;

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Volume constraints  (tetrahedral incompressibility)
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  A volume-preservation constraint over a tetrahedron (i, j, k, l).
    ///
    /// Projects all four particles to restore the rest volume.  Essential for
    /// squashy deformable solids that shouldn't collapse under pressure.
    struct VolumeConstraint {
        std::uint32_t i;          ///< Tetrahedron vertex A.
        std::uint32_t j;          ///< Tetrahedron vertex B.
        std::uint32_t k;          ///< Tetrahedron vertex C.
        std::uint32_t l;          ///< Tetrahedron vertex D.
        float         restVolume; ///< Resting volume (signed, positive for valid tets).
    };
    std::vector<VolumeConstraint> volumeConstraints;

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Simulation parameters
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Gravity applied to all particles each step.
    Vec3f gravity = Vec3f(0.0f, -9.81f, 0.0f);

    /// @brief  Velocity damping factor [0, 1).  0 = no damping.
    float damping = 0.01f;

    /// @brief  Friction coefficient for particle-rigid body sliding [0, ∞).
    float friction = 0.5f;

    /// @brief  Collision skin margin (metres).  Particles this far from a shape
    ///         surface are considered touching and will be projected out.
    float collisionMargin = 0.005f;

    /// @brief  Number of constraint-iteration passes per timestep.
    int constraintIterations = 5;

    /// @brief  Constraint stiffness [0, 1].  1.0 = infinitely stiff.
    float constraintStiffness = 0.8f;

    /// @brief  When true, the soft body is treated as cloth and can be rendered
    ///         as a triangle mesh.  Factory methods set this automatically.
    bool isCloth = false;

    /// @brief  Enable particle-particle self-collision repulsion.
    ///         When true, particles that come within `2 × collisionMargin`
    ///         of each other are pushed apart via the spatial-hash accelerated
    ///         self-collision solver.  Disabled by default.
    bool enableSelfCollision = false;

    /// @brief  Configuration for self-collision repulsion.
    ///         Only relevant when `enableSelfCollision == true`.
    SelfCollisionConfig selfCollisionConfig;

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Accessors
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Number of particles in this body.
    [[nodiscard]] std::size_t particleCount() const noexcept { return positions.size(); }

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Factory methods
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Create a rectangular cloth mesh with grid topology.
    ///
    /// The cloth lies in the XZ plane at y=0, with `width`×`height` vertices.
    /// Each quad is split into two triangles.  Distance constraints connect
    /// structural edges (horizontal, vertical) and shear edges (diagonal).
    /// Bend constraints connect every adjacent triangle pair.
    ///
    /// @param width    Number of vertices along the X axis (>= 2).
    /// @param height   Number of vertices along the Z axis (>= 2).
    /// @param spacing  Distance between adjacent vertices (metres).
    /// @param mass     Mass of each particle (0 = pinned; negative = auto-compute).
    ///                 If mass <= 0, a uniform mass is computed from total particle count.
    /// @return A fully-constructed SoftBody ready to pass to `PhysicsWorld::addSoftBody`.
    static SoftBody createCloth(int width, int height,
                                float spacing = 0.5f,
                                float mass = -1.0f) noexcept {
        SoftBody sb;
        sb.isCloth = true;

        int totalVerts = width * height;

        // ── Allocate particle storage ──────────────────────────────────────────────────
        sb.positions.reserve(totalVerts);
        sb.positionsPrev.reserve(totalVerts);
        sb.inverseMasses.reserve(totalVerts);
        sb.pinned.reserve(totalVerts);

        float halfW = (width - 1) * spacing * 0.5f;
        float halfH = (height - 1) * spacing * 0.5f;

        float pmass = mass > 0.0f ? mass : 1.0f / static_cast<float>(totalVerts);
        float invMass = 1.0f / pmass;

        for (int iz = 0; iz < height; ++iz) {
            for (int ix = 0; ix < width; ++ix) {
                Vec3f pos(static_cast<float>(ix) * spacing - halfW,
                          0.0f,
                          static_cast<float>(iz) * spacing - halfH);
                sb.positions.emplace_back(pos);
                sb.positionsPrev.emplace_back(pos);
                sb.inverseMasses.emplace_back(invMass);
                sb.pinned.emplace_back(0);
            }
        }

        // ── Helper: index from grid coordinates ────────────────────────────────────────
        auto idx = [width](int x, int z) { return static_cast<std::uint32_t>(z * width + x); };

        // ── Distance constraints (structural + shear) ──────────────────────────────────
        // Reserve capacity to avoid repeated reallocation.
        int structuralCount = (width - 1) * height + (height - 1) * width;
        int shearCount      = (width - 1) * (height - 1) * 2;
        sb.distanceConstraints.reserve(static_cast<std::size_t>(structuralCount + shearCount));

        for (int iz = 0; iz < height; ++iz) {
            for (int ix = 0; ix < width; ++ix) {
                if (ix < width - 1) {
                    std::uint32_t a = idx(ix, iz);
                    std::uint32_t b = idx(ix + 1, iz);
                    float rest = (sb.positions[a] - sb.positions[b]).length();
                    sb.distanceConstraints.push_back({a, b, rest});
                }
                if (iz < height - 1) {
                    std::uint32_t a = idx(ix, iz);
                    std::uint32_t b = idx(ix, iz + 1);
                    float rest = (sb.positions[a] - sb.positions[b]).length();
                    sb.distanceConstraints.push_back({a, b, rest});
                }
                // Shear (diagonal) constraints.
                if (ix < width - 1 && iz < height - 1) {
                    std::uint32_t a = idx(ix, iz);
                    std::uint32_t b = idx(ix + 1, iz + 1);
                    float rest = (sb.positions[a] - sb.positions[b]).length();
                    sb.distanceConstraints.push_back({a, b, rest});
                }
                if (ix > 0 && iz < height - 1) {
                    std::uint32_t a = idx(ix, iz);
                    std::uint32_t b = idx(ix - 1, iz + 1);
                    float rest = (sb.positions[a] - sb.positions[b]).length();
                    sb.distanceConstraints.push_back({a, b, rest});
                }
            }
        }

        // ── Bend constraints (cross-edge distance model) ────────────────────────────────
        //
        // For each pair of adjacent triangles sharing an edge, we add a distance constraint
        // between the two *non-shared* vertices.  This naturally resists out-of-plane bending
        // without complex dihedral-angle gradient computations.
        //
        // Each quad (i0,i1,i2,i3) with triangles T1=(i0,i1,i2), T2=(i1,i3,i2) shares edge
        // i1-i2.  The cross-edge constraint is between i0 and i3.  The other diagonal split
        // (i0,i2,i3) shares edge i0-i2 with cross-edge (i1,i3).  We create both for symmetry.
        //
        // This approach is used in many production cloth solvers (e.g. ObiCloth, AMD TressFX)
        // and avoids the numerical pitfalls of angle-based bending entirely.
        for (int iz = 0; iz < height - 1; ++iz) {
            for (int ix = 0; ix < width - 1; ++ix) {
                std::uint32_t i0 = idx(ix, iz);
                std::uint32_t i1 = idx(ix + 1, iz);
                std::uint32_t i2 = idx(ix, iz + 1);
                std::uint32_t i3 = idx(ix + 1, iz + 1);

                // Cross-edge across diagonal i1-i2 (triangles T1 and T2).
                float restA = (sb.positions[i0] - sb.positions[i3]).length();
                sb.distanceConstraints.push_back({i0, i3, restA});

                // Cross-edge across diagonal i0-i2 (triangles T1 and T3).
                float restB = (sb.positions[i1] - sb.positions[i2]).length();
                sb.distanceConstraints.push_back({i1, i2, restB});
            }
        }

        return sb;
    }

    /// @brief  Create a tetrahedral volumetric body approximating a sphere.
    ///
    /// Generates particles on a spherical shell with internal vertices, then
    /// connects them into tetrahedra.  Every tetrahedron gets a volume-preservation
    /// constraint.  This produces a squashy, incompressible ball.
    ///
    /// @param center  Centre of the soft volume (world space).
    /// @param radius  Radius of the spherical region.
    /// @param mass    Per-particle mass.  0 = pinned.
    /// @return A fully-constructed SoftBody.
    static SoftBody createVolume(const Vec3f& center, float radius,
                                 float mass = 1.0f) noexcept {
        SoftBody sb;
        sb.isCloth = false;

        // We generate a simple 8-particle hexahedron (cube) split into 5 tetrahedra.
        // This is the minimal deformable volume.  For production, use a proper
        // tetrahedralisation (e.g. TetGen) to generate the mesh.
        //
        // Vertices of a cube:
        //    7──────6
        //   /│     /│
        //  3──────2 │
        //  │ 4─────│5
        //  │/      │/
        //  0──────1
        //
        // Split into 5 tets: the standard minimal tetrahedralisation of a cube.

        const float h = radius * 0.577f;  // half-extent so vertex lies on sphere surface
        const Vec3f verts[8] = {
            center + Vec3f(-h, -h, -h),
            center + Vec3f( h, -h, -h),
            center + Vec3f( h,  h, -h),
            center + Vec3f(-h,  h, -h),
            center + Vec3f(-h, -h,  h),
            center + Vec3f( h, -h,  h),
            center + Vec3f( h,  h,  h),
            center + Vec3f(-h,  h,  h),
        };

        // Tetrahedralisation (5 tets for a cube).
        const std::uint32_t tets[5][4] = {
            {0, 1, 3, 4},
            {1, 2, 3, 6},
            {1, 4, 5, 6},
            {1, 3, 4, 6},
            {3, 4, 6, 7},
        };

        float invMass = mass > 0.0f ? 1.0f / mass : 1.0f;

        for (int i = 0; i < 8; ++i) {
            sb.positions.emplace_back(verts[i]);
            sb.positionsPrev.emplace_back(verts[i]);
            sb.inverseMasses.emplace_back(invMass);
            sb.pinned.emplace_back(0);
        }

        // ── Distance constraints (all 12 edges of the hexahedron) ──────────────────────
        const std::uint32_t edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom face
            {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top face
            {0, 4}, {1, 5}, {2, 6}, {3, 7},  // vertical edges
        };

        for (const auto& e : edges) {
            float rest = (verts[e[0]] - verts[e[1]]).length();
            sb.distanceConstraints.push_back({e[0], e[1], rest});
        }

        // ── Volume constraints ─────────────────────────────────────────────────────────
        for (const auto& tet : tets) {
            auto v = [&](int i) -> const Vec3f& { return verts[tet[i]]; };
            // Compute signed volume of tetrahedron.
            Vec3f a = v(1) - v(0);
            Vec3f b = v(2) - v(0);
            Vec3f c = v(3) - v(0);
            float vol = a.dot(b.cross(c)) / 6.0f;
            sb.volumeConstraints.push_back({tet[0], tet[1], tet[2], tet[3], vol});
        }

        return sb;
    }

    /// @brief  Create a simple rope (chain of particles with distance constraints).
    ///
    /// @param start     World-space start position.
    /// @param end       World-space end position.
    /// @param segments  Number of segments (particles = segments + 1).
    /// @param mass      Per-particle mass.  0 = pinned.
    /// @return A fully-constructed SoftBody.
    static SoftBody createRope(const Vec3f& start, const Vec3f& end,
                               int segments, float mass = 1.0f) noexcept {
        SoftBody sb;
        sb.isCloth = false;

        int particleCount = segments + 1;
        float invMass = mass > 0.0f ? 1.0f / mass : 1.0f;

        for (int i = 0; i < particleCount; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(segments);
            Vec3f pos = start + (end - start) * t;
            sb.positions.emplace_back(pos);
            sb.positionsPrev.emplace_back(pos);
            sb.inverseMasses.emplace_back(invMass);
            sb.pinned.emplace_back(0);
        }

        // Distance constraints between consecutive particles.
        for (int i = 0; i < segments; ++i) {
            float rest = (sb.positions[i] - sb.positions[i + 1]).length();
            sb.distanceConstraints.push_back({
                static_cast<std::uint32_t>(i),
                static_cast<std::uint32_t>(i + 1),
                rest
            });
        }

        return sb;
    }

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Utility: apply a user callback to pin/unpin particles
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// @brief  Pin a particle so it cannot move.
    /// @param index  Particle index.
    void pinParticle(std::size_t index) noexcept {
        if (index < pinned.size()) {
            pinned[index] = 1;
            inverseMasses[index] = 0.0f;
        }
    }

    /// @brief  Unpin a previously-pinned particle.
    /// @param index  Particle index.
    /// @param mass   New mass for the particle.
    void unpinParticle(std::size_t index, float mass = 1.0f) noexcept {
        if (index < pinned.size()) {
            pinned[index] = 0;
            inverseMasses[index] = mass > 0.0f ? 1.0f / mass : 1.0f;
        }
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// SoftBodySolver
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Solves soft body dynamics using Position-Based Dynamics (PBD).
///
/// ## Pipeline (per fixed step)
///
/// ```
/// for each soft body:
///   1. Verlet integrate     —  x' = x + (x - x_prev) + a * dt²
///   2. Project constraints  —  N iterations of distance → bend → volume
///   3. Collide with world   —  push particles out of rigid body shapes
///   4. Recover velocities   —  v = (x' - x) / dt
/// ```
///
/// ## Numerical Stability
///
/// PBD is unconditionally stable for any timestep, provided the constraint
/// projection is correct.  Large timesteps produce more compliance (softer
/// behaviour).  For stiff materials, increase `constraintIterations` rather
/// than reducing the timestep.
///
/// ## Thread Safety
///
/// The solver processes each soft body sequentially.  Soft bodies are
/// independent and could be parallelised per-body in a future task-graph
/// integration.
class SoftBodySolver {
public:
    /// @brief  Step a single soft body through one fixed timestep.
    ///
    /// @param dt                  Fixed timestep duration (seconds).
    /// @param body                Soft body to simulate (modified in-place).
    /// @param numRigidBodies      Number of rigid bodies in the world.
    /// @param bodyPositions       World positions of each rigid body (for shape transforms).
    /// @param bodyRotations       World rotations of each rigid body.
    /// @param shapeStart          Per-body sub-shape start indices.
    /// @param shapeCount          Per-body sub-shape counts.
    /// @param flatShapeHandles    Flat sub-shape handle array.
    /// @param flatShapeLocalPos   Flat sub-shape local positions.
    /// @param flatShapeLocalRot   Flat sub-shape local rotations.
    /// @param shapes              Shape registry (indexed by ShapeHandle).
    /// @param activeFlags         Active flags of each rigid body (vector).
    void step(float dt, SoftBody& body,
              std::size_t numRigidBodies,
              const Vec3f* bodyPositions,
              const Quaternionf* bodyRotations,
              const std::uint32_t* shapeStart,
              const std::uint32_t* shapeCount,
              const ShapeHandle* flatShapeHandles,
              const Vec3f* flatShapeLocalPositions,
              const Quaternionf* flatShapeLocalRotations,
              const Shape* shapes,
              const std::vector<bool>& activeFlags) const noexcept {

        if (body.particleCount() == 0) return;

        // ── Phase 1: Verlet integration ────────────────────────────────────────────────
        verletIntegrate(dt, body);

        // ── Phase 2: Project constraints ───────────────────────────────────────────────
        int iters = std::max(1, body.constraintIterations);
        for (int iter = 0; iter < iters; ++iter) {
            solveDistanceConstraints(body);
            solveBendConstraints(body);
            solveVolumeConstraints(body);

            // Collide particles with rigid body shapes.
            collideWithRigidBodies(body, numRigidBodies,
                                   bodyPositions, bodyRotations,
                                   shapeStart, shapeCount,
                                   flatShapeHandles,
                                   flatShapeLocalPositions,
                                   flatShapeLocalRotations,
                                   shapes, activeFlags);

            // Self-collision: particle-particle repulsion within this body.
            if (body.enableSelfCollision) {
                solveSelfCollision(body, body.selfCollisionConfig);
            }
        }

        // ── Phase 3b: Collide with other soft bodies (deferred) ───────────────────────

        // ── Phase 4: Recover velocities from position delta ────────────────────────────
        updateVelocities(dt, body);
    }

private:
    // ─── Phase 1: Verlet integration ────────────────────────────────────────────────────────────

    /// @brief  Integrate particle positions using the Verlet method.
    ///
    /// Verlet:  x_{t+dt} = 2*x_t - x_{t-dt} + a * dt²
    ///
    /// This is derived from a second-order Taylor expansion and is **symplectic** —
    /// it conserves energy in the absence of damping, making it ideal for spring
    /// networks.  Gravity enters as the acceleration term.  Damping is applied by
    /// scaling the velocity contribution.
    static void verletIntegrate(float dt, SoftBody& body) noexcept {
        float dtSq = dt * dt;
        Vec3f grav = body.gravity;

        for (std::size_t i = 0; i < body.particleCount(); ++i) {
            if (body.pinned[i]) continue;

            Vec3f pos = body.positions[i];
            Vec3f prev = body.positionsPrev[i];

            // Verlet with damping: x_new = pos + (1 - damp) * (pos - prev) + a * dt²
            Vec3f vel = pos - prev;
            Vec3f accel = grav;

            body.positionsPrev[i] = pos;
            body.positions[i] = pos + vel * (1.0f - body.damping) + accel * dtSq;
        }
    }

    // ─── Phase 2a: Distance constraint projection ──────────────────────────────────────────────

    /// @brief  Project all distance constraints (Gauss-Seidel).
    ///
    /// For each constraint:
    ///   1. Compute current distance between particles.
    ///   2. Compute correction delta = (dist - rest) / (w_i + w_j) * stiffness
    ///   3. Move each particle along the edge direction proportional to its inverse mass.
    static void solveDistanceConstraints(SoftBody& body) noexcept {
        float stiffness = body.constraintStiffness;
        for (const auto& c : body.distanceConstraints) {
            Vec3f& pA = body.positions[c.i];
            Vec3f& pB = body.positions[c.j];
            float wA = body.inverseMasses[c.i];
            float wB = body.inverseMasses[c.j];
            float totalW = wA + wB;
            if (totalW < 1e-12f) continue;

            Vec3f delta = pB - pA;
            float dist = delta.length();
            if (dist < 1e-12f) continue;

            Vec3f dir = delta / dist;
            float correction = (dist - c.restLength) * stiffness / totalW;

            pA += dir * (correction * wA);
            pB -= dir * (correction * wB);
        }
    }

    // ─── Phase 2b: Bend constraint projection ──────────────────────────────────────────────────

    /// @brief  Project bending (dihedral angle) constraints.
    ///
    /// Preserves the angle between two triangles sharing an edge.  The gradient
    /// of the dihedral angle w.r.t. each vertex position leads to the correction
    /// formula below (derived from Bridson et al.'s "Simulation of Clothing").
    static void solveBendConstraints(SoftBody& body) noexcept {
        for (const auto& c : body.bendConstraints) {
            Vec3f& p1 = body.positions[c.i];  // shared edge A
            Vec3f& p2 = body.positions[c.j];  // shared edge B
            Vec3f& p3 = body.positions[c.k];  // opposite in triangle 1
            Vec3f& p4 = body.positions[c.l];  // opposite in triangle 2

            float w1 = body.inverseMasses[c.i];
            float w2 = body.inverseMasses[c.j];
            float w3 = body.inverseMasses[c.k];
            float w4 = body.inverseMasses[c.l];

            // Sum of inverse masses for all four particles.
            float wSum = w1 + w2 + w3 + w4;
            if (wSum < 1e-12f) continue;

            Vec3f e = p2 - p1;
            float eLen = e.length();
            if (eLen < 1e-12f) continue;
            Vec3f eHat = e / eLen;

            Vec3f n1 = (p1 - p3).cross(p2 - p3);
            float n1LenSq = n1.lengthSquared();
            Vec3f n2 = (p2 - p4).cross(p1 - p4);
            float n2LenSq = n2.lengthSquared();

            if (n1LenSq < 1e-12f || n2LenSq < 1e-12f) continue;

            // Current dihedral angle.
            Vec3f n1Hat = n1 / std::sqrt(n1LenSq);
            Vec3f n2Hat = n2 / std::sqrt(n2LenSq);
            float d = n1Hat.dot(n2Hat);
            d = std::max(-1.0f, std::min(1.0f, d));
            float angle = std::acos(d);

            // Sign (check orientation).
            Vec3f crossN = n1Hat.cross(n2Hat);
            float sign = crossN.dot(eHat) >= 0.0f ? 1.0f : -1.0f;
            angle *= sign;

            float diff = angle - c.restAngle;
            if (std::abs(diff) < 1e-6f) continue;

            // Correction magnitude (simplified PBD gradient).
            // The factor (eLen / (n1LenSq + n2LenSq)) comes from the gradient derivation.
            float correction = diff * c.stiffness * body.constraintStiffness;
            float factor = correction * eLen / (n1LenSq + n2LenSq);

            // Compute gradients.
            Vec3f grad1 = eHat.cross(p3 - p2) / n1LenSq * factor;
            Vec3f grad2 = eHat.cross(p1 - p3) / n1LenSq * factor;
            Vec3f grad3 = eHat.cross(p2 - p1) / n1LenSq * factor;

            Vec3f g1 = -(n2 / n2LenSq).cross(eHat) * factor;
            Vec3f g2 = -(eHat).cross(n1 / n1LenSq) * (-factor);  // note sign
            Vec3f g4 = -(eHat).cross(p2 - p1) / n2LenSq * (-factor);

            // Normalise and apply.
            Vec3f q1 = (grad1 - g1);
            Vec3f q2 = (grad2 - g2);
            Vec3f q3 = grad3;
            Vec3f q4 = -g4;

            // Project
            p1 += q1 * (w1 / wSum);
            p2 += q2 * (w2 / wSum);
            p3 += q3 * (w3 / wSum);
            p4 += q4 * (w4 / wSum);
        }
    }

    // ─── Phase 2c: Volume constraint projection ────────────────────────────────────────────────

    /// @brief  Project tetrahedral volume constraints.
    ///
    /// Maintains the rest volume of each tetrahedron by moving all four vertices
    /// along the gradient of the volume function:
    ///   V = (1/6) * (p_j - p_i) · ((p_k - p_i) × (p_l - p_i))
    ///
    /// The gradient w.r.t. each vertex is given by the cross product of the
    /// two opposing edges.
    static void solveVolumeConstraints(SoftBody& body) noexcept {
        float stiffness = body.constraintStiffness;
        for (const auto& c : body.volumeConstraints) {
            Vec3f& p1 = body.positions[c.i];
            Vec3f& p2 = body.positions[c.j];
            Vec3f& p3 = body.positions[c.k];
            Vec3f& p4 = body.positions[c.l];

            float w1 = body.inverseMasses[c.i];
            float w2 = body.inverseMasses[c.j];
            float w3 = body.inverseMasses[c.k];
            float w4 = body.inverseMasses[c.l];

            Vec3f a = p2 - p1;
            Vec3f b = p3 - p1;
            Vec3f cVec = p4 - p1;

            float volume = a.dot(b.cross(cVec)) / 6.0f;
            float diff = volume - c.restVolume;
            if (std::abs(diff) < 1e-12f) continue;

            // Gradient of volume w.r.t. each vertex.
            Vec3f grad1 = (p3 - p2).cross(p4 - p2) / 6.0f;
            Vec3f grad2 = (p3 - p1).cross(p4 - p1) / 6.0f;
            Vec3f grad3 = (p4 - p1).cross(p2 - p1) / 6.0f;
            Vec3f grad4 = (p2 - p1).cross(p3 - p1) / 6.0f;

            float wSum = w1 * grad1.lengthSquared()
                       + w2 * grad2.lengthSquared()
                       + w3 * grad3.lengthSquared()
                       + w4 * grad4.lengthSquared();

            if (wSum < 1e-20f) continue;

            float lambda = -diff * stiffness / wSum;
            p1 += grad1 * (lambda * w1);
            p2 += grad2 * (lambda * w2);
            p3 += grad3 * (lambda * w3);
            p4 += grad4 * (lambda * w4);
        }
    }

    // ─── Phase 2d: Self-collision (particle-particle repulsion) ───────────────────────────────

    /// @brief  Spatial hash key for 3D cell coordinates.
    struct CellHashKey {
        int x, y, z;
        bool operator==(const CellHashKey& o) const noexcept {
            return x == o.x && y == o.y && z == o.z;
        }
    };

    struct CellHash {
        [[nodiscard]] std::size_t operator()(const CellHashKey& k) const noexcept {
            // Well-known spatial hashing: multiply each axis by a large prime
            // and XOR.  These specific primes (from Teschner et al. 2003)
            // produce good distribution for 3D grids.
            return static_cast<std::size_t>(
                static_cast<std::uint64_t>(k.x) * 73856093ULL ^
                static_cast<std::uint64_t>(k.y) * 19349663ULL ^
                static_cast<std::uint64_t>(k.z) * 83492791ULL);
        }
    };

    /// @brief  Resolve particle-particle repulsion within a soft body using a
    ///         spatial hash for O(n) expected-time neighbourhood queries.
    ///
    /// ## Algorithm
    ///
    ///   1. Build a spatial hash: partition particles into 3D grid cells of size
    ///      `repulsionRadius`.
    ///   2. Build a set of structural pairs (distance-constrained edges) to skip
    ///      — directly-connected particles should not repel each other, otherwise
    ///      the cloth fights its own stretch constraints.
    ///   3. For each particle, query its own cell and the 26 neighbouring cells.
    ///   4. For each candidate pair with index > self (to avoid double-testing),
    ///      compute the separation distance.  If less than `2 × repulsionRadius`,
    ///      push both particles apart weighted by inverse mass.
    ///
    /// ## Performance
    ///
    ///   - Spatial hash build: O(N) where N = particle count.
    ///   - Neighbour queries: O(9 × N × average_cell_density) which is nearly
    ///     linear for uniform particle distributions.
    ///   - Structural pair set: O(C) where C = distance constraint count.
    ///
    /// ## Numerical stability
    ///
    /// The repulsion is applied as a position correction inside the constraint
    /// loop (not as a force), ensuring it participates in the Gauss-Seidel
    /// projection.  This makes self-collision stable even at high stiffness
    /// values with large timesteps.
    static void solveSelfCollision(SoftBody& body,
                                   const SelfCollisionConfig& config) noexcept {
        std::size_t n = body.particleCount();
        if (n < 2) return;

        float margin = body.collisionMargin;
        float repulsionRadius = config.repulsionRadiusMult * margin;
        if (repulsionRadius <= 0.0f) return;

        float cellSize = config.spatialHashCellSize > 0.0f
                             ? config.spatialHashCellSize
                             : repulsionRadius * 2.0f;
        float invCellSize = 1.0f / cellSize;
        float repulsionRadiusSq = repulsionRadius * repulsionRadius;
        float stiffness = std::clamp(config.stiffness, 0.0f, 1.0f);

        // ── Step 1: Build spatial hash ──────────────────────────────────────────────────
        // Map from cell coordinate → list of particle indices in that cell.
        std::unordered_map<CellHashKey, std::vector<std::uint32_t>, CellHash> grid;
        grid.reserve(n / 4 + 1);

        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
            if (body.pinned[i]) continue;
            if (body.inverseMasses[i] <= 0.0f) continue;

            CellHashKey cell;
            cell.x = static_cast<int>(std::floor(body.positions[i].x * invCellSize));
            cell.y = static_cast<int>(std::floor(body.positions[i].y * invCellSize));
            cell.z = static_cast<int>(std::floor(body.positions[i].z * invCellSize));
            grid[cell].push_back(i);
        }

        // ── Step 2: Build structural pair skip set ───────────────────────────────────────
        // Encode each pair as a 64-bit key: ((uint64_t)min << 32) | max.
        // This prevents directly-connected particles from repelling each other
        // and fighting the distance constraints.
        std::unordered_set<std::uint64_t> structuralPairs;
        structuralPairs.reserve(body.distanceConstraints.size());
        for (const auto& dc : body.distanceConstraints) {
            std::uint32_t a = dc.i;
            std::uint32_t b = dc.j;
            if (a > b) std::swap(a, b);
            structuralPairs.insert((static_cast<std::uint64_t>(a) << 32) | b);
        }

        // ── Step 3: Query neighbours and apply repulsion ────────────────────────────────
        // For each particle, check its cell + 26 neighbours.
        // We only test i < j to avoid double-processing each pair.

        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
            if (body.pinned[i]) continue;
            if (body.inverseMasses[i] <= 0.0f) continue;

            float wI = body.inverseMasses[i];
            CellHashKey cell;
            cell.x = static_cast<int>(std::floor(body.positions[i].x * invCellSize));
            cell.y = static_cast<int>(std::floor(body.positions[i].y * invCellSize));
            cell.z = static_cast<int>(std::floor(body.positions[i].z * invCellSize));

            // Check 3×3×3 neighbourhood (27 cells).
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        CellHashKey neighbour{cell.x + dx, cell.y + dy, cell.z + dz};
                        auto it = grid.find(neighbour);
                        if (it == grid.end()) continue;

                        const auto& candidates = it->second;
                        for (std::uint32_t j : candidates) {
                            // Avoid self and only process i < j.
                            if (j <= i) continue;
                            if (body.pinned[j]) continue;
                            if (body.inverseMasses[j] <= 0.0f) continue;

                            // Skip structural pairs (distance-constrained edges).
                            std::uint64_t pairKey = (static_cast<std::uint64_t>(i) << 32) | j;
                            if (structuralPairs.count(pairKey)) continue;

                            // Compute separation.
                            Vec3f delta = body.positions[j] - body.positions[i];
                            float distSq = delta.lengthSquared();
                            if (distSq >= repulsionRadiusSq) continue;

                            float dist;
                            Vec3f dir;

                            if (distSq < 1e-20f) {
                                // Degenerate: particles at exactly the same position.
                                // Use an arbitrary separation direction based on
                                // particle index to break symmetry.
                                dist = 0.0f;
                                float angle = static_cast<float>(i * 3 + j * 7);
                                float ca = std::cos(angle), sa = std::sin(angle);
                                dir = Vec3f(ca, sa * 0.707f, sa * 0.707f);
                                dir = dir.normalized();
                            } else {
                                dist = std::sqrt(distSq);
                                dir = delta * (1.0f / dist);
                            }

                            float overlap = repulsionRadius - dist;
                            if (overlap <= 0.0f) continue;

                            // Weighted push: each particle moves inversely to its mass.
                            float wJ = body.inverseMasses[j];
                            float totalW = wI + wJ;
                            if (totalW <= 0.0f) continue;

                            float correction = overlap * stiffness;
                            Vec3f push = dir * correction;

                            body.positions[i] -= push * (wI / totalW);
                            body.positions[j] += push * (wJ / totalW);
                        }
                    }
                }
            }
        }
    }

    // ─── Phase 3: Particle-vs-rigid-body collision ─────────────────────────────────────────────

    /// @brief  Collide all particles of a soft body against all rigid body shapes.
    ///
    /// For each active rigid body, iterate its sub-shapes and project particles
    /// out of the shape volume.  Friction is applied by damping the velocity
    /// along the tangent direction (recovered later via position deltas).
    ///
    /// @note  This is the hot loop for soft body simulation.  At scale, use a
    ///        broadphase (spatial hash / Dbvt) to cull particle-vs-shape tests.
    static void collideWithRigidBodies(SoftBody& body,
                                       std::size_t numRigidBodies,
                                       const Vec3f* bodyPositions,
                                       const Quaternionf* bodyRotations,
                                       const std::uint32_t* shapeStart,
                                       const std::uint32_t* shapeCount,
                                       const ShapeHandle* flatShapeHandles,
                                       const Vec3f* flatShapeLocalPositions,
                                       const Quaternionf* flatShapeLocalRotations,
                                       const Shape* shapes,
                                       const std::vector<bool>& activeFlags) noexcept {
        float margin = body.collisionMargin;

        // For each particle, test against all rigid bodies.
        for (std::size_t pIdx = 0; pIdx < body.particleCount(); ++pIdx) {
            if (body.pinned[pIdx]) continue;
            if (body.inverseMasses[pIdx] <= 0.0f) continue;

            Vec3f& pos = body.positions[pIdx];

            for (std::size_t rb = 0; rb < numRigidBodies; ++rb) {
                if (rb >= activeFlags.size() || !activeFlags[rb]) continue;

                std::uint32_t sStart = shapeStart[rb];
                std::uint32_t sCount = shapeCount[rb];

                for (std::uint32_t s = 0; s < sCount; ++s) {
                    ShapeHandle sh = flatShapeHandles[sStart + s];
                    if (sh >= std::numeric_limits<std::uint32_t>::max()) continue;

                    const Shape& shape = shapes[sh];

                    // Compute world-space sub-shape transform.
                    Vec3f localPos = flatShapeLocalPositions[sStart + s];
                    Quaternionf localRot = flatShapeLocalRotations[sStart + s];
                    Vec3f shapeWorldPos = bodyPositions[rb]
                                        + bodyRotations[rb].rotate(localPos);
                    Quaternionf shapeWorldRot = bodyRotations[rb] * localRot;

                    // Dispatch based on shape type.
                    Vec3f correction = Vec3f::zero();
                    Vec3f normal = Vec3f::zero();
                    bool hit = false;

                    switch (shape.type) {
                    case ShapeType::Sphere:
                        hit = collidePointSphere(pos, shape.sphere,
                                                  shapeWorldPos, margin,
                                                  correction, normal);
                        break;
                    case ShapeType::Box:
                        hit = collidePointBox(pos, shape.box,
                                               shapeWorldPos, shapeWorldRot,
                                               margin, correction, normal);
                        break;
                    case ShapeType::Plane:
                        hit = collidePointPlane(pos, shape.plane,
                                                 shapeWorldPos, shapeWorldRot,
                                                 margin, correction, normal);
                        break;
                    case ShapeType::Capsule:
                        hit = collidePointCapsule(pos, shape.capsule,
                                                   shapeWorldPos, shapeWorldRot,
                                                   margin, correction, normal);
                        break;
                    case ShapeType::ConvexMesh:
                    case ShapeType::TriangleMesh:
                        // Conservative: approximate convex mesh as a sphere
                        // with the bounding radius.  Triangle mesh collision
                        // is deferred to a future PBD-rigid coupling pass.
                        {
                            float br = shape.boundingRadius();
                            Sphere approxSphere{br};
                            hit = collidePointSphere(pos, approxSphere,
                                                      shapeWorldPos, margin,
                                                      correction, normal);
                        }
                        break;
                    }

                    if (hit && correction.lengthSquared() > 0.0f) {
                        // Apply correction immediately.
                        pos += correction;

                        // Apply friction: project the implicit velocity
                        // (pos - prev) onto the tangent plane.
                        Vec3f vel = pos - body.positionsPrev[pIdx];
                        float vn = vel.dot(normal);
                        if (vn < 0.0f) {
                            // Particle is moving into the surface.
                            Vec3f vt = vel - normal * vn;
                            // Damp tangent velocity by friction coefficient.
                            body.positionsPrev[pIdx] = pos - (normal * vn + vt * (1.0f - body.friction));
                        }
                    }
                }
            }
        }
    }

    /// @brief  Collide a point against a sphere shape.
    /// @return True if the point is inside the sphere (with margin).
    static bool collidePointSphere(const Vec3f& point, const Sphere& sphere,
                                   const Vec3f& shapeWorldPos, float margin,
                                   Vec3f& outCorrection, Vec3f& outNormal) noexcept {
        Vec3f delta = point - shapeWorldPos;
        float distSq = delta.lengthSquared();
        float radius = sphere.radius + margin;
        if (distSq >= radius * radius) return false;

        float dist = std::sqrt(distSq);
        if (dist < 1e-12f) {
            // Point is exactly at centre — push radially outward.
            outNormal = Vec3f(0.0f, 1.0f, 0.0f);
            outCorrection = outNormal * radius;
            return true;
        }

        outNormal = delta / dist;
        float pen = radius - dist;
        outCorrection = outNormal * pen;
        return true;
    }

    /// @brief  Collide a point against a plane shape.
    /// @return True if the point is on the negative side of the plane (with margin).
    static bool collidePointPlane(const Vec3f& point, const Plane& plane,
                                  const Vec3f& shapeWorldPos,
                                  const Quaternionf& shapeWorldRot,
                                  float margin,
                                  Vec3f& outCorrection, Vec3f& outNormal) noexcept {
        Vec3f planeNormal = shapeWorldRot.rotate(plane.normal).normalized();
        float d = plane.distance;
        // The world-space plane equation: n · (p - shapePos) = d
        //   → n · p - n · shapePos = d
        float worldD = d + planeNormal.dot(shapeWorldPos);
        float signedDist = planeNormal.dot(point) - worldD;

        if (signedDist > margin) return false;

        outNormal = planeNormal;
        outCorrection = outNormal * (margin - signedDist);
        return true;
    }

    /// @brief  Collide a point against a box shape.
    /// @return True if the point is inside the box (with margin).
    static bool collidePointBox(const Vec3f& point, const Box& box,
                                const Vec3f& shapeWorldPos,
                                const Quaternionf& shapeWorldRot,
                                float margin,
                                Vec3f& outCorrection, Vec3f& outNormal) noexcept {
        // Transform point into box local space.
        Vec3f localP = shapeWorldRot.conjugate().rotate(point - shapeWorldPos);
        Vec3f he = box.halfExtents + Vec3f(margin);

        // Check if point is inside the box.
        float overlapX = he.x - std::abs(localP.x);
        float overlapY = he.y - std::abs(localP.y);
        float overlapZ = he.z - std::abs(localP.z);

        if (overlapX < 0.0f || overlapY < 0.0f || overlapZ < 0.0f) return false;

        // Find the axis with smallest overlap (push along this axis).
        Vec3f localNormal;
        float minOverlap = overlapX;
        localNormal = Vec3f(localP.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
        if (overlapY < minOverlap) {
            minOverlap = overlapY;
            localNormal = Vec3f(0.0f, localP.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
        }
        if (overlapZ < minOverlap) {
            minOverlap = overlapZ;
            localNormal = Vec3f(0.0f, 0.0f, localP.z >= 0.0f ? 1.0f : -1.0f);
        }

        outNormal = shapeWorldRot.rotate(localNormal).normalized();
        outCorrection = outNormal * minOverlap;
        return true;
    }

    /// @brief  Collide a point against a capsule shape.
    /// @return True if the point is inside the capsule (with margin).
    static bool collidePointCapsule(const Vec3f& point, const Capsule& capsule,
                                    const Vec3f& shapeWorldPos,
                                    const Quaternionf& shapeWorldRot,
                                    float margin,
                                    Vec3f& outCorrection, Vec3f& outNormal) noexcept {
        // Transform point into capsule local space.
        Vec3f localP = shapeWorldRot.conjugate().rotate(point - shapeWorldPos);

        // Capsule is a segment along Y axis from -halfHeight to +halfHeight,
        // with a sphere of `radius` around the segment.
        float halfH = capsule.halfHeight;
        float rad = capsule.radius + margin;

        // Clamp point Y to segment range, then compute distance from the segment.
        // The capsule segment is along the local Y axis from -halfHeight to +halfHeight.
        float clampedY = std::max(-halfH, std::min(halfH, localP.y));
        Vec3f delta(
            localP.x,
            localP.y - clampedY,
            localP.z
        );

        float distSq = delta.lengthSquared();
        if (distSq >= rad * rad) return false;

        float dist = std::sqrt(distSq);
        if (dist < 1e-12f) {
            // Point is exactly on segment line.
            Vec3f localNormal(0.0f, localP.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
            outNormal = shapeWorldRot.rotate(localNormal).normalized();
            outCorrection = outNormal * rad;
            return true;
        }

        Vec3f localNormal = delta / dist;
        float pen = rad - dist;
        outNormal = shapeWorldRot.rotate(localNormal).normalized();
        outCorrection = outNormal * pen;
        return true;
    }

    // ─── Phase 4: Velocity recovery ────────────────────────────────────────────────────────────

    /// @brief  Recover velocities from the Verlet position delta.
    ///
    /// v = (position - positionPrev) / dt
    ///
    /// This produces a velocity consistent with the new positions, which is
    /// essential for any external systems reading velocities (e.g. rendering,
    /// two-way coupling).
    static void updateVelocities(float dt, SoftBody& body) noexcept {
        // Velocities are implicit in the Verlet scheme (v = (x - x_prev) / dt).
        // They are recovered on-the-fly by readers that need them.  The prev
        // positions array already reflects the updated state from this step's
        // Verlet integration.
        (void)dt; (void)body;
    }
};

} // namespace ciphyxs
