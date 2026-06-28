//==================================================================================================
/// @file  Shape.hpp
/// @brief  Collision primitives (Sphere, Box, Plane, Capsule), AABB, and type-erased shape storage.
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include "../math/Mat3.hpp"
#include "../core/RigidBody.hpp"
#include <cstdint>
#include <limits>
#include <cstddef>

namespace ciphyxs { struct Bvh; }

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// ShapeType
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  The concrete primitive type stored in a Shape union.
enum class ShapeType : std::uint8_t {
    Sphere,    ///< Sphere defined by radius.
    Box,       ///< Oriented box defined by half-extents.
    Plane,     ///< Infinite plane:  n · p = d  (d = signed distance from origin).
    Capsule,   ///< Capsule defined by radius and half-height (Y-axis).
    ConvexMesh,///< Convex hull defined by a set of vertices (GJK/EPA collision).
    TriangleMesh ///< Triangle mesh with BVH acceleration (static/deformable geometry).
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Shape primitives
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Sphere collision primitive.
struct Sphere {
    float radius = 0.5f;
};

/// @brief  Axis-aligned box primitive (orientation provided by the body's transform).
struct Box {
    Vec3f halfExtents = Vec3f(0.5f, 0.5f, 0.5f);
};

/// @brief  Infinite plane primitive:  `normal · p = distance`.
struct Plane {
    Vec3f normal   = Vec3f::unitY();
    float distance = 0.0f;
};

/// @brief  Capsule primitive aligned to the local Y axis.
///
/// The capsule is defined by a line segment from `(0, -halfHeight, 0)` to
/// `(0, +halfHeight, 0)` with `radius` around it.  Body rotation reorients
/// the capsule in world space.
struct Capsule {
    float radius      = 0.5f;  ///< Radius of the capsule (>= 0).
    float halfHeight  = 0.5f;  ///< Half-length of the medial segment.
};

/// @brief  Convex hull primitive defined by a set of vertices.
///
/// The hull is the convex hull of the provided vertices.  Vertices are stored
/// as a pointer to an external array (owned by the caller or by the PhysicsWorld's
/// shape registry).  The hull is assumed to be in **local space** and is transformed
/// by the body's position and rotation.
///
/// For collision detection, GJK (Gilbert-Johnson-Keerthi) + EPA (Expanding Polytope
/// Algorithm) is used, which supports any convex-convex pair natively.
///
/// @note  The vertex pointer must remain valid for the lifetime of the shape.
struct ConvexMesh {
    const Vec3f* vertices     = nullptr; ///< Vertex array (local space).
    int          vertexCount  = 0;       ///< Number of vertices.
    Vec3f        halfExtents  = Vec3f::zero(); ///< Local-space AABB half-extents.
    Vec3f        center       = Vec3f::zero(); ///< Geometric centre of the hull.
};

/// @brief  Triangle mesh primitive.
///
/// Stores a set of triangles defined by vertex and index arrays.  A BVH (Bounding
/// Volume Hierarchy) is used to accelerate ray casting and collision queries.
///
/// @note  The vertex, index, and BVH pointers must remain valid for the lifetime
///        of the shape (caller's responsibility).
struct TriangleMesh {
    const Vec3f* vertices        = nullptr; ///< Vertex array (local space).
    const int*   indices         = nullptr; ///< Triangle index array (3 per triangle).
    int          vertexCount     = 0;       ///< Number of vertices.
    int          triangleCount   = 0;       ///< Number of triangles.
    Vec3f        halfExtents     = Vec3f::zero(); ///< Local-space AABB half-extents.
    Vec3f        center          = Vec3f::zero(); ///< Geometric centre.
    const class Bvh* bvh        = nullptr; ///< BVH for acceleration (optional, externally managed).
};


// AABB  (Axis-Aligned Bounding Box)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Axis-aligned bounding box for broad-phase culling.
struct AABB {
    Vec3f min = Vec3f( std::numeric_limits<float>::max());
    Vec3f max = Vec3f(-std::numeric_limits<float>::max());

    /// @brief  Grow the box to contain `point`.
    void expand(const Vec3f& point) noexcept {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }

    /// @brief  True when this box overlaps `other`.
    [[nodiscard]] bool overlaps(const AABB& other) const noexcept {
        if (max.x < other.min.x || min.x > other.max.x) return false;
        if (max.y < other.min.y || min.y > other.max.y) return false;
        if (max.z < other.min.z || min.z > other.max.z) return false;
        return true;
    }

    /// @brief  Centre of the box.
    [[nodiscard]] Vec3f center() const noexcept { return (min + max) * 0.5f; }

    /// @brief  Half-extents along each axis.
    [[nodiscard]] Vec3f extents() const noexcept { return (max - min) * 0.5f; }

    /// @brief  Return the union of this AABB and `other` (no modification).
    [[nodiscard]] AABB united(const AABB& other) const noexcept {
        AABB r;
        r.min.x = std::min(min.x, other.min.x);
        r.min.y = std::min(min.y, other.min.y);
        r.min.z = std::min(min.z, other.min.z);
        r.max.x = std::max(max.x, other.max.x);
        r.max.y = std::max(max.y, other.max.y);
        r.max.z = std::max(max.z, other.max.z);
        return r;
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Shape  (type-erased wrapper)
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Type-erased shape that owns one primitive in a tightly-packed union.
///
/// Access the concrete type through the `type` discriminator and the corresponding member:
/// @code
/// if (shape.type == ShapeType::Sphere) { use(shape.sphere); }
/// @endcode
struct Shape {
    ShapeType type = ShapeType::Sphere;

    union {
            Sphere    sphere;
            Box       box;
            Plane     plane;
            Capsule   capsule;
            ConvexMesh convexMesh;
            TriangleMesh triangleMesh;
        };

        Shape() noexcept : type(ShapeType::Sphere), sphere() {}
        Shape(const Sphere& s)     noexcept : type(ShapeType::Sphere),     sphere(s)      {}
        Shape(const Box& b)        noexcept : type(ShapeType::Box),        box(b)         {}
        Shape(const Plane& p)      noexcept : type(ShapeType::Plane),      plane(p)       {}
        Shape(const Capsule& c)    noexcept : type(ShapeType::Capsule),    capsule(c)     {}
        Shape(const ConvexMesh& m) noexcept : type(ShapeType::ConvexMesh), convexMesh(m)  {}
        Shape(const TriangleMesh& m) noexcept : type(ShapeType::TriangleMesh), triangleMesh(m) {}

    /// @brief  Compute the world-space AABB for this shape at a given transform.
    [[nodiscard]] AABB getAABB(const Vec3f& position, const Quaternionf& rotation) const noexcept {
        switch (type) {
        case ShapeType::Sphere: {
            Vec3f r(sphere.radius);
            return AABB{position - r, position + r};
        }
        case ShapeType::Box: {
            // Efficient box AABB: rotate half-extents into world and sum absolute axes.
            Mat3f rot = rotation.toMat3();
            Vec3f he = box.halfExtents;
            Vec3f worldHe(
                std::abs(rot(0,0)) * he.x + std::abs(rot(0,1)) * he.y + std::abs(rot(0,2)) * he.z,
                std::abs(rot(1,0)) * he.x + std::abs(rot(1,1)) * he.y + std::abs(rot(1,2)) * he.z,
                std::abs(rot(2,0)) * he.x + std::abs(rot(2,1)) * he.y + std::abs(rot(2,2)) * he.z
            );
            return AABB{position - worldHe, position + worldHe};
        }
        case ShapeType::Plane:
            // Infinite – return a sentinel large enough for any practical scene.
            return AABB{
                Vec3f(-1e10f),
                Vec3f( 1e10f)
            };
        case ShapeType::Capsule: {
            // Capsule AABB: sphere of radius `radius` swept along the Y segment.
            // World-space half-height = halfHeight along rotated Y axis.
            Vec3f localEnd(0.0f, capsule.halfHeight, 0.0f);
            Vec3f worldEnd = rotation.rotate(localEnd);
            Vec3f absEnd(std::abs(worldEnd.x), std::abs(worldEnd.y), std::abs(worldEnd.z));
            Vec3f r(capsule.radius);
            return AABB{position - absEnd - r, position + absEnd + r};
        }
        case ShapeType::ConvexMesh: {
            // Transform local AABB half-extents into world space.
            Mat3f rot = rotation.toMat3();
            Vec3f he = convexMesh.halfExtents;
            Vec3f worldHe(
                std::abs(rot(0,0)) * he.x + std::abs(rot(0,1)) * he.y + std::abs(rot(0,2)) * he.z,
                std::abs(rot(1,0)) * he.x + std::abs(rot(1,1)) * he.y + std::abs(rot(1,2)) * he.z,
                std::abs(rot(2,0)) * he.x + std::abs(rot(2,1)) * he.y + std::abs(rot(2,2)) * he.z
            );
            Vec3f worldCenter = position + rotation.rotate(convexMesh.center);
            return AABB{worldCenter - worldHe, worldCenter + worldHe};
        }
        case ShapeType::TriangleMesh: {
            // Transform local AABB half-extents into world space (same as ConvexMesh).
            Mat3f rot = rotation.toMat3();
            Vec3f he = triangleMesh.halfExtents;
            Vec3f worldHe(
                std::abs(rot(0,0)) * he.x + std::abs(rot(0,1)) * he.y + std::abs(rot(0,2)) * he.z,
                std::abs(rot(1,0)) * he.x + std::abs(rot(1,1)) * he.y + std::abs(rot(1,2)) * he.z,
                std::abs(rot(2,0)) * he.x + std::abs(rot(2,1)) * he.y + std::abs(rot(2,2)) * he.z
            );
            Vec3f worldCenter = position + rotation.rotate(triangleMesh.center);
            return AABB{worldCenter - worldHe, worldCenter + worldHe};
        }
        }
        return AABB{};
    }

    /// @brief  World-space bounding-sphere radius approximation (for broadphase culling).
    [[nodiscard]] float boundingRadius() const noexcept {
        switch (type) {
        case ShapeType::Sphere: return sphere.radius;
        case ShapeType::Box:    return box.halfExtents.length();
        case ShapeType::Plane:  return 1e10f;
        case ShapeType::Capsule: return capsule.halfHeight + capsule.radius;
        case ShapeType::ConvexMesh: return convexMesh.halfExtents.length();
        case ShapeType::TriangleMesh: return triangleMesh.halfExtents.length();
        }
        return 0.0f;
    }

    /// @brief  Compute the diagonal of the inertia tensor in local space for a given mass.
    ///
    /// Uses the standard closed-form formulas for each primitive:
    ///   - **Sphere**:  I = (2/5) · m · r² · diag(1, 1, 1)
    ///   - **Box**:     I_xx = (m/3) · (h² + d²)   where h = halfHeight, d = halfDepth
    ///                  I_yy = (m/3) · (w² + d²)   where w = halfWidth
    ///                  I_zz = (m/3) · (w² + h²)
    ///   - **Plane**:   Returns (1, 1, 1) — static bodies ignore inertia.
    ///
    /// @param mass  The mass of the body (must be > 0 for meaningful results).
    /// @return  The diagonal of the inertia tensor (local space).
    [[nodiscard]] Vec3f computeInertia(float mass) const noexcept {
        switch (type) {
        case ShapeType::Sphere: {
            float r = sphere.radius;
            float i = 0.4f * mass * r * r;  // 2/5 = 0.4
            return Vec3f(i, i, i);
        }
        case ShapeType::Box: {
            Vec3f he = box.halfExtents;
            // Inertia of a solid box with full dimensions (2a, 2b, 2c):
            //   Ixx = m/12 * ((2b)² + (2c)²) = m/3 * (b² + c²)
            float ix = (1.0f / 3.0f) * mass * (he.y * he.y + he.z * he.z);
            float iy = (1.0f / 3.0f) * mass * (he.x * he.x + he.z * he.z);
            float iz = (1.0f / 3.0f) * mass * (he.x * he.x + he.y * he.y);
            // Guard: ensure all axes are at least a small positive value.
            float eps = 1e-8f;
            return Vec3f(std::max(ix, eps), std::max(iy, eps), std::max(iz, eps));
        }
        case ShapeType::Plane:
            break;
        case ShapeType::Capsule: {
            // Inertia of a solid capsule (approximated as cylinder + hemispheres):
            //   I_yy (axial)    = m * r²
            //   I_xx = I_zz = m/12 * (3*r² + 4*h²) + m*r²
            // where h = halfHeight, r = radius.
            float r = capsule.radius;
            float h = capsule.halfHeight;
            float r2 = r * r;
            float h2 = h * h;
            float mI = mass / 12.0f;
            // Axial inertia (Y axis — capsule axis).
            float iy = mass * r2;
            // Transverse inertia.
            float ixz = mI * (3.0f * r2 + 4.0f * h2) + mass * r2;
            float eps = 1e-8f;
            return Vec3f(std::max(ixz, eps), std::max(iy, eps), std::max(ixz, eps));
        }
        case ShapeType::ConvexMesh: {
            // Approximate inertia from AABB half-extents (conservative).
            // Treat the hull as a solid box with the same bounding-box dimensions.
            Vec3f he = convexMesh.halfExtents;
            float ix = (1.0f / 3.0f) * mass * (he.y * he.y + he.z * he.z);
            float iy = (1.0f / 3.0f) * mass * (he.x * he.x + he.z * he.z);
            float iz = (1.0f / 3.0f) * mass * (he.x * he.x + he.y * he.y);
            float eps = 1e-8f;
            return Vec3f(std::max(ix, eps), std::max(iy, eps), std::max(iz, eps));
        }
        case ShapeType::TriangleMesh: {
            // Approximate inertia from AABB half-extents (solid box approximation).
            Vec3f he = triangleMesh.halfExtents;
            float ix = (1.0f / 3.0f) * mass * (he.y * he.y + he.z * he.z);
            float iy = (1.0f / 3.0f) * mass * (he.x * he.x + he.z * he.z);
            float iz = (1.0f / 3.0f) * mass * (he.x * he.x + he.y * he.y);
            float eps = 1e-8f;
            return Vec3f(std::max(ix, eps), std::max(iy, eps), std::max(iz, eps));
        }
        }
        return Vec3f(1.0f, 1.0f, 1.0f);
    }
};

} // namespace ciphyxs
