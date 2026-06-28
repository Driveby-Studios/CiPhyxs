//==================================================================================================
/// @file  DebugRenderer.hpp
/// @brief  Renderer-agnostic debug visualization interface for CiPhyxs.
///
/// Implement this interface to draw the physics world state (bodies, contacts, joints,
/// AABBs, axes) in any rendering system — OpenGL, DirectX, Vulkan, debug overlay,
/// or a file dump.
///
/// ## Usage
///
/// @code
///   class MyDebugDraw : public ciphyxs::IDebugRenderer {
///       void drawLine(Vec3f from, Vec3f to, Color c) override { /* GL calls */ }
///       void drawTriangle(Vec3f a, Vec3f b, Vec3f c, Color clr) override { /* ... */ }
///   };
///
///   MyDebugDraw renderer;
///   world.debugDraw(&renderer);
/// @endcode
//==================================================================================================
#pragma once

#include "../math/Vec3.hpp"
#include "../math/Quaternion.hpp"
#include "../collision/Shape.hpp"
#include "RigidBody.hpp"
#include <cstdint>
#include <cmath>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Color
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Simple RGBA colour for debug visualization.
struct Color {
    float r = 1.0f; ///< Red   [0, 1]
    float g = 1.0f; ///< Green [0, 1]
    float b = 1.0f; ///< Blue  [0, 1]
    float a = 1.0f; ///< Alpha [0, 1]

    constexpr Color() noexcept = default;
    constexpr Color(float r_, float g_, float b_, float a_ = 1.0f) noexcept
        : r(r_), g(g_), b(b_), a(a_) {}

    static constexpr Color white()   noexcept { return {1.0f, 1.0f, 1.0f}; }
    static constexpr Color red()     noexcept { return {1.0f, 0.0f, 0.0f}; }
    static constexpr Color green()   noexcept { return {0.0f, 1.0f, 0.0f}; }
    static constexpr Color blue()    noexcept { return {0.0f, 0.0f, 1.0f}; }
    static constexpr Color yellow()  noexcept { return {1.0f, 1.0f, 0.0f}; }
    static constexpr Color cyan()    noexcept { return {0.0f, 1.0f, 1.0f}; }
    static constexpr Color magenta() noexcept { return {1.0f, 0.0f, 1.0f}; }
    static constexpr Color grey()    noexcept { return {0.5f, 0.5f, 0.5f}; }
    static constexpr Color black()   noexcept { return {0.0f, 0.0f, 0.0f}; }
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// IDebugRenderer
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Abstract interface for debug drawing.
///
/// Implement this interface and pass it to `PhysicsWorld::debugDraw()` to visualise
/// the physics simulation state.  All coordinates are in world space.
struct IDebugRenderer {
    virtual ~IDebugRenderer() = default;

    /// @brief  Draw a line segment from `from` to `to`.
    virtual void drawLine(const Vec3f& from, const Vec3f& to, const Color& color) noexcept = 0;

    /// @brief  Draw a filled triangle.
    virtual void drawTriangle(const Vec3f& a, const Vec3f& b, const Vec3f& c,
                              const Color& color) noexcept = 0;

    /// @brief  Draw a point (small cross or dot).
    virtual void drawPoint(const Vec3f& pos, const Color& color) noexcept = 0;

    /// @brief  Draw a 3D arrow from `origin` in `direction` with `length`.
    virtual void drawArrow(const Vec3f& origin, const Vec3f& direction, float length,
                           const Color& color) noexcept = 0;

    /// @brief  Begin a new frame.  Called once per `debugDraw()` invocation.
    virtual void beginFrame() noexcept {}

    /// @brief  End the current frame.  Called after all draw calls for the frame.
    virtual void endFrame() noexcept {}
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
//  Debug drawing utilities  (used by PhysicsWorld::debugDraw)
// ────────────────────────────────────────────────────────────────────────────────────────────────

namespace debug_detail {

/// @brief  Draw an oriented box as 12 line segments.
inline void drawBox(IDebugRenderer& r, const Vec3f& pos, const Quaternionf& rot,
                    const Vec3f& halfExtents, const Color& color) noexcept {
    Vec3f corners[8];
    Mat3f m = rot.toMat3();
    Vec3f ax = m.col(0) * halfExtents.x;
    Vec3f ay = m.col(1) * halfExtents.y;
    Vec3f az = m.col(2) * halfExtents.z;

    corners[0] = pos - ax - ay - az;
    corners[1] = pos + ax - ay - az;
    corners[2] = pos - ax + ay - az;
    corners[3] = pos + ax + ay - az;
    corners[4] = pos - ax - ay + az;
    corners[5] = pos + ax - ay + az;
    corners[6] = pos - ax + ay + az;
    corners[7] = pos + ax + ay + az;

    // Bottom face.
    r.drawLine(corners[0], corners[1], color);
    r.drawLine(corners[1], corners[3], color);
    r.drawLine(corners[3], corners[2], color);
    r.drawLine(corners[2], corners[0], color);

    // Top face.
    r.drawLine(corners[4], corners[5], color);
    r.drawLine(corners[5], corners[7], color);
    r.drawLine(corners[7], corners[6], color);
    r.drawLine(corners[6], corners[4], color);

    // Pillars.
    r.drawLine(corners[0], corners[4], color);
    r.drawLine(corners[1], corners[5], color);
    r.drawLine(corners[2], corners[6], color);
    r.drawLine(corners[3], corners[7], color);
}

/// @brief  Draw a sphere as a wireframe circle in the XY, XZ, and YZ planes.
inline void drawSphere(IDebugRenderer& r, const Vec3f& pos, float radius,
                       const Color& color) noexcept {
    constexpr int kSegments = 16;
    auto ring = [&](const Vec3f& u, const Vec3f& v) {
        Vec3f prev = pos + u * radius;
        for (int i = 1; i <= kSegments; ++i) {
            float a = 2.0f * 3.14159265f * float(i) / float(kSegments);
            Vec3f cur = pos + (u * std::cos(a) + v * std::sin(a)) * radius;
            r.drawLine(prev, cur, color);
            prev = cur;
        }
    };
    ring(Vec3f::unitX(), Vec3f::unitY());
    ring(Vec3f::unitX(), Vec3f::unitZ());
    ring(Vec3f::unitY(), Vec3f::unitZ());
}

/// @brief  Draw a capsule as a wireframe cylinder with hemispherical caps.
inline void drawCapsule(IDebugRenderer& r, const Vec3f& pos, const Quaternionf& rot,
                        float radius, float halfHeight, const Color& color) noexcept {
    // Capsule axis is Y in local space.
    Vec3f localA(0.0f, -halfHeight, 0.0f);
    Vec3f localB(0.0f, +halfHeight, 0.0f);
    Vec3f worldA = pos + rot.rotate(localA);
    Vec3f worldB = pos + rot.rotate(localB);

    // Cylinder body (4 lines along axis).
    constexpr int kSeg = 12;
    for (int i = 0; i < kSeg; ++i) {
        float a0 = 2.0f * 3.14159265f * float(i) / float(kSeg);
        float a1 = 2.0f * 3.14159265f * float(i + 1) / float(kSeg);
        Vec3f dir0(0.0f, 0.0f, 0.0f);
        Vec3f dir1(0.0f, 0.0f, 0.0f);
        dir0.x = std::cos(a0); dir0.z = std::sin(a0);
        dir1.x = std::cos(a1); dir1.z = std::sin(a1);

        Vec3f wDir0 = rot.rotate(dir0);
        Vec3f wDir1 = rot.rotate(dir1);

        r.drawLine(worldA + wDir0 * radius, worldA + wDir1 * radius, color);
        r.drawLine(worldB + wDir0 * radius, worldB + wDir1 * radius, color);
        r.drawLine(worldA + wDir0 * radius, worldB + wDir0 * radius, color);
    }

    // Hemisphere caps approximated as fan rings.
    auto drawCap = [&](const Vec3f& centre, const Vec3f& axisDir) {
        for (int ring = 1; ring <= 4; ++ring) {
            float phi = 3.14159265f * 0.5f * float(ring) / 4.0f;
            float rRing = radius * std::cos(phi);
            float zOff  = radius * std::sin(phi);
            Vec3f prev = centre + axisDir * zOff + Vec3f(rRing, 0, 0);
            for (int i = 1; i <= kSeg; ++i) {
                float a = 2.0f * 3.14159265f * float(i) / float(kSeg);
                Vec3f cur = centre + axisDir * zOff
                          + Vec3f(rRing * std::cos(a), 0, rRing * std::sin(a));
                cur = rot.rotate(cur - centre) + centre;  // reorient
                r.drawLine(prev, cur, color);
                prev = cur;
            }
        }
    };

    drawCap(worldA, rot.rotate(Vec3f::unitY()) * (-1.0f));
    drawCap(worldB, rot.rotate(Vec3f::unitY()));
}

/// @brief  Draw a plane as a grid of lines in world space.
inline void drawPlane(IDebugRenderer& r, const Vec3f& pos, const Quaternionf& rot,
                      float extent, const Color& color) noexcept {
    Vec3f u = rot.rotate(Vec3f::unitX());
    Vec3f v = rot.rotate(Vec3f::unitZ());
    constexpr int kGrid = 10;
    for (int i = -kGrid; i <= kGrid; ++i) {
        float t = extent * float(i) / float(kGrid);
        r.drawLine(pos + u * t - v * extent, pos + u * t + v * extent, color);
        r.drawLine(pos - u * extent + v * t, pos + u * extent + v * t, color);
    }
}

} // namespace debug_detail
} // namespace ciphyxs
