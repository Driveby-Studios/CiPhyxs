//==================================================================================================
/// @file  Quaternion.hpp
/// @brief  Quaternion rotation primitive with full constexpr arithmetic (C++20).
//==================================================================================================
#pragma once

#include "Vec3.hpp"
#include "Mat3.hpp"
#include <algorithm>
#include <cmath>
#include <ostream>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Quaternion
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Unit quaternion representing a 3D rotation.
/// @tparam T  Floating-point scalar (default: float).
///
/// Uses the **Hamilton convention** where `i*j = k`, `j*k = i`, `k*i = j`.
/// A rotation of vector `v` is:  `v' = q * v * q^{-1}`.
///
/// Layout: `[w, x, y, z]` — aligned to 16 bytes (a natural fit for SSE / NEON).
template <Scalar T = float>
struct alignas(16) Quaternion {
    T w{T(1)}, x{}, y{}, z{};   // default = identity

    // ─── Constructors ───────────────────────────────────────────────────────────────────────────
    constexpr Quaternion() noexcept = default;

    /// @brief  Construct from components.
    constexpr Quaternion(T w_, T x_, T y_, T z_) noexcept : w(w_), x(x_), y(y_), z(z_) {}

    // ─── Access ─────────────────────────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr T&       operator[](std::size_t i)       noexcept { return (&w)[i]; }
    [[nodiscard]] constexpr T const& operator[](std::size_t i) const noexcept { return (&w)[i]; }

    /// @brief  The vector (imaginary) part – x, y, z wrapped as a Vec3.
    [[nodiscard]] constexpr Vec3<T> vec() const noexcept { return Vec3<T>(x, y, z); }

    // ─── Arithmetic ─────────────────────────────────────────────────────────────────────────────

    /// @brief  Quaternion multiplication (composition):  `(*this) * rhs`.
    [[nodiscard]] constexpr Quaternion operator*(const Quaternion& rhs) const noexcept {
        return Quaternion(
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w
        );
    }

    constexpr Quaternion& operator*=(const Quaternion& rhs) noexcept { return *this = *this * rhs; }

    /// @brief  Conjugate (negate vector part).
    [[nodiscard]] constexpr Quaternion conjugate() const noexcept {
        return Quaternion(w, -x, -y, -z);
    }

    /// @brief  Inverse (conjugate / squared-norm).  For unit quaternions this is just the conjugate.
    [[nodiscard]] constexpr Quaternion inverse() const noexcept {
        T lenSq = w * w + x * x + y * y + z * z;
        if (lenSq < T(1e-12)) return identity();
        T invLenSq = T(1) / lenSq;
        return Quaternion(w * invLenSq, -x * invLenSq, -y * invLenSq, -z * invLenSq);
    }

    /// @brief  Squared length.
    [[nodiscard]] constexpr T lengthSquared() const noexcept { return w * w + x * x + y * y + z * z; }

    /// @brief  Length.
    [[nodiscard]] T length() const noexcept { return std::sqrt(lengthSquared()); }

    /// @brief  Return a normalized copy.
    [[nodiscard]] Quaternion normalized() const noexcept {
        T len = length();
        if (len < T(1e-12)) return identity();
        T invLen = T(1) / len;
        return Quaternion(w * invLen, x * invLen, y * invLen, z * invLen);
    }

    /// @brief  Normalise in-place.
    Quaternion& normalize() noexcept {
        T len = length();
        if (len > T(1e-12)) {
            T invLen = T(1) / len;
            w *= invLen; x *= invLen; y *= invLen; z *= invLen;
        } else {
            *this = identity();
        }
        return *this;
    }

    // ─── Rotate a vector ────────────────────────────────────────────────────────────────────────

    /// @brief  Rotate `v` by this quaternion.
    /// @note   Uses the efficient formula:
    ///         `t = 2 * cross(q.xyz, v)`
    ///         `v' = v + q.w * t + cross(q.xyz, t)`
    [[nodiscard]] Vec3<T> rotate(const Vec3<T>& v) const noexcept {
        Vec3<T> qv(x, y, z);
        Vec3<T> t = T(2) * qv.cross(v);
        return v + w * t + qv.cross(t);
    }

    /// @brief  Equivalent to `conjugate().rotate(v)` (inverse rotation).
    [[nodiscard]] Vec3<T> rotateInverse(const Vec3<T>& v) const noexcept {
        return conjugate().rotate(v);
    }

    // ─── Factories ──────────────────────────────────────────────────────────────────────────────

    /// @brief  Identity quaternion.
    [[nodiscard]] static constexpr Quaternion identity() noexcept {
        return Quaternion(T(1), T(0), T(0), T(0));
    }

    /// @brief  Build a quaternion from an **axis**-**angle** representation (radians).
    [[nodiscard]] static Quaternion fromAxisAngle(const Vec3<T>& axis, T angle) noexcept {
        Vec3<T> n = axis.normalized();
        T half   = angle * T(0.5);
        T s      = std::sin(half);
        return Quaternion(std::cos(half), n.x * s, n.y * s, n.z * s);
    }

    /// @brief  Build the shortest-arc quaternion rotating `from` onto `to`.
    [[nodiscard]] static Quaternion fromTo(const Vec3<T>& from, const Vec3<T>& to) noexcept {
        Vec3<T> f = from.normalized();
        Vec3<T> t = to.normalized();
        T d = f.dot(t);

        if (d > T(1.0 - 1e-6)) return identity();     // parallel, same direction
        if (d < T(-1.0 + 1e-6)) {                      // parallel, opposite
            Vec3<T> axis = Vec3<T>::unitX().cross(f);
            if (axis.lengthSquared() < T(1e-8))
                axis = Vec3<T>::unitY().cross(f);
            return fromAxisAngle(axis.normalized(), T(3.14159265358979));
        }

        Vec3<T> axis = f.cross(t);
        T s = std::sqrt(T(2) * (T(1) + d));
        return Quaternion(s * T(0.5), axis.x / s, axis.y / s, axis.z / s);
    }

    /// @brief  Build from Euler angles (XYZ intrinsic, radians).
    [[nodiscard]] static Quaternion fromEuler(T x, T y, T z) noexcept {
        T cx = std::cos(x * T(0.5)), sx = std::sin(x * T(0.5));
        T cy = std::cos(y * T(0.5)), sy = std::sin(y * T(0.5));
        T cz = std::cos(z * T(0.5)), sz = std::sin(z * T(0.5));

        return Quaternion(
            cx * cy * cz - sx * sy * sz,
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz
        );
    }

    /// @brief  Convert to a 3×3 rotation matrix.
    [[nodiscard]] Mat3<T> toMat3() const noexcept {
        T xx = x * x, yy = y * y, zz = z * z;
        T xy = x * y, xz = x * z, yz = y * z;
        T wx = w * x, wy = w * y, wz = w * z;

        return Mat3<T>(
            T(1) - T(2) * (yy + zz), T(2) * (xy + wz),       T(2) * (xz - wy),
            T(2) * (xy - wz),         T(1) - T(2) * (xx + zz), T(2) * (yz + wx),
            T(2) * (xz + wy),         T(2) * (yz - wx),         T(1) - T(2) * (xx + yy)
        );
    }

    /// @brief  Extract the rotation axis (normalised).  Zero-rotation returns unitX().
    [[nodiscard]] Vec3<T> axis() const noexcept {
        T s = T(1) - w * w;
        if (s < T(1e-12)) return Vec3<T>::unitX();
        T invS = T(1) / std::sqrt(s);
        return Vec3<T>(x * invS, y * invS, z * invS);
    }

    /// @brief  Extract the rotation angle in radians.  Always in [0, π].
    [[nodiscard]] T angle() const noexcept {
        return T(2) * std::acos(std::clamp(w, T(-1), T(1)));
    }

    // ─── Interpolation ──────────────────────────────────────────────────────────────────────────

    /// @brief  Spherical linear interpolation (Slerp) between `a` and `b`.
    /// @param t  Interpolation factor in [0, 1].
    [[nodiscard]] static Quaternion slerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
        T cosTheta = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
        Quaternion bc = b;
        if (cosTheta < T(0)) {        // take the shorter arc
            bc = -bc;
            cosTheta = -cosTheta;
        }

        constexpr T kThreshold = T(0.999);
        if (cosTheta >= kThreshold) { // nearly parallel – linear interpolation avoids division by ~0
            Quaternion result = a + t * (bc - a);
            return result.normalized();
        }

        T theta = std::acos(cosTheta);
        T sinTheta = std::sin(theta);
        T invSin = T(1) / sinTheta;
        T tA = std::sin((T(1) - t) * theta) * invSin;
        T tB = std::sin(t * theta) * invSin;

        return Quaternion(
            a.w * tA + bc.w * tB,
            a.x * tA + bc.x * tB,
            a.y * tA + bc.y * tB,
            a.z * tA + bc.z * tB
        );
    }

    // ─── Convenience ────────────────────────────────────────────────────────────────────────────

    /// @brief  Unary negation – flips the rotation (represents the same orientation on the
    ///         double-cover, but useful for the shortest-path slerp).
    [[nodiscard]] constexpr Quaternion operator-() const noexcept {
        return Quaternion(-w, -x, -y, -z);
    }

    /// @brief  Scalar multiply (useful for blends).
    [[nodiscard]] friend constexpr Quaternion operator*(Quaternion q, T s) noexcept {
        return Quaternion(q.w * s, q.x * s, q.y * s, q.z * s);
    }
    [[nodiscard]] friend constexpr Quaternion operator*(T s, Quaternion q) noexcept { return q * s; }
    [[nodiscard]] friend constexpr Quaternion operator+(Quaternion a, const Quaternion& b) noexcept {
        return Quaternion(a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z);
    }
    [[nodiscard]] friend constexpr Quaternion operator-(Quaternion a, const Quaternion& b) noexcept {
        return Quaternion(a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z);
    }

    // ─── Comparison ─────────────────────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr bool operator==(const Quaternion&) const noexcept = default;

    // ─── Stream output ──────────────────────────────────────────────────────────────────────────
    friend std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
        return os << "Quaternion(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
    }
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

} // namespace ciphyxs
