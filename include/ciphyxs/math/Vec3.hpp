//==================================================================================================
/// @file  Vec3.hpp
/// @brief 3-component vector with SIMD-friendly alignment.  Fully constexpr for C++20.
//==================================================================================================
#pragma once

#include <concepts>
#include <cmath>
#include <cstddef>
#include <ostream>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Concepts
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  A floating-point scalar type suitable for physics math (float, double, etc.).
template <typename T>
concept Scalar = std::floating_point<T>;

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Vec3
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  3-component vector aligned to 16 bytes for efficient SSE/NEON load/store.
/// @tparam T  Floating-point scalar (default: float).
///
/// All algebraic operations are **constexpr** in C++20.  Transcendental operations (length,
/// normalize, etc.) call `std::sqrt` and are **not** constexpr (available C++23 onward).
///
/// Memory layout: `[x, y, z]` with trailing padding to satisfy `alignas(16)`.
/// This guarantees that a contiguous array of Vec3 values is SIMD-friendly.
template <Scalar T = float>
struct alignas(16) Vec3 {
    T x{}, y{}, z{};

    // ─── Constructors ───────────────────────────────────────────────────────────────────────────

    /// @brief  Default constructor – zero-initialises all components.
    constexpr Vec3() noexcept = default;

    /// @brief  Construct from three components.
    constexpr Vec3(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}

    /// @brief  Uniform constructor – sets all components to the same value.
    explicit constexpr Vec3(T s) noexcept : x(s), y(s), z(s) {}

    // ─── Assignment ─────────────────────────────────────────────────────────────────────────────
    constexpr Vec3& operator+=(const Vec3& rhs) noexcept { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
    constexpr Vec3& operator-=(const Vec3& rhs) noexcept { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
    constexpr Vec3& operator*=(T s)                  noexcept { x *= s;    y *= s;    z *= s;    return *this; }
    constexpr Vec3& operator/=(T s)                  noexcept { T inv = T(1) / s; return *this *= inv; }

    // ─── Unary ──────────────────────────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr Vec3 operator-() const noexcept { return Vec3(-x, -y, -z); }

    // ─── Binary arithmetic ──────────────────────────────────────────────────────────────────────
    [[nodiscard]] friend constexpr Vec3 operator+(Vec3 lhs, const Vec3& rhs) noexcept { lhs += rhs; return lhs; }
    [[nodiscard]] friend constexpr Vec3 operator-(Vec3 lhs, const Vec3& rhs) noexcept { lhs -= rhs; return lhs; }
    [[nodiscard]] friend constexpr Vec3 operator*(Vec3 v, T s)                noexcept { v *= s;    return v; }
    [[nodiscard]] friend constexpr Vec3 operator*(T s, Vec3 v)                noexcept { v *= s;    return v; }
    [[nodiscard]] friend constexpr Vec3 operator/(Vec3 v, T s)                noexcept { v /= s;    return v; }

    // ─── Index access ───────────────────────────────────────────────────────────────────────────
    /// @brief  Access component by index (0 = x, 1 = y, 2 = z).  No bounds checking.
    [[nodiscard]] constexpr T&       operator[](std::size_t i)       noexcept { return (&x)[i]; }
    [[nodiscard]] constexpr T const& operator[](std::size_t i) const noexcept { return (&x)[i]; }

    // ─── Algebraic operations ───────────────────────────────────────────────────────────────────

    /// @brief  Dot product.
    [[nodiscard]] constexpr T dot(const Vec3& rhs) const noexcept {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    /// @brief  Cross product.
    [[nodiscard]] constexpr Vec3 cross(const Vec3& rhs) const noexcept {
        return Vec3(
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        );
    }

    /// @brief  Squared length (avoids `std::sqrt` – useful in early-out tests).
    [[nodiscard]] constexpr T lengthSquared() const noexcept { return dot(*this); }

    /// @brief  Euclidean length.
    /// @note   Not constexpr because `std::sqrt` is not required to be constexpr before C++23.
    [[nodiscard]] T length() const noexcept {
        return std::sqrt(lengthSquared());
    }

    /// @brief  Return a normalized copy (unit vector).  Zero-vector returns itself.
    [[nodiscard]] Vec3 normalized() const noexcept {
        T lenSq = lengthSquared();
        if (lenSq < T(1e-12)) return *this;
        return *this * (T(1) / std::sqrt(lenSq));
    }

    /// @brief  Normalise in-place.  Returns *this for chaining.
    Vec3& normalize() noexcept {
        T lenSq = lengthSquared();
        if (lenSq > T(1e-12)) *this *= T(1) / std::sqrt(lenSq);
        return *this;
    }

    // ─── Comparison ─────────────────────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr bool operator==(const Vec3&) const noexcept = default;

    // ─── Stream output ──────────────────────────────────────────────────────────────────────────
    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        return os << "Vec3(" << v.x << ", " << v.y << ", " << v.z << ")";
    }

    // ─── Constants ──────────────────────────────────────────────────────────────────────────────
    static constexpr Vec3 zero()  noexcept { return Vec3(T(0)); }
    static constexpr Vec3 one()   noexcept { return Vec3(T(1)); }
    static constexpr Vec3 unitX() noexcept { return Vec3(T(1), T(0), T(0)); }
    static constexpr Vec3 unitY() noexcept { return Vec3(T(0), T(1), T(0)); }
    static constexpr Vec3 unitZ() noexcept { return Vec3(T(0), T(0), T(1)); }
};

/// @brief  Common aliases.
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

} // namespace ciphyxs
