//==================================================================================================
/// @file  Mat3.hpp
/// @brief  3×3 column-major matrix with full constexpr arithmetic in C++20.
//==================================================================================================
#pragma once

#include "Vec3.hpp"
#include <cstddef>
#include <ostream>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Mat3
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  3×3 matrix stored in **column-major** order.
/// @tparam T  Floating-point scalar (default: float).
///
/// Storage is a flat `T[9]` array so that columns can be loaded into SIMD registers without
/// worrying about struct padding.  The element at `(row, col)` is `data[col * 3 + row]`.
///
/// All operations are **constexpr** in C++20.  `inverse()` calls `std::abs` for a near-zero
/// guard and is therefore runtime-only.
template <Scalar T = float>
struct /* not over-aligned – users should heap-allocate with 16-byte allocators for hot data */
    Mat3 {
    T data[9]{};   // column-major: data[col*3 + row]

    // ─── Constructors ───────────────────────────────────────────────────────────────────────────
    constexpr Mat3() noexcept = default;

    /// @brief  Construct from three column vectors.
    constexpr Mat3(const Vec3<T>& c0, const Vec3<T>& c1, const Vec3<T>& c2) noexcept {
        setCol(0, c0);
        setCol(1, c1);
        setCol(2, c2);
    }

    /// @brief  Construct from nine elements supplied in **row-major** order (natural for
    ///         mathematicians).  For example:
    ///         @code
    ///         Mat3 m(a, b, c,   // row 0
    ///                 d, e, f,   // row 1
    ///                 g, h, i);  // row 2
    ///         @endcode
    constexpr Mat3(T m00, T m01, T m02,
                   T m10, T m11, T m12,
                   T m20, T m21, T m22) noexcept
        : data{ m00, m10, m20,    // col 0
                m01, m11, m21,    // col 1
                m02, m12, m22 }   // col 2
    {}

    // ─── Element / column access ────────────────────────────────────────────────────────────────

    /// @brief  Read/write element at `(row, col)`.  0-based, no bounds checking.
    [[nodiscard]] constexpr T&       operator()(int row, int col)       noexcept { return data[col * 3 + row]; }
    [[nodiscard]] constexpr T const& operator()(int row, int col) const noexcept { return data[col * 3 + row]; }

    /// @brief  Read column `i` (0–2) as a Vec3.
    [[nodiscard]] constexpr Vec3<T> col(int i) const noexcept {
        int idx = i * 3;
        return Vec3<T>(data[idx], data[idx + 1], data[idx + 2]);
    }

    /// @brief  Write column `i` from a Vec3.
    constexpr void setCol(int i, const Vec3<T>& v) noexcept {
        int idx   = i * 3;
        data[idx]     = v.x;
        data[idx + 1] = v.y;
        data[idx + 2] = v.z;
    }

    /// @brief  Row `i` as a Vec3 (uses the column-major data but presents a row).
    [[nodiscard]] constexpr Vec3<T> row(int i) const noexcept {
        return Vec3<T>(data[i], data[3 + i], data[6 + i]);
    }

    // ─── Arithmetic ─────────────────────────────────────────────────────────────────────────────
    constexpr Mat3& operator+=(const Mat3& rhs) noexcept { for (int i = 0; i < 9; ++i) data[i] += rhs.data[i]; return *this; }
    constexpr Mat3& operator-=(const Mat3& rhs) noexcept { for (int i = 0; i < 9; ++i) data[i] -= rhs.data[i]; return *this; }
    constexpr Mat3& operator*=(T s)                noexcept { for (int i = 0; i < 9; ++i) data[i] *= s;        return *this; }
    constexpr Mat3& operator/=(T s)                noexcept { T inv = T(1) / s; return *this *= inv; }

    [[nodiscard]] constexpr Mat3 operator-() const noexcept { Mat3 r; for (int i = 0; i < 9; ++i) r.data[i] = -data[i]; return r; }

    [[nodiscard]] friend constexpr Mat3 operator+(Mat3 lhs, const Mat3& rhs) noexcept { lhs += rhs; return lhs; }
    [[nodiscard]] friend constexpr Mat3 operator-(Mat3 lhs, const Mat3& rhs) noexcept { lhs -= rhs; return lhs; }
    [[nodiscard]] friend constexpr Mat3 operator*(Mat3 m, T s)                noexcept { m *= s;    return m; }
    [[nodiscard]] friend constexpr Mat3 operator*(T s, Mat3 m)                noexcept { m *= s;    return m; }
    [[nodiscard]] friend constexpr Mat3 operator/(Mat3 m, T s)                noexcept { m /= s;    return m; }

    // ─── Matrix-matrix multiplication ───────────────────────────────────────────────────────────
    /// @brief  Multiply two matrices: `(*this) * rhs`.
    [[nodiscard]] constexpr Mat3 operator*(const Mat3& rhs) const noexcept {
        Mat3 result;
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 3; ++r) {
                result(r, c) = row(r).dot(rhs.col(c));
            }
        }
        return result;
    }

    constexpr Mat3& operator*=(const Mat3& rhs) noexcept { return *this = *this * rhs; }

    // ─── Matrix-vector multiplication ──────────────────────────────────────────────────────────
    /// @brief  Transform a column vector:  `result = M * v`.
    [[nodiscard]] constexpr Vec3<T> operator*(const Vec3<T>& v) const noexcept {
        return col(0) * v.x + col(1) * v.y + col(2) * v.z;
    }

    // ─── Transform helpers ──────────────────────────────────────────────────────────────────────
    /// @brief  Transpose.
    [[nodiscard]] constexpr Mat3 transpose() const noexcept {
        Mat3 t;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                t(c, r) = (*this)(r, c);
        return t;
    }

    /// @brief  Determinant.
    [[nodiscard]] constexpr T determinant() const noexcept {
        const auto& d = data;
        //             a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        return d[0] * (d[4] * d[8] - d[5] * d[7])
             - d[1] * (d[3] * d[8] - d[5] * d[6])
             + d[2] * (d[3] * d[7] - d[4] * d[6]);
    }

    /// @brief  Inverse via cofactor method.  Returns identity if singular.
    /// @note   Not constexpr because of `std::abs`.
    [[nodiscard]] Mat3 inverse() const noexcept {
        T det = determinant();
        if (std::abs(det) < T(1e-12)) return identity();

        T invDet = T(1) / det;
        const auto& d = data;

        // Cofactor matrix (transposed for adjugate), then multiply by invDet.
        Mat3 adj;
        // clang-format off
        adj(0, 0) =  (d[4] * d[8] - d[5] * d[7]) * invDet;
        adj(0, 1) = -(d[3] * d[8] - d[5] * d[6]) * invDet;
        adj(0, 2) =  (d[3] * d[7] - d[4] * d[6]) * invDet;

        adj(1, 0) = -(d[1] * d[8] - d[2] * d[7]) * invDet;
        adj(1, 1) =  (d[0] * d[8] - d[2] * d[6]) * invDet;
        adj(1, 2) = -(d[0] * d[7] - d[1] * d[6]) * invDet;

        adj(2, 0) =  (d[1] * d[5] - d[2] * d[4]) * invDet;
        adj(2, 1) = -(d[0] * d[5] - d[2] * d[3]) * invDet;
        adj(2, 2) =  (d[0] * d[4] - d[1] * d[3]) * invDet;
        // clang-format on
        return adj;
    }

    // ─── Factories ──────────────────────────────────────────────────────────────────────────────
    /// @brief  Identity matrix.
    [[nodiscard]] static constexpr Mat3 identity() noexcept {
        return Mat3(T(1), T(0), T(0),
                    T(0), T(1), T(0),
                    T(0), T(0), T(1));
    }

    /// @brief  Zero matrix.
    [[nodiscard]] static constexpr Mat3 zero() noexcept { return Mat3{}; }

    /// @brief  Rotation matrix from an **axis** and **angle** (right-handed, radians).
    [[nodiscard]] static Mat3 rotationAxisAngle(const Vec3<T>& axis, T angle) noexcept {
        Vec3<T> n   = axis.normalized();
        T c        = std::cos(angle);
        T s        = std::sin(angle);
        T oneMinusC = T(1) - c;
        T x = n.x, y = n.y, z = n.z;

        return Mat3(
            c + x*x*oneMinusC,   x*y*oneMinusC - z*s,   x*z*oneMinusC + y*s,
            y*x*oneMinusC + z*s, c + y*y*oneMinusC,     y*z*oneMinusC - x*s,
            z*x*oneMinusC - y*s, z*y*oneMinusC + x*s,   c + z*z*oneMinusC
        );
    }

    /// @brief  Rotation matrix from Euler angles (XYZ intrinsic order, in radians).
    [[nodiscard]] static Mat3 rotationEuler(T x, T y, T z) noexcept {
        T cx = std::cos(x), sx = std::sin(x);
        T cy = std::cos(y), sy = std::sin(y);
        T cz = std::cos(z), sz = std::sin(z);

        return Mat3(
            cy * cz,  cy * sz,  -sy,
            sx * sy * cz - cx * sz,
            sx * sy * sz + cx * cz,
            sx * cy,
            cx * sy * cz + sx * sz,
            cx * sy * sz - sx * cz,
            cx * cy
        );
    }

    // ─── Comparison ─────────────────────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr bool operator==(const Mat3& rhs) const noexcept {
        for (int i = 0; i < 9; ++i)
            if (data[i] != rhs.data[i]) return false;
        return true;
    }

    // ─── Stream output ──────────────────────────────────────────────────────────────────────────
    friend std::ostream& operator<<(std::ostream& os, const Mat3& m) {
        return os << "Mat3(" << m.row(0) << ", " << m.row(1) << ", " << m.row(2) << ")";
    }
};

using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;

} // namespace ciphyxs
