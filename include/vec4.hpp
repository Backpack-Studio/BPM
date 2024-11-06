/**
 * Copyright (c) 2023-2024 Le Juez Victor
 *
 * This software is provided "as-is", without any express or implied warranty. In no event 
 * will the authors be held liable for any damages arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose, including commercial 
 * applications, and to alter it and redistribute it freely, subject to the following restrictions:
 *
 *   1. The origin of this software must not be misrepresented; you must not claim that you 
 *   wrote the original software. If you use this software in a product, an acknowledgment 
 *   in the product documentation would be appreciated but is not required.
 *
 *   2. Altered source versions must be plainly marked as such, and must not be misrepresented
 *   as being the original software.
 *
 *   3. This notice may not be removed or altered from any source distribution.
 */

#ifndef BPM_VEC4_HPP
#define BPM_VEC4_HPP

#include "./vec3.hpp"

#include <type_traits>
#include <algorithm>
#include <cstdint>
#include <ostream>
#include <cmath>

namespace bpm {

template <typename T>
struct Vector4;

using Vec4 = Vector4<float>;
using DVec4 = Vector4<double>;
using IVec4 = Vector4<int32_t>;
using UVec4 = Vector4<uint32_t>;

template <typename T>
struct Vector4
{
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    static constexpr int DIMENSIONS = 4;
    typedef T value_type;

    T x, y, z, w;

    /**
     * @brief Default constructor. Constructs a vector with all components set to zero.
     */
    constexpr Vector4()
        : x(0), y(0),
            z(0), w(0)
    { }

    /**
     * @brief Constructs a vector with all components set to the specified value.
     *
     * @param value The value to set for all components.
     */
    constexpr explicit Vector4(T value)
        : x(value), y(value),
            z(value), w(value)
    { }

    /**
     * @brief Constructs a vector with the specified components.
     *
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     * @param w The w-component.
     */
    constexpr Vector4(T x, T y, T z, T w = 1.0f)
        : x(x), y(y),
            z(z), w(w)
    { }

    /**
     * @brief Constructs a Vector4 from a Vector3 with an optional w-component.
     *
     * @param Vec3 The Vector3 to use for the x, y, and z components.
     * @param w The w-component.
     */
    constexpr Vector4(const Vector3<T>& Vec3, float w = 1.0f)
        : x(Vec3.x), y(Vec3.y),
            z(Vec3.z), w(w)
    { }

    /**
     * @brief Conversion operator to convert the vector to a Vector4 of a different type.
     *
     * Converts the current vector to a Vector4 of a different type by casting its components to the new type.
     *
     * @tparam U The type to convert the components of the vector to.
     * @return A Vector4 containing the components of the current vector casted to the type U.
     */
    template <typename U>
    constexpr operator Vector4<U>() const {
        return Vector4<U>(
            static_cast<U>(x),
            static_cast<U>(y),
            static_cast<U>(z),
            static_cast<U>(w));
    }

    /**
     * @brief Accesses the component at the specified index.
     *
     * @param axis The index of the component to access.
     * @return T& A reference to the component at the specified index.
     */
    constexpr T& operator[](int axis) {
        return *(reinterpret_cast<T*>(this) + axis);
    }

    /**
     * @brief Accesses the component at the specified index.
     *
     * @param axis The index of the component to access.
     * @return const T& A const reference to the component at the specified index.
     */
    constexpr const T& operator[](int axis) const {
        return *(reinterpret_cast<const T*>(this) + axis);
    }

    /**
     * @brief Negates each component of the vector.
     *
     * @return Vector4 The resulting negated vector.
     */
    constexpr Vector4 operator-() {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] = -(*reinterpret_cast<T>(&result))[i];
        }
        return result;
    }

    /**
     * @brief Subtracts a scalar value from each component of the vector.
     *
     * @param scalar The scalar value to subtract.
     * @return Vector4 The resulting vector.
     */
    constexpr Vector4 operator-(T scalar) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] -= scalar;
        }
        return result;
    }

    /**
     * @brief Adds a scalar value to each component of the vector.
     *
     * @param scalar The scalar value to add.
     * @return Vector4 The resulting vector.
     */
    constexpr Vector4 operator+(T scalar) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] += scalar;
        }
        return result;
    }

    /**
     * @brief Multiplies each component of the vector by a scalar value.
     *
     * @param scalar The scalar value to multiply by.
     * @return Vector4 The resulting vector.
     */
    constexpr Vector4 operator*(T scalar) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] *= scalar;
        }
        return result;
    }

    /**
     * @brief Scalar division operator.
     * 
     * Divides each component of the vector by the given scalar value.
     * 
     * @warning If the scalar is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param scalar The scalar value to divide by.
     * @return Vector4 The result of the division.
     */
    constexpr Vector4 operator/(T scalar) const {
        Vector4 result = *this;
        if constexpr (std::is_floating_point<T>::value) {
            const T inv = 1.0 / scalar;
            for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
                (*reinterpret_cast<T>(&result))[i] *= inv;
            }
        } else {
            for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
                (*reinterpret_cast<T>(&result))[i] /= scalar;
            }
        }
        return result;
    }

    /**
     * @brief Vector subtraction operator.
     *
     * @param other The vector to subtract.
     * @return Vector4 The result of the subtraction.
     */
    constexpr Vector4 operator-(const Vector4& other) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] -= (*reinterpret_cast<T>(&other))[i];
        }
        return result;
    }

    /**
     * @brief Vector addition operator.
     *
     * @param other The vector to add.
     * @return Vector4 The result of the addition.
     */
    constexpr Vector4 operator+(const Vector4& other) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] += (*reinterpret_cast<T>(&other))[i];
        }
        return result;
    }

    /**
     * @brief Vector multiplication operator.
     *
     * @param other The vector to multiply by.
     * @return Vector4 The result of the multiplication.
     */
    constexpr Vector4 operator*(const Vector4& other) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] *= (*reinterpret_cast<T>(&other))[i];
        }
        return result;
    }

    /**
     * @brief Vector division operator.
     * 
     * Divides each component of the current vector by the corresponding component of the other vector.
     * 
     * @warning If any component of the `other` vector is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param other The vector to divide by.
     * @return Vector4 The result of the division.
     */
    constexpr Vector4 operator/(const Vector4& other) const {
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] /= (*reinterpret_cast<T>(&other))[i];
        }
        return result;
    }

    /**
     * @brief Equality operator.
     * 
     * @param other The vector to compare with.
     * @return bool True if the vectors are equal, false otherwise.
     */
    constexpr bool operator==(const Vector4& other) const {
        return (x == other.x) && (y == other.y) && (z == other.z) && (w == other.w);
    }

    /**
     * @brief Inequality operator.
     * 
     * @param other The vector to compare with.
     * @return bool True if the vectors are not equal, false otherwise.
     */
    constexpr bool operator!=(const Vector4& other) const {
        return (x != other.x) || (y != other.y) || (z != other.z) || (w != other.w);
    }

    /**
     * @brief Scalar subtraction and assignment operator.
     *
     * @param scalar The scalar value to subtract.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator-=(T scalar) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] -= scalar;
        }
        return *this;
    }

    /**
     * @brief Scalar addition and assignment operator.
     *
     * @param scalar The scalar value to add.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator+=(T scalar) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] += scalar;
        }
        return *this;
    }

    /**
     * @brief Scalar multiplication and assignment operator.
     *
     * @param scalar The scalar value to multiply by.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator*=(T scalar) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] *= scalar;
        }
        return *this;
    }

    /**
     * @brief Scalar division and assignment operator.
     * 
     * Divides each component of the vector by the given scalar value and assigns the result back to the vector.
     * 
     * @warning If the scalar is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param scalar The scalar value to divide by.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator/=(T scalar) {
        if constexpr (std::is_floating_point<T>::value) {
            const T inv = 1.0 / scalar;
            for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
                (*reinterpret_cast<T>(this))[i] *= inv;
            }
        } else {
            for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
                (*reinterpret_cast<T>(this))[i] /= scalar;
            }
        }
        return *this;
    }

    /**
     * @brief Vector subtraction and assignment operator.
     *
     * @param other The vector to subtract.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator-=(const Vector4& other) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] -= (*reinterpret_cast<T>(&other))[i];
        }
        return *this;
    }

    /**
     * @brief Vector addition and assignment operator.
     *
     * @param other The vector to add.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator+=(const Vector4& other) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] += (*reinterpret_cast<T>(&other))[i];
        }
        return *this;
    }

    /**
     * @brief Vector multiplication and assignment operator.
     *
     * @param other The vector to multiply by.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator*=(const Vector4& other) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] *= (*reinterpret_cast<T>(&other))[i];
        }
        return *this;
    }

    /**
     * @brief Vector division and assignment operator.
     * 
     * Divides each component of the current vector by the corresponding component of the other vector and assigns the result back to the current vector.
     * 
     * @warning If any component of the `other` vector is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param other The vector to divide by.
     * @return Vector4& Reference to the modified vector.
     */
    Vector4& operator/=(const Vector4& other) {
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(this))[i] /= (*reinterpret_cast<T>(&other))[i];
        }
        return *this;
    }

    /**
     * @brief Overload of the output stream operator for Vector4.
     *
     * Allows a Vector4 to be output to an output stream (e.g., std::cout) in the format: Vec4(x, y, z, w).
     *
     * @param os The output stream (e.g., std::cout).
     * @param v The vector to be output.
     * @return A reference to the modified output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Vector4& v) {
        os << "Vec4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
        return os;
    }

    /**
     * @brief Method to check if the vector is equal to (0,0,0,0).
     *
     * @return bool True if the vector is equal to (0,0,0,0), false otherwise.
     */
    bool is_zero() const {
        return !(x + y + z + w);
    }

    /**
    * @brief Calculate the reciprocal of the vector components.
    *
    * @return A new Vector2 object with each component as the reciprocal of the original.
    */
    Vector4 rcp() const {
        static_assert(std::is_floating_point<T>::value, "T must be a floating-point type.");
        Vector4 result = *this;
        for (int_fast8_t i = 0; i < DIMENSIONS; i++) {
            (*reinterpret_cast<T>(&result))[i] =
                static_cast<T>(1.0) / (*reinterpret_cast<T>(this))[i];
        }
        return result;
    }

    /**
     * @brief Function to calculate the length (magnitude) of the vector.
     *
     * @return T The length of the vector.
     */
    T length() const {
        return std::sqrt(x * x + y * y + z * z + w * w);
    }

    /**
     * @brief Function to calculate the length squared of the vector.
     *
     * @return T The length squared of the vector.
     */
    T length_sq() const {
        return x * x + y * y + z * z + w * w;
    }

    /**
     * @brief Method to calculate the dot product of the vector with another vector.
     *
     * @param other The other vector.
     * @return T The dot product of the two vectors.
     */
    T dot(const Vector4& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    /**
     * @brief Method to normalize the vector.
     */
    void normalize() {
        const T mag = length();
        if (mag != 0.0) (*this) *= 1.0 / mag;
    }

    /**
     * @brief Method to get a normalized vector.
     *
     * @return Vector4 The normalized vector.
     */
    Vector4 normalized() const {
        Vector4 result(*this);
        result.normalize();
        return result;
    }

    /**
     * @brief Method to calculate the distance between two vectors.
     *
     * @param other The other vector.
     * @return T The distance between the two vectors.
     */
    T distance(const Vector4& other) const {
        return (*this - other).length();
    }

    /**
     * @brief Function to calculate the squared distance between two vectors.
     *
     * @param other The other vector.
     * @return T The squared distance between the two vectors.
     */
    T distance_sq(const Vector4& other) const {
        const Vector4 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }

    /**
     * @brief Function to transform the vector by a 4x4 matrix.
     *
     * @param matrix The 4x4 matrix.
     */
    void transform(const Mat4& matrix); ///< NOTE: Defined in 'mat4.hpp'

    /**
     * @brief Function to get the vector transformed by a 4x4 matrix.
     *
     * @param matrix The 4x4 matrix.
     * @return Vector4 The transformed vector.
     */
    Vector4 transformed(const Mat4& matrix) const; ///< NOTE: Defined in 'mat4.hpp'

    /**
     * @brief Move this vector towards another vector 'b' by a specified 'delta'.
     *
     * This method calculates the new position of the vector by moving towards
     * the target vector 'b', limited by the distance 'delta' for each component.
     *
     * @param b The target vector to move towards.
     * @param delta The maximum distance to move towards 'b'.
     * @return A new vector that is moved towards 'b' by 'delta'.
     */
    constexpr Vector4 move_towards(Vector4 b, T delta) {
        T dx = b.x - x;
        T dy = b.y - y;
        T dz = b.z - z;
        T dw = b.w - w;

        return {
            (std::abs(dx) > delta) ? x + (delta * (dx > T(0) ? T(1) : T(-1))) : b.x,
            (std::abs(dy) > delta) ? y + (delta * (dy > T(0) ? T(1) : T(-1))) : b.y,
            (std::abs(dz) > delta) ? z + (delta * (dz > T(0) ? T(1) : T(-1))) : b.z,
            (std::abs(dw) > delta) ? w + (delta * (dw > T(0) ? T(1) : T(-1))) : b.w
        };
    }

    /**
     * @brief Linearly interpolate between this vector and another vector 'b' based on a parameter 't'.
     *
     * This method computes a point along the line connecting this vector and 'b'.
     * The interpolation parameter 't' should be in the range [0, 1], where
     * t=0 returns this vector and t=1 returns vector 'b'.
     *
     * @param b The target vector to interpolate towards.
     * @param t The interpolation factor (should be in the range [0, 1]).
     * @return A new vector that is the result of the linear interpolation.
     */
    constexpr Vector4 lerp(Vector4 b, T t) {
        static_assert(std::is_floating_point<T>::value, "Only floating-point types are allowed.");
        return {
            x + t * (b.x - x),
            y + t * (b.y - y),
            z + t * (b.z - z),
            w + t * (b.w - w)
        };
    }

    /**
     * @brief Function to clamp the vector components within a specified range.
     *
     * @param min The minimum value vector.
     * @param max The maximum value vector.
     * @return Vector4 The clamped vector.
     */
    Vector4 clamp(const Vector4& min, const Vector4& max) {
        return {
            std::clamp(x, min.x, max.x),
            std::clamp(y, min.y, max.y),
            std::clamp(z, min.z, max.z),
            std::clamp(w, min.w, max.w)
        };
    }

    /**
     * @brief Function to clamp the vector components within a specified range.
     *
     * @param min The minimum value.
     * @param max The maximum value.
     * @return Vector4 The clamped vector.
     */
    Vector4 clamp(T min, T max) {
        return {
            std::clamp(x, min, max),
            std::clamp(y, min, max),
            std::clamp(z, min, max),
            std::clamp(w, min, max)
        };
    }

    /**
     * @brief Function to get the absolute values of the vector components.
     *
     * @return Vector4 The vector with absolute values.
     */
    Vector4 abs() {
        return {
            std::abs(x),
            std::abs(y),
            std::abs(z),
            std::abs(w)
        };
    }

    /**
     * @brief Returns a pointer to the underlying data of the vector.
     * 
     * This method provides a pointer to the raw data of the vector, allowing direct
     * access to the components. The returned pointer is of type `T*`, where `T` is the
     * data type (e.g., `float`, `double`) of the vector's components.
     * 
     * @return T* Pointer to the underlying data of the vector.
     */
    T* ptr() {
        return reinterpret_cast<T*>(this);
    }

    /**
     * @brief Returns a constant pointer to the underlying data of the vector.
     * 
     * This method provides a pointer to the raw data of the vector, allowing direct
     * access to the components in a read-only manner. The returned pointer is of type `const T*`,
     * where `T` is the data type (e.g., `float`, `double`) of the vector's components.
     * 
     * @return const T* Constant pointer to the underlying data of the vector.
     */
    const T* ptr() const {
        return reinterpret_cast<const T*>(this);
    }
};

} // namespace bpm

#endif // BPM_VEC4_HPP
