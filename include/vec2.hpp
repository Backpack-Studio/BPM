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

#ifndef BPM_VEC2_HPP
#define BPM_VEC2_HPP

#include "./mat3.hpp"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <cmath>

namespace bpm {

template <typename T>
struct Vector2;

using Vec2 = Vector2<float>;
using DVec2 = Vector2<double>;
using IVec2 = Vector2<int32_t>;
using UVec2 = Vector2<uint32_t>;

/**
 * @brief Represents a 2-dimensional vector.
 *
 * This class template represents a 2-dimensional vector with components x and y.
 * It provides various mathematical operations and utilities for manipulating 2D vectors.
 *
 * @tparam T The type of the components of the vector.
 */
template <typename T>
struct Vector2
{
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    static constexpr int DIMENSIONS = 2;    ///< The number of dimensions of the vector.
    typedef T value_type;                   ///< The type of each component of the vector.
    T x, y;                                 ///< The x and y components of the vector.

    /**
     * @brief Default constructor.
     *
     * Initializes both components x and y to zero.
     */
    constexpr Vector2()
        : x(0), y(0)
    { }

    /**
     * @brief Constructor initializing both components with a single value.
     *
     * Initializes both components x and y with the same given value.
     *
     * @param value The value to set for both x and y components.
     */
    constexpr explicit Vector2(T value)
        : x(value), y(value)
    { }

    /**
     * @brief Constructor initializing both components with specific values.
     *
     * Initializes the x component with the given x value and the y component with the given y value.
     *
     * @param x The value to set for the x component.
     * @param y The value to set for the y component.
     */
    constexpr Vector2(T x, T y)
        : x(x), y(y)
    { }

    /**
     * @brief Conversion operator to convert the vector to a Vector2 of a different type.
     *
     * Converts the current vector to a Vector2 of a different type by casting its components to the new type.
     *
     * @tparam U The type to convert the components of the vector to.
     * @return A Vector2 containing the components of the current vector casted to the type U.
     */
    template <typename U>
    constexpr operator Vector2<U>() const
    {
        return Vector2<U>(static_cast<U>(x), static_cast<U>(y));
    }

    /**
     * @brief Subscript operator to access the components of the vector.
     *
     * Allows accessing the components of the vector using array-style indexing.
     *
     * @param axis The index of the component to access (0 for x, 1 for y).
     * @return A reference to the component at the specified index.
     */
    constexpr T& operator[](int axis)
    {
        switch (axis) {
            case 0:  return x;
            default: return y;
        }
    }

    /**
     * @brief Subscript operator to access the components of the vector (const version).
     *
     * Allows accessing the components of the vector using array-style indexing.
     * This is the const version of the subscript operator.
     *
     * @param axis The index of the component to access (0 for x, 1 for y).
     * @return A const reference to the component at the specified index.
     */
    constexpr const T& operator[](int axis) const
    {
        switch (axis) {
            case 0:  return x;
            default: return y;
        }
    }

    /**
     * @brief Unary negation operator.
     *
     * Returns the negation of the current vector by negating each component.
     *
     * @return The negation of the current vector.
     */
    constexpr Vector2 operator-() const
    {
        return { -x, -y };
    }

    /**
     * @brief Subtraction operator with a scalar.
     *
     * Subtracts the given scalar value from each component of the current vector.
     *
     * @param scalar The scalar value to subtract from each component.
     * @return A new vector resulting from the subtraction operation.
     */
    constexpr Vector2 operator-(T scalar) const
    {
        return Vector2(x - scalar, y - scalar);
    }

    /**
     * @brief Addition operator with a scalar.
     *
     * Adds the given scalar value to each component of the current vector.
     *
     * @param scalar The scalar value to add to each component.
     * @return A new vector resulting from the addition operation.
     */
    constexpr Vector2 operator+(T scalar) const
    {
        return Vector2(x + scalar, y + scalar);
    }

    /**
     * @brief Multiplication operator with a scalar.
     *
     * Multiplies each component of the current vector by the given scalar value.
     *
     * @param scalar The scalar value to multiply each component by.
     * @return A new vector resulting from the multiplication operation.
     */
    constexpr Vector2 operator*(T scalar) const
    {
        return Vector2(x * scalar, y * scalar);
    }

    /**
     * @brief Division operator by a scalar.
     *
     * Divides each component of the current vector by the given scalar value.
     *
     * @warning If the scalar is zero, the behavior is undefined. This function does not check for zero scalar values.
     *          For floating-point scalars, division by zero may result in infinity or NaN.
     *
     * @param scalar The scalar value to divide each component by.
     * @return A new vector resulting from the division operation.
     */
    constexpr Vector2 operator/(T scalar) const
    {
        if constexpr (std::is_floating_point<T>::value) {
            const T inv = static_cast<T>(1.0) / scalar;
            return Vector2(x * inv, y * inv);
        }
        return Vector2(x / scalar, y / scalar);
    }

    /**
     * @brief Subtraction operator between vectors.
     *
     * Subtracts each component of the other vector from the corresponding component of the current vector.
     *
     * @param other The vector to subtract from the current vector.
     * @return A new vector resulting from the subtraction operation.
     */
    constexpr Vector2 operator-(const Vector2& other) const
    {
        return Vector2(x - other.x, y - other.y);
    }

    /**
     * @brief Addition operator between vectors.
     *
     * Adds each component of the other vector to the corresponding component of the current vector.
     *
     * @param other The vector to add to the current vector.
     * @return A new vector resulting from the addition operation.
     */
    constexpr Vector2 operator+(const Vector2& other) const
    {
        return Vector2(x + other.x, y + other.y);
    }

    /**
     * @brief Multiplication operator between vectors.
     *
     * Multiplies each component of the other vector by the corresponding component of the current vector.
     *
     * @param other The vector to multiply with the current vector.
     * @return A new vector resulting from the multiplication operation.
     */
    constexpr Vector2 operator*(const Vector2& other) const
    {
        return Vector2(x * other.x, y * other.y);
    }

    /**
     * @brief Division operator between vectors.
     *
     * Divides each component of the current vector by the corresponding component of the other vector.
     *
     * @warning If any component of the `other` vector is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param other The vector by which to divide the current vector.
     * @return A new vector resulting from the division operation.
     */
    constexpr Vector2 operator/(const Vector2& other) const
    {
        return Vector2(x / other.x, y / other.y);
    }

    /**
     * @brief Equality operator.
     *
     * Checks if each component of the current vector is equal to the corresponding component of the other vector.
     *
     * @param other The vector to compare with the current vector.
     * @return True if the vectors are equal (i.e., all components are equal), false otherwise.
     */
    constexpr bool operator==(const Vector2& other) const
    {
        return (x == other.x) && (y == other.y);
    }

    /**
     * @brief Inequality operator.
     *
     * Checks if any component of the current vector is not equal to the corresponding component of the other vector.
     *
     * @param other The vector to compare with the current vector.
     * @return True if the vectors are not equal (i.e., any component is not equal), false otherwise.
     */
    constexpr bool operator!=(const Vector2& other) const
    {
        return (x != other.x) || (y != other.y);
    }

    /**
     * @brief Subtraction and assignment operator with a scalar.
     *
     * Subtracts the given scalar value from each component of the current vector and assigns the result back to the current vector.
     *
     * @param scalar The scalar value to subtract from each component.
     * @return A reference to the modified current vector.
     */
    Vector2& operator-=(T scalar)
    {
        x -= scalar;
        y -= scalar;
        return *this;
    }

    /**
     * @brief Addition and assignment operator with a scalar.
     *
     * Adds the given scalar value to each component of the current vector and assigns the result back to the current vector.
     *
     * @param scalar The scalar value to add to each component.
     * @return A reference to the modified current vector.
     */
    Vector2& operator+=(T scalar)
    {
        x += scalar;
        y += scalar;
        return *this;
    }

    /**
     * @brief Multiplication and assignment operator with a scalar.
     *
     * Multiplies each component of the current vector by the given scalar value and assigns the result back to the current vector.
     *
     * @param scalar The scalar value to multiply each component by.
     * @return A reference to the modified current vector.
     */
    Vector2& operator*=(T scalar)
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    /**
     * @brief Division and assignment operator with a scalar.
     *
     * Divides each component of the current vector by the given scalar value and assigns the result back to the current vector.
     *
     * @param scalar The scalar value to divide each component by.
     * @return A reference to the modified current vector.
     */
    Vector2& operator/=(T scalar)
    {
        if constexpr (std::is_floating_point<T>::value) {
            const T inv = static_cast<T>(1.0) / scalar;
            x *= inv, y *= inv;
            return *this;
        }
        x /= scalar;
        y /= scalar;
        return *this;
    }

    /**
     * @brief Subtraction and assignment operator.
     *
     * Subtracts each component of the other vector from the corresponding component of the current vector and assigns the result back to the current vector.
     *
     * @param other The vector to subtract from the current vector.
     * @return A reference to the modified current vector.
     */
    Vector2& operator-=(const Vector2& other)
    {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    /**
     * @brief Addition and assignment operator.
     *
     * Adds each component of the other vector to the corresponding component of the current vector and assigns the result back to the current vector.
     *
     * @param other The vector to add to the current vector.
     * @return A reference to the modified current vector.
     */
    Vector2& operator+=(const Vector2& other)
    {
        x += other.x;
        y += other.y;
        return *this;
    }

    /**
     * @brief Multiplication and assignment operator with another vector.
     *
     * Multiplies each component of the current vector by the corresponding component of the other vector and assigns the result back to the current vector.
     *
     * @param other The vector to multiply with the current vector.
     * @return A reference to the modified current vector.
     */
    Vector2& operator*=(const Vector2& other)
    {
        x *= other.x;
        y *= other.y;
        return *this;
    }

    /**
     * @brief Division and assignment operator with another vector.
     *
     * Divides each component of the current vector by the corresponding component of the other vector
     * and assigns the result back to the current vector.
     * 
     * @warning No check is performed for division by zero. If any component of the `other` vector is zero,
     * the result will be undefined (typically resulting in infinity or NaN).
     *
     * @param other The vector by which to divide the current vector.
     * @return A reference to the modified current vector.
     */
    Vector2& operator/=(const Vector2& other)
    {
        x /= other.x;
        y /= other.y;
        return *this;
    }

    /**
     * @brief Overload of the output stream operator for Vector2.
     *
     * Allows a Vector2 to be output to an output stream (e.g., std::cout) in the format: Vec2(x, y).
     *
     * @param os The output stream (e.g., std::cout).
     * @param v The vector to be output.
     * @return A reference to the modified output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
        os << "Vec2(" << v.x << ", " << v.y << ")";
        return os;
    }

    /**
     * @brief Method to check if the vector is equal to (0,0).
     *
     * @return True if the vector is equal to (0,0), false otherwise.
     */
    constexpr bool is_zero() const
    {
        return !(x != 0 || y != 0);
    }

    /**
    * @brief Calculate the reciprocal of the vector components.
    *
    * @return A new Vector2 object with each component as the reciprocal of the original.
    */
    Vector2 rcp() const
    {
        static_assert(std::is_floating_point<T>::value, "T must be a floating-point type.");
        return { T(1.0) / x, T(1.0) / y };
    }

    /**
     * @brief Function to calculate the length (magnitude) of the vector.
     *
     * @return The length (magnitude) of the vector.
     */
    T length() const
    {
        return std::sqrt(x * x + y * y);
    }

    /**
     * @brief Function to calculate the squared length of the vector.
     *
     * @return The squared length of the vector.
     */
    constexpr T length_sq() const
    {
        return x * x + y * y;
    }

    /**
     * @brief Method to calculate the dot product of the vector with another vector.
     *
     * @param other The other vector to calculate the dot product with.
     * @return The dot product of the two vectors.
     */
    constexpr T dot(const Vector2& other) const
    {
        return x * other.x + y * other.y;
    }

    /**
     * @brief Method to normalize the vector.
     *
     * Normalizes the current vector, making it a unit vector (a vector with a magnitude of 1).
     * If the magnitude of the vector is zero, no operation is performed.
     */
    void normalize()
    {
        const T len = length();
        if (len != 0.0) (*this) *= 1.0 / len;
    }

    /**
     * @brief Method to get a normalized vector.
     *
     * Returns a new vector that is the normalized version of the current vector.
     *
     * @return A normalized vector.
     */
    Vector2 normalized() const
    {
        Vector2 result(*this);
        result.normalize();
        return result;
    }

    /**
     * @brief Method to calculate the Euclidean distance between two vectors.
     *
     * @param other The other vector to calculate the distance to.
     * @return The Euclidean distance between the two vectors.
     */
    T distance(const Vector2& other) const
    {
        return (*this - other).length();
    }

    /**
     * @brief Function to calculate the squared Euclidean distance between two vectors.
     *
     * @param other The other vector to calculate the squared distance to.
     * @return The squared Euclidean distance between the two vectors.
     */
    constexpr T distance_sq(const Vector2& other) const
    {
        const Vector2 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y;
    }

    /**
     * @brief Function to rotate the vector around the origin by an angle in radians.
     *
     * @param angle The angle in radians by which to rotate the vector.
     */
    void rotate(T angle)
    {
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        x = x * c - y * s, y = x * s + y * c;
    }

    /**
     * @brief Function to get a rotated copy of the vector around the origin by an angle in radians.
     *
     * @param angle The angle in radians by which to rotate the vector.
     * @return A rotated copy of the vector.
     */
    Vector2 rotated(T angle) const
    {
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        return Vector2(x * c - y * s, x * s + y * c);
    }

    /**
     * @brief Function to rotate the vector around the origin by given cosine and sine values.
     *
     * @param c The cosine value of the rotation angle.
     * @param s The sine value of the rotation angle.
     */
    void rotate(T c, T s)
    {
        x = x * c - y * s;
        y = x * s + y * c;
    }

    /**
     * @brief Function to get a rotated copy of the vector around the origin by given cosine and sine values.
     *
     * @param c The cosine value of the rotation angle.
     * @param s The sine value of the rotation angle.
     * @return A rotated copy of the vector.
     */
    Vector2 rotated(T c, T s) const
    {
        return Vector2(x * c - y * s, x * s + y * c);
    }

    /**
     * @brief Function to reflect the vector with respect to a given normal vector.
     *
     * Reflects the vector with respect to the provided normal vector assuming 'normal' is a unit vector.
     * The reflection is computed using the formula: v - 2 * dot(v, normal) * normal.
     *
     * @param normal The normal vector used for reflection.
     * @return The reflected vector.
     */
    constexpr Vector2 reflect(const Vector2& normal) const
    {
        T dot = this->Dot(normal);
        return Vector2(
            x - 2.0 * dot * normal.x,
            y - 2.0 * dot * normal.y
        );
    }

    /**
     * @brief Function to calculate the angle in radians of the vector with respect to the positive x-axis.
     *
     * @return The angle in radians of the vector with respect to the positive x-axis.
     */
    T angle() const
    {
        return std::atan2(y, x);
    }

    /**
     * @brief Function to calculate the angle in radians between two vectors with respect to the positive x-axis.
     *
     * @param other The other vector to calculate the angle with respect to.
     * @return The angle in radians between the two vectors with respect to the positive x-axis.
     */
    T angle(const Vector2& other) const
    {
        return std::atan2(y - other.y, x - other.x);
    }

    /**
     * @brief Function to transform the vector by a 2D transformation matrix.
     *
     * Transforms the vector using the provided 2D transformation matrix.
     * The transformation is applied as a 2x3 matrix multiplication: [x' y'] = [x y 1] * matrix.
     *
     * @param matrix The 2D transformation matrix to apply.
     */
    constexpr void transform(const Mat3& matrix)
    {
     *this = {
            x * matrix.m[0] + y * matrix.m[3] + matrix.m[6],
            x * matrix.m[1] + y * matrix.m[4] + matrix.m[7]
        };
    }

    /**
     * @brief Function to transform the vector by a 2D transformation matrix and return the result.
     *
     * Transforms the vector using the provided 2D transformation matrix and returns the transformed vector.
     * The transformation is applied as a 2x3 matrix multiplication: [x' y'] = [x y 1] * matrix.
     *
     * @param matrix The 2D transformation matrix to apply.
     * @return The transformed vector.
     */
    constexpr Vector2 transformed(const Mat3& matrix)
    {
        return {
            x * matrix.m[0] + y * matrix.m[3] + matrix.m[6],
            x * matrix.m[1] + y * matrix.m[4] + matrix.m[7]
        };
    }

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
    constexpr Vector2 move_towards(Vector2 b, T delta)
    {
        T dx = b.x - x;
        T dy = b.y - y;
        return {
            (std::abs(dx) > delta) ? x + (delta * (dx > T(0) ? T(1) : T(-1))) : b.x,
            (std::abs(dy) > delta) ? y + (delta * (dy > T(0) ? T(1) : T(-1))) : b.y
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
    constexpr Vector2 lerp(Vector2 b, T t)
    {
        static_assert(std::is_floating_point<T>::value, "Only floating-point types are allowed.");
        return {
            x + t * (b.x - x),
            y + t * (b.y - y)
        };
    }

    /**
     * @brief Function to calculate the component-wise minimum between the current vector and another vector.
     *
     * @param other The other vector to compare with.
     * @return A vector where each component is the minimum of the corresponding components of the current vector and the other vector.
     */
    constexpr Vector2 min(const Vector2& other) const
    {
        return {
            std::min(x, other.x),
            std::min(y, other.y)
        };
    }

    /**
     * @brief Function to calculate the component-wise maximum between the current vector and another vector.
     *
     * @param other The other vector to compare with.
     * @return A vector where each component is the maximum of the corresponding components of the current vector and the other vector.
     */
    constexpr Vector2 max(const Vector2& other) const
    {
        return {
            std::max(x, other.x),
            std::max(y, other.y)
        };
    }

    /**
     * @brief Function to clamp each component of the current vector within a specified range defined by minimum and maximum vectors.
     *
     * @param min The minimum vector defining the lower bounds for each component.
     * @param max The maximum vector defining the upper bounds for each component.
     * @return A vector where each component is clamped within the specified range.
     */
    constexpr Vector2 clamp(const Vector2& min, const Vector2& max) const
    {
        return {
            std::clamp(x, min.x, max.x),
            std::clamp(y, min.y, max.y)
        };
    }

    /**
     * @brief Function to clamp each component of the current vector within a specified scalar range defined by minimum and maximum values.
     *
     * @param min The minimum value defining the lower bounds for each component.
     * @param max The maximum value defining the upper bounds for each component.
     * @return A vector where each component is clamped within the specified scalar range.
     */
    constexpr Vector2 clamp(T min, T max) const
    {
        return {
            std::clamp(x, min, max),
            std::clamp(y, min, max)
        };
    }

    /**
     * @brief Function to calculate the absolute value of each component of the vector.
     *
     * @return A vector where each component is the absolute value of the corresponding component of the current vector.
     */
    Vector2 abs() const
    {
        return {
            std::abs(x),
            std::abs(y)
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

#endif // BPM_VEC2_HPP
