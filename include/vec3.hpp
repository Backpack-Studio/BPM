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

#ifndef BPM_VEC3_HPP
#define BPM_VEC3_HPP

#include "./vec2.hpp"

#include <algorithm>
#include <ostream>
#include <cmath>

namespace bpm {

struct Quat;
struct Mat4;

template <typename T>
struct Vector3;

using Vec3 = Vector3<float>;
using DVec3 = Vector3<double>;
using IVec3 = Vector3<int32_t>;
using UVec3 = Vector3<uint32_t>;

/**
 * @brief Represents a 3-dimensional vector.
 *
 * This class template represents a 3-dimensional vector with components x, y, and z.
 * It provides various mathematical operations and utilities for manipulating 3D vectors.
 *
 * @tparam T The type of the components of the vector.
 */
template <typename T>
struct Vector3
{
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    static constexpr int DIMENSIONS = 3;    ///< The number of dimensions of the vector.
    typedef T value_type;                   ///< The type of each component of the vector.
    T x, y, z;                              ///< The x, y, and z components of the vector.

    /**
     * @brief Default constructor.
     *
     * Initializes all components x, y, and z to zero.
     */
    constexpr Vector3()
        : x(0), y(0), z(0)
    { }

    /**
     * @brief Constructor initializing all components with a single value.
     *
     * Initializes all components x, y, and z with the same given value.
     *
     * @param value The value to set for all components.
     */
    constexpr explicit Vector3(T value)
        : x(value), y(value), z(value)
    { }

    /**
     * @brief Constructor initializing all components with specific values.
     *
     * Initializes the x, y, and z components with the given values.
     *
     * @param x The value to set for the x component.
     * @param y The value to set for the y component.
     * @param z The value to set for the z component.
     */
    constexpr Vector3(T x, T y, T z = 0)
        : x(x), y(y), z(z)
    { }

    /**
     * @brief Constructor initializing the vector from a 2D vector and an optional z value.
     *
     * Initializes the x and y components of the vector with the x and y values of the given 2D vector,
     * and sets the z component to the specified value (default is 0.0).
     *
     * @param Vec2 The 2D vector to initialize the x and y components with.
     * @param z The value to set for the z component (default is 0.0).
     */
    constexpr Vector3(const Vector2<T>& Vec2, T z = 0)
        : x(Vec2.x), y(Vec2.y), z(z)
    { }

    /**
     * @brief Conversion operator to convert the vector to a Vector3 of a different type.
     *
     * Converts the current vector to a Vector3 of a different type by casting its components to the new type.
     *
     * @tparam U The type to convert the components of the vector to.
     * @return A Vector3 containing the components of the current vector casted to the type U.
     */
    template <typename U>
    constexpr operator Vector3<U>() const {
        return Vector3<U>(
            static_cast<U>(x),
            static_cast<U>(y),
            static_cast<U>(z));
    }

    /**
     * @brief Subscript operator to access individual components of the vector.
     *
     * @param axis The index of the component to access (0 for x, 1 for y, 2 for z).
     * @return A reference to the component at the specified index.
     */
    constexpr T& operator[](int axis) {
        switch (axis) {
            case 0:  return x;
            case 1:  return y;
            default: return z;
        }
    }

    /**
     * @brief Subscript operator to access individual components of the vector (const version).
     *
     * @param axis The index of the component to access (0 for x, 1 for y, 2 for z).
     * @return A const reference to the component at the specified index.
     */
    constexpr const T& operator[](int axis) const {
        switch (axis) {
            case 0:  return x;
            case 1:  return y;
            default: return z;
        }
    }

    /**
     * @brief Unary negation operator.
     *
     * @return A new vector with each component negated.
     */
    constexpr Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    /**
     * @brief Subtraction operator with a scalar value.
     *
     * @param scalar The scalar value to subtract from each component of the vector.
     * @return A new vector resulting from the subtraction operation.
     */
    constexpr Vector3 operator-(T scalar) const {
        return Vector3(x - scalar, y - scalar, z - scalar);
    }

    /**
     * @brief Addition operator with a scalar value.
     *
     * @param scalar The scalar value to add to each component of the vector.
     * @return A new vector resulting from the addition operation.
     */
    constexpr Vector3 operator+(T scalar) const {
        return Vector3(x + scalar, y + scalar, z + scalar);
    }

    /**
     * @brief Multiplication operator with a scalar value.
     *
     * @param scalar The scalar value to multiply each component of the vector by.
     * @return A new vector resulting from the multiplication operation.
     */
    constexpr Vector3 operator*(T scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    /**
     * @brief Division operator by a scalar value.
     *
     * Divides each component of the vector by the given scalar value.
     *
     * @warning If the scalar is zero, the behavior is undefined. This function does not check for division by zero.
     *          For floating-point types, division by zero may result in infinity or NaN.
     *
     * @param scalar The scalar value to divide each component of the vector by.
     * @return A new vector resulting from the division operation.
     */
    constexpr Vector3 operator/(T scalar) const {
        if constexpr (std::is_floating_point<T>::value) {
            const T inv = static_cast<T>(1.0) / scalar;
            return Vector3(x * inv, y * inv, z * inv);
        }
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    /**
     * @brief Subtraction operator between two vectors.
     *
     * @param other The vector to subtract from the current vector.
     * @return A new vector resulting from the subtraction operation.
     */
    constexpr Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    /**
     * @brief Addition operator between two vectors.
     *
     * @param other The vector to add to the current vector.
     * @return A new vector resulting from the addition operation.
     */
    constexpr Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    /**
     * @brief Multiplication operator with another vector.
     *
     * @param other The vector to multiply element-wise.
     * @return A new vector resulting from the element-wise multiplication.
     */
    constexpr Vector3 operator*(const Vector3& other) const {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }

    /**
     * @brief Division operator by another vector.
     *
     * Divides each component of the current vector by the corresponding component of the other vector.
     *
     * @warning If any component of the `other` vector is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param other The vector to divide element-wise.
     * @return A new vector resulting from the element-wise division.
     */
    constexpr Vector3 operator/(const Vector3& other) const {
        return Vector3(x / other.x, y / other.y, z / other.z);
    }

    /**
     * @brief Equality operator.
     *
     * @param other The vector to compare with.
     * @return True if the vectors are equal (all components are equal), false otherwise.
     */
    constexpr bool operator==(const Vector3& other) const {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }

    /**
     * @brief Inequality operator.
     *
     * @param other The vector to compare with.
     * @return True if the vectors are not equal (at least one component is different), false otherwise.
     */
    constexpr bool operator!=(const Vector3& other) const {
        return (x != other.x) || (y != other.y) || (z != other.z);
    }

    /**
     * @brief Subtraction and assignment operator with a scalar value.
     *
     * @param scalar The scalar value to subtract from each component of the vector.
     * @return A reference to the modified vector.
     */
    Vector3& operator-=(T scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }

    /**
     * @brief Addition and assignment operator with a scalar value.
     *
     * @param scalar The scalar value to add to each component of the vector.
     * @return A reference to the modified vector.
     */
    Vector3& operator+=(T scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }

    /**
     * @brief Multiplication and assignment operator with a scalar value.
     *
     * @param scalar The scalar value to multiply each component of the vector by.
     * @return A reference to the modified vector.
     */
    Vector3& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    /**
     * @brief Division and assignment operator with a scalar value.
     *
     * Divides each component of the vector by the given scalar value and assigns the result back to the vector.
     *
     * @warning If the scalar is zero, the behavior is undefined. This function does not check for division by zero.
     *          For floating-point types, division by zero may result in infinity or NaN.
     *
     * @param scalar The scalar value to divide each component of the vector by.
     * @return A reference to the modified vector.
     */
    Vector3& operator/=(T scalar) {
        if constexpr (std::is_floating_point<T>::value) {
            const T inv = static_cast<T>(1.0) / scalar;
            x *= inv, y *= inv, z *= inv;
            return *this;
        }
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    /**
     * @brief Subtraction and assignment operator with another vector.
     *
     * @param other The vector to subtract.
     * @return Reference to the modified vector after subtraction.
     */
    Vector3& operator-=(const Vector3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    /**
     * @brief Addition and assignment operator with another vector.
     *
     * @param other The vector to add.
     * @return Reference to the modified vector after addition.
     */
    Vector3& operator+=(const Vector3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    /**
     * @brief Multiplication and assignment operator with another vector.
     *
     * @param other The vector to multiply.
     * @return Reference to the modified vector after multiplication.
     */
    Vector3& operator*=(const Vector3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    /**
     * @brief Division and assignment operator with another vector.
     *
     * Divides each component of the current vector by the corresponding component of the other vector
     * and assigns the result back to the current vector.
     *
     * @warning If any component of the `other` vector is zero, the behavior is undefined. This function does not check for division by zero,
     *          which may result in infinity or NaN for floating-point types.
     *
     * @param other The vector to divide.
     * @return A reference to the modified vector after division.
     */
    Vector3& operator/=(const Vector3& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        return *this;
    }

    /**
     * @brief Overload of the output stream operator for Vector3.
     *
     * Allows a Vector3 to be output to an output stream (e.g., std::cout) in the format: Vec3(x, y, z).
     *
     * @param os The output stream (e.g., std::cout).
     * @param v The vector to be output.
     * @return A reference to the modified output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
        os << "Vec3(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }

    /**
     * @brief Method to check if the vector is equal to (0,0,0).
     *
     * @return True if the vector is equal to (0,0,0), false otherwise.
     */
    bool is_zero() const {
        return !(x + y + z);
    }

    /**
    * @brief Calculate the reciprocal of the vector components.
    *
    * @return A new Vector2 object with each component as the reciprocal of the original.
    */
    Vector3 rcp() const {
        static_assert(std::is_floating_point<T>::value, "T must be a floating-point type.");
        return { T(1.0) / x, T(1.0) / y, T(1.0) / z };
    }

    /**
     * @brief Function to calculate the length (magnitude) of the vector.
     *
     * @return The length (magnitude) of the vector.
     */
    T length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    /**
     * @brief Function to calculate the length squared of the vector.
     *
     * @return The length squared of the vector.
     */
    T length_sq() const {
        return x * x + y * y + z * z;
    }

    /**
     * @brief Method to calculate the dot product of the vector with another vector.
     *
     * @param other The other vector for dot product calculation.
     * @return The dot product of the two vectors.
     */
    T dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    /**
     * @brief Method to normalize the vector.
     *
     * If the magnitude of the vector is not zero, this method normalizes the vector.
     */
    void normalize() {
        const T mag = length();
        if (mag != 0.0) (*this) *= 1.0 / mag;
    }

    /**
     * @brief Method to obtain a normalized vector.
     *
     * @return A normalized vector.
     */
    Vector3 normalized() const {
        Vector3 result(*this);
        result.normalize();
        return result;
    }

    /**
     * @brief Makes vectors normalized and orthogonal to each other using Gram-Schmidt process.
     *
     * @param tangent A vector orthogonal to this vector after normalization.
     */
    void ortho_normalize(Vector3& tangent) {
        this->normalize();
        tangent = this->cross(tangent).normalized().cross(*this);
    }

    /**
     * @brief Method to calculate the distance between two vectors.
     *
     * @param other The other vector to calculate the distance to.
     * @return The distance between this vector and the other vector.
     */
    T distance(const Vector3& other) const {
        return (*this - other).length();
    }

    /**
     * @brief Function to calculate the distance squared between two vectors.
     *
     * @param other The other vector to calculate the distance to.
     * @return The distance squared between this vector and the other vector.
     */
    T distance_sq(const Vector3& other) const {
        const Vector3 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }

    /**
     * @brief Function to calculate the angle between two vectors.
     *
     * @param other The other vector.
     * @return The angle between the two vectors in radians.
     */
    T angle(const Vector3& other) {
        return std::atan2(this->cross(other).length(), this->dot(other));
    }

    /**
     * @brief Function to rotate the vector around an axis by a certain angle (Euler-Rodrigues).
     *
     * @param axis The axis of rotation.
     * @param angle The angle of rotation in radians.
     */
    void rotate(Vector3 axis, T angle) {
        axis.normalize();
        angle *= 0.5f;

        Vector3 w = axis * std::sin(angle);
        Vector3 wv = w.cross(*this);
        Vector3 wwv = w.cross(wv);

        wv *= 2 * std::cos(angle);
        wwv *= 2;

        (*this) += wv + wwv;
    }

    /**
    * @brief Function to rotate the vector using Euler angles (yaw, pitch, roll).
    *
    * @param euler The Euler angles in radians, where x is pitch, y is yaw, and z is roll.
    */
    void rotate(const Vector3& euler) {
        T pitch = euler.x;
        T yaw = euler.y;
        T roll = euler.z;

        T cosYaw = std::cos(yaw);
        T sinYaw = std::sin(yaw);
        Vector3<T> yawRot(
            cosYaw * x + sinYaw * z,
            y,
            -sinYaw * x + cosYaw * z
        );

        T cosPitch = std::cos(pitch);
        T sinPitch = std::sin(pitch);
        Vector3<T> pitchRot(
            yawRot.x,
            cosPitch * yawRot.y - sinPitch * yawRot.z,
            sinPitch * yawRot.y + cosPitch * yawRot.z
        );

        T cosRoll = std::cos(roll);
        T sinRoll = std::sin(roll);
        Vector3<T> rollRot(
            cosRoll * pitchRot.x - sinRoll * pitchRot.y,
            sinRoll * pitchRot.x + cosRoll * pitchRot.y,
            pitchRot.z
        );

        x = rollRot.x;
        y = rollRot.y;
        z = rollRot.z;
    }

    /**
     * @brief Function to rotate the vector by a quaternion.
     *
     * @param q The quaternion representing the rotation.
     */
    void rotate(const Quat& q); ///< NOTE: Implemented in 'quat.hpp'

    /**
     * @brief Function to calculate the rotation of the vector by an axis and an angle.
     *
     * @param axis The axis of rotation.
     * @param angle The angle of rotation in radians.
     * @return The rotated vector.
     */
    Vector3 rotated(const Vector3& axis, T angle) const {
        Vector3 result(*this);
        result.rotate(axis, angle);
        return result;
    }

    /**
    * @brief Function to rotate the vector using Euler angles (yaw, pitch, roll).
    *
    * @param euler The Euler angles in radians, where x is pitch, y is yaw, and z is roll.
    * @return The rotated vector.
    */
    Vector3 rotated(const Vector3& euler) {
        Vector3 result(*this);
        result.rotate(euler);
        return result;
    }

    /**
     * @brief Function to calculate the rotation of the vector by a quaternion.
     *
     * @param q The quaternion representing the rotation.
     * @return The rotated vector.
     */
    Vector3 rotated(const Quat& q) {
        Vector3 result(*this);
        result.rotate(q);
        return result;
    }

    /**
     * @brief Function to perform a reflection of the vector with respect to another vector.
     *
     * @param normal The normal vector (assumed to be a unit vector).
     * @return The reflected vector.
     */
    Vector3 reflect(const Vector3& normal) const {
        T dot = this->dot(normal);
        return Vector3(
            x - 2.0f * this->dot(normal) * normal.x,
            y - 2.0f * this->dot(normal) * normal.y,
            z - 2.0f * this->dot(normal) * normal.z);
    }

    /**
     * @brief Function to perform a cross product of the vector with another vector.
     *
     * @param other The other vector.
     * @return The cross product vector.
     */
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    /**
     * @brief Function to obtain the direction vector from this vector to another vector.
     *
     * @param other The target vector.
     * @return The direction vector from this vector to the target vector.
     */
    Vector3 direction(const Vector3& other) const {
        return (other - *this).normalized();
    }

    /**
     * @brief Function to transform the vector by a 3x3 matrix.
     *
     * @param matrix The 3x3 transformation matrix.
     */
    void transform(const Mat3& matrix) {
     *this = {
            x * matrix.m[0] + y * matrix.m[3] + z * matrix.m[6],
            x * matrix.m[1] + y * matrix.m[4] + z * matrix.m[7],
            x * matrix.m[2] + y * matrix.m[5] + z * matrix.m[8]
        };
    }

    /**
     * @brief Function to obtain the vector transformed by a 3x3 matrix.
     *
     * @param matrix The 3x3 transformation matrix.
     * @return The transformed vector.
     */
    Vector3 transformed(const Mat3& matrix) const {
        return {
            x * matrix.m[0] + y * matrix.m[3] + z * matrix.m[6],
            x * matrix.m[1] + y * matrix.m[4] + z * matrix.m[7],
            x * matrix.m[2] + y * matrix.m[5] + z * matrix.m[8]
        };
    }

    /**
     * @brief Function to transform the vector by a 4x4 matrix.
     *
     * @param matrix The 4x4 transformation matrix.
     */
    void transform(const Mat4& matrix); ///< NOTE: Defined in 'mat4.hpp'

    /**
     * @brief Function to obtain the vector transformed by a 4x4 matrix.
     *
     * @param matrix The 4x4 transformation matrix.
     * @return The transformed vector.
     */
    Vector3 transformed(const Mat4& matrix) const;  ///< NOTE: Defined in 'mat4.hpp'

    /**
     * @brief Calculate a perpendicular vector to the given vector.
     *
     * @param other The input vector.
     * @return A perpendicular vector to the input vector.
     */
    static Vector3 perpendicular(const Vector3& other) {
        Vector3 cardinalAxis = {1.0, 0.0, 0.0};
        const Vector3 oabs = other.Abs();
        T min = oabs.x;
        if (oabs.y < min) {
            min = oabs.y;
            cardinalAxis = {0.0, 1.0, 0.0};
        }
        if (oabs.z < min) {
            cardinalAxis = {0.0, 0.0, 1.0};
        }
        return other.Cross(cardinalAxis);
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
    constexpr Vector3 move_towards(Vector3 b, T delta) {
        T dx = b.x - x;
        T dy = b.y - y;
        T dz = b.z - z;
        return {
            (std::abs(dx) > delta) ? x + (delta * (dx > T(0) ? T(1) : T(-1))) : b.x,
            (std::abs(dy) > delta) ? y + (delta * (dy > T(0) ? T(1) : T(-1))) : b.y,
            (std::abs(dz) > delta) ? z + (delta * (dz > T(0) ? T(1) : T(-1))) : b.z
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
    constexpr Vector3 lerp(Vector3 b, T t) {
        static_assert(std::is_floating_point<T>::value, "Only floating-point types are allowed.");
        return {
            x + t * (b.x - x),
            y + t * (b.y - y),
            z + t * (b.z - z)
        };
    }

    /**
     * @brief Get the component-wise minimum of this vector and another vector.
     *
     * @param other The other vector.
     * @return A vector containing the component-wise minimum values.
     */
    Vector3 min(const Vector3& other) const {
        return {
            std::min(x, other.x),
            std::min(y, other.y),
            std::min(z, other.z)
        };
    }

    /**
     * @brief Get the component-wise maximum of this vector and another vector.
     *
     * @param other The other vector.
     * @return A vector containing the component-wise maximum values.
     */
    Vector3 max(const Vector3& other) const {
        return {
            std::max(x, other.x),
            std::max(y, other.y),
            std::max(z, other.z)
        };
    }

    /**
     * @brief Clamp each component of this vector between the corresponding components of min and max vectors.
     *
     * @param min The vector containing the minimum values.
     * @param max The vector containing the maximum values.
     * @return A vector with each component clamped between the corresponding components of min and max.
     */
    Vector3 clamp(const Vector3& min, const Vector3& max) const {
        return {
            std::clamp(x, min.x, max.x),
            std::clamp(y, min.y, max.y),
            std::clamp(z, min.z, max.z)
        };
    }

    /**
     * @brief Clamp each component of this vector between a minimum and maximum value.
     *
     * @param min The minimum value.
     * @param max The maximum value.
     * @return A vector with each component clamped between the minimum and maximum values.
     */
    Vector3 clamp(T min, T max) const {
        return {
            std::clamp(x, min, max),
            std::clamp(y, min, max),
            std::clamp(z, min, max)
        };
    }

    /**
     * @brief Get the absolute value of each component of this vector.
     *
     * @return A vector containing the absolute values of each component.
     */
    Vector3 abs() const {
        return {
            std::abs(x),
            std::abs(y),
            std::abs(z)
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

#endif // BPM_VEC3_HPP
