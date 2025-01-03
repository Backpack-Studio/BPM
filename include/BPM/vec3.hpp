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

#include "./vecx.hpp"
#include "./vec2.hpp"

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
class Vector3 : public Vector<T, 3, Vector3<T>>
{
public:
    /**
     * @brief Default constructor.
     *
     * Initializes all components x, y, and z to zero.
     */
    constexpr Vector3()
        : Vector<T, 3, Vector3<T>>()
    { }

    /**
     * @brief Constructor initializing all components with a single value.
     *
     * Initializes all components x, y, and z with the same given value.
     *
     * @param value The value to set for all components.
     */
    constexpr explicit Vector3(T value)
        : Vector<T, 3, Vector3<T>>({ value, value, value })
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
        : Vector<T, 3, Vector3<T>>({ x, y, z })
    { }

    /**
     * @brief Constructor initializing the vector from a 2D vector and an optional z value.
     *
     * Initializes the x and y components of the vector with the x and y values of the given 2D vector,
     * and sets the z component to the specified value (default is 0.0).
     *
     * @param vec2 The 2D vector to initialize the x and y components with.
     * @param z The value to set for the z component (default is 0.0).
     */
    constexpr Vector3(const Vector2<T>& v, T z = 0)
        : Vector<T, 3, Vector3<T>>({ v[0], v[1], z })
    { }

    /**
     * @brief Constructor initializing all three components from a tuple.
     *
     * This constructor extracts the three elements from the given tuple
     * and uses them to initialize the x, y, and z components of the vector.
     *
     * @param t A tuple containing three elements, where the first element is used to 
     * initialize the x component, the second element to initialize the y component,
     * and the third element to initialize the z component.
     */
    constexpr Vector3(const std::tuple<T, T, T>& t)
        : Vector<T, 3, Vector2<T>>({
            std::get<0>(t),
            std::get<1>(t),
            std::get<2>(t)
        })
    { }

    /**
     * @brief Constructor that converts a `Vector3<U>` to a `Vector3<T>`.
     *
     * This constructor creates a `Vector3<T>` by copying the components from a given 
     * `Vector3<U>`. Each component of the input vector `v` is used to initialize the
     * corresponding component of the output vector, where `T` and `U` may be different
     * types. This is useful for converting between different types of vector components 
     * (e.g., from `Vector3<float>` to `Vector3<double>`).
     *
     * @tparam U The type of the components in the input vector `v`.
     * @tparam T The type of the components in the output vector (the type of the current vector).
     *
     * @param v The input `Vector3<U>` to convert to a `Vector3<T>`.
     */
    template <typename U>
    constexpr Vector3(const Vector3<U>& v)
        : Vector3<T>(v[0], v[1], v[2])
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
        return Vector<U, 3, Vector3<U>>({ this->v[0], this->v[1], this->v[2] });
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
        os << "Vec3(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
        return os;
    }

    /**
     * @brief Converts a 3D vector to a 2D vector by discarding the z component.
     * 
     * This method returns a `Vector2` containing the x and y components of the original
     * 3D vector. The z component is discarded, which can be useful in certain 2D contexts
     * where only the x and y components are relevant.
     * 
     * @tparam T Type of the vector components.
     * @return Vector2<T> A new 2D vector containing the x and y components.
     */
    constexpr Vector2<T> xy() const { return { this->v[0], this->v[1] }; }

    /**
     * @brief Accessor for the x component of the vector.
     * 
     * This method returns a reference to the x component of the vector. The x component 
     * is typically the first element in a 3D vector.
     * 
     * @return T& Reference to the x component.
     */
    constexpr T& x() { return this->v[0]; }

    /**
     * @brief Accessor for the y component of the vector.
     * 
     * This method returns a reference to the y component of the vector. The y component 
     * is typically the second element in a 3D vector.
     * 
     * @return T& Reference to the y component.
     */
    constexpr T& y() { return this->v[1]; }

    /**
     * @brief Accessor for the z component of the vector.
     * 
     * This method returns a reference to the z component of the vector. The z component 
     * is typically the third element in a 3D vector.
     * 
     * @return T& Reference to the z component.
     */
    constexpr T& z() { return this->v[2]; }

    /**
     * @brief Const accessor for the x component of the vector.
     * 
     * This method returns a const reference to the x component of the vector. The x component 
     * is typically the first element in a 3D vector.
     * 
     * @return const T& Const reference to the x component.
     */
    constexpr const T& x() const { return this->v[0]; }

    /**
     * @brief Const accessor for the y component of the vector.
     * 
     * This method returns a const reference to the y component of the vector. The y component 
     * is typically the second element in a 3D vector.
     * 
     * @return const T& Const reference to the y component.
     */
    constexpr const T& y() const { return this->v[1]; }

    /**
     * @brief Const accessor for the z component of the vector.
     * 
     * This method returns a const reference to the z component of the vector. The z component 
     * is typically the third element in a 3D vector.
     * 
     * @return const T& Const reference to the z component.
     */
    constexpr const T& z() const { return this->v[2]; }

    /**
     * @brief Mutator for the x component of the vector.
     * 
     * This method sets the x component of the vector. The x component is typically the first 
     * element in a 3D vector.
     * 
     * @param value The new value to set for the x component.
     */
    constexpr void x(T value) { this->v[0] = value; }

    /**
     * @brief Mutator for the y component of the vector.
     * 
     * This method sets the y component of the vector. The y component is typically the second 
     * element in a 3D vector.
     * 
     * @param value The new value to set for the y component.
     */
    constexpr void y(T value) { this->v[1] = value; }

    /**
     * @brief Mutator for the z component of the vector.
     * 
     * This method sets the z component of the vector. The z component is typically the third 
     * element in a 3D vector.
     * 
     * @param value The new value to set for the z component.
     */
    constexpr void z(T value) { this->v[2] = value; }
};


/* Vector3 Algorithms Implementation */

/**
 * @brief Ortho-normalizes two 3D vectors using the Gram-Schmidt process.
 * 
 * This function applies the Gram-Schmidt process to ortho-normalize two 3D vectors, `v1` and `v2`.
 * After this operation, `v1` remains normalized with a magnitude of 1, and `v2` is transformed to 
 * be orthogonal to `v1` and normalized. This results in an orthonormal basis where both vectors 
 * are unit length and mutually perpendicular.
 * 
 * @tparam T The type of the vector components (typically a floating-point type, e.g., float or double).
 * 
 * @param v1 A pointer to the first 3D vector. After the operation, `v1` will be normalized but 
 *           otherwise unchanged in direction.
 * 
 * @param v2 A pointer to the second 3D vector. After the operation, `v2` will be orthogonal 
 *           to `v1` and normalized, completing an orthonormal basis.
 * 
 * @note This function requires that the type `T` is a floating-point type.
 * 
 * @details
 * The process follows these steps:
 * 1. Normalize `v1`, preserving its direction.
 * 2. Adjust `v2` to be orthogonal to `v1` by projecting it onto the plane orthogonal to `v1`.
 * 3. Normalize the adjusted `v2` to ensure it has a unit length.
 */
template <typename T>
inline void ortho_normalize(Vector3<T>* v1, Vector3<T>* v2) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    normalize(*v1); *v2 = cross(normalized(cross(*v1, *v2)), *v1);
}

/**
 * @brief Computes the cross product of two 3D vectors.
 * 
 * This function computes the cross product of two 3D vectors `v1` and `v2`.
 * The cross product yields a vector that is perpendicular to both `v1` and `v2`.
 * 
 * @tparam T The type of the vector components (should be a floating-point type).
 * @param v1 The first 3D vector.
 * @param v2 The second 3D vector.
 * @return The cross product vector of `v1` and `v2`.
 */
template <typename T>
inline constexpr Vector3<T> cross(const Vector3<T>& v1, const Vector3<T>& v2) {
    return Vector3<T> {
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    };
}

/**
 * @brief Function to calculate the angle between two vectors.
 *
 * @param other The other vector.
 * @return The angle between the two vectors in radians.
 */
template <typename T>
inline T angle(const Vector3<T>& v1, const Vector3<T>& v2) {
    return std::atan2(length(cross(v1, v2)), dot(v1, v2));
}

/**
 * @brief Function to rotate the vector around an axis by a certain angle (Euler-Rodrigues).
 *
 * @param axis The axis of rotation.
 * @param angle The angle of rotation in radians.
 */
template <typename T>
inline Vector3<T> rotate(const Vector3<T>& v, const Vector3<T>& axis, T angle) {

    angle *= 0.5f;

    Vector3 w = axis * std::sin(angle);
    Vector3 wv = cross(w, v);
    Vector3 wwv = cross(w, wv);

    wv *= 2 * std::cos(angle);
    wwv *= 2;

    return v + wv + wwv;
}

/**
 * @brief Function to rotate the vector using Euler angles (yaw, pitch, roll).
 *
 * @param euler The Euler angles in radians, where x is pitch, y is yaw, and z is roll.
 */
template <typename T>
inline Vector3<T> rotate(const Vector3<T>& v, const Vector3<T>& euler) {
    T pitch = euler[0];
    T yaw = euler[1];
    T roll = euler[2];

    T cos_yaw = std::cos(yaw);
    T sin_yaw = std::sin(yaw);
    Vector3<T> yaw_rot(
        cos_yaw * v[0] + sin_yaw * v[2],
        v[1],
        -sin_yaw * v[0] + cos_yaw * v[2]
    );

    T cos_pitch = std::cos(pitch);
    T sin_pitch = std::sin(pitch);
    Vector3<T> pitch_rot(
        yaw_rot[0],
        cos_pitch * yaw_rot[1] - sin_pitch * yaw_rot[2],
        sin_pitch * yaw_rot[1] + cos_pitch * yaw_rot[2]
    );

    T cos_roll = std::cos(roll);
    T sin_roll = std::sin(roll);
    Vector3<T> roll_rot(
        cos_roll * pitch_rot[0] - sin_roll * pitch_rot[1],
        sin_roll * pitch_rot[0] + cos_roll * pitch_rot[1],
        pitch_rot[2]
    );

    return roll_rot;
}

/**
 * @brief Function to perform a reflection of the vector with respect to another vector.
 *
 * @param normal The normal vector (assumed to be a unit vector).
 * @return The reflected vector.
 */
template <typename T>
inline Vector3<T> reflect(const Vector3<T>& v, const Vector3<T>& normal) {
    T d = dot(v, normal);
    return Vector3(
        v[0] - 2.0f * dot(v, normal) * normal[0],
        v[1] - 2.0f * dot(v, normal) * normal[1],
        v[2] - 2.0f * dot(v, normal) * normal[2]);
}

/**
 * @brief Calculate a perpendicular vector to the given vector.
 *
 * @param other The input vector.
 * @return A perpendicular vector to the input vector.
 */
template <typename T>
Vector3<T> perpendicular(const Vector3<T>& v) {
    Vector3<T> cardinal_axis = { 1.0, 0.0, 0.0 };
    const Vector3 oabs = abs(v);
    T min = oabs[0];
    if (oabs[1] < min) {
        min = oabs[1];
        cardinal_axis = { 0.0, 1.0, 0.0 };
    }
    if (oabs[2] < min) {
        cardinal_axis = { 0.0, 0.0, 1.0 };
    }
    return cross(v, cardinal_axis);
}

} // namespace bpm

#endif // BPM_VEC3_HPP
