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

#include "./vecx.hpp"

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
class Vector2 : Vector<T, 2, Vector2<T>>
{
public:
    /**
     * @brief Default constructor.
     *
     * Initializes all components x and y to zero.
     */
    constexpr Vector2() noexcept
        : Vector<T, 2, Vector2<T>>()
    { }

    /**
     * @brief Constructor initializing both components with a single value.
     *
     * Initializes both components x and y with the same given value.
     *
     * @param value The value to set for both x and y components.
     */
    constexpr explicit Vector2(T value) noexcept
        : Vector<T, 2, Vector2<T>>({ value, value })
    { }

    /**
     * @brief Constructor initializing both components with specific values.
     *
     * Initializes the x component with the given x value and the y component with the given y value.
     *
     * @param x The value to set for the x component.
     * @param y The value to set for the y component.
     */
    constexpr Vector2(T x, T y) noexcept
        : Vector<T, 2, Vector2<T>>({ x, y })
    { }

    /**
     * @brief Constructor initializing both components from a tuple.
     *
     * This constructor extracts the first and second elements from the given tuple
     * and uses them to initialize the x and y components of the vector, respectively.
     *
     * @param t A tuple containing two elements, where the first element is used to 
     * initialize the x component and the second element to initialize the y component.
     */
    constexpr Vector2(const std::tuple<T, T>& t)
        : Vector<T, 2, Vector2<T>>({
            std::get<0>(t),
            std::get<1>(t)
        })
    { }

    /**
     * @brief Constructor that converts a `Vector2<U>` to a `Vector2<T>`.
     *
     * This constructor creates a `Vector2<T>` by copying the components from a given 
     * `Vector2<U>`. Each component of the input vector `v` is used to initialize the
     * corresponding component of the output vector, where `T` and `U` may be different
     * types. This is useful for converting between different types of vector components 
     * (e.g., from `Vector2<float>` to `Vector2<double>`).
     *
     * @tparam U The type of the components in the input vector `v`.
     * @tparam T The type of the components in the output vector (the type of the current vector).
     *
     * @param v The input `Vector2<U>` to convert to a `Vector2<T>`.
     */
    template <typename U>
    constexpr Vector2(const Vector2<U>& v)
        : Vector2<T>(v[0], v[1])
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
    constexpr operator Vector2<U>() const {
        return Vector<U, 3, Vector2<U>>({ this->v[0], this->v[1] });
    }

    /**
     * @brief Overload of the output stream operator for Vector2.
     *
     * Allows a Vector2 to be output to an output stream (e.g., std::cout) in the format: Vec2(x, v[1]).
     *
     * @param os The output stream (e.g., std::cout).
     * @param v The vector to be output.
     * @return A reference to the modified output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
        os << "Vec2(" << v.v[0] << ", " << v.v[1] << ")";
        return os;
    }

    /**
     * @brief Accessor for the x component of the vector.
     * 
     * This method returns a reference to the x component of the vector. The x component 
     * is typically the first element in a 2D vector.
     * 
     * @return T& Reference to the x component.
     */
    constexpr T& x() { return this->v[0]; }

    /**
     * @brief Accessor for the y component of the vector.
     * 
     * This method returns a reference to the y component of the vector. The y component 
     * is typically the second element in a 2D vector.
     * 
     * @return T& Reference to the y component.
     */
    constexpr T& y() { return this->v[1]; }

    /**
     * @brief Const accessor for the x component of the vector.
     * 
     * This method returns a const reference to the x component of the vector. The x component 
     * is typically the first element in a 2D vector.
     * 
     * @return const T& Const reference to the x component.
     */
    constexpr const T& x() const { return this->v[0]; }

    /**
     * @brief Const accessor for the y component of the vector.
     * 
     * This method returns a const reference to the y component of the vector. The y component 
     * is typically the second element in a 2D vector.
     * 
     * @return const T& Const reference to the y component.
     */
    constexpr const T& y() const { return this->v[1]; }

    /**
     * @brief Mutator for the x component of the vector.
     * 
     * This method sets the x component of the vector. The x component is typically the first 
     * element in a 2D vector.
     * 
     * @param value The new value to set for the x component.
     */
    constexpr void x(T value) { this->v[0] = value; }

    /**
     * @brief Mutator for the y component of the vector.
     * 
     * This method sets the y component of the vector. The y component is typically the second 
     * element in a 2D vector.
     * 
     * @param value The new value to set for the y component.
     */
    constexpr void y(T value) { this->v[1] = value; }
};

/**
 * @brief Function to rotate the vector around the origin by an angle in radians.
 *
 * @param angle The angle in radians by which to rotate the vector.
 */
template <typename T>
inline Vector2<T> rotate(const Vector2<T>& v, T angle) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    const T c = std::cos(angle);
    const T s = std::sin(angle);
    return Vector2<T> {
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c
    };
}

/**
 * @brief Function to get a rotated copy of the vector around the origin by given cosine and sine values.
 *
 * @param c The cosine value of the rotation angle.
 * @param s The sine value of the rotation angle.
 * @return A rotated copy of the vector.
 */
template <typename T>
inline constexpr Vector2<T> rotated(const Vector2<T>& v, T c, T s) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return Vector2<T> {
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c
    };
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
template <typename T>
inline constexpr Vector2<T> reflect(const Vector2<T>& v, const Vector2<T>& normal) {
    T d = dot(v, normal);
    return Vector2(
        v[0] - 2.0 * d * normal[0],
        v[1] - 2.0 * d * normal[1]
    );
}

/**
 * @brief Function to calculate the angle in radians of the vector with respect to the positive x-axis.
 *
 * @return The angle in radians of the vector with respect to the positive x-axis.
 */
template <typename T>
inline T angle(const Vector2<T>& v) {
    return std::atan2(v[1], v[0]);
}

/**
 * @brief Function to calculate the angle in radians between two vectors with respect to the positive x-axis.
 *
 * @param other The other vector to calculate the angle with respect to.
 * @return The angle in radians between the two vectors with respect to the positive x-axis.
 */
template <typename T>
inline T angle(const Vector2<T>& v1, const Vector2<T>& v2) {
    return std::atan2(v1[1] - v2[1], v1[0] - v2[0]);
}

} // namespace bpm

#endif // BPM_VEC2_HPP
