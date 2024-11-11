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

#include "./vecx.hpp"
#include "./vec3.hpp"

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
class Vector4 : public Vector<T, 4, Vector4<T>>
{
public:
    /**
     * @brief Default constructor. Constructs a vector with all components set to zero.
     */
    constexpr Vector4()
        : Vector<T, 4, Vector4<T>>()
    { }

    /**
     * @brief Constructs a vector with all components set to the specified value.
     *
     * @param value The value to set for all components.
     */
    constexpr explicit Vector4(T value)
        : Vector<T, 4, Vector4<T>>({
            value, value, value, value
        })
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
        : Vector<T, 4, Vector4<T>>({ x, y, z, w })
    { }

    /**
     * @brief Constructs a Vector4 from a Vector3 with an optional w-component.
     *
     * @param Vec3 The Vector3 to use for the x, y, and z components.
     * @param w The w-component.
     */
    constexpr Vector4(const Vector3<T>& v, float w = 1.0f)
        : Vector<T, 4, Vector4<T>>({ v[0], v[1], v[2], w })
    { }

    /**
     * @brief Constructor initializing all four components from a tuple.
     *
     * This constructor extracts the four elements from the given tuple
     * and uses them to initialize the x, y, z, and w components of the vector.
     *
     * @param t A tuple containing four elements, where each element is used to 
     * initialize the respective component (x, y, z, w) of the vector.
     */
    constexpr Vector4(const std::tuple<T, T, T, T>& t)
        : Vector<T, 4, Vector4<T>>({
            std::get<0>(t),
            std::get<1>(t),
            std::get<2>(t),
            std::get<3>(t)
        })
    { }

    /**
     * @brief Constructor that converts a `Vector4<U>` to a `Vector4<T>`.
     *
     * This constructor creates a `Vector4<T>` by copying the components from a given 
     * `Vector4<U>`. Each component of the input vector `v` is used to initialize the
     * corresponding component of the output vector, where `T` and `U` may be different
     * types. This is useful for converting between different types of vector components 
     * (e.g., from `Vector4<float>` to `Vector4<double>`).
     *
     * @tparam U The type of the components in the input vector `v`.
     * @tparam T The type of the components in the output vector (the type of the current vector).
     *
     * @param v The input `Vector4<U>` to convert to a `Vector4<T>`.
     */
    template <typename U>
    constexpr Vector4(const Vector4<U>& v)
        : Vector4<T>(v[0], v[1], v[2], v[3])
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
        return Vector4<U>({ this->v[0], this->v[1], this->v[2], this->v[3] });
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
        os << "Vec4(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ")";
        return os;
    }

    /**
     * @brief Accessor for the x component of the vector.
     * 
     * This method returns a reference to the x component of the vector. The x component 
     * is typically the first element in a 4D vector.
     * 
     * @return T& Reference to the x component.
     */
    constexpr T& x() { return this->v[0]; }

    /**
     * @brief Accessor for the y component of the vector.
     * 
     * This method returns a reference to the y component of the vector. The y component 
     * is typically the second element in a 4D vector.
     * 
     * @return T& Reference to the y component.
     */
    constexpr T& y() { return this->v[1]; }

    /**
     * @brief Accessor for the z component of the vector.
     * 
     * This method returns a reference to the z component of the vector. The z component 
     * is typically the third element in a 4D vector.
     * 
     * @return T& Reference to the z component.
     */
    constexpr T& z() { return this->v[2]; }

    /**
     * @brief Accessor for the w component of the vector.
     * 
     * This method returns a reference to the w component of the vector. The w component 
     * is typically the fourth element in a 4D vector.
     * 
     * @return T& Reference to the w component.
     */
    constexpr T& w() { return this->v[3]; }

    /**
     * @brief Const accessor for the x component of the vector.
     * 
     * This method returns a const reference to the x component of the vector. The x component 
     * is typically the first element in a 4D vector.
     * 
     * @return const T& Const reference to the x component.
     */
    constexpr const T& x() const { return this->v[0]; }

    /**
     * @brief Const accessor for the y component of the vector.
     * 
     * This method returns a const reference to the y component of the vector. The y component 
     * is typically the second element in a 4D vector.
     * 
     * @return const T& Const reference to the y component.
     */
    constexpr const T& y() const { return this->v[1]; }

    /**
     * @brief Const accessor for the z component of the vector.
     * 
     * This method returns a const reference to the z component of the vector. The z component 
     * is typically the third element in a 4D vector.
     * 
     * @return const T& Const reference to the z component.
     */
    constexpr const T& z() const { return this->v[2]; }

    /**
     * @brief Const accessor for the w component of the vector.
     * 
     * This method returns a const reference to the w component of the vector. The w component 
     * is typically the fourth element in a 4D vector.
     * 
     * @return const T& Const reference to the w component.
     */
    constexpr const T& w() const { return this->v[3]; }

    /**
     * @brief Mutator for the x component of the vector.
     * 
     * This method sets the x component of the vector. The x component is typically the first 
     * element in a 4D vector.
     * 
     * @param value The new value to set for the x component.
     */
    constexpr void x(T value) { this->v[0] = value; }

    /**
     * @brief Mutator for the y component of the vector.
     * 
     * This method sets the y component of the vector. The y component is typically the second 
     * element in a 4D vector.
     * 
     * @param value The new value to set for the y component.
     */
    constexpr void y(T value) { this->v[1] = value; }

    /**
     * @brief Mutator for the z component of the vector.
     * 
     * This method sets the z component of the vector. The z component is typically the third 
     * element in a 4D vector.
     * 
     * @param value The new value to set for the z component.
     */
    constexpr void z(T value) { this->v[2] = value; }

    /**
     * @brief Mutator for the w component of the vector.
     * 
     * This method sets the w component of the vector. The w component is typically the fourth 
     * element in a 4D vector.
     * 
     * @param value The new value to set for the w component.
     */
    constexpr void w(T value) { this->v[3] = value; }
};

} // namespace bpm

#endif // BPM_VEC4_HPP
