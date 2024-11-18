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

#ifndef BPM_VECX_HPP
#define BPM_VECX_HPP

#include "./math.hpp"

#include <algorithm>
#include <cstdint>
#include <array>
#include <cmath>

namespace bpm {

/**
 * @brief N-dimensional vector base class using CRTP.
 * 
 * Provides a foundation for vector operations in N dimensions.
 * Supports only numeric types.
 *
 * @tparam T       Numeric type of the vector components.
 * @tparam N       Number of dimensions.
 * @tparam Derived Type of the derived class (CRTP).
 */
template <typename T, uint_fast8_t N, typename Derived>
class Vector
{
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");

public:
    static constexpr uint_fast8_t DIMENSIONS = N;   ///< The number of dimensions of the vector.
    using Array = std::array<T, N>;                 ///< The type of the vector container.
    using Type = Derived;
    using ValueType = T;                            ///< The type of each component of the vector.

    /**
     * @brief Default constructor.
     *
     * Initializes all components to zero.
     */
    constexpr Vector() noexcept
        : v{}
    { }

    /**
     * @brief Constructor from std::array.
     *
     * Allows direct initialization from an std::array.
     */
    constexpr Vector(const Array& a) noexcept
        : v(a)
    { }

    /**
     * @brief Conversion operator to allow implicit conversion to an `std::array&`.
     * 
     * This operator provides access to the underlying array representation of the vector,
     * enabling seamless interaction with APIs or code that require a reference to an `std::array`.
     * 
     * @return std::array& A non-const reference to the underlying array representation.
     */
    constexpr operator Array&() noexcept {
        return v;
    }

    /**
     * @brief Conversion operator to allow implicit conversion to a `const std::array&`.
     * 
     * This operator provides read-only access to the underlying array representation of the vector,
     * allowing const-qualified `Vector` objects to be used where a `const std::array&` is expected.
     * 
     * @return const std::array& A const reference to the underlying array representation.
     */
    constexpr operator const Array&() const noexcept {
        return v;
    }

    /**
     * @brief Subscript operator to access the components of the vector.
     *
     * Allows accessing the components of the vector using array-style indexing.
     *
     * @param axis The index of the component to access (0 for x, 1 for y).
     * @return A reference to the component at the specified index.
     */
    constexpr T& operator[](uint_fast8_t axis) noexcept {
        return v[axis];
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
    constexpr const T& operator[](uint_fast8_t axis) const noexcept {
        return v[axis];
    }

    /**
     * @brief Unary negation operator.
     *
     * Returns the negation of the current vector by negating each component.
     *
     * @return The negation of the current vector.
     */
    constexpr Type operator-() const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = -v[i];
        }
        return result;
    }

    /**
     * @brief Subtraction operator with a scalar.
     *
     * Subtracts the given scalar value from each component of the current vector.
     *
     * @param scalar The scalar value to subtract from each component.
     * @return A new vector resulting from the subtraction operation.
     */
    constexpr Type operator-(T scalar) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] - scalar;
        }
        return result;
    }

    /**
     * @brief Addition operator with a scalar.
     *
     * Adds the given scalar value to each component of the current vector.
     *
     * @param scalar The scalar value to add to each component.
     * @return A new vector resulting from the addition operation.
     */
    constexpr Type operator+(T scalar) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] + scalar;
        }
        return result;
    }

    /**
     * @brief Multiplication operator with a scalar.
     *
     * Multiplies each component of the current vector by the given scalar value.
     *
     * @param scalar The scalar value to multiply each component by.
     * @return A new vector resulting from the multiplication operation.
     */
    constexpr Type operator*(T scalar) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] * scalar;
        }
        return result;
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
    constexpr Type operator/(T scalar) const noexcept {
        Type result;
        if constexpr (std::is_floating_point_v<T>) {
            const T rcp = static_cast<T>(1.0) / scalar;
            for (uint_fast8_t i = 0; i < N; i++) {
                result[i] = v[i] * rcp;
            }
        } else {
            for (uint_fast8_t i = 0; i < N; i++) {
                result[i] = v[i] / scalar;
            }
        }
        return result;
    }

    /**
     * @brief Subtraction operator between vectors.
     *
     * Subtracts each component of the other vector from the corresponding component of the current vector.
     *
     * @param other The vector to subtract from the current vector.
     * @return A new vector resulting from the subtraction operation.
     */
    constexpr Type operator-(const Vector& other) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] - other.v[i];
        }
        return result;
    }

    /**
     * @brief Addition operator between vectors.
     *
     * Adds each component of the other vector to the corresponding component of the current vector.
     *
     * @param other The vector to add to the current vector.
     * @return A new vector resulting from the addition operation.
     */
    constexpr Type operator+(const Vector& other) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] + other.v[i];
        }
        return result;
    }

    /**
     * @brief Multiplication operator between vectors.
     *
     * Multiplies each component of the other vector by the corresponding component of the current vector.
     *
     * @param other The vector to multiply with the current vector.
     * @return A new vector resulting from the multiplication operation.
     */
    constexpr Type operator*(const Vector& other) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] * other.v[i];
        }
        return result;
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
    constexpr Type operator/(const Vector& other) const noexcept {
        Type result;
        for (uint_fast8_t i = 0; i < N; i++) {
            result[i] = v[i] / other.v[i];
        }
        return result;
    }

    /**
     * @brief Equality operator.
     *
     * Checks if each component of the current vector is equal to the corresponding component of the other vector.
     *
     * @param other The vector to compare with the current vector.
     * @return True if the vectors are equal (i.e., all components are equal), false otherwise.
     */
    constexpr bool operator==(const Vector& other) const noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            if (v[i] != other.v[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Inequality operator.
     *
     * Checks if any component of the current vector is not equal to the corresponding component of the other vector.
     *
     * @param other The vector to compare with the current vector.
     * @return True if the vectors are not equal (i.e., any component is not equal), false otherwise.
     */
    constexpr bool operator!=(const Vector& other) const noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            if (v[i] == other.v[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Subtraction and assignment operator with a scalar.
     *
     * Subtracts the given scalar value from each component of the current vector and assigns the result back to the current vector.
     *
     * @param scalar The scalar value to subtract from each component.
     * @return A reference to the modified current vector.
     */
    constexpr Vector& operator-=(T scalar) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] -= scalar;
        }
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
    constexpr Vector& operator+=(T scalar) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] += scalar;
        }
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
    constexpr Vector& operator*=(T scalar) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] *= scalar;
        }
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
    constexpr Vector& operator/=(T scalar) noexcept {
        if constexpr (std::is_floating_point_v<T>) {
            const T rcp = static_cast<T>(1.0) / scalar;
            for (uint_fast8_t i = 0; i < N; i++) {
                v[i] *= rcp;
            }
        } else {
            for (uint_fast8_t i = 0; i < N; i++) {
                v[i] /= scalar;
            }
        }
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
    constexpr Vector& operator-=(const Vector& other) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] -= other[i];
        }
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
    constexpr Vector& operator+=(const Vector& other) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] += other[i];
        }
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
    constexpr Vector& operator*=(const Vector& other) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] *= other[i];
        }
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
    constexpr Vector& operator/=(const Vector& other) noexcept {
        for (uint_fast8_t i = 0; i < N; i++) {
            v[i] /= other[i];
        }
        return *this;
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
    constexpr T* data() noexcept {
        return v.data();
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
    constexpr const T* data() const noexcept {
        return v.data();
    }

protected:
    std::array<T, N> v;
};


/* Vector Algorithms Implementation */

/**
 * @brief Element-wise minimum of two vectors.
 *
 * Computes a new vector where each component is the minimum of the corresponding components in `a` and `b`.
 *
 * @tparam T Numeric type of the vector components.
 * @tparam N Number of dimensions of the vectors.
 * @tparam D Derived class type (CRTP).
 * @param a First input vector.
 * @param b Second input vector.
 * @return A vector of type `Vector<T, N, D>::Type` with the element-wise minimum values.
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr typename Vector<T, N, D>::Type min(Vector<T, N, D> a, Vector<T, N, D> b) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    typename Vector<T, N, D>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = std::min(a[i], b[i]);
    }
    return result;
}

/**
 * @brief Element-wise maximum of two vectors.
 *
 * Computes a new vector where each component is the maximum of the corresponding components in `a` and `b`.
 *
 * @tparam T Numeric type of the vector components.
 * @tparam N Number of dimensions of the vectors.
 * @tparam D Derived class type (CRTP).
 * @param a First input vector.
 * @param b Second input vector.
 * @return A vector of type `Vector<T, N, D>::Type` with the element-wise maximum values.
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr typename Vector<T, N, D>::Type max(Vector<T, N, D> a, Vector<T, N, D> b) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    typename Vector<T, N, D>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = std::max(a[i], b[i]);
    }
    return result;
}

/**
 * @brief Clamps each component of a vector within the specified bounds.
 *
 * Computes a new vector where each component of `v` is clamped between the corresponding
 * components in `lo` and `hi`.
 *
 * @tparam T Numeric type of the vector components.
 * @tparam N Number of dimensions of the vectors.
 * @tparam D Derived class type (CRTP).
 * @param v Input vector to clamp.
 * @param lo Vector containing the lower bounds for each component.
 * @param hi Vector containing the upper bounds for each component.
 * @return A vector of type `Vector<T, N, D>::Type` with each component clamped to the specified range.
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr typename Vector<T, N, D>::Type clamp(Vector<T, N, D> v, Vector<T, N, D> lo, Vector<T, N, D> hi) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    typename Vector<T, N, D>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = std::clamp(v[i], lo[i], hi[i]);
    }
    return result;
}

/**
 * @brief Computes the element-wise absolute value of a vector.
 *
 * Creates a new vector where each component is the absolute value of the corresponding component in `v`.
 *
 * @tparam T Numeric type of the vector components.
 * @tparam N Number of dimensions of the vector.
 * @tparam D Derived class type (CRTP).
 * @param v Input vector from which to compute the absolute values.
 * @return A vector of type `Vector<T, N, D>::Type` with the absolute values of each component.
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr typename Vector<T, N, D>::Type abs(const Vector<T, N, D>& v) {
    typename Vector<T, N, D>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = std::abs(v[i]);
    }
    return result;
}

/**
 * @brief Computes the sign of each component in an N-dimensional vector.
 *
 * Produces a new vector where each component represents the sign of the corresponding component in the input vector `v`.
 * The sign of a component is typically -1, 0, or 1, depending on whether the component is negative, zero, or positive.
 *
 * @tparam T  Type of the components in the output vector, must be a signed integer type.
 * @tparam U  Type of the components in the input vector.
 * @tparam N  Number of dimensions of the vector.
 * @tparam DT Derived class type of the output vector (CRTP).
 * @tparam DU Derived class type of the input vector (CRTP).
 * @param v   Input vector whose component signs will be computed.
 * @return An N-dimensional vector of type `Vector<T, N, DT>::Type` where each component represents the sign of the corresponding component in `v`.
 */
template <typename T, typename U, uint_fast8_t N, typename DT, typename DU>
inline constexpr typename Vector<T, N, DT>::Type sign(const Vector<U, N, DU>& v) {
    static_assert(std::is_signed<T>::value, "Type T must be a signed integer");
    typename Vector<T, N, DT>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = (v[i] > 0) - (v[i] < 0);  // Compute sign as -1, 0, or 1
    }
    return result;
}

/**
 * @brief Computes the element-wise reciprocal of each component in a vector.
 * 
 * Generates a new vector where each component is the reciprocal of the corresponding component 
 * in the input vector `v` (i.e., `1 / v[i]`). The resulting vector has the same dimensionality as `v`.
 * 
 * @tparam T The type of each component in the output vector, typically a floating-point type.
 * @tparam U The type of each component in the input vector `v`, which must be a floating-point type.
 * @tparam N The number of components in the vector (dimensionality).
 * @tparam DT The derived type of the output vector (for CRTP).
 * @tparam DU The derived type of the input vector (for CRTP).
 * 
 * @param v The input vector for which element-wise reciprocals are calculated.
 * @return A vector of type `Vector<T, N, DT>::Type` where each component is the reciprocal of the corresponding component in `v`.
 * 
 * @note This function requires that `U` is a floating-point type, and it is recommended that `T` is also floating-point to preserve precision.
 */
template <typename T, typename U, uint_fast8_t N, typename DT, typename DU>
constexpr inline typename Vector<T, N, DT>::Type reciprocal(Vector<U, N, DU> v) {
    static_assert(std::is_floating_point<U>::value, "Type U must be a floating-point");
    typename Vector<T, N, DT>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = static_cast<T>(1.0) / static_cast<T>(v[i]);
    }
    return result;
}

/**
 * @brief Checks if two vectors are approximately equal within a specified tolerance.
 * 
 * Compares each component of vectors `a` and `b` to determine if they are approximately 
 * equal within the tolerance `epsilon`. The vectors are considered approximately equal 
 * if the absolute difference between each corresponding component is within `epsilon`.
 * 
 * @tparam T The type of the vector components, which must be a floating-point type.
 * @tparam N The dimensionality of the vectors (e.g., 3 for 3D vectors).
 * @tparam D The derived class type for CRTP.
 * 
 * @param a The first vector to compare.
 * @param b The second vector to compare.
 * @param epsilon The tolerance for comparing each component, with a default of `std::numeric_limits<T>::epsilon()`.
 * 
 * @return `true` if all components of `a` and `b` are approximately equal within the specified tolerance; `false` otherwise.
 * 
 * @note This function requires `T` to be a floating-point type.
 */
template <typename T, uint_fast8_t N, typename D>
inline bool approx(const Vector<T, N, D>& a, const Vector<T, N, D>& b, T epsilon = std::numeric_limits<T>::epsilon()) {
    static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point");
    for (uint_fast8_t i = 0; i < N; i++) {
        if (!approx(a[i], b[i], epsilon)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Moves a vector 'a' towards another vector 'b' by a specified distance 'delta'.
 * 
 * This function computes a new vector that is moved towards the target vector `b` 
 * by a distance of `delta`. The movement is done component-wise, where each component 
 * of the vector `a` is moved towards the corresponding component in vector `b` by the specified distance.
 * 
 * @tparam T The numeric type of the vector components, typically a floating-point type.
 * @tparam N The dimensionality of the vectors (e.g., 3 for 3D vectors).
 * @tparam D The derived class type for CRTP.
 * 
 * @param a The starting vector, from which we move.
 * @param b The target vector, towards which we move.
 * @param delta The distance to move each component of `a` towards the corresponding component in `b`.
 * 
 * @return A new vector that is moved towards `b` by `delta`, with the same dimensionality as `a` and `b`.
 * 
 * @note The resulting vector is a component-wise move towards `b`, but does not exceed the distance `delta` for any component.
 */
template <typename T, uint_fast8_t N, typename D>
inline typename Vector<T, N, D>::Type move_towards(const Vector<T, N, D>& a, const Vector<T, N, D>& b, T delta) {
    typename Vector<T, N, D>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = move_towards(a[i], b[i], delta);
    }
    return result;
}

/**
 * @brief Performs linear interpolation between two vectors.
 * 
 * This function computes the linear interpolation (lerp) between two vectors `a` and `b` using the formula:
 * 
 *     result = a + t * (b - a)
 * 
 * where `t` is the interpolation parameter. The value of `t` should be in the range [0, 1], where:
 * - `t = 0` returns the vector `a`,
 * - `t = 1` returns the vector `b`, and
 * - `t` values between 0 and 1 return intermediate points between `a` and `b`.
 * 
 * The interpolation is performed component-wise, so each component of the resulting vector is computed
 * as the interpolation of the corresponding components in `a` and `b`.
 * 
 * @tparam T The type of the vector components (should be a floating-point type).
 * @tparam N The dimensionality of the vectors (e.g., 2 for 2D vectors, 3 for 3D vectors).
 * @tparam D The derived class type (used for CRTP).
 *
 * @param a The starting vector for interpolation.
 * @param b The ending vector for interpolation.
 * @param t The interpolation parameter, which should be in the range [0, 1].
 * 
 * @return A new vector that represents the interpolated result between `a` and `b` based on `t`.
 * 
 * @note This function requires that `T` is a floating-point type.
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr typename Vector<T, N, D>::Type lerp(const Vector<T, N, D>& a, const Vector<T, N, D>& b, T t) {
    static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point");
    typename Vector<T, N, D>::Type result;
    for (uint_fast8_t i = 0; i < N; i++) {
        result[i] = lerp(a[i], b[i], t);
    }
    return result;
}

/**
 * @brief Computes the dot product of two vectors.
 * 
 * This function calculates the dot product (also known as the scalar product) of two vectors `v1` and `v2`.
 * The dot product is computed as the sum of the products of the corresponding components of the two vectors:
 * 
 *     dot(v1, v2) = v1[0] * v2[0] + v1[1] * v2[1] + ... + v1[N-1] * v2[N-1]
 * 
 * The dot product is used in various applications, including vector projection, calculating angles between vectors, 
 * and more.
 * 
 * @tparam T The type of the vector components, which should be a numeric type (typically floating-point).
 * @tparam N The dimensionality of the vectors (e.g., 2 for 2D vectors, 3 for 3D vectors).
 * @tparam D The derived class type (used for CRTP).
 * 
 * @param v1 The first vector.
 * @param v2 The second vector.
 *
 * @return The dot product of vectors `v1` and `v2`, which is a scalar of type `T`.
 * 
 * @note This function requires that `T` is a numeric type (either integer or floating-point).
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr T dot(const Vector<T, N, D>& v1, const Vector<T, N, D>& v2) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    T sum = 0;
    for (uint_fast8_t i = 0; i < N; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

/**
 * @brief Calculates the squared Euclidean length (squared magnitude) of a vector.
 * 
 * This function computes the squared Euclidean length (magnitude squared) of a vector by summing the squares 
 * of its components. It avoids the computational cost of taking a square root, which is useful when only 
 * relative lengths or comparisons are needed. This is particularly efficient in algorithms that don't require 
 * the exact length but only need to compare distances.
 * 
 * @tparam T The type of the vector components, which should be a numeric type (typically floating-point).
 * @tparam N The dimensionality of the vector (e.g., 3 for a 3D vector, 4 for a 4D vector).
 * @tparam D The derived class type (used for CRTP).
 *
 * @param v The input vector for which the squared length is calculated.
 *
 * @return The squared Euclidean length of the vector as a scalar of type `T`.
 * 
 * @note This function requires that `T` is a numeric type (either integer or floating-point).
 */
template <typename T, uint_fast8_t N, typename D>
inline constexpr T length_sq(const Vector<T, N, D>& v) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    T sum = 0;
    for (uint_fast8_t i = 0; i < N; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

/**
 * @brief Calculates the Euclidean length (magnitude) of a vector.
 * 
 * This function computes the Euclidean length of a vector by summing the squares 
 * of each component and taking the square root of the total. This is equivalent 
 * to calculating the L2 norm of the vector.
 * 
 * @tparam T The type of each component in the vector, typically a floating-point type.
 * @tparam N The dimension of the vector (e.g., 3 for a 3D vector, 4 for a 4D vector).
 * @tparam D The derived class type (used for CRTP).
 * 
 * @param v The input vector whose length is to be calculated.
 * 
 * @return The Euclidean length (magnitude) of the vector as a scalar of type `T`.
 * 
 * @note This function requires that `T` is a floating-point type.
 */
template <typename T, uint_fast8_t N, typename D>
inline T length(const Vector<T, N, D>& v) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    T sum = 0;
    for (uint_fast8_t i = 0; i < N; i++) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

/**
 * @brief Normalizes a vector.
 * 
 * This function normalizes the given vector `v`, scaling it so that its length (magnitude) becomes 1.
 * The resulting vector is a unit vector in the same direction as the original vector.
 * 
 * @tparam T The type of the vector components, which should be a floating-point type.
 * @tparam N The dimensionality of the vector (e.g., 3 for a 3D vector, 4 for a 4D vector).
 * @tparam D The derived class type (used for CRTP).
 * 
 * @param v The vector to normalize.
 * 
 * @return The normalized vector, a vector of the same type and dimensionality as `v`, but with a length of 1.
 * 
 * @note This function requires that `T` is a floating-point type.
 */
template <typename T, uint_fast8_t N, typename D>
inline typename Vector<T, N, D>::Type normalize(const Vector<T, N, D>& v) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return v * static_cast<T>(1.0) / length(v);
}

/**
 * @brief Computes the direction vector from one vector to another.
 * 
 * This function computes the direction vector that points from the vector `v1` 
 * to the vector `v2`. The direction vector is normalized, meaning it has a 
 * length of 1 and indicates the direction from `v1` to `v2`.
 * 
 * @tparam T The type of the vector components, typically a floating-point type.
 * @tparam N The dimensionality of the vectors (e.g., 3 for a 3D vector).
 * @tparam D The derived class type (used for CRTP).
 * 
 * @param v1 The starting vector (the origin of the direction).
 * @param v2 The target vector (the destination of the direction).
 * 
 * @return A normalized direction vector from `v1` to `v2`, with a length of 1.
 * 
 * @note This function assumes that `v1` and `v2` are of the same type and dimension.
 */
template <typename T, uint_fast8_t N, typename D>
inline typename Vector<T, N, D>::Type direction(const Vector<T, N, D>& v1, const Vector<T, N, D>& v2) {
    return normalize(v2 - v1);
}

/**
 * @brief Computes the squared Euclidean distance between two vectors.
 * 
 * This function calculates the squared Euclidean distance between two vectors `v1` and `v2`.
 * It returns the square of the distance, which avoids the computational cost of taking the square root.
 * This is particularly useful when only relative distances are needed, as it is faster 
 * and avoids potential issues with floating-point precision.
 * 
 * @tparam T The type of the vector components, typically a floating-point type.
 * @tparam N The dimensionality of the vectors (e.g., 3 for 3D vectors).
 * @tparam D The derived class type (used for CRTP).
 * 
 * @param v1 The first vector in the distance calculation.
 * @param v2 The second vector in the distance calculation.
 * 
 * @return The squared Euclidean distance between `v1` and `v2` as a scalar of type `T`.
 * 
 * @note This function returns the square of the distance between `v1` and `v2`. 
 *       To obtain the actual Euclidean distance, you can call the `distance` function.
 */
template <typename T, uint_fast8_t N, typename D>
inline T distance_sq(const Vector<T, N, D>& v1, const Vector<T, N, D>& v2) {
    static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point");
    return length_sq(v1 - v2);
}

/**
 * @brief Computes the Euclidean distance between two vectors.
 * 
 * This function calculates the Euclidean distance between two vectors `v1` and `v2`.
 * The distance is defined as the length (magnitude) of the vector difference `v1 - v2`, 
 * and represents the shortest path between the two points in N-dimensional space. 
 * It is computed using the formula:
 * 
 *     distance(v1, v2) = || v1 - v2 ||
 * 
 * where `||v||` denotes the Euclidean norm (length) of the vector `v`.
 * 
 * @tparam T The type of the vector components, which must be a floating-point type (e.g., `float`, `double`).
 * @tparam N The dimensionality of the vectors (e.g., 3 for 3D vectors).
 * @tparam D The derived class type (used for CRTP).
 * 
 * @param v1 The first vector in the distance calculation.
 * @param v2 The second vector in the distance calculation.
 * 
 * @return The Euclidean distance between `v1` and `v2`, a scalar of type `T`.
 * 
 * @note This function requires that `T` is a floating-point type.
 */
template <typename T, uint_fast8_t N, typename D>
inline T distance(const Vector<T, N, D>& v1, const Vector<T, N, D>& v2) {
    static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point");
    return length(v1 - v2);
}

} // namespace bpm

#endif // BPM_VECX_HPP
