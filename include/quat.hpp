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

#ifndef BPM_QUAT_HPP
#define BPM_QUAT_HPP

#include "./vecx.hpp"
#include "./vec3.hpp"

#include <iostream>
#include <cmath>

namespace bpm {

/**
 * @brief Quaternion structure for representing rotations in 3D space
 * 
 * A quaternion is a four-dimensional complex number that can be used to represent
 * 3D rotations. This implementation provides methods for common quaternion operations
 * including composition, normalization, and conversion to/from other rotation formats.
 */
class Quat : public Vector<float, 4, Quat>
{
public:
    /**
     * @brief Default constructor.
     *
     * Initializes all components w, x, y, and z to zero.
     */
    constexpr Quat() noexcept
        : Vector<float, 4, Quat>()
    { }

    /**
     * @brief Constructor with explicit components
     * 
     * @param w Scalar (real) component
     * @param x First vector (imaginary) component
     * @param y Second vector (imaginary) component
     * @param z Third vector (imaginary) component
     */
    constexpr Quat(float w, float x, float y, float z) noexcept
        : Vector<float, 4, Quat>({ w, x, y, z })
    { }

    /**
     * @brief Constructor from axis and angle
     * 
     * Creates a quaternion representing a rotation around a specified axis
     * 
     * @tparam T Type of the vector components
     * @param axis Unit vector representing the rotation axis
     * @param angle Rotation angle in radians
     */
    template<typename T>
    constexpr Quat(const Vector3<T>& axis, float angle) {
        float half_angle = angle * 0.5f;
        float s = std::sin(half_angle);
        v[0] = std::cos(half_angle);
        v[1] = axis[0] * s;
        v[2] = axis[1] * s;
        v[3] = axis[2] * s;
    }

    /**
     * @brief Constructor from Euler angles
     * 
     * Creates a quaternion from pitch, yaw, and roll angles
     * 
     * @param pitch Rotation around X-axis in radians
     * @param yaw Rotation around Y-axis in radians
     * @param roll Rotation around Z-axis in radians
     */
    Quat(float pitch, float yaw, float roll) {
        float cy = std::cos(yaw * 0.5f);
        float sy = std::sin(yaw * 0.5f);
        float cp = std::cos(pitch * 0.5f);
        float sp = std::sin(pitch * 0.5f);
        float cr = std::cos(roll * 0.5f);
        float sr = std::sin(roll * 0.5f);

        v[0] = cr * cp * cy + sr * sp * sy;
        v[1] = sr * cp * cy - cr * sp * sy;
        v[2] = cr * sp * cy + sr * cp * sy;
        v[3] = cr * cp * sy - sr * sp * cy;
    }

    /**
     * @brief Quaternion multiplication operator
     * 
     * Performs quaternion multiplication which represents composition of rotations
     * 
     * @param q The quaternion to multiply with
     * @return Quat The resulting quaternion
     */
    constexpr Quat operator*(const Quat& q) const noexcept {
        return Quat(
            v[0] * q[0] - v[1] * q[1] - v[2] * q[2] - v[3] * q[2],
            v[0] * q[1] + v[1] * q[0] + v[2] * q[3] - v[3] * q[1],
            v[0] * q[2] - v[1] * q[3] + v[2] * q[0] + v[3] * q[0],
            v[0] * q[3] + v[1] * q[2] - v[2] * q[1] + v[3] * q[3]
        );
    }

    /**
     * @brief Stream output operator
     * 
     * Enables printing quaternion to output streams
     * 
     * @param os Output stream
     * @param q Quaternion to output
     * @return std::ostream& Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Quat& q) {
        os << "Quat(" << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << ")";
        return os;
    }

    /**
     * @brief Accessor for the w component (real part) of the quaternion.
     * 
     * This method returns a reference to the w component of the quaternion. The w component 
     * is typically used as the scalar (real) part of the quaternion.
     * 
     * @return float& Reference to the w component.
     */
    constexpr float& w() { return v[0]; }

    /**
     * @brief Accessor for the x component (imaginary part) of the quaternion.
     * 
     * This method returns a reference to the x component of the quaternion. The x component 
     * is typically used as part of the imaginary components of the quaternion.
     * 
     * @return float& Reference to the x component.
     */
    constexpr float& x() { return v[1]; }

    /**
     * @brief Accessor for the y component (imaginary part) of the quaternion.
     * 
     * This method returns a reference to the y component of the quaternion. The y component 
     * is typically used as part of the imaginary components of the quaternion.
     * 
     * @return float& Reference to the y component.
     */
    constexpr float& y() { return v[2]; }

    /**
     * @brief Accessor for the z component (imaginary part) of the quaternion.
     * 
     * This method returns a reference to the z component of the quaternion. The z component 
     * is typically used as part of the imaginary components of the quaternion.
     * 
     * @return float& Reference to the z component.
     */
    constexpr float& z() { return v[3]; }

    /**
     * @brief Const accessor for the w component (real part) of the quaternion.
     * 
     * This method returns a const reference to the w component of the quaternion. The w component 
     * is typically used as the scalar (real) part of the quaternion.
     * 
     * @return const float& Const reference to the w component.
     */
    constexpr const float& w() const { return v[0]; }

    /**
     * @brief Const accessor for the x component (imaginary part) of the quaternion.
     * 
     * This method returns a const reference to the x component of the quaternion. The x component 
     * is typically used as part of the imaginary components of the quaternion.
     * 
     * @return const float& Const reference to the x component.
     */
    constexpr const float& x() const { return v[1]; }

    /**
     * @brief Const accessor for the y component (imaginary part) of the quaternion.
     * 
     * This method returns a const reference to the y component of the quaternion. The y component 
     * is typically used as part of the imaginary components of the quaternion.
     * 
     * @return const float& Const reference to the y component.
     */
    constexpr const float& y() const { return v[2]; }

    /**
     * @brief Const accessor for the z component (imaginary part) of the quaternion.
     * 
     * This method returns a const reference to the z component of the quaternion. The z component 
     * is typically used as part of the imaginary components of the quaternion.
     * 
     * @return const float& Const reference to the z component.
     */
    constexpr const float& z() const { return v[3]; }

    /**
     * @brief Mutator for the w component (real part) of the quaternion.
     * 
     * This method sets the w component of the quaternion. The w component is typically used 
     * as the scalar (real) part of the quaternion.
     * 
     * @param value The new value to set for the w component.
     */
    constexpr void w(float value) { v[0] = value; }

    /**
     * @brief Mutator for the x component (imaginary part) of the quaternion.
     * 
     * This method sets the x component of the quaternion. The x component is typically used 
     * as part of the imaginary components of the quaternion.
     * 
     * @param value The new value to set for the x component.
     */
    constexpr void x(float value) { v[1] = value; }

    /**
     * @brief Mutator for the y component (imaginary part) of the quaternion.
     * 
     * This method sets the y component of the quaternion. The y component is typically used 
     * as part of the imaginary components of the quaternion.
     * 
     * @param value The new value to set for the y component.
     */
    constexpr void y(float value) { v[2] = value; }

    /**
     * @brief Mutator for the z component (imaginary part) of the quaternion.
     * 
     * This method sets the z component of the quaternion. The z component is typically used 
     * as part of the imaginary components of the quaternion.
     * 
     * @param value The new value to set for the z component.
     */
    constexpr void z(float value) { v[3] = value; }
};


/* Quaternion Algorithms Implementation */

/**
 * @brief Performs a normalized linear interpolation between two quaternions.
 * 
 * This function performs a normalized linear interpolation (NLerp) between two quaternions.
 * The interpolation parameter `t` is clamped between 0 and 1.
 * 
 * @tparam T The type of the components in the vectors (should be a floating-point type).
 * @param a The start vector.
 * @param b The end vector.
 * @param t The interpolation parameter. It should be in the range [0, 1].
 * @return The result of the NLerp operation, normalized.
 */
inline Quat nlerp(const Quat& a, const Quat& b, float t) {
    return normalize(lerp(a, b, t));
}

/**
 * @brief Performs a spherical linear interpolation between two quaternions.
 * 
 * This function performs a spherical linear interpolation (SLerp) between two quaternions.
 * The interpolation is performed using the provided interpolation amount `amount`.
 * 
 * @tparam T The type of the components in the quaternions (should be a floating-point type).
 * @param q1 The first quaternion.
 * @param q2 The second quaternion.
 * @param amount The interpolation amount. It should be in the range [0, 1].
 * @return The result of the SLerp operation.
 */
template <typename T>
inline Quat slerp(const Quat& q1, Quat q2, T amount) {
    float cos_half_theta = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
    if (cos_half_theta < 0) {
        cos_half_theta = -cos_half_theta;
        q2[0] = -q2[0];
        q2[1] = -q2[1];
        q2[2] = -q2[2];
        q2[3] = -q2[3];
    }
    if (std::fabs(cos_half_theta) >= 1.0f) {
        return q1;
    } else if (cos_half_theta > 0.95f) {
        return nlerp(q1, q2, amount);
    }
    float half_theta = std::acos(cos_half_theta);
    float sin_half_theta = std::sqrt(1.0f - cos_half_theta*cos_half_theta);
    if (std::fabs(sin_half_theta) < std::numeric_limits<float>::epsilon()) {
        return {
            q1[0] * 0.5f + q2[0] * 0.5f,
            q1[1] * 0.5f + q2[1] * 0.5f,
            q1[2] * 0.5f + q2[2] * 0.5f,
            q1[3] * 0.5f + q2[3] * 0.5f
        };
    }
    float ratioA = std::sin((1.0f - amount) * half_theta) / sin_half_theta;
    float ratioB = std::sin(amount * half_theta) / sin_half_theta;
    return {
        q1[0] * ratioA + q2[0] * ratioB,
        q1[1] * ratioA + q2[1] * ratioB,
        q1[2] * ratioA + q2[2] * ratioB,
        q1[3] * ratioA + q2[3] * ratioB
    };
}

/**
 * @brief Computes the conjugate of the quaternion.
 * 
 * The conjugate of a quaternion is computed by negating its vector part (the imaginary components). 
 * The conjugate is particularly useful for calculating the inverse of a quaternion, which is used in
 * rotation inversions and certain transformations in 3D graphics and physics simulations. 
 * 
 * Given a quaternion `q = w + xi + yj + zk`, its conjugate is defined as:
 * 
 *     conjugate(q) = w - xi - yj - zk
 * 
 * The conjugate is used when applying inverse rotations, as it represents the reverse of the quaternion's rotation.
 * 
 * @param q The input quaternion to compute the conjugate of.
 * 
 * @return Quat The conjugate quaternion, where the vector part is negated and the scalar part remains the same.
 */
inline constexpr Quat conjugate(const Quat& q) {
    return Quat(q[0], -q[1], -q[2], -q[3]);
}

/**
 * @brief Computes the inverse of the quaternion.
 * 
 * The inverse of a quaternion is used to "undo" the rotation represented by the quaternion.
 * It is computed by dividing the conjugate of the quaternion by its squared norm. 
 * The inverse quaternion, when applied to the original quaternion, yields the identity rotation.
 * 
 * Given a quaternion `q = w + xi + yj + zk`, its inverse is given by:
 * 
 *     inverse(q) = conjugate(q) / ||q||^2
 * 
 * where `conjugate(q)` is the conjugate of `q` and `||q||^2` is the squared norm of `q`. 
 * The squared norm of the quaternion is calculated as `w^2 + x^2 + y^2 + z^2`.
 * 
 * The inverse is used when performing inverse rotations, or when applying transformations that 
 * need to be undone or reversed.
 * 
 * @param q The input quaternion to compute the inverse of.
 * 
 * @return Quat The inverse quaternion, which can be used to reverse the rotation represented by `q`.
 * 
 * @note If the quaternion's norm squared is 0 (i.e., it's a zero quaternion), the function returns a default quaternion, 
 * as the inverse of a zero quaternion is undefined.
 */
inline constexpr Quat inverse(const Quat& q) {
    float norm_sq = length_sq(q);
    if (norm_sq > 0.0f) {
        float invNorm = 1.0f / norm_sq;
        return Quat(q[0] * invNorm, -q[1] * invNorm, -q[2] * invNorm, -q[3] * invNorm);
    }
    return Quat();
}

/**
 * @brief Converts quaternion to axis-angle representation.
 * 
 * This function converts a quaternion `q` into an axis-angle representation. The axis is a unit vector that represents 
 * the axis of rotation, and the angle is the amount of rotation around that axis in radians.
 * 
 * The quaternion `q` is assumed to be normalized (its length should be 1), but if it is not, the function will 
 * normalize it internally to extract the axis and angle. The result is stored in the provided `axis` and `angle` pointers.
 * 
 * The axis-angle representation is often used in computer graphics and 3D transformations because it is an intuitive 
 * way to represent rotations.
 * 
 * @param q The quaternion to convert to axis-angle representation.
 * @param axis Pointer to a `Vec3` that will store the rotation axis as a unit vector.
 * @param angle Pointer to a `float` that will store the rotation angle in radians.
 * 
 * @note If the quaternion represents the identity rotation (i.e., no rotation), the axis is set arbitrarily (usually along the x-axis) 
 * and the angle is set to 0 radians.
 */
inline void to_axis_angle(const Quat& q, Vec3* axis, float* angle) {
    float scale = length(q);
    if (scale > 0.0f) {
        const float inv = 1.0f / scale;
        *angle = 2.0f * std::acos(q[0]);
        (*axis)[0] = q[1] * inv;
        (*axis)[1] = q[2] * inv;
        (*axis)[2] = q[3] * inv;
    } else {
        // If quaternion is close to identity, axis is arbitrary
        (*axis)[0] = 1.0f;
        (*axis)[1] = 0.0f;
        (*axis)[2] = 0.0f;
        *angle = 0.0f;
    }
}

/**
 * @brief Converts quaternion to Euler angles.
 * 
 * This function converts a quaternion `q` into Euler angles: pitch, yaw, and roll. These angles 
 * represent rotations around the X, Y, and Z axes, respectively, in a 3D space. The angles are returned 
 * as a `Vec3` where:
 * - `pitch` is the rotation around the X-axis (in radians),
 * - `yaw` is the rotation around the Y-axis (in radians),
 * - `roll` is the rotation around the Z-axis (in radians).
 * 
 * Euler angles are commonly used in 3D graphics, robotics, and physics simulations to represent orientation 
 * and rotation in space. However, they can suffer from gimbal lock when the pitch angle approaches ±90 degrees, 
 * causing a loss of one degree of freedom in rotation representation.
 * 
 * @return Vec3 A vector containing the Euler angles (pitch, yaw, roll) in radians.
 * 
 * @note This function may experience gimbal lock when the pitch angle approaches ±90 degrees. 
 * In such cases, the yaw and roll values may become undefined or result in large jumps.
 */
inline Vec3 to_euler(const Quat& q) {
    float sinr_cosp = 2.0f * (q[0] * q[1] + q[2] * q[3]);
    float cosr_cosp = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    float roll = std::atan2(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (q[0] * q[2] - q[3] * q[1]);
    float pitch;
    if (std::abs(sinp) >= 1.0f) {
        pitch = std::copysign(1.57079632679f, sinp); // Gimbal lock
    } else {
        pitch = std::asin(sinp);
    }

    float siny_cosp = 2.0f * (q[0] * q[3] + q[1] * q[2]);
    float cosy_cosp = 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]);
    float yaw = std::atan2(siny_cosp, cosy_cosp);

    return Vec3(pitch, yaw, roll);
}

/**
 * @brief Rotates a 3D vector by a quaternion.
 * 
 * This function rotates a 3D vector `v` by the quaternion `q`. The rotation is performed using the 
 * following formula:
 * 
 * \[
 * v' = v + 2 * q_0 * (q \times v) + 2 * (q \times (q \times v))
 * \]
 * 
 * Where `q` is a quaternion representing the rotation, and `v` is the vector being rotated. 
 * The result is the vector `v'` after applying the rotation defined by the quaternion `q`.
 * 
 * The formula uses the quaternion as a representation of a rotation in 3D space, and the vector is 
 * rotated through a combination of cross products and scalar multiplication.
 * 
 * @param v The vector to rotate.
 * @param q The quaternion that defines the rotation.
 * @return Vector3<T> The rotated vector.
 * 
 * @note This function assumes that the quaternion `q` is normalized. If the quaternion is not normalized, 
 * the result may not represent a pure rotation.
 * 
 * @warning The type `T` must be a floating-point type (`float`, `double`, etc.).
 */
template <typename T>
inline Vector3<T> rotate(const Vector3<T>& v, const Quat& q) {
    static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point");
    Vector3<T> qvec(q[1], q[2], q[3]);
    Vector3<T> temp = cross(qvec, v);
    return (v + temp * static_cast<T>(2) * q[0]) + cross(qvec, temp) * static_cast<T>(2);
}

} // namespace bpm

#endif // BPM_QUAT_HPP
