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
struct Quat
{
    float w, x, y, z;

    /**
     * @brief Default constructor
     * 
     * Initializes to identity quaternion (no rotation)
     */
    constexpr Quat()
        : w(1), x(0)
        , y(0), z(0)
    { }

    /**
     * @brief Constructor with explicit components
     * 
     * @param w Scalar (real) component
     * @param x First vector (imaginary) component
     * @param y Second vector (imaginary) component
     * @param z Third vector (imaginary) component
     */
    constexpr Quat(float w, float x, float y, float z)
        : w(w), x(x)
        , y(y), z(z)
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
    constexpr Quat(const Vector3<T>& axis, float angle)
    {
        float halfAngle = angle * 0.5f;
        float s = std::sin(halfAngle);
        w = std::cos(halfAngle);
        x = axis.x * s;
        y = axis.y * s;
        z = axis.z * s;
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
    Quat(float pitch, float yaw, float roll)
    {
        float cy = std::cos(yaw * 0.5f);
        float sy = std::sin(yaw * 0.5f);
        float cp = std::cos(pitch * 0.5f);
        float sp = std::sin(pitch * 0.5f);
        float cr = std::cos(roll * 0.5f);
        float sr = std::sin(roll * 0.5f);

        w = cr * cp * cy + sr * sp * sy;
        x = sr * cp * cy - cr * sp * sy;
        y = cr * sp * cy + sr * cp * sy;
        z = cr * cp * sy - sr * sp * cy;
    }

    /**
     * @brief Quaternion multiplication operator
     * 
     * Performs quaternion multiplication which represents composition of rotations
     * 
     * @param q The quaternion to multiply with
     * @return Quat The resulting quaternion
     */
    constexpr Quat operator*(const Quat& q) const
    {
        return Quat(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
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
    friend std::ostream& operator<<(std::ostream& os, const Quat& q)
    {
        os << "Quat(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
        return os;
    }

    /**
     * @brief Normalizes the quaternion
     * 
     * Scales the quaternion so that its magnitude equals 1
     * 
     * @return Quat& Reference to this quaternion after normalization
     */
    Quat& normalize()
    {
        float magnitude = std::sqrt(w * w + x * x + y * y + z * z);
        if (magnitude > 0.0f) {
            float invMag = 1.0f / magnitude;
            w *= invMag;
            x *= invMag;
            y *= invMag;
            z *= invMag;
        }
        return *this;
    }

    /**
     * @brief Computes the conjugate of the quaternion
     * 
     * The conjugate is used for inverse rotation
     * 
     * @return Quat The conjugate quaternion
     */
    constexpr Quat conjugate() const
    {
        return Quat(w, -x, -y, -z);
    }

    /**
     * @brief Computes the inverse of the quaternion
     * 
     * The inverse quaternion will undo the rotation of this quaternion
     * 
     * @return Quat The inverse quaternion
     */
    constexpr Quat inverse() const
    {
        float normSq = w * w + x * x + y * y + z * z;
        if (normSq > 0.0f) {
            float invNorm = 1.0f / normSq;
            return Quat(w * invNorm, -x * invNorm, -y * invNorm, -z * invNorm);
        }
        return Quat(); // Returns identity quaternion if norm is zero
    }

    /**
     * @brief Converts quaternion to axis-angle representation
     * 
     * Extracts the rotation axis and angle from the quaternion
     * 
     * @param axis Pointer to vector that will store the rotation axis
     * @param angle Pointer to float that will store the rotation angle in radians
     */
    void to_axis_angle(Vec3* axis, float* angle) const
    {
        float scale = std::sqrt(x * x + y * y + z * z);
        if (scale > 0.0f) {
            axis->x = x / scale;
            axis->y = y / scale;
            axis->z = z / scale;
            *angle = 2.0f * std::acos(w);
        } else {
            // If quaternion is close to identity, axis is arbitrary
            axis->x = 1.0f;
            axis->y = 0.0f;
            axis->z = 0.0f;
            *angle = 0.0f;
        }
    }

    /**
     * @brief Converts quaternion to Euler angles
     * 
     * Extracts pitch, yaw, and roll angles from the quaternion
     * 
     * @return Vec3 Vector containing (pitch, yaw, roll) in radians
     * @note May experience gimbal lock when pitch approaches ±90 degrees
     */
    Vec3 to_euler() const
    {
        float sinr_cosp = 2.0f * (w * x + y * z);
        float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        float roll = std::atan2(sinr_cosp, cosr_cosp);

        float sinp = 2.0f * (w * y - z * x);
        float pitch;
        if (std::abs(sinp) >= 1.0f)
            pitch = std::copysign(1.57079632679f, sinp); // Gimbal lock
        else
            pitch = std::asin(sinp);

        float siny_cosp = 2.0f * (w * z + x * y);
        float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        float yaw = std::atan2(siny_cosp, cosy_cosp);

        return Vec3(pitch, yaw, roll);
    }
};

} // namespace bpm

#endif // BPM_QUAT_HPP
