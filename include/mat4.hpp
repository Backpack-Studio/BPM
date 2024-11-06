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

#ifndef BPM_MAT4_HPP
#define BPM_MAT4_HPP

#include "./vec3.hpp"
#include "./vec4.hpp"
#include "./quat.hpp"

#include <algorithm>

namespace bpm {

struct Mat4
{
    float m[16]{};  ///< 4x4 matrix elements

    /**
     * @brief Default constructor
     */
    constexpr Mat4() = default;

    /**
     * @brief Constructor from array
     * @param mat Pointer to an array of 16 floats representing the matrix elements
     */
    Mat4(const float* mat)
    {
        std::copy(mat, mat + 16, m);
    }

    /**
     * @brief Constructor from individual elements
     * @param m0 Element at index (0, 0)
     * @param m4 Element at index (1, 1)
     * @param m8 Element at index (2, 2)
     * @param m12 Element at index (3, 3)
     * @param m1 Element at index (0, 1)
     * @param m5 Element at index (1, 1)
     * @param m9 Element at index (2, 1)
     * @param m13 Element at index (3, 1)
     * @param m2 Element at index (0, 2)
     * @param m6 Element at index (1, 2)
     * @param m10 Element at index (2, 2)
     * @param m14 Element at index (3, 2)
     * @param m3 Element at index (0, 3)
     * @param m7 Element at index (1, 3)
     * @param m11 Element at index (2, 3)
     * @param m15 Element at index (3, 3)
     */
    constexpr Mat4(float m0, float m4, float m8,  float m12,
                    float m1, float m5, float m9,  float m13,
                    float m2, float m6, float m10, float m14,
                    float m3, float m7, float m11, float m15);

    /**
     * @brief Returns the identity matrix
     * @return Identity matrix
     */
    static constexpr Mat4 identity();

    /**
     * @brief Creates a translation matrix
     * @param x Translation in the x-axis
     * @param y Translation in the y-axis
     * @param z Translation in the z-axis
     * @return Translation matrix
     */
    static constexpr Mat4 translate(float x, float y, float z);

    /**
     * @brief Creates a translation matrix from a 3D vector
     * @tparam T Type of vector components
     * @param v Translation vector
     * @return Translation matrix
     */
    static constexpr Mat4 translate(const Vec3& v);

    /**
     * @brief Creates a rotation matrix
     * @param x X component of the rotation axis
     * @param y Y component of the rotation axis
     * @param z Z component of the rotation axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate(float x, float y, float z, float angle);

    /**
     * @brief Creates a rotation matrix from an axis and angle
     * @tparam T Type of vector components
     * @param axis Rotation axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate(const Vec3& axis, float angle);

    /**
     * @brief Creates a rotation matrix around the X-axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_x(float angle);

    /**
     * @brief Creates a rotation matrix around the Y-axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_y(float angle);

    /**
     * @brief Creates a rotation matrix around the Z-axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_z(float angle);

    /**
     * @brief Creates a rotation matrix around the X, Y, and Z axes in the order specified
     * @param angle_x Rotation angle around the X-axis in radians
     * @param angle_y Rotation angle around the Y-axis in radians
     * @param angle_z Rotation angle around the Z-axis in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_xyz(float angle_x, float angle_y, float angle_z);

    /**
     * @brief Creates a rotation matrix around the Z, Y, and X axes in the order specified
     * @param angle_z Rotation angle around the Z-axis in radians
     * @param angle_y Rotation angle around the Y-axis in radians
     * @param angle_x Rotation angle around the X-axis in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_zyx(float angle_z, float angle_y, float angle_x);

    /**
     * @brief Creates a scaling matrix
     * @param sx Scaling factor in the x-axis
     * @param sy Scaling factor in the y-axis
     * @param sz Scaling factor in the z-axis
     * @return Scaling matrix
     */
    static constexpr Mat4 scale(float sx, float sy, float sz);

    /**
     * @brief Creates a scaling matrix from a 3D vector
     * @tparam T Type of vector components
     * @param v Scaling factors in each axis
     * @return Scaling matrix
     */
    static constexpr Mat4 scale(const Vec3& v);

    /**
     * @brief Creates a transformation matrix from translation, scale, and rotation
     * @tparam T Type of vector components
     * @param translate Translation vector
     * @param scale Scaling vector
     * @param axis Axis of rotation
     * @param angle Angle of rotation in radians
     * @return Transformation matrix
     */
    static Mat4 transform(const Vec3& translate, const Vec3& scale, const Vec3 axis, float angle);

    /**
     * @brief Creates a transformation matrix from translation, scale, and quaternion rotation
     * @tparam T Type of vector components
     * @param translate Translation vector
     * @param scale Scaling vector
     * @param quaternion Quaternion representing rotation
     * @return Transformation matrix
     */
    static Mat4 transform(const Vec3& translate, const Vec3& scale, const Quat& quaternion);

    /**
     * @brief Creates a perspective frustum matrix
     * @param left Coordinate for the left vertical clipping plane
     * @param right Coordinate for the right vertical clipping plane
     * @param bottom Coordinate for the bottom horizontal clipping plane
     * @param top Coordinate for the top horizontal clipping plane
     * @param near Distance to the near depth clipping plane (positive)
     * @param far Distance to the far depth clipping plane (positive)
     * @return Perspective frustum matrix
     */
    static constexpr Mat4 frustum(float left, float right, float bottom, float top, float near, float far);

    /**
     * @brief Creates a perspective projection matrix
     * @param fovy Field of view angle in the y direction in radians
     * @param aspect Aspect ratio, defined as width divided by height
     * @param near Distance to the near depth clipping plane (positive)
     * @param far Distance to the far depth clipping plane (positive)
     * @return Perspective projection matrix
     */
    static Mat4 perspective(float fovy, float aspect, float near, float far);

    /**
     * @brief Creates an orthographic projection matrix
     * @param left Coordinate for the left vertical clipping plane
     * @param right Coordinate for the right vertical clipping plane
     * @param bottom Coordinate for the bottom horizontal clipping plane
     * @param top Coordinate for the top horizontal clipping plane
     * @param near Distance to the near depth clipping plane (positive)
     * @param far Distance to the far depth clipping plane (positive)
     * @return Orthographic projection matrix
     */
    static constexpr Mat4 ortho(float left, float right, float bottom, float top, float near, float far);

    /**
     * @brief Creates a view matrix using the look-at algorithm
     * @tparam T Type of vector components
     * @param eye Position of the camera
     * @param target Position where the camera is looking
     * @param up Up direction in world space (usually [0, 1, 0])
     * @return View matrix
     */
    static Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up);

    /**
     * @brief Creates a matrix from a quaternion rotation
     * @tparam T Type of vector components
     * @param q Quaternion rotation
     * @return Matrix representing the quaternion rotation
     */
    static constexpr Mat4 from_quat(const Quat& q);

    /**
     * @brief Returns the array/pointer of the matrix
     * @return Pointer to the matrix array
     */
    constexpr operator const float*() const { return m; }

    /**
     * @brief Addition operator for the 4x4 matrix
     * @param other Matrix to add
     * @return Result of addition
     */
    constexpr Mat4 operator+(const Mat4& other) const;

    /**
     * @brief Subtraction operator for the 4x4 matrix
     * @param other Matrix to subtract
     * @return Result of subtraction
     */
    constexpr Mat4 operator-(const Mat4& other) const;

    /**
     * @brief Multiplication operator for matrices (4x4)
     * @param other Matrix to multiply with
     * @return Result of multiplication
     */
    constexpr Mat4 operator*(const Mat4& other) const;

    /**
     * @brief Scalar multiplication operator for the 4x4 matrix
     * @param scalar Scalar value to multiply with
     * @return Result of multiplication
     */
    constexpr Mat4 operator*(float scalar) const;

    /**
     * @brief Addition and assignment operator for the 4x4 matrix
     * @param other Matrix to add
     */
    constexpr void operator+=(const Mat4& other) { *this = *this + other; }

    /**
     * @brief Subtraction and assignment operator for the 4x4 matrix
     * @param other Matrix to subtract
     */
    constexpr void operator-=(const Mat4& other) { *this = *this - other; }

    /**
     * @brief Multiplication and assignment operator for matrices (4x4)
     * @param other Matrix to multiply with
     */
    constexpr void operator*=(const Mat4& other) { *this = *this * other; }

    /**
     * @brief Scalar multiplication and assignment operator for the 4x4 matrix
     * @param scalar Scalar value to multiply with
     */
    constexpr void operator*=(float scalar) { *this = *this * scalar; }

    /**
     * @brief Equality operator for the 4x4 matrix
     * @param other Matrix to compare with
     * @return True if matrices are equal, false otherwise
     */
    constexpr bool operator==(const Mat4& other) const;

    /**
     * @brief Inequality operator for the 4x4 matrix
     * @param other Matrix to compare with
     * @return True if matrices are not equal, false otherwise
     */
    constexpr bool operator!=(const Mat4& other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Gets the translation component of the matrix
     * @tparam T Type of vector components
     * @return Translation vector
     */
    constexpr Vec3 get_translation() const;

    /**
     * @brief Gets the rotation component of the matrix as a quaternion
     * @tparam T Type of vector components
     * @return Rotation quaternion
     */
    Quat get_rotation() const;

    /**
     * @brief Calculates the determinant of the matrix
     * @return Determinant value
     */
    constexpr float determinant() const;

    /**
     * @brief Calculates the trace of the matrix (sum of values on the diagonal)
     * @return Trace value
     */
    constexpr float trace() const;

    /**
     * @brief Transposes the matrix
     * @return Transposed matrix
     */
    constexpr Mat4 transpose() const;

    /**
     * @brief Inverts the matrix (if it is invertible)
     * @return Inverted matrix
     */
    constexpr Mat4 invert() const;
};

/* Constructors */

constexpr Mat4::Mat4(float m0, float m4, float m8,  float m12,
                        float m1, float m5, float m9,  float m13,
                        float m2, float m6, float m10, float m14,
                        float m3, float m7, float m11, float m15)
{
    m[0] = m0; m[4] = m4; m[8]  = m8;  m[12] = m12;
    m[1] = m1; m[5] = m5; m[9]  = m9;  m[13] = m13;
    m[2] = m2; m[6] = m6; m[10] = m10; m[14] = m14;
    m[3] = m3; m[7] = m7; m[11] = m11; m[15] = m15;
}

/* Static Methods */

constexpr Mat4 Mat4::identity() {
    return Mat4 {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

constexpr Mat4 Mat4::translate(float x, float y, float z) {
    return Mat4 {
        1.0f, 0.0f, 0.0f, x,
        0.0f, 1.0f, 0.0f, y,
        0.0f, 0.0f, 1.0f, z,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

constexpr Mat4 Mat4::translate(const Vec3& v) {
    return translate(v.x, v.y, v.z);
}

inline Mat4 Mat4::rotate(float x, float y, float z, float angle) {
    float lenSq = x * x + y * y + z * z;
    if (lenSq != 1.0f && lenSq != 0.0f) {
        float len = 1.0f / std::sqrt(lenSq);
        x *= len, y *= len, z *= len;
    }
    float s = std::sin(angle);
    float c = std::cos(angle);
    float t = 1.0f - c;
    return Mat4 {
        x * x * t + c, x * y * t - z * s, x * z * t + y * s, 0.0f,
        y * x * t + z * s, y * y * t + c, y * z * t - x * s, 0.0f,
        z * x * t - y * s, z * y * t + x * s, z * z * t + c, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

inline Mat4 Mat4::rotate(const Vec3& axis, float angle) {
    return rotate(axis.x, axis.y, axis.z, angle);
}

inline Mat4 Mat4::rotate_x(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return Mat4 {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, c,    -s,   0.0f,
        0.0f, s,     c,   0.0f,
        0.0f, 0.0f,  0.0f, 1.0f
    };
}

inline Mat4 Mat4::rotate_y(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return Mat4 {
        c,    0.0f, s,    0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        -s,   0.0f, c,   0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

inline Mat4 Mat4::rotate_z(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return Mat4 {
        c,    -s,   0.0f, 0.0f,
        s,     c,   0.0f, 0.0f,
        0.0f,  0.0f, 1.0f, 0.0f,
        0.0f,  0.0f, 0.0f, 1.0f
    };
}

inline Mat4 Mat4::rotate_xyz(float angle_x, float angle_y, float angle_z) {
    float cx = std::cos(angle_x);
    float sx = std::sin(angle_x);
    float cy = std::cos(angle_y);
    float sy = std::sin(angle_y);
    float cz = std::cos(angle_z);
    float sz = std::sin(angle_z);
    return Mat4 {
        cy * cz, -cy * sz, sy, 0.0f,
        sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy, 0.0f,
        -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

inline Mat4 Mat4::rotate_zyx(float angle_z, float angle_y, float angle_x) {
    float cx = std::cos(angle_x);
    float sx = std::sin(angle_x);
    float cy = std::cos(angle_y);
    float sy = std::sin(angle_y);
    float cz = std::cos(angle_z);
    float sz = std::sin(angle_z);
    return Mat4 {
        cy * cz, -sz, cz * sy, 0.0f,
        cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy, 0.0f,
        sx * sz - cx * sy * cz, cx * sy * sz + sx * cz, cx * cy, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

constexpr Mat4 Mat4::scale(float sx, float sy, float sz) {
    return {
        sx, 0.0f, 0.0f, 0.0f,
        0.0f, sy, 0.0f, 0.0f,
        0.0f, 0.0f, sz, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

constexpr Mat4 Mat4::scale(const Vec3& v) {
    return Mat4::scale(v.x, v.y, v.z);
}

inline Mat4 Mat4::transform(const Vec3& translate, const Vec3& scale, const Vec3 axis, float angle) {
    return Mat4::scale(scale) * Mat4::rotate(axis, angle) * Mat4::translate(translate);
}

inline Mat4 Mat4::transform(const Vec3& translate, const Vec3& scale, const Quat& quaternion) {
    return Mat4::scale(scale) * Mat4::from_quat(quaternion) * Mat4::translate(translate);
}

constexpr Mat4 Mat4::frustum(float left, float right, float bottom, float top, float near, float far) {
    float rl = right - left;
    float tb = top - bottom;
    float fn = far - near;
    return Mat4 {
        2.0f * near / rl, 0.0f, (right + left) / rl, 0.0f,
        0.0f, 2.0f * near / tb, (top + bottom) / tb, 0.0f,
        0.0f, 0.0f, -(far + near) / fn, -2.0f * far * near / fn,
        0.0f, 0.0f, -1.0f, 0.0f
    };
}

inline Mat4 Mat4::perspective(float fovy, float aspect, float near, float far) {
    float tanHalfFovy = std::tan(fovy / 2.0f);
    return Mat4 {
        1.0f / (aspect * tanHalfFovy), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tanHalfFovy, 0.0f, 0.0f,
        0.0f, 0.0f, -(far + near) / (far - near), -2.0f * far * near / (far - near),
        0.0f, 0.0f, -1.0f, 0.0f
    };
}

constexpr Mat4 Mat4::ortho(float left, float right, float bottom, float top, float near, float far) {
    float rl = right - left;
    float tb = top - bottom;
    float fn = far - near;
    return Mat4 {
        2.0f / rl, 0.0f, 0.0f, -(right + left) / rl,
        0.0f, 2.0f / tb, 0.0f, -(top + bottom) / tb,
        0.0f, 0.0f, -2.0f / fn, -(far + near) / fn,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

inline Mat4 Mat4::look_at(const Vec3& eye, const Vec3& target, const Vec3& up) {
    Vector3 zaxis = (eye - target).normalized();
    Vector3 xaxis = up.cross(zaxis).normalized();
    Vector3 yaxis = zaxis.cross(xaxis);

    Mat4 mat_view;

    mat_view.m[0]  = xaxis.x;
    mat_view.m[1]  = yaxis.x;
    mat_view.m[2]  = zaxis.x;
    mat_view.m[3]  = 0.0f;

    mat_view.m[4]  = xaxis.y;
    mat_view.m[5]  = yaxis.y;
    mat_view.m[6]  = zaxis.y;
    mat_view.m[7]  = 0.0f;

    mat_view.m[8]  = xaxis.z;
    mat_view.m[9]  = yaxis.z;
    mat_view.m[10] = zaxis.z;
    mat_view.m[11] = 0.0f;

    mat_view.m[12] = -xaxis.dot(eye);
    mat_view.m[13] = -yaxis.dot(eye);
    mat_view.m[14] = -zaxis.dot(eye);
    mat_view.m[15] = 1.0f;

    return mat_view;
}

constexpr Mat4 Mat4::from_quat(const Quat& q) {
    Mat4 result;

    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    result.m[0]  = 1 - 2 * (yy + zz);   // 1st row, 1st column
    result.m[1]  = 2 * (xy - wz);       // 2nd row, 1st column
    result.m[2]  = 2 * (xz + wy);       // 3rd row, 1st column
    result.m[3]  = 0;                   // 4th row, 1st column (homogeneous)

    result.m[4]  = 2 * (xy + wz);       // 1st row, 2nd column
    result.m[5]  = 1 - 2 * (xx + zz);   // 2nd row, 2nd column
    result.m[6]  = 2 * (yz - wx);       // 3rd row, 2nd column
    result.m[7]  = 0;                   // 4th row, 2nd column (homogeneous)

    result.m[8]  = 2 * (xz - wy);       // 1st row, 3rd column
    result.m[9]  = 2 * (yz + wx);       // 2nd row, 3rd column
    result.m[10] = 1 - 2 * (xx + yy);   // 3rd row, 3rd column
    result.m[11] = 0;                   // 4th row, 3rd column (homogeneous)

    result.m[12] = 0;                   // 1st row, 4th column (translation x)
    result.m[13] = 0;                   // 2nd row, 4th column (translation y)
    result.m[14] = 0;                   // 3rd row, 4th column (translation z)
    result.m[15] = 1;                   // 4th row, 4th column (homogeneous)

    return result;
}

/* Operators */

constexpr Mat4 Mat4::operator+(const Mat4& other) const {
    Mat4 result;
    for (int i = 0; i < 16; ++i) {
        result.m[i] = m[i] + other.m[i];
    }
    return result;
}

constexpr Mat4 Mat4::operator-(const Mat4& other) const {
    Mat4 result;
    for (int i = 0; i < 16; ++i) {
        result.m[i] = m[i] - other.m[i];
    }
    return result;
}

constexpr Mat4 Mat4::operator*(const Mat4& other) const {
    Mat4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += m[i * 4 + k] * other.m[k * 4 + j];
            }
            result.m[i * 4 + j] = sum;
        }
    }
    return result;
}

constexpr Mat4 Mat4::operator*(float scalar) const {
    Mat4 result;
    for (int i = 0; i < 16; i++) {
        result.m[i] = m[i] * scalar;
    }
    return result;
}

constexpr bool Mat4::operator==(const Mat4& other) const {
    for (int i = 0; i < 16; i++) {
        if (m[i] != other.m[i]) {
            return false;
        }
    }
    return true;
}

/* Mat4 Methods */

constexpr Vec3 Mat4::get_translation() const {
    return { m[12], m[13], m[14] };
}

inline Quat Mat4::get_rotation() const {
    // Extract the elements from the matrix
    float trace = m[0] + m[5] + m[10]; // Trace of the matrix

    Quat q;

    if (trace > 0) {
        // If the trace is positive, we use the quaternion method
        float s = std::sqrt(trace + 1.0) * 2; // S = 4 * w
        q.w = 0.25f * s;
        q.x = (m[6] - m[9]) / s;
        q.y = (m[8] - m[2]) / s;
        q.z = (m[1] - m[4]) / s;
    } else {
        // If the trace is non-positive, we determine which index is the largest
        if (m[0] > m[5] && m[0] > m[10]) {
            float s = std::sqrt(1.0 + m[0] - m[5] - m[10]) * 2; // S = 4 * x
            q.w = (m[6] - m[9]) / s;
            q.x = 0.25f * s;
            q.y = (m[1] + m[4]) / s;
            q.z = (m[2] + m[8]) / s;
        } else if (m[5] > m[10]) {
            float s = std::sqrt(1.0 + m[5] - m[0] - m[10]) * 2; // S = 4 * y
            q.w = (m[8] - m[2]) / s;
            q.x = (m[1] + m[4]) / s;
            q.y = 0.25f * s;
            q.z = (m[6] + m[9]) / s;
        } else {
            float s = std::sqrt(1.0 + m[10] - m[0] - m[5]) * 2; // S = 4 * z
            q.w = (m[1] - m[4]) / s;
            q.x = (m[2] + m[8]) / s;
            q.y = (m[6] + m[9]) / s;
            q.z = 0.25f * s;
        }
    }

    // Normalize the quaternion before returning it
    return q.normalize();
}

constexpr float Mat4::determinant() const {
    return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
}

constexpr float Mat4::trace() const {
    return m[0] + m[5] + m[10] + m[15];
}

constexpr Mat4 Mat4::transpose() const {
    return Mat4 {
        m[0], m[1], m[2], m[3],
        m[4], m[5], m[6], m[7],
        m[8], m[9], m[10], m[11],
        m[12], m[13], m[14], m[15]
    };
}

constexpr Mat4 Mat4::invert() const {
    float a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
    float a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    float a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
    float a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

    float b00 = a00*a11 - a01*a10;
    float b01 = a00*a12 - a02*a10;
    float b02 = a00*a13 - a03*a10;
    float b03 = a01*a12 - a02*a11;
    float b04 = a01*a13 - a03*a11;
    float b05 = a02*a13 - a03*a12;
    float b06 = a20*a31 - a21*a30;
    float b07 = a20*a32 - a22*a30;
    float b08 = a20*a33 - a23*a30;
    float b09 = a21*a32 - a22*a31;
    float b10 = a21*a33 - a23*a31;
    float b11 = a22*a33 - a23*a32;

    float invDet = 1.0f / (b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06);

    return Mat4 {
        (a11 * b11 - a12 * b10 + a13 * b09) * invDet,
        (-a10 * b11 + a12 * b08 - a13 * b07) * invDet,
        (a10 * b10 - a11 * b08 + a13 * b06) * invDet,
        (-a10 * b09 + a11 * b07 - a12 * b06) * invDet,
        (-a01 * b11 + a02 * b10 - a03 * b09) * invDet,
        (a00 * b11 - a02 * b08 + a03 * b07) * invDet,
        (-a00 * b10 + a01 * b08 - a03 * b06) * invDet,
        (a00 * b09 - a01 * b07 + a02 * b06) * invDet,
        (a31 * b05 - a32 * b04 + a33 * b03) * invDet,
        (-a30 * b05 + a32 * b02 - a33 * b01) * invDet,
        (a30 * b04 - a31 * b02 + a33 * b00) * invDet,
        (-a30 * b03 + a31 * b01 - a32 * b00) * invDet,
        (-a21 * b05 + a22 * b04 - a23 * b03) * invDet,
        (a20 * b05 - a22 * b02 + a23 * b01) * invDet,
        (-a20 * b04 + a21 * b02 - a23 * b00) * invDet,
        (a20 * b03 - a21 * b01 + a22 * b00) * invDet
    };
}

/* Vec3 Transformation Methods */

template <typename T>
void Vector3<T>::transform(const Mat4& matrix) {
    *this = {
        x * matrix.m[0] + y * matrix.m[4] + z * matrix.m[8] + matrix.m[12],
        x * matrix.m[1] + y * matrix.m[5] + z * matrix.m[9] + matrix.m[13],
        x * matrix.m[2] + y * matrix.m[6] + z * matrix.m[10] + matrix.m[14],
    };
}

template <typename T>
Vector3<T> Vector3<T>::transformed(const Mat4& matrix) const {
    return {
        x * matrix.m[0] + y * matrix.m[4] + z * matrix.m[8] + matrix.m[12],
        x * matrix.m[1] + y * matrix.m[5] + z * matrix.m[9] + matrix.m[13],
        x * matrix.m[2] + y * matrix.m[6] + z * matrix.m[10] + matrix.m[14]
    };
}

/* Vec4 Transformation Methods */

template <typename T>
void Vector4<T>::transform(const Mat4& matrix) {
    *this = {
        x * matrix.m[0] + y * matrix.m[4] + z * matrix.m[8] + w * matrix.m[12],
        x * matrix.m[1] + y * matrix.m[5] + z * matrix.m[9] + w * matrix.m[13],
        x * matrix.m[2] + y * matrix.m[6] + z * matrix.m[10] + w * matrix.m[14],
        x * matrix.m[3] + y * matrix.m[7] + z * matrix.m[11] + w * matrix.m[15]
    };
}

template <typename T>
Vector4<T> Vector4<T>::transformed(const Mat4& matrix) const {
    return {
        x * matrix.m[0] + y * matrix.m[4] + z * matrix.m[8] + w * matrix.m[12],
        x * matrix.m[1] + y * matrix.m[5] + z * matrix.m[9] + w * matrix.m[13],
        x * matrix.m[2] + y * matrix.m[6] + z * matrix.m[10] + w * matrix.m[14],
        x * matrix.m[3] + y * matrix.m[7] + z * matrix.m[11] + w * matrix.m[15]
    };
}

} // namespace bpm

#endif // BPM_MAT4_HPP
