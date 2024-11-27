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

#include "./vecx.hpp"
#include "./vec3.hpp"
#include "./vec4.hpp"
#include "./quat.hpp"
#include <iomanip>

namespace bpm {

class Mat4 : public Vector<float, 4*4, Mat4>
{
public:
    /**
     * @brief Default constructor
     */
    constexpr Mat4() = default;

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
                   float m3, float m7, float m11, float m15)
    : Vector<float, 4*4, Mat4>({
        m0, m1, m2, m3,
        m4, m5, m6, m7,
        m8, m9, m10, m11,
        m12, m13, m14, m15
    })
    { }

    /**
     * @brief Returns the identity matrix
     * @return Identity matrix
     */
    static constexpr Mat4 identity() {
        return Mat4 {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a translation matrix
     * @param x Translation in the x-axis
     * @param y Translation in the y-axis
     * @param z Translation in the z-axis
     * @return Translation matrix
     */
    static constexpr Mat4 translate(float x, float y, float z) {
        return Mat4 {
            1.0f, 0.0f, 0.0f, x,
            0.0f, 1.0f, 0.0f, y,
            0.0f, 0.0f, 1.0f, z,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a translation matrix from a 3D vector
     * @tparam T Type of vector components
     * @param v Translation vector
     * @return Translation matrix
     */
    static constexpr Mat4 translate(const Vec3& v) {
        return Mat4 {
            1.0f, 0.0f, 0.0f, v[0],
            0.0f, 1.0f, 0.0f, v[1],
            0.0f, 0.0f, 1.0f, v[2],
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a rotation matrix
     * @param x X component of the rotation axis
     * @param y Y component of the rotation axis
     * @param z Z component of the rotation axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate(float x, float y, float z, float angle) {
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

    /**
     * @brief Creates a rotation matrix from an axis and angle
     * @tparam T Type of vector components
     * @param axis Rotation axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate(const Vec3& axis, float angle) {
        return Mat4::rotate(axis.x(), axis.y(), axis.z(), angle);
    }

    /**
     * @brief Creates a rotation matrix around the X-axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_x(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Mat4 {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, c,    -s,   0.0f,
            0.0f, s,     c,   0.0f,
            0.0f, 0.0f,  0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a rotation matrix around the Y-axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_y(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Mat4 {
            c,    0.0f, s,    0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            -s,   0.0f, c,   0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a rotation matrix around the Z-axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_z(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Mat4 {
            c,    -s,   0.0f, 0.0f,
            s,     c,   0.0f, 0.0f,
            0.0f,  0.0f, 1.0f, 0.0f,
            0.0f,  0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a rotation matrix around the X, Y, and Z axes in the order specified
     * @param angle_x Rotation angle around the X-axis in radians
     * @param angle_y Rotation angle around the Y-axis in radians
     * @param angle_z Rotation angle around the Z-axis in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_xyz(float angle_x, float angle_y, float angle_z) {
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

    /**
     * @brief Creates a rotation matrix around the Z, Y, and X axes in the order specified
     * @param angle_z Rotation angle around the Z-axis in radians
     * @param angle_y Rotation angle around the Y-axis in radians
     * @param angle_x Rotation angle around the X-axis in radians
     * @return Rotation matrix
     */
    static Mat4 rotate_zyx(float angle_z, float angle_y, float angle_x) {
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

    /**
     * @brief Creates a scaling matrix
     * @param sx Scaling factor in the x-axis
     * @param sy Scaling factor in the y-axis
     * @param sz Scaling factor in the z-axis
     * @return Scaling matrix
     */
    static constexpr Mat4 scale(float sx, float sy, float sz) {
        return {
            sx, 0.0f, 0.0f, 0.0f,
            0.0f, sy, 0.0f, 0.0f,
            0.0f, 0.0f, sz, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a scaling matrix from a 3D vector
     * @tparam T Type of vector components
     * @param v Scaling factors in each axis
     * @return Scaling matrix
     */
    static constexpr Mat4 scale(const Vec3& v) {
        return Mat4::scale(v[0], v[1], v[2]);
    }

    /**
     * @brief Creates a transformation matrix from translation, scale, and rotation
     * @tparam T Type of vector components
     * @param translate Translation vector
     * @param scale Scaling vector
     * @param axis Axis of rotation
     * @param angle Angle of rotation in radians
     * @return Transformation matrix
     */
    static Mat4 transform(const Vec3& translate, const Vec3& scale, const Vec3 axis, float angle) {
        return Mat4::scale(scale) * Mat4::rotate(axis, angle) * Mat4::translate(translate);
    }

    /**
     * @brief Creates a transformation matrix from translation, scale, and quaternion rotation
     * @tparam T Type of vector components
     * @param translate Translation vector
     * @param scale Scaling vector
     * @param quaternion Quaternion representing rotation
     * @return Transformation matrix
     */
    static Mat4 transform(const Vec3& translate, const Vec3& scale, const Quat& quaternion) {
        return Mat4::scale(scale) * Mat4::from_quat(quaternion) * Mat4::translate(translate);
    }

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
    static constexpr Mat4 frustum(double left, double right, double bottom, double top, double near, double far) {
        double rl = right - left;
        double tb = top - bottom;
        double fn = far - near;
        return Mat4 {
            static_cast<float>(2.0 * near / rl), 0.0f, static_cast<float>((right + left) / rl), 0.0f,
            0.0f, static_cast<float>(2.0 * near / tb), static_cast<float>((top + bottom) / tb), 0.0f,
            0.0f, 0.0f, static_cast<float>(-(far + near) / fn), static_cast<float>(-2.0 * far * near / fn),
            0.0f, 0.0f, -1.0f, 0.0f
        };
    }

    /**
     * @brief Creates a perspective projection matrix
     * @param fovy Field of view angle in the y direction in radians
     * @param aspect Aspect ratio, defined as width divided by height
     * @param near Distance to the near depth clipping plane (positive)
     * @param far Distance to the far depth clipping plane (positive)
     * @return Perspective projection matrix
     */
    static Mat4 perspective(double fovy, double aspect, double near, double far) {
        double tan_half_fovy = std::tan(fovy / 2.0f);
        return Mat4 {
            static_cast<float>(1.0 / (aspect * tan_half_fovy)), 0.0f, 0.0f, 0.0f,
            0.0f, static_cast<float>(1.0 / tan_half_fovy), 0.0f, 0.0f,
            0.0f, 0.0f, static_cast<float>(-(far + near) / (far - near)), static_cast<float>(-2.0 * far * near / (far - near)),
            0.0f, 0.0f, -1.0f, 0.0f
        };
    }

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
    static constexpr Mat4 ortho(double left, double right, double bottom, double top, double near, double far) {
        double rl = right - left;
        double tb = top - bottom;
        double fn = far - near;
        return Mat4 {
            static_cast<float>(2.0 / rl), 0.0f, 0.0f, static_cast<float>(-(right + left) / rl),
            0.0f, static_cast<float>(2.0 / tb), 0.0f, static_cast<float>(-(top + bottom) / tb),
            0.0f, 0.0f, static_cast<float>(-2.0 / fn), static_cast<float>(-(far + near) / fn),
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a view matrix using the look-at algorithm
     * @tparam T Type of vector components
     * @param eye Position of the camera
     * @param target Position where the camera is looking
     * @param up Up direction in world space (usually [0, 1, 0])
     * @return View matrix
     */
    static Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Vector3 zaxis = normalize(eye - target);
        Vector3 xaxis = normalize(cross(up, zaxis));
        Vector3 yaxis = cross(zaxis, xaxis);

        Mat4 mat_view;

        mat_view[0]  = xaxis[0];
        mat_view[1]  = yaxis[0];
        mat_view[2]  = zaxis[0];
        mat_view[3]  = 0.0f;

        mat_view[4]  = xaxis[1];
        mat_view[5]  = yaxis[1];
        mat_view[6]  = zaxis[1];
        mat_view[7]  = 0.0f;

        mat_view[8]  = xaxis[2];
        mat_view[9]  = yaxis[2];
        mat_view[10] = zaxis[2];
        mat_view[11] = 0.0f;

        mat_view[12] = -dot(xaxis, eye);
        mat_view[13] = -dot(yaxis, eye);
        mat_view[14] = -dot(zaxis, eye);
        mat_view[15] = 1.0f;

        return mat_view;
    }

    /**
     * @brief Creates a matrix from a quaternion rotation
     * @tparam T Type of vector components
     * @param q Quaternion rotation
     * @return Matrix representing the quaternion rotation
     */
    static constexpr Mat4 from_quat(const Quat& q) {
        Mat4 result;

        float xx = q[1] * q[1];
        float yy = q[2] * q[2];
        float zz = q[3] * q[3];
        float xy = q[1] * q[2];
        float xz = q[1] * q[3];
        float yz = q[2] * q[3];
        float wx = q[0] * q[1];
        float wy = q[0] * q[2];
        float wz = q[0] * q[3];

        result[0]  = 1 - 2 * (yy + zz);   // 1st row, 1st column
        result[1]  = 2 * (xy - wz);       // 2nd row, 1st column
        result[2]  = 2 * (xz + wy);       // 3rd row, 1st column
        result[3]  = 0;                   // 4th row, 1st column (homogeneous)

        result[4]  = 2 * (xy + wz);       // 1st row, 2nd column
        result[5]  = 1 - 2 * (xx + zz);   // 2nd row, 2nd column
        result[6]  = 2 * (yz - wx);       // 3rd row, 2nd column
        result[7]  = 0;                   // 4th row, 2nd column (homogeneous)

        result[8]  = 2 * (xz - wy);       // 1st row, 3rd column
        result[9]  = 2 * (yz + wx);       // 2nd row, 3rd column
        result[10] = 1 - 2 * (xx + yy);   // 3rd row, 3rd column
        result[11] = 0;                   // 4th row, 3rd column (homogeneous)

        result[12] = 0;                   // 1st row, 4th column (translation x)
        result[13] = 0;                   // 2nd row, 4th column (translation y)
        result[14] = 0;                   // 3rd row, 4th column (translation z)
        result[15] = 1;                   // 4th row, 4th column (homogeneous)

        return result;
    }

    /**
     * @brief Multiplication operator for matrices (4x4)
     * @param other Matrix to multiply with
     * @return Result of multiplication
     */
    constexpr Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += v[i * 4 + k] * other.v[k * 4 + j];
                }
                result.v[i * 4 + j] = sum;
            }
        }
        return result;
    }

    /**
     * @brief Overload of the stream insertion operator for a 4x4 matrix (Mat4).
     * 
     * This operator formats and prints a 4x4 matrix in a readable way, with each element
     * enclosed in square brackets and aligned in columns.
     * 
     * @param os The output stream to write to.
     * @param m The Mat4 matrix to print.
     * @return The output stream (to allow chaining).
     */
    friend std::ostream& operator<<(std::ostream& os, const Mat4& m) {
        os << "Mat4(\n";
        for (int row = 0; row < 4; ++row) {
            os << "  ";  // Indentation for clean display
            for (int col = 0; col < 4; ++col) {
                os << "[";  // Start of the element
                os << std::setw(8) << std::setprecision(4) << std::fixed << m.v[col * 4 + row];  // Print element with formatting
                os << "]";  // End of the element
                if (col < 3) os << ", ";  // Space between columns
            }
            os << '\n';  // New line after each row
        }
        os << ")";
        return os;
    }
};


/* Matrix 4x4 Algorithms Implementation */

/**
 * @brief Extracts the translation vector from a 4x4 transformation matrix.
 * 
 * This function extracts the translation component of a 4x4 transformation matrix `m`.
 * A 4x4 transformation matrix typically encodes transformations like rotation, scaling, 
 * and translation. The translation part is contained in the last column of the matrix, 
 * specifically the elements at indices 12, 13, and 14.
 * 
 * The function returns a `Vec3` vector that contains the translation values, 
 * corresponding to the X, Y, and Z components of the transformation.
 * 
 * @param m The 4x4 transformation matrix from which to extract the translation.
 * 
 * @return A `Vec3` containing the translation vector (x, y, z).
 */
inline constexpr Vec3 get_translation(const Mat4& m) {
    return { m[12], m[13], m[14] };
}

/**
 * @brief Extracts the rotation component from a 4x4 transformation matrix.
 * 
 * This function extracts the rotation component of a 4x4 transformation matrix `m` and returns it as a quaternion.
 * The rotation matrix is represented by the top-left 3x3 part of the matrix. The function uses the trace of the 
 * matrix to determine the best method to compute the quaternion. It handles different cases based on the value of 
 * the trace and the largest diagonal element of the matrix.
 * 
 * The algorithm follows the method for converting a rotation matrix to a quaternion, which is well-known in 3D 
 * graphics and physics.
 * 
 * @param m The 4x4 transformation matrix from which to extract the rotation.
 * 
 * @return A quaternion representing the rotation component of the matrix.
 * 
 * @note The quaternion is normalized before being returned to ensure it is a unit quaternion.
 */
inline Quat get_rotation(const Mat4& m) {
    // Extract the elements from the matrix
    float trace = m[0] + m[5] + m[10]; // Trace of the matrix

    Quat q;

    if (trace > 0) {
        // If the trace is positive, we use the quaternion method
        float s = std::sqrt(trace + 1.0) * 2; // S = 4 * w
        q[0] = 0.25f * s;
        q[1] = (m[6] - m[9]) / s;
        q[2] = (m[8] - m[2]) / s;
        q[3] = (m[1] - m[4]) / s;
    } else {
        // If the trace is non-positive, we determine which index is the largest
        if (m[0] > m[5] && m[0] > m[10]) {
            float s = std::sqrt(1.0 + m[0] - m[5] - m[10]) * 2; // S = 4 * x
            q[0] = (m[6] - m[9]) / s;
            q[1] = 0.25f * s;
            q[2] = (m[1] + m[4]) / s;
            q[3] = (m[2] + m[8]) / s;
        } else if (m[5] > m[10]) {
            float s = std::sqrt(1.0 + m[5] - m[0] - m[10]) * 2; // S = 4 * y
            q[0] = (m[8] - m[2]) / s;
            q[1] = (m[1] + m[4]) / s;
            q[2] = 0.25f * s;
            q[3] = (m[6] + m[9]) / s;
        } else {
            float s = std::sqrt(1.0 + m[10] - m[0] - m[5]) * 2; // S = 4 * z
            q[0] = (m[1] - m[4]) / s;
            q[1] = (m[2] + m[8]) / s;
            q[2] = (m[6] + m[9]) / s;
            q[3] = 0.25f * s;
        }
    }

    // Normalize the quaternion before returning it
    return normalize(q);
}

/**
 * @brief Computes the determinant of a 4x4 matrix.
 * 
 * This function calculates the determinant of a 4x4 matrix `m` using cofactor expansion 
 * along the first row. The determinant is a scalar value that can be used to determine 
 * properties such as whether a matrix is invertible (a non-zero determinant implies invertibility).
 * 
 * The formula used for the determinant calculation is:
 * 
 *     det(m) = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8])
 * 
 * This approach calculates the determinant by expanding along the first row of the matrix.
 * 
 * @param m The 4x4 matrix for which to calculate the determinant.
 * 
 * @return The determinant of the matrix as a scalar value of type `float`.
 * 
 * @note The determinant is computed using a direct approach based on matrix components. 
 *       This is suitable for small matrices like 4x4, but for larger matrices, more efficient 
 *       algorithms may be necessary.
 */
inline constexpr float determinant(const Mat4& m) {
    return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
}

/**
 * @brief Computes the trace of a 4x4 matrix.
 * 
 * This function calculates the trace of a 4x4 matrix `m`. The trace of a matrix is the sum 
 * of the diagonal elements, which are the elements where the row and column indices are equal 
 * (i.e., m[0], m[5], m[10], and m[15] for a 4x4 matrix).
 * 
 * The formula for the trace of a 4x4 matrix is:
 * 
 *     trace(m) = m[0] + m[5] + m[10] + m[15]
 * 
 * The trace has applications in various fields, such as in determining the properties of 
 * transformations and in certain matrix operations like the computation of eigenvalues.
 * 
 * @param m The 4x4 matrix for which to calculate the trace.
 * 
 * @return The trace of the matrix as a scalar value of type `float`.
 * 
 * @note The trace is only defined for square matrices, but this function assumes the 
 *       matrix is 4x4.
 */
inline constexpr float trace(const Mat4& m) {
    return m[0] + m[5] + m[10] + m[15];
}

/**
 * @brief Computes the transpose of a 4x4 matrix.
 * 
 * This function computes the transpose of a given 4x4 matrix `m`. The transpose of a matrix 
 * is obtained by swapping its rows and columns. For a matrix `m`, the transpose `m^T` is defined as:
 * 
 *     m^T[i][j] = m[j][i]
 * 
 * Specifically, for a 4x4 matrix, the elements of the matrix are rearranged such that:
 * 
 *     m^T[0] = m[0], m^T[1] = m[4], m^T[2] = m[8], m^T[3] = m[12]
 *     m^T[4] = m[1], m^T[5] = m[5], m^T[6] = m[9], m^T[7] = m[13]
 *     m^T[8] = m[2], m^T[9] = m[6], m^T[10] = m[10], m^T[11] = m[14]
 *     m^T[12] = m[3], m^T[13] = m[7], m^T[14] = m[11], m^T[15] = m[15]
 * 
 * The transposition of a matrix is often used in various mathematical and geometric operations,
 * such as in vector transformations, rotation operations, and changing coordinate systems.
 * 
 * @param m The 4x4 matrix to transpose.
 * 
 * @return A new 4x4 matrix that is the transpose of `m`.
 * 
 * @note The transposition operation does not alter the original matrix `m`; instead, it creates 
 *       and returns a new matrix with the rows and columns swapped.
 */
inline constexpr Mat4 transpose(const Mat4& m) {
    return Mat4 {
        m[0], m[1], m[2], m[3],
        m[4], m[5], m[6], m[7],
        m[8], m[9], m[10], m[11],
        m[12], m[13], m[14], m[15]
    };
}

/**
 * @brief Computes the inverse of a 4x4 matrix.
 * 
 * This function computes the inverse of a given 4x4 matrix `m`. The inverse of a matrix `A`, 
 * denoted `A^-1`, is a matrix such that when multiplied by `A`, the result is the identity matrix:
 * 
 *     A * A^-1 = I
 * 
 * In the case of a 4x4 matrix, the inverse is calculated using the adjoint (or adjugate) matrix 
 * and the determinant. If the determinant is non-zero, the matrix is invertible. The formula used 
 * for the inverse is based on the cofactor matrix and the determinant of the original matrix.
 * 
 * The matrix inversion formula for a 4x4 matrix involves several intermediate calculations and 
 * is applied in this function to compute the inverse.
 * 
 * @param m The 4x4 matrix to invert.
 * 
 * @return The inverted 4x4 matrix.
 * 
 * @note If the determinant of the matrix is zero, the matrix is not invertible, and this function 
 *       will return an incorrect result. It is assumed that the matrix is invertible when calling 
 *       this function.
 */
inline constexpr Mat4 invert(const Mat4& m) {
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

/**
 * @brief Transforms a 3D vector by a 4x4 transformation matrix.
 * 
 * This function applies a 4x4 matrix transformation to a 3D vector `v`. The transformation is applied 
 * using a standard matrix multiplication, and the result is a new transformed 3D vector. The matrix 
 * represents a combination of translation, rotation, and scaling.
 * 
 * The matrix multiplication is performed as follows:
 * 
 *     [ x' ]   =   [ x ] * matrix[0..3] + weight * matrix[12]
 *     [ y' ]       [ y ] * matrix[4..7] + weight * matrix[13]
 *     [ z' ]       [ z ] * matrix[8..11] + weight * matrix[14]
 * 
 * Where:
 * - `x, y, z` are the components of the vector `v`.
 * - `matrix[0..3]` are the elements of the first column of the matrix, representing the linear part 
 *   of the transformation (rotation and scaling).
 * - `matrix[12..14]` are the elements of the translation part of the matrix, which is applied to 
 *   the vector `v` after the linear transformation.
 * - The optional `weight` parameter scales the translation components, providing flexibility in the 
 *   transformation (useful for weighted transformations or homogenous coordinates).
 * 
 * This function is useful for applying affine transformations such as translation, rotation, and scaling 
 * to vectors in 3D space.
 * 
 * @param v The 3D vector to transform.
 * @param matrix The 4x4 transformation matrix.
 * @param weight The scaling factor for the translation components (defaults to 1).
 * 
 * @return A new 3D vector that is the result of the transformation.
 */
template <typename T>
inline constexpr Vector3<T> transform(const Vector3<T>& v, const Mat4& matrix, T weight = 1) {
    return Vector3<T> {
        v[0] * matrix[0] + v[1] * matrix[4] + v[2] * matrix[8] + weight * matrix[12],
        v[0] * matrix[1] + v[1] * matrix[5] + v[2] * matrix[9] + weight * matrix[13],
        v[0] * matrix[2] + v[1] * matrix[6] + v[2] * matrix[10] + weight * matrix[14],
    };
}

/**
 * @brief Transforms a 4D vector by a 4x4 transformation matrix.
 * 
 * This function applies a 4x4 matrix transformation to a 4D vector `v`. The transformation is performed 
 * using matrix multiplication, and the result is a new transformed 4D vector. The matrix may represent 
 * translation, rotation, scaling, or a combination of these transformations.
 * 
 * The matrix multiplication is performed as follows:
 * 
 *     [ x' ]   =   [ x ] * matrix[0..3] + y * matrix[4..7] + z * matrix[8..11] + w * matrix[12..15]
 *     [ y' ]       [ y ]
 *     [ z' ]       [ z ]
 *     [ w' ]       [ w ]
 * 
 * Where:
 * - `x, y, z, w` are the components of the input 4D vector `v`.
 * - `matrix[0..3]` represent the linear transformation part (rotation and scaling).
 * - `matrix[12..15]` represent the translation component of the transformation.
 * 
 * This function is particularly useful for applying affine transformations (like translation, rotation, 
 * and scaling) to 4D vectors, which may represent homogeneous coordinates in computer graphics or physics.
 * 
 * @param v The 4D vector to transform.
 * @param matrix The 4x4 transformation matrix.
 * 
 * @return A new 4D vector that is the result of the transformation.
 */
template <typename T>
inline constexpr Vector4<T> transform(const Vector4<T>& v, const Mat4& matrix) {
    return Vector4<T> {
        v[0] * matrix[0] + v[1] * matrix[4] + v[2] * matrix[8] + v[3] * matrix[12],
        v[0] * matrix[1] + v[1] * matrix[5] + v[2] * matrix[9] + v[3] * matrix[13],
        v[0] * matrix[2] + v[1] * matrix[6] + v[2] * matrix[10] + v[3] * matrix[14],
        v[0] * matrix[3] + v[1] * matrix[7] + v[2] * matrix[11] + v[3] * matrix[15]
    };
}

} // namespace bpm

#endif // BPM_MAT4_HPP
