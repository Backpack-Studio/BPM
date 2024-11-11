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

#ifndef BPM_MAT3_HPP
#define BPM_MAT3_HPP

#include "./vecx.hpp"
#include "./vec2.hpp"
#include "./vec3.hpp"

#include <cmath>

namespace bpm {

/**
 * @brief 3x3 Matrix structure
 */
class Mat3 : public Vector<float, 3*3, Mat3>
{
public:
    /**
     * @brief Default constructor
     */
    constexpr Mat3()
        : Vector<float, 3*3, Mat3>()
    { }

    /**
     * @brief Constructor from individual elements
     * @param m0 Element at index (0, 0)
     * @param m3 Element at index (0, 1)
     * @param m6 Element at index (0, 2)
     * @param m1 Element at index (1, 0)
     * @param m4 Element at index (1, 1)
     * @param m7 Element at index (1, 2)
     * @param m2 Element at index (2, 0)
     * @param m5 Element at index (2, 1)
     * @param m8 Element at index (2, 2)
     */
    constexpr Mat3(float m0, float m3, float m6,
                   float m1, float m4, float m7,
                   float m2, float m5, float m8)
    : Vector<float, 3*3, Mat3>({
        m0, m1, m2,
        m3, m4, m5,
        m6, m7, m8
    })
    { }

    /**
     * @brief Returns the identity matrix
     * @return Identity matrix
     */
    static constexpr Mat3 identity() {
        return {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a translation matrix
     * @param x Translation in the x-axis
     * @param y Translation in the y-axis
     * @return Translation matrix
     */
    static constexpr Mat3 translate(float x, float y) {
        return {
            1.0f, 0.0f, x,
            0.0f, 1.0f, y,
            0.0f, 0.0f, 1.0f,
        };
    }

    /**
     * @brief Creates a translation matrix from a 2D vector
     * @tparam T Type of vector components
     * @param v Translation vector
     * @return Translation matrix
     */
    template <typename T>
    static constexpr Mat3 translate(const Vector2<T>& v) {
        return translate(v.x, v.y);
    }

    /**
     * @brief Creates a rotation matrix
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     */
    static constexpr Mat3 rotate(float angle) {
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        return {
            c, -s, 0.0f,
            s, c, 0.0f,
            0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a scaling matrix
     * @param sx Scaling factor in the x-axis
     * @param sy Scaling factor in the y-axis
     * @return Scaling matrix
     */
    static constexpr Mat3 scale(float sx, float sy) {
        return {
            sx, 0.0f, 0.0f,
            0.0f, sy, 0.0f,
            0.0f, 0.0f, 1.0f
        };
    }

    /**
     * @brief Creates a scaling matrix from a 2D vector
     * @tparam T Type of vector components
     * @param v Scaling vector
     * @return Scaling matrix
     */
    template <typename T>
    static constexpr Mat3 scale(const Vector2<T>& v) {
        return scale(v.x, v.y);
    }

    /**
     * @brief Multiplies this 3x3 matrix by another 3x3 matrix
     * @param other Matrix to multiply by
     * @return Result of matrix multiplication
     */
    constexpr Mat3 operator*(const Mat3& other) const {
        Mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 3; k++) {
                    sum += v[i * 3 + k] * other.v[k * 3 + j];
                }
                result.v[i * 3 + j] = sum;
            }
        }
        return result;
    }
};

/* Matrix 3x3 Algorithms Implementation */

/**
 * @brief Calculates the determinant of the 3x3 matrix
 * @return Determinant of the matrix
 */
inline constexpr float determinant(const Mat3& m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/**
 * @brief Calculates the trace of the 3x3 matrix (sum of diagonal elements)
 * @return Trace of the matrix
 */
inline constexpr float trace(const Mat3& m) {
    return m[0] + m[4] + m[8];
}

/**
 * @brief Transposes the 3x3 matrix
 * @return Transposed matrix
 */
inline constexpr Mat3 transpose(const Mat3& m) {
    return Mat3 {
        m[0], m[1], m[2],
        m[3], m[4], m[5],
        m[6], m[7], m[8]
    };
}

/**
 * @brief Inverts the 3x3 matrix if it is invertible
 * @return Inverted matrix
 */
inline constexpr Mat3 invert(const Mat3 m) {
    float det = determinant(m);
    if (det != 0.0f) {
        return Mat3::identity();
    }
    float inv_det = 1.0f / det;
    Mat3 result;
    result[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    result[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    result[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    result[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    result[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    result[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    result[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    result[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    result[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
    return result;
}

/**
 * @brief Transforms a 2D vector using a 2D transformation matrix.
 * 
 * This function applies a 2D transformation to the input vector `v` using a 
 * 3x3 matrix `matrix`. The transformation is performed by multiplying the 
 * 2D vector with the transformation matrix in homogeneous coordinates, 
 * where the vector is treated as [x, y, 1] and the result is [x', y'].
 * 
 * The matrix multiplication is performed as:
 * 
 *     [x' y'] = [x y 1] * matrix
 * 
 * @tparam T The type of the components in the input and output vectors (typically floating-point).
 * 
 * @param v The input 2D vector to be transformed.
 * @param matrix The 3x3 transformation matrix to apply to the vector.
 * 
 * @return A new 2D vector that is the result of the transformation.
 */
template <typename T>
inline constexpr Vector2<T> transform(const Vector2<T>& v, const Mat3& matrix) {
    return Vector2<T> {
        v[0] * matrix[0] + v[1] * matrix[3] + matrix[6],
        v[0] * matrix[1] + v[1] * matrix[4] + matrix[7]
    };
}

/**
 * @brief Transforms a 3D vector using a 3x3 transformation matrix.
 * 
 * This function applies a 3D transformation to the input vector `v` using a 
 * 3x3 matrix `matrix`. The transformation is performed by multiplying the 
 * 3D vector with the transformation matrix.
 * 
 * The matrix multiplication is performed as:
 * 
 *     [x' y' z'] = [x y z] * matrix
 * 
 * @tparam T The type of the components in the input and output vectors (typically floating-point).
 * 
 * @param v The input 3D vector to be transformed.
 * @param matrix The 3x3 transformation matrix to apply to the vector.
 * 
 * @return A new 3D vector that is the result of the transformation.
 */
template <typename T>
inline constexpr Vector3<T> transform(const Vector3<T>& v, const Mat3& matrix) {
    return Vector3<T> {
        v[0] * matrix[0] + v[1] * matrix[3] + v[2] * matrix[6],
        v[0] * matrix[1] + v[1] * matrix[4] + v[2] * matrix[7],
        v[0] * matrix[2] + v[1] * matrix[5] + v[2] * matrix[8]
    };
}

} // namespace bpm

#endif // BPM_MAT3_HPP
