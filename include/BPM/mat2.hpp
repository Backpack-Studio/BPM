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

#ifndef BPM_MAT2_HPP
#define BPM_MAT2_HPP

#include "./vecx.hpp"
#include <iomanip>

namespace bpm {

/**
 * @brief 2x2 Matrix structure
 */
class Mat2 : public Vector<float, 2*2, Mat2>
{
public:
    /**
     * @brief Default constructor
     */
    constexpr Mat2()
        : Vector<float, 2*2, Mat2>()
    { }

    /**
     * @brief Constructor from individual elements
     * @param m0 Element at index (0, 0)
     * @param m2 Element at index (0, 1)
     * @param m1 Element at index (1, 0)
     * @param m3 Element at index (1, 1)
     */
    constexpr Mat2(float m0, float m2, float m1, float m3) {
        v[0] = m0; v[2] = m1;
        v[1] = m2; v[3] = m3;
    }

    /**
     * @brief Returns the identity matrix
     * @return Identity matrix
     */
    static constexpr Mat2 identity() {
        return {
            1.0f, 0.0f,
            0.0f, 1.0f
        };
    }

    /**
     * @brief Multiplies two matrices
     * @param other Matrix to multiply by
     * @return Result of matrix multiplication
     */
    constexpr Mat2 operator*(const Mat2& other) const {
        return {
            v[0] * other.v[0] + v[1] * other.v[2], v[2] * other.v[0] + v[3] * other.v[2],
            v[0] * other.v[1] + v[1] * other.v[3], v[2] * other.v[1] + v[3] * other.v[3]
        };
    }

    /**
     * @brief Overload of the stream insertion operator for a 2x2 matrix (Mat2).
     * 
     * This operator formats and prints a 2x2 matrix in a readable way, with each element
     * enclosed in square brackets and aligned in columns.
     * 
     * @param os The output stream to write to.
     * @param m The Mat2 matrix to print.
     * @return The output stream (to allow chaining).
     */
    friend std::ostream& operator<<(std::ostream& os, const Mat2& m) {
        os << "Mat2(\n";
        for (int row = 0; row < 2; ++row) {
            os << "  ";  // Indentation for clean display
            for (int col = 0; col < 2; ++col) {
                os << "[";  // Start of the element
                os << std::setw(10) << std::setprecision(4) << std::fixed << m.v[col * 2 + row];  // Print element with formatting
                os << "]";  // End of the element
                if (col < 1) os << ", ";  // Space between columns
            }
            os << '\n';  // New line after each row
        }
        os << ")";
        return os;
    }
};

/* Matrix 2x2 Algorithms Implementation */

/**
 * @brief Calculates the determinant of the matrix
 * @return Determinant of the matrix
 */
inline constexpr float determinant(const Mat2& m) {
    return m[0] * m[3] - m[1] * m[2];
}

/**
 * @brief Calculates the trace of the matrix
 * @return Trace of the matrix
 */
inline constexpr float trace(const Mat2& m) {
    return m[0] + m[3];
}

/**
 * @brief Transposes the matrix
 * @return Transposed matrix
 */
inline constexpr Mat2 transpose(const Mat2& m) {
    return {
        m[0], m[1],
        m[2], m[3]
    };
}

/**
 * @brief Inverts the matrix if it is invertible
 * @return Inverted matrix
 */
inline constexpr Mat2 invert(const Mat2& m) {
    float det = determinant(m);
    if (det == 0.0f) {
        return Mat2::identity();
    }
    float invDet = 1.0f / det;
    return Mat2(m[3] * invDet, -m[1] * invDet, -m[2] * invDet, m[0] * invDet);
}

} // namespace bpm

#endif // BPM_MAT2_HPP
