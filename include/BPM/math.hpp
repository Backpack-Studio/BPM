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

#ifndef BPM_MATH_HPP
#define BPM_MATH_HPP

#include <type_traits>
#include <cstdint>
#include <limits>
#include <cmath>

namespace bpm {

constexpr double RCP_255    = 0.00392156862745098039;   ///< The reciprocal of 255, useful for converting 8-bit color values to the range [0, 1].
constexpr double SQRT_2     = 1.41421356237309504880;   ///< The square root of 2.
constexpr double SQRT_3     = 1.73205080756887729352;   ///< The square root of 3.
constexpr double PHI        = 1.61803398874989484820;   ///< The golden ratio, approximately (1 + sqrt(5)) / 2.
constexpr double PI         = 3.14159265358979323846;   ///< The mathematical constant pi.
constexpr double TAU        = 2.0 * PI;                 ///< Tau is equal to 2 times pi, representing one full turn in radians.
constexpr double DEG_TO_RAD = PI / 180.0;               ///< Conversion factor from degrees to radians.
constexpr double RAD_TO_DEG = 180.0 / PI;               ///< Conversion factor from radians to degrees.
constexpr double E          = 2.71828182845904523536;   ///< The base of the natural logarithm (e).
constexpr double LOG2_E     = 1.44269504088896340736;   ///< The logarithm of e to the base 2 (log2(e)).
constexpr double LOG10_E    = 0.43429448190325182765;   ///< The logarithm of e to the base 10 (log10(e)).
constexpr double INV_SQRT_2 = 0.70710678118654752440;   ///< The inverse square root of 2, often used in optimizations (1 / sqrt(2)).
constexpr double INV_SQRT_3 = 0.57735026918962576451;   ///< The inverse square root of 3, often used in optimizations (1 / sqrt(3)).
constexpr double GOLDEN_ANGLE = 137.50776405003785508499; ///< The golden angle in radians, approximately 137.5°, important in geometry and art.

/**
 * @brief Converts radians to degrees.
 * 
 * This function takes an angle in radians and converts it to degrees.
 * 
 * @tparam T The type of the angle (should be a floating-point type).
 * @param radians The angle in radians to be converted.
 * @return The equivalent angle in degrees.
 */
template <typename T>
inline constexpr T degrees(T radians) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return radians * RAD_TO_DEG; // RAD_TO_DEG is assumed to be a constant representing the conversion factor from radians to degrees
}

/**
 * @brief Converts degrees to radians.
 * 
 * This function takes an angle in degrees and converts it to radians.
 * 
 * @tparam T The type of the angle (should be a floating-point type).
 * @param degrees The angle in degrees to be converted.
 * @return The equivalent angle in radians.
 */
template <typename T>
inline constexpr T radians(T degrees) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return degrees * DEG_TO_RAD; // DEG_TO_RAD is assumed to be a constant representing the conversion factor from degrees to radians
}

/**
 * @brief Computes the fractional part of a floating-point number.
 * 
 * This function calculates the fractional part of a floating-point number.
 * For example, Fract(3.14) returns 0.14.
 * 
 * @tparam T The type of the value (should be a floating-point type).
 * @param value The value whose fractional part is to be computed.
 * @return The fractional part of the input value.
 */
template <typename T>
inline constexpr T fract(T value) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return value - static_cast<int32_t>(value); // Subtracting the integer part to get the fractional part
}

/**
 * @brief Computes the sign of a value.
 * 
 * This function determines the sign of a numeric value.
 * It returns -1 if the value is negative, 0 if the value is zero, and 1 if the value is positive.
 * 
 * @tparam T The type of the value.
 * @param value The value whose sign is to be computed.
 * @return -1 if value is negative, 0 if value is zero, and 1 if value is positive.
 */
template <typename T, typename U>
inline constexpr T sign(U value) {
    static_assert(std::is_signed<T>::value, "Type T must be a signed integer");
    return (static_cast<U>(0) < value) - (value < static_cast<U>(0));
}

/**
 * @brief Computes the factorial of an integer.
 * 
 * This function calculates the factorial of a non-negative integer.
 * The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n.
 * For example, factorial(5) returns 5*4*3*2*1 = 120.
 * 
 * @param n The non-negative integer for which the factorial is to be computed.
 * @return The factorial of the input integer.
 */
template <typename T>
inline constexpr T factorial(T n) {
    static_assert(std::is_unsigned<T>::value, "Type T must be an unsigned integer");
    return (n == 0 || n == 1) ? 1 : n * factorial(n - 1);
}

/**
 * @brief Finds the next power of two greater than or equal to the given value.
 * 
 * This function calculates the next power of two that is greater than or equal to the given value.
 * If the given value is already a power of two, it returns the value itself doubled.
 * 
 * @param value The input value.
 * @return The next power of two greater than or equal to the input value.
 * 
 * @tparam T The type of the input value. It must be an unsigned integer type.
 * 
 * @note This function uses bitwise operations to efficiently find the next power of two.
 *       It works for unsigned integer types of various sizes (e.g., uint8_t, uint32_t, uint64_t).
 *       The function also handles edge cases like `value == 0`, returning `1` in this case.
 */
template <typename T>
inline constexpr T next_po2(T value) {
    static_assert(std::is_unsigned<T>::value, "Type T must be an unsigned integer");

    if (value == 0) return 1;

    // If the value is already a power of two, double it
    if ((value & (value - 1)) == 0) return value << 1;

    // Perform bitwise operations to propagate the highest bit
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    if constexpr (sizeof(T) > 1) value |= value >> 8;   // Necessary for types larger than 8 bits
    if constexpr (sizeof(T) > 2) value |= value >> 16;  // Necessary for types larger than 16 bits
    if constexpr (sizeof(T) > 4) value |= value >> 32;  // Necessary for types larger than 32 bits
    value++;

    return value;
}

/**
 * @brief Finds the next power of two greater than or equal to the given value using logarithms.
 * 
 * This function uses the `log2` function to calculate the next power of two greater than or equal to
 * the given value. It works by calculating the ceiling of `log2(value)` and then returning 2 raised to that power.
 * 
 * @param value The input value.
 * @return The next power of two greater than or equal to the input value.
 * 
 * @tparam T The type of the input value. The function works for all numeric types.
 * 
 * @note This function is less efficient than the bitwise approach, especially for larger types,
 *       but it provides a simple and reliable way to calculate the next power of two using standard math functions.
 */
template <typename T>
inline T next_po2_log(T value) {
    return static_cast<T>(std::pow(2, std::ceil(std::log2(value))));
}

/**
 * @brief Finds the previous power of two less than or equal to the given value.
 * 
 * This function calculates the previous power of two that is less than or equal to the given value.
 * If the given value is already a power of two, it returns the value itself divided by two.
 * 
 * @param value The input value.
 * @return The previous power of two less than or equal to the input value.
 * 
 * @tparam T The type of the input value. It must be an unsigned integer type.
 * 
 * @note This function uses bitwise operations to efficiently find the previous power of two.
 *       It works for unsigned integer types of various sizes (e.g., uint8_t, uint32_t, uint64_t).
 *       The function also handles edge cases like `value == 0`, returning `0` in this case.
 */
template <typename T>
inline constexpr T previous_po2(T value) {
    static_assert(std::is_unsigned<T>::value, "Type T must be an unsigned integer");

    if (value == 0) return 0;

    // If the value is already a power of two, divide it by 2
    if ((value & (value - 1)) == 0) return value >> 1;

    // Perform bitwise operations to propagate the highest bit
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    if constexpr (sizeof(T) > 1) value |= value >> 8;  // Necessary for types larger than 8 bits
    if constexpr (sizeof(T) > 2) value |= value >> 16; // Necessary for types larger than 16 bits
    if constexpr (sizeof(T) > 4) value |= value >> 32; // Necessary for types larger than 32 bits

    // Calculate the previous power of two
    return value - (value >> 1);
}

/**
 * @brief Finds the previous power of two less than or equal to the given value using logarithms.
 * 
 * This function uses the `log2` function to calculate the previous power of two less than or equal to
 * the given value. It works by calculating the floor of `log2(value)` and then returning 2 raised to that power.
 * 
 * @param value The input value.
 * @return The previous power of two less than or equal to the input value.
 * 
 * @tparam T The type of the input value. The function works for all numeric types.
 * 
 * @note This function is less efficient than the bitwise approach, especially for larger types,
 *       but it provides a simple and reliable way to calculate the previous power of two using standard math functions.
 */
template <typename T>
inline T previous_po2_log(T value) {
    return static_cast<T>(std::pow(2, std::floor(std::log2(value))));
}

/**
 * @brief Finds the closest power of two to the given value.
 * 
 * This function calculates the closest power of two to the given value.
 * If the given value is already a power of two, it returns the value itself.
 * The function calculates both the next and previous powers of two and returns the closest one.
 * 
 * @param value The input value.
 * @return The closest power of two to the input value.
 * 
 * @tparam T The type of the input value. It must be an unsigned integer type.
 * 
 * @note This function uses bitwise operations to efficiently find the next and previous powers of two.
 *       It works for unsigned integer types of various sizes (e.g., uint8_t, uint32_t, uint64_t).
 *       The function also handles edge cases like `value == 0`, returning `1` in this case.
 */
template <typename T>
inline constexpr T nearest_po2(T value) {
    static_assert(std::is_unsigned<T>::value, "Type T must be an unsigned integer");

    if (value == 0) return 1;

    // Find the next power of two greater than or equal to the value
    T next = value;
    next--;
    next |= next >> 1;
    next |= next >> 2;
    next |= next >> 4;
    if constexpr (sizeof(T) > 1) next |= next >> 8;  // Necessary for types larger than 8 bits
    if constexpr (sizeof(T) > 2) next |= next >> 16; // Necessary for types larger than 16 bits
    if constexpr (sizeof(T) > 4) next |= next >> 32; // Necessary for types larger than 32 bits
    next++;

    // Find the previous power of two less than or equal to the value
    T prev = next >> 1;

    // Compare the difference to find the closest power of two
    return (next - value) > (value - prev) ? prev : next;
}

/**
 * @brief Finds the closest power of two to the given value using logarithms.
 * 
 * This function calculates the closest power of two to the given value using logarithms.
 * It calculates both the next and previous powers of two using `log2` and returns the closest one.
 * 
 * @param value The input value.
 * @return The closest power of two to the input value.
 * 
 * @tparam T The type of the input value. The function works for all numeric types.
 * 
 * @note This function is less efficient than the bitwise approach, especially for larger types,
 *       but it provides a simple and reliable way to calculate the closest power of two using standard math functions.
 */
template <typename T>
inline T nearest_po2_log(T value) {
    if (value == 0) return 1;

    // Calculate the previous power of two using log2 and floor
    T lower = static_cast<T>(std::pow(2, std::floor(std::log2(value))));

    // Calculate the next power of two using log2 and ceil
    T upper = static_cast<T>(std::pow(2, std::ceil(std::log2(value))));

    // Return the closest power of two
    return (value - lower) <= (upper - value) ? lower : upper;
}

/**
 * @brief Computes the reciprocal of a scalar value.
 * 
 * This function returns the reciprocal (1 / value) of the provided scalar `value`.
 * It is a constexpr function, meaning the computation is evaluated at compile-time
 * if the input is known at compile-time.
 * 
 * @tparam T The return type for the reciprocal, typically a floating-point type.
 * @tparam U The type of the input value, which must be a floating-point type.
 * 
 * @param value The scalar value for which the reciprocal is calculated.
 * @return The reciprocal of `value`, with the result cast to type `T`.
 * 
 * @note This function requires that `U` is a floating-point type.
 */
template <typename T, typename U>
constexpr inline T reciprocal(U value) {
    static_assert(std::is_floating_point<U>::value, "Type U must be a floating-point");
    return static_cast<T>(1.0) / static_cast<T>(value);
}

/**
 * @brief Checks if two floating-point numbers are approximately equal within a tolerance.
 * 
 * This function compares two floating-point numbers `a` and `b` to determine if they are 
 * approximately equal, within a specified tolerance `epsilon`. The comparison considers 
 * values equal if their absolute difference is less than `epsilon`, which defaults to 
 * the smallest representable difference for the type `T`.
 * 
 * @tparam T The type of the numbers being compared, which must be a floating-point type.
 * 
 * @param a The first floating-point number.
 * @param b The second floating-point number.
 * @param epsilon The tolerance for comparison, defaulting to `std::numeric_limits<T>::epsilon()`.
 * 
 * @return `true` if the absolute difference between `a` and `b` is less than `epsilon`; `false` otherwise.
 * 
 * @note This function requires that `T` is a floating-point type.
 */
template <typename T>
inline bool approx(T a, T b, T epsilon = std::numeric_limits<T>::epsilon()) {
    static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point");
    return std::abs(a - b) < epsilon;
}

/**
 * @brief Move a value 'a' towards 'b' by a specified 'delta'.
 *
 * @tparam T The numeric type of the values.
 * @param a The starting value.
 * @param b The target value.
 * @param delta The distance to move towards 'b'.
 * @return The new value of 'a' moved towards 'b' by 'delta'.
 */
template <typename T>
inline T move_towards(T a, T b, T delta) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    T diff = b - a;
    if (std::abs(diff) <= delta) {
        return b;
    }
    return a + (delta * (diff > T(0) ? T(1) : T(-1)));
}

/**
 * @brief Performs linear interpolation between two values.
 * 
 * This function performs linear interpolation (lerp) between two values.
 * It returns the value that is linearly interpolated between start and end based on the interpolation parameter t.
 * 
 * @tparam T The type of the values.
 * @param start The starting value.
 * @param end The ending value.
 * @param t The interpolation parameter (should be in the range [0, 1]).
 * @return The linearly interpolated value between start and end based on t.
 */
template <typename T>
inline constexpr T lerp(T start, T end, T t) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return start + t * (end - start);
}

/**
 * @brief Applies a smoothstep interpolation function to the input value.
 * 
 * The smoothstep function produces a smooth interpolation curve that starts and ends
 * with a zero derivative (no abrupt changes), making it useful for smooth transitions
 * in animations and procedural shading. The function is mathematically defined as:
 * 
 *     smoothstep(t) = t * t * (3 - 2 * t)
 * 
 * where `t` is expected to be in the range [0, 1]. This formula creates an S-shaped curve
 * that transitions smoothly from 0 to 1 as `t` moves from 0 to 1. If `t` is outside
 * this range, the output will not be clamped and may exceed [0, 1].
 * 
 * @tparam T The type of the input, which must be a floating-point type (e.g., float, double).
 *           A static assertion ensures only floating-point types are allowed.
 * @param t The interpolation parameter, typically in the range [0, 1].
 * @return T The result of the smoothstep interpolation, which lies in [0, 1] for input `t` in [0, 1].
 * 
 * @note If `t` is outside the range [0, 1], the output may exceed [0, 1]. For strict clamping,
 *       apply additional logic to ensure `t` is within bounds before calling this function.
 */
template <typename T>
inline constexpr T smoothstep(T t) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return t * t * (static_cast<T>(3.0) - static_cast<T>(2.0) * t);
}

/**
 * @brief Applies a smootherstep interpolation function to the input value.
 * 
 * The smootherstep function generates an even smoother interpolation curve than smoothstep,
 * with zero first and second derivatives at the endpoints, making it particularly useful
 * for applications requiring very smooth transitions. This function is mathematically defined as:
 * 
 *     smootherstep(t) = t^3 * (t * (t * 6 - 15) + 10)
 * 
 * where `t` is typically in the range [0, 1]. This formula produces a steeper S-shaped curve
 * than smoothstep, providing a more gradual transition near the start and end, which makes it 
 * ideal for use in physical simulations and procedural noise functions.
 * 
 * @tparam T The type of the input, which must be a floating-point type (e.g., float, double).
 *           A static assertion ensures only floating-point types are allowed.
 * @param t The interpolation parameter, typically in the range [0, 1].
 * @return T The result of the smootherstep interpolation, which lies in [0, 1] for input `t` in [0, 1].
 * 
 * @note If `t` is outside the range [0, 1], the output may exceed [0, 1]. As with smoothstep,
 *       additional clamping may be applied as needed to enforce strict boundaries.
 */
template <typename T>
inline constexpr T smootherstep(T t) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return t * t * t * (t * (t * static_cast<T>(6) - static_cast<T>(15)) + static_cast<T>(10));
}

/**
 * @brief Normalizes a value within a specified range.
 * 
 * This function normalizes the value within the range defined by `start` and `end`.
 * 
 * @tparam T The type of the value and the range (should be a floating-point type).
 * @param value The value to normalize.
 * @param start The start of the range.
 * @param end The end of the range.
 * @return The normalized value within the range [0, 1].
 */
template <typename T>
inline T normalize(T value, T start, T end) {
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");
    return (value - start) / (end - start);
}

/**
 * @brief Remaps a value from one range to another linearly.
 * 
 * This function remaps a value `value` from the input range defined by `in_start` and `in_end`
 * to the output range defined by `out_start` and `out_end`. The remapping is done linearly.
 * 
 * @tparam T The type of the value and the ranges (should be a floating-point type).
 * @param value The value to remap.
 * @param in_start The start of the input range.
 * @param in_end The end of the input range.
 * @param out_start The start of the output range.
 * @param out_end The end of the output range.
 * @return The remapped value in the output range.
 */
template <typename T>
inline constexpr T remap(T value, T in_start, T in_end, T out_start, T out_end) {
    return (value - in_start) / (in_end - in_start) * (out_end - out_start) + out_start;
}

/**
 * @brief Wraps a value within a range.
 * 
 * This function wraps a value `value` within the range defined by `min` and `max`.
 * If `value` exceeds the range, it wraps around to the other end of the range.
 * 
 * @tparam T The type of the value and the range (should be a floating-point type).
 * @param value The value to wrap.
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return The wrapped value within the range [min, max].
 */
template <typename T>
inline T wrap(T value, T min, T max) {
    return value - (max - min) * std::floor((value - min) / (max - min));
}

/**
 * @brief Wraps an angle in radians to the range from -π to π.
 * 
 * This function wraps an angle `th` in radians to the range from -π to π.
 * If the angle exceeds this range, it wraps around accordingly.
 * 
 * @tparam T The type of the angle (should be a floating-point type).
 * @param th The angle to wrap.
 * @return The wrapped angle within the range [-π, π].
 */
template <typename T>
inline T wrap_rad(T th) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return std::atan2(std::sin(th), std::cos(th));
}

/**
 * @brief Computes the angular difference between two angles in radians.
 * 
 * This function computes the angular difference between two angles `current` and `target` given in radians.
 * It returns the difference angle, which may be positive or negative depending on the direction of rotation.
 * 
 * @tparam T The type of the angles (should be a floating-point type).
 * @param current The current angle in radians.
 * @param target The target angle in radians.
 * @return The angular difference between `current` and `target`.
 */
template <typename T>
inline T delta_rad(T current, T target) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    const T c0 = std::cos(current), s0 = std::sin(current);
    const T c1 = std::cos(target), s1 = std::sin(target);
    return std::atan2(c0 * s1 - c1 * s0, c0 * c1 + s1 * s0);
}

/**
 * @brief Performs linear interpolation (lerp) between two angles in radians.
 * 
 * This function performs linear interpolation (lerp) between two angles `start` and `end` given in radians,
 * using the parameter `t` to determine the interpolation factor.
 * 
 * @tparam T The type of the angles (should be a floating-point type).
 * @param start The starting angle in radians.
 * @param end The ending angle in radians.
 * @param t The interpolation factor between 0 and 1.
 * @return The interpolated angle between `start` and `end`.
 */
template <typename T>
inline T lerp_rad(T start, T end, T t) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    start = wrap_rad(start), end = wrap_rad(end);
    return wrap_rad(start + t * delta_rad(start, end));
}

/**
 * @brief Wraps an angle to the range [0, 360).
 * 
 * This function wraps an angle `angle` to the range [0, 360) degrees.
 * If the angle is negative, it wraps around to the positive range.
 * 
 * @tparam T The type of the angle.
 * @param angle The angle to wrap in degrees.
 * @return The wrapped angle within the range [0, 360).
 */
template <typename T>
inline T wrap_deg(T angle) {
    return std::fmod(std::abs(angle), static_cast<T>(360.0));
}

/**
 * @brief Computes the angular difference between two angles in degrees.
 * 
 * This function computes the angular difference between two angles `current` and `target`
 * given in degrees. It returns the difference angle in degrees, which may be positive or negative
 * depending on the direction of rotation.
 * 
 * @tparam T The type of the angles (should be a floating-point type).
 * @param current The current angle in degrees.
 * @param target The target angle in degrees.
 * @return The angular difference between `current` and `target` in degrees.
 */
template <typename T>
inline T delta_deg(T current, T target) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return delta_rad(current * DEG_TO_RAD, target * DEG_TO_RAD) * RAD_TO_DEG;
}

/**
 * @brief Performs linear interpolation (lerp) between two angles in degrees.
 * 
 * This function performs linear interpolation (lerp) between two angles `start` and `end`
 * given in degrees, using the parameter `t` to determine the interpolation factor.
 * 
 * @tparam T The type of the angles (should be a floating-point type).
 * @param start The starting angle in degrees.
 * @param end The ending angle in degrees.
 * @param t The interpolation factor between 0 and 1.
 * @return The interpolated angle between `start` and `end`.
 */
template <typename T>
inline T lerp_deg(T start, T end, T t) {
    static_assert(std::is_floating_point_v<T>, "Type T must be an floating-point");
    return std::fmod(start + t * delta_deg(start, end) + 360.0f, 360.0f);
}

} // namespace bpm

#endif // BPM_MATH_HPP
