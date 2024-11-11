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

#ifndef BPM_EASE_HPP
#define BPM_EASE_HPP

#include "./math.hpp"

namespace bpm {

/**
 * @brief Applies the sine easing-in function to a parameter.
 * 
 * This function applies the sine easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a gentle starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_sine_in(T t) {
    return std::sin(PI / 2 * t);
}

/**
 * @brief Applies the sine easing-out function to a parameter.
 * 
 * This function applies the sine easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a gentle ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_sine_out(T t) {
    return 1 + std::sin(PI / 2 * (--t));
}

/**
 * @brief Applies the sine easing-in-out function to a parameter.
 * 
 * This function applies the sine easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_sine_in_out(T t) {
    return 0.5 * (1 + std::sin(PI * (t - 0.5)));
}

/**
 * @brief Applies the quadratic easing-in function to a parameter.
 * 
 * This function applies the quadratic easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quad_in(T t) {
    return t * t;
}

/**
 * @brief Applies the quadratic easing-out function to a parameter.
 * 
 * This function applies the quadratic easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quad_out(T t) {
    return t * (2 - t);
}

/**
 * @brief Applies the quadratic easing-in-out function to a parameter.
 * 
 * This function applies the quadratic easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quad_in_out(T t) {
    return t < 0.5 ? 2 * t * t : t * (4 - 2 * t) - 1;
}

/**
 * @brief Applies the cubic easing-in function to a parameter.
 * 
 * This function applies the cubic easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_cubic_in(T t) {
    return t * t * t;
}

/**
 * @brief Applies the cubic easing-out function to a parameter.
 * 
 * This function applies the cubic easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_cubic_out(T t) {
    return 1 + (--t) * t * t;
}

/**
 * @brief Applies the cubic easing-in-out function to a parameter.
 * 
 * This function applies the cubic easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_cubic_in_out(T t) {
    return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
}

/**
 * @brief Applies the quartic easing-in function to a parameter.
 * 
 * This function applies the quartic easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quart_in(T t) {
    t *= t;
    return t * t;
}

/**
 * @brief Applies the quartic easing-out function to a parameter.
 * 
 * This function applies the quartic easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quart_out(T t) {
    t = (--t) * t;
    return 1 - t * t;
}

/**
 * @brief Applies the quartic easing-in-out function to a parameter.
 * 
 * This function applies the quartic easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quart_in_out(T t) {
    if (t < 0.5) {
        t *= t;
        return 8 * t * t;
    } else {
        t = (--t) * t;
        return 1 - 8 * t * t;
    }
}

/**
 * @brief Applies the quintic easing-in function to a parameter.
 * 
 * This function applies the quintic easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quint_in(T t) {
    T t2 = t * t;
    return t * t2 * t2;
}

/**
 * @brief Applies the quintic easing-out function to a parameter.
 * 
 * This function applies the quintic easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quint_out(T t) {
    T t2 = (--t) * t;
    return 1 + t * t2 * t2;
}

/**
 * @brief Applies the quintic easing-in-out function to a parameter.
 * 
 * This function applies the quintic easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_quint_in_out(T t) {
    T t2;
    if (t < 0.5) {
        t2 = t * t;
        return 16 * t * t2 * t2;
    }
    t2 = (--t) * t;
    return 1 + 16 * t * t2 * t2;
}

/**
 * @brief Applies the exponential easing-in function to a parameter.
 * 
 * This function applies the exponential easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_expo_in(T t) {
    return (std::pow(2, 8 * t) - 1) / 255;
}

/**
 * @brief Applies the exponential easing-out function to a parameter.
 * 
 * This function applies the exponential easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_expo_out(T t) {
    return 1 - std::pow(2, -8 * t);
}

/**
 * @brief Applies the exponential easing-in-out function to a parameter.
 * 
 * This function applies the exponential easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_expo_in_out(T t) {
    if (t < 0.5) {
        return (std::pow(2, 16 * t) - 1) / 510;
    }
    return 1 - 0.5 * std::pow(2, -16 * (t - 0.5));
}

/**
 * @brief Applies the circular easing-in function to a parameter.
 * 
 * This function applies the circular easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_circ_in(T t) {
    return 1 - std::sqrt(1 - t);
}

/**
 * @brief Applies the circular easing-out function to a parameter.
 * 
 * This function applies the circular easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_circ_out(T t) {
    return std::sqrt(t);
}

/**
 * @brief Applies the circular easing-in-out function to a parameter.
 * 
 * This function applies the circular easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_circ_in_out(T t) {
    if (t < 0.5) {
        return (1 - std::sqrt(1 - 2 * t)) * 0.5;
    }
    return (1 + std::sqrt(2 * t - 1)) * 0.5;
}

/**
 * @brief Applies the back easing-in function to a parameter.
 * 
 * This function applies the back easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration and overshoot.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_back_in(T t) {
    return t * t * (2.70158 * t - 1.70158);
}

/**
 * @brief Applies the back easing-out function to a parameter.
 * 
 * This function applies the back easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with an ending deceleration and overshoot.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_back_out(T t) {
    return 1 + (--t) * t * (2.70158 * t + 1.70158);
}

/**
 * @brief Applies the back easing-in-out function to a parameter.
 * 
 * This function applies the back easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration
 * and overshoot.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline constexpr T ease_back_in_out(T t) {
    if (t < 0.5) {
        return t * t * (7 * t - 2.5) * 2;
    }
    return 1 + (--t) * t * 2 * (7 * t + 2.5);
}

/**
 * @brief Applies the elastic easing-in function to a parameter.
 * 
 * This function applies the elastic easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a starting acceleration and overshoot followed by
 * a sinusoidal oscillation.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_elastic_in(T t) {
    T t2 = t * t;
    return t2 * t2 * std::sin(t * PI * 4.5);
}

/**
 * @brief Applies the elastic easing-out function to a parameter.
 * 
 * This function applies the elastic easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a deceleration followed by a sinusoidal oscillation
 * and ending deceleration.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_elastic_out(T t) {
    T t2 = (t - 1) * (t - 1);
    return 1 - t2 * t2 * std::cos(t * PI * 4.5);
}

/**
 * @brief Applies the elastic easing-in-out function to a parameter.
 * 
 * This function applies the elastic easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with both starting and ending acceleration/deceleration
 * followed by a sinusoidal oscillation.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_elastic_in_out(T t) {
    T t2;
    if (t < 0.45) {
        t2 = t * t;
        return 8 * t2 * t2 * std::sin(t * PI * 9);
    } else if (t < 0.55) {
        return 0.5 + 0.75 * std::sin(t * PI * 4);
    }
    t2 = (t - 1) * (t - 1);
    return 1 - 8 * t2 * t2 * std::sin(t * PI * 9);
}

/**
 * @brief Applies the bounce easing-in function to a parameter.
 * 
 * This function applies the bounce easing-in function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a bouncing effect at the beginning.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_bounce_in(T t) {
    return std::pow(2, 6 * (t - 1)) * std::abs(std::sin(t * PI * 3.5));
}

/**
 * @brief Applies the bounce easing-out function to a parameter.
 * 
 * This function applies the bounce easing-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a bouncing effect at the end.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_bounce_out(T t) {
    return 1 - std::pow(2, -6 * t) * std::abs(std::cos(t * PI * 3.5));
}

/**
 * @brief Applies the bounce easing-in-out function to a parameter.
 * 
 * This function applies the bounce easing-in-out function to a parameter `t`.
 * The function modifies the rate of change of `t` over time to create
 * smooth transitions with a bouncing effect both at the beginning and at the end.
 * 
 * @tparam T The type of the parameter (should be a floating-point type).
 * @param t The parameter to apply the easing function to.
 * @return The eased parameter value.
 */
template <typename T>
inline T ease_bounce_in_out(T t) {
    if (t < 0.5) {
        return 8 * std::pow(2, 8 * (t - 1)) * std::abs(std::sin(t * PI * 7));
    }
    return 1 - 8 * std::pow(2, -8 * t) * std::abs(std::sin(t * PI * 7));
}

} // namespace bpm

#endif // BPM_EASE_HPP