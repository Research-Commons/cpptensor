#pragma once

#include <cmath>
#include <limits>

namespace cpptensor::detail {

inline bool pow_exponent_is_integer(float exponent) {
    if (!std::isfinite(exponent)) {
        return false;
    }

    float integral_part = 0.0f;
    return std::modf(exponent, &integral_part) == 0.0f;
}

inline float real_pow(float base, float exponent) {
    if (base < 0.0f && !pow_exponent_is_integer(exponent)) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    return std::pow(base, exponent);
}

} // namespace cpptensor::detail
