#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor {

inline bool is_floating_dtype(DType dtype) {
    return dtype == DType::FLOAT32 || dtype == DType::FLOAT64;
}

inline bool is_integer_dtype(DType dtype) {
    return dtype == DType::INT32;
}

inline bool is_bool_dtype(DType dtype) {
    return dtype == DType::BOOL;
}

inline int dtype_promotion_rank(DType dtype) {
    switch (dtype) {
        case DType::BOOL:
            return 0;
        case DType::INT32:
            return 1;
        case DType::FLOAT32:
            return 2;
        case DType::FLOAT64:
            return 3;
    }
    return -1;
}

inline DType promote_dtype(DType lhs, DType rhs) {
    const int rank = std::max(dtype_promotion_rank(lhs), dtype_promotion_rank(rhs));
    switch (rank) {
        case 0:
            return DType::BOOL;
        case 1:
            return DType::INT32;
        case 2:
            return DType::FLOAT32;
        case 3:
            return DType::FLOAT64;
        default:
            return DType::FLOAT32;
    }
}

template <typename T>
struct CppTypeToDType;

template <>
struct CppTypeToDType<bool> {
    static constexpr DType value = DType::BOOL;
};

template <>
struct CppTypeToDType<std::uint8_t> {
    static constexpr DType value = DType::BOOL;
};

template <>
struct CppTypeToDType<std::int32_t> {
    static constexpr DType value = DType::INT32;
};

template <>
struct CppTypeToDType<float> {
    static constexpr DType value = DType::FLOAT32;
};

template <>
struct CppTypeToDType<double> {
    static constexpr DType value = DType::FLOAT64;
};

template <typename T>
inline constexpr DType cpp_type_to_dtype_v = CppTypeToDType<std::remove_cv_t<T>>::value;

} // namespace cpptensor
