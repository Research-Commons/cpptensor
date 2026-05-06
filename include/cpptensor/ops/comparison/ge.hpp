#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise greater than or equal comparison
     * 
     * Returns a tensor where each element is true if the corresponding
     * element of a is greater than or equal to b , false otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Bool tensor (dtype=bool)
     */
    Tensor ge(const Tensor& a, const Tensor& b);
    Tensor ge(const Tensor& a, float scalar);
    Tensor ge(float scalar, const Tensor& b);
}
