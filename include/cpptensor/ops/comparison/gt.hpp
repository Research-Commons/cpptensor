#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise greater than comparison
     * 
     * Returns a tensor where each element is true if the corresponding
     * element of a is greater than b , false otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Bool tensor (dtype=bool)
     */
    Tensor gt(const Tensor& a, const Tensor& b);
    Tensor gt(const Tensor& a, float scalar);
    Tensor gt(float scalar, const Tensor& b);
}
