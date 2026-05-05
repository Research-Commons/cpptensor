#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise less than comparison
     * 
     * Returns a tensor where each element is 1.0f if the corresponding
     * element of a is less than b, 0.0f otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Tensor with boolean values (1.0f = true, 0.0f = false)
     */
    Tensor lt(const Tensor& a, const Tensor& b);
    Tensor lt(const Tensor& a, float scalar);
    Tensor lt(float scalar, const Tensor& b);
}
