#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise less than or equal comparison
     * 
     * Returns a tensor where each element is 1.0f if the corresponding
     * element of a is less than or equal to b, 0.0f otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Tensor with boolean values (1.0f = true, 0.0f = false)
     */
    Tensor le(const Tensor& a, const Tensor& b);
    Tensor le(const Tensor& a, float scalar);
    Tensor le(float scalar, const Tensor& b);
}
