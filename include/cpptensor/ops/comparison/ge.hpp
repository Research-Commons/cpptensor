#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise greater than or equal comparison
     * 
     * Returns a tensor where each element is 1.0f if the corresponding
     * element of a is greater than or equal to b, 0.0f otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Tensor with boolean values (1.0f = true, 0.0f = false)
     */
    Tensor ge(const Tensor& a, const Tensor& b);
    Tensor ge(const Tensor& a, float scalar);
    Tensor ge(float scalar, const Tensor& b);
}
