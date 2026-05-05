#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise inequality comparison
     * 
     * Returns a tensor where each element is 1.0f if the corresponding
     * elements of a and b are not equal, 0.0f otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Tensor with boolean values (1.0f = true, 0.0f = false)
     */
    Tensor ne(const Tensor& a, const Tensor& b);
    Tensor ne(const Tensor& a, float scalar);
    Tensor ne(float scalar, const Tensor& b);
}
