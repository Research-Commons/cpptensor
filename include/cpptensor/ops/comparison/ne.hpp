#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise inequality comparison
     * 
     * Returns a tensor where each element is true if the corresponding
     * elements of a and b are not equal , false otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Bool tensor (dtype=bool)
     */
    Tensor ne(const Tensor& a, const Tensor& b);
    Tensor ne(const Tensor& a, float scalar);
    Tensor ne(float scalar, const Tensor& b);
}
