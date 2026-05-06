#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise less than comparison
     * 
     * Returns a tensor where each element is true if the corresponding
     * element of a is less than b , false otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Bool tensor (dtype=bool)
     */
    Tensor lt(const Tensor& a, const Tensor& b);
    Tensor lt(const Tensor& a, float scalar);
    Tensor lt(float scalar, const Tensor& b);
}
