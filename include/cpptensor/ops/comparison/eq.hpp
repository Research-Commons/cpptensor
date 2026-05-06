#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise equality comparison
     * 
     * Returns a tensor where each element is true if the corresponding
     * elements of a and b are equal , false otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Bool tensor (dtype=bool)
     */
    Tensor eq(const Tensor& a, const Tensor& b);
    Tensor eq(const Tensor& a, float scalar);
    Tensor eq(float scalar, const Tensor& b);
}
