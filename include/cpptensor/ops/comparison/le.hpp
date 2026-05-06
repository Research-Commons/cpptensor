#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Element-wise less than or equal comparison
     * 
     * Returns a tensor where each element is true if the corresponding
     * element of a is less than or equal to b , false otherwise.
     * 
     * @param a First tensor
     * @param b Second tensor
     * @return Bool tensor (dtype=bool)
     */
    Tensor le(const Tensor& a, const Tensor& b);
    Tensor le(const Tensor& a, float scalar);
    Tensor le(float scalar, const Tensor& b);
}
