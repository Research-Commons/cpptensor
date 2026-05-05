#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    /**
     * @brief Compute minimum values along a dimension
     *
     * Returns the minimum value(s) in the input tensor. Can operate on the entire
     * tensor or along a specific dimension.
     *
     * @param input Input tensor
     * @param dim Dimension along which to compute min (-1 for all elements)
     * @param keepdim If true, output has same number of dimensions as input
     * @return Tensor containing minimum values
     *
     * @example
     * ```cpp
     * Tensor A = Tensor::randn({3, 4});
     * Tensor min_all = min(A);              // Min of all elements, shape: ()
     * Tensor min_dim0 = min(A, 0);          // Min along dim 0, shape: (4,)
     * Tensor min_dim0_keep = min(A, 0, true); // Min along dim 0, shape: (1, 4)
     * ```
     */
    Tensor min(const Tensor& input, int dim = -1, bool keepdim = false);

} // namespace cpptensor
