#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    /**
     * @brief Compute maximum values along a dimension
     *
     * Returns the maximum value(s) in the input tensor. Can operate on the entire
     * tensor or along a specific dimension.
     *
     * @param input Input tensor
     * @param dim Dimension along which to compute max (-1 for all elements)
     * @param keepdim If true, output has same number of dimensions as input
     * @return Tensor containing maximum values
     *
     * @example
     * ```cpp
     * Tensor A = Tensor::randn({3, 4});
     * Tensor max_all = max(A);              // Max of all elements, shape: ()
     * Tensor max_dim0 = max(A, 0);          // Max along dim 0, shape: (4,)
     * Tensor max_dim0_keep = max(A, 0, true); // Max along dim 0, shape: (1, 4)
     * ```
     */
    Tensor max(const Tensor& input, int dim = -1, bool keepdim = false);

} // namespace cpptensor
