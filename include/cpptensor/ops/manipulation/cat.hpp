#pragma once
#include "cpptensor/tensor/tensor.hpp"
#include <vector>

namespace cpptensor {
    /**
     * @brief Concatenate tensors along a dimension
     *
     * Concatenates the given sequence of tensors along an existing dimension.
     * All tensors must have the same shape except in the concatenating dimension.
     * All tensors must have the same number of dimensions.
     *
     * @param tensors Vector of tensors to concatenate (must be non-empty)
     * @param dim Dimension along which to concatenate (supports negative indexing)
     * @return New tensor with concatenated data
     *
     * @throws std::runtime_error if tensors is empty
     * @throws std::runtime_error if tensors have incompatible shapes
     * @throws std::runtime_error if dim is out of range
     *
     * @example
     * ```cpp
     * Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
     * Tensor B({2, 3}, {7, 8, 9, 10, 11, 12});
     *
     * // Concatenate along dim 0: result shape [4, 3]
     * Tensor C = cat({A, B}, 0);
     * // Result: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
     *
     * // Concatenate along dim 1: result shape [2, 6]
     * Tensor D = cat({A, B}, 1);
     * // Result: [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
     *
     * // Negative indexing (dim=-1 is last dimension)
     * Tensor E = cat({A, B}, -1);  // Same as dim=1 for 2D tensors
     *
     * // Concatenate multiple tensors
     * Tensor C_tensor({2, 3}, {13, 14, 15, 16, 17, 18});
     * Tensor F = cat({A, B, C_tensor}, 0);  // Shape: [6, 3]
     * ```
     */
    Tensor cat(const std::vector<Tensor>& tensors, int dim);

} // namespace cpptensor
