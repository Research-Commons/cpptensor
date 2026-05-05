#pragma once
#include "cpptensor/tensor/tensor.hpp"
#include <vector>

namespace cpptensor {
    /**
     * @brief Stack tensors along a new dimension
     *
     * Creates a new dimension and stacks tensors along it. Unlike cat(), which
     * concatenates along an existing dimension, stack() creates a new dimension
     * at the specified position and stacks the tensors there.
     *
     * All input tensors must have exactly the same shape and live on the same
     * device.
     *
     * The output tensor preserves the common input device. Non-contiguous
     * inputs are stacked by their logical values; view operands may be
     * materialized internally before inserting the new dimension.
     *
     * @param tensors Vector of tensors to stack (must be non-empty)
     * @param dim Position to insert new dimension (supports negative indexing)
     *            Valid range: [-ndim-1, ndim] where ndim is the number of dimensions
     * @return New tensor with stacked data
     *
     * @throws std::runtime_error if tensors is empty
     * @throws std::runtime_error if tensors have different shapes
     * @throws std::runtime_error if tensors are on different devices
     * @throws std::runtime_error if dim is out of range
     *
     * @example
     * ```cpp
     * Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
     * Tensor B({2, 3}, {7, 8, 9, 10, 11, 12});
     *
     * // Stack along new dim 0 (prepend): result shape [2, 2, 3]
     * Tensor C = stack({A, B}, 0);
     *
     * // Stack along new dim 1 (insert in middle): result shape [2, 2, 3]
     * Tensor D = stack({A, B}, 1);
     *
     * // Stack along new dim 2 (append at end): result shape [2, 3, 2]
     * Tensor E = stack({A, B}, 2);
     *
     * // Negative indexing: -1 means after last dimension
     * Tensor F = stack({A, B}, -1);  // Same as dim=2 for 2D tensors
     *
     * // Stack multiple tensors
     * Tensor C_tensor({2, 3}, {13, 14, 15, 16, 17, 18});
     * Tensor G = stack({A, B, C_tensor}, 0);  // Shape: [3, 2, 3]
     * ```
     *
     * @note Difference from cat():
     *       - cat({A, B}, 0) with shape [2,3] → [4,3] (concat existing dim)
     *       - stack({A, B}, 0) with shape [2,3] → [2,2,3] (new dim)
     */
    Tensor stack(const std::vector<Tensor>& tensors, int dim);

} // namespace cpptensor
