#pragma once
#include "cpptensor/tensor/tensor.hpp"
#include <optional>

namespace cpptensor {
    /**
     * @brief Sum of tensor elements along a dimension
     *
     * Reduces the input tensor by summing along the specified dimension.
     * If no dimension is specified, sums all elements and returns a scalar.
     *
     * @param A Input tensor
     * @param dim Dimension to reduce (nullopt = reduce all dimensions)
     * @param keepdim Keep reduced dimension as size 1 if true
     * @return Tensor with reduced dimension(s)
     *
     * @throws std::runtime_error if dim is out of range
     *
     * @example
     * ```cpp
     * Tensor A({2, 3, 4}, values);
     *
     * // Sum all elements -> scalar tensor with shape [1]
     * auto total = sum(A);
     *
     * // Sum along dimension 1 -> shape [2, 4]
     * auto B = sum(A, 1);
     *
     * // Sum along dimension 0, keep dimension -> shape [1, 3, 4]
     * auto C = sum(A, 0, true);
     * ```
     */
    Tensor sum(const Tensor& A,
               std::optional<int> dim = std::nullopt,
               bool keepdim = false);
}
