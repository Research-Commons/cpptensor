#pragma once

#include <vector>
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    // tensordot contracts specified axes of A and B, similar to NumPy's tensordot
    // Overload 1: axes = k (int)
    // - Contracts last k axes of A with first k axes of B
    Tensor tensordot(const Tensor& A, const Tensor& B, int axes);

    // Overload 2: explicit axes lists
    // - Contracts A axes in axesA with B axes in axesB (must be same length)
    Tensor tensordot(const Tensor& A, const Tensor& B, const std::vector<int>& axesA, const std::vector<int>& axesB);

} // namespace cpptensor
