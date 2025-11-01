#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    // Shape rules:
    // - A.shape() == [N], B.shape() == [N]
    // - Returns shape [] (scalar)
    Tensor dot(const Tensor& A, const Tensor& B);

} // namespace cpptensor
