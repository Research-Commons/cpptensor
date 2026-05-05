#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    Tensor matmul(const Tensor& A, const Tensor& B);
    Tensor gemm(const Tensor& A, const Tensor& B);
    Tensor gemv(const Tensor& A, const Tensor& x);

} // namespace cpptensor