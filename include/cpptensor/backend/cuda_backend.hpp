#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    class CUDA {
    public:

        static void addKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void mulKernel(const Tensor& A, const Tensor& B, Tensor& out);
    };

} // namespace cppgrad