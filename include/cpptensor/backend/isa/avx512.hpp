#pragma once
#include <vector>
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    class AVX512 {
        public:
            static void add_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);
            static void mul_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);
            static void gemm_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);
    };

} // namespace cppgrad