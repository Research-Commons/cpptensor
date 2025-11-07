#pragma once
#include <vector>
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    class AVX512 {
    public:
        static void add_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);
        static void mul_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);
        static void gemm_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);
        static void dot_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out);

        // Reduction operations
        static void sum_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim);
        static void mean_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim);
        static void max_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim);
        static void min_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim);
    };

} // namespace cppgrad