#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    class AVX2 {
        public:
            static void add_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
            static void mul_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
            static void sub_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
            static void div_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
            static void pow_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
            static void exp_f32_avx2(const Tensor& A, Tensor& Out);
            static void log_f32_avx2(const Tensor& A, Tensor& Out);
            static void abs_f32_avx2(const Tensor& A, Tensor& Out);
            static void sqrt_f32_avx2(const Tensor& A, Tensor& Out);
            static void sin_f32_avx2(const Tensor& A, Tensor& Out);
            static void cos_f32_avx2(const Tensor& A, Tensor& Out);
            static void tan_f32_avx2(const Tensor& A, Tensor& Out);
            static void sigmoid_f32_avx2(const Tensor& A, Tensor& Out);
            static void relu_f32_avx2(const Tensor& A, Tensor& Out);
            static void gemm_f32_avx2(const Tensor& A, const Tensor& B, Tensor& C);
            static void dot_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    };
} // namespace cppgrad



