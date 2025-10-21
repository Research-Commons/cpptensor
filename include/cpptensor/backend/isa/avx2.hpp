#pragma once
#include <vector>
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    void add_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    void mul_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    void sub_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    void div_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    void pow_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out);
    void exp_f32_avx2(const Tensor& A, Tensor& Out);
    void log_f32_avx2(const Tensor& A, Tensor& Out);
    void abs_f32_avx2(const Tensor& A, Tensor& Out);
    void sqrt_f32_avx2(const Tensor& A, Tensor& Out);
    void sin_f32_avx2(const Tensor& A, Tensor& Out);
    void cos_f32_avx2(const Tensor& A, Tensor& Out);
    void tan_f32_avx2(const Tensor& A, Tensor& Out);
    void sigmoid_f32_avx2(const Tensor& A, Tensor& Out);
    void relu_f32_avx2(const Tensor& A, Tensor& Out);
} // namespace cppgrad



