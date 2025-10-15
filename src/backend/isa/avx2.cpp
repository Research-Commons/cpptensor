#include <immintrin.h>
#include "cppgrad/backend/isa/avx2.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/enums/dispatcherEnum.h"

namespace cppgrad {

void add_f32_avx2(const cppgrad::Tensor& A,
                  const cppgrad::Tensor& B,
                  cppgrad::Tensor& Out) {
    // Basic sanity checks
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX2 add: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX2 add: shape mismatch");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 8; // 8 floats per __m256
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }

    // remainder
    for (; i < n; ++i) {
        o[i] = a[i] + b[i];
    }
}

void mul_f32_avx2(const cppgrad::Tensor& A,
                  const cppgrad::Tensor& B,
                  cppgrad::Tensor& Out) {
    // Basic sanity checks
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX2 mul: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX2 mul: shape mismatch");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 8; // 8 floats
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }

    // remainder
    for (; i < n; ++i) {
        o[i] = a[i] * b[i];
    }
}

} // namespace cppgrad


CPPGRAD_REGISTER_BACKEND(avx2, {
    std::cout << "[cppgrad] Registering AVX2 kernels...\n";
    auto& R = cppgrad::KernelRegistry::instance();
    R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cppgrad::add_f32_avx2);
    R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cppgrad::mul_f32_avx2);
});
