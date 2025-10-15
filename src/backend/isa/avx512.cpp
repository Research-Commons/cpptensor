#include "cpptensor/backend/isa/avx512.hpp"
#include <immintrin.h>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h" // adjust include to your actual path
#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor{

void add_f32_avx512(const cpptensor::Tensor& A,
                    const cpptensor::Tensor& B,
                    cpptensor::Tensor& Out) {
    using namespace cpptensor;

    // Basic sanity checks
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX512 add: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX512 add: shape mismatch");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 16; // 16 floats per __m512
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(o + i, vc);
    }

    int rem = static_cast<int>(n - i);
    if (rem > 0) {
        __mmask16 k = static_cast<__mmask16>((1u << rem) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(k, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(k, b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_mask_storeu_ps(o + i, k, vc);
    }
}

void mul_f32_avx512(const cpptensor::Tensor& A,
                    const cpptensor::Tensor& B,
                    cpptensor::Tensor& Out) {
    using namespace cpptensor;

    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX512 mul: only CPU tensors supported");
    }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("AVX512 mul: shape mismatch");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* o = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    const int stride = 16;
    std::int64_t i = 0;

    for (; i + stride <= n; i += stride) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(o + i, vc);
    }

    int rem = static_cast<int>(n - i);
    if (rem > 0) {
        __mmask16 k = static_cast<__mmask16>((1u << rem) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(k, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(k, b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_mask_storeu_ps(o + i, k, vc);
    }
}

} // namespace cppgrad

// registration (called once at startup)
CPPGRAD_REGISTER_BACKEND(avx512, {
    std::cout << "[cppgrad] Registering AVX512 kernels...\n";
    auto& R = cpptensor::KernelRegistry::instance();
    R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX512, cpptensor::add_f32_avx512);
    R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX512, cpptensor::mul_f32_avx512);
});