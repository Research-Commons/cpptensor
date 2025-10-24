#include "cpptensor/backend/isa/avx512.hpp"
#include <immintrin.h>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h" // adjust include to your actual path
#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor{

    void AVX512::add_f32_avx512(const cpptensor::Tensor& A,
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

    void AVX512::mul_f32_avx512(const cpptensor::Tensor& A,
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

    void AVX512::gemm_f32_avx512(const Tensor &A, const Tensor &B, Tensor &Out) {
        const int64_t M = A.shape()[0];
        const int64_t K = A.shape()[1];
        const int64_t KB = B.shape()[0];
        const int64_t N = B.shape()[1];

        if (K != KB || Out.shape()[0] != M || Out.shape()[1] != N) {
            throw std::runtime_error("AVX512 GEMM: shape mismatch (A: MxK, B: KxN, C: MxN)");
        }

        const float* a = A.data().data();
        const float* b = B.data().data();
        float* c = Out.data().data();

        constexpr int TILE_M = 16;
        constexpr int TILE_N = 16;
        constexpr int TILE_K = 16;

        for (int64_t i = 0; i < M; i += TILE_M) {
            for (int64_t j = 0; j < N; j += TILE_N) {

                //init accumulator
                __m512 acc[TILE_M];
                for (int ii = 0; ii < TILE_M; ++ii) {
                    acc[ii] = _mm512_setzero_ps();
                }

                //compute
                for (int64_t k = 0; k < K; k += TILE_K) {
                    const int64_t kMax = std::min<int64_t>(K, k + TILE_K);

                    for (int64_t kk = k; kk < kMax; ++kk) {
                        __m512 bcol = _mm512_loadu_ps(&b[kk * N + j]);

                        for (int ii = 0; ii < TILE_M; ++ii) {
                            if (i + ii >= M) break;

                            __m512 aval = _mm512_set1_ps(a[(i + ii) * K + kk]);
                            acc[ii] = _mm512_fmadd_ps(aval, bcol, acc[ii]);
                        }
                    }
                }

                //store
                for (int ii = 0; ii < TILE_M; ++ii) {
                    if (i + ii >= M) break;
                    _mm512_storeu_ps(&c[(i + ii) * N + j], acc[ii]);
                }
            }
        }
    }
} // namespace cppgrad
