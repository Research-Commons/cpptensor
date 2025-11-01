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

    //my version of gemm

    // void AVX512::gemm_f32_avx512(const Tensor &A, const Tensor &B, Tensor &Out) {
    //     const int64_t M = A.shape()[0];
    //     const int64_t K = A.shape()[1];
    //     const int64_t KB = B.shape()[0];
    //     const int64_t N = B.shape()[1];
    //
    //     if (K != KB || Out.shape()[0] != M || Out.shape()[1] != N) {
    //         throw std::runtime_error("AVX512 GEMM: shape mismatch (A: MxK, B: KxN, C: MxN)");
    //     }
    //
    //     const float* a = A.data().data();
    //     const float* b = B.data().data();
    //     float* c = Out.data().data();
    //
    //     constexpr int TILE_M = 16;
    //     constexpr int TILE_N = 16;
    //     constexpr int TILE_K = 16;
    //
    //     for (int64_t i = 0; i < M; i += TILE_M) {
    //         for (int64_t j = 0; j < N; j += TILE_N) {
    //
    //             //init accumulator
    //             __m512 acc[TILE_M];
    //             for (int ii = 0; ii < TILE_M; ++ii) {
    //                 acc[ii] = _mm512_setzero_ps();
    //             }
    //
    //             //compute
    //             for (int64_t k = 0; k < K; k += TILE_K) {
    //                 const int64_t kMax = std::min<int64_t>(K, k + TILE_K);
    //
    //                 for (int64_t kk = k; kk < kMax; ++kk) {
    //                     // Load B[kk, j:j+16) with masking for tail
    //                     int remain = static_cast<int>(std::min<int64_t>(TILE_N, N - j));
    //                     __mmask16 km = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);
    //                     __m512 bcol = (remain >= 16)
    //                                   ? _mm512_loadu_ps(&b[kk * N + j])
    //                                   : _mm512_maskz_loadu_ps(km, &b[kk * N + j]);
    //
    //                     for (int ii = 0; ii < TILE_M; ++ii) {
    //                         if (i + ii >= M) break;
    //
    //                         __m512 aval = _mm512_set1_ps(a[(i + ii) * K + kk]);
    //                         acc[ii] = _mm512_fmadd_ps(aval, bcol, acc[ii]);
    //                     }
    //                 }
    //             }
    //
    //             //store
    //             for (int ii = 0; ii < TILE_M; ++ii) {
    //                 if (i + ii >= M) break;
    //                 int remain = static_cast<int>(std::min<int64_t>(TILE_N, N - j));
    //                 __mmask16 km = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);
    //                 if (remain >= 16) {
    //                     _mm512_storeu_ps(&c[(i + ii) * N + j], acc[ii]);
    //                 } else {
    //                     _mm512_mask_storeu_ps(&c[(i + ii) * N + j], km, acc[ii]);
    //                 }
    //             }
    //         }
    //     }
    // }

    //optimised version of gemm i found online (1.5x-2x better)

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

        // Tunable tile sizes. Smaller TILE_M improves register pressure; larger TILE_K improves cache locality.
        constexpr int TILE_M = 12;
        constexpr int TILE_N = 16;
        constexpr int TILE_K = 64;

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

                    // Unroll the K loop by 2 for better ILP
                    int64_t kk = k;
                    for (; kk + 1 < kMax; kk += 2) {
                        // Prefetch upcoming A and B rows to L1
                        _mm_prefetch((const char*)(&b[(kk + 16) * N + j]), _MM_HINT_T0);
                        if (i < M)
                            _mm_prefetch((const char*)(&a[(i) * K + kk + 16]), _MM_HINT_T0);

                        int remain = static_cast<int>(std::min<int64_t>(TILE_N, N - j));
                        __mmask16 km = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);

                        // Load two consecutive B rows for current j-tile
                        __m512 b0 = (remain >= 16)
                                    ? _mm512_loadu_ps(&b[kk * N + j])
                                    : _mm512_maskz_loadu_ps(km, &b[kk * N + j]);
                        __m512 b1 = (remain >= 16)
                                    ? _mm512_loadu_ps(&b[(kk + 1) * N + j])
                                    : _mm512_maskz_loadu_ps(km, &b[(kk + 1) * N + j]);

                        // Accumulate for TILE_M rows
                        for (int ii = 0; ii < TILE_M; ++ii) {
                            if (i + ii >= M) break;
                            float a0 = a[(i + ii) * K + kk];
                            float a1 = a[(i + ii) * K + (kk + 1)];
                            __m512 va0 = _mm512_set1_ps(a0);
                            __m512 va1 = _mm512_set1_ps(a1);
                            acc[ii] = _mm512_fmadd_ps(va0, b0, acc[ii]);
                            acc[ii] = _mm512_fmadd_ps(va1, b1, acc[ii]);
                        }
                    }

                    // Handle leftover kk
                    for (; kk < kMax; ++kk) {
                        int remain = static_cast<int>(std::min<int64_t>(TILE_N, N - j));
                        __mmask16 km = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);
                        __m512 bcol = (remain >= 16)
                                    ? _mm512_loadu_ps(&b[kk * N + j])
                                    : _mm512_maskz_loadu_ps(km, &b[kk * N + j]);
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
                    int remain = static_cast<int>(std::min<int64_t>(TILE_N, N - j));
                    __mmask16 km = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);
                    if (remain >= 16) {
                        _mm512_storeu_ps(&c[(i + ii) * N + j], acc[ii]);
                    } else {
                        _mm512_mask_storeu_ps(&c[(i + ii) * N + j], km, acc[ii]);
                    }
                }
            }
        }
    }


    //----DOT------

    static inline float hsum256(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        return _mm_cvtss_f32(vsum);
    }

    void AVX512::dot_f32_avx512(const Tensor& A, const Tensor& B, Tensor& Out) {
        // Sanity checks
        if (A.device_type() != DeviceType::CPU ||
            B.device_type() != DeviceType::CPU ||
            Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX512 dot: only CPU tensors supported");
            }

        if (A.shape().size() != 1 || B.shape().size() != 1) {
            throw std::runtime_error("AVX512 dot: inputs must be 1D vectors");
        }
        if (A.shape()[0] != B.shape()[0]) {
            throw std::runtime_error("AVX512 dot: size mismatch");
        }
        if (!Out.shape().empty()) {
            throw std::runtime_error("AVX512 dot: output must be scalar (shape [])");
        }

        const float* a = A.data().data();
        const float* b = B.data().data();
        const std::int64_t n = static_cast<std::int64_t>(A.shape()[0]);

        __m512 vacc = _mm512_setzero_ps();
        std::int64_t i = 0;
        constexpr int stride = 16;
        for (; i + stride <= n; i += stride) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            vacc = _mm512_fmadd_ps(va, vb, vacc); // FMA accumulate
        }

        // Reduce 512 -> two 256 then scalar
        __m256 lo = _mm512_castps512_ps256(vacc);
        __m256 hi = _mm512_extractf32x8_ps(vacc, 1);
        float sum = hsum256(lo) + hsum256(hi);

        // Remainder
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }

        Out.data()[0] = sum;
    }

} // namespace cppgrad
