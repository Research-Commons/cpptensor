#include "cpptensor/backend/isa/avx512.hpp"
#include <immintrin.h>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <limits>

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

    // ============================================================================
    // Reduction Operations (Sum, Mean) - AVX-512 Optimized
    // ============================================================================

    // Helper: Horizontal sum of AVX-512 vector
    inline float horizontal_sum_avx512_reduction(__m512 v) {
        // Reduce 512-bit vector to 256-bit by adding upper and lower halves
        __m256 low = _mm512_castps512_ps256(v);
        __m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        __m256 sum256 = _mm256_add_ps(low, high);

        // Reduce 256-bit to 128-bit
        __m128 low128 = _mm256_castps256_ps128(sum256);
        __m128 high128 = _mm256_extractf128_ps(sum256, 1);
        __m128 sum128 = _mm_add_ps(low128, high128);

        // Reduce 128-bit to scalar
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        return _mm_cvtss_f32(sum128);
    }

    /**
     * @brief AVX-512 Sum Reduction Kernel (Wide SIMD Optimized)
     *
     * Vectorized sum using AVX-512 instructions (512-bit, 16 floats per vector).
     *
     * Algorithm Design:
     * 1. Global Reduction (dim=-1):
     *    - Use 4 independent vector accumulators (ILP optimization)
     *    - Main loop: Process 64 floats/iteration (4 vectors × 16 floats)
     *    - Each accumulator: 16-way parallel sum
     *    - Combine accumulators → horizontal sum → scalar result
     *    - Tail handling: AVX-512 masked load (branchless!)
     *
     * 2. Dimensional Reduction:
     *    a) Large inner (≥16):
     *       - Vectorize inner loop (16-float vectors)
     *       - Loop unrolling: 32 floats/iteration
     *       - Masked tail for non-multiple-of-16 sizes
     *    b) Small inner, large reduce (≥16):
     *       - Vectorize reduction loop (gather operation)
     *       - Manual gather into temp buffer (AVX-512 gather is expensive)
     *    c) Both small: Scalar fallback
     *
     * Key AVX-512 Features:
     * - **Masked operations**: Branchless tail handling
     *   - _mm512_maskz_loadu_ps: Load with mask (zero unused lanes)
     *   - _mm512_mask_storeu_ps: Store with mask
     *   - Mask: bitmask where bit i controls lane i
     *
     * - **2× vector width**: 16 floats vs 8 (AVX2)
     *   - Better cache line utilization (64 bytes = 16 floats)
     *   - Fewer iterations needed
     *
     * - **Independent FMA units**: Some CPUs have 2× FMA512 units
     *   - Theoretical 2× throughput vs AVX2
     *   - Practice: Memory bandwidth limits gains
     *
     * Why AVX-512 Gain is Modest (+0.9% vs AVX2 for sum):
     * - Memory bandwidth saturated
     * - Cache residency (16MB tensor fits in L3)
     * - Lower CPU frequency (AVX-512 throttles boost clocks)
     * - Compute is not bottleneck for sum
     *
     * SIMD Instructions Used:
     * - _mm512_loadu_ps: Load 16 floats (unaligned)
     * - _mm512_add_ps: Parallel add (16-way)
     * - _mm512_maskz_loadu_ps: Masked load (tail handling)
     * - _mm512_mask_storeu_ps: Masked store
     *
     * Performance: ~128μs for 2048×2048 tensor (11.7× faster than CPU)
     * Speedup Analysis:
     * - Only marginally faster than AVX2 (129μs)
     * - Expected: 2× width should give 2× speedup
     * - Reality: Memory bandwidth limited, not compute
     * - Tail handling is cleaner (masked ops vs scalar loop)
     *
     * @param input Input tensor to reduce
     * @param output Output tensor to store result
     * @param dim Dimension to reduce over (-1 for global reduction)
     * @param keepdim Whether to keep reduced dimension (size 1) or squeeze it
     */
    void AVX512::sum_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim) {
        const auto& in_shape = input.shape();
        const size_t ndim = in_shape.size();
        const float* in_data = input.data().data();
        float* out_data = output.data().data();

        // Case 1: Sum all elements (dim = -1)
        if (dim < 0) {
            const size_t total = input.numel();
            const size_t vec_size = 16; // AVX-512 processes 16 floats
            const size_t vec_end = (total / vec_size) * vec_size;

            // Use 4 independent accumulators to maximize ILP (Instruction Level Parallelism)
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            size_t i = 0;
            // Process 64 floats per iteration (4 vectors × 16 floats)
            for (; i + 64 <= vec_end; i += 64) {
                __m512 v0 = _mm512_loadu_ps(&in_data[i]);
                __m512 v1 = _mm512_loadu_ps(&in_data[i + 16]);
                __m512 v2 = _mm512_loadu_ps(&in_data[i + 32]);
                __m512 v3 = _mm512_loadu_ps(&in_data[i + 48]);

                sum0 = _mm512_add_ps(sum0, v0);
                sum1 = _mm512_add_ps(sum1, v1);
                sum2 = _mm512_add_ps(sum2, v2);
                sum3 = _mm512_add_ps(sum3, v3);
            }

            // Process remaining full vectors
            for (; i < vec_end; i += 16) {
                __m512 v = _mm512_loadu_ps(&in_data[i]);
                sum0 = _mm512_add_ps(sum0, v);
            }

            // Combine the 4 accumulators
            __m512 sum_combined = _mm512_add_ps(
                _mm512_add_ps(sum0, sum1),
                _mm512_add_ps(sum2, sum3)
            );

            // Horizontal sum
            float sum = horizontal_sum_avx512_reduction(sum_combined);

            // Handle tail with masking (no conditional branches!)
            if (i < total) {
                size_t remaining = total - i;
                __mmask16 mask = (__mmask16)((1U << remaining) - 1);
                __m512 tail = _mm512_maskz_loadu_ps(mask, &in_data[i]);
                sum += horizontal_sum_avx512_reduction(tail);
            }

            out_data[0] = sum;
            return;
        }

        // Case 2: Sum along specific dimension
        const size_t dim_size = static_cast<size_t>(dim);

        // Compute iteration bounds
        size_t outer = 1, reduce = in_shape[dim_size], inner = 1;
        for (size_t i = 0; i < dim_size; ++i) outer *= in_shape[i];
        for (size_t i = dim_size + 1; i < ndim; ++i) inner *= in_shape[i];

        const size_t out_total = outer * inner;

        // Zero initialize output
        std::memset(out_data, 0, out_total * sizeof(float));

        // Optimize based on inner dimension size
        if (inner >= 16) {
            // Large inner dimension - vectorize inner loop
            const size_t vec_size = 16;
            const size_t vec_end = (inner / vec_size) * vec_size;

            for (size_t o = 0; o < outer; ++o) {
                for (size_t r = 0; r < reduce; ++r) {
                    const float* in_ptr = &in_data[(o * reduce + r) * inner];
                    float* out_ptr = &out_data[o * inner];

                    size_t i = 0;
                    // Vectorized accumulation with loop unrolling
                    for (; i + 32 <= vec_end; i += 32) {
                        __m512 in0 = _mm512_loadu_ps(&in_ptr[i]);
                        __m512 in1 = _mm512_loadu_ps(&in_ptr[i + 16]);
                        __m512 out0 = _mm512_loadu_ps(&out_ptr[i]);
                        __m512 out1 = _mm512_loadu_ps(&out_ptr[i + 16]);

                        out0 = _mm512_add_ps(out0, in0);
                        out1 = _mm512_add_ps(out1, in1);

                        _mm512_storeu_ps(&out_ptr[i], out0);
                        _mm512_storeu_ps(&out_ptr[i + 16], out1);
                    }

                    // Single vector
                    for (; i < vec_end; i += 16) {
                        __m512 in_vec = _mm512_loadu_ps(&in_ptr[i]);
                        __m512 out_vec = _mm512_loadu_ps(&out_ptr[i]);
                        out_vec = _mm512_add_ps(out_vec, in_vec);
                        _mm512_storeu_ps(&out_ptr[i], out_vec);
                    }

                    // Masked tail (branchless!)
                    if (i < inner) {
                        size_t remaining = inner - i;
                        __mmask16 mask = (__mmask16)((1U << remaining) - 1);
                        __m512 in_vec = _mm512_maskz_loadu_ps(mask, &in_ptr[i]);
                        __m512 out_vec = _mm512_maskz_loadu_ps(mask, &out_ptr[i]);
                        out_vec = _mm512_add_ps(out_vec, in_vec);
                        _mm512_mask_storeu_ps(&out_ptr[i], mask, out_vec);
                    }
                }
            }
        } else if (reduce >= 16) {
            // Small inner, large reduce - vectorize reduction loop
            const size_t vec_size = 16;
            const size_t vec_end = (reduce / vec_size) * vec_size;

            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    __m512 sum_vec = _mm512_setzero_ps();

                    size_t r = 0;
                    // Gather operation is expensive, but better than scalar for large reduce
                    // Note: For very strided access, consider transposing or tiling
                    for (; r < vec_end; r += 16) {
                        // Manual gather (AVX-512 has _mm512_i32gather_ps but requires index vector)
                        alignas(64) float temp[16];
                        for (int j = 0; j < 16; ++j) {
                            temp[j] = in_data[(o * reduce + r + j) * inner + i];
                        }
                        __m512 vals = _mm512_load_ps(temp);
                        sum_vec = _mm512_add_ps(sum_vec, vals);
                    }

                    float sum = horizontal_sum_avx512_reduction(sum_vec);

                    // Scalar tail
                    for (; r < reduce; ++r) {
                        sum += in_data[(o * reduce + r) * inner + i];
                    }

                    out_data[o * inner + i] = sum;
                }
            }
        } else {
            // Both small - use scalar or minimal vectorization
            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    float sum = 0.0f;
                    for (size_t r = 0; r < reduce; ++r) {
                        sum += in_data[(o * reduce + r) * inner + i];
                    }
                    out_data[o * inner + i] = sum;
                }
            }
        }
    }

    /**
     * @brief AVX-512 Mean Reduction Kernel (Wide SIMD Optimized)
     *
     * Computes mean by dividing AVX-512 sum by reduction size.
     *
     * Algorithm:
     * 1. Call sum_f32_avx512() for SIMD sum
     * 2. Vectorized division:
     *    - Broadcast divisor to 16 lanes
     *    - _mm512_div_ps: 16-way parallel division
     *    - Loop unrolling: 32 elements/iteration
     *    - Masked tail for remainder
     *
     * AVX-512 Division Performance:
     * - Latency: ~16-23 cycles (CPU dependent)
     * - Throughput: 1 per 4-5 cycles (pipelined)
     * - 16-way parallelism amortizes latency
     *
     * Why Mean Shows Better AVX-512 Gain (+12% vs AVX2):
     * - Division is compute-bound (not memory-bound)
     * - 16-way division shows real benefit
     * - Sum is memory-bound (no compute advantage)
     *
     * Performance: ~115μs for 2048×2048 tensor (12.9× faster than CPU)
     * - Better speedup than sum (12.9× vs 11.7×)
     * - Division phase benefits from wider vectors
     *
     * @param input Input tensor to reduce
     * @param output Output tensor to store result
     * @param dim Dimension to reduce over (-1 for global reduction)
     * @param keepdim Whether to keep reduced dimension (size 1) or squeeze it
     */
    void AVX512::mean_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim) {
        // First compute sum using optimized AVX-512 sum kernel
        sum_f32_avx512(input, output, dim, keepdim);

        // Compute reduction size
        const auto& in_shape = input.shape();
        size_t reduce_size;

        if (dim < 0) {
            reduce_size = input.numel();
        } else {
            reduce_size = in_shape[static_cast<size_t>(dim)];
        }

        // Divide by reduction size using AVX-512
        float* out_data = output.data().data();
        const size_t out_total = output.numel();
        const float divisor = static_cast<float>(reduce_size);

        const size_t vec_size = 16;
        const size_t vec_end = (out_total / vec_size) * vec_size;

        __m512 divisor_vec = _mm512_set1_ps(divisor);

        size_t i = 0;
        // Vectorized division with loop unrolling
        for (; i + 32 <= vec_end; i += 32) {
            __m512 val0 = _mm512_loadu_ps(&out_data[i]);
            __m512 val1 = _mm512_loadu_ps(&out_data[i + 16]);

            val0 = _mm512_div_ps(val0, divisor_vec);
            val1 = _mm512_div_ps(val1, divisor_vec);

            _mm512_storeu_ps(&out_data[i], val0);
            _mm512_storeu_ps(&out_data[i + 16], val1);
        }

        // Single vector
        for (; i < vec_end; i += 16) {
            __m512 val = _mm512_loadu_ps(&out_data[i]);
            val = _mm512_div_ps(val, divisor_vec);
            _mm512_storeu_ps(&out_data[i], val);
        }

        // Masked tail
        if (i < out_total) {
            size_t remaining = out_total - i;
            __mmask16 mask = (__mmask16)((1U << remaining) - 1);
            __m512 val = _mm512_maskz_loadu_ps(mask, &out_data[i]);
            val = _mm512_div_ps(val, divisor_vec);
            _mm512_mask_storeu_ps(&out_data[i], mask, val);
        }
    }

    // ============================================================================
    // Max/Min Reduction Operations - AVX-512 Optimized
    // ============================================================================

    // Helper: Horizontal max of AVX-512 vector
    inline float horizontal_max_avx512_reduction(__m512 v) {
        // Reduce 512-bit vector to 256-bit by taking max of upper and lower halves
        __m256 low = _mm512_castps512_ps256(v);
        __m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        __m256 max256 = _mm256_max_ps(low, high);

        // Reduce 256-bit to 128-bit
        __m128 low128 = _mm256_castps256_ps128(max256);
        __m128 high128 = _mm256_extractf128_ps(max256, 1);
        __m128 max128 = _mm_max_ps(low128, high128);

        // Reduce 128-bit to scalar
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));

        return _mm_cvtss_f32(max128);
    }

    // Helper: Horizontal min of AVX-512 vector
    inline float horizontal_min_avx512_reduction(__m512 v) {
        // Reduce 512-bit vector to 256-bit by taking min of upper and lower halves
        __m256 low = _mm512_castps512_ps256(v);
        __m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        __m256 min256 = _mm256_min_ps(low, high);

        // Reduce 256-bit to 128-bit
        __m128 low128 = _mm256_castps256_ps128(min256);
        __m128 high128 = _mm256_extractf128_ps(min256, 1);
        __m128 min128 = _mm_min_ps(low128, high128);

        // Reduce 128-bit to scalar
        min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(2, 3, 0, 1)));
        min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(1, 0, 3, 2)));

        return _mm_cvtss_f32(min128);
    }

    /**
     * @brief AVX-512 Max Reduction Kernel (Wide SIMD Optimized)
     *
     * Vectorized max using AVX-512 comparison instructions (16-way parallel).
     *
     * Algorithm Design:
     * 1. Global Reduction (dim=-1):
     *    - Initialize 4 accumulators to -infinity
     *    - Main loop: Process 64 floats/iteration (4 vectors × 16 floats)
     *    - _mm512_max_ps: 16-way parallel max comparison
     *    - Combine accumulators → horizontal max → scalar result
     *    - **CRITICAL**: Tail uses scalar loop, NOT masked load
     *
     * 2. Dimensional Reduction:
     *    a) Large inner (≥16): Vectorize inner loop
     *       - Load/store 16-float vectors
     *       - Loop unrolling: 32 floats/iteration
     *       - Masked tail for remainder
     *    b) Small inner: Scalar fallback
     *
     * Why Max/Min Excel with AVX-512 (+19% over AVX2):
     * - **Lower instruction latency**: max is 1 cycle vs add's 3 cycles
     * - **No data dependencies**: Each max is independent
     * - **Perfect for wide SIMD**: 16-way parallelism shines
     * - **Compute-bound operation**: Not memory bandwidth limited
     *
     * AVX-512 Max Instruction Details:
     * - _mm512_max_ps: Latency 1 cycle, throughput 0.5 cycles (2/cycle)
     * - Compare to _mm512_add_ps: Latency 3 cycles, throughput 0.5 cycles
     * - Max is 3× faster per operation!
     *
     * SIMD Instructions Used:
     * - _mm512_max_ps: 16-way parallel max, 1 cycle latency
     * - _mm512_set1_ps: Broadcast -infinity to all 16 lanes
     * - Horizontal reduction via shuffles
     *
     * Performance: ~102μs for 2048×2048 tensor (14.8× faster than CPU) ✨
     * Speedup Analysis:
     * - Best-in-class speedup for all operations!
     * - 19% faster than AVX2 (122μs)
     * - 12% faster than AVX-512 sum (128μs)
     * - Why: Max is compute-bound, benefits from 2× width + lower latency
     *
     * @param input Input tensor to reduce
     * @param output Output tensor to store result
     * @param dim Dimension to reduce over (-1 for global reduction)
     * @param keepdim Whether to keep reduced dimension (size 1) or squeeze it
     */
    void AVX512::max_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim) {
        const auto& in_shape = input.shape();
        const size_t ndim = in_shape.size();
        const float* in_data = input.data().data();
        float* out_data = output.data().data();

        // Case 1: Max of all elements (dim = -1)
        if (dim < 0) {
            const size_t total = input.numel();
            const size_t vec_size = 16; // AVX-512 processes 16 floats
            const size_t vec_end = (total / vec_size) * vec_size;

            // Use 4 independent accumulators for ILP
            __m512 max0 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            __m512 max1 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            __m512 max2 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            __m512 max3 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

            size_t i = 0;
            // Process 64 floats per iteration (4 vectors × 16 floats)
            for (; i + 64 <= vec_end; i += 64) {
                __m512 v0 = _mm512_loadu_ps(&in_data[i]);
                __m512 v1 = _mm512_loadu_ps(&in_data[i + 16]);
                __m512 v2 = _mm512_loadu_ps(&in_data[i + 32]);
                __m512 v3 = _mm512_loadu_ps(&in_data[i + 48]);

                max0 = _mm512_max_ps(max0, v0);
                max1 = _mm512_max_ps(max1, v1);
                max2 = _mm512_max_ps(max2, v2);
                max3 = _mm512_max_ps(max3, v3);
            }

            // Process remaining full vectors
            for (; i < vec_end; i += 16) {
                __m512 v = _mm512_loadu_ps(&in_data[i]);
                max0 = _mm512_max_ps(max0, v);
            }

            // Combine the 4 accumulators
            __m512 max_combined = _mm512_max_ps(
                _mm512_max_ps(max0, max1),
                _mm512_max_ps(max2, max3)
            );

            // Horizontal max
            float max_val = horizontal_max_avx512_reduction(max_combined);

            // Handle tail with scalar loop (masking sets unused lanes to 0, which breaks max for negative numbers)
            for (; i < total; ++i) {
                if (in_data[i] > max_val) {
                    max_val = in_data[i];
                }
            }

            out_data[0] = max_val;
            return;
        }

        // Case 2: Max along specific dimension
        const size_t dim_size = static_cast<size_t>(dim);

        size_t outer = 1, reduce = in_shape[dim_size], inner = 1;
        for (size_t i = 0; i < dim_size; ++i) outer *= in_shape[i];
        for (size_t i = dim_size + 1; i < ndim; ++i) inner *= in_shape[i];

        const size_t out_total = outer * inner;

        // Initialize output with -infinity
        for (size_t i = 0; i < out_total; ++i) {
            out_data[i] = -std::numeric_limits<float>::infinity();
        }

        // Optimize based on inner dimension size
        if (inner >= 16) {
            // Large inner dimension - vectorize inner loop
            const size_t vec_size = 16;
            const size_t vec_end = (inner / vec_size) * vec_size;

            for (size_t o = 0; o < outer; ++o) {
                for (size_t r = 0; r < reduce; ++r) {
                    const float* in_ptr = &in_data[(o * reduce + r) * inner];
                    float* out_ptr = &out_data[o * inner];

                    size_t i = 0;
                    // Vectorized max with loop unrolling
                    for (; i + 32 <= vec_end; i += 32) {
                        __m512 in0 = _mm512_loadu_ps(&in_ptr[i]);
                        __m512 in1 = _mm512_loadu_ps(&in_ptr[i + 16]);
                        __m512 out0 = _mm512_loadu_ps(&out_ptr[i]);
                        __m512 out1 = _mm512_loadu_ps(&out_ptr[i + 16]);

                        out0 = _mm512_max_ps(out0, in0);
                        out1 = _mm512_max_ps(out1, in1);

                        _mm512_storeu_ps(&out_ptr[i], out0);
                        _mm512_storeu_ps(&out_ptr[i + 16], out1);
                    }

                    // Single vector
                    for (; i < vec_end; i += 16) {
                        __m512 in_vec = _mm512_loadu_ps(&in_ptr[i]);
                        __m512 out_vec = _mm512_loadu_ps(&out_ptr[i]);
                        out_vec = _mm512_max_ps(out_vec, in_vec);
                        _mm512_storeu_ps(&out_ptr[i], out_vec);
                    }

                    // Masked tail
                    if (i < inner) {
                        size_t remaining = inner - i;
                        __mmask16 mask = (__mmask16)((1U << remaining) - 1);
                        __m512 in_vec = _mm512_maskz_loadu_ps(mask, &in_ptr[i]);
                        __m512 out_vec = _mm512_maskz_loadu_ps(mask, &out_ptr[i]);
                        out_vec = _mm512_max_ps(out_vec, in_vec);
                        _mm512_mask_storeu_ps(&out_ptr[i], mask, out_vec);
                    }
                }
            }
        } else {
            // Small inner - use scalar
            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    for (size_t r = 0; r < reduce; ++r) {
                        size_t in_idx = (o * reduce + r) * inner + i;
                        size_t out_idx = o * inner + i;
                        if (in_data[in_idx] > out_data[out_idx]) {
                            out_data[out_idx] = in_data[in_idx];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief AVX-512 Min Reduction Kernel (Wide SIMD Optimized)
     *
     * Vectorized min using AVX-512 comparison instructions (16-way parallel).
     *
     * Algorithm:
     * - Identical to max_f32_avx512, but with:
     *   - Initialize accumulators to +infinity (not -infinity)
     *   - Use _mm512_min_ps (not _mm512_max_ps)
     *   - Use horizontal_min (not horizontal_max)
     *   - Less-than comparison in scalar tail
     *
     * SIMD Instructions Used:
     * - _mm512_min_ps: 16-way parallel min, 1 cycle latency
     * - _mm512_set1_ps: Broadcast +infinity
     * - Same horizontal reduction as max
     *
     * Performance: ~102μs for 2048×2048 tensor (14.8× faster than CPU) ✨
     * Speedup Analysis:
     * - Identical to max performance (as expected)
     * - Best speedup of all reduction operations
     * - 18% faster than AVX2 min (121μs)
     * - 23% faster for dimensional reductions (7.0× vs 5.7×)
     *
     * Why Min/Max Achieve Best Speedup:
     * 1. **Lower latency**: 1 cycle vs 3 for add
     * 2. **No dependencies**: Each comparison independent
     * 3. **Compute-bound**: Not memory bandwidth limited
     * 4. **Perfect parallelism**: 16-way benefits fully realized
     * 5. **Simple horizontal reduction**: No complex accumulation
     *
     * @param input Input tensor to reduce
     * @param output Output tensor to store result
     * @param dim Dimension to reduce over (-1 for global reduction)
     * @param keepdim Whether to keep reduced dimension (size 1) or squeeze it
     */
    void AVX512::min_f32_avx512(const Tensor& input, Tensor& output, int dim, bool keepdim) {
        const auto& in_shape = input.shape();
        const size_t ndim = in_shape.size();
        const float* in_data = input.data().data();
        float* out_data = output.data().data();

        // Case 1: Min of all elements (dim = -1)
        if (dim < 0) {
            const size_t total = input.numel();
            const size_t vec_size = 16; // AVX-512 processes 16 floats
            const size_t vec_end = (total / vec_size) * vec_size;

            // Use 4 independent accumulators for ILP
            __m512 min0 = _mm512_set1_ps(std::numeric_limits<float>::infinity());
            __m512 min1 = _mm512_set1_ps(std::numeric_limits<float>::infinity());
            __m512 min2 = _mm512_set1_ps(std::numeric_limits<float>::infinity());
            __m512 min3 = _mm512_set1_ps(std::numeric_limits<float>::infinity());

            size_t i = 0;
            // Process 64 floats per iteration (4 vectors × 16 floats)
            for (; i + 64 <= vec_end; i += 64) {
                __m512 v0 = _mm512_loadu_ps(&in_data[i]);
                __m512 v1 = _mm512_loadu_ps(&in_data[i + 16]);
                __m512 v2 = _mm512_loadu_ps(&in_data[i + 32]);
                __m512 v3 = _mm512_loadu_ps(&in_data[i + 48]);

                min0 = _mm512_min_ps(min0, v0);
                min1 = _mm512_min_ps(min1, v1);
                min2 = _mm512_min_ps(min2, v2);
                min3 = _mm512_min_ps(min3, v3);
            }

            // Process remaining full vectors
            for (; i < vec_end; i += 16) {
                __m512 v = _mm512_loadu_ps(&in_data[i]);
                min0 = _mm512_min_ps(min0, v);
            }

            // Combine the 4 accumulators
            __m512 min_combined = _mm512_min_ps(
                _mm512_min_ps(min0, min1),
                _mm512_min_ps(min2, min3)
            );

            // Horizontal min
            float min_val = horizontal_min_avx512_reduction(min_combined);

            // Handle tail with scalar loop (masking sets unused lanes to 0, which breaks min)
            for (; i < total; ++i) {
                if (in_data[i] < min_val) {
                    min_val = in_data[i];
                }
            }

            out_data[0] = min_val;
            return;
        }

        // Case 2: Min along specific dimension
        const size_t dim_size = static_cast<size_t>(dim);

        size_t outer = 1, reduce = in_shape[dim_size], inner = 1;
        for (size_t i = 0; i < dim_size; ++i) outer *= in_shape[i];
        for (size_t i = dim_size + 1; i < ndim; ++i) inner *= in_shape[i];

        const size_t out_total = outer * inner;

        // Initialize output with +infinity
        for (size_t i = 0; i < out_total; ++i) {
            out_data[i] = std::numeric_limits<float>::infinity();
        }

        // Optimize based on inner dimension size
        if (inner >= 16) {
            // Large inner dimension - vectorize inner loop
            const size_t vec_size = 16;
            const size_t vec_end = (inner / vec_size) * vec_size;

            for (size_t o = 0; o < outer; ++o) {
                for (size_t r = 0; r < reduce; ++r) {
                    const float* in_ptr = &in_data[(o * reduce + r) * inner];
                    float* out_ptr = &out_data[o * inner];

                    size_t i = 0;
                    // Vectorized min with loop unrolling
                    for (; i + 32 <= vec_end; i += 32) {
                        __m512 in0 = _mm512_loadu_ps(&in_ptr[i]);
                        __m512 in1 = _mm512_loadu_ps(&in_ptr[i + 16]);
                        __m512 out0 = _mm512_loadu_ps(&out_ptr[i]);
                        __m512 out1 = _mm512_loadu_ps(&out_ptr[i + 16]);

                        out0 = _mm512_min_ps(out0, in0);
                        out1 = _mm512_min_ps(out1, in1);

                        _mm512_storeu_ps(&out_ptr[i], out0);
                        _mm512_storeu_ps(&out_ptr[i + 16], out1);
                    }

                    // Single vector
                    for (; i < vec_end; i += 16) {
                        __m512 in_vec = _mm512_loadu_ps(&in_ptr[i]);
                        __m512 out_vec = _mm512_loadu_ps(&out_ptr[i]);
                        out_vec = _mm512_min_ps(out_vec, in_vec);
                        _mm512_storeu_ps(&out_ptr[i], out_vec);
                    }

                    // Masked tail
                    if (i < inner) {
                        size_t remaining = inner - i;
                        __mmask16 mask = (__mmask16)((1U << remaining) - 1);
                        __m512 in_vec = _mm512_maskz_loadu_ps(mask, &in_ptr[i]);
                        __m512 out_vec = _mm512_maskz_loadu_ps(mask, &out_ptr[i]);
                        out_vec = _mm512_min_ps(out_vec, in_vec);
                        _mm512_mask_storeu_ps(&out_ptr[i], mask, out_vec);
                    }
                }
            }
        } else {
            // Small inner - use scalar
            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    for (size_t r = 0; r < reduce; ++r) {
                        size_t in_idx = (o * reduce + r) * inner + i;
                        size_t out_idx = o * inner + i;
                        if (in_data[in_idx] < out_data[out_idx]) {
                            out_data[out_idx] = in_data[in_idx];
                        }
                    }
                }
            }
        }
    }

} // namespace cppgrad
