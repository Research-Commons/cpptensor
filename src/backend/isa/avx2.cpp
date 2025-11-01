#include <immintrin.h>
#include "cpptensor/backend/isa/avx2.hpp"

#include <cmath>
#include <cstring>

#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor {

    void AVX2::add_f32_avx2(const cpptensor::Tensor& A,
                      const cpptensor::Tensor& B,
                      cpptensor::Tensor& Out) {
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

    void AVX2::mul_f32_avx2(const cpptensor::Tensor& A,
                      const cpptensor::Tensor& B,
                      cpptensor::Tensor& Out) {
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

    void AVX2::sub_f32_avx2(const Tensor& A,
                  const Tensor& B,
                  Tensor& Out) {

        if (A.device_type() != DeviceType::CPU ||
            B.device_type() != DeviceType::CPU ||
            Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 sub: only CPU tensors supported");
            }

        if (A.shape() != B.shape() || A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 sub: shape mismatch");
        }

        const float* a = A.data().data();
        const float* b = B.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per AVX2 vector
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(o + i, vc);
        }

        // remainder
        for (; i < n; ++i) {
            o[i] = a[i] - b[i];
        }
    }

    void AVX2::div_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 div: only CPU tensors supported");
        }

        if (A.shape() != B.shape() || A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 div: shape mismatch");
        }

        const float* a = A.data().data();
        const float* b = B.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per AVX2 register
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(o + i, vc);
        }

        for (; i < n; ++i) {
            o[i] = a[i] / b[i];
        }
    }

    //----------EXP--------------

    /**
     * @brief Fast AVX2 exponential approximation (8 floats at once)
     *
     * Algorithm: Cephes-style range reduction
     * 1. Express x as: x = n*ln(2) + r, where n is integer and |r| < ln(2)/2
     * 2. Then: exp(x) = 2^n * exp(r)
     * 3. Approximate exp(r) with polynomial (r is small)
     * 4. Scale by 2^n using exponent manipulation
     *
     * @param x Input vector of 8 floats
     * @return exp(x) for each element
     */
    inline __m256 exp256_ps(__m256 x) {
        // Constants
        const __m256 ln2_inv = _mm256_set1_ps(1.44269504088896341f); // 1/ln(2) for converting to base-2
        const __m256 ln2     = _mm256_set1_ps(0.69314718056f);       // ln(2) for range reduction
        const __m256 max_x   = _mm256_set1_ps(88.3762626647949f);    // exp(88.4) ≈ FLT_MAX
        const __m256 min_x   = _mm256_set1_ps(-88.3762626647949f);   // exp(-88.4) ≈ 0

        // Step 1: Clamp input to valid range to prevent overflow/underflow
        x = _mm256_min_ps(x, max_x);
        x = _mm256_max_ps(x, min_x);

        // Step 2: Range reduction - find n where x = n*ln(2) + r
        // Compute n = round(x / ln(2))
        __m256 fx = _mm256_mul_ps(x, ln2_inv);
        fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // Compute remainder: r = x - n*ln(2)
        // Now |r| < ln(2)/2 ≈ 0.347, which makes polynomial accurate
        __m256 n_ln2 = _mm256_mul_ps(fx, ln2);
        __m256 r = _mm256_sub_ps(x, n_ln2);

        // Step 3: Polynomial approximation for exp(r) where r is small
        // Using Taylor series: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
        // Coefficients arranged for Horner's method
        const __m256 c1 = _mm256_set1_ps(1.9875691500E-4f);  // ~1/5040
        const __m256 c2 = _mm256_set1_ps(1.3981999507E-3f);  // ~1/720
        const __m256 c3 = _mm256_set1_ps(8.3334519073E-3f);  // ~1/120
        const __m256 c4 = _mm256_set1_ps(4.1665795894E-2f);  // ~1/24
        const __m256 c5 = _mm256_set1_ps(1.6666665459E-1f);  // ~1/6
        const __m256 c6 = _mm256_set1_ps(5.0000001201E-1f);  // ~1/2

        __m256 r2 = _mm256_mul_ps(r, r);  // r²

        // Evaluate polynomial using nested multiplication
        // p(r) = c1 + r*c2 + r²*c3 + r³*c4 + r⁴*c5 + r⁵*c6
        __m256 y = _mm256_add_ps(c1, _mm256_mul_ps(r, c2));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, c3));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, _mm256_mul_ps(r, c4)));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, _mm256_mul_ps(r2, c5)));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, _mm256_mul_ps(r2, _mm256_mul_ps(r, c6))));

        // Add linear and constant terms: exp(r) ≈ 1 + r + p(r)
        y = _mm256_add_ps(y, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

        // Step 4: Compute 2^n by manipulating float exponent bits
        // Float format: [sign(1) | exponent(8) | mantissa(23)]
        // For 2^n: exponent = n + 127 (bias), mantissa = 0
        __m256i mm = _mm256_cvtps_epi32(fx);               // Convert n to integer
        mm = _mm256_add_epi32(mm, _mm256_set1_epi32(127)); // Add bias (127)
        mm = _mm256_slli_epi32(mm, 23);                    // Shift into exponent position
        __m256 pow2n = _mm256_castsi256_ps(mm);            // Reinterpret as float

        // Final result: exp(x) = exp(r) * 2^n
        return _mm256_mul_ps(y, pow2n);
    }

    void AVX2::exp_f32_avx2(const Tensor& A, Tensor& Out) {
        // Basic sanity checks (mirror your add kernel style)
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 exp: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 exp: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per __m256
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vo = exp256_ps(va);
            _mm256_storeu_ps(o + i, vo);
        }

        // remainder (scalar fallback)
        for (; i < n; ++i) {
            o[i] = std::exp(a[i]);
        }
    }

    //////////


    // --------- LOG------------

    /**
     * @brief Fast AVX2 natural logarithm approximation (8 floats at once)
     *
     * Algorithm: IEEE 754 exponent/mantissa decomposition
     * 1. Decompose x into: x = 2^e * m, where 1 ≤ m < 2 (mantissa) and e is exponent
     * 2. Then: ln(x) = e*ln(2) + ln(m)
     * 3. Approximate ln(m) with polynomial (m is in [1,2])
     * 4. Combine exponent and mantissa contributions
     *
     * @param x Input vector of 8 floats (must be positive)
     * @return ln(x) for each element
     */
    inline __m256 log256_ps(__m256 x) {
        // Constants
        const __m256 one           = _mm256_set1_ps(1.0f);
        const __m256 ln2           = _mm256_set1_ps(0.69314718056f);     // ln(2)
        const __m256 min_norm_pos  = _mm256_set1_ps(1.17549435e-38f);    // Smallest positive normal float

        // Step 1: Clamp input to avoid log(0) or log(negative)
        // For values below min_norm_pos, result will be clamped
        x = _mm256_max_ps(x, min_norm_pos);

        // Step 2: Extract exponent and normalize mantissa
        // Float format: [sign(1) | exponent(8) | mantissa(23)]
        __m256i xi = _mm256_castps_si256(x);

        // Extract exponent: shift right 23 bits to get exponent field
        __m256i emm0 = _mm256_srli_epi32(xi, 23);
        // Remove bias (127) to get actual exponent e
        emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(127));

        // Normalize mantissa to range [1, 2):
        // Keep mantissa bits (lower 23), clear exponent, set exponent to 126 (bias for 0.5)
        // This effectively multiplies mantissa by 0.5, giving range [0.5, 1)
        xi = _mm256_and_si256(xi, _mm256_set1_epi32(0x7FFFFF));          // Mask mantissa bits
        xi = _mm256_or_si256(xi, _mm256_castps_si256(_mm256_set1_ps(0.5f))); // Set exponent to -1
        x = _mm256_castsi256_ps(xi);
        // Now x is in [0.5, 1), but we multiply by 2 conceptually to get [1, 2)

        // Convert exponent to float for ln(2) multiplication
        __m256 e = _mm256_cvtepi32_ps(emm0);

        // Step 3: Polynomial approximation for ln(mantissa)
        // For x in [1, 2], use ln(x) = ln(1+y) where y = x-1, so y in [0, 1]
        // Taylor series: ln(1+y) ≈ y - y²/2 + y³/3 - y⁴/4 + ...
        x = _mm256_sub_ps(x, one);  // y = mantissa - 1
        __m256 y = x;
        __m256 z = _mm256_mul_ps(y, y);  // y²

        // Polynomial coefficients for ln(1+y) approximation
        // These come from rational approximation or Chebyshev polynomials
        const __m256 c1 = _mm256_set1_ps(0.666666666666735130e+0f);  // ~2/3
        const __m256 c2 = _mm256_set1_ps(0.399999999994094190e+0f);  // ~2/5
        const __m256 c3 = _mm256_set1_ps(0.285714287436623910e+0f);  // ~2/7
        const __m256 c4 = _mm256_set1_ps(0.222221984321497840e+0f);  // ~2/9
        const __m256 c5 = _mm256_set1_ps(0.181835721616180500e+0f);  // ~2/11
        const __m256 c6 = _mm256_set1_ps(0.153138376992093730e+0f);  // ~2/13
        const __m256 c7 = _mm256_set1_ps(0.147981986051165860e+0f);  // ~2/15

        // Evaluate polynomial using Horner's method with FMA
        // p(y) = c1*y + c2*y² + c3*y³ + c4*y⁴ + c5*y⁵ + c6*y⁶ + c7*y⁷
        __m256 p = c1;
        p = _mm256_fmadd_ps(p, y, c2);  // p = p*y + c2
        p = _mm256_fmadd_ps(p, y, c3);  // p = p*y + c3
        p = _mm256_fmadd_ps(p, y, c4);  // p = p*y + c4
        p = _mm256_fmadd_ps(p, y, c5);  // p = p*y + c5
        p = _mm256_fmadd_ps(p, y, c6);  // p = p*y + c6
        p = _mm256_fmadd_ps(p, y, c7);  // p = p*y + c7

        // Compute ln(mantissa): log_m = y + y²*p(y)
        __m256 log_m = _mm256_add_ps(y, _mm256_mul_ps(z, p));

        // Step 4: Combine exponent and mantissa contributions
        // ln(x) = ln(2^e * m) = e*ln(2) + ln(m)
        __m256 result = _mm256_fmadd_ps(e, ln2, log_m);  // result = e*ln2 + log_m
        return result;
    }

    void AVX2::log_f32_avx2(const Tensor& A, Tensor& Out) {
        // Basic sanity checks
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 log: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 log: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per __m256
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vo = log256_ps(va);
            _mm256_storeu_ps(o + i, vo);
        }

        // remainder (scalar fallback)
        for (; i < n; ++i) {
            o[i] = std::log(a[i]);
        }
    }

    //////////////////////////////////////////

    //----------POW--------------
    void AVX2::pow_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU ||
            B.device_type() != DeviceType::CPU ||
            Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 pow: only CPU tensors supported");
            }

        if (A.shape() != B.shape() || A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 pow: shape mismatch");
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

            // log(a)
            __m256 v_log = log256_ps(va);

            // b * log(a)
            __m256 v_mul = _mm256_mul_ps(vb, v_log);

            // exp(b * log(a))
            __m256 v_pow = exp256_ps(v_mul);

            _mm256_storeu_ps(o + i, v_pow);
        }

        // rem
        for (; i < n; ++i) {
            o[i] = std::pow(a[i], b[i]);
        }
    }

    // ------ABS---------
    void AVX2::abs_f32_avx2(const Tensor& A, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 abs: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 abs: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        const int stride = 8;
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vabs = _mm256_and_ps(va, mask);  // clear sign bit
            _mm256_storeu_ps(o + i, vabs);
        }

        for (; i < n; ++i) {
            o[i] = std::fabs(a[i]);
        }
    }

    //-----------SQRT----------------
    void AVX2::sqrt_f32_avx2(const Tensor& A, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 sqrt: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 sqrt: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per __m256
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);

            // Optional safety: clamp negatives to 0 before sqrt
            __m256 zeros = _mm256_setzero_ps();
            __m256 mask = _mm256_cmp_ps(va, zeros, _CMP_LT_OS); // va < 0
            va = _mm256_blendv_ps(va, zeros, mask);

            __m256 vsqrt = _mm256_sqrt_ps(va);
            _mm256_storeu_ps(o + i, vsqrt);
        }

        for (; i < n; ++i) {
            float val = a[i];
            o[i] = (val < 0.0f) ? std::numeric_limits<float>::quiet_NaN() : std::sqrt(val);
        }
    }

    //-----------SIN----------------
    inline __m256 sin256_ps(__m256 x) {
        const __m256 pi     = _mm256_set1_ps(3.14159265358979323846f);
        const __m256 inv_pi2 = _mm256_set1_ps(1.0f / (2.0f * 3.14159265358979323846f));
        const __m256 two_pi = _mm256_set1_ps(2.0f * 3.14159265358979323846f);

        // Range reduction: x = x mod (2*pi)
        __m256 n = _mm256_round_ps(_mm256_mul_ps(x, inv_pi2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 xr = _mm256_sub_ps(x, _mm256_mul_ps(n, two_pi));

        // Polynomial coefficients for sin(x)
        const __m256 c3 = _mm256_set1_ps(-1.0f / 6.0f);        // -x^3/6
        const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);       // +x^5/120
        const __m256 c7 = _mm256_set1_ps(-1.0f / 5040.0f);     // -x^7/5040

        __m256 x2 = _mm256_mul_ps(xr, xr);
        __m256 x3 = _mm256_mul_ps(x2, xr);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 x7 = _mm256_mul_ps(x5, x2);

        __m256 result = xr;
        result = _mm256_add_ps(result, _mm256_mul_ps(c3, x3));
        result = _mm256_add_ps(result, _mm256_mul_ps(c5, x5));
        result = _mm256_add_ps(result, _mm256_mul_ps(c7, x7));

        return result;
    }

    void AVX2::sin_f32_avx2(const Tensor& A, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 sin: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 sin: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per __m256
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vsin = sin256_ps(va);
            _mm256_storeu_ps(o + i, vsin);
        }

        for (; i < n; ++i) {
            o[i] = std::sin(a[i]);
        }
    }

    //---------COS-------------
    inline __m256 cos256_ps(__m256 x) {
        const __m256 two_pi  = _mm256_set1_ps(6.28318530717958647692f);  // 2π
        const __m256 inv_two_pi = _mm256_set1_ps(1.0f / 6.28318530717958647692f);

        // Range reduction: x = x mod (2π)
        __m256 n = _mm256_round_ps(_mm256_mul_ps(x, inv_two_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 xr = _mm256_sub_ps(x, _mm256_mul_ps(n, two_pi));

        // Coefficients for cosine polynomial
        const __m256 c2 = _mm256_set1_ps(-1.0f / 2.0f);
        const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
        const __m256 c6 = _mm256_set1_ps(-1.0f / 720.0f);

        __m256 x2 = _mm256_mul_ps(xr, xr);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 x6 = _mm256_mul_ps(x4, x2);

        __m256 result = _mm256_set1_ps(1.0f);
        result = _mm256_add_ps(result, _mm256_mul_ps(c2, x2));
        result = _mm256_add_ps(result, _mm256_mul_ps(c4, x4));
        result = _mm256_add_ps(result, _mm256_mul_ps(c6, x6));

        return result;
    }

    void AVX2::cos_f32_avx2(const Tensor& A, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 cos: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 cos: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per __m256
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vcos = cos256_ps(va);
            _mm256_storeu_ps(o + i, vcos);
        }

        // Remainder loop (scalar)
        for (; i < n; ++i) {
            o[i] = std::cos(a[i]);
        }
    }

    //---------TAN-------------
    void AVX2::tan_f32_avx2(const Tensor& A, Tensor& Out) {
        // ==== Sanity checks ====
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 tan: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 tan: shape mismatch");
        }

        // ==== Data pointers ====
        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per __m256
        std::int64_t i = 0;

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);

            // Compute sin and cos
            __m256 vsin = sin256_ps(va);
            __m256 vcos = cos256_ps(va);

            // Avoid divide-by-zero: clamp very small cos values
            const __m256 eps = _mm256_set1_ps(1e-8f);
            __m256 sign = _mm256_and_ps(vcos, _mm256_set1_ps(-0.0f)); // extract sign
            __m256 abs_cos = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vcos);
            __m256 safe_cos = _mm256_add_ps(abs_cos, eps);
            safe_cos = _mm256_or_ps(safe_cos, sign);

            // tan(x) = sin(x) / cos(x)
            __m256 vtan = _mm256_div_ps(vsin, safe_cos);
            _mm256_storeu_ps(o + i, vtan);
        }

        // Scalar fallback for tail
        for (; i < n; ++i) {
            o[i] = std::tan(a[i]);
        }
    }

    //---------SIGMOID-------------
    void AVX2::sigmoid_f32_avx2(const Tensor& A, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 sigmoid: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 sigmoid: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8;
        std::int64_t i = 0;

        const __m256 one = _mm256_set1_ps(1.0f);

        for (; i + stride <= n; i += stride) {
            __m256 x = _mm256_loadu_ps(a + i);
            __m256 negx = _mm256_sub_ps(_mm256_setzero_ps(), x);
            __m256 exp_negx = exp256_ps(negx);
            __m256 denom = _mm256_add_ps(one, exp_negx);
            __m256 result = _mm256_div_ps(one, denom);
            _mm256_storeu_ps(o + i, result);
        }

        for (; i < n; ++i) {
            float x = a[i];
            if (x >= 0.0f) {
                float exp_neg_x = std::exp(-x);
                o[i] = 1.0f / (1.0f + exp_neg_x);
            } else {
                float exp_x = std::exp(x);
                o[i] = exp_x / (1.0f + exp_x);
            }
        }
    }

    //---------RELU-------------
    void AVX2::relu_f32_avx2(const Tensor& A, Tensor& Out) {
        if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 relu: only CPU tensors supported");
        }

        if (A.shape() != Out.shape()) {
            throw std::runtime_error("AVX2 relu: shape mismatch");
        }

        const float* a = A.data().data();
        float* o = Out.data().data();
        const std::int64_t n = static_cast<std::int64_t>(Out.numel());

        const int stride = 8; // 8 floats per AVX register
        std::int64_t i = 0;

        const __m256 zeros = _mm256_setzero_ps();

        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_max_ps(va, zeros);
            _mm256_storeu_ps(o + i, vout);
        }

        // remainder
        for (; i < n; ++i) {
            float x = a[i];
            o[i] = (x > 0.0f) ? x : 0.0f;
        }
    }

void AVX2::gemm_f32_avx2(const Tensor& A, const Tensor& B, Tensor& C) {
    // --- Sanity checks ---
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        C.device_type() != DeviceType::CPU) {
        throw std::runtime_error("AVX2 matmul: only CPU tensors supported");
    }

    if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 2) {
        throw std::runtime_error("AVX2 matmul: only supports 2D tensors");
    }

    const int M = static_cast<int>(A.shape()[0]);
    const int K = static_cast<int>(A.shape()[1]);
    const int KB = static_cast<int>(B.shape()[0]);
    const int N = static_cast<int>(B.shape()[1]);

    if (K != KB)
        throw std::runtime_error("AVX2 matmul: dimension mismatch (A.cols != B.rows)");
    if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N))
        throw std::runtime_error("AVX2 matmul: output shape mismatch");

    const float* Adata = A.data().data();
    const float* Bdata = B.data().data();
    float* Cdata = C.data().data();

    // Micro-tile sizes (tuned): 8 rows x 16 columns, K-block 64
    constexpr int TM = 8;
    constexpr int TN = 16;   // two 8-wide zmm lanes
    constexpr int TK = 64;   // K blocking for cache locality

    // Zero C tile-by-tile when storing, so no need to pre-clear C

    for (int i0 = 0; i0 < M; i0 += TM) {
        const int iMax = std::min(M, i0 + TM);
        for (int j0 = 0; j0 < N; j0 += TN) {
            const int nRemain = N - j0;
            const int n0 = std::min(8, std::max(0, nRemain));
            const int n1 = std::min(8, std::max(0, nRemain - 8));

            // Accumulators: acc[row][lane] where lane=0 -> j0..j0+7, lane=1 -> j0+8..j0+15
            __m256 acc[TM][2];
            for (int ii = 0; ii < TM; ++ii) {
                acc[ii][0] = _mm256_setzero_ps();
                acc[ii][1] = _mm256_setzero_ps();
            }

            for (int k0 = 0; k0 < K; k0 += TK) {
                const int kMax = std::min(K, k0 + TK);
                for (int k = k0; k < kMax; ++k) {
                    // Load B panels for this k and j0 with tail-safe loads
                    __m256 b0, b1;
                    if (n0 == 8) {
                        b0 = _mm256_loadu_ps(&Bdata[k * N + j0]);
                    } else {
                        alignas(32) float btmp0[8] = {0};
                        if (n0 > 0) std::memcpy(btmp0, &Bdata[k * N + j0], sizeof(float) * n0);
                        b0 = _mm256_loadu_ps(btmp0);
                    }
                    if (n1 == 8) {
                        b1 = _mm256_loadu_ps(&Bdata[k * N + j0 + 8]);
                    } else if (n1 > 0) {
                        alignas(32) float btmp1[8] = {0};
                        std::memcpy(btmp1, &Bdata[k * N + j0 + 8], sizeof(float) * n1);
                        b1 = _mm256_loadu_ps(btmp1);
                    } else {
                        b1 = _mm256_setzero_ps();
                    }

                    // Accumulate over up to TM rows
                    for (int ii = 0; ii < TM && (i0 + ii) < M; ++ii) {
                        __m256 a_brd = _mm256_broadcast_ss(&Adata[(i0 + ii) * K + k]);
                        acc[ii][0] = _mm256_fmadd_ps(a_brd, b0, acc[ii][0]);
                        if (n1 > 0) acc[ii][1] = _mm256_fmadd_ps(a_brd, b1, acc[ii][1]);
                    }
                }
            }

            // Store accumulators back to C with tail-safe stores
            for (int ii = 0; ii < TM && (i0 + ii) < M; ++ii) {
                float* Crow = &Cdata[(i0 + ii) * N + j0];

                if (n0 == 8) {
                    _mm256_storeu_ps(Crow, acc[ii][0]);
                } else if (n0 > 0) {
                    alignas(32) float ctmp0[8];
                    _mm256_storeu_ps(ctmp0, acc[ii][0]);
                    std::memcpy(Crow, ctmp0, sizeof(float) * n0);
                }

                if (n1 == 8) {
                    _mm256_storeu_ps(Crow + 8, acc[ii][1]);
                } else if (n1 > 0) {
                    alignas(32) float ctmp1[8];
                    _mm256_storeu_ps(ctmp1, acc[ii][1]);
                    std::memcpy(Crow + 8, ctmp1, sizeof(float) * n1);
                }
            }
        }
    }
}

    void AVX2::dot_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out) {
        // Sanity checks
        if (A.device_type() != DeviceType::CPU ||
            B.device_type() != DeviceType::CPU ||
            Out.device_type() != DeviceType::CPU) {
            throw std::runtime_error("AVX2 dot: only CPU tensors supported");
            }

        if (A.shape().size() != 1 || B.shape().size() != 1) {
            throw std::runtime_error("AVX2 dot: inputs must be 1D vectors");
        }
        if (A.shape()[0] != B.shape()[0]) {
            throw std::runtime_error("AVX2 dot: size mismatch");
        }
        if (!Out.shape().empty()) {
            throw std::runtime_error("AVX2 dot: output must be scalar (shape [])");
        }

        const float* a = A.data().data();
        const float* b = B.data().data();
        const std::int64_t n = static_cast<std::int64_t>(A.shape()[0]);

        __m256 vacc = _mm256_setzero_ps();
        std::int64_t i = 0;
        constexpr int stride = 8;
        for (; i + stride <= n; i += stride) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vacc = _mm256_fmadd_ps(va, vb, vacc); // FMA accumulate
        }

        // Horizontal sum of vacc
        __m128 vlow = _mm256_castps256_ps128(vacc);
        __m128 vhigh = _mm256_extractf128_ps(vacc, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        float sum = _mm_cvtss_f32(vsum);

        // Remainder
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }

        Out.data()[0] = sum;
    }
} // namespace cppgrad

