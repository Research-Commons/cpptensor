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

    // fast 8-wide exp approximation (Cephes-style)
    inline __m256 exp256_ps(__m256 x) {
        const __m256 ln2_inv = _mm256_set1_ps(1.44269504088896341f); // 1/ln(2)
        const __m256 ln2     = _mm256_set1_ps(0.69314718056f);       // ln(2)
        const __m256 max_x   = _mm256_set1_ps(88.3762626647949f);
        const __m256 min_x   = _mm256_set1_ps(-88.3762626647949f);

        // clamp
        x = _mm256_min_ps(x, max_x);
        x = _mm256_max_ps(x, min_x);

        // n = round(x / ln2)
        __m256 fx = _mm256_mul_ps(x, ln2_inv);
        fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2
        __m256 n_ln2 = _mm256_mul_ps(fx, ln2);
        __m256 r = _mm256_sub_ps(x, n_ln2);

        // polynomial approximation for exp(r)
        const __m256 c1 = _mm256_set1_ps(1.9875691500E-4f);
        const __m256 c2 = _mm256_set1_ps(1.3981999507E-3f);
        const __m256 c3 = _mm256_set1_ps(8.3334519073E-3f);
        const __m256 c4 = _mm256_set1_ps(4.1665795894E-2f);
        const __m256 c5 = _mm256_set1_ps(1.6666665459E-1f);
        const __m256 c6 = _mm256_set1_ps(5.0000001201E-1f);

        __m256 r2 = _mm256_mul_ps(r, r);

        __m256 y = _mm256_add_ps(c1, _mm256_mul_ps(r, c2));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, c3));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, _mm256_mul_ps(r, c4)));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, _mm256_mul_ps(r2, c5)));
        y = _mm256_add_ps(y, _mm256_mul_ps(r2, _mm256_mul_ps(r2, _mm256_mul_ps(r, c6))));
        y = _mm256_add_ps(y, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

        // 2^n scaling using float exponent bits
        __m256i mm = _mm256_cvtps_epi32(fx);
        mm = _mm256_add_epi32(mm, _mm256_set1_epi32(127)); // bias
        mm = _mm256_slli_epi32(mm, 23);                    // shift into exponent
        __m256 pow2n = _mm256_castsi256_ps(mm);

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

    inline __m256 log256_ps(__m256 x) {
        // Constants
        const __m256 one     = _mm256_set1_ps(1.0f);
        const __m256 halff   = _mm256_set1_ps(0.5f);
        const __m256 ln2     = _mm256_set1_ps(0.69314718056f);
        const __m256 min_norm_pos = _mm256_set1_ps(1.17549435e-38f); // FLT_MIN

        // Clamp input to avoid log(0)
        x = _mm256_max_ps(x, min_norm_pos);

        // Extract exponent and mantissa
        __m256i xi = _mm256_castps_si256(x);
        __m256i emm0 = _mm256_srli_epi32(xi, 23);
        emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(127));

        // Set mantissa to [1, 2)
        xi = _mm256_and_si256(xi, _mm256_set1_epi32(0x7FFFFF));
        xi = _mm256_or_si256(xi, _mm256_castps_si256(_mm256_set1_ps(0.5f)));
        x = _mm256_castsi256_ps(xi);

        __m256 e = _mm256_cvtepi32_ps(emm0);

        // Polynomial approximation of log(mantissa)
        // log(x) ≈ y + y^2 * P(y), where y = x - 1
        x = _mm256_sub_ps(x, one);
        __m256 y = x;
        __m256 z = _mm256_mul_ps(y, y);

        // Coefficients for log(1+y)
        const __m256 c1 = _mm256_set1_ps(0.666666666666735130e+0f);
        const __m256 c2 = _mm256_set1_ps(0.399999999994094190e+0f);
        const __m256 c3 = _mm256_set1_ps(0.285714287436623910e+0f);
        const __m256 c4 = _mm256_set1_ps(0.222221984321497840e+0f);
        const __m256 c5 = _mm256_set1_ps(0.181835721616180500e+0f);
        const __m256 c6 = _mm256_set1_ps(0.153138376992093730e+0f);
        const __m256 c7 = _mm256_set1_ps(0.147981986051165860e+0f);

        __m256 p = c1;
        p = _mm256_fmadd_ps(p, y, c2);
        p = _mm256_fmadd_ps(p, y, c3);
        p = _mm256_fmadd_ps(p, y, c4);
        p = _mm256_fmadd_ps(p, y, c5);
        p = _mm256_fmadd_ps(p, y, c6);
        p = _mm256_fmadd_ps(p, y, c7);

        __m256 log_m = _mm256_add_ps(y, _mm256_mul_ps(z, p));

        // Combine exponent and mantissa: log(x) = e*ln(2) + log_m
        __m256 result = _mm256_fmadd_ps(e, ln2, log_m);
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

void AVX2::matmul_f32_avx2(const Tensor& A, const Tensor& B, Tensor& C) {
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
    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::runtime_error("AVX2 matmul: output shape mismatch");

    const float* Adata = A.data().data();
    const float* Bdata = B.data().data();
    float* Cdata = C.data().data();

    // zero the output (row-major)
    std::memset(Cdata, 0, sizeof(float) * size_t(M) * N);

    constexpr int BLOCK = 8;      // vector width
    constexpr int BLOCK_X = 4;    // number of 8-wide accumulators per tile
    constexpr int BLOCK_Y = 4;    // number of rows per tile

    // We'll use Bfm as a pointer reinterpret to __m256* view of B (column-packed)
    // For simplicity we treat B row-major directly, loading as needed.

    for (int y = 0; y < M; y += BLOCK_Y) {
        for (int x = 0; x < N; x += BLOCK * BLOCK_X) {

            __m256 acc[BLOCK_Y][BLOCK_X];
            for (int iy = 0; iy < BLOCK_Y; ++iy)
                for (int ix = 0; ix < BLOCK_X; ++ix)
                    acc[iy][ix] = _mm256_setzero_ps();

            // Main K loop
            for (int k = 0; k < K; ++k) {
                for (int iy = 0; iy < BLOCK_Y && (y + iy) < M; ++iy) {
                    __m256 ta = _mm256_broadcast_ss(&Adata[(y + iy) * K + k]);

                    for (int ix = 0; ix < BLOCK_X; ++ix) {
                        int xoff = x + ix * BLOCK;
                        if (xoff >= N) break;

                        // safe load of B[k, xoff : xoff+8]
                        alignas(32) float btmp[BLOCK] = {0};
                        int remain = std::min(BLOCK, N - xoff);
                        std::memcpy(btmp, &Bdata[k * N + xoff], remain * sizeof(float));
                        __m256 bv = _mm256_loadu_ps(btmp);

                        acc[iy][ix] = _mm256_fmadd_ps(ta, bv, acc[iy][ix]);
                    }
                }
            }

            // store accumulators
            for (int iy = 0; iy < BLOCK_Y && (y + iy) < M; ++iy) {
                for (int ix = 0; ix < BLOCK_X; ++ix) {
                    int xoff = x + ix * BLOCK;
                    if (xoff >= N) break;

                    alignas(32) float ctmp[BLOCK];
                    _mm256_storeu_ps(ctmp, acc[iy][ix]);

                    int remain = std::min(BLOCK, N - xoff);
                    std::memcpy(&Cdata[(y + iy) * N + xoff], ctmp, remain * sizeof(float));
                }
            }
        }
    }
}
} // namespace cppgrad

