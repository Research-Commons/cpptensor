#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/utils/broadcastUtils.hpp"
#include <limits>
#include <experimental/simd>

//Need to fix these ( reduce so much repetitions. dont care that much rn as its just basic C++ kernels not even being called mostly)

void cpptensor::CPU::addKernel(const Tensor &A, const Tensor &B, Tensor &out) {
    const auto& a_sh = A.shape();
    const auto& b_sh = B.shape();
    const auto& out_sh = out.shape();
    size_t n = out_sh.size();

    // Build padded shapes
    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    size_t na = a_sh.size(), nb = b_sh.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n-na ? 1 : a_sh[i-(n-na)]);
        b_pad[i] = (i < n-nb ? 1 : b_sh[i-(n-nb)]);
    }

    // Compute prefix (stride) for A, B, and output
    std::vector<size_t> strideA(n), strideB(n), strideOut(n);
    if (n > 0) {
        strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
    }
    for (int i = (int)n-2; i >= 0; --i) {
        strideA[i]   = strideA[i+1] * a_pad[i+1];
        strideB[i]   = strideB[i+1] * b_pad[i+1];
        strideOut[i] = strideOut[i+1] * out_sh[i+1];
    }

    // Elementwise loop over output tensor
    size_t total = 1;
    for (size_t dim : out_sh) total *= dim;
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idxA = 0, idxB = 0;
        // Convert flat index to multi-index, then to each input index
        for (size_t dim = 0; dim < n; ++dim) {
            size_t i = (pos / strideOut[dim]) % out_sh[dim];
            if (a_pad[dim] != 1) idxA += i * strideA[dim];
            if (b_pad[dim] != 1) idxB += i * strideB[dim];
        }
        out.data()[pos] = A.data()[idxA] + B.data()[idxB];
    }
}

void cpptensor::CPU::mulKernel(const Tensor &A, const Tensor &B, Tensor &out) {
    const auto& a_sh = A.shape();
    const auto& b_sh = B.shape();
    const auto& out_sh = out.shape();
    size_t n = out_sh.size();

    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    size_t na = a_sh.size(), nb = b_sh.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n-na ? 1 : a_sh[i-(n-na)]);
        b_pad[i] = (i < n-nb ? 1 : b_sh[i-(n-nb)]);
    }

    std::vector<size_t> strideA(n), strideB(n), strideOut(n);
    if (n > 0) {
        strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
    }
    for (int i = (int)n-2; i >= 0; --i) {
        strideA[i]   = strideA[i+1] * a_pad[i+1];
        strideB[i]   = strideB[i+1] * b_pad[i+1];
        strideOut[i] = strideOut[i+1] * out_sh[i+1];
    }

    size_t total = 1;
    for (size_t dim : out_sh) total *= dim;
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idxA = 0, idxB = 0;
        for (size_t dim = 0; dim < n; ++dim) {
            size_t i = (pos / strideOut[dim]) % out_sh[dim];
            if (a_pad[dim] != 1) idxA += i * strideA[dim];
            if (b_pad[dim] != 1) idxB += i * strideB[dim];
        }
        out.data()[pos] = A.data()[idxA] * B.data()[idxB];
    }
}


void cpptensor::CPU::subKernel(const Tensor &A, const Tensor &B, Tensor &out) {
    const auto& a_sh = A.shape();
    const auto& b_sh = B.shape();
    const auto& out_sh = out.shape();
    size_t n = out_sh.size();

    // Build padded shapes
    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    size_t na = a_sh.size(), nb = b_sh.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n-na ? 1 : a_sh[i-(n-na)]);
        b_pad[i] = (i < n-nb ? 1 : b_sh[i-(n-nb)]);
    }

    // Compute prefix (stride) for A, B, and output
    std::vector<size_t> strideA(n), strideB(n), strideOut(n);
    if (n > 0) {
        strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
    }
    for (int i = (int)n-2; i >= 0; --i) {
        strideA[i]   = strideA[i+1] * a_pad[i+1];
        strideB[i]   = strideB[i+1] * b_pad[i+1];
        strideOut[i] = strideOut[i+1] * out_sh[i+1];
    }

    // Elementwise loop over output tensor
    size_t total = 1;
    for (size_t dim : out_sh) total *= dim;
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idxA = 0, idxB = 0;
        // Convert flat index to multi-index, then to each input index
        for (size_t dim = 0; dim < n; ++dim) {
            size_t i = (pos / strideOut[dim]) % out_sh[dim];
            if (a_pad[dim] != 1) idxA += i * strideA[dim];
            if (b_pad[dim] != 1) idxB += i * strideB[dim];
        }
        out.data()[pos] = A.data()[idxA] - B.data()[idxB];
    }
}

void cpptensor::CPU::divKernel(const Tensor &A, const Tensor &B, Tensor &out) {
    const auto& a_sh = A.shape();
    const auto& b_sh = B.shape();
    const auto& out_sh = out.shape();
    size_t n = out_sh.size();

    // Build padded shapes
    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    size_t na = a_sh.size(), nb = b_sh.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n-na ? 1 : a_sh[i-(n-na)]);
        b_pad[i] = (i < n-nb ? 1 : b_sh[i-(n-nb)]);
    }

    // Compute prefix (stride) for A, B, and output
    std::vector<size_t> strideA(n), strideB(n), strideOut(n);
    if (n > 0) {
        strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
    }
    for (int i = (int)n-2; i >= 0; --i) {
        strideA[i]   = strideA[i+1] * a_pad[i+1];
        strideB[i]   = strideB[i+1] * b_pad[i+1];
        strideOut[i] = strideOut[i+1] * out_sh[i+1];
    }

    // Elementwise loop over output tensor
    size_t total = 1;
    for (size_t dim : out_sh) total *= dim;
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idxA = 0, idxB = 0;
        // Convert flat index to multi-index, then to each input index
        for (size_t dim = 0; dim < n; ++dim) {
            size_t i = (pos / strideOut[dim]) % out_sh[dim];
            if (a_pad[dim] != 1) idxA += i * strideA[dim];
            if (b_pad[dim] != 1) idxB += i * strideB[dim];
        }
        float denom = B.data()[idxB];
        if (denom == 0.0f) {
            out.data()[pos] = std::numeric_limits<float>::infinity();
        } else {
            out.data()[pos] = A.data()[idxA] / denom;
        }
    }
}

void cpptensor::CPU::expKernel(const Tensor& A, Tensor& Out) {
    // ==== Sanity checks ====
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("exp_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("exp_f32_generic: shape mismatch");
    }

    // ==== Get raw data pointers ====
    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    // ==== Elementwise exponential ====
    for (std::int64_t i = 0; i < n; ++i) {
        out_data[i] = std::exp(in_data[i]);
    }
}

void cpptensor::CPU::logKernel(const Tensor& A, Tensor& Out) {
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("log_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("log_f32_generic: shape mismatch");
    }

    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    for (std::int64_t i = 0; i < n; ++i) {
        float val = in_data[i];
        if (val <= 0.0f) {
            out_data[i] = -INFINITY;
        } else {
            out_data[i] = std::log(val);
        }
    }
}

void cpptensor::CPU::powKernel(const Tensor& A, const Tensor& B, Tensor& Out) {
    if (A.device_type() != DeviceType::CPU ||
        B.device_type() != DeviceType::CPU ||
        Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("pow_f32_generic: only CPU tensors supported");
        }

    if (A.shape() != B.shape() || A.shape() != Out.shape()) {
        throw std::runtime_error("pow_f32_generic: shape mismatch");
    }

    const float* a_data = A.data().data();
    const float* b_data = B.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    for (std::int64_t i = 0; i < n; ++i) {
        float base = a_data[i];
        float exp  = b_data[i];

        if (base < 0.0f) {
            // NaN for non-integer powers of negative numbers
            out_data[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            out_data[i] = std::pow(base, exp);
        }
    }
}

void cpptensor::CPU::absKernel(const Tensor& A, Tensor& Out) {
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("abs_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("abs_f32_generic: shape mismatch");
    }

    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    for (std::int64_t i = 0; i < n; ++i) {
        out_data[i] = std::fabs(in_data[i]);
    }
}

void cpptensor::CPU::sqrtKernel(const Tensor& A, Tensor& Out) {
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("sqrt_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("sqrt_f32_generic: shape mismatch");
    }

    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    for (std::int64_t i = 0; i < n; ++i) {
        float val = in_data[i];
        if (val < 0.0f) {
            out_data[i] = std::numeric_limits<float>::quiet_NaN(); // domain error
        } else {
            out_data[i] = std::sqrt(val);
        }
    }
}

void cpptensor::CPU::sinKernel(const Tensor& A, Tensor& Out) {
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("sin_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("sin_f32_generic: shape mismatch");
    }

    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    for (std::int64_t i = 0; i < n; ++i) {
        out_data[i] = std::sin(in_data[i]);
    }
}

void cpptensor::CPU::cosKernel(const Tensor& A, Tensor& Out) {
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("cos_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("cos_f32_generic: shape mismatch");
    }

    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    for (std::int64_t i = 0; i < n; ++i) {
        out_data[i] = std::cos(in_data[i]);
    }
}

void cpptensor::CPU::tanKernel(const Tensor& A, Tensor& Out) {
    // ==== Sanity checks ====
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("tan_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("tan_f32_generic: shape mismatch");
    }

    // ==== Get raw data ====
    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    // ==== Elementwise tangent ====
    for (std::int64_t i = 0; i < n; ++i) {
        out_data[i] = std::tan(in_data[i]);
    }
}

void cpptensor::CPU::sigmoidKernel(const Tensor& A, Tensor& Out) {
    // ==== Sanity checks ====
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("sigmoid_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("sigmoid_f32_generic: shape mismatch");
    }

    // ==== Get raw pointers ====
    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    // ==== Forward pass: numerically stable sigmoid ====
    for (std::int64_t i = 0; i < n; ++i) {
        float x = in_data[i];
        if (x >= 0.0f) {
            float exp_neg_x = std::exp(-x);
            out_data[i] = 1.0f / (1.0f + exp_neg_x);
        } else {
            float exp_x = std::exp(x);
            out_data[i] = exp_x / (1.0f + exp_x);
        }
    }
}

void cpptensor::CPU::reluKernel(const Tensor& A, Tensor& Out) {
    // ==== Sanity checks ====
    if (A.device_type() != DeviceType::CPU || Out.device_type() != DeviceType::CPU) {
        throw std::runtime_error("relu_f32_generic: only CPU tensors supported");
    }

    if (A.shape() != Out.shape()) {
        throw std::runtime_error("relu_f32_generic: shape mismatch");
    }

    // ==== Get raw data pointers ====
    const float* in_data = A.data().data();
    float* out_data = Out.data().data();
    const std::int64_t n = static_cast<std::int64_t>(Out.numel());

    // ==== Compute ReLU elementwise ====
    for (std::int64_t i = 0; i < n; ++i) {
        float x = in_data[i];
        out_data[i] = (x > 0.0f) ? x : 0.0f;
    }
}

void cpptensor::CPU::gemmf32kernel(const Tensor &A, const Tensor &B, Tensor &Out) {

    //-----this is way too slow. not gonna work. commenting----
    // const int64_t M = A.shape()[0];
    // const int64_t K = A.shape()[1];
    // const int64_t KB = B.shape()[0];
    // const int64_t N = B.shape()[1];
    //
    // if (K != KB || Out.shape()[0] != M || Out.shape()[1] != N) {
    //     throw std::runtime_error("CPU Matmul: shape mismatch (A: MxK, B: KxN, C: MxN)");
    // }
    //
    // const float* a = A.data().data();
    // const float* b = B.data().data();
    // float* c = Out.data().data();
    //
    // for (int64_t i = 0; i < M; ++i) {
    //     for (int64_t j = 0; j < N; ++j) {
    //         float sum = 0.0f;
    //         for (int64_t k = 0; k < K; ++k) {
    //             sum += a[i * K + k] * b[k * N + j];
    //         }
    //         c[i * N + j] = sum;
    //     }
    // }

    //-----much faster cache friendly version with tiling and accumulator(fma)----

    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t KB = B.shape()[0];
    const int64_t N = B.shape()[1];

    if (K != KB || Out.shape()[0] != M || Out.shape()[1] != N) {
        throw std::runtime_error("CPU GEMM: shape mismatch (A: MxK, B: KxN, C: MxN)");
    }

    const float* a = A.data().data();
    const float* b = B.data().data();
    float* c = Out.data().data();

    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 32;

    std::fill(c, c + M * N, 0.0f);

    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {

                int64_t iMax = std::min<int64_t>(M, i0 + TILE_M);
                int64_t jMax = std::min<int64_t>(N, j0 + TILE_N);
                int64_t kMax = std::min<int64_t>(K, k0 + TILE_K);

                for (int64_t i = i0; i < iMax; ++i) {
                    for (int64_t k = k0; k < kMax; ++k) {
                        float a_ik = a[i * K + k];
                        for (int64_t j = j0; j < jMax; ++j) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}




