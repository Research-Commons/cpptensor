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

void cpptensor::CPU::addBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out, Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    // accumulate in padded buffers
    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }
        float v = gout[pos];
        a_padded[idx_a] += v;
        b_padded[idx_b] += v;
    }

    // squeeze to original shapes
    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

void cpptensor::CPU::mulBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                    Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    const auto &Adata = A.data();
    const auto &Bdata = B.data();

    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }
        float gout_v = gout[pos];
        a_padded[idx_a] += gout_v * Bdata[idx_b];
        b_padded[idx_b] += gout_v * Adata[idx_a];
    }

    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

void cpptensor::CPU::subBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                     Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    // accumulate in padded buffers
    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }
        float v = gout[pos];
        a_padded[idx_a] += v;    // grad wrt A is +grad_out
        b_padded[idx_b] += -v;   // grad wrt B is -grad_out
    }

    // squeeze to original shapes
    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}

void cpptensor::CPU::divBackwardKernel(const Tensor &A, const Tensor &B, const Tensor &grad_out,
                                     Tensor &grad_a, Tensor &grad_b) {
    auto a_sh = A.shape();
    auto b_sh = B.shape();
    auto out_sh = compute_broadcast_shape(a_sh, b_sh);

    int n = static_cast<int>(out_sh.size());
    auto a_pad = pad_shape_right(a_sh, n);
    auto b_pad = pad_shape_right(b_sh, n);

    auto stride_out = compute_strides(out_sh);
    auto stride_a = compute_strides(a_pad);
    auto stride_b = compute_strides(b_pad);

    size_t out_total = numel(out_sh);
    size_t a_p_total = numel(a_pad);
    size_t b_p_total = numel(b_pad);

    std::vector<float> a_padded(a_p_total, 0.0f);
    std::vector<float> b_padded(b_p_total, 0.0f);

    const auto &gout = grad_out.data();
    const auto &Adata = A.data();
    const auto &Bdata = B.data();

    for (size_t pos = 0; pos < out_total; ++pos) {
        size_t idx_a = 0, idx_b = 0;
        for (int d = 0; d < n; ++d) {
            size_t coord = (pos / stride_out[(size_t)d]) % out_sh[(size_t)d];
            if (a_pad[(size_t)d] != 1) idx_a += coord * stride_a[(size_t)d];
            if (b_pad[(size_t)d] != 1) idx_b += coord * stride_b[(size_t)d];
        }

        float gout_v = gout[pos];
        float aval = Adata[idx_a];
        float bval = Bdata[idx_b];

        // grad wrt A: grad_out / b  (handle b==0)
        float contrib_a = (bval == 0.0f) ? std::numeric_limits<float>::infinity() : (gout_v / bval);
        // grad wrt B: -grad_out * a / (b*b)  (handle denom==0)
        float denom = bval * bval;
        float contrib_b = (denom == 0.0f) ? -std::numeric_limits<float>::infinity()
                                          : (-gout_v * aval / denom);

        a_padded[idx_a] += contrib_a;
        b_padded[idx_b] += contrib_b;
    }

    auto a_squeezed = squeeze_padded_to_unpadded(a_padded, a_pad, a_sh);
    auto b_squeezed = squeeze_padded_to_unpadded(b_padded, b_pad, b_sh);

    grad_a.data() = std::move(a_squeezed);
    grad_b.data() = std::move(b_squeezed);
}



