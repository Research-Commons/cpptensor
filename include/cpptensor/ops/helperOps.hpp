#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cpptensor/dispatcher/kernelRegistry.h"

namespace cpptensor {

inline std::string formatShape(const std::vector<size_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

inline const char* deviceTypeName(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
    }

    return "Unknown";
}

inline Tensor materialize_for_backend_input(const Tensor& tensor) {
    if (tensor.is_contiguous()) {
        return tensor;
    }
    return tensor.contiguous();
}

// Helper: compute broadcasted output shape for two input shapes.
inline std::vector<size_t> computeBroadcastShape(const std::vector<size_t>& a,
                                                 const std::vector<size_t>& b) {
    const size_t na = a.size();
    const size_t nb = b.size();
    const size_t n = std::max(na, nb);
    std::vector<size_t> out(n);

    for (size_t i = 0; i < n; ++i) {
        const size_t dimA = (i < n - na) ? 1 : a[i - (n - na)];
        const size_t dimB = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (dimA != dimB && dimA != 1 && dimB != 1) {
            throw std::runtime_error(
                "Binary op operands with shapes " + formatShape(a) +
                " and " + formatShape(b) +
                " are not broadcastable at aligned axis " + std::to_string(i) +
                " (" + std::to_string(dimA) + " vs " + std::to_string(dimB) + ")");
        }
        out[i] = std::max(dimA, dimB);
    }

    return out;
}

// Helper: check if two shapes require broadcasting (used for hybrid SIMD dispatch).
inline bool needsBroadcast(const std::vector<size_t>& shape_a,
                           const std::vector<size_t>& shape_b) {
    if (shape_a.size() != shape_b.size()) {
        return true;
    }
    for (size_t i = 0; i < shape_a.size(); ++i) {
        if (shape_a[i] != shape_b[i]) {
            return true;
        }
    }
    return false;
}

struct BinaryOpContext {
    std::vector<size_t> output_shape;
    DeviceType device;
    bool use_cpu_broadcast_kernel;
    DType output_dtype;
};

inline BinaryOpContext prepareBinaryOp(const Tensor& lhs, const Tensor& rhs) {
    const DeviceType lhs_device = lhs.device_type();
    const DeviceType rhs_device = rhs.device_type();
    if (lhs_device != rhs_device) {
        throw std::runtime_error(
            "Binary op requires matching devices, got lhs=" + std::string(deviceTypeName(lhs_device)) +
            " and rhs=" + std::string(deviceTypeName(rhs_device)));
    }

    const auto lhs_shape = lhs.shape();
    const auto rhs_shape = rhs.shape();

    const DType lhs_dtype = lhs.dtype();
    const DType rhs_dtype = rhs.dtype();
    const DType promoted_dtype = promote_dtype(lhs_dtype, rhs_dtype);
    if (promoted_dtype != DType::FLOAT32) {
        throw std::runtime_error(
            "Binary op kernels currently support float32 compute only; got lhs dtype " +
            std::string(dtype_name(lhs_dtype)) + " and rhs dtype " + std::string(dtype_name(rhs_dtype)) +
            ". Cast with astype(float32) before arithmetic.");
    }

    return BinaryOpContext{
        computeBroadcastShape(lhs_shape, rhs_shape),
        lhs_device,
        lhs_device == DeviceType::CPU && needsBroadcast(lhs_shape, rhs_shape),
        promoted_dtype,
    };
}

inline Tensor allocateBinaryOpOutput(const BinaryOpContext& context) {
    return Tensor::full(context.output_shape, 0.0f, context.device, context.output_dtype);
}

template <typename CpuBroadcastKernel>
inline Tensor dispatchBinaryOp(const Tensor& lhs,
                               const Tensor& rhs,
                               OpType op,
                               CpuBroadcastKernel&& cpu_broadcast_kernel) {
    const BinaryOpContext context = prepareBinaryOp(lhs, rhs);
    Tensor out = allocateBinaryOpOutput(context);
    const Tensor prepared_lhs = materialize_for_backend_input(lhs);
    const Tensor prepared_rhs = materialize_for_backend_input(rhs);

    if (context.use_cpu_broadcast_kernel) {
        std::forward<CpuBroadcastKernel>(cpu_broadcast_kernel)(prepared_lhs, prepared_rhs, out);
    } else {
        KernelRegistry::instance().getKernel(op, context.device)(prepared_lhs, prepared_rhs, out);
    }

    return out;
}

} // namespace cpptensor
