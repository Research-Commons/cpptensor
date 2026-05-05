#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/backend/cpu_backend.h"
#ifdef BUILD_AVX2
#include "cpptensor/backend/isa/avx2.hpp"
#endif
#include <stdexcept>

namespace cpptensor {

Tensor lt(const Tensor& a, const Tensor& b) {
    if (a.device_type() != b.device_type()) {
        throw std::runtime_error("Device mismatch in lt");
    }
    
    std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
    Tensor out = Tensor::full(out_shape, 0.0f, a.device_type());
    
    // Hybrid dispatch: use AVX2 for same-shape, CPU generic for broadcasting
    if (a.device_type() == DeviceType::CPU) {
        if (needsBroadcast(a.shape(), b.shape())) {
            // Broadcasting needed - use CPU generic kernel
            CPU::ltKernel(a, b, out);
        } else {
#ifdef BUILD_AVX2
            // Same shape - use AVX2 fast path when it is built in.
            AVX2::lt_f32_avx2(a, b, out);
#else
            CPU::ltKernel(a, b, out);
#endif
        }
    } else {
        // For non-CPU devices, fall back to kernel registry
        KernelRegistry::instance()
            .getKernel(OpType::Lt, a.device_type())(a, b, out);
    }
    
    return out;
}

Tensor lt(const Tensor& a, float scalar) {
    return lt(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor lt(float scalar, const Tensor& b) {
    return lt(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

// Operator overloads
Tensor operator<(const Tensor& a, const Tensor& b) {
    return lt(a, b);
}

Tensor operator<(const Tensor& a, float scalar) {
    return lt(a, scalar);
}

Tensor operator<(float scalar, const Tensor& b) {
    return lt(scalar, b);
}

} // namespace cpptensor
