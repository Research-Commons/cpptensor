#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/backend/cpu_backend.h"
#include <stdexcept>

namespace cpptensor {

Tensor lt(const Tensor& a, const Tensor& b) {
    if (a.device_type() != b.device_type()) {
        throw std::runtime_error("Device mismatch in lt");
    }
    
    std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
    Tensor out = Tensor::full(out_shape, 0.0f, a.device_type());
    
    // Broadcasting stays on the generic CPU kernel; same-shape comparisons use runtime ISA dispatch.
    if (a.device_type() == DeviceType::CPU && needsBroadcast(a.shape(), b.shape())) {
        CPU::ltKernel(a, b, out);
    } else {
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
