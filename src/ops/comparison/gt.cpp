#include "cpptensor/ops/comparison/gt.hpp"
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/backend/cpu_backend.h"
#include <stdexcept>

namespace cpptensor {

Tensor gt(const Tensor& a, const Tensor& b) {
    if (a.device_type() != b.device_type()) {
        throw std::runtime_error("Device mismatch in gt");
    }
    
    std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
    Tensor out = Tensor::full(out_shape, 0.0f, a.device_type());
    const Tensor lhs = materialize_for_backend_input(a);
    const Tensor rhs = materialize_for_backend_input(b);
    
    // Broadcasting stays on the generic CPU kernel; same-shape comparisons use runtime ISA dispatch.
    if (a.device_type() == DeviceType::CPU && needsBroadcast(a.shape(), b.shape())) {
        CPU::gtKernel(lhs, rhs, out);
    } else {
        KernelRegistry::instance()
            .getKernel(OpType::Gt, a.device_type())(lhs, rhs, out);
    }
    
    return out;
}

Tensor gt(const Tensor& a, float scalar) {
    return gt(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor gt(float scalar, const Tensor& b) {
    return gt(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

// Operator overloads
Tensor operator>(const Tensor& a, const Tensor& b) {
    return gt(a, b);
}

Tensor operator>(const Tensor& a, float scalar) {
    return gt(a, scalar);
}

Tensor operator>(float scalar, const Tensor& b) {
    return gt(scalar, b);
}

} // namespace cpptensor
