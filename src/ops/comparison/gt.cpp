#include "cpptensor/ops/comparison/gt.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

Tensor gt(const Tensor& a, const Tensor& b) {
    return dispatchBinaryOp(a, b, OpType::Gt, CPU::gtKernel);
}

Tensor gt(const Tensor& a, float scalar) {
    return gt(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor gt(float scalar, const Tensor& b) {
    return gt(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

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
