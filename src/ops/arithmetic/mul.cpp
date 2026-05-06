#include "cpptensor/ops/arithmetic/mul.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

Tensor operator*(const Tensor& a, const Tensor& b) {
    return dispatchBinaryOp(a, b, OpType::Mul, CPU::mulKernel);
}

Tensor operator*(const Tensor& lhs, float scalar) {
    return lhs * Tensor::full(lhs.shape(), scalar, lhs.device_type(), lhs.dtype());
}

Tensor operator*(float scalar, const Tensor& rhs) {
    return rhs * Tensor::full(rhs.shape(), scalar, rhs.device_type(), rhs.dtype());
}

} // namespace cpptensor
