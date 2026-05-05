#include "cpptensor/ops/arithmetic/add.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

Tensor operator+(const Tensor& a, const Tensor& b) {
    return dispatchBinaryOp(a, b, OpType::Add, CPU::addKernel);
}

Tensor operator+(const Tensor& lhs, float scalar) {
    return lhs + Tensor::full(lhs.shape(), scalar, lhs.device_type());
}

Tensor operator+(float scalar, const Tensor& rhs) {
    return rhs + Tensor::full(rhs.shape(), scalar, rhs.device_type());
}

} // namespace cpptensor
