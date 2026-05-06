#include "cpptensor/ops/comparison/eq.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor eq(const Tensor& a, const Tensor& b) {
    autograd::throw_if_requires_grad(a, b, "comparison");
    return dispatchBinaryOp(a, b, OpType::Eq, CPU::eqKernel);
}

Tensor eq(const Tensor& a, float scalar) {
    return eq(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor eq(float scalar, const Tensor& b) {
    return eq(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

Tensor operator==(const Tensor& a, const Tensor& b) {
    return eq(a, b);
}

Tensor operator==(const Tensor& a, float scalar) {
    return eq(a, scalar);
}

Tensor operator==(float scalar, const Tensor& b) {
    return eq(scalar, b);
}

} // namespace cpptensor
