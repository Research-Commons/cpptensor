#include "cpptensor/ops/comparison/le.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor le(const Tensor& a, const Tensor& b) {
    autograd::throw_if_requires_grad(a, b, "comparison");
    return dispatchBinaryOp(a, b, OpType::Le, CPU::leKernel);
}

Tensor le(const Tensor& a, float scalar) {
    return le(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor le(float scalar, const Tensor& b) {
    return le(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

Tensor operator<=(const Tensor& a, const Tensor& b) {
    return le(a, b);
}

Tensor operator<=(const Tensor& a, float scalar) {
    return le(a, scalar);
}

Tensor operator<=(float scalar, const Tensor& b) {
    return le(scalar, b);
}

} // namespace cpptensor
