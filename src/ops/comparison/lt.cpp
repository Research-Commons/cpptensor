#include "cpptensor/ops/comparison/lt.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor lt(const Tensor& a, const Tensor& b) {
    autograd::throw_if_requires_grad(a, b, "comparison");
    return dispatchBinaryOp(a, b, OpType::Lt, CPU::ltKernel);
}

Tensor lt(const Tensor& a, float scalar) {
    return lt(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor lt(float scalar, const Tensor& b) {
    return lt(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

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
