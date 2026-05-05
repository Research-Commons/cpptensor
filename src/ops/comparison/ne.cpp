#include "cpptensor/ops/comparison/ne.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

Tensor ne(const Tensor& a, const Tensor& b) {
    return dispatchBinaryOp(a, b, OpType::Ne, CPU::neKernel);
}

Tensor ne(const Tensor& a, float scalar) {
    return ne(a, Tensor::full(a.shape(), scalar, a.device_type()));
}

Tensor ne(float scalar, const Tensor& b) {
    return ne(Tensor::full(b.shape(), scalar, b.device_type()), b);
}

Tensor operator!=(const Tensor& a, const Tensor& b) {
    return ne(a, b);
}

Tensor operator!=(const Tensor& a, float scalar) {
    return ne(a, scalar);
}

Tensor operator!=(float scalar, const Tensor& b) {
    return ne(scalar, b);
}

} // namespace cpptensor
