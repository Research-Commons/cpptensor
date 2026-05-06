#include "cpptensor/ops/comparison/gt.hpp"

#include "cpptensor/ops/comparison/comparison_common.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor gt(const Tensor& a, const Tensor& b) {
    autograd::throw_if_requires_grad(a, b, "comparison");
    return compare_tensors(a, b, std::greater<float>{});
}

Tensor gt(const Tensor& a, float scalar) {
    return gt(a, Tensor::full(a.shape(), scalar, a.device_type(), a.dtype()));
}

Tensor gt(float scalar, const Tensor& b) {
    return gt(Tensor::full(b.shape(), scalar, b.device_type(), b.dtype()), b);
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
