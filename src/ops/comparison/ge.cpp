#include "cpptensor/ops/comparison/ge.hpp"

#include "cpptensor/ops/comparison/comparison_common.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor ge(const Tensor& a, const Tensor& b) {
    autograd::throw_if_requires_grad(a, b, "comparison");
    return compare_tensors(a, b, std::greater_equal<float>{});
}

Tensor ge(const Tensor& a, float scalar) {
    return ge(a, Tensor::full(a.shape(), scalar, a.device_type(), a.dtype()));
}

Tensor ge(float scalar, const Tensor& b) {
    return ge(Tensor::full(b.shape(), scalar, b.device_type(), b.dtype()), b);
}

Tensor operator>=(const Tensor& a, const Tensor& b) {
    return ge(a, b);
}

Tensor operator>=(const Tensor& a, float scalar) {
    return ge(a, scalar);
}

Tensor operator>=(float scalar, const Tensor& b) {
    return ge(scalar, b);
}

} // namespace cpptensor
