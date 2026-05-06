#include "cpptensor/ops/comparison/ne.hpp"

#include "cpptensor/ops/comparison/comparison_common.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor ne(const Tensor& a, const Tensor& b) {
    autograd::throw_if_requires_grad(a, b, "comparison");
    return compare_tensors(a, b, std::not_equal_to<float>{});
}

Tensor ne(const Tensor& a, float scalar) {
    return ne(a, Tensor::full(a.shape(), scalar, a.device_type(), a.dtype()));
}

Tensor ne(float scalar, const Tensor& b) {
    return ne(Tensor::full(b.shape(), scalar, b.device_type(), b.dtype()), b);
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
