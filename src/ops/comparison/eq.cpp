#include "cpptensor/ops/comparison/eq.hpp"

#include "cpptensor/ops/comparison/comparison_common.hpp"

namespace cpptensor {

Tensor eq(const Tensor& a, const Tensor& b) {
    return compare_tensors(a, b, std::equal_to<float>{});
}

Tensor eq(const Tensor& a, float scalar) {
    return eq(a, Tensor::full(a.shape(), scalar, a.device_type(), a.dtype()));
}

Tensor eq(float scalar, const Tensor& b) {
    return eq(Tensor::full(b.shape(), scalar, b.device_type(), b.dtype()), b);
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
