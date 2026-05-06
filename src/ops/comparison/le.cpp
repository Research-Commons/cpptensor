#include "cpptensor/ops/comparison/le.hpp"

#include "cpptensor/ops/comparison/comparison_common.hpp"

namespace cpptensor {

Tensor le(const Tensor& a, const Tensor& b) {
    return compare_tensors(a, b, std::less_equal<float>{});
}

Tensor le(const Tensor& a, float scalar) {
    return le(a, Tensor::full(a.shape(), scalar, a.device_type(), a.dtype()));
}

Tensor le(float scalar, const Tensor& b) {
    return le(Tensor::full(b.shape(), scalar, b.device_type(), b.dtype()), b);
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
