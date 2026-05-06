#include "cpptensor/ops/arithmetic/add.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor operator+(const Tensor& a, const Tensor& b) {
    Tensor out = dispatchBinaryOp(a, b, OpType::Add, CPU::addKernel);

    const bool requires_grad = a.requires_grad() || b.requires_grad();
    out.set_requires_grad(requires_grad);
    if (!requires_grad) {
        return out;
    }

    const auto out_shape = out.shape();
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const auto a_impl = a.impl();
    const auto b_impl = b.impl();

    out.impl()->set_grad_fn([a_impl, b_impl, out_shape, a_shape, b_shape](const std::vector<float>& grad_out) {
        if (a_impl->requires_grad()) {
            a_impl->backward(autograd::reduce_sum_to_shape(grad_out, out_shape, a_shape));
        }
        if (b_impl->requires_grad()) {
            b_impl->backward(autograd::reduce_sum_to_shape(grad_out, out_shape, b_shape));
        }
    });

    return out;
}

Tensor operator+(const Tensor& lhs, float scalar) {
    return lhs + Tensor::full(lhs.shape(), scalar, lhs.device_type(), lhs.dtype());
}

Tensor operator+(float scalar, const Tensor& rhs) {
    return rhs + Tensor::full(rhs.shape(), scalar, rhs.device_type(), rhs.dtype());
}

} // namespace cpptensor
