#include "cpptensor/ops/arithmetic/mul.hpp"

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

Tensor operator*(const Tensor& a, const Tensor& b) {
    Tensor out = dispatchBinaryOp(a, b, OpType::Mul, CPU::mulKernel);

    const bool requires_grad = a.requires_grad() || b.requires_grad();
    out.set_requires_grad(requires_grad);
    if (!requires_grad) {
        return out;
    }

    const auto out_shape = out.shape();
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const auto a_data = a.data();
    const auto b_data = b.data();
    const auto a_impl = a.impl();
    const auto b_impl = b.impl();

    out.impl()->set_grad_fn([a_impl, b_impl, out_shape, a_shape, b_shape, a_data, b_data]
                            (const std::vector<float>& grad_out) {
        const auto a_broadcast = autograd::broadcast_to_shape(a_data, a_shape, out_shape);
        const auto b_broadcast = autograd::broadcast_to_shape(b_data, b_shape, out_shape);

        if (a_impl->requires_grad()) {
            std::vector<float> grad_a_full(grad_out.size(), 0.0f);
            for (size_t i = 0; i < grad_out.size(); ++i) {
                grad_a_full[i] = grad_out[i] * b_broadcast[i];
            }
            a_impl->backward(autograd::reduce_sum_to_shape(grad_a_full, out_shape, a_shape));
        }

        if (b_impl->requires_grad()) {
            std::vector<float> grad_b_full(grad_out.size(), 0.0f);
            for (size_t i = 0; i < grad_out.size(); ++i) {
                grad_b_full[i] = grad_out[i] * a_broadcast[i];
            }
            b_impl->backward(autograd::reduce_sum_to_shape(grad_b_full, out_shape, b_shape));
        }
    });

    return out;
}

Tensor operator*(const Tensor& lhs, float scalar) {
    return lhs * Tensor::full(lhs.shape(), scalar, lhs.device_type());
}

Tensor operator*(float scalar, const Tensor& rhs) {
    return rhs * Tensor::full(rhs.shape(), scalar, rhs.device_type());
}

} // namespace cpptensor
