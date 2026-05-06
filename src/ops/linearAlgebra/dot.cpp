#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include "cpptensor/ops/helperOps.hpp"

#include <stdexcept>

namespace cpptensor {

Tensor dot(const Tensor& A, const Tensor& B) {
    if (A.device_type() != B.device_type()) {
        throw std::runtime_error("dot: device mismatch");
    }

    const auto& Ash = A.shape();
    const auto& Bsh = B.shape();
    if (Ash.size() != 1 || Bsh.size() != 1) {
        throw std::runtime_error("dot: inputs must be 1D tensors (vectors)");
    }
    if (Ash[0] != Bsh[0]) {
        throw std::runtime_error("dot: size mismatch");
    }

    Tensor out = Tensor::full({}, 0.0f, A.device_type());

    const Tensor prepared_a = materialize_for_backend_input(A);
    const Tensor prepared_b = materialize_for_backend_input(B);
    KernelRegistry::instance().getKernel(OpType::Dot, A.device_type())(prepared_a, prepared_b, out);

    const bool requires_grad = A.requires_grad() || B.requires_grad();
    out.set_requires_grad(requires_grad);
    if (!requires_grad) {
        return out;
    }

    const auto a_data = A.data();
    const auto b_data = B.data();
    const auto a_impl = A.impl();
    const auto b_impl = B.impl();

    out.impl()->set_grad_fn([a_data, b_data, a_impl, b_impl](const std::vector<float>& grad_out) {
        const float g = grad_out.empty() ? 0.0f : grad_out[0];
        if (a_impl->requires_grad()) {
            std::vector<float> grad_a(a_data.size(), 0.0f);
            for (size_t i = 0; i < a_data.size(); ++i) {
                grad_a[i] = g * b_data[i];
            }
            a_impl->backward(grad_a);
        }
        if (b_impl->requires_grad()) {
            std::vector<float> grad_b(b_data.size(), 0.0f);
            for (size_t i = 0; i < b_data.size(); ++i) {
                grad_b[i] = g * a_data[i];
            }
            b_impl->backward(grad_b);
        }
    });

    return out;
}

} // namespace cpptensor
