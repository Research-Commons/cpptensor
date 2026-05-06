#include "cpptensor/ops/reduction/mean.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"
#include <stdexcept>

namespace cpptensor {
    Tensor mean(const Tensor& A, std::optional<int> dim, bool keepdim) {
        const auto& in_shape = A.shape();
        const size_t ndim = in_shape.size();

        // Compute output shape
        std::vector<size_t> out_shape;
        int actual_dim = -1;

        if (!dim.has_value()) {
            // Mean of all elements -> scalar unless keepdim preserves singleton axes
            out_shape = keepdim ? std::vector<size_t>(ndim, 1) : std::vector<size_t>{};
            actual_dim = -1;
        } else {
            int d = dim.value();

            // Handle negative indexing
            if (d < 0) {
                d += static_cast<int>(ndim);
            }

            // Validate dimension
            if (d < 0 || d >= static_cast<int>(ndim)) {
                throw std::runtime_error("Mean dimension out of range");
            }

            actual_dim = d;
            out_shape = in_shape;

            if (keepdim) {
                out_shape[d] = 1;
            } else {
                out_shape.erase(out_shape.begin() + d);
            }

        }

        // Create output tensor
        Tensor out = Tensor::zeros(out_shape, A.device_type());
        const Tensor input = materialize_for_backend_input(A);

        // Get and call the reduction kernel
        KernelRegistry::instance()
            .getReductionKernel(OpType::Mean, A.device_type())
            (input, out, actual_dim, keepdim);

        const bool requires_grad = A.requires_grad();
        out.set_requires_grad(requires_grad);
        if (!requires_grad) {
            return out;
        }

        size_t reduction_size = 1;
        if (actual_dim == -1) {
            for (size_t d : in_shape) {
                reduction_size *= d;
            }
        } else {
            reduction_size = in_shape[static_cast<size_t>(actual_dim)];
        }
        if (reduction_size == 0) {
            throw std::runtime_error("autograd: mean backward undefined for empty reduction");
        }

        const auto out_shape_copy = out.shape();
        const auto in_shape_copy = in_shape;
        const auto input_impl = A.impl();
        const std::optional<int> backward_dim = (actual_dim == -1)
                                                ? std::nullopt
                                                : std::optional<int>(actual_dim);
        const float scale = 1.0f / static_cast<float>(reduction_size);

        out.impl()->set_grad_fn([input_impl, out_shape_copy, in_shape_copy, backward_dim, keepdim, scale]
                                (const std::vector<float>& grad_out) {
            std::vector<float> expanded = autograd::expand_reduction_grad(grad_out,
                                                                           out_shape_copy,
                                                                           in_shape_copy,
                                                                           backward_dim,
                                                                           keepdim);
            for (float& v : expanded) {
                v *= scale;
            }
            input_impl->backward(expanded);
        });

        return out;
    }
}
