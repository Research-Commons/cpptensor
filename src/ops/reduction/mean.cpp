#include "cpptensor/ops/reduction/mean.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include <stdexcept>

namespace cpptensor {
    Tensor mean(const Tensor& A, std::optional<int> dim, bool keepdim) {
        const auto& in_shape = A.shape();
        const size_t ndim = in_shape.size();

        // Compute output shape
        std::vector<size_t> out_shape;
        int actual_dim = -1;

        if (!dim.has_value()) {
            // Mean of all elements -> scalar
            out_shape = {1};
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

            // Ensure output shape is not empty (at least a scalar)
            if (out_shape.empty()) {
                out_shape = {1};
            }
        }

        // Create output tensor
        Tensor out = Tensor::zeros(out_shape, A.device_type());

        // Get and call the reduction kernel
        KernelRegistry::instance()
            .getReductionKernel(OpType::Mean, A.device_type())
            (A, out, actual_dim, keepdim);

        return out;
    }
}
