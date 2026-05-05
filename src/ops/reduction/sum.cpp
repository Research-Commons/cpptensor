#include "cpptensor/ops/reduction/sum.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"
#include <stdexcept>

namespace cpptensor {
    Tensor sum(const Tensor& A, std::optional<int> dim, bool keepdim) {
        const auto& in_shape = A.shape();
        const size_t ndim = in_shape.size();

        // Compute output shape
        std::vector<size_t> out_shape;
        int actual_dim = -1;

        if (!dim.has_value()) {
            // Sum all elements -> scalar unless keepdim preserves singleton axes
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
                throw std::runtime_error("Sum dimension out of range");
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
            .getReductionKernel(OpType::Sum, A.device_type())
            (input, out, actual_dim, keepdim);

        return out;
    }
}
