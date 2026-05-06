#include "cpptensor/ops/reduction/max.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include "cpptensor/ops/helperOps.hpp"

#include <stdexcept>

namespace cpptensor {

    Tensor max(const Tensor& input, std::optional<int> dim, bool keepdim) {
        const auto& in_shape = input.shape();
        const size_t ndim = in_shape.size();

        int actual_dim = -1;
        std::vector<size_t> out_shape;

        if (!dim.has_value()) {
            // Global max: true scalar unless keepdim preserves existing axes.
            out_shape = keepdim ? std::vector<size_t>(ndim, 1) : std::vector<size_t>{};
        } else {
            int d = dim.value();
            if (d < 0) {
                d += static_cast<int>(ndim);
            }

            if (d < 0 || d >= static_cast<int>(ndim)) {
                throw std::invalid_argument(
                    "max: dimension " + std::to_string(dim.value()) +
                    " is out of range for tensor with " +
                    std::to_string(ndim) + " dimensions"
                );
            }

            actual_dim = d;
            out_shape = in_shape;

            if (keepdim) {
                out_shape[static_cast<size_t>(d)] = 1;
            } else {
                out_shape.erase(out_shape.begin() + d);
            }
        }

        // Create output tensor
        Tensor output = Tensor::zeros(out_shape, input.device_type());
        const Tensor prepared_input = materialize_for_backend_input(input);

        // Dispatch to appropriate backend kernel
        auto& registry = KernelRegistry::instance();
        auto kernel = registry.getReductionKernel(
            OpType::Max,
            input.device_type()
        );

        if (!kernel) {
            throw std::runtime_error("No kernel registered for max operation on " +
                                    std::string(input.device_type() == DeviceType::CPU ? "CPU" : "CUDA"));
        }

        // Execute kernel
        kernel(prepared_input, output, actual_dim, keepdim);

        return output;
    }

} // namespace cpptensor
