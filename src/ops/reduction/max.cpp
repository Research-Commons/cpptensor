#include "cpptensor/ops/reduction/max.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"

#include <stdexcept>

namespace cpptensor {

    Tensor max(const Tensor& input, int dim, bool keepdim) {
        // Validate dimension
        if (dim >= static_cast<int>(input.ndim())) {
            throw std::invalid_argument(
                "max: dimension " + std::to_string(dim) +
                " is out of range for tensor with " +
                std::to_string(input.ndim()) + " dimensions"
            );
        }

        // Compute output shape
        std::vector<size_t> out_shape;
        if (dim < 0) {
            // Max over all elements
            if (keepdim) {
                out_shape = std::vector<size_t>(input.ndim(), 1);
            } else {
                out_shape = {};  // Scalar output
            }
        } else {
            // Max along specific dimension
            const auto& in_shape = input.shape();
            for (size_t i = 0; i < in_shape.size(); ++i) {
                if (static_cast<int>(i) == dim) {
                    if (keepdim) {
                        out_shape.push_back(1);
                    }
                } else {
                    out_shape.push_back(in_shape[i]);
                }
            }
        }

        // Create output tensor
        Tensor output = Tensor::zeros(out_shape, input.device_type());

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
        kernel(input, output, dim, keepdim);

        return output;
    }

} // namespace cpptensor
