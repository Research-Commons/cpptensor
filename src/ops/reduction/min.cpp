#include "cpptensor/ops/reduction/min.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

#include <stdexcept>

namespace cpptensor {

Tensor min(const Tensor& input, std::optional<int> dim, bool keepdim) {
    autograd::throw_if_requires_grad(input, "min");

    const auto& in_shape = input.shape();
    const size_t ndim = in_shape.size();

    int actual_dim = -1;
    std::vector<size_t> out_shape;

    if (!dim.has_value()) {
        out_shape = keepdim ? std::vector<size_t>(ndim, 1) : std::vector<size_t>{};
    } else {
        int d = dim.value();
        if (d < 0) {
            d += static_cast<int>(ndim);
        }

        if (d < 0 || d >= static_cast<int>(ndim)) {
            throw std::invalid_argument(
                "min: dimension " + std::to_string(dim.value()) +
                " is out of range for tensor with " +
                std::to_string(ndim) + " dimensions");
        }

        actual_dim = d;
        out_shape = in_shape;

        if (keepdim) {
            out_shape[static_cast<size_t>(d)] = 1;
        } else {
            out_shape.erase(out_shape.begin() + d);
        }
    }

    Tensor output = Tensor::zeros(out_shape, input.device_type());
    const Tensor prepared_input = materialize_for_backend_input(input);

    auto& registry = KernelRegistry::instance();
    auto kernel = registry.getReductionKernel(OpType::Min, input.device_type());

    if (!kernel) {
        throw std::runtime_error("No kernel registered for min operation on " +
                                 std::string(input.device_type() == DeviceType::CPU ? "CPU" : "CUDA"));
    }

    kernel(prepared_input, output, actual_dim, keepdim);

    return output;
}

} // namespace cpptensor
