#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"
#include <stdexcept>

namespace cpptensor {

    namespace {
        const char* deviceTypeName(DeviceType device) {
            switch (device) {
                case DeviceType::CPU:
                    return "CPU";
                case DeviceType::CUDA:
                    return "CUDA";
                default:
                    return "Unknown";
            }
        }
    } // namespace

    Tensor stack(const std::vector<Tensor>& tensors, int dim) {
        // 1. Validate input: empty tensor list
        if (tensors.empty()) {
            throw std::runtime_error("stack: cannot stack empty tensor list");
        }

        // 2. Get reference shape from first tensor
        const auto& first_tensor = tensors[0];
        auto ref_shape = first_tensor.shape();
        const DeviceType common_device = first_tensor.device_type();
        int ndim = static_cast<int>(ref_shape.size());

        // 3. Normalize dimension (handle negative indexing)
        // For stack, valid range is [-ndim-1, ndim] (can insert at any position including end)
        int stack_dim = dim;
        if (stack_dim < 0) {
            // Negative indexing: -1 means insert at end (after last dimension)
            stack_dim = ndim + 1 + stack_dim;
        }

        // Validate dimension range
        if (stack_dim < 0 || stack_dim > ndim) {
            throw std::runtime_error("stack: dimension " + std::to_string(dim) +
                                    " out of range. For tensor with " + std::to_string(ndim) +
                                    " dimensions, valid range is [" + std::to_string(-ndim-1) +
                                    ", " + std::to_string(ndim) + "]");
        }

        // 4. Validate all tensors have the same shape
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& t = tensors[i];
            autograd::throw_if_requires_grad(t, "stack");
            auto t_shape = t.shape();

            if (t.device_type() != common_device) {
                throw std::runtime_error("stack: all tensors must be on the same device. Tensor 0 is on " +
                                         std::string(deviceTypeName(common_device)) +
                                         ", but tensor " + std::to_string(i) + " is on " +
                                         std::string(deviceTypeName(t.device_type())));
            }

            // Check number of dimensions matches
            if (static_cast<int>(t_shape.size()) != ndim) {
                throw std::runtime_error("stack: all tensors must have the same number of dimensions. "
                                       "Expected " + std::to_string(ndim) + " dimensions, but tensor " +
                                       std::to_string(i) + " has " + std::to_string(t_shape.size()) +
                                       " dimensions");
            }

            // Check all dimensions match
            for (int d = 0; d < ndim; ++d) {
                if (t_shape[d] != ref_shape[d]) {
                    throw std::runtime_error("stack: all tensors must have the same shape. "
                                           "Dimension " + std::to_string(d) + " mismatch: expected " +
                                           std::to_string(ref_shape[d]) + " but got " +
                                           std::to_string(t_shape[d]) + " for tensor " +
                                           std::to_string(i));
                }
            }
        }

        // 5. Unsqueeze each tensor at the stack dimension to add a new dimension of size 1
        std::vector<Tensor> unsqueezed;
        unsqueezed.reserve(tensors.size());

        for (const auto& t : tensors) {
            // reshape-backed unsqueeze() requires contiguous storage. Preserve
            // the logical contents of non-contiguous operands explicitly before
            // inserting the new dimension.
            Tensor contiguous_input = t.is_contiguous() ? t : t.contiguous();
            unsqueezed.push_back(contiguous_input.unsqueeze(stack_dim));
        }

        // 6. Use cat to concatenate along the new dimension
        // After unsqueeze, all tensors have shape with an extra dimension of size 1 at stack_dim
        // cat will concatenate along stack_dim, resulting in size = number of tensors
        return cat(unsqueezed, stack_dim);
    }

} // namespace cpptensor
