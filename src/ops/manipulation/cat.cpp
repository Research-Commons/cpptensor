#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"
#include <stdexcept>
#include <cstring>

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

    // Helper function to copy a slice from src tensor to dst tensor at given
    // offset along concat_dim.
    void copySlice(Tensor& dst, const Tensor& src, int concat_dim, size_t offset_in_concat_dim) {
        auto src_shape = src.shape();
        auto dst_shape = dst.shape();
        auto src_stride = src.stride();
        auto dst_stride = dst.stride();

        int ndim = static_cast<int>(src_shape.size());

        // Use offset-aware raw pointers so zero-copy views copy their logical
        // contents instead of starting from the base tensor's storage origin.
        float* dst_data = dst.impl()->data_ptr();
        const float* src_data = src.impl()->data_ptr();

        // Compute total iterations needed (product of all dims except concat_dim)
        size_t total_iterations = 1;
        for (int i = 0; i < ndim; ++i) {
            if (i != concat_dim) {
                total_iterations *= src_shape[i];
            }
        }

        // Size of contiguous chunk to copy along concat dimension
        size_t concat_size = src_shape[concat_dim];

        // If the concat dimension is the last dimension and both tensors are
        // contiguous, we can optimize with fewer, larger memcpy calls.
        bool can_optimize = (concat_dim == ndim - 1) && src.is_contiguous() && dst.is_contiguous();

        if (can_optimize) {
            // Fast path: copy entire rows at once
            size_t elements_per_row = concat_size;
            size_t dst_row_offset = offset_in_concat_dim;

            for (size_t iter = 0; iter < total_iterations; ++iter) {
                size_t src_row_start = iter * elements_per_row;
                size_t dst_row_start = iter * dst_shape[concat_dim] + dst_row_offset;

                std::memcpy(dst_data + dst_row_start,
                           src_data + src_row_start,
                           elements_per_row * sizeof(float));
            }
        } else {
            // General path: iterate through all indices
            std::vector<size_t> indices(ndim, 0);

            for (size_t iter = 0; iter < total_iterations; ++iter) {
                // Copy the slice along concat_dim
                for (size_t c = 0; c < concat_size; ++c) {
                    // Compute source offset
                    size_t src_offset = 0;
                    for (int d = 0; d < ndim; ++d) {
                        size_t idx = (d == concat_dim) ? c : indices[d];
                        src_offset += idx * src_stride[d];
                    }

                    // Compute destination offset
                    size_t dst_offset = 0;
                    for (int d = 0; d < ndim; ++d) {
                        size_t idx = (d == concat_dim) ? (c + offset_in_concat_dim) : indices[d];
                        dst_offset += idx * dst_stride[d];
                    }

                    dst_data[dst_offset] = src_data[src_offset];
                }

                // Increment indices (skip concat_dim)
                for (int d = ndim - 1; d >= 0; --d) {
                    if (d == concat_dim) continue;

                    indices[d]++;
                    if (indices[d] < src_shape[d]) {
                        break;
                    }
                    indices[d] = 0;
                }
            }
        }
    }
} // anonymous namespace

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    // 1. Validate input: empty tensor list
    if (tensors.empty()) {
        throw std::runtime_error("cat: cannot concatenate empty tensor list");
    }

    // 2. Get reference shape and ndim from first tensor
    const auto& first_tensor = tensors[0];
    auto ref_shape = first_tensor.shape();
    const DeviceType common_device = first_tensor.device_type();
    int ndim = static_cast<int>(ref_shape.size());

    if (ndim == 0) {
        throw std::runtime_error("cat: cannot concatenate 0-dimensional tensors");
    }

    // 3. Normalize dimension (handle negative indexing)
    int concat_dim = dim;
    if (concat_dim < 0) {
        concat_dim += ndim;
    }

    if (concat_dim < 0 || concat_dim >= ndim) {
        throw std::runtime_error("cat: dimension " + std::to_string(dim) +
                                " out of range for tensor with " + std::to_string(ndim) +
                                " dimensions");
    }

    // 4. Validate all tensors and compute total size along concat dimension
    size_t total_concat_size = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        autograd::throw_if_requires_grad(t, "cat");
        auto t_shape = t.shape();

        if (t.device_type() != common_device) {
            throw std::runtime_error("cat: all tensors must be on the same device. Tensor 0 is on " +
                                     std::string(deviceTypeName(common_device)) +
                                     ", but tensor " + std::to_string(i) + " is on " +
                                     std::string(deviceTypeName(t.device_type())));
        }

        // Check number of dimensions matches
        if (static_cast<int>(t_shape.size()) != ndim) {
            throw std::runtime_error("cat: all tensors must have the same number of dimensions. "
                                   "Expected " + std::to_string(ndim) + " dimensions, but tensor " +
                                   std::to_string(i) + " has " + std::to_string(t_shape.size()) +
                                   " dimensions");
        }

        // Check all dimensions except concat_dim match the reference
        for (int d = 0; d < ndim; ++d) {
            if (d != concat_dim && t_shape[d] != ref_shape[d]) {
                throw std::runtime_error("cat: all tensors must have the same shape except in the "
                                       "concatenating dimension. Dimension " + std::to_string(d) +
                                       " mismatch: expected " + std::to_string(ref_shape[d]) +
                                       " but got " + std::to_string(t_shape[d]) +
                                       " for tensor " + std::to_string(i));
            }
        }

        total_concat_size += t_shape[concat_dim];
    }

    // 5. Compute output shape
    std::vector<size_t> out_shape = ref_shape;
    out_shape[concat_dim] = total_concat_size;

    // 6. Allocate output tensor (initialized to zero)
    Tensor result = Tensor::zeros(out_shape, common_device);

    // 7. Copy data from each input tensor to the output
    size_t offset_in_concat_dim = 0;

    for (const auto& t : tensors) {
        // Copy directly from the logical tensor view. copySlice() handles
        // offsets for contiguous views and strides for non-contiguous ones.
        copySlice(result, t, concat_dim, offset_in_concat_dim);

        // Move offset forward by this tensor's size in concat dimension.
        offset_in_concat_dim += t.shape()[concat_dim];
    }

    return result;
}

} // namespace cpptensor
