#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/ops/reduction/sum.hpp"
#include "cpptensor/ops/reduction/mean.hpp"
#include "cpptensor/ops/reduction/max.hpp"
#include "cpptensor/ops/reduction/min.hpp"

#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace cpptensor {

    namespace {

        struct NormalizedSlice {
            int64_t start = 0;
            int64_t stop = 0;
            int64_t step = 1;
            size_t length = 0;
        };

        int64_t normalize_scalar_index(int64_t index,
                                       int64_t dim_size,
                                       const char* op_name) {
            int64_t normalized = index;
            if (normalized < 0) {
                normalized += dim_size;
            }

            if (normalized < 0 || normalized >= dim_size) {
                throw std::runtime_error(std::string(op_name) + ": scalar index " +
                                         std::to_string(index) + " out of bounds for dimension size " +
                                         std::to_string(dim_size));
            }

            return normalized;
        }

        NormalizedSlice normalize_slice_spec(const Tensor::SliceSpec& slice,
                                             int64_t dim_size,
                                             const char* op_name) {
            const int64_t step = slice.step.value_or(1);
            if (step == 0) {
                throw std::runtime_error(std::string(op_name) + ": slice step cannot be zero");
            }

            if (step > 0) {
                int64_t start = slice.start.value_or(0);
                int64_t stop = slice.end.value_or(dim_size);

                if (start < 0) start += dim_size;
                if (stop < 0) stop += dim_size;

                start = std::clamp(start, int64_t{0}, dim_size);
                stop = std::clamp(stop, int64_t{0}, dim_size);

                size_t length = 0;
                if (start < stop) {
                    const int64_t span = stop - start;
                    length = static_cast<size_t>((span + step - 1) / step);
                }

                return NormalizedSlice{
                    .start = start,
                    .stop = stop,
                    .step = step,
                    .length = length,
                };
            }

            int64_t start = slice.start.value_or(dim_size - 1);
            int64_t stop = slice.end.value_or(-1);

            if (slice.start.has_value() && start < 0) {
                start += dim_size;
            }
            if (slice.start.has_value()) {
                start = std::clamp(start, int64_t{-1}, dim_size - 1);
            } else if (dim_size == 0) {
                start = -1;
            }

            if (slice.end.has_value() && stop < 0) {
                stop += dim_size;
            }
            if (slice.end.has_value()) {
                stop = std::clamp(stop, int64_t{-1}, dim_size - 1);
            } else {
                stop = -1;
            }

            size_t length = 0;
            if (start > stop) {
                const int64_t span = start - stop;
                const int64_t abs_step = -step;
                length = static_cast<size_t>((span + abs_step - 1) / abs_step);
            }

            return NormalizedSlice{
                .start = start,
                .stop = stop,
                .step = step,
                .length = length,
            };
        }

        std::vector<float> copy_logical_data(const Tensor& tensor) {
            const auto sh = tensor.shape();
            const auto& st = tensor.stride();
            const float* src = tensor.impl()->data_ptr();
            const size_t total = tensor.numel();

            std::vector<float> copied(total);
            if (total == 0) {
                return copied;
            }

            if (sh.empty()) {
                copied[0] = src[0];
                return copied;
            }

            std::vector<size_t> indices(sh.size(), 0);
            for (size_t i = 0; i < total; ++i) {
                size_t src_offset = 0;
                for (size_t d = 0; d < sh.size(); ++d) {
                    src_offset += indices[d] * st[d];
                }

                copied[i] = src[src_offset];

                for (int d = static_cast<int>(sh.size()) - 1; d >= 0; --d) {
                    if (++indices[static_cast<size_t>(d)] < sh[static_cast<size_t>(d)]) {
                        break;
                    }
                    indices[static_cast<size_t>(d)] = 0;
                }
            }

            return copied;
        }

    } // namespace

    // ---------- Constructors ----------
    Tensor::Tensor(const std::vector<size_t>& shape,
                   const std::vector<float>& values,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, values, device))
    {}

    Tensor::Tensor(const std::vector<size_t>& shape,
                   float value,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, value, device))
    {}

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
        : impl_(std::move(impl))
    {}

    std::shared_ptr<TensorImpl> Tensor::require_impl(const char* method) const {
        if (!impl_) {
            throw std::runtime_error(std::string("Tensor::") + method +
                                     ": tensor is uninitialized; default-constructed tensors must be assigned before use");
        }
        return impl_;
    }

    // ---------- Factories ----------
    Tensor Tensor::zeros(const std::vector<size_t>& shape,
                         DeviceType device) {
        return Tensor(shape, 0.0f, device);
    }

    Tensor Tensor::ones(const std::vector<size_t>& shape,
                        DeviceType device) {
        return Tensor(shape, 1.0f, device);
    }

    Tensor Tensor::full(const std::vector<size_t>& shape,
                        float value,
                        DeviceType device) {
        return Tensor(shape, value, device);
    }

    Tensor Tensor::randn(const std::vector<size_t>& shape,
                         DeviceType device) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        std::vector<float> data(total);
        static thread_local std::mt19937_64 gen((unsigned)std::random_device{}());
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (size_t i = 0; i < total; ++i) data[i] = d(gen);
        return Tensor(shape, data, device);
    }

    Tensor Tensor::from_ptr(const std::vector<size_t>& shape,
                           float* data_ptr,
                           std::shared_ptr<TensorImpl> owner,
                           DeviceType device) {
        auto impl = std::make_shared<TensorImpl>(shape, data_ptr, owner, device);
        return Tensor(std::move(impl));
    }

    // ---------- Shape & Info ----------
    std::vector<size_t> Tensor::shape() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).shape();
    }
    size_t Tensor::numel() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).numel();
    }
    size_t Tensor::ndim() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).shape().size();
    }
    DeviceType Tensor::device_type() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).device();
    }


    void Tensor::print() const {
        const auto impl = require_impl(__func__);
        const auto &s = impl->shape();
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << s[i];
        }
        std::cout << "], values=[";

        // Use stride-aware access for views/sliced tensors
        const auto &strides = impl->stride();
        const float* data_ptr = impl->data_ptr();
        size_t total_elements = numel();

        // Helper to convert flat index to multi-dimensional indices
        auto flat_to_indices = [&](size_t flat_idx) -> std::vector<size_t> {
            std::vector<size_t> indices(s.size());
            for (int i = (int)s.size() - 1; i >= 0; --i) {
                indices[i] = flat_idx % s[i];
                flat_idx /= s[i];
            }
            return indices;
        };

        // Helper to compute strided offset from multi-dimensional indices
        auto compute_offset = [&](const std::vector<size_t>& indices) -> size_t {
            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                offset += indices[i] * strides[i];
            }
            return offset;
        };

        for (size_t i = 0; i < total_elements; ++i) {
            if (i) std::cout << ", ";
            auto indices = flat_to_indices(i);
            size_t offset = compute_offset(indices);
            std::cout << data_ptr[offset];
            if (i >= 31) { std::cout << ", ..."; break; }
        }
        std::cout << "])\n";
    }

    void Tensor::print_pretty() const {
        // small pretty printer: only for 1D or 2D tensors
        const auto impl = require_impl(__func__);
        const auto &s = impl->shape();
        const auto &strides = impl->stride();
        const float* data_ptr = impl->data_ptr();

        if (s.size() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < s[0]; ++i) {
                if (i) std::cout << ", ";
                std::cout << data_ptr[i * strides[0]];
            }
            std::cout << "]\n";
        } else if (s.size() == 2) {
            for (size_t r = 0; r < s[0]; ++r) {
                std::cout << "[";
                for (size_t c = 0; c < s[1]; ++c) {
                    if (c) std::cout << ", ";
                    size_t offset = r * strides[0] + c * strides[1];
                    std::cout << data_ptr[offset];
                }
                std::cout << "]\n";
            }
        } else {
            print();
        }
    }

    // Data access
    const std::vector<float>& Tensor::data() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).data();
    }
    std::vector<float>& Tensor::data() { return require_impl(__func__)->data(); }
    const std::vector<size_t>& Tensor::stride() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).stride();
    }
    std::vector<size_t>& Tensor::stride(){ return require_impl(__func__)->stride(); }
    std::shared_ptr<TensorImpl> Tensor::impl() const { return require_impl(__func__); }

    // =============== Tensor Manipulation Operations ===============

    Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
        const auto impl = require_impl(__func__);
        // Validate total elements match
        size_t new_numel = 1;
        for (auto s : new_shape) new_numel *= s;

        if (numel() != new_numel) {
            throw std::runtime_error("view: cannot reshape tensor of size " +
                                    std::to_string(numel()) + " to size " +
                                    std::to_string(new_numel));
        }

        // Check if tensor is contiguous (required for view)
        if (!is_contiguous()) {
            throw std::runtime_error("view: tensor must be contiguous. Call contiguous() first.");
        }

        // Create view TensorImpl that shares data with this tensor
        auto view_impl = std::make_shared<TensorImpl>(impl, new_shape);

        Tensor result;
        result.impl_ = view_impl;
        return result;
    }

    Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
        require_impl(__func__);
        if (is_contiguous()) {
            return view(new_shape);  // Zero-copy if possible
        } else {
            // Must copy to make contiguous first
            return contiguous().view(new_shape);
        }
    }

    Tensor Tensor::flatten(int start_dim, int end_dim) const {
        require_impl(__func__);
        auto sh = shape();
        int ndims = static_cast<int>(sh.size());

        if (ndims == 0) {
            throw std::runtime_error("flatten: cannot flatten scalar tensor");
        }

        // Normalize negative indices
        if (start_dim < 0) start_dim += ndims;
        if (end_dim < 0) end_dim += ndims;

        // Validate range
        if (start_dim < 0 || start_dim >= ndims ||
            end_dim < 0 || end_dim >= ndims ||
            start_dim > end_dim) {
            throw std::runtime_error("flatten: invalid dimension range");
        }

        // Compute new shape
        std::vector<size_t> new_shape;

        // Keep dimensions before start_dim
        for (int i = 0; i < start_dim; ++i) {
            new_shape.push_back(sh[i]);
        }

        // Flatten dimensions from start_dim to end_dim
        size_t flat_size = 1;
        for (int i = start_dim; i <= end_dim; ++i) {
            flat_size *= sh[i];
        }
        new_shape.push_back(flat_size);

        // Keep dimensions after end_dim
        for (int i = end_dim + 1; i < ndims; ++i) {
            new_shape.push_back(sh[i]);
        }

        return reshape(new_shape);
    }

    Tensor Tensor::slice(int dim,
                         std::optional<int64_t> start,
                         std::optional<int64_t> end,
                         std::optional<int64_t> step) const {
        require_impl(__func__);
        const int rank = static_cast<int>(ndim());

        int norm_dim = dim;
        if (norm_dim < 0) {
            norm_dim += rank;
        }

        if (norm_dim < 0 || norm_dim >= rank) {
            throw std::runtime_error("slice: dimension " + std::to_string(dim) +
                                   " out of range for tensor with " + std::to_string(rank) + " dimensions");
        }

        std::vector<IndexSpec> specs(static_cast<size_t>(rank), IndexSpec(SliceSpec{}));
        specs[static_cast<size_t>(norm_dim)] = SliceSpec(start, end, step);
        return index(specs);
    }

    Tensor Tensor::index(const std::vector<IndexSpec>& indices) const {
        const auto impl = require_impl(__func__);
        const auto& src_shape = impl->shape();
        const auto& src_stride = impl->stride();
        const int rank = static_cast<int>(src_shape.size());

        if (static_cast<int>(indices.size()) > rank) {
            throw std::runtime_error("index: received " + std::to_string(indices.size()) +
                                     " indices for tensor with rank " + std::to_string(rank));
        }

        struct AxisPlan {
            bool is_scalar = false;
            int64_t scalar_index = 0;
            int64_t start = 0;
            int64_t step = 1;
            size_t length = 0;
            bool has_negative_step = false;
        };

        std::vector<AxisPlan> axis_plans(static_cast<size_t>(rank));
        bool has_negative_step = false;

        for (int axis = 0; axis < rank; ++axis) {
            const int64_t dim_size = static_cast<int64_t>(src_shape[static_cast<size_t>(axis)]);
            const bool user_provided = static_cast<size_t>(axis) < indices.size();
            const IndexSpec spec = user_provided
                ? indices[static_cast<size_t>(axis)]
                : IndexSpec(SliceSpec{});

            AxisPlan plan;
            if (std::holds_alternative<int64_t>(spec.value)) {
                plan.is_scalar = true;
                plan.scalar_index = normalize_scalar_index(std::get<int64_t>(spec.value), dim_size, "index");
                axis_plans[static_cast<size_t>(axis)] = plan;
                continue;
            }

            const auto normalized = normalize_slice_spec(std::get<SliceSpec>(spec.value), dim_size, "index");
            plan.start = normalized.start;
            plan.step = normalized.step;
            plan.length = normalized.length;
            plan.has_negative_step = normalized.step < 0;
            has_negative_step = has_negative_step || plan.has_negative_step;

            axis_plans[static_cast<size_t>(axis)] = plan;
        }

        if (!has_negative_step) {
            std::vector<size_t> out_shape;
            std::vector<size_t> out_stride;
            out_shape.reserve(src_shape.size());
            out_stride.reserve(src_shape.size());

            size_t offset_delta = 0;
            for (int axis = 0; axis < rank; ++axis) {
                const auto& plan = axis_plans[static_cast<size_t>(axis)];
                const size_t axis_stride = src_stride[static_cast<size_t>(axis)];

                if (plan.is_scalar) {
                    offset_delta += static_cast<size_t>(plan.scalar_index) * axis_stride;
                    continue;
                }

                if (plan.length > 0 && plan.start > 0) {
                    offset_delta += static_cast<size_t>(plan.start) * axis_stride;
                }

                out_shape.push_back(plan.length);
                out_stride.push_back(axis_stride * static_cast<size_t>(plan.step));
            }

            auto view_impl = std::make_shared<TensorImpl>(impl, out_shape, out_stride, offset_delta);
            return Tensor(std::move(view_impl));
        }

        std::vector<size_t> out_shape;
        out_shape.reserve(src_shape.size());
        for (int axis = 0; axis < rank; ++axis) {
            const auto& plan = axis_plans[static_cast<size_t>(axis)];
            if (!plan.is_scalar) {
                out_shape.push_back(plan.length);
            }
        }

        size_t out_numel = 1;
        for (const size_t d : out_shape) {
            out_numel *= d;
        }

        std::vector<float> out_data(out_numel, 0.0f);
        const float* src = impl->data_ptr();

        if (out_numel > 0) {
            std::vector<size_t> out_indices(out_shape.size(), 0);
            for (size_t out_linear = 0; out_linear < out_numel; ++out_linear) {
                int64_t src_offset = 0;
                size_t out_axis = 0;

                for (int axis = 0; axis < rank; ++axis) {
                    const auto& plan = axis_plans[static_cast<size_t>(axis)];
                    int64_t src_index = 0;

                    if (plan.is_scalar) {
                        src_index = plan.scalar_index;
                    } else {
                        src_index = plan.start +
                                    static_cast<int64_t>(out_indices[out_axis]) * plan.step;
                        ++out_axis;
                    }

                    src_offset += src_index * static_cast<int64_t>(src_stride[static_cast<size_t>(axis)]);
                }

                if (src_offset < 0) {
                    throw std::runtime_error("index: internal error produced negative source offset");
                }

                out_data[out_linear] = src[static_cast<size_t>(src_offset)];

                for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                    const size_t dim = static_cast<size_t>(d);
                    if (++out_indices[dim] < out_shape[dim]) {
                        break;
                    }
                    out_indices[dim] = 0;
                }
            }
        }

        return Tensor(out_shape, out_data, device_type());
    }

    Tensor Tensor::index(std::initializer_list<IndexSpec> indices) const {
        return index(std::vector<IndexSpec>(indices.begin(), indices.end()));
    }

    Tensor Tensor::squeeze(int dim) const {
        require_impl(__func__);
        auto sh = shape();
        std::vector<size_t> new_shape;

        if (dim == -1) {
            // Remove all dimensions of size 1
            for (size_t d : sh) {
                if (d != 1) new_shape.push_back(d);
            }
        } else {
            // Remove specific dimension if size 1
            int ndims = static_cast<int>(sh.size());
            int norm_dim = (dim < 0) ? dim + ndims : dim;

            if (norm_dim < 0 || norm_dim >= ndims) {
                throw std::runtime_error("squeeze: dimension out of range");
            }

            if (sh[norm_dim] != 1) {
                throw std::runtime_error("squeeze: dimension " + std::to_string(dim) +
                                        " has size " + std::to_string(sh[norm_dim]) +
                                        ", expected 1");
            }

            for (int i = 0; i < ndims; ++i) {
                if (i != norm_dim) {
                    new_shape.push_back(sh[i]);
                }
            }
        }

        return reshape(new_shape);
    }

    Tensor Tensor::unsqueeze(int dim) const {
        require_impl(__func__);
        auto sh = shape();
        int ndims = static_cast<int>(sh.size());

        // Normalize dimension (allow -1 for last index)
        int norm_dim = dim;
        if (dim < 0) norm_dim = dim + ndims + 1;  // +1 cuz we're adding a dimension

        if (norm_dim < 0 || norm_dim > ndims) {
            throw std::runtime_error("unsqueeze: dimension out of range");
        }

        // Insert dimension of size 1
        std::vector<size_t> new_shape;
        for (int i = 0; i < norm_dim; ++i) {
            new_shape.push_back(sh[i]);
        }
        new_shape.push_back(1);
        for (int i = norm_dim; i < ndims; ++i) {
            new_shape.push_back(sh[i]);
        }

        return reshape(new_shape);
    }

    Tensor Tensor::permute(const std::vector<int>& dims) const {
        const auto impl = require_impl(__func__);
        auto old_shape = shape();
        auto old_stride = stride();
        int ndims = static_cast<int>(old_shape.size());

        // Validate dimensions
        if (static_cast<int>(dims.size()) != ndims) {
            throw std::runtime_error("permute: dims size mismatch. Expected " +
                                    std::to_string(ndims) + ", got " +
                                    std::to_string(dims.size()));
        }

        // Check for valid permutation
        std::vector<bool> seen(ndims, false);
        for (int d : dims) {
            int norm_d = (d < 0) ? d + ndims : d;
            if (norm_d < 0 || norm_d >= ndims) {
                throw std::runtime_error("permute: dimension out of range");
            }
            if (seen[norm_d]) {
                throw std::runtime_error("permute: duplicate dimension");
            }
            seen[norm_d] = true;
        }

        // Compute new shape and stride by reordering
        std::vector<size_t> new_shape(ndims);
        std::vector<size_t> new_stride(ndims);

        for (int i = 0; i < ndims; ++i) {
            int d = dims[i];
            if (d < 0) d += ndims;
            new_shape[i] = old_shape[d];
            new_stride[i] = old_stride[d];
        }

        // Create view with modified shape and stride
        auto view_impl = std::make_shared<TensorImpl>(impl, new_shape, new_stride);

        Tensor result;
        result.impl_ = view_impl;
        return result;
    }

    Tensor Tensor::transpose(int dim0, int dim1) const {
        require_impl(__func__);
        int ndims = static_cast<int>(ndim());

        // Default: transpose last two dimensions for 2D case
        if (ndims < 2) {
            throw std::runtime_error("transpose: tensor must have at least 2 dimensions");
        }

        // Normalize dimensions
        if (dim0 < 0) dim0 += ndims;
        if (dim1 < 0) dim1 += ndims;

        if (dim0 < 0 || dim0 >= ndims || dim1 < 0 || dim1 >= ndims) {
            throw std::runtime_error("transpose: dimension out of range");
        }

        // Create permutation that swaps dim0 and dim1
        std::vector<int> perm(ndims);
        for (int i = 0; i < ndims; ++i) {
            perm[i] = i;
        }
        std::swap(perm[dim0], perm[dim1]);

        return permute(perm);
    }

    bool Tensor::is_contiguous() const {
        require_impl(__func__);
        auto sh = shape();
        auto st = stride();

        if (sh.empty()) return true;

        // Check if strides match row-major layout
        size_t expected_stride = 1;
        for (int i = static_cast<int>(sh.size()) - 1; i >= 0; --i) {
            if (st[i] != expected_stride) return false;
            expected_stride *= sh[i];
        }
        return true;
    }

    Tensor Tensor::contiguous() const {
        const auto impl = require_impl(__func__);
        if (impl->can_expose_direct_data_buffer()) {
            return *this;  // Already backed by a direct compact buffer
        }

        return Tensor(shape(), copy_logical_data(*this), device_type());
    }

    Tensor Tensor::clone() const {
        require_impl(__func__);
        return Tensor(shape(), copy_logical_data(*this), device_type());
    }

    // =============== Reduction Operations Implementation ===============

    // Global reduction overloads (no dim parameter)
    Tensor Tensor::sum(bool keepdim) const {
        require_impl(__func__);
        return cpptensor::sum(*this, std::nullopt, keepdim);
    }

    Tensor Tensor::mean(bool keepdim) const {
        require_impl(__func__);
        return cpptensor::mean(*this, std::nullopt, keepdim);
    }

    Tensor Tensor::max(bool keepdim) const {
        require_impl(__func__);
        return cpptensor::max(*this, -1, keepdim);
    }

    Tensor Tensor::min(bool keepdim) const {
        require_impl(__func__);
        return cpptensor::min(*this, -1, keepdim);
    }

    // Dimensional reduction overloads (with dim parameter)
    Tensor Tensor::sum(int dim, bool keepdim) const {
        require_impl(__func__);
        return cpptensor::sum(*this, std::optional<int>(dim), keepdim);
    }

    Tensor Tensor::mean(int dim, bool keepdim) const {
        require_impl(__func__);
        return cpptensor::mean(*this, std::optional<int>(dim), keepdim);
    }

    Tensor Tensor::max(int dim, bool keepdim) const {
        require_impl(__func__);
        int actual_dim = dim;
        if (actual_dim < 0) {
            actual_dim += static_cast<int>(ndim());
        }
        return cpptensor::max(*this, actual_dim, keepdim);
    }

    Tensor Tensor::min(int dim, bool keepdim) const {
        require_impl(__func__);
        int actual_dim = dim;
        if (actual_dim < 0) {
            actual_dim += static_cast<int>(ndim());
        }
        return cpptensor::min(*this, actual_dim, keepdim);
    }

} // namespace cpptensor
