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
#include <cstring>

namespace cpptensor {

    // ---------- Constructors ----------
    Tensor::Tensor(const std::vector<size_t>& shape,
                   const std::vector<float>& values,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, values, device))
    {}

    Tensor::Tensor(const std::vector<size_t>& shape,
                   std::initializer_list<float> values,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, std::vector<float>(values), device))
    {}

    Tensor::Tensor(const std::vector<size_t>& shape,
                   std::initializer_list<int> values,
                   DeviceType device)
        : impl_(nullptr)
    {
        std::vector<float> converted;
        converted.reserve(values.size());
        for (int value : values) {
            converted.push_back(static_cast<float>(value));
        }
        impl_ = std::make_shared<TensorImpl>(shape, converted, device);
    }

    Tensor::Tensor(const std::vector<size_t>& shape,
                   const std::vector<double>& values,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, values, device))
    {}

    Tensor::Tensor(const std::vector<size_t>& shape,
                   const std::vector<std::int32_t>& values,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, values, device))
    {}

    Tensor::Tensor(const std::vector<size_t>& shape,
                   const std::vector<bool>& values,
                   DeviceType device)
        : impl_(std::make_shared<TensorImpl>(shape, values, device))
    {}

    Tensor::Tensor(const std::vector<size_t>& shape,
                   float value,
                   DeviceType device,
                   DType dtype)
        : impl_(nullptr)
    {
        switch (dtype) {
            case DType::FLOAT32:
                impl_ = std::make_shared<TensorImpl>(shape, value, device);
                break;
            case DType::FLOAT64:
                impl_ = std::make_shared<TensorImpl>(shape, static_cast<double>(value), device);
                break;
            case DType::INT32:
                impl_ = std::make_shared<TensorImpl>(shape, static_cast<std::int32_t>(value), device);
                break;
            case DType::BOOL:
                impl_ = std::make_shared<TensorImpl>(shape, value != 0.0f, device);
                break;
        }
    }

    Tensor::Tensor(const std::vector<size_t>& shape,
                   double value,
                   DeviceType device,
                   DType dtype)
        : impl_(nullptr)
    {
        switch (dtype) {
            case DType::FLOAT32:
                impl_ = std::make_shared<TensorImpl>(shape, static_cast<float>(value), device);
                break;
            case DType::FLOAT64:
                impl_ = std::make_shared<TensorImpl>(shape, value, device);
                break;
            case DType::INT32:
                impl_ = std::make_shared<TensorImpl>(shape, static_cast<std::int32_t>(value), device);
                break;
            case DType::BOOL:
                impl_ = std::make_shared<TensorImpl>(shape, value != 0.0, device);
                break;
        }
    }

    Tensor::Tensor(const std::vector<size_t>& shape,
                   std::int32_t value,
                   DeviceType device,
                   DType dtype)
        : impl_(nullptr)
    {
        switch (dtype) {
            case DType::FLOAT32:
                impl_ = std::make_shared<TensorImpl>(shape, static_cast<float>(value), device);
                break;
            case DType::FLOAT64:
                impl_ = std::make_shared<TensorImpl>(shape, static_cast<double>(value), device);
                break;
            case DType::INT32:
                impl_ = std::make_shared<TensorImpl>(shape, value, device);
                break;
            case DType::BOOL:
                impl_ = std::make_shared<TensorImpl>(shape, value != 0, device);
                break;
        }
    }

    Tensor::Tensor(const std::vector<size_t>& shape,
                   bool value,
                   DeviceType device,
                   DType dtype)
        : impl_(nullptr)
    {
        switch (dtype) {
            case DType::FLOAT32:
                impl_ = std::make_shared<TensorImpl>(shape, value ? 1.0f : 0.0f, device);
                break;
            case DType::FLOAT64:
                impl_ = std::make_shared<TensorImpl>(shape, value ? 1.0 : 0.0, device);
                break;
            case DType::INT32:
                impl_ = std::make_shared<TensorImpl>(shape, value ? std::int32_t{1} : std::int32_t{0}, device);
                break;
            case DType::BOOL:
                impl_ = std::make_shared<TensorImpl>(shape, value, device);
                break;
        }
    }

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
                         DeviceType device,
                         DType dtype) {
        return Tensor(shape, 0.0f, device, dtype);
    }

    Tensor Tensor::ones(const std::vector<size_t>& shape,
                        DeviceType device,
                        DType dtype) {
        return Tensor(shape, 1.0f, device, dtype);
    }

    Tensor Tensor::full(const std::vector<size_t>& shape,
                        float value,
                        DeviceType device,
                        DType dtype) {
        return Tensor(shape, value, device, dtype);
    }

    Tensor Tensor::full(const std::vector<size_t>& shape,
                        double value,
                        DeviceType device,
                        DType dtype) {
        return Tensor(shape, value, device, dtype);
    }

    Tensor Tensor::full(const std::vector<size_t>& shape,
                        std::int32_t value,
                        DeviceType device,
                        DType dtype) {
        return Tensor(shape, value, device, dtype);
    }

    Tensor Tensor::full(const std::vector<size_t>& shape,
                        bool value,
                        DeviceType device,
                        DType dtype) {
        return Tensor(shape, value, device, dtype);
    }

    Tensor Tensor::randn(const std::vector<size_t>& shape,
                         DeviceType device,
                         DType dtype) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        static thread_local std::mt19937_64 gen((unsigned)std::random_device{}());

        if (dtype == DType::FLOAT64) {
            std::normal_distribution<double> d(0.0, 1.0);
            std::vector<double> data(total);
            for (size_t i = 0; i < total; ++i) data[i] = d(gen);
            return Tensor(shape, data, device);
        }

        std::normal_distribution<float> d(0.0f, 1.0f);
        std::vector<float> data(total);
        for (size_t i = 0; i < total; ++i) data[i] = d(gen);

        if (dtype == DType::FLOAT32) {
            return Tensor(shape, data, device);
        }

        if (dtype == DType::INT32) {
            std::vector<std::int32_t> out(total);
            for (size_t i = 0; i < total; ++i) {
                out[i] = static_cast<std::int32_t>(std::lrint(data[i]));
            }
            return Tensor(shape, out, device);
        }

        std::vector<bool> out(total);
        for (size_t i = 0; i < total; ++i) {
            out[i] = data[i] > 0.0f;
        }
        return Tensor(shape, out, device);
    }

    Tensor Tensor::from_ptr(const std::vector<size_t>& shape,
                           float* data_ptr,
                           std::shared_ptr<TensorImpl> owner,
                           DeviceType device,
                           DType dtype) {
        auto impl = std::make_shared<TensorImpl>(shape, data_ptr, owner, device, dtype);
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

    DType Tensor::dtype() const {
        const auto impl = require_impl(__func__);
        return static_cast<const TensorImpl&>(*impl).dtype();
    }


    void Tensor::print() const {
        const auto impl = require_impl(__func__);
        const auto &s = impl->shape();
        const auto& logical = data();

        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << s[i];
        }
        std::cout << "], dtype=" << dtype_name(impl->dtype()) << ", values=[";

        for (size_t i = 0; i < logical.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << logical[i];
            if (i >= 31) { std::cout << ", ..."; break; }
        }
        std::cout << "])\n";
    }

    void Tensor::print_pretty() const {
        // small pretty printer: only for 1D or 2D tensors
        const auto impl = require_impl(__func__);
        const auto &s = impl->shape();
        const auto &logical = data();

        if (s.size() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < s[0]; ++i) {
                if (i) std::cout << ", ";
                std::cout << logical[i];
            }
            std::cout << "]\n";
        } else if (s.size() == 2) {
            for (size_t r = 0; r < s[0]; ++r) {
                std::cout << "[";
                for (size_t c = 0; c < s[1]; ++c) {
                    if (c) std::cout << ", ";
                    const size_t offset = r * s[1] + c;
                    std::cout << logical[offset];
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
        const auto impl = require_impl(__func__);
        const int rank = static_cast<int>(ndim());

        // Normalize negative dimension
        int norm_dim = dim;
        if (norm_dim < 0) {
            norm_dim += rank;
        }

        if (norm_dim < 0 || norm_dim >= rank) {
            throw std::runtime_error("slice: dimension " + std::to_string(dim) +
                                   " out of range for tensor with " + std::to_string(rank) + " dimensions");
        }

        auto new_shape = shape();
        const auto& base_stride = impl->stride();
        std::vector<size_t> new_stride = base_stride;

        const int64_t dim_size = static_cast<int64_t>(new_shape[norm_dim]);

        // Default step is 1
        const int64_t step_value = step.value_or(1);
        if (step_value <= 0) {
            throw std::runtime_error("slice: step must be positive, got " + std::to_string(step_value));
        }

        // Helper function to clamp indices to valid range
        const auto clamp_index = [dim_size](int64_t idx) -> int64_t {
            if (dim_size == 0) {
                return 0;
            }
            // Handle negative indices (Python-style)
            if (idx < 0) {
                idx += dim_size;
            }
            // Clamp to valid range [0, dim_size]
            if (idx < 0) {
                idx = 0;
            }
            if (idx > dim_size) {
                idx = dim_size;
            }
            return idx;
        };

        // Normalize start and end indices
        int64_t start_idx = clamp_index(start.value_or(0));
        int64_t end_idx = clamp_index(end.value_or(dim_size));

        // Compute slice length
        size_t slice_len = 0;
        if (end_idx > start_idx && dim_size > 0) {
            const int64_t distance = end_idx - start_idx;
            slice_len = static_cast<size_t>((distance + step_value - 1) / step_value);
        }

        // Update shape and stride for sliced dimension
        new_shape[norm_dim] = slice_len;
        new_stride[norm_dim] = base_stride[norm_dim] * static_cast<size_t>(step_value);

        // Calculate offset from base data
        size_t offset_delta = 0;
        if (dim_size > 0 && start_idx > 0) {
            offset_delta = static_cast<size_t>(start_idx) * base_stride[norm_dim];
        }

        // Create view with modified shape, stride, and offset
        auto view_impl = std::make_shared<TensorImpl>(impl, new_shape, new_stride, offset_delta);
        return Tensor(std::move(view_impl));
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
        if (impl->dtype() == DType::FLOAT32 && impl->can_expose_direct_data_buffer()) {
            return *this;  // Already backed by a direct compact buffer
        }
        return clone();
    }

    Tensor Tensor::clone() const {
        const auto impl = require_impl(__func__);
        std::vector<std::uint8_t> copied;
        impl->materialize_logical_data_bytes(copied);
        const auto total = numel();

        switch (impl->dtype()) {
            case DType::BOOL: {
                std::vector<bool> out(total, false);
                for (size_t i = 0; i < total; ++i) {
                    out[i] = copied[i] != 0;
                }
                return Tensor(shape(), out, device_type());
            }
            case DType::INT32: {
                std::vector<std::int32_t> out(total);
                std::memcpy(out.data(), copied.data(), total * sizeof(std::int32_t));
                return Tensor(shape(), out, device_type());
            }
            case DType::FLOAT32: {
                std::vector<float> out(total);
                std::memcpy(out.data(), copied.data(), total * sizeof(float));
                return Tensor(shape(), out, device_type());
            }
            case DType::FLOAT64: {
                std::vector<double> out(total);
                std::memcpy(out.data(), copied.data(), total * sizeof(double));
                return Tensor(shape(), out, device_type());
            }
        }

        throw std::runtime_error("clone: unsupported dtype");
    }

    Tensor Tensor::astype(DType target_dtype) const {
        const auto impl = require_impl(__func__);
        const auto source_dtype = impl->dtype();

        if (target_dtype == source_dtype) {
            return clone();
        }

        const auto total = numel();
        const auto& logical = data();

        switch (target_dtype) {
            case DType::BOOL: {
                std::vector<bool> out(total, false);
                for (size_t i = 0; i < total; ++i) {
                    out[i] = logical[i] != 0.0f;
                }
                return Tensor(shape(), out, device_type());
            }
            case DType::INT32: {
                std::vector<std::int32_t> out(total, 0);
                for (size_t i = 0; i < total; ++i) {
                    out[i] = static_cast<std::int32_t>(logical[i]);
                }
                return Tensor(shape(), out, device_type());
            }
            case DType::FLOAT32:
                return Tensor(shape(), logical, device_type());
            case DType::FLOAT64: {
                std::vector<double> out(total, 0.0);
                for (size_t i = 0; i < total; ++i) {
                    out[i] = static_cast<double>(logical[i]);
                }
                return Tensor(shape(), out, device_type());
            }
        }

        throw std::runtime_error("astype: unsupported target dtype");
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
