#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/ops/reduction/sum.hpp"
#include "cpptensor/ops/reduction/mean.hpp"
#include "cpptensor/ops/reduction/max.hpp"
#include "cpptensor/ops/reduction/min.hpp"

#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <bit>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <array>
#include <fstream>
#include <limits>

namespace cpptensor {

    namespace {
        constexpr std::array<char, 8> kTensorCheckpointMagic = {'C', 'P', 'T', 'E', 'N', 'S', 'R', '\0'};
        constexpr uint16_t kTensorCheckpointVersion = 1;
        constexpr uint8_t kTensorCheckpointDTypeF32 = 1;

        [[noreturn]] void throw_checkpoint_error(const std::string& msg) {
            throw std::runtime_error("tensor checkpoint I/O error: " + msg);
        }

        void ensure_stream_ok(const std::ios& stream, const std::string& context) {
            if (!stream) {
                throw_checkpoint_error(context);
            }
        }

        void write_u8(std::ostream& out, uint8_t value) {
            out.put(static_cast<char>(value));
            ensure_stream_ok(out, "failed to write checkpoint byte");
        }

        void write_u16_le(std::ostream& out, uint16_t value) {
            write_u8(out, static_cast<uint8_t>(value & 0xFFu));
            write_u8(out, static_cast<uint8_t>((value >> 8u) & 0xFFu));
        }

        void write_u64_le(std::ostream& out, uint64_t value) {
            for (int shift = 0; shift < 64; shift += 8) {
                write_u8(out, static_cast<uint8_t>((value >> shift) & 0xFFu));
            }
        }

        uint8_t read_u8(std::istream& in) {
            const int value = in.get();
            if (value == EOF) {
                throw_checkpoint_error("unexpected end-of-file");
            }
            return static_cast<uint8_t>(value);
        }

        uint16_t read_u16_le(std::istream& in) {
            uint16_t value = 0;
            value |= static_cast<uint16_t>(read_u8(in));
            value |= static_cast<uint16_t>(read_u8(in)) << 8u;
            return value;
        }

        uint64_t read_u64_le(std::istream& in) {
            uint64_t value = 0;
            for (int shift = 0; shift < 64; shift += 8) {
                value |= static_cast<uint64_t>(read_u8(in)) << shift;
            }
            return value;
        }

        void write_f32_le(std::ostream& out, float value) {
            const uint32_t bits = std::bit_cast<uint32_t>(value);
            for (int shift = 0; shift < 32; shift += 8) {
                write_u8(out, static_cast<uint8_t>((bits >> shift) & 0xFFu));
            }
        }

        float read_f32_le(std::istream& in) {
            uint32_t bits = 0;
            for (int shift = 0; shift < 32; shift += 8) {
                bits |= static_cast<uint32_t>(read_u8(in)) << shift;
            }
            return std::bit_cast<float>(bits);
        }

        uint8_t encode_device_type(DeviceType device) {
            switch (device) {
                case DeviceType::CPU: return 0;
                case DeviceType::CUDA: return 1;
            }
            throw_checkpoint_error("unsupported device enum value");
        }

        DeviceType decode_device_type(uint8_t value) {
            switch (value) {
                case 0: return DeviceType::CPU;
                case 1: return DeviceType::CUDA;
                default:
                    throw_checkpoint_error("unsupported device code in checkpoint");
            }
        }

        size_t checked_mul(size_t a, size_t b) {
            if (a == 0 || b == 0) {
                return 0;
            }
            if (a > std::numeric_limits<size_t>::max() / b) {
                throw_checkpoint_error("shape metadata overflows size_t");
            }
            return a * b;
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
        if (impl->can_expose_direct_data_buffer()) {
            return *this;  // Already backed by a direct compact buffer
        }

        return Tensor(shape(), copy_logical_data(*this), device_type());
    }

    Tensor Tensor::clone() const {
        require_impl(__func__);
        return Tensor(shape(), copy_logical_data(*this), device_type());
    }

    void Tensor::save(const std::string& path) const {
        require_impl(__func__);

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            throw_checkpoint_error("unable to open file for writing: " + path);
        }

        const auto tensor_shape = shape();
        const auto tensor_data = copy_logical_data(*this);  // materializes views

        out.write(kTensorCheckpointMagic.data(), static_cast<std::streamsize>(kTensorCheckpointMagic.size()));
        ensure_stream_ok(out, "failed to write checkpoint magic");
        write_u16_le(out, kTensorCheckpointVersion);
        write_u16_le(out, 0);  // reserved flags for future compatibility
        write_u8(out, kTensorCheckpointDTypeF32);
        write_u8(out, encode_device_type(device_type()));
        write_u16_le(out, 0);  // reserved padding

        write_u64_le(out, static_cast<uint64_t>(tensor_shape.size()));
        write_u64_le(out, static_cast<uint64_t>(tensor_data.size()));

        for (size_t dim : tensor_shape) {
            write_u64_le(out, static_cast<uint64_t>(dim));
        }
        for (float value : tensor_data) {
            write_f32_le(out, value);
        }

        out.flush();
        ensure_stream_ok(out, "failed while finalizing checkpoint write");
    }

    Tensor Tensor::load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw_checkpoint_error("unable to open file for reading: " + path);
        }

        std::array<char, kTensorCheckpointMagic.size()> magic{};
        in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (in.gcount() != static_cast<std::streamsize>(magic.size()) || magic != kTensorCheckpointMagic) {
            throw_checkpoint_error("file is not a cpptensor checkpoint");
        }

        const uint16_t version = read_u16_le(in);
        if (version != kTensorCheckpointVersion) {
            throw_checkpoint_error("unsupported checkpoint version: " + std::to_string(version));
        }

        const uint16_t flags = read_u16_le(in);
        if (flags != 0) {
            throw_checkpoint_error("unsupported checkpoint flags for version 1");
        }

        const uint8_t dtype_code = read_u8(in);
        if (dtype_code != kTensorCheckpointDTypeF32) {
            throw_checkpoint_error("unsupported dtype code in checkpoint");
        }

        const DeviceType device = decode_device_type(read_u8(in));
        (void)read_u16_le(in);  // reserved padding

        const uint64_t ndim_u64 = read_u64_le(in);
        const uint64_t numel_u64 = read_u64_le(in);

        if (ndim_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw_checkpoint_error("ndim metadata exceeds platform size limits");
        }
        if (numel_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw_checkpoint_error("numel metadata exceeds platform size limits");
        }

        const size_t ndim = static_cast<size_t>(ndim_u64);
        const size_t numel = static_cast<size_t>(numel_u64);

        std::vector<size_t> loaded_shape;
        loaded_shape.reserve(ndim);
        size_t expected_numel = 1;
        for (size_t i = 0; i < ndim; ++i) {
            const uint64_t dim_u64 = read_u64_le(in);
            if (dim_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw_checkpoint_error("shape metadata exceeds platform size limits");
            }
            const size_t dim = static_cast<size_t>(dim_u64);
            loaded_shape.push_back(dim);
            expected_numel = checked_mul(expected_numel, dim);
        }

        if (loaded_shape.empty()) {
            expected_numel = 1;  // scalar convention
        }

        if (expected_numel != numel) {
            throw_checkpoint_error("shape/numel metadata mismatch");
        }

        std::vector<float> loaded_data(numel);
        for (size_t i = 0; i < numel; ++i) {
            loaded_data[i] = read_f32_le(in);
        }

        if (in.peek() != EOF) {
            throw_checkpoint_error("unexpected trailing bytes in checkpoint");
        }

        return Tensor(loaded_shape, loaded_data, device);
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
