#include "cpptensor/tensor/tensor.hpp"

#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>

namespace cpptensor {

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
        Tensor result;
        result.impl_ = impl;
        return result;
    }

    // ---------- Shape & Info ----------
    std::vector<size_t> Tensor::shape() const { return impl_->shape(); }
    size_t Tensor::numel() const { return impl_->numel(); }
    size_t Tensor::ndim() const { return impl_->shape().size(); }
    DeviceType Tensor::device_type() const { return impl_->device(); }


    void Tensor::print() const {
        const auto &s = impl_->shape();
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << s[i];
        }
        std::cout << "], values=[";
        const auto &d = impl_->data();
        for (size_t i = 0; i < d.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << d[i];
            if (i >= 31) { std::cout << ", ..."; break; }
        }
        std::cout << "])\n";
    }

    void Tensor::print_pretty() const {
        // small pretty printer: only for 1D or 2D tensors
        const auto s = impl_->shape();
        const auto &d = impl_->data();
        if (s.size() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < s[0]; ++i) {
                if (i) std::cout << ", ";
                std::cout << d[i];
            }
            std::cout << "]\n";
        } else if (s.size() == 2) {
            for (size_t r = 0; r < s[0]; ++r) {
                std::cout << "[";
                for (size_t c = 0; c < s[1]; ++c) {
                    if (c) std::cout << ", ";
                    std::cout << d[r * s[1] + c];
                }
                std::cout << "]\n";
            }
        } else {
            print();
        }
    }

    // Data access
    const std::vector<float>& Tensor::data() const { return impl_->data(); }
    std::vector<float>& Tensor::data() { return impl_->data(); }
    const std::vector<size_t>& Tensor::stride() const { return impl_->stride(); }
    std::vector<size_t>& Tensor::stride(){ return impl_->stride(); }
    std::shared_ptr<TensorImpl> Tensor::impl() const { return impl_; }

    // =============== Tensor Manipulation Operations ===============

    Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
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
        auto view_impl = std::make_shared<TensorImpl>(impl_, new_shape);

        Tensor result;
        result.impl_ = view_impl;
        return result;
    }

    Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
        if (is_contiguous()) {
            return view(new_shape);  // Zero-copy if possible
        } else {
            // Must copy to make contiguous first
            return contiguous().view(new_shape);
        }
    }

    Tensor Tensor::flatten(int start_dim, int end_dim) const {
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

    Tensor Tensor::squeeze(int dim) const {
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

        // Handle edge case: squeezing all dims results in scalar
        if (new_shape.empty() && numel() == 1) {
            new_shape.push_back(1);  // Keep as 1D tensor with single element
        }

        return reshape(new_shape);
    }

    Tensor Tensor::unsqueeze(int dim) const {
        auto sh = shape();
        int ndims = static_cast<int>(sh.size());

        // Normalize dimension (allow -1 for "append")
        int norm_dim = dim;
        if (dim < 0) norm_dim = dim + ndims + 1;  // +1 because we're adding a dimension

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
        // This is zero-copy - just changes how we interpret the data
        auto view_impl = std::make_shared<TensorImpl>(impl_, new_shape, new_stride);

        Tensor result;
        result.impl_ = view_impl;
        return result;
    }

    Tensor Tensor::transpose(int dim0, int dim1) const {
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
        if (is_contiguous()) {
            return *this;  // Already contiguous, return shallow copy
        }

        // Need to actually copy and reorder data
        auto sh = shape();
        auto st = stride();
        size_t total = numel();
        std::vector<float> new_data(total);

        // Copy data in contiguous (row-major) order
        std::vector<size_t> indices(sh.size(), 0);
        const float* src = impl_->data().data();

        for (size_t i = 0; i < total; ++i) {
            // Compute offset in original tensor using strides
            size_t src_offset = 0;
            for (size_t d = 0; d < sh.size(); ++d) {
                src_offset += indices[d] * st[d];
            }

            new_data[i] = src[src_offset];

            // Increment multi-dimensional index
            for (int d = static_cast<int>(sh.size()) - 1; d >= 0; --d) {
                if (++indices[d] < sh[d]) break;
                indices[d] = 0;
            }
        }

        return Tensor(sh, new_data, device_type());
    }

    Tensor Tensor::clone() const {
        // Deep copy - create new data buffer
        return Tensor(shape(), impl_->data(), device_type());
    }

} // namespace cpptensor

// // -------- Reductions (only global dim implemented for now) --------
// Tensor Tensor::sum(int dim, bool keepdim) const {
//     if (dim != -1) throw std::runtime_error("sum(dim) not implemented yet (only global sum)");
//     float acc = 0.0f;
//     for (float v : impl_->data()) acc += v;
//     // return scalar as shape {1}
//     return Tensor({1}, std::vector<float>{acc}, false);
// }
//
// Tensor Tensor::mean(int dim, bool keepdim) const {
//     if (dim != -1) throw std::runtime_error("mean(dim) not implemented yet (only global mean)");
//     float acc = 0.0f;
//     for (float v : impl_->data()) acc += v;
//     return Tensor({1}, std::vector<float>{acc / (float)numel()}, false);
// }
//
// Tensor Tensor::max(int dim, bool keepdim) const {
//     if (dim != -1) throw std::runtime_error("max(dim) not implemented yet (only global max)");
//     const auto &d = impl_->data();
//     if (d.empty()) throw std::runtime_error("max on empty tensor");
//     float m = d[0];
//     for (auto v : d) if (v > m) m = v;
//     return Tensor({1}, std::vector<float>{m}, false);
// }
//
// // -------- Elementwise operators (simple implementations) --------
// static void check_shapes_match(const Tensor& a, const Tensor& b) {
//     if (a.shape() != b.shape()) throw std::runtime_error("shape mismatch in binary op");
// }
//
// Tensor operator+(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) Od[i] = Ad[i] + Bd[i];
//     return out;
// }
//
// Tensor operator-(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) Od[i] = Ad[i] - Bd[i];
//     return out;
// }
//
// Tensor operator*(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) Od[i] = Ad[i] * Bd[i];
//     return out;
// }
//
// Tensor operator/(const Tensor& a, const Tensor& b) {
//     check_shapes_match(a,b);
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad() || b.requires_grad());
//     const auto &Ad = a.data();
//     const auto &Bd = b.data();
//     auto &Od = out.data();
//     size_t N = Ad.size();
//     for (size_t i = 0; i < N; ++i) {
//         if (Bd[i] == 0.0f) Od[i] = std::numeric_limits<float>::infinity();
//         else Od[i] = Ad[i] / Bd[i];
//     }
//     return out;
// }
//
// // Scalar ops
// Tensor operator+(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] + s;
//     return out;
// }
// Tensor operator+(float s, const Tensor& a) { return a + s; }
//
// Tensor operator-(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] - s;
//     return out;
// }
// Tensor operator-(float s, const Tensor& a) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = s - Ad[i];
//     return out;
// }
//
// Tensor operator*(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] * s;
//     return out;
// }
// Tensor operator*(float s, const Tensor& a) { return a * s; }
//
// Tensor operator/(const Tensor& a, float s) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     if (s == 0.0f) {
//         for (size_t i = 0; i < Ad.size(); ++i) Od[i] = std::numeric_limits<float>::infinity();
//     } else {
//         for (size_t i = 0; i < Ad.size(); ++i) Od[i] = Ad[i] / s;
//     }
//     return out;
// }
// Tensor operator/(float s, const Tensor& a) {
//     auto out = Tensor::full(a.shape(), 0.0f, a.requires_grad());
//     const auto &Ad = a.data();
//     auto &Od = out.data();
//     for (size_t i = 0; i < Ad.size(); ++i) {
//         if (Ad[i] == 0.0f) Od[i] = std::numeric_limits<float>::infinity();
//         else Od[i] = s / Ad[i];
//     }
//     return out;
// }
//
// Tensor operator-(const Tensor& a) {
//     return a * (-1.0f);
// }
