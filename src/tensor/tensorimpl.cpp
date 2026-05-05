#include "cpptensor/tensor/tensorimpl.hpp"
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cpptensor {

    namespace {
        bool has_row_major_stride(const std::vector<size_t>& shape,
                                  const std::vector<size_t>& stride) {
            if (shape.size() != stride.size()) {
                return false;
            }

            size_t expected_stride = 1;
            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                if (stride[static_cast<size_t>(i)] != expected_stride) {
                    return false;
                }
                expected_stride *= shape[static_cast<size_t>(i)];
            }
            return true;
        }
    } // namespace

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           const std::vector<float>& data,
                           DeviceType device)
        : data_(data),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        if (data_.size() != total) {
            throw std::runtime_error("TensorImpl: data size does not match shape");
        }
        stride_ = compute_strides(shape_);
    }

    //protected const
    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           float fill_value,
                           DeviceType device)
        : data_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        data_.assign(total, fill_value);
        stride_ = compute_strides(shape_);
    }

    // View constructor - shares data with base
    TensorImpl::TensorImpl(std::shared_ptr<TensorImpl> base,
                           const std::vector<size_t>& new_shape,
                           const std::vector<size_t>& new_stride,
                           size_t offset)
        : data_(),
          base_impl_(std::move(base)),
          data_ptr_(nullptr),
          stride_(),
          offset_(offset),
          grad_fn_(nullptr),
          shape_(new_shape),
          device_(base_impl_->device_)
    {
        if (new_stride.empty()) {
            stride_ = compute_strides(new_shape);
        } else {
            stride_ = new_stride;
        }
        // data_ is empty - we delegate to base
    }

    // Pointer-based view constructor - wraps raw pointer
    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           float* data_ptr,
                           std::shared_ptr<TensorImpl> owner,
                           DeviceType device)
        : data_(),
          base_impl_(std::move(owner)),
          data_ptr_(data_ptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device)
    {
        stride_ = compute_strides(shape);
        // data_ is empty - we use data_ptr_ instead
    }

    bool TensorImpl::can_expose_direct_data_buffer() const {
        if (data_ptr_ != nullptr) {
            return false;
        }

        if (!base_impl_) {
            return true;
        }

        if (offset_ != 0) {
            return false;
        }

        if (!has_row_major_stride(shape_, stride_)) {
            return false;
        }

        if (numel() != base_impl_->numel()) {
            return false;
        }

        return base_impl_->can_expose_direct_data_buffer();
    }

    void TensorImpl::materialize_logical_data(std::vector<float>& out) const {
        const size_t total = numel();
        out.resize(total);

        if (total == 0) {
            return;
        }

        const float* src = data_ptr();
        if (shape_.empty()) {
            out[0] = src[0];
            return;
        }

        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < total; ++i) {
            size_t src_offset = 0;
            for (size_t d = 0; d < shape_.size(); ++d) {
                src_offset += indices[d] * stride_[d];
            }

            out[i] = src[src_offset];

            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                const size_t dim = static_cast<size_t>(d);
                if (++indices[dim] < shape_[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    const std::vector<float>& TensorImpl::data() const {
        if (can_expose_direct_data_buffer()) {
            if (base_impl_) {
                return base_impl_->data();
            }
            return data_;
        }

        materialize_logical_data(logical_data_cache_);
        return logical_data_cache_;
    }

    std::vector<float>& TensorImpl::data() {
        if (can_expose_direct_data_buffer()) {
            if (base_impl_) {
                return base_impl_->data();
            }
            return data_;
        }

        throw std::runtime_error(
            "Tensor::data(): mutable access is unavailable for sliced, permuted, "
            "transposed, or pointer-backed views. Call contiguous() or clone() first.");
    }

    const float* TensorImpl::data_ptr() const {
        // Pointer-based view: return the raw pointer
        if (data_ptr_) {
            return data_ptr_ + offset_;
        }
        // View: delegate to base
        if (base_impl_) {
            return base_impl_->data_ptr() + offset_;
        }
        // Own data: return pointer to vector
        return data_.data() + offset_;
    }

    float* TensorImpl::data_ptr() {
        // Pointer-based view: return the raw pointer
        if (data_ptr_) {
            return data_ptr_ + offset_;
        }
        // View: delegate to base
        if (base_impl_) {
            return base_impl_->data_ptr() + offset_;
        }
        // Own data: return pointer to vector
        return data_.data() + offset_;
    }

    std::vector<size_t>& TensorImpl::stride(){ return stride_; }
    const std::vector<size_t>& TensorImpl::stride() const { return stride_; }

    size_t TensorImpl::offset() const { return offset_; }

    const std::vector<size_t>& TensorImpl::shape() const { return shape_; }
    size_t TensorImpl::numel() const {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        return total;
    }

    DeviceType TensorImpl::device() const { return device_; }
    void TensorImpl::set_device(DeviceType dev) { device_ = dev; }

    bool TensorImpl::has_called_backward() const {
        // TODO: Implement autograd backward tracking
        // For now, always return false as autograd is not fully implemented
        return false;
    }

    void TensorImpl::set_has_called_backward(bool val) {
        // TODO: Implement autograd backward tracking
        // Placeholder for future autograd implementation
        (void)val;  // Suppress unused parameter warning
    }

    std::vector<size_t> TensorImpl::compute_strides(const std::vector<size_t>& shape){
        int n = static_cast<int>(shape.size());
        std::vector<size_t> stride((size_t)n, 0);
        if (n == 0) return stride;
        stride[(size_t)n - 1] = 1;
        for (int i = n - 2; i >= 0; --i) {
            stride[(size_t)i] = stride[(size_t)i + 1] * shape[(size_t)i + 1];
        }
        return stride;
    }

} // namespace cpptensor
