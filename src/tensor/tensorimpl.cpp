#include "cpptensor/tensor/tensorimpl.hpp"
#include <numeric>
#include <stdexcept>

namespace cpptensor {

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           const std::vector<float>& data,
                           DeviceType device)
        : data_(data),
          shape_(shape),
          device_(device),
          data_ptr_(nullptr)
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
        : shape_(shape),
          device_(device),
          data_ptr_(nullptr)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        data_.assign(total, fill_value);
        stride_ = compute_strides(shape_);
    }

    // View constructor - shares data with base
    TensorImpl::TensorImpl(std::shared_ptr<TensorImpl> base,
                           const std::vector<size_t>& new_shape,
                           const std::vector<size_t>& new_stride)
        : base_impl_(base),  // Keep base alive
          shape_(new_shape),
          device_(base->device_),
          data_ptr_(nullptr)  // Views delegate through base_impl_
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
        : base_impl_(owner),  // Keep owner alive
          shape_(shape),
          device_(device),
          data_ptr_(data_ptr)  // Store raw pointer
    {
        stride_ = compute_strides(shape);
        // data_ is empty - we use data_ptr_ instead
    }

    const std::vector<float>& TensorImpl::data() const {
        // If this is a view, delegate to base
        if (base_impl_) {
            return base_impl_->data();
        }
        return data_;
    }

    std::vector<float>& TensorImpl::data() {
        // If this is a view, delegate to base
        if (base_impl_) {
            return base_impl_->data();
        }
        return data_;
    }

    const float* TensorImpl::data_ptr() const {
        // Pointer-based view: return the raw pointer
        if (data_ptr_) {
            return data_ptr_;
        }
        // View: delegate to base
        if (base_impl_) {
            return base_impl_->data_ptr();
        }
        // Own data: return pointer to vector
        return data_.data();
    }

    float* TensorImpl::data_ptr() {
        // Pointer-based view: return the raw pointer
        if (data_ptr_) {
            return data_ptr_;
        }
        // View: delegate to base
        if (base_impl_) {
            return base_impl_->data_ptr();
        }
        // Own data: return pointer to vector
        return data_.data();
    }

    std::vector<size_t>& TensorImpl::stride(){ return stride_; }
    const std::vector<size_t>& TensorImpl::stride() const { return stride_; }

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