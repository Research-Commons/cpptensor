#include "cpptensor/tensor/tensorimpl.hpp"
#include <numeric>
#include <stdexcept>

namespace cpptensor {

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           const std::vector<float>& data,
                           DeviceType device)
        : data_(data),
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
        : shape_(shape),
          device_(device)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        data_.assign(total, fill_value);
        stride_ = compute_strides(shape_);
    }

    const std::vector<float>& TensorImpl::data() const { return data_; }
    std::vector<float>& TensorImpl::data() { return data_; }

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

} // namespace cppgrad