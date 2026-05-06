#include "cpptensor/tensor/tensorimpl.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef BUILD_CUDA
#include <cuda_runtime.h>
#endif

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

#ifdef BUILD_CUDA
        inline void cuda_check(cudaError_t status, const char* what) {
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA failure during ") + what + ": " + cudaGetErrorString(status));
            }
        }
#endif

        [[noreturn]] void throw_cuda_not_built() {
            throw std::runtime_error(
                "CUDA transfer requested, but cpptensor was built without BUILD_CUDA support");
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

        if (device_ == DeviceType::CUDA) {
#ifdef BUILD_CUDA
            ensure_resident(DeviceType::CUDA);
#endif
        }
    }

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

        if (device_ == DeviceType::CUDA) {
#ifdef BUILD_CUDA
            ensure_resident(DeviceType::CUDA);
#endif
        }
    }

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
    }

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
    }

    TensorImpl::~TensorImpl() {
        release_cuda_storage();
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
            ensure_resident(DeviceType::CPU);
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
            ensure_resident(DeviceType::CPU);
            host_data_valid_ = true;
            cuda_data_valid_ = false;
            return data_;
        }

        throw std::runtime_error(
            "Tensor::data(): mutable access is unavailable for sliced, permuted, "
            "transposed, or pointer-backed views. Call contiguous() or clone() first.");
    }

    void TensorImpl::ensure_resident(DeviceType dev) const {
        if (data_ptr_ != nullptr) {
            throw std::runtime_error(
                "TensorImpl::ensure_resident: pointer-backed views do not support explicit device transfer");
        }

        if (base_impl_) {
            base_impl_->ensure_resident(dev);
            return;
        }

        const size_t total = numel();
        const size_t bytes = total * sizeof(float);

        if (dev == DeviceType::CPU) {
            if (host_data_valid_) {
                return;
            }

            data_.resize(total);

#ifdef BUILD_CUDA
            if (total != 0) {
                cuda_check(cudaMemcpy(data_.data(), cuda_data_, bytes, cudaMemcpyDeviceToHost),
                           "device-to-host copy");
            }
#else
            throw_cuda_not_built();
#endif

            host_data_valid_ = true;
            return;
        }

#ifdef BUILD_CUDA
        if (cuda_data_valid_) {
            return;
        }

        if (total != 0 && cuda_data_ == nullptr) {
            cuda_check(cudaMalloc(&cuda_data_, bytes), "device allocation");
        }

        if (host_data_valid_ && total != 0) {
            cuda_check(cudaMemcpy(cuda_data_, data_.data(), bytes, cudaMemcpyHostToDevice),
                       "host-to-device copy");
        }

        cuda_data_valid_ = true;
#else
        throw_cuda_not_built();
#endif
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

        ensure_resident(DeviceType::CPU);
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

        ensure_resident(DeviceType::CPU);
        host_data_valid_ = true;
        cuda_data_valid_ = false;
        return data_.data() + offset_;
    }

    const float* TensorImpl::backend_data_ptr(DeviceType dev) const {
        if (dev == DeviceType::CPU) {
            return data_ptr();
        }

        if (data_ptr_) {
            throw std::runtime_error(
                "TensorImpl::backend_data_ptr(CUDA): pointer-backed views do not expose CUDA storage");
        }

        if (base_impl_) {
            return base_impl_->backend_data_ptr(dev) + offset_;
        }

        ensure_resident(DeviceType::CUDA);
        return cuda_data_ + offset_;
    }

    float* TensorImpl::backend_data_ptr(DeviceType dev) {
        if (dev == DeviceType::CPU) {
            return data_ptr();
        }

        if (data_ptr_) {
            throw std::runtime_error(
                "TensorImpl::backend_data_ptr(CUDA): pointer-backed views do not expose CUDA storage");
        }

        if (base_impl_) {
            return base_impl_->backend_data_ptr(dev) + offset_;
        }

        ensure_resident(DeviceType::CUDA);
        host_data_valid_ = false;
        cuda_data_valid_ = true;
        return cuda_data_ + offset_;
    }

    std::shared_ptr<TensorImpl> TensorImpl::copy_to(DeviceType dev) const {
        std::vector<float> logical;
        materialize_logical_data(logical);

        auto copied = std::make_shared<TensorImpl>(shape_, logical, DeviceType::CPU);
        copied->set_device(dev);
        return copied;
    }

    void TensorImpl::release_cuda_storage() const {
#ifdef BUILD_CUDA
        if (cuda_data_ != nullptr) {
            cudaFree(cuda_data_);
            cuda_data_ = nullptr;
        }
#else
        cuda_data_ = nullptr;
#endif
        cuda_data_valid_ = false;
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
    void TensorImpl::set_device(DeviceType dev) {
        if (device_ == dev) {
            return;
        }
        ensure_resident(dev);
        device_ = dev;
    }

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
