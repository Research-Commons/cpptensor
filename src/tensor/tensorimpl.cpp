#include "cpptensor/tensor/tensorimpl.hpp"

#include <cstring>
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

        size_t compute_numel(const std::vector<size_t>& shape) {
            size_t total = 1;
            for (size_t dim : shape) {
                total *= dim;
            }
            return total;
        }

        std::vector<std::uint8_t> bool_to_u8_storage(const std::vector<bool>& values) {
            std::vector<std::uint8_t> out(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                out[i] = values[i] ? std::uint8_t{1} : std::uint8_t{0};
            }
            return out;
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
        : storage_(data),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(),
          grad_data_(),
          requires_grad_(false),
          has_called_backward_(false),
          shape_(shape),
          device_(device),
          dtype_(DType::FLOAT32)
    {
        const size_t total = compute_numel(shape_);
        if (data.size() != total) {
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
                           const std::vector<double>& data,
                           DeviceType device)
        : storage_(data),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(),
          grad_data_(),
          requires_grad_(false),
          has_called_backward_(false),
          shape_(shape),
          device_(device),
          dtype_(DType::FLOAT64)
    {
        const size_t total = compute_numel(shape_);
        if (data.size() != total) {
            throw std::runtime_error("TensorImpl: data size does not match shape");
        }
        stride_ = compute_strides(shape_);
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           const std::vector<std::int32_t>& data,
                           DeviceType device)
        : storage_(data),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device),
          dtype_(DType::INT32)
    {
        const size_t total = compute_numel(shape_);
        if (data.size() != total) {
            throw std::runtime_error("TensorImpl: data size does not match shape");
        }
        stride_ = compute_strides(shape_);
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           const std::vector<bool>& data,
                           DeviceType device)
        : storage_(bool_to_u8_storage(data)),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device),
          dtype_(DType::BOOL)
    {
        const size_t total = compute_numel(shape_);
        if (data.size() != total) {
            throw std::runtime_error("TensorImpl: data size does not match shape");
        }
        stride_ = compute_strides(shape_);
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           float fill_value,
                           DeviceType device)
        : storage_(std::vector<float>(compute_numel(shape), fill_value)),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device),
          dtype_(DType::FLOAT32)
    {
        stride_ = compute_strides(shape_);

        if (device_ == DeviceType::CUDA) {
#ifdef BUILD_CUDA
            ensure_resident(DeviceType::CUDA);
#endif
        }
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           double fill_value,
                           DeviceType device)
        : storage_(std::vector<double>(compute_numel(shape), fill_value)),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device),
          dtype_(DType::FLOAT64)
    {
        stride_ = compute_strides(shape_);
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           std::int32_t fill_value,
                           DeviceType device)
        : storage_(std::vector<std::int32_t>(compute_numel(shape), fill_value)),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device),
          dtype_(DType::INT32)
    {
        stride_ = compute_strides(shape_);
    }

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           bool fill_value,
                           DeviceType device)
        : storage_(std::vector<std::uint8_t>(compute_numel(shape), fill_value ? std::uint8_t{1} : std::uint8_t{0})),
          logical_data_cache_(),
          base_impl_(nullptr),
          data_ptr_(nullptr),
          stride_(),
          offset_(0),
          grad_fn_(nullptr),
          shape_(shape),
          device_(device),
          dtype_(DType::BOOL)
    {
        stride_ = compute_strides(shape_);
    }

    TensorImpl::TensorImpl(std::shared_ptr<TensorImpl> base,
                           const std::vector<size_t>& new_shape,
                           const std::vector<size_t>& new_stride,
                           size_t offset)
        : storage_(std::vector<float>{}),
          logical_data_cache_(),
          base_impl_(std::move(base)),
          data_ptr_(nullptr),
          stride_(),
          offset_(offset),
          grad_fn_(),
          grad_data_(),
          requires_grad_(false),
          has_called_backward_(false),
          shape_(new_shape),
          device_(base_impl_->device_),
          dtype_(base_impl_->dtype_)
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
                           DeviceType device,
                           DType dtype)
        : storage_(std::vector<float>{}),
          logical_data_cache_(),
          base_impl_(std::move(owner)),
          data_ptr_(data_ptr),
          stride_(),
          offset_(0),
          grad_fn_(),
          grad_data_(),
          requires_grad_(false),
          has_called_backward_(false),
          shape_(shape),
          device_(device),
          dtype_(dtype)
    {
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error("TensorImpl::from_ptr currently supports only float32 pointers");
        }
        stride_ = compute_strides(shape);
    }

    TensorImpl::~TensorImpl() {
        release_cuda_storage();
    }

    bool TensorImpl::is_pointer_backed_view() const {
        return data_ptr_ != nullptr;
    }

    std::size_t TensorImpl::element_size_bytes() const {
        return dtype_size_bytes(dtype_);
    }

    bool TensorImpl::can_expose_direct_data_buffer() const {
        if (dtype_ != DType::FLOAT32) {
            return false;
        }

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

        const std::uint8_t* raw = static_cast<const std::uint8_t*>(raw_data_ptr());

        if (shape_.empty()) {
            switch (dtype_) {
                case DType::BOOL:
                    out[0] = static_cast<float>(*reinterpret_cast<const std::uint8_t*>(raw) != 0);
                    return;
                case DType::INT32:
                    out[0] = static_cast<float>(*reinterpret_cast<const std::int32_t*>(raw));
                    return;
                case DType::FLOAT32:
                    out[0] = *reinterpret_cast<const float*>(raw);
                    return;
                case DType::FLOAT64:
                    out[0] = static_cast<float>(*reinterpret_cast<const double*>(raw));
                    return;
            }
        }

        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < total; ++i) {
            size_t src_offset = 0;
            for (size_t d = 0; d < shape_.size(); ++d) {
                src_offset += indices[d] * stride_[d];
            }

            const std::uint8_t* elem_ptr = raw + src_offset * element_size_bytes();
            switch (dtype_) {
                case DType::BOOL:
                    out[i] = static_cast<float>(*reinterpret_cast<const std::uint8_t*>(elem_ptr) != 0);
                    break;
                case DType::INT32:
                    out[i] = static_cast<float>(*reinterpret_cast<const std::int32_t*>(elem_ptr));
                    break;
                case DType::FLOAT32:
                    out[i] = *reinterpret_cast<const float*>(elem_ptr);
                    break;
                case DType::FLOAT64:
                    out[i] = static_cast<float>(*reinterpret_cast<const double*>(elem_ptr));
                    break;
            }

            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                const size_t dim = static_cast<size_t>(d);
                if (++indices[dim] < shape_[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    void TensorImpl::materialize_logical_data_bytes(std::vector<std::uint8_t>& out) const {
        const size_t total = numel();
        const std::size_t element_size = element_size_bytes();
        out.resize(total * element_size);

        if (total == 0) {
            return;
        }

        const std::uint8_t* raw = static_cast<const std::uint8_t*>(raw_data_ptr());

        if (shape_.empty()) {
            std::memcpy(out.data(), raw, element_size);
            return;
        }

        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < total; ++i) {
            size_t src_offset = 0;
            for (size_t d = 0; d < shape_.size(); ++d) {
                src_offset += indices[d] * stride_[d];
            }

            const std::uint8_t* src = raw + src_offset * element_size;
            std::uint8_t* dst = out.data() + i * element_size;
            std::memcpy(dst, src, element_size);

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
            return std::get<std::vector<float>>(storage_);
        }

        materialize_logical_data(logical_data_cache_);
        return logical_data_cache_;
    }

    std::vector<float>& TensorImpl::data() {
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::data(): mutable access is only supported for float32 tensors. "
                "Use astype(float32) first if you need mutable numeric buffer access.");
        }

        if (can_expose_direct_data_buffer()) {
            if (base_impl_) {
                return base_impl_->data();
            }
            ensure_resident(DeviceType::CPU);
            host_data_valid_ = true;
            cuda_data_valid_ = false;
            return std::get<std::vector<float>>(storage_);
        }

        throw std::runtime_error(
            "Tensor::data(): mutable access is unavailable for sliced, permuted, "
            "transposed, or pointer-backed views. Call contiguous() or clone() first.");
    }

    void TensorImpl::ensure_resident(DeviceType dev) const {
        if (dev == DeviceType::CUDA && dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::to(DeviceType::CUDA) currently supports only float32 tensors");
        }

        if (data_ptr_ != nullptr) {
            throw std::runtime_error(
                "TensorImpl::ensure_resident: pointer-backed views do not support explicit device transfer");
        }

        if (base_impl_) {
            base_impl_->ensure_resident(dev);
            return;
        }

        if (dtype_ != DType::FLOAT32) {
            return;
        }

        auto& host = const_cast<std::vector<float>&>(std::get<std::vector<float>>(storage_));
        const size_t total = numel();
        const size_t bytes = total * sizeof(float);

        if (dev == DeviceType::CPU) {
            if (host_data_valid_) {
                return;
            }

#ifdef BUILD_CUDA
            if (total != 0) {
                cuda_check(cudaMemcpy(host.data(), cuda_data_, bytes, cudaMemcpyDeviceToHost),
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
            cuda_check(cudaMemcpy(cuda_data_, host.data(), bytes, cudaMemcpyHostToDevice),
                       "host-to-device copy");
        }

        cuda_data_valid_ = true;
#else
        throw_cuda_not_built();
#endif
    }

    const void* TensorImpl::raw_data_ptr() const {
        if (data_ptr_) {
            if (dtype_ != DType::FLOAT32) {
                throw std::runtime_error("TensorImpl: pointer-backed views currently require float32 dtype");
            }
            return static_cast<const void*>(data_ptr_ + offset_);
        }

        if (base_impl_) {
            const auto* base = static_cast<const std::uint8_t*>(base_impl_->raw_data_ptr());
            return static_cast<const void*>(base + offset_ * element_size_bytes());
        }

        if (dtype_ == DType::FLOAT32) {
            ensure_resident(DeviceType::CPU);
        }

        return std::visit(
            [this](const auto& vec) -> const void* {
                return static_cast<const void*>(vec.data() + offset_);
            },
            storage_);
    }

    void* TensorImpl::raw_data_ptr() {
        if (data_ptr_) {
            if (dtype_ != DType::FLOAT32) {
                throw std::runtime_error("TensorImpl: pointer-backed views currently require float32 dtype");
            }
            return static_cast<void*>(data_ptr_ + offset_);
        }

        if (base_impl_) {
            if (dtype_ == DType::FLOAT32) {
                return static_cast<void*>(base_impl_->data_ptr() + offset_);
            }
            auto* base = static_cast<std::uint8_t*>(base_impl_->raw_data_ptr());
            return static_cast<void*>(base + offset_ * element_size_bytes());
        }

        if (dtype_ == DType::FLOAT32) {
            ensure_resident(DeviceType::CPU);
            host_data_valid_ = true;
            cuda_data_valid_ = false;
        }

        return std::visit(
            [this](auto& vec) -> void* {
                return static_cast<void*>(vec.data() + offset_);
            },
            storage_);
    }

    const float* TensorImpl::data_ptr() const {
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::data_ptr(): float pointer access requires float32 tensor dtype.");
        }
        if (data_ptr_) {
            return data_ptr_ + offset_;
        }
        if (base_impl_) {
            return base_impl_->data_ptr() + offset_;
        }
        ensure_resident(DeviceType::CPU);
        return std::get<std::vector<float>>(storage_).data() + offset_;
    }

    float* TensorImpl::data_ptr() {
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::data_ptr(): float pointer access requires float32 tensor dtype.");
        }
        if (data_ptr_) {
            return data_ptr_ + offset_;
        }
        if (base_impl_) {
            return base_impl_->data_ptr() + offset_;
        }
        ensure_resident(DeviceType::CPU);
        host_data_valid_ = true;
        cuda_data_valid_ = false;
        return std::get<std::vector<float>>(storage_).data() + offset_;
    }

    const float* TensorImpl::backend_data_ptr(DeviceType dev) const {
        if (dev == DeviceType::CPU) {
            return data_ptr();
        }
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "TensorImpl::backend_data_ptr(CUDA): CUDA backend is currently float32-only");
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
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "TensorImpl::backend_data_ptr(CUDA): CUDA backend is currently float32-only");
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
        if (dev == DeviceType::CUDA && dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::copy_to(DeviceType::CUDA) currently supports only float32 tensors");
        }

        if (dtype_ == DType::FLOAT32) {
            std::vector<float> logical;
            materialize_logical_data(logical);
            auto copied = std::make_shared<TensorImpl>(shape_, logical, DeviceType::CPU);
            copied->set_device(dev);
            return copied;
        }

        if (dtype_ == DType::FLOAT64) {
            std::vector<std::uint8_t> bytes;
            materialize_logical_data_bytes(bytes);
            std::vector<double> values(numel());
            if (!values.empty()) {
                std::memcpy(values.data(), bytes.data(), bytes.size());
            }
            auto copied = std::make_shared<TensorImpl>(shape_, values, DeviceType::CPU);
            copied->set_device(dev);
            return copied;
        }

        if (dtype_ == DType::INT32) {
            std::vector<std::uint8_t> bytes;
            materialize_logical_data_bytes(bytes);
            std::vector<std::int32_t> values(numel());
            if (!values.empty()) {
                std::memcpy(values.data(), bytes.data(), bytes.size());
            }
            auto copied = std::make_shared<TensorImpl>(shape_, values, DeviceType::CPU);
            copied->set_device(dev);
            return copied;
        }

        std::vector<std::uint8_t> bytes;
        materialize_logical_data_bytes(bytes);
        std::vector<bool> values(numel(), false);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = bytes[i] != 0;
        }
        auto copied = std::make_shared<TensorImpl>(shape_, values, DeviceType::CPU);
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
        return compute_numel(shape_);
    }

    DeviceType TensorImpl::device() const { return device_; }
    DType TensorImpl::dtype() const { return dtype_; }
    void TensorImpl::set_device(DeviceType dev) {
        if (device_ == dev) {
            return;
        }
        if (dev == DeviceType::CUDA && dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::set_device(DeviceType::CUDA) currently supports only float32 tensors");
        }
        ensure_resident(dev);
        device_ = dev;
    }

    bool TensorImpl::has_called_backward() const {
        return has_called_backward_;
    }

    void TensorImpl::set_has_called_backward(bool val) {
        has_called_backward_ = val;
    }

    bool TensorImpl::requires_grad() const {
        return requires_grad_;
    }

    void TensorImpl::set_requires_grad(bool val) {
        requires_grad_ = val;
    }

    void TensorImpl::zero_grad() {
        grad_data_.assign(numel(), 0.0f);
    }

    const std::vector<float>& TensorImpl::grad_data() const {
        return grad_data_;
    }

    void TensorImpl::accumulate_grad(const std::vector<float>& grad) {
        if (grad.size() != numel()) {
            throw std::runtime_error("autograd: gradient shape mismatch while accumulating gradient");
        }

        if (grad_data_.empty()) {
            grad_data_ = grad;
            return;
        }

        if (grad_data_.size() != grad.size()) {
            throw std::runtime_error("autograd: internal gradient buffer size mismatch");
        }

        for (size_t i = 0; i < grad.size(); ++i) {
            grad_data_[i] += grad[i];
        }
    }

    void TensorImpl::backward(const std::vector<float>& grad) {
        if (!requires_grad_) {
            return;
        }

        has_called_backward_ = true;
        accumulate_grad(grad);

        if (grad_fn_) {
            grad_fn_(grad);
        }
    }

    void TensorImpl::set_grad_fn(std::function<void(const std::vector<float>&)> fn) {
        grad_fn_ = std::move(fn);
    }

    void TensorImpl::clear_grad_fn() {
        grad_fn_ = {};
    }

    bool TensorImpl::has_grad_fn() const {
        return static_cast<bool>(grad_fn_);
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
