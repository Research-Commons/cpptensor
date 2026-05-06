#include "cpptensor/tensor/tensorimpl.hpp"

#include <cstring>
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
            return std::get<std::vector<float>>(storage_);
        }

        throw std::runtime_error(
            "Tensor::data(): mutable access is unavailable for sliced, permuted, "
            "transposed, or pointer-backed views. Call contiguous() or clone() first.");
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
            auto* base = static_cast<std::uint8_t*>(base_impl_->raw_data_ptr());
            return static_cast<void*>(base + offset_ * element_size_bytes());
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
        return static_cast<const float*>(raw_data_ptr());
    }

    float* TensorImpl::data_ptr() {
        if (dtype_ != DType::FLOAT32) {
            throw std::runtime_error(
                "Tensor::data_ptr(): float pointer access requires float32 tensor dtype.");
        }
        return static_cast<float*>(raw_data_ptr());
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
    void TensorImpl::set_device(DeviceType dev) { device_ = dev; }

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
