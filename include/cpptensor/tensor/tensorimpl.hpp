#pragma once
#include <vector>
#include <memory>
#include <stdexcept>

#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor {

    class Function; // forward

    /**
     * TensorImpl
     *
     * Internal storage for Tensor. Replaces the old ArrayFire af::array storage
     * with std::vector<float>. Holds optional autograd metadata (grad vector,
     * grad_fn pointer, bookkeeping).
     */
    class TensorImpl {
    public:
        // Construct from shape and initial data (row-major)
        // device defaults to DeviceType::CPU for backward compatibility
        TensorImpl(const std::vector<size_t>& shape,
                   const std::vector<float>& data,
                   DeviceType device = DeviceType::CPU);

        // Construct from shape and a single fill value
        TensorImpl(const std::vector<size_t>& shape,
                   float fill_value,
                   DeviceType device = DeviceType::CPU);

        // Accessors
        const std::vector<float>& data() const;
        std::vector<float>& data();

        const std::vector<size_t>& stride() const;
        std::vector<size_t>& stride();

        bool has_called_backward() const;
        void set_has_called_backward(bool val);

        const std::vector<size_t>& shape() const;
        size_t numel() const;

        // Device accessor / mutator
        DeviceType device() const;
        void set_device(DeviceType dev);

    private:
        std::vector<float> data_;
        std::vector<size_t> stride_;
        std::shared_ptr<Function> grad_fn_;
        std::vector<size_t> shape_;
        DeviceType device_ = DeviceType::CPU;

        std::vector<size_t> compute_strides(const std::vector<size_t>& shape);
    };

} // namespace cppgrad