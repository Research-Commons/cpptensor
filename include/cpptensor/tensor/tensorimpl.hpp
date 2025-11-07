#pragma once
#include <vector>
#include <memory>
#include <stdexcept>

#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor {

    class Function; // forward declaration for autograd support

    /**
     * @class TensorImpl
     * @brief Internal implementation class for Tensor data storage
     *
     * TensorImpl is the actual storage backend for Tensor objects, implementing
     * the PIMPL (Pointer to Implementation) pattern. It manages:
     * - Raw data buffer (std::vector<float> in row-major order)
     * - Shape and stride information for n-dimensional indexing
     * - Device placement (CPU, CUDA, etc.)
     * - Autograd metadata (gradient function, backward propagation state)
     *
     * Memory Layout:
     * - Data stored in contiguous row-major (C-style) order
     * - For shape [D₀, D₁, ..., Dₙ], element at index [i₀, i₁, ..., iₙ] is at:
     *   offset = i₀*stride[0] + i₁*stride[1] + ... + iₙ*stride[n]
     * - Strides computed as: stride[i] = ∏(j=i+1 to n) shape[j]
     *
     * Example Memory Layout:
     * ```
     * Shape: [2, 3]
     * Data:  [a, b, c, d, e, f]
     *
     * Logical view:      Physical memory:
     * [[a, b, c],   →    [a, b, c, d, e, f]
     *  [d, e, f]]
     *
     * Strides: [3, 1]
     * Element [i,j] at offset: i*3 + j*1
     * ```
     *
     * Design Rationale:
     * - Separation from Tensor enables lightweight handle semantics
     * - Shared ownership (via shared_ptr) allows efficient copying
     * - Enables future optimizations (COW, view semantics, lazy evaluation)
     *
     * @note This class is an internal implementation detail. Users should
     *       interact with Tensor class instead.
     */
    class TensorImpl {
    public:
        // =============== Constructors ===============

        /**
         * @brief Construct TensorImpl from shape and data vector
         *
         * Creates a tensor implementation with the specified shape and initializes
         * it with the provided data. Data must be in row-major order and its size
         * must exactly match the product of shape dimensions.
         *
         * @param shape Dimensions of the tensor (e.g., {2, 3, 4})
         * @param data Initial data in row-major order
         * @param device Target device for tensor storage (CPU, CUDA, etc.)
         * @throws std::runtime_error if data.size() != product(shape)
         *
         * @example
         * ```cpp
         * // Create 2×3 tensor
         * TensorImpl impl({2, 3}, {1, 2, 3, 4, 5, 6}, DeviceType::CPU);
         * // Represents: [[1, 2, 3],
         * //              [4, 5, 6]]
         * ```
         */
        TensorImpl(const std::vector<size_t>& shape,
                   const std::vector<float>& data,
                   DeviceType device = DeviceType::CPU);

        /**
         * @brief Construct TensorImpl filled with a constant value
         *
         * Creates a tensor implementation where all elements are initialized
         * to the same fill value. More memory-efficient than constructing
         * a full data vector for uniform initialization.
         *
         * @param shape Dimensions of the tensor
         * @param fill_value Value to initialize all elements
         * @param device Target device for tensor storage
         *
         * @example
         * ```cpp
         * // Create 100×100 matrix filled with zeros
         * TensorImpl zeros({100, 100}, 0.0f, DeviceType::CPU);
         * ```
         */
        TensorImpl(const std::vector<size_t>& shape,
                   float fill_value,
                   DeviceType device = DeviceType::CPU);

        /**
         * @brief Construct view TensorImpl that shares data with base tensor
         *
         * Creates a view tensor that references the same underlying data as
         * the base tensor but with different shape/stride. Used for zero-copy
         * operations like reshape, view, permute, etc.
         *
         * @param base Base TensorImpl to share data with
         * @param new_shape Shape for the view
         * @param new_stride Stride for the view (optional, computed if empty)
         */
        TensorImpl(std::shared_ptr<TensorImpl> base,
                   const std::vector<size_t>& new_shape,
                   const std::vector<size_t>& new_stride = {});

        /**
         * @brief Construct view TensorImpl that wraps raw pointer (zero-copy)
         *
         * Creates a view that wraps an existing raw pointer without copying data.
         * Used for efficient batch slicing and sub-tensor views. The caller
         * must ensure data validity through the owner parameter.
         *
         * @param shape Shape of the view
         * @param data_ptr Raw pointer to existing data
         * @param owner Base TensorImpl that owns the data (keeps it alive)
         * @param device Device type of the data
         */
        TensorImpl(const std::vector<size_t>& shape,
                   float* data_ptr,
                   std::shared_ptr<TensorImpl> owner,
                   DeviceType device = DeviceType::CPU);

        // =============== Data Accessors ===============

        /**
         * @brief Get const reference to raw data buffer
         *
         * Returns the underlying flattened data vector in row-major order.
         *
         * @return Const reference to data vector
         *
         * @note Direct access to raw data. Useful for interfacing with
         *       external libraries (BLAS, LAPACK, etc.)
         */
        const std::vector<float>& data() const;

        /**
         * @brief Get mutable reference to raw data buffer
         *
         * Allows direct modification of underlying data. Use with caution
         * as it bypasses tensor abstraction and autograd tracking.
         *
         * @return Mutable reference to data vector
         *
         * @warning Modifying data directly breaks autograd computation graph
         */
        std::vector<float>& data();

        /**
         * @brief Get raw pointer to data (for pointer-based views and BLAS)
         *
         * Returns a pointer to the actual data, whether it's stored in data_,
         * accessed via base_impl_, or wrapped via data_ptr_. This is the
         * preferred method for interfacing with external libraries like BLAS.
         *
         * @return Const pointer to data
         */
        const float* data_ptr() const;

        /**
         * @brief Get mutable raw pointer to data
         *
         * @return Mutable pointer to data
         */
        float* data_ptr();

        /**
         * @brief Get const reference to stride information
         *
         * Strides define memory layout for multi-dimensional indexing.
         * stride[i] indicates how many elements to skip to move one position
         * along dimension i.
         *
         * @return Const reference to stride vector
         *
         * @note For row-major layout: stride[i] = ∏(j=i+1 to n) shape[j]
         *
         * @example
         * ```cpp
         * // Shape [2, 3, 4], strides = [12, 4, 1]
         * // To access element [i, j, k]: data[i*12 + j*4 + k]
         * ```
         */
        const std::vector<size_t>& stride() const;

        /**
         * @brief Get mutable reference to stride information
         *
         * @return Mutable reference to stride vector
         *
         * @warning Manual stride modification can corrupt tensor indexing.
         *          Advanced use only (e.g., implementing views/slicing).
         */
        std::vector<size_t>& stride();

        /**
         * @brief Check if backward pass has been executed
         *
         * Used by autograd system to track whether gradients have been
         * computed for this tensor during backpropagation.
         *
         * @return true if backward() has been called, false otherwise
         *
         * @note Currently declared but not fully implemented. Part of
         *       future autograd infrastructure.
         */
        bool has_called_backward() const;

        /**
         * @brief Set backward execution flag
         *
         * Marks whether backward pass has been executed for this tensor.
         *
         * @param val true to mark as executed, false otherwise
         *
         * @note Part of autograd bookkeeping for preventing duplicate gradients
         */
        void set_has_called_backward(bool val);

        // =============== Shape and Metadata ===============

        /**
         * @brief Get tensor dimensions
         *
         * @return Const reference to shape vector
         *
         * @example
         * ```cpp
         * auto& shape = impl.shape();  // {batch, channels, height, width}
         * size_t batch_size = shape[0];
         * ```
         */
        const std::vector<size_t>& shape() const;

        /**
         * @brief Get total number of elements
         *
         * Computes the product of all dimensions.
         *
         * @return Total element count
         *
         * @example
         * ```cpp
         * TensorImpl impl({2, 3, 4}, 0.0f);
         * size_t count = impl.numel();  // 24
         * ```
         */
        size_t numel() const;

        /**
         * @brief Get device where tensor data is stored
         *
         * @return DeviceType enum value (CPU, CUDA, etc.)
         */
        DeviceType device() const;

        /**
         * @brief Set device for tensor storage
         *
         * Changes the device type. Note: This does NOT actually move data.
         * It only updates the metadata flag. Actual data transfer must be
         * handled separately.
         *
         * @param dev New device type
         *
         * @warning Does not perform actual data migration. Use with caution.
         */
        void set_device(DeviceType dev);

    private:
        /**
         * @brief Raw data buffer in row-major order
         */
        std::vector<float> data_;

        /**
         * @brief Base tensor for views (keeps base alive)
         *
         * When this TensorImpl is a view, base_impl_ points to the original
         * tensor that owns the data. This keeps the data alive as long as
         * any view exists.
         */
        std::shared_ptr<TensorImpl> base_impl_;

        /**
         * @brief Raw pointer for pointer-based views (zero-copy)
         *
         * When this TensorImpl wraps a raw pointer (created via from_ptr),
         * this stores the pointer. The base_impl_ keeps the owner alive.
         * If null, uses data_ vector instead.
         */
        float* data_ptr_;

        /**
         * @brief Stride information for each dimension
         *
         * stride_[i] = number of elements to skip to move one step along dimension i
         */
        std::vector<size_t> stride_;

        /**
         * @brief Gradient function for autograd
         *
         * Pointer to the operation that created this tensor, used for
         * automatic differentiation during backward pass.
         */
        std::shared_ptr<Function> grad_fn_;

        /**
         * @brief Tensor dimensions
         */
        std::vector<size_t> shape_;

        /**
         * @brief Device where tensor is stored (CPU, CUDA, etc.)
         */
        DeviceType device_ = DeviceType::CPU;

        /**
         * @brief Compute strides from shape (row-major layout)
         *
         * Calculates stride values for efficient multi-dimensional indexing.
         * For row-major layout: stride[i] = product of shape[i+1:end]
         *
         * @param shape Tensor dimensions
         * @return Vector of stride values
         *
         * @example
         * ```cpp
         * compute_strides({2, 3, 4}) → {12, 4, 1}
         * compute_strides({5, 10})   → {10, 1}
         * ```
         */
        std::vector<size_t> compute_strides(const std::vector<size_t>& shape);
    };

} // namespace cpptensor