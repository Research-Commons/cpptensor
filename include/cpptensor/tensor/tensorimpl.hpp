#pragma once
#include <vector>
#include <memory>
#include <stdexcept>

#include "cpptensor/enums/dispatcherEnum.h"

namespace cpptensor {

    class Function; // forward declaration for autograd support
    class Tensor;

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
        friend class Tensor;

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
         * @param offset Offset from base data start (for slicing)
         */
        TensorImpl(std::shared_ptr<TensorImpl> base,
                   const std::vector<size_t>& new_shape,
                   const std::vector<size_t>& new_stride = {},
                   size_t offset = 0);

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

        ~TensorImpl();

        // =============== Data Accessors ===============

        /**
         * @brief Get const reference to flattened logical tensor contents
         *
         * Returns the tensor values in logical row-major order. Owning tensors
         * and full contiguous views expose their backing storage directly.
         * Non-trivial views (for example slices, permutations, transposes, or
         * pointer-backed views) materialize a compact row-major snapshot.
         *
         * @return Const reference to flattened logical contents
         */
        const std::vector<float>& data() const;

        /**
         * @brief Get mutable reference to direct backing storage
         *
         * Allows direct modification only when the tensor is backed by a
         * single contiguous storage vector whose logical contents exactly
         * match the exposed buffer.
         *
         * @return Mutable reference to backing data vector
         *
         * @warning Throws for sliced, permuted, transposed, or pointer-backed
         *          views. Call contiguous() or clone() first if you need a
         *          mutable compact buffer.
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
         * @brief Get pointer for a specific backend and ensure residency
         */
        const float* backend_data_ptr(DeviceType dev) const;

        /**
         * @brief Get mutable pointer for a specific backend and mark it dirty
         */
        float* backend_data_ptr(DeviceType dev);

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
         * @brief Get offset from base tensor data start
         *
         * For view tensors (created via slicing), returns the element offset
         * from the base tensor's data pointer. For non-view tensors, returns 0.
         *
         * @return Offset in number of elements
         */
        size_t offset() const;

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
         * Changes the preferred device and performs the required data transfer
         * for owning tensors.
         *
         * @param dev New device type
         */
        void set_device(DeviceType dev);

        /**
         * @brief Copy tensor storage to another device and return a new impl
         */
        std::shared_ptr<TensorImpl> copy_to(DeviceType dev) const;

        /**
         * @brief Ensure the current storage is resident on the target device
         */
        void ensure_resident(DeviceType dev) const;

    private:
        /**
         * @brief Check whether data() can expose direct mutable storage
         */
        bool can_expose_direct_data_buffer() const;

        /**
         * @brief Materialize logical row-major contents into a compact buffer
         */
        void materialize_logical_data(std::vector<float>& out) const;

        /**
         * @brief Raw data buffer in row-major order
         */
        mutable std::vector<float> data_;

        /**
         * @brief Cached compact logical contents for const data() on views
         */
        mutable std::vector<float> logical_data_cache_;

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
         * @brief Offset from base tensor data (for sliced views)
         *
         * When this TensorImpl is a sliced view, offset_ indicates how many
         * elements to skip from the base tensor's data pointer. For non-view
         * tensors or non-sliced views, this is 0.
         */
        size_t offset_ = 0;

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
         * @brief CUDA device storage for owning contiguous tensors
         */
        mutable float* cuda_data_ = nullptr;

        /**
         * @brief Whether host storage contains up-to-date logical contents
         */
        mutable bool host_data_valid_ = true;

        /**
         * @brief Whether CUDA storage contains up-to-date logical contents
         */
        mutable bool cuda_data_valid_ = false;

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

        /**
         * @brief Free CUDA storage owned by this impl
         */
        void release_cuda_storage() const;
    };

} // namespace cpptensor
