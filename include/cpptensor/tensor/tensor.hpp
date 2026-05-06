#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <optional>
#include <cstdint>

#include "cpptensor/tensor/tensorimpl.hpp"
#include "cpptensor/enums/dispatcherEnum.h"   // for DeviceType

namespace cpptensor {

/**
 * @class Tensor
 * @brief Multi-dimensional array container for numerical computations
 *
 * The Tensor class provides a high-level interface for n-dimensional arrays
 * with support for various operations, device placement (CPU/CUDA), and
 * automatic differentiation (via grad_fn when autograd is enabled).
 *
 * Memory Layout:
 * - Data is stored in row-major (C-style) order
 * - Strides are computed automatically based on shape
 * - Example: [2×3] tensor stores as [row0_col0, row0_col1, row0_col2, row1_col0, ...]
 *
 * Design:
 * - Uses copy-on-write semantics via shared_ptr to TensorImpl
 * - Lightweight wrapper enabling efficient copying and passing
 * - Actual data resides in TensorImpl for memory efficiency
 *
 * @example
 * ```cpp
 * // Create from data
 * Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
 *
 * // Factory methods
 * Tensor B = Tensor::zeros({3, 3});
 * Tensor C = Tensor::randn({100, 100});
 *
 * // Operations
 * Tensor D = A + B * 2.0f;
 * D.print();
 * ```
 */
class Tensor {
    public:
        // =============== Constructors ===============

        /**
         * @brief Construct tensor from shape and data vector
         *
         * Creates a tensor with the specified shape and initializes it with
         * the provided data in row-major order. Data size must match the
         * product of shape dimensions.
         *
         * @param shape Dimensions of the tensor (e.g., {2, 3, 4})
         * @param values Initial data in row-major order (must have shape.prod() elements)
         * @param device Target device (CPU, CUDA, etc.)
         * @throws std::runtime_error if values.size() != product(shape)
         *
         * @example
         * ```cpp
         * Tensor A({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});  // [[1, 2], [3, 4]]
         * ```
         */
        Tensor(const std::vector<size_t>& shape,
               const std::vector<float>& values,
               bool requires_grad,
               DeviceType device = DeviceType::CPU);

        Tensor(const std::vector<size_t>& shape,
               const std::vector<float>& values,
               DeviceType device = DeviceType::CPU);

        /**
         * @brief Default constructor - creates an uninitialized tensor handle
         *
         * Creates a tensor with a null implementation pointer. Public API calls
         * on a default-constructed tensor throw a std::runtime_error until the
         * tensor is assigned a valid implementation.
         */
        Tensor() = default;

        /**
         * @brief Copy constructor - shallow copy with shared ownership
         *
         * Creates a new Tensor that shares the underlying TensorImpl with the
         * source. Both tensors reference the same data (copy-on-write semantics).
         */
        Tensor(const Tensor&) = default;

        /**
         * @brief Move constructor - transfers ownership
         *
         * Efficiently transfers ownership of TensorImpl without copying.
         */
        Tensor(Tensor&&) noexcept = default;

        /**
         * @brief Copy assignment - shallow copy with shared ownership
         */
        Tensor& operator=(const Tensor&) = default;

        /**
         * @brief Move assignment - transfers ownership
         */
        Tensor& operator=(Tensor&&) noexcept = default;

        // =============== Factory Methods ===============

        /**
         * @brief Create tensor filled with zeros
         *
         * @param shape Dimensions of the tensor
         * @param device Target device for tensor allocation
         * @return Tensor filled with 0.0f
         *
         * @example
         * ```cpp
         * Tensor A = Tensor::zeros({3, 3});  // 3×3 matrix of zeros
         * ```
         */
        static Tensor zeros(const std::vector<size_t>& shape,
                            bool requires_grad,
                            DeviceType device = DeviceType::CPU);

        static Tensor zeros(const std::vector<size_t>& shape,
                            DeviceType device = DeviceType::CPU);

        /**
         * @brief Create tensor filled with ones
         *
         * @param shape Dimensions of the tensor
         * @param device Target device for tensor allocation
         * @return Tensor filled with 1.0f
         *
         * @example
         * ```cpp
         * Tensor identity_diag = Tensor::ones({100});  // 100-element vector of ones
         * ```
         */
        static Tensor ones(const std::vector<size_t>& shape,
                           bool requires_grad,
                           DeviceType device = DeviceType::CPU);

        static Tensor ones(const std::vector<size_t>& shape,
                           DeviceType device = DeviceType::CPU);

        /**
         * @brief Create tensor with random normal distribution (mean=0, std=1)
         *
         * Generates random values from standard normal distribution N(0, 1)
         * using C++ standard library random number generator.
         *
         * @param shape Dimensions of the tensor
         * @param device Target device for tensor allocation
         * @return Tensor with random values from N(0, 1)
         *
         * @example
         * ```cpp
         * Tensor weights = Tensor::randn({784, 128});  // Random initialization
         * ```
         */
        static Tensor randn(const std::vector<size_t>& shape,
                            bool requires_grad,
                            DeviceType device = DeviceType::CPU);

        static Tensor randn(const std::vector<size_t>& shape,
                            DeviceType device = DeviceType::CPU);

        /**
         * @brief Create tensor filled with a constant value
         *
         * @param shape Dimensions of the tensor
         * @param value Constant value to fill all elements
         * @param device Target device for tensor allocation
         * @return Tensor filled with the specified value
         *
         * @example
         * ```cpp
         * Tensor mask = Tensor::full({10, 10}, -1.0f);  // 10×10 matrix of -1s
         * ```
         */
        static Tensor full(const std::vector<size_t>& shape,
                           float value,
                           bool requires_grad,
                           DeviceType device = DeviceType::CPU);

        static Tensor full(const std::vector<size_t>& shape,
                           float value,
                           DeviceType device = DeviceType::CPU);

        /**
         * @brief Create zero-copy view from raw pointer (advanced use)
         *
         * Creates a tensor that wraps existing data without copying. The caller
         * must ensure the data remains valid for the lifetime of this tensor
         * and any views derived from it. This is achieved by passing the owner's
         * shared_ptr to keep it alive.
         *
         * **Use Case:** Efficient batch slicing in matmul, creating sub-tensor
         * views without memory allocation.
         *
         * @param shape Dimensions of the tensor view
         * @param data_ptr Pointer to existing data (must be valid for tensor lifetime)
         * @param owner Shared pointer to owner tensor that keeps data alive
         * @param device Device type of the data
         * @return Tensor view wrapping the raw pointer (zero-copy)
         *
         * @warning Advanced feature. Incorrect use can lead to use-after-free.
         *          Always pass valid owner to ensure data lifetime.
         *
         * @example
         * ```cpp
         * // Create view of batch slice without copying
         * Tensor parent({10, 64, 64}, ...);
         * float* slice_ptr = parent.data().data() + (5 * 64 * 64);  // Batch 5
         * Tensor slice = Tensor::from_ptr({64, 64}, slice_ptr,
         *                                  parent.impl(), parent.device_type());
         * // slice aliases the parent storage for zero-copy reads/internal kernels
         * // Mutable data() access is intentionally rejected for pointer-backed views
         * ```
         */
        static Tensor from_ptr(const std::vector<size_t>& shape,
                              float* data_ptr,
                              std::shared_ptr<TensorImpl> owner,
                              DeviceType device = DeviceType::CPU);

        // =============== Shape and Metadata ===============

        /**
         * @brief Get tensor dimensions
         *
         * @return Vector of dimension sizes (e.g., {2, 3, 4} for 2×3×4 tensor)
         *
         * @example
         * ```cpp
         * auto sh = tensor.shape();  // {batch, height, width}
         * size_t height = sh[1];
         * ```
         */
        std::vector<size_t> shape() const;

        /**
         * @brief Get total number of elements
         *
         * @return Product of all dimensions (total element count)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4});
         * size_t count = A.numel();  // 24
         * ```
         */
        size_t numel() const;

        /**
         * @brief Get number of dimensions (rank)
         *
         * @return Number of axes in the tensor
         *
         * @example
         * ```cpp
         * Tensor scalar({});       // ndim() = 0
         * Tensor vector({10});     // ndim() = 1
         * Tensor matrix({3, 4});   // ndim() = 2
         * ```
         */
        size_t ndim() const;

        /**
         * @brief Get device type where tensor is stored
         *
         * @return DeviceType enum (CPU, CUDA, etc.)
         */
        DeviceType device_type() const;

        /**
         * @brief Print tensor data in compact format
         *
         * Prints tensor shape and flattened data values.
         * Format: "Tensor(shape=[...], values=[...])"
         *
         * @note For large tensors, only first elements are shown
         */
        void print() const;

        /**
         * @brief Print tensor with formatted layout
         *
         * Displays tensor with proper matrix/array formatting for better readability.
         * Multi-dimensional tensors show nested structure with indentation.
         */
        void print_pretty() const;

        // =============== Data Access ===============

        /**
         * @brief Get const reference to flattened logical tensor contents
         *
         * For owning tensors and full contiguous views, this exposes the
         * underlying storage directly. For non-trivial views, it returns a
         * compact row-major snapshot of the view contents.
         *
         * @return Const reference to flattened logical values
         *
         * @warning Direct data access bypasses abstraction. Use carefully.
         *
         * @example
         * ```cpp
         * const auto& data = tensor.data();
         * float first_element = data[0];
         * ```
         */
        const std::vector<float>& data() const;

        /**
         * @brief Get mutable reference to direct backing storage
         *
         * Only tensors backed by a single compact storage vector expose mutable
         * data directly. Non-trivial views throw instead of silently exposing
         * unrelated base storage.
         *
         * @return Mutable reference to flattened backing storage
         *
         * @warning Modifying data directly may break invariants. Prefer
         *          operations. Non-trivial views must be materialized with
         *          contiguous() or clone() first.
         *
         * @example
         * ```cpp
         * auto& data = tensor.data();
         * data[0] = 1.0f;  // Direct modification
         * ```
         */
        std::vector<float>& data();

        /**
         * @brief Get const reference to stride information
         *
         * Strides define memory layout - how many elements to skip
         * to move one position in each dimension.
         *
         * @return Vector of strides for each dimension (row-major)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4});
         * auto strides = A.stride();  // {12, 4, 1}
         * // To access A[i][j][k]: offset = i*12 + j*4 + k*1
         * ```
         */
        const std::vector<size_t>& stride() const;

        /**
         * @brief Get mutable reference to stride information
         *
         * @return Mutable reference to stride vector
         *
         * @warning Modifying strides manually can corrupt tensor. Advanced use only.
         */
        std::vector<size_t>& stride();

        /**
         * @brief Get shared pointer to underlying TensorImpl
         *
         * Provides access to the implementation for advanced operations
         * or when interfacing with internal APIs.
         *
         * @return Shared pointer to TensorImpl
         */
        std::shared_ptr<TensorImpl> impl() const;

        // =============== Tensor Manipulation Operations ===============

        /**
         * @brief Create zero-copy view with new shape (requires contiguous tensor)
         *
         * Creates a new tensor that shares the underlying data but with a different
         * shape. The total number of elements must remain the same. This is a
         * zero-copy operation - no data is copied.
         *
         * @param new_shape New dimensions for the view
         * @return Tensor view with new shape sharing the same data
         * @throws std::runtime_error if total elements don't match or tensor not contiguous
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, ...);  // 24 elements
         * Tensor B = A.view({4, 6}); // Same 24 elements, different shape
         * B.data()[0] = 1.0f;        // Also modifies A!
         * ```
         */
        Tensor view(const std::vector<size_t>& new_shape) const;

        /**
         * @brief Reshape tensor (zero-copy if contiguous, otherwise copies)
         *
         * Returns a tensor with the specified shape. If the original tensor is
         * contiguous, this is zero-copy (same as view). Otherwise, data is copied
         * to create a contiguous layout first.
         *
         * @param new_shape New dimensions
         * @return Reshaped tensor
         * @throws std::runtime_error if total elements don't match
         *
         * @example
         * ```cpp
         * Tensor A({2, 6}, ...);
         * Tensor B = A.reshape({3, 4});  // Zero-copy if A is contiguous
         * ```
         */
        Tensor reshape(const std::vector<size_t>& new_shape) const;

        /**
         * @brief Flatten tensor into 1D vector
         *
         * @param start_dim First dimension to flatten (default: 0)
         * @param end_dim Last dimension to flatten (default: -1, meaning last dim)
         * @return Flattened tensor view or copy
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, ...);
         * Tensor B = A.flatten();           // Shape: {24}
         * Tensor C = A.flatten(1, 2);       // Shape: {2, 12}
         * ```
         */
        Tensor flatten(int start_dim = 0, int end_dim = -1) const;

        /**
         * @brief Create zero-copy slice along a single dimension
         *
         * Produces a view into the tensor without copying data. The slice is
         * defined by [start, end) with optional step controlling stride between
         * selected elements. Negative indices are interpreted relative to the
         * dimension size (e.g., -1 = last element). Only positive step values
         * are supported.
         *
         * @param dim Dimension to slice (0-indexed, supports negative indexing)
         * @param start Starting index (inclusive). Defaults to 0 if not set.
         * @param end Ending index (exclusive). Defaults to dim size if not set.
         * @param step Step between indices. Defaults to 1. Must be positive.
         * @return Tensor view representing the requested slice (zero-copy)
         * @throws std::runtime_error if dim out of range or step <= 0
         *
         * @example
         * ```cpp
         * Tensor A({10, 20, 30}, ...);
         * Tensor B = A.slice(0, 2, 5);           // rows [2:5) → shape {3, 20, 30}
         * Tensor C = A.slice(1, -10, -1);        // last 9 cols → shape {10, 9, 30}
         * Tensor D = A.slice(2, 0, std::nullopt, 2);  // every 2nd → shape {10, 20, 15}
         *
         * // Non-trivial views materialize const data() and reject mutable data()
         * const auto& logical = B.data();
         * Tensor B_copy = B.contiguous();  // Compact mutable copy
         * ```
         */
        Tensor slice(int dim,
                     std::optional<int64_t> start = std::nullopt,
                     std::optional<int64_t> end = std::nullopt,
                     std::optional<int64_t> step = std::nullopt) const;

        /**
         * @brief Remove dimensions of size 1
         *
         * @param dim Dimension to squeeze (-1 means squeeze all size-1 dims)
         * @return Squeezed tensor (zero-copy view)
         *
         * @example
         * ```cpp
         * Tensor A({2, 1, 3, 1}, ...);
         * Tensor B = A.squeeze();      // Shape: {2, 3}
         * Tensor C = A.squeeze(1);     // Shape: {2, 3, 1}
         * Tensor D({1}, ...);
         * Tensor E = D.squeeze();      // Shape: {}
         * ```
         */
        Tensor squeeze(int dim = -1) const;

        /**
         * @brief Add dimension of size 1
         *
         * @param dim Position to insert new dimension
         * @return Unsqueezed tensor (zero-copy view)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3}, ...);
         * Tensor B = A.unsqueeze(0);   // Shape: {1, 2, 3}
         * Tensor C = A.unsqueeze(1);   // Shape: {2, 1, 3}
         * Tensor D = A.unsqueeze(-1);  // Shape: {2, 3, 1}
         * ```
         */
        Tensor unsqueeze(int dim) const;

        /**
         * @brief Permute tensor dimensions (generalized transpose)
         *
         * Reorders dimensions according to the given permutation. This creates
         * a view with modified strides - the resulting tensor is NOT contiguous.
         * Call contiguous() if you need contiguous memory layout.
         *
         * @param dims Permutation of dimensions (must be a permutation of 0..ndim-1)
         * @return Permuted tensor (zero-copy view with non-contiguous strides)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, ...);
         * Tensor B = A.permute({2, 0, 1});  // Shape: {4, 2, 3}
         * Tensor C = A.permute({1, 0, 2});  // Swap first two dims
         * ```
         */
        Tensor permute(const std::vector<int>& dims) const;

        /**
         * @brief Transpose two dimensions (2D transpose if no args given)
         *
         * @param dim0 First dimension to swap (default: 0 for 2D case)
         * @param dim1 Second dimension to swap (default: 1 for 2D case)
         * @return Transposed tensor (zero-copy view)
         *
         * @example
         * ```cpp
         * Tensor A({3, 4}, ...);
         * Tensor B = A.transpose();        // Shape: {4, 3} (swap dims 0,1)
         *
         * Tensor C({2, 3, 4}, ...);
         * Tensor D = C.transpose(0, 2);    // Shape: {4, 3, 2}
         * ```
         */
        Tensor transpose(int dim0 = 0, int dim1 = 1) const;

        /**
         * @brief Check if tensor has contiguous memory layout
         *
         * A tensor is contiguous if its strides match row-major ordering.
         * Non-contiguous tensors may result from permute/transpose operations.
         *
         * @return true if tensor is contiguous (row-major), false otherwise
         */
        bool is_contiguous() const;

        /**
         * @brief Ensure tensor has contiguous memory layout
         *
         * Returns the tensor itself if already contiguous, otherwise returns
         * a copy with contiguous (row-major) layout.
         *
         * @return Contiguous tensor (may be a copy)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3}, ...);
         * Tensor B = A.transpose();     // Not contiguous
         * Tensor C = B.contiguous();    // Contiguous copy
         * ```
         */
        Tensor contiguous() const;

        /**
         * @brief Create deep copy of tensor
         *
         * Creates a new tensor with its own data buffer (no sharing),
         * preserving the current tensor's logical contents even for views.
         *
         * @return Independent copy of the tensor
         */
        Tensor clone() const;

        // =============== Autograd API ===============

        /**
         * @brief Whether this tensor should track gradients
         */
        bool requires_grad() const;

        /**
         * @brief Enable or disable gradient tracking on this tensor
         */
        void set_requires_grad(bool requires_grad);

        /**
         * @brief Get accumulated gradient as a tensor with matching shape
         */
        Tensor grad() const;

        /**
         * @brief Reset accumulated gradient buffer to zeros
         */
        void zero_grad();

        /**
         * @brief Backpropagate gradients using ones_like as seed
         */
        void backward() const;

        /**
         * @brief Backpropagate with an explicit output gradient
         */
        void backward(const Tensor& grad_output) const;

        // =============== Reduction Operations ===============

        // =============== Reduction Operations ===============

        /**
         * @brief Sum of all tensor elements (global reduction)
         *
         * Reduces the tensor by summing all elements and returns a scalar.
         *
         * @param keepdim Keep all reduced dimensions as size 1 if true
         * @return Tensor scalar (shape [] or [1,1,...,1] if keepdim=true)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto total = A.sum();         // Global sum, shape []
         * auto total2 = A.sum(true);    // Global sum with keepdim, shape [1, 1, 1]
         * ```
         */
        Tensor sum(bool keepdim = false) const;

        /**
         * @brief Sum of tensor elements along a specific dimension
         *
         * Reduces the tensor by summing along the specified dimension.
         * Supports negative indexing (e.g., -1 for last dimension, -2 for second-to-last).
         *
         * @param dim Dimension to reduce (supports negative indexing)
         * @param keepdim Keep reduced dimension as size 1 if true
         * @return Tensor with one dimension reduced
         *
         * @throws std::runtime_error if dim is out of range
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto B = A.sum(1);            // Sum along dim 1 → shape [2, 4]
         * auto C = A.sum(-1);           // Sum along last dim → shape [2, 3]
         * auto D = A.sum(0, true);      // Sum along dim 0 with keepdim → shape [1, 3, 4]
         * ```
         */
        Tensor sum(int dim, bool keepdim = false) const;

        /**
         * @brief Mean (average) of all tensor elements (global reduction)
         *
         * Reduces the tensor by computing the mean of all elements and returns a scalar.
         *
         * @param keepdim Keep all reduced dimensions as size 1 if true
         * @return Tensor scalar (shape [] or [1,1,...,1] if keepdim=true)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto avg = A.mean();          // Global mean, shape []
         * auto avg2 = A.mean(true);     // Global mean with keepdim, shape [1, 1, 1]
         * ```
         */
        Tensor mean(bool keepdim = false) const;

        /**
         * @brief Mean (average) of tensor elements along a specific dimension
         *
         * Reduces the tensor by computing the mean along the specified dimension.
         * Supports negative indexing (e.g., -1 for last dimension, -2 for second-to-last).
         *
         * @param dim Dimension to reduce (supports negative indexing)
         * @param keepdim Keep reduced dimension as size 1 if true
         * @return Tensor with one dimension reduced
         *
         * @throws std::runtime_error if dim is out of range
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto B = A.mean(1);           // Mean along dim 1 → shape [2, 4]
         * auto C = A.mean(-1);          // Mean along last dim → shape [2, 3]
         * auto D = A.mean(0, true);     // Mean along dim 0 with keepdim → shape [1, 3, 4]
         * ```
         */
        Tensor mean(int dim, bool keepdim = false) const;

        /**
         * @brief Maximum of all tensor elements (global reduction)
         *
         * Reduces the tensor by finding the maximum of all elements and returns a scalar.
         *
         * @param keepdim Keep all reduced dimensions as size 1 if true
         * @return Tensor scalar (shape [] or [1,1,...,1] if keepdim=true)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto max_val = A.max();       // Global max, shape []
         * auto max_val2 = A.max(true);  // Global max with keepdim, shape [1, 1, 1]
         * ```
         */
        Tensor max(bool keepdim = false) const;

        /**
         * @brief Maximum of tensor elements along a specific dimension
         *
         * Reduces the tensor by finding the maximum along the specified dimension.
         * Supports negative indexing (e.g., -1 for last dimension, -2 for second-to-last).
         *
         * @param dim Dimension to reduce (supports negative indexing)
         * @param keepdim Keep reduced dimension as size 1 if true
         * @return Tensor with one dimension reduced
         *
         * @throws std::runtime_error if dim is out of range
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto B = A.max(1);            // Max along dim 1 → shape [2, 4]
         * auto C = A.max(-1);           // Max along last dim → shape [2, 3]
         * auto D = A.max(0, true);      // Max along dim 0 with keepdim → shape [1, 3, 4]
         * ```
         */
        Tensor max(int dim, bool keepdim = false) const;

        /**
         * @brief Minimum of all tensor elements (global reduction)
         *
         * Reduces the tensor by finding the minimum of all elements and returns a scalar.
         *
         * @param keepdim Keep all reduced dimensions as size 1 if true
         * @return Tensor scalar (shape [] or [1,1,...,1] if keepdim=true)
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto min_val = A.min();       // Global min, shape []
         * auto min_val2 = A.min(true);  // Global min with keepdim, shape [1, 1, 1]
         * ```
         */
        Tensor min(bool keepdim = false) const;

        /**
         * @brief Minimum of tensor elements along a specific dimension
         *
         * Reduces the tensor by finding the minimum along the specified dimension.
         * Supports negative indexing (e.g., -1 for last dimension, -2 for second-to-last).
         *
         * @param dim Dimension to reduce (supports negative indexing)
         * @param keepdim Keep reduced dimension as size 1 if true
         * @return Tensor with one dimension reduced
         *
         * @throws std::runtime_error if dim is out of range
         *
         * @example
         * ```cpp
         * Tensor A({2, 3, 4}, values);
         * auto B = A.min(1);            // Min along dim 1 → shape [2, 4]
         * auto C = A.min(-1);           // Min along last dim → shape [2, 3]
         * auto D = A.min(0, true);      // Min along dim 0 with keepdim → shape [1, 3, 4]
         * ```
         */
        Tensor min(int dim, bool keepdim = false) const;

        // =============== Operator Overloads (Element-wise) ===============

        /**
         * @brief Element-wise addition: C = A + B
         *
         * Performs element-wise addition with broadcasting support.
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return New tensor containing element-wise sum
         * @throws std::runtime_error if shapes are incompatible for broadcasting
         *
         * @note Broadcasting rules follow NumPy conventions
         */
        friend Tensor operator+(const Tensor& A, const Tensor& B);

        /**
         * @brief Element-wise subtraction: C = A - B
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return New tensor containing element-wise difference
         */
        friend Tensor operator-(const Tensor& A, const Tensor& B);

        /**
         * @brief Element-wise multiplication: C = A * B (Hadamard product)
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return New tensor containing element-wise product
         *
         * @note This is NOT matrix multiplication. Use matmul() for that.
         */
        friend Tensor operator*(const Tensor& A, const Tensor& B);

        /**
         * @brief Element-wise division: C = A / B
         *
         * @param A Numerator tensor
         * @param B Denominator tensor
         * @return New tensor containing element-wise quotient
         *
         * @warning Division by zero yields inf or nan
         */
        friend Tensor operator/(const Tensor& A, const Tensor& B);

        // =============== Scalar Operations ===============

        /**
         * @brief Add scalar to all elements: C = A + scalar
         */
        friend Tensor operator+(const Tensor& A, float scalar);

        /**
         * @brief Add scalar to all elements (commutative): C = scalar + A
         */
        friend Tensor operator+(float scalar, const Tensor& A);

        /**
         * @brief Subtract scalar from all elements: C = A - scalar
         */
        friend Tensor operator-(const Tensor& A, float scalar);

        /**
         * @brief Subtract tensor from scalar: C = scalar - A
         */
        friend Tensor operator-(float scalar, const Tensor& A);

        /**
         * @brief Multiply all elements by scalar: C = A * scalar
         */
        friend Tensor operator*(const Tensor& A, float scalar);

        /**
         * @brief Multiply all elements by scalar (commutative): C = scalar * A
         */
        friend Tensor operator*(float scalar, const Tensor& A);

        /**
         * @brief Divide all elements by scalar: C = A / scalar
         */
        friend Tensor operator/(const Tensor& A, float scalar);

        /**
         * @brief Divide scalar by all elements: C = scalar / A
         */
        friend Tensor operator/(float scalar, const Tensor& A);

        // =============== Comparison Operations ===============

        /**
         * @brief Element-wise equality comparison: C = (A == B)
         *
         * Performs element-wise equality comparison with broadcasting support.
         * Returns a tensor with 1.0f where elements are equal, 0.0f otherwise.
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return Boolean tensor (1.0f = true, 0.0f = false)
         */
        friend Tensor operator==(const Tensor& A, const Tensor& B);
        friend Tensor operator==(const Tensor& A, float scalar);
        friend Tensor operator==(float scalar, const Tensor& B);

        /**
         * @brief Element-wise inequality comparison: C = (A != B)
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return Boolean tensor (1.0f = true, 0.0f = false)
         */
        friend Tensor operator!=(const Tensor& A, const Tensor& B);
        friend Tensor operator!=(const Tensor& A, float scalar);
        friend Tensor operator!=(float scalar, const Tensor& B);

        /**
         * @brief Element-wise greater-than comparison: C = (A > B)
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return Boolean tensor (1.0f = true, 0.0f = false)
         */
        friend Tensor operator>(const Tensor& A, const Tensor& B);
        friend Tensor operator>(const Tensor& A, float scalar);
        friend Tensor operator>(float scalar, const Tensor& B);

        /**
         * @brief Element-wise less-than comparison: C = (A < B)
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return Boolean tensor (1.0f = true, 0.0f = false)
         */
        friend Tensor operator<(const Tensor& A, const Tensor& B);
        friend Tensor operator<(const Tensor& A, float scalar);
        friend Tensor operator<(float scalar, const Tensor& B);

        /**
         * @brief Element-wise greater-or-equal comparison: C = (A >= B)
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return Boolean tensor (1.0f = true, 0.0f = false)
         */
        friend Tensor operator>=(const Tensor& A, const Tensor& B);
        friend Tensor operator>=(const Tensor& A, float scalar);
        friend Tensor operator>=(float scalar, const Tensor& B);

        /**
         * @brief Element-wise less-or-equal comparison: C = (A <= B)
         *
         * @param A Left operand tensor
         * @param B Right operand tensor
         * @return Boolean tensor (1.0f = true, 0.0f = false)
         */
        friend Tensor operator<=(const Tensor& A, const Tensor& B);
        friend Tensor operator<=(const Tensor& A, float scalar);
        friend Tensor operator<=(float scalar, const Tensor& B);

        /**
         * @brief Unary negation: C = -A
         *
         * Negates all elements in the tensor.
         *
         * @param A Input tensor
         * @return New tensor with negated values
         */
        friend Tensor operator-(const Tensor& A);

    protected:
        /**
         * @brief Protected constructor for creating filled tensors
         *
         * Used internally by factory methods (zeros, ones, full) to create
         * tensors initialized with a single value.
         *
         * @param shape Tensor dimensions
         * @param value Fill value for all elements
         * @param device Target device
         */
        Tensor(const std::vector<size_t>& shape,
               float value,
               bool requires_grad,
               DeviceType device = DeviceType::CPU);

        Tensor(const std::vector<size_t>& shape,
               float value,
               DeviceType device = DeviceType::CPU);

        /**
         * @brief Protected constructor from TensorImpl pointer
         *
         * Used internally to create Tensor from existing TensorImpl (e.g., views).
         *
         * @param impl Shared pointer to TensorImpl
         */
        explicit Tensor(std::shared_ptr<TensorImpl> impl);

    private:
        /**
         * @brief Validate that the tensor has an implementation before use
         *
         * @param method Name of the Tensor API entry point being called
         * @return Shared pointer to the validated implementation
         * @throws std::runtime_error if this tensor is default-constructed /
         *         uninitialized
         */
        std::shared_ptr<TensorImpl> require_impl(const char* method) const;

        /**
         * @brief Shared pointer to implementation (PIMPL pattern)
         *
         * Uses shared ownership to enable efficient copying and move semantics.
         * Multiple Tensor objects can reference the same TensorImpl for
         * copy-on-write behavior.
         */
        std::shared_ptr<TensorImpl> impl_;
    };

} // namespace cpptensor
