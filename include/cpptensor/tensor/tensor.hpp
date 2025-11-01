#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>

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
               DeviceType device = DeviceType::CPU);

        /**
         * @brief Default constructor - creates empty tensor
         *
         * Creates a tensor with null implementation pointer. Operations on
         * default-constructed tensors are undefined until properly initialized.
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
         * @brief Get const reference to underlying data
         *
         * @return Const reference to flattened data vector (row-major order)
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
         * @brief Get mutable reference to underlying data
         *
         * @return Mutable reference to flattened data vector
         *
         * @warning Modifying data directly may break invariants. Prefer operations.
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

        // =============== Reduction Operations ===============
        // TODO: Implement reduction operations
        // Tensor sum(int dim = -1, bool keepdim = false) const;
        // Tensor mean(int dim = -1, bool keepdim = false) const;
        // Tensor max(int dim = -1, bool keepdim = false) const;

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
               DeviceType device = DeviceType::CPU);

    private:
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