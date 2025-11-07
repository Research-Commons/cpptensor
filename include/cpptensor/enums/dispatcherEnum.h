#pragma once

/**
 * @file dispatcherEnum.h
 * @brief Core enumerations for kernel dispatch system
 *
 * This file defines the type system for cpptensor's multi-backend kernel
 * dispatch mechanism. The dispatch system routes operations to optimized
 * implementations based on:
 * - Operation type (Add, Mul, Matmul, etc.)
 * - Target device (CPU, CUDA, etc.)
 * - CPU instruction set (Generic, AVX2, AVX512)
 *
 * Architecture Overview:
 * ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
 * │   OpType    │────▶│ DispatchKey  │────▶│   Kernel    │
 * │ (Add, Mul)  │     │  (routing)   │     │ (optimized) │
 * └─────────────┘     └──────────────┘     └─────────────┘
 *        │                    │                    │
 *   DeviceType            CpuIsa          Actual implementation
 *   (CPU/CUDA)         (AVX2/AVX512)      (matmul_avx2.cpp)
 *
 * Example Dispatch Flow:
 * 1. User calls: C = A + B
 * 2. System creates DispatchKey{Add, CPU, AVX512}
 * 3. Registry looks up registered kernel for this key
 * 4. If AVX512 kernel exists → use it
 *    Else if AVX2 exists → fallback to AVX2
 *    Else → fallback to generic CPU implementation
 *
 * @see KernelRegistry for dispatch implementation
 * @see backend_loader.cpp for kernel registration
 */

/**
 * @enum DeviceType
 * @brief Target device for tensor operations
 *
 * Specifies where tensor data is stored and where operations execute.
 * Different devices require different memory management and kernel
 * implementations.
 *
 * Supported Devices:
 * - CPU: Main system memory, uses CPU instructions (scalar/AVX2/AVX512)
 * - CUDA: NVIDIA GPU memory, uses CUDA kernels (future support)
 *
 * @note Current implementation primarily supports CPU. CUDA is defined
 *       but not fully implemented.
 *
 * @example
 * ```cpp
 * Tensor A = Tensor::zeros({100, 100}, DeviceType::CPU);
 * Tensor B = Tensor::randn({100, 100}, DeviceType::CUDA);  // Future support
 * ```
 */
enum class DeviceType {
    CPU,   ///< Central Processing Unit (x86/ARM)
    CUDA   ///< NVIDIA CUDA-capable GPU (future)
};

/**
 * @enum OpType
 * @brief Enumeration of all supported tensor operations
 *
 * Each operation has corresponding kernel implementations for different
 * device types and ISA levels. Operations are categorized as:
 *
 * Element-wise Binary: Add, Sub, Mul, Div, Pow
 * - Apply operation to corresponding elements: C[i] = A[i] op B[i]
 * - Support broadcasting (NumPy-style)
 *
 * Element-wise Unary: Exp, Log, Abs, Sqrt, Sin, Cos, Tan, Sigmoid, Relu
 * - Apply operation to each element: B[i] = op(A[i])
 *
 * Linear Algebra: Matmul, Dot
 * - Matrix multiplication and dot products
 * - Optimized with BLAS when available
 *
 * @note When adding a new operation:
 *       1. Add enum value here
 *       2. Implement kernels (e.g., add_f32_avx2.cpp)
 *       3. Register in backend_loader.cpp
 *       4. Create high-level API function (e.g., ops/arithmetic/add.cpp)
 *
 * @see "HOW TO ADD A NEW OPERATION" section below for detailed steps
 */
enum class OpType {
    // =============== Arithmetic Operations ===============
    Add,      ///< Element-wise addition: C = A + B
    Mul,      ///< Element-wise multiplication: C = A * B (Hadamard product)
    Sub,      ///< Element-wise subtraction: C = A - B
    Div,      ///< Element-wise division: C = A / B
    Pow,      ///< Element-wise power: C = A^B

    // =============== Transcendental Functions ===============
    Exp,      ///< Element-wise exponential: B = e^A
    Log,      ///< Element-wise natural logarithm: B = ln(A)
    Sqrt,     ///< Element-wise square root: B = √A

    // =============== Trigonometric Functions ===============
    Sin,      ///< Element-wise sine: B = sin(A)
    Cos,      ///< Element-wise cosine: B = cos(A)
    Tan,      ///< Element-wise tangent: B = tan(A)

    // =============== Activation Functions ===============
    Sigmoid,  ///< Sigmoid activation: B = 1/(1 + e^(-A))
    Relu,     ///< ReLU activation: B = max(0, A)

    // =============== Utility Functions ===============
    Abs,      ///< Element-wise absolute value: B = |A|

    // =============== Reduction Operations ===============
    Sum,      ///< Reduction sum: B = sum(A, dim, keepdim)
    Mean,     ///< Reduction mean: B = mean(A, dim, keepdim)
    Max,      ///< Reduction max: B = max(A, dim, keepdim)
    Min,      ///< Reduction min: B = min(A, dim, keepdim)

    // =============== Linear Algebra ===============
    Matmul,   ///< Matrix multiplication: C = A @ B
    Dot       ///< Dot product (vector inner product): scalar = A · B
};

/**
 * @enum CpuIsa
 * @brief CPU instruction set architecture levels for optimization
 *
 * Modern CPUs support different SIMD (Single Instruction Multiple Data)
 * instruction sets. Higher ISA levels provide better performance through
 * wider vector operations and specialized instructions.
 *
 * ISA Capabilities:
 *
 * GENERIC (Baseline):
 * - Scalar operations only (one element at a time)
 * - Works on all x86-64 CPUs
 * - Slowest but most compatible
 * - Example: for(int i=0; i<n; i++) c[i] = a[i] + b[i];
 *
 * AVX2 (Advanced Vector Extensions 2):
 * - 256-bit vectors (8 floats at once)
 * - FMA (Fused Multiply-Add) support
 * - Available on Intel Haswell (2013+), AMD Excavator (2015+)
 * - ~2-4× faster than generic for element-wise ops
 * - Example: __m256 va = _mm256_loadu_ps(a); // Load 8 floats
 *
 * AVX512 (Advanced Vector Extensions 512):
 * - 512-bit vectors (16 floats at once)
 * - Masking for efficient tail handling
 * - Available on Intel Skylake-X (2017+), AMD Zen 4 (2022+)
 * - ~4-8× faster than generic for element-wise ops
 * - Example: __m512 va = _mm512_loadu_ps(a); // Load 16 floats
 *
 * Performance Hierarchy (typical):
 * - AVX512: 100-200 GFLOPS (element-wise)
 * - AVX2:   50-100 GFLOPS (element-wise)
 * - GENERIC: 10-30 GFLOPS (element-wise)
 *
 * @note The dispatch system automatically selects the best available ISA
 *       at runtime based on CPU capabilities (detected via CPUID).
 *
 * @see initialize_kernels() in backend_loader.cpp for ISA detection
 */
enum class CpuIsa {
    GENERIC,  ///< Portable C++ code, no SIMD (works everywhere)
    AVX2,     ///< AVX2 SIMD: 256-bit vectors, 8 floats/cycle
    AVX512    ///< AVX512 SIMD: 512-bit vectors, 16 floats/cycle
};

/**
 * @struct DispatchKey
 * @brief Routing key for kernel dispatch system
 *
 * Combines operation type, device, and ISA into a unique key for looking
 * up the appropriate kernel implementation. The kernel registry uses this
 * key to route operations to optimized implementations.
 *
 * Key Components:
 * - op:  What operation to perform (Add, Matmul, etc.)
 * - dev: Where to execute (CPU, CUDA)
 * - isa: CPU optimization level (Generic, AVX2, AVX512)
 *
 * Dispatch Strategy:
 * 1. Try exact match: (op, dev, best_isa)
 * 2. Fallback to lower ISA: (op, dev, AVX2) if AVX512 not found
 * 3. Final fallback: (op, dev, GENERIC) if AVX2 not found
 * 4. Throw error if no implementation exists
 *
 * Example Dispatch Scenarios:
 *
 * Scenario 1 - Full optimization available:
 *   Request: {Add, CPU, AVX512}
 *   Result:  Uses add_f32_avx512() → Maximum performance
 *
 * Scenario 2 - Fallback to AVX2:
 *   Request: {Sin, CPU, AVX512}
 *   Missing: sin_f32_avx512()
 *   Result:  Uses sin_f32_avx2() → Good performance
 *
 * Scenario 3 - Fallback to generic:
 *   Request: {CustomOp, CPU, AVX512}
 *   Missing: AVX512 and AVX2 versions
 *   Result:  Uses generic C++ implementation → Works everywhere
 *
 * @note The operator< is required for std::map key comparison
 *
 * @example
 * ```cpp
 * // Create dispatch key
 * DispatchKey key{OpType::Add, DeviceType::CPU, CpuIsa::AVX512};
 *
 * // Look up kernel
 * auto kernel = registry.getKernel(key);
 *
 * // Execute operation
 * kernel(A, B, C);  // C = A + B using AVX512
 * ```
 */
struct DispatchKey {
    OpType op;        ///< Operation to perform (Add, Mul, etc.)
    DeviceType dev;   ///< Target device (CPU, CUDA)
    CpuIsa isa;       ///< CPU instruction set level (Generic, AVX2, AVX512)

    /**
     * @brief Comparison operator for std::map ordering
     *
     * Provides lexicographic ordering: first by op, then dev, then isa.
     * Required for using DispatchKey as std::map key in KernelRegistry.
     *
     * @param o Other DispatchKey to compare against
     * @return true if this key is less than o
     */
    bool operator<(const DispatchKey& o) const {
        if (op != o.op) return op < o.op;
        if (dev != o.dev) return dev < o.dev;
        return isa < o.isa;
    }
};

