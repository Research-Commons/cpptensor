# How to Add a New Operation to `cpptensor`

This guide explains how to add a new operation (op) to the `cpptensor` library — from defining the op type to testing and building.

---

## ⚙Step 1: Add `OpType` Enum Value

Add your operation to the `OpType` enum:

```cpp
enum class OpType {
    // ... existing ops ...
    YourNewOp,  ///< Brief description of what it does
};
```

---

## Step 2: Implement Kernel Functions

Create optimized implementations for different ISA levels.

### A) Generic CPU Kernel (required)

File: `src/backend/cpu/your_op_cpu.cpp`

```cpp
void your_op_f32_cpu(const Tensor& A, const Tensor& B, Tensor& Out) {
    const float* a = A.data().data();
    const float* b = B.data().data();
    float* out = Out.data().data();
    size_t n = A.numel();
    
    for (size_t i = 0; i < n; ++i) {
        out[i] = /* your operation on a[i], b[i] */;
    }
}
```

---

### B) AVX2 Kernel (optional but recommended)

File: `src/backend/isa/avx2.cpp`

Header: `avx2.hpp`

```cpp
void AVX2::your_op_f32_avx2(const Tensor& A, const Tensor& B, Tensor& Out) {
    const float* a = A.data().data();
    const float* b = B.data().data();
    float* out = Out.data().data();
    size_t n = A.numel();
    
    const int stride = 8;  // 8 floats per __m256
    size_t i = 0;
    
    // SIMD loop
    for (; i + stride <= n; i += stride) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = /* AVX2 intrinsic for your op */;
        _mm256_storeu_ps(out + i, vout);
    }
    
    // Scalar remainder
    for (; i < n; ++i) {
        out[i] = /* scalar version */;
    }
}
```

---

### C) AVX512 Kernel (optional, highest performance)

File: `src/backend/isa/avx512.cpp`

Similar to AVX2 but with 512-bit vectors (`__m512`, 16 floats).

---

## Step 3: Register Kernels in Backend Loader

File: `src/backend/backend_loader.cpp`

In `initialize_kernels()`:

```cpp
void initialize_kernels() {
    auto& reg = KernelRegistry::instance();
    
    // ... existing registrations ...
    
    // Register your new op
    reg.registerKernel(OpType::YourNewOp, DeviceType::CPU,
                      CpuIsa::GENERIC, your_op_f32_cpu);
    
    #ifdef BUILD_AVX2
    reg.registerKernel(OpType::YourNewOp, DeviceType::CPU,
                      CpuIsa::AVX2, AVX2::your_op_f32_avx2);
    #endif
    
    #ifdef BUILD_AVX512
    reg.registerKernel(OpType::YourNewOp, DeviceType::CPU,
                      CpuIsa::AVX512, AVX512::your_op_f32_avx512);
    #endif
}
```

---

## Step 4: Create High-Level API Function

### A) Header

File: `include/cpptensor/ops/category/your_op.hpp`

```cpp
#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

/**
 * @brief Your operation description
 * @param A First input tensor
 * @param B Second input tensor
 * @return Result tensor
 */
Tensor your_op(const Tensor& A, const Tensor& B);

} // namespace cpptensor
```

---

### B) Implementation

File: `src/ops/category/your_op.cpp`

```cpp
#include "cpptensor/ops/category/your_op.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"

namespace cpptensor {

Tensor your_op(const Tensor& A, const Tensor& B) {
    // Validate inputs
    if (A.shape() != B.shape()) {
        throw std::runtime_error("Shape mismatch");
    }
    
    // Create output tensor
    Tensor Out = Tensor::zeros(A.shape(), A.device_type());
    
    // Dispatch to appropriate kernel
    auto& registry = KernelRegistry::instance();
    auto kernel = registry.getKernel(OpType::YourNewOp, A.device_type());
    kernel(A, B, Out);
    
    return Out;
}

} // namespace cpptensor
```

---

## Optional: Add Operator Overload

If your operation has a natural operator syntax (e.g. `%`):

In `include/cpptensor/tensor/tensor.hpp`:

```cpp
friend Tensor operator%(const Tensor&, const Tensor&);
```

In `src/tensor/tensor.cpp`:

```cpp
Tensor operator%(const Tensor& A, const Tensor& B) {
    return your_op(A, B);
}
```

---

## Examples

Refer to existing operations for reference:

| Type              | Example                       | Description                         |
| ----------------- | ----------------------------- | ----------------------------------- |
| Simple unary      | `src/ops/math/abs.cpp`        | Uses simple scalar and SIMD loops   |
| Binary arithmetic | `src/ops/arithmetic/add.cpp`  | Demonstrates elementwise binary ops |
| Complex op        | `src/ops/math/matmul.cpp`     | Uses OpenBLAS if available          |
| Activation        | `src/ops/activation/relu.cpp` | Simple and common example           |
