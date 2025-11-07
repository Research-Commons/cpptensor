# cpptensor Operations Catalog

**Comprehensive inventory of implemented and missing tensor operations**

**Last Updated:** November 5, 2025  
**Repository:** cpptensor  
**Status:** Active Development

---

## Table of Contents

1. [✅ Implemented Operations](#implemented-operations)
2. [❌ Missing Operations](#missing-operations)
3. [🔄 Partially Implemented](#partially-implemented)
4. [📋 Operation Categories](#operation-categories)
5. [🎯 Priority Roadmap](#priority-roadmap)

---

## ✅ Implemented Operations

### 1. Tensor Creation & Initialization

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| **From data** | `Tensor(shape, values)` | Create from vector of values | ✅ Working |
| **Zeros** | `Tensor::zeros(shape)` | Create tensor filled with zeros | ✅ Working |
| **Ones** | `Tensor::ones(shape)` | Create tensor filled with ones | ✅ Working |
| **Full** | `Tensor::full(shape, value)` | Create tensor filled with scalar | ✅ Working |
| **Random normal** | `Tensor::randn(shape)` | Random values from N(0,1) | ✅ Working |
| **Random uniform** | `Tensor::rand(shape)` | Random values from U(0,1) | ✅ Working |
| **From pointer** | `Tensor::from_ptr(shape, ptr, owner)` | Zero-copy view from raw pointer | ✅ Working |

### 2. Element-wise Arithmetic Operations

| Operation | Operator/Function | Description | Status |
|-----------|------------------|-------------|--------|
| **Addition** | `A + B`, `add(A, B)` | Element-wise addition | ✅ Working |
| **Subtraction** | `A - B`, `sub(A, B)` | Element-wise subtraction | ✅ Working |
| **Multiplication** | `A * B`, `mul(A, B)` | Element-wise multiplication (Hadamard) | ✅ Working |
| **Division** | `A / B`, `div(A, B)` | Element-wise division | ✅ Working |
| **Power** | `pow(A, B)` | Element-wise power | ✅ Working |
| **Negation** | `-A`, `neg(A)` | Element-wise negation | ✅ Working |

### 3. Mathematical Functions (Unary)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| **Exponential** | `exp(A)` | e^x for each element | ✅ Working |
| **Natural log** | `log(A)` | ln(x) for each element | ✅ Working |
| **Square root** | `sqrt(A)` | √x for each element | ✅ Working |
| **Absolute value** | `abs(A)` | \|x\| for each element | ✅ Working |
| **Sine** | `sin(A)` | sin(x) for each element | ✅ Working |
| **Cosine** | `cos(A)` | cos(x) for each element | ✅ Working |
| **Tangent** | `tan(A)` | tan(x) for each element | ✅ Working |

### 4. Activation Functions

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| **ReLU** | `relu(A)` | max(0, x) for each element | ✅ Working |
| **Sigmoid** | `sigmoid(A)` | 1 / (1 + e^(-x)) | ✅ Working |

### 5. Linear Algebra Operations

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| **Matrix multiply** | `matmul(A, B)` | Matrix multiplication (2D and ND batched) | ✅ Working (Optimized) |
| **Dot product** | `dot(A, B)` | Vector dot product (1D tensors) | ✅ Working |
| **Tensor dot** | `tensordot(A, B, axes)` | Generalized tensor contraction | ✅ Working |
| **SVD** | `svd(A, full_matrices, compute_uv)` | Singular value decomposition | ✅ Working |
| **Eigenvalue** | `eig(A, compute_eigenvectors)` | General eigenvalue decomposition | ✅ Working |
| **Symmetric eig** | `eig_symmetric(A, compute_eigenvectors)` | Symmetric eigenvalue decomposition | ✅ Working |

### 6. Tensor Manipulation (Views & Reshaping)

| Operation | Method | Description | Status |
|-----------|--------|-------------|--------|
| **View** | `A.view(new_shape)` | Reshape without copying (zero-copy) | ✅ Working |
| **Reshape** | `A.reshape(new_shape)` | Change shape (may copy if not contiguous) | ✅ Working |
| **Flatten** | `A.flatten(start_dim, end_dim)` | Flatten dimensions into 1D | ✅ Working |
| **Squeeze** | `A.squeeze(dim)` | Remove size-1 dimensions | ✅ Working |
| **Unsqueeze** | `A.unsqueeze(dim)` | Add size-1 dimension | ✅ Working |
| **Permute** | `A.permute(dims)` | Reorder dimensions (generalized transpose) | ✅ Working |
| **Transpose** | `A.transpose(dim0, dim1)` | Swap two dimensions | ✅ Working |
| **Contiguous** | `A.contiguous()` | Ensure contiguous memory layout | ✅ Working |
| **Clone** | `A.clone()` | Deep copy of tensor | ✅ Working |

### 7. Tensor Properties & Inspection

| Operation | Method | Description | Status |
|-----------|--------|-------------|--------|
| **Shape** | `A.shape()` | Get tensor dimensions | ✅ Working |
| **Size/numel** | `A.numel()` | Total number of elements | ✅ Working |
| **Dimensions** | `A.ndim()` | Number of dimensions | ✅ Working |
| **Device** | `A.device_type()` | Get device (CPU/CUDA) | ✅ Working |
| **Strides** | `A.strides()` | Get stride information | ✅ Working |
| **Is contiguous** | `A.is_contiguous()` | Check if contiguous layout | ✅ Working |
| **Data access** | `A.data()` | Access underlying data vector | ✅ Working |
| **Print** | `A.print()` | Print tensor contents | ✅ Working |

### 8. Backend & Dispatch

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| **Initialize kernels** | `initialize_kernels()` | Load backend (OpenBLAS/AVX/CUDA) | ✅ Working |
| **Backend selection** | Device type enum | CPU, CUDA, AVX2, AVX512 | ✅ Working |

---

## ❌ Missing Operations

### 1. Reduction Operations (High Priority)

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Sum** | `A.sum(dim, keepdim)` | Sum along dimension(s) | 🔴 P0 |
| **Mean** | `A.mean(dim, keepdim)` | Average along dimension(s) | 🔴 P0 |
| **Max** | `A.max(dim, keepdim)` | Maximum along dimension | 🟡 P1 |
| **Min** | `A.min(dim, keepdim)` | Minimum along dimension | 🟡 P1 |
| **Argmax** | `A.argmax(dim, keepdim)` | Index of maximum value | 🟡 P1 |
| **Argmin** | `A.argmin(dim, keepdim)` | Index of minimum value | 🟡 P1 |
| **Prod** | `A.prod(dim, keepdim)` | Product along dimension | 🟢 P2 |
| **Std** | `A.std(dim, keepdim)` | Standard deviation | 🟢 P2 |
| **Var** | `A.var(dim, keepdim)` | Variance | 🟢 P2 |
| **Norm** | `A.norm(p, dim)` | p-norm along dimension | 🟢 P2 |

**Impact:** Critical for neural networks (softmax, normalization layers)

### 2. Tensor Manipulation (High Priority)

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Concatenate** | `torch.cat([A, B], dim)` | Join tensors along dimension | 🔴 P0 |
| **Stack** | `torch.stack([A, B], dim)` | Stack tensors (new dimension) | 🔴 P0 |
| **Split** | `torch.split(A, size, dim)` | Split tensor into chunks | 🟡 P1 |
| **Chunk** | `torch.chunk(A, chunks, dim)` | Split into N chunks | 🟡 P1 |
| **Expand** | `A.expand(shape)` | Broadcast to new shape (no copy) | 🟡 P1 |
| **Repeat** | `A.repeat(counts)` | Repeat tensor along dimensions | 🟡 P1 |
| **Tile** | `A.tile(reps)` | Repeat entire tensor | 🟢 P2 |
| **Gather** | `torch.gather(A, dim, idx)` | Gather values along dimension | 🟢 P2 |
| **Scatter** | `torch.scatter(A, dim, idx, val)` | Scatter values along dimension | 🟢 P2 |

**Impact:** Essential for advanced network architectures (ResNet, Transformers)

### 3. Indexing & Slicing (High Priority)

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Basic slicing** | `A[0:10, :, 5]` | NumPy-style slicing | 🔴 P0 |
| **Advanced indexing** | `A[[0, 2, 4], [1, 3, 5]]` | Index with arrays | 🟡 P1 |
| **Boolean indexing** | `A[mask]` | Index with boolean mask | 🟡 P1 |
| **Where** | `torch.where(cond, A, B)` | Select elements based on condition | 🟡 P1 |
| **Masked select** | `torch.masked_select(A, mask)` | Select elements where mask is true | 🟢 P2 |
| **Index select** | `torch.index_select(A, dim, idx)` | Select along dimension | 🟢 P2 |

**Impact:** Critical for data manipulation and masking in NLP/Vision

### 4. Comparison & Logical Operations

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Equal** | `A == B`, `torch.eq(A, B)` | Element-wise equality | 🟡 P1 |
| **Not equal** | `A != B`, `torch.ne(A, B)` | Element-wise inequality | 🟡 P1 |
| **Greater** | `A > B`, `torch.gt(A, B)` | Element-wise greater than | 🟡 P1 |
| **Less** | `A < B`, `torch.lt(A, B)` | Element-wise less than | 🟡 P1 |
| **Greater/equal** | `A >= B`, `torch.ge(A, B)` | Element-wise >= | 🟡 P1 |
| **Less/equal** | `A <= B`, `torch.le(A, B)` | Element-wise <= | 🟡 P1 |
| **Logical AND** | `torch.logical_and(A, B)` | Element-wise AND | 🟢 P2 |
| **Logical OR** | `torch.logical_or(A, B)` | Element-wise OR | 🟢 P2 |
| **Logical NOT** | `torch.logical_not(A)` | Element-wise NOT | 🟢 P2 |
| **Allclose** | `torch.allclose(A, B)` | Check approximate equality | 🟢 P2 |

**Impact:** Needed for control flow and validation

### 5. Advanced Activation Functions

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Tanh** | `torch.tanh(A)` | Hyperbolic tangent | 🟡 P1 |
| **Softmax** | `torch.softmax(A, dim)` | Softmax along dimension | 🔴 P0 |
| **Log softmax** | `torch.log_softmax(A, dim)` | Log of softmax (numerically stable) | 🟡 P1 |
| **Leaky ReLU** | `torch.nn.functional.leaky_relu(A)` | Leaky ReLU | 🟡 P1 |
| **ELU** | `torch.nn.functional.elu(A)` | Exponential Linear Unit | 🟢 P2 |
| **GELU** | `torch.nn.functional.gelu(A)` | Gaussian Error Linear Unit | 🟡 P1 |
| **Swish/SiLU** | `torch.nn.functional.silu(A)` | Sigmoid Linear Unit | 🟢 P2 |
| **Softplus** | `torch.nn.functional.softplus(A)` | Smooth approximation of ReLU | 🟢 P2 |

**Impact:** Essential for modern deep learning (GELU in Transformers)

### 6. Broadcasting & Type Conversions

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Broadcasting** | Automatic | NumPy-style broadcasting | 🔴 P0 |
| **Type casting** | `A.to(dtype)` | Convert data type | 🟡 P1 |
| **Device transfer** | `A.to(device)` | Move tensor to device | 🟡 P1 |
| **Fill** | `A.fill_(value)` | Fill tensor with scalar | 🟢 P2 |
| **Copy** | `A.copy_(B)` | Copy B's data into A | 🟢 P2 |

**Impact:** Broadcasting critical for vectorized operations

### 7. Linear Algebra (Advanced)

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **QR decomposition** | `torch.linalg.qr(A)` | QR factorization | 🟢 P2 |
| **Cholesky** | `torch.linalg.cholesky(A)` | Cholesky decomposition | 🟢 P2 |
| **LU decomposition** | `torch.linalg.lu(A)` | LU factorization | 🟢 P2 |
| **Matrix inverse** | `torch.linalg.inv(A)` | Matrix inverse | 🟡 P1 |
| **Determinant** | `torch.linalg.det(A)` | Matrix determinant | 🟢 P2 |
| **Matrix rank** | `torch.linalg.matrix_rank(A)` | Rank of matrix | 🟢 P2 |
| **Solve linear** | `torch.linalg.solve(A, b)` | Solve Ax = b | 🟡 P1 |
| **Cross product** | `torch.cross(A, B)` | Cross product | 🟢 P3 |
| **Outer product** | `torch.outer(A, B)` | Outer product | 🟢 P3 |

**Impact:** Useful for numerical computing, less critical for DL

### 8. Convolution Operations

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Conv1d** | `torch.nn.functional.conv1d` | 1D convolution | 🟡 P1 |
| **Conv2d** | `torch.nn.functional.conv2d` | 2D convolution | 🔴 P0 |
| **Conv3d** | `torch.nn.functional.conv3d` | 3D convolution | 🟢 P2 |
| **ConvTranspose2d** | `torch.nn.functional.conv_transpose2d` | Transposed convolution | 🟡 P1 |
| **MaxPool2d** | `torch.nn.functional.max_pool2d` | 2D max pooling | 🔴 P0 |
| **AvgPool2d** | `torch.nn.functional.avg_pool2d` | 2D average pooling | 🟡 P1 |
| **AdaptiveAvgPool** | `torch.nn.functional.adaptive_avg_pool2d` | Adaptive pooling | 🟡 P1 |

**Impact:** Critical for CNN architectures

### 9. Loss Functions

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **MSE Loss** | `torch.nn.functional.mse_loss` | Mean squared error | 🔴 P0 |
| **Cross entropy** | `torch.nn.functional.cross_entropy` | Cross entropy loss | 🔴 P0 |
| **BCE Loss** | `torch.nn.functional.binary_cross_entropy` | Binary cross entropy | 🟡 P1 |
| **L1 Loss** | `torch.nn.functional.l1_loss` | L1 (MAE) loss | 🟡 P1 |
| **KL Divergence** | `torch.nn.functional.kl_div` | KL divergence | 🟢 P2 |
| **NLL Loss** | `torch.nn.functional.nll_loss` | Negative log likelihood | 🟢 P2 |

**Impact:** Essential for training neural networks

### 10. Normalization Operations

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Batch norm** | `torch.nn.functional.batch_norm` | Batch normalization | 🔴 P0 |
| **Layer norm** | `torch.nn.functional.layer_norm` | Layer normalization | 🟡 P1 |
| **Instance norm** | `torch.nn.functional.instance_norm` | Instance normalization | 🟢 P2 |
| **Group norm** | `torch.nn.functional.group_norm` | Group normalization | 🟢 P2 |

**Impact:** Critical for modern architectures (Transformers, ResNets)

### 11. Autograd & Gradient Operations

| Operation | PyTorch Equivalent | Description | Priority |
|-----------|-------------------|-------------|----------|
| **Backward** | `loss.backward()` | Compute gradients | 🔴 P0 |
| **Grad accumulation** | `A.grad` | Access gradient | 🔴 P0 |
| **Zero grad** | `optimizer.zero_grad()` | Clear gradients | 🔴 P0 |
| **Detach** | `A.detach()` | Detach from computation graph | 🟡 P1 |
| **No grad context** | `torch.no_grad()` | Disable gradient tracking | 🟡 P1 |
| **Gradient clipping** | `torch.nn.utils.clip_grad_norm_` | Clip gradients | 🟡 P1 |

**Impact:** Fundamental for training (currently partially implemented)

---

## 🔄 Partially Implemented

### Autograd System
- **Status:** Infrastructure exists but incomplete
- **What works:** Basic gradient tracking hooks (`grad_fn`)
- **What's missing:**
    - Backward pass implementation
    - Gradient accumulation
    - Higher-order derivatives
- **Files:** `include/cpptensor/autograd/`

### Broadcasting
- **Status:** Not implemented
- **What works:** Operations on same-shaped tensors
- **What's missing:**
    - Automatic shape broadcasting
    - Broadcasting rules (NumPy-compatible)
- **Impact:** Limits vectorization and expressiveness

### Device Management
- **Status:** Enum exists, limited functionality
- **What works:** Device type specification (CPU/CUDA)
- **What's missing:**
    - Actual CUDA implementation
    - Device-to-device transfers
    - Multi-GPU support
- **Files:** `include/cpptensor/enums/dispatcherEnum.h`

---

## 📋 Operation Categories Summary

| Category | Implemented | Missing | Coverage |
|----------|-------------|---------|----------|
| **Creation** | 7 | 0 | 100% ✅ |
| **Arithmetic** | 6 | 0 | 100% ✅ |
| **Math (Unary)** | 7 | 0 | 100% ✅ |
| **Activation** | 2 | 7 | 22% ⚠️ |
| **Linear Algebra** | 6 | 9 | 40% ⚠️ |
| **Manipulation** | 9 | 9 | 50% ⚠️ |
| **Reduction** | 0 | 10 | 0% ❌ |
| **Comparison** | 0 | 10 | 0% ❌ |
| **Indexing** | 0 | 6 | 0% ❌ |
| **Convolution** | 0 | 7 | 0% ❌ |
| **Loss Functions** | 0 | 6 | 0% ❌ |
| **Normalization** | 0 | 4 | 0% ❌ |
| **Autograd** | 10% | 90% | 10% ❌ |

**Overall Coverage:** ~35% of essential PyTorch operations

---

## 🎯 Priority Roadmap

### Phase 1: Core Operations (Next Sprint) 🔴

**Goal:** Enable basic neural network training

| Priority | Operation | Reason | Estimated Effort |
|----------|-----------|--------|------------------|
| **P0** | `sum(dim)`, `mean(dim)` | Needed for loss functions | 2-3 days |
| **P0** | `softmax(dim)` | Essential for classification | 1-2 days |
| **P0** | Broadcasting support | Critical for vectorization | 3-5 days |
| **P0** | `cat(tensors, dim)` | Data manipulation | 2 days |
| **P0** | Basic slicing `A[i:j]` | Tensor indexing | 3-4 days |
| **P0** | Cross entropy loss | Training loss | 2 days |
| **P0** | MSE loss | Regression loss | 1 day |

**Total:** ~15-20 days

### Phase 2: Neural Network Essentials (Month 2) 🟡

**Goal:** Support CNN and basic architectures

| Priority | Operation | Reason | Estimated Effort |
|----------|-----------|--------|------------------|
| **P1** | `conv2d` | Convolutional layers | 5-7 days |
| **P1** | `max_pool2d` | Pooling layers | 2-3 days |
| **P1** | `batch_norm` | Normalization | 3-4 days |
| **P1** | `max/min(dim)` | Pooling & statistics | 2 days |
| **P1** | Comparison ops (`>`, `<`, etc.) | Control flow | 2-3 days |
| **P1** | `tanh`, `leaky_relu` | More activations | 1-2 days |
| **P1** | `stack`, `split` | Data manipulation | 3 days |

**Total:** ~18-25 days

### Phase 3: Advanced Features (Month 3) 🟢

**Goal:** Transformer support and optimization

| Priority | Operation | Reason | Estimated Effort |
|----------|-----------|--------|------------------|
| **P2** | `layer_norm` | Transformer architecture | 2-3 days |
| **P2** | `gelu` activation | Modern activations | 1 day |
| **P2** | `einsum` | Flexible tensor ops | 5-7 days |
| **P2** | Advanced indexing | Complex slicing | 4-5 days |
| **P2** | `gather`, `scatter` | Embedding layers | 3-4 days |
| **P2** | Gradient clipping | Training stability | 2 days |

**Total:** ~17-25 days

### Phase 4: Production Readiness (Month 4+)

- Complete autograd backward pass
- CUDA implementation
- Optimization passes (fusion, memory planning)
- Comprehensive testing
- Documentation
- Benchmarking suite

---

## Implementation Notes

### Quick Wins (< 1 day each)
1. `tanh` - just wrap `std::tanh`
2. `neg` - already implemented as unary `-`
3. `MSE loss` - simple: `mean((A - B).pow(2))`
4. `L1 loss` - simple: `mean(abs(A - B))`
5. `fill_(value)` - straightforward loop

### Medium Complexity (2-4 days)
1. `sum(dim)` - reduction with axis handling
2. `softmax(dim)` - exp + normalize (watch numerical stability!)
3. `cat/stack` - memory layout + copying
4. Basic slicing - view system extension
5. `conv2d` - use existing im2col or direct approach

### High Complexity (5+ days)
1. Broadcasting - affects entire operation system
2. Advanced indexing - complex memory patterns
3. `einsum` - generalized tensor contraction
4. Autograd backward - dependency graph traversal
5. CUDA kernels - entirely new backend

---

## Design Considerations

### Memory Efficiency
- ✅ **Zero-copy views** implemented (view, reshape, transpose)
- ❌ **Broadcasting** needs view-based implementation (avoid copies)
- ❌ **Slicing** should return views, not copies

### Performance
- ✅ **Matmul** highly optimized (OpenBLAS + transpose detection)
- ⚠️ **Element-wise ops** could benefit from vectorization (AVX/SIMD)
- ❌ **Reductions** need parallel implementation
- ❌ **Convolutions** need im2col or Winograd optimization

### API Design
- ✅ **Method chaining** works well (`A.transpose().contiguous()`)
- ⚠️ **In-place ops** missing (need `_` suffix: `A.add_(B)`)
- ❌ **Operator overloading** could be extended (`A[i:j]`)

---

## Comparison with Major Frameworks

| Feature | cpptensor | PyTorch | NumPy | TensorFlow | Status |
|---------|-----------|---------|-------|------------|--------|
| **Basic arithmetic** | ✅ | ✅ | ✅ | ✅ | Complete |
| **Matmul** | ✅ | ✅ | ✅ | ✅ | Optimized |
| **Views/reshaping** | ✅ | ✅ | ✅ | ✅ | Complete |
| **Reductions** | ❌ | ✅ | ✅ | ✅ | Missing |
| **Broadcasting** | ❌ | ✅ | ✅ | ✅ | Missing |
| **Slicing** | ❌ | ✅ | ✅ | ✅ | Missing |
| **Convolutions** | ❌ | ✅ | ❌ | ✅ | Missing |
| **Autograd** | 10% | ✅ | ❌ | ✅ | Partial |
| **GPU support** | ❌ | ✅ | ❌ | ✅ | Planned |

---

## Contributing Guidelines

### Adding New Operations

1. **Create header file:** `include/cpptensor/ops/<category>/<op>.hpp`
2. **Implement:** `src/ops/<category>/<op>.cpp`
3. **Add dispatcher:** Update backend dispatcher if needed
4. **Write tests:** Add to test suite (once test infrastructure is fixed)
5. **Document:** Add to this catalog
6. **Benchmark:** Compare with PyTorch/NumPy

### Operation Template

```cpp
// include/cpptensor/ops/category/operation.hpp
#pragma once
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    /**
     * @brief Brief description
     * 
     * @param A Input tensor
     * @param param Operation parameter
     * @return Tensor Result
     */
    Tensor operation(const Tensor& A, int param);
}

// src/ops/category/operation.cpp
#include "cpptensor/ops/category/operation.hpp"

namespace cpptensor {
    Tensor operation(const Tensor& A, int param) {
        // Implementation
        // 1. Validate inputs
        // 2. Allocate output
        // 3. Call backend/kernel
        // 4. Return result
    }
}
```

---

## FAQ

**Q: Why are reductions missing?**  
A: Reductions require careful dimension handling and are the next priority.

**Q: When will autograd be complete?**  
A: Backward pass implementation is planned for Phase 3-4. Infrastructure exists.

**Q: Is CUDA support planned?**  
A: Yes, but after core CPU operations are complete. Backend system is designed for it.

**Q: Why no broadcasting?**  
A: Complex feature affecting all operations. Being implemented in Phase 1.

**Q: Can I use this for production?**  
A: Currently suitable for research/prototyping. Production use after Phase 4.

---

## Resources

- **PyTorch Docs:** https://pytorch.org/docs/stable/torch.html
- **NumPy Reference:** https://numpy.org/doc/stable/reference/
- **Project README:** [README.md](README.md)
- **Performance Analysis:** [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)
- **View Architecture:** [docs/VIEW_ARCHITECTURE.md](docs/VIEW_ARCHITECTURE.md)

---

*Last updated: November 5, 2025*  
*Maintainer: cpptensor team*  
*License: MIT*
