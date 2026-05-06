# cpptensor Operations Status (Source of Truth)

**Last verified:** 2026-05-06
**Verification inputs:**
- Public API headers under `include/cpptensor/`
- Kernel registration in `src/backend/backend_loader.cpp`
- Behavior tests in `test/test_ops.cpp`, `test/test_tensor.cpp`, `test/test_tensor_views.cpp`, `test/test_linear_algebra.cpp`, `test/test_cuda_dispatch.cpp`, `test/test_tensor_serialization.cpp`

This page tracks **shipped behavior only**. Roadmap material is listed separately.

---

## Implemented public API

### Tensor creation & persistence

| API | Status | Coverage note |
|---|---|---|
| `Tensor(shape, values[, device])` | ✅ Implemented | `test/test_tensor.cpp` |
| `Tensor::zeros`, `ones`, `full`, `randn` | ✅ Implemented | `test/test_tensor.cpp` |
| `Tensor::from_ptr(...)` | ✅ Implemented (advanced/view use) | view + pointer behavior in `test/test_ops.cpp` |
| `Tensor::save(path)`, `Tensor::load(path)` | ✅ Implemented | `test/test_tensor_serialization.cpp` |

> Note: there is **no** `Tensor::rand` (uniform) in the current public API.

### Elementwise arithmetic

| API | Status | Coverage note |
|---|---|---|
| `operator+`, `-`, `*`, `/` (tensor/tensor + scalar overloads) | ✅ Implemented | `test/test_ops.cpp`, `test/test_tensor.cpp` |
| `cpptensor::add/sub/mul/div/neg/pow` | ✅ Implemented | `test/test_ops.cpp`, `test/test_tensor.cpp` |
| NumPy-style broadcasting for binary arithmetic | ✅ Implemented | `test/test_ops.cpp` |
| IEEE/domain edge semantics (`div`, `log`, `sqrt`, `pow`) | ✅ Implemented on CPU | `test/test_ops.cpp` |

### Comparison operations

| API | Status | Coverage note |
|---|---|---|
| `==`, `!=`, `>`, `<`, `>=`, `<=` (tensor/tensor + scalar overloads) | ✅ Implemented | `test/test_ops.cpp` |
| Broadcasting for comparisons | ✅ Implemented | `test/test_ops.cpp` |

### Unary math + activations

| API | Status | Coverage note |
|---|---|---|
| `exp`, `log`, `sqrt` | ✅ Implemented | `test/test_ops.cpp`, `test/test_tensor.cpp`, `test/test_cuda_dispatch.cpp` |
| `abs`, `sin`, `cos`, `tan` | ✅ Implemented | no dedicated direct tests currently |
| `relu`, `sigmoid` | ✅ Implemented | no dedicated direct tests currently |

### Reductions

| API | Status | Coverage note |
|---|---|---|
| `sum`, `mean`, `max`, `min` (global + `dim` + `keepdim`) | ✅ Implemented | `test/test_ops.cpp`, `test/test_tensor.cpp` |
| Negative dimension indexing for reductions | ✅ Implemented | `test/test_ops.cpp` |

### Linear algebra

| API | Status | Coverage note |
|---|---|---|
| `matmul`, `dot`, `tensordot` | ✅ Implemented | `test/test_ops.cpp` |
| `svd`, `eig`, `eig_symmetric` | ✅ Implemented | `test/test_linear_algebra.cpp` |

### Tensor views & manipulation

| API | Status | Coverage note |
|---|---|---|
| `view` | ✅ Implemented | `test/test_tensor_views.cpp`, `test/test_ops.cpp` |
| `reshape`, `flatten` | ✅ Implemented | covered indirectly through view/contiguity paths; no dedicated standalone tests currently |
| `slice` (positive step), `squeeze`, `unsqueeze` | ✅ Implemented | `test/test_ops.cpp`, `test/test_tensor_views.cpp` |
| `permute`, `transpose`, `is_contiguous`, `contiguous`, `clone` | ✅ Implemented | `test/test_ops.cpp`, `test/test_tensor_views.cpp` |
| `cat`, `stack` | ✅ Implemented | `test/test_ops.cpp` |

### Device/dispatch behavior (current)

| Behavior | Status | Coverage note |
|---|---|---|
| CPU kernels registered for shipped ops | ✅ Implemented | runtime use across all op tests |
| AVX2/AVX512 dispatch where available | ✅ Implemented | ISA override coverage in `test/test_ops.cpp` |
| CUDA-tagged tensors: only limited kernel registration; clear missing-kernel failures for others | ✅ Current behavior documented | `test/test_cuda_dispatch.cpp` |
| Mixed-device binary ops fail fast | ✅ Implemented | `test/test_cuda_dispatch.cpp` |

---

## Roadmap / not-yet-shipped API

These are **not** currently part of the public API and should not be treated as implemented:

- Reductions: `argmax`, `argmin`, `prod`, `std`, `var`, `norm`
- Manipulation/indexing: `split`, `chunk`, `expand`, `repeat`, `tile`, `gather`, `scatter`, advanced indexing/boolean indexing helpers

---

## Maintenance checklist (doc drift guard)

When adding/changing ops:
1. Update/verify headers in `include/cpptensor/ops/*` and/or `include/cpptensor/tensor/tensor.hpp`.
2. Add/update tests in `test/` for API behavior and error boundaries.
3. Update this page’s status table and `Last verified` date.
4. Keep roadmap items separate from shipped behavior.
