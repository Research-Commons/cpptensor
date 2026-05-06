# Tensor Views in cpptensor

**Last verified:** 2026-05-06  
**Verified against:** `include/cpptensor/tensor/tensor.hpp`, `src/tensor/tensor.cpp`, `src/tensor/tensorimpl.cpp`, `test/test_tensor_views.cpp`, `test/test_ops.cpp`, `test/test_tensor_serialization.cpp`

This page documents **current shipped view semantics**.

---

## Overview

cpptensor uses zero-copy views for layout-changing tensor operations. A view reuses base storage and carries its own shape/stride/offset metadata.

View-producing operations:
- `view(...)` (requires contiguous input)
- `reshape(...)` (view when possible; copy via `contiguous()` when needed)
- `flatten(...)`
- `slice(...)`
- `squeeze(...)`, `unsqueeze(...)`
- `permute(...)`, `transpose(...)`

---

## Data access contract

### `const data()`
`const Tensor::data()` returns logical row-major values for the tensor view:
- Owning tensors and full contiguous views can expose direct storage.
- Non-trivial views (slice/transpose/permute/pointer-backed view) materialize a compact logical buffer.

### Mutable `data()`
Mutable `Tensor::data()` is intentionally restricted:
- ✅ Allowed for direct compact storage (owning tensors, full contiguous views).
- ❌ Throws for sliced/permuted/transposed/pointer-backed views.

If mutable compact storage is needed, call `contiguous()` or `clone()` first.

---

## Contiguity rules

- `view(new_shape)` requires `is_contiguous() == true` and matching element count.
- `reshape(new_shape)`:
  - contiguous input -> zero-copy `view`
  - non-contiguous input -> materializes via `contiguous()` then views
- `contiguous()` returns self when direct compact storage is already exposable; otherwise copies logical view contents.
- `clone()` always deep-copies logical contents.

---

## Pointer-backed views (`from_ptr`)

`Tensor::from_ptr(...)` creates a tensor view over external memory and tracks an owner `shared_ptr<TensorImpl>` for lifetime safety.

Current behavior:
- Logical const reads work.
- Mutable `data()` throws (pointer-backed views are non-exposable for mutable vector storage).
- Kernels use `data_ptr()` internally.

---

## Serialization behavior

`Tensor::save()` materializes logical tensor contents. Saving a non-contiguous view and reloading yields a contiguous tensor with the same logical values and shape.

---

## Coverage anchors

- `test/test_tensor_views.cpp`
  - const logical data for views
  - mutable view access restrictions
  - contiguous full-view shared storage behavior
- `test/test_ops.cpp`
  - kernels over transposed/sliced views
  - `clone()`/`contiguous()` correctness for views
  - pointer-backed view behavior
- `test/test_tensor_serialization.cpp`
  - save/load materialization of non-contiguous views
