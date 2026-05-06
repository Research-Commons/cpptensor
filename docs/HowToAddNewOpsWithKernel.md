# How to Add a New Operation to `cpptensor`

**Last verified:** 2026-05-06  
**Verified against:** `include/cpptensor/enums/dispatcherEnum.h`, `src/backend/backend_loader.cpp`, `src/backend/cpu_backend.cpp`, `src/ops/*`, `test/test_ops.cpp`

This guide reflects the **current repository layout**.

---

## 1) Define operation surface

Pick the API shape first:
- Kernel-dispatched op (most arithmetic/unary/reduction/comparison ops)
- Direct high-level op (some manipulation ops such as `cat`/`stack` do not rely on `OpType` dispatch)

If dispatch-based, add an enum entry in `include/cpptensor/enums/dispatcherEnum.h`.

---

## 2) Implement kernels

### Required: generic CPU kernel

Implement in `src/backend/cpu_backend.cpp` and declare in `include/cpptensor/backend/cpu_backend.h`.

### Optional: SIMD kernels

- AVX2: `src/backend/isa/avx2.cpp` (+ declaration in `include/cpptensor/backend/isa/avx2.hpp`)
- AVX512: `src/backend/isa/avx512.cpp` (+ declaration in `include/cpptensor/backend/isa/avx512.hpp`)

Use `Tensor::data_ptr()`/`numel()`-style access in kernels where view-safe logical layout handling is required.

---

## 3) Register kernels

Wire registrations in `src/backend/backend_loader.cpp` inside `register_kernels()`:
- Generic CPU registration
- Optional AVX2/AVX512 registration
- Optional CUDA registration if supported

Match registration type to op kind:
- `registerKernel(...)` for binary ops
- `registerUnaryKernel(...)` for unary ops
- `registerReductionKernel(...)` for reductions

---

## 4) Add high-level API

- Header: `include/cpptensor/ops/<category>/<op>.hpp`
- Impl: `src/ops/<category>/<op>.cpp`

Typical responsibilities:
- Validate shapes/devices/dim args
- Handle broadcasting or dim normalization as needed
- Materialize non-contiguous inputs when backend kernels require compact layout
- Dispatch to `KernelRegistry`

If ergonomic operator overloads are needed, update `include/cpptensor/tensor/tensor.hpp` and `src/tensor/tensor.cpp`.

---

## 5) Add or update tests

Prefer adding coverage in `test/test_ops.cpp`, and use targeted test files when appropriate (e.g. `test/test_linear_algebra.cpp`, `test/test_cuda_dispatch.cpp`).

Minimum test set:
- Correctness on representative shapes
- Error boundaries (shape mismatch, invalid dim/device)
- View/non-contiguous input behavior (when relevant)
- ISA dispatch behavior if SIMD path added

---

## 6) Keep docs in sync (required)

When behavior ships, update:
- `docs/Ops_Status.md` (implemented status + coverage note)
- `docs/TensorViews.md` if view/memory semantics changed
- This guide if workflow/repo layout changed

Do not mix roadmap ideas into shipped-status sections.

---

## 7) Validate build/tests in the `cpptensor` conda env

```bash
conda run -n cpptensor cmake -S . -B build
conda run -n cpptensor cmake --build build -j
conda run -n cpptensor ctest --test-dir build --output-on-failure
```

(If you also run benchmarks, run them via `conda run -n cpptensor ...` as well.)
