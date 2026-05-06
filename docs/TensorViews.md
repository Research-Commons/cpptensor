# View-Based Tensor Architecture

This document explains the zero-copy view architecture implementation in cpptensor, including the design rationale, memory management, and implementation details.

## Table of Contents

- [Overview](#overview)
- [Tuple-Style Indexing and Slicing](#tuple-style-indexing-and-slicing)
- [cat/stack behavior with views and devices](#catstack-behavior-with-views-and-devices)
- [Core Architecture Changes](#core-architecture-changes)
- [The `base_impl_` Design](#the-base_impl_-design)
- [Tensor Manipulation Functions](#tensor-manipulation-functions)
- [Memory Layout Examples](#memory-layout-examples)
- [Key Design Principles](#key-design-principles)
- [Alternative Designs](#alternative-designs)

---

## Overview

cpptensor implements a **PyTorch-style zero-copy view architecture** for tensor manipulation operations. This enables efficient tensor reshaping, transposition, and dimension reordering without copying data.

**Key Benefits:**
- ✅ **Zero-copy operations**: View, reshape, permute, transpose without data duplication
- ✅ **Automatic memory management**: Safe shared ownership via `shared_ptr`
- ✅ **PyTorch compatibility**: Familiar API and semantics
- ✅ **Contiguity tracking**: Automatic detection and handling of memory layout
- ✅ **Explicit fallback policy**: Kernels that are not stride-aware consume logical compact
  materializations via `contiguous()`/logical flattening rather than raw backing offsets

---

## Tuple-Style Indexing and Slicing

Issue #19 adds a multi-axis indexing API:

```cpp
Tensor out = x.index({
    Tensor::SliceSpec{1, 4},         // axis 0 slice
    -1,                              // axis 1 scalar selection (drops axis)
    Tensor::SliceSpec{0, 8, 2},      // axis 2 stepped slice
});
```

### Supported patterns

- Multiple axes in one call (`index({...})`)
- Scalar indexing (`int64_t`) with negative indices
- Slice indexing (`SliceSpec{start, end, step}`) with negative start/end
- Negative-step slices (`step < 0`)

### Shape behavior

- Scalar axis selection removes that axis from the output shape.
- Slice axes remain, with length computed from `[start:end:step]`.
- Selecting scalar indices on every axis returns a rank-0 tensor (`shape() == {}`).

### View vs copy semantics

- **Zero-copy view path**: all axes are scalar or slice with `step > 0`.
- **Materialized copy path**: any axis uses `step < 0`.

This keeps aliasing behavior predictable:
- positive-step indexing/slicing aliases source storage,
- negative-step indexing/slicing produces an independent compact tensor.

---

## cat/stack behavior with views and devices

For tensor lists passed to `cat` and `stack`:

- All operands must be on the same device (`CPU` or `CUDA` tags). Mixed-device lists fail fast with a clear error.
- Output device is preserved from the validated common input device.
- Non-contiguous inputs (e.g. slices/transposes/views) are supported:
  - `cat` copies logical values using shape/stride traversal.
  - `stack` first preserves logical contents for non-contiguous operands before concatenation.

Regression coverage lives in `test/test_ops.cpp` under:
- `[manipulation][cat][device]`
- `[manipulation][stack][device]`
- `[manipulation][cat][views]`
- `[manipulation][stack][views]`

---

## Core Architecture Changes

### TensorImpl: Added View Support

**Files:** `include/cpptensor/tensor/tensorimpl.hpp`, `src/tensor/tensorimpl.cpp`

#### New Member Variable

```cpp
std::shared_ptr<TensorImpl> base_impl_;
```

**Purpose:**
- When a tensor is a "view", this points to the original tensor that owns the data
- Keeps the base tensor alive as long as any view exists (shared ownership)
- Enables zero-copy operations by sharing data between tensors

#### New View Constructor

```cpp
TensorImpl(std::shared_ptr<TensorImpl> base,
           const std::vector<size_t>& new_shape,
           const std::vector<size_t>& new_stride = {});
```

**Behavior:**
- Creates views that **share data** instead of copying
- Takes a base tensor, new shape, and optionally new strides
- If `new_stride` is empty, computes default row-major strides
- Sets `base_impl_` to keep base alive via shared ownership

#### Logical vs Backing Data Access

`cpptensor` now distinguishes **backing storage** from **logical tensor order**:

- `Tensor::data() const` returns flattened values in logical row-major order.
  - Owning tensors and full contiguous views expose storage directly.
  - Non-contiguous/pointer-backed views materialize a compact logical snapshot.
- `Tensor::data()` (mutable) is intentionally restricted to direct compact backing storage.
  - Sliced/transposed/permuted/pointer-backed views throw and must be materialized with
    `contiguous()` or `clone()` before mutable flat-buffer writes.

This preserves correctness for view-based tensors while keeping zero-copy behavior where safe.

---

## The `base_impl_` Design

### Why `base_impl_` is Needed

The `base_impl_` pointer solves a **critical memory management problem** with shared data in views.

#### ❌ Problem Without `base_impl_`

```cpp
void problematic_example() {
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});  // Create original tensor
    Tensor B = A.view({3, 2});              // Create view
    
    // At this point:
    // - A.impl_ points to TensorImpl with data [1,2,3,4,5,6]
    // - B.impl_ points to NEW TensorImpl that needs access to A's data
    
    // A goes out of scope here...
}  // 💥 A is destroyed, its data_ vector is freed!

// B still exists but its data is now INVALID - use after free!
```

**The Problem:** When A is destroyed, its `data_` vector is freed. If B just had a pointer to A's data, that pointer would be dangling.

#### ✅ Solution WITH `base_impl_`

```cpp
void safe_example() {
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = A.view({3, 2});
    
    // At this point:
    // - A.impl_ = shared_ptr<TensorImpl> (owns data)
    // - B.impl_ = shared_ptr<TensorImpl> (view)
    // - B.impl_->base_impl_ = A.impl_ (shared_ptr keeping A alive!)
    
    // A goes out of scope...
}  // ✅ A.impl_ not destroyed because B.impl_->base_impl_ still holds a reference!

// B is still valid! The original data is kept alive by shared_ptr
```

---

### Memory Ownership Diagrams

#### Scenario 1: Normal Tensor (No Views)

```cpp
Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
```

```
Stack:                          Heap:
┌─────────┐                    ┌──────────────────────┐
│ Tensor A│──────shared_ptr───▶│ TensorImpl           │
└─────────┘                    │  ├─ data_: [1,2,3,..] │ ◀── Owns the data
                               │  ├─ base_impl_: null  │ ◀── No base (original)
                               │  ├─ shape_: [2, 3]    │
                               │  └─ stride_: [3, 1]   │
                               └──────────────────────┘
                               Reference count = 1
```

**What happens when A goes out of scope?**
- `A.impl_` is destroyed (shared_ptr)
- Reference count drops to 0
- TensorImpl is deleted
- `data_` vector is freed ✅

---

#### Scenario 2: Tensor With View

```cpp
Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor B = A.view({3, 2});
```

```
Stack:                          Heap:
┌─────────┐                    ┌──────────────────────┐
│ Tensor A│──────shared_ptr───▶│ TensorImpl (Base)    │
└─────────┘        │           │  ├─ data_: [1,2,3,..] │ ◀── Owns the data
                   │           │  ├─ base_impl_: null  │
                   │           │  ├─ shape_: [2, 3]    │
┌─────────┐        │           │  └─ stride_: [3, 1]   │
│ Tensor B│──┐     │           └──────────────────────┘
└─────────┘  │     │           Reference count = 2 ◀──┐
             │     │                                   │
       shared_ptr  │           ┌──────────────────────┤
             │     │           │ TensorImpl (View)    │
             └─────┼──────────▶│  ├─ data_: EMPTY     │
                   │           │  ├─ base_impl_────────┘ ◀── Points to base!
                   │           │  ├─ shape_: [3, 2]    │
                   └───────────│  └─ stride_: [2, 1]   │
                               └──────────────────────┘
```

**Key Points:**
1. **Base TensorImpl has reference count = 2** (A and B's view both point to it)
2. **View's `base_impl_` keeps base alive** via `shared_ptr`
3. **View's `data_` is EMPTY** - logical reads use shape/stride/offset over shared backing storage

---

#### Scenario 3: Original Tensor Dies, View Lives

```cpp
Tensor B;
{
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    B = A.view({3, 2});
    // A goes out of scope here...
}
// B is still alive and valid!
```

```
After A is destroyed:

Stack:                          Heap:
┌─────────┐                    ┌──────────────────────┐
│ Tensor B│──┐                 │ TensorImpl (Base)    │
└─────────┘  │                 │  ├─ data_: [1,2,3,..] │ ◀── Still alive!
             │                 │  ├─ base_impl_: null  │
       shared_ptr               │  ├─ shape_: [2, 3]    │
             │                 │  └─ stride_: [3, 1]   │
             │                 └──────────────────────┘
             │                 Reference count = 1 ◀──┐
             │                                         │
             │                 ┌──────────────────────┤
             └────────────────▶│ TensorImpl (View)    │
                               │  ├─ data_: EMPTY     │
                               │  ├─ base_impl_────────┘
                               │  ├─ shape_: [3, 2]    │
                               │  └─ stride_: [2, 1]   │
                               └──────────────────────┘
```

**What happened?**
1. A's `impl_` shared_ptr was destroyed
2. But reference count only dropped from 2 → 1 (B's view still holds reference via `base_impl_`)
3. **Data stays alive!** ✅
4. B can still access the same backing storage through `base_impl_`

---

### Data Access Flow

For a view tensor, logical indexing is resolved using:

1. the shared backing storage pointer (`base_impl_` / `data_ptr_`)
2. the view's `offset_`
3. the view's `stride_`

For `const data()`:
- full compact layouts can expose direct storage
- non-contiguous layouts materialize a compact logical snapshot

For mutable `data()`:
- only direct compact backing storage is allowed
- non-trivial views throw and require `contiguous()`/`clone()` first

This is what keeps operations on slices/transposes/from_ptr views correct even when the
physical backing layout does not match logical tensor order.

---

### What Happens to Original Tensor If No Views?

#### Case 1: No Views Created

```cpp
{
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    // Do some operations...
    float sum = A.data()[0] + A.data()[1];
}  // A goes out of scope
```

**Destruction Sequence:**
1. `A` destructor called
2. `A.impl_` (shared_ptr) destructor called
3. Reference count = 0 (only A had reference)
4. `TensorImpl` destructor called
5. `data_` vector destroyed
6. Memory freed ✅

**`base_impl_` is null**, so it doesn't affect anything.

---

#### Case 2: Views Were Created But Also Destroyed

```cpp
{
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    {
        Tensor B = A.view({3, 2});
        // Use B...
    }  // B destroyed here (reference count: 2 → 1)
    // A still alive
}  // A destroyed here (reference count: 1 → 0)
```

**Timeline:**
- Initially: A.impl_ reference count = 1
- After creating B: A.impl_ reference count = 2 (A and B's `base_impl_`)
- B destroyed: A.impl_ reference count = 2 → 1
- A destroyed: A.impl_ reference count = 1 → 0 → memory freed ✅

---

## Tensor Manipulation Functions

### 1. `view(new_shape)` - Zero-Copy Reshape

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    // 1. Validate total elements match
    // 2. Check tensor is contiguous (REQUIRED for view)
    // 3. Create view TensorImpl sharing data with impl_
    auto view_impl = std::make_shared<TensorImpl>(impl_, new_shape);
    return Tensor(view_impl);
}
```

**Characteristics:**
- **Zero-copy**: No data duplication
- **Requirement**: Tensor must be contiguous (throws error if not)
- **Shares data**: Modifying view modifies original

**Example:**
```cpp
Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor B = A.view({3, 2});  // Shape: {3, 2}, shares data with A
B.data()[0] = 99;            // A.data()[0] is now also 99!
```

---

### 2. `reshape(new_shape)` - Smart Reshape

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (is_contiguous()) {
        return view(new_shape);  // Zero-copy path
    } else {
        return contiguous().view(new_shape);  // Copy then view
    }
}
```

**Smart Behavior:**
- **Contiguous tensor** → Uses `view()` (zero-copy)
- **Non-contiguous tensor** → Makes contiguous copy first, then views
- **User-friendly**: Works in all cases automatically

**Example:**
```cpp
Tensor A({2, 3}, ...);
Tensor B = A.reshape({6});     // Zero-copy if A is contiguous

Tensor C = A.transpose();       // C is non-contiguous
Tensor D = C.reshape({6});      // Makes a copy first
```

---

### 3. `flatten(start_dim, end_dim)` - Dimension Flattening

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::flatten(int start_dim, int end_dim) const {
    // 1. Normalize negative indices
    // 2. Validate dimension range
    // 3. Build new shape:
    //    - Keep dims before start_dim
    //    - Flatten dims from start_dim to end_dim into one
    //    - Keep dims after end_dim
    return reshape(new_shape);
}
```

**Examples:**
```cpp
Tensor A({2, 3, 4}, ...);

Tensor B = A.flatten();         // Shape: {24} (flatten all)
Tensor C = A.flatten(1, 2);     // Shape: {2, 12} (flatten dims 1-2)
Tensor D = A.flatten(0, 1);     // Shape: {6, 4} (flatten dims 0-1)
```

---

### 4. `squeeze(dim)` - Remove Size-1 Dimensions

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::squeeze(int dim) const {
    if (dim == -1) {
        // Remove ALL dimensions of size 1
    } else {
        // Remove specific dimension (must be size 1)
    }
    return reshape(new_shape);
}
```

**Examples:**
```cpp
Tensor A({2, 1, 3, 1, 4}, ...);

Tensor B = A.squeeze();         // Shape: {2, 3, 4} (remove all size-1)
Tensor C = A.squeeze(1);        // Shape: {2, 3, 1, 4} (remove dim 1)
Tensor D = A.squeeze(3);        // Shape: {2, 1, 3, 4} (remove dim 3)
```

---

### 5. `unsqueeze(dim)` - Add Size-1 Dimension

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::unsqueeze(int dim) const {
    // Normalize dimension (allow negative indexing)
    // Insert dimension of size 1 at specified position
    return reshape(new_shape);
}
```

**Examples:**
```cpp
Tensor A({2, 3}, ...);

Tensor B = A.unsqueeze(0);      // Shape: {1, 2, 3}
Tensor C = A.unsqueeze(1);      // Shape: {2, 1, 3}
Tensor D = A.unsqueeze(-1);     // Shape: {2, 3, 1}
```

---

### 6. `permute(dims)` - Arbitrary Dimension Reordering

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::permute(const std::vector<int>& dims) const {
    // 1. Validate: dims must be valid permutation of [0..ndim-1]
    // 2. Reorder shape and stride according to permutation
    std::vector<size_t> new_shape(ndims);
    std::vector<size_t> new_stride(ndims);
    for (int i = 0; i < ndims; ++i) {
        int d = dims[i];
        new_shape[i] = old_shape[d];
        new_stride[i] = old_stride[d];  // Key: reorder strides!
    }
    // 3. Create view with modified shape AND stride
    auto view_impl = std::make_shared<TensorImpl>(impl_, new_shape, new_stride);
    return Tensor(view_impl);
}
```

**Zero-Copy Magic:**
- **Doesn't copy data** - only changes strides
- Result is **non-contiguous** (strides no longer row-major)
- Can be made contiguous with `.contiguous()`

**Examples:**
```cpp
Tensor A({2, 3, 4}, ...);

Tensor B = A.permute({2, 0, 1});  // Shape: {4, 2, 3}, non-contiguous
Tensor C = A.permute({1, 0, 2});  // Swap first two dims
Tensor D = A.permute({2, 1, 0});  // Reverse dimension order
```

---

### 7. `transpose(dim0, dim1)` - Swap Two Dimensions

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::transpose(int dim0, int dim1) const {
    // Default: swap last two dimensions (classic 2D transpose)
    // Create permutation that swaps dim0 and dim1
    std::vector<int> perm(ndims);
    for (int i = 0; i < ndims; ++i) perm[i] = i;
    std::swap(perm[dim0], perm[dim1]);
    return permute(perm);  // Reuse permute implementation
}
```

**Convenience wrapper around permute:**

**Examples:**
```cpp
Tensor A({3, 4}, ...);
Tensor B = A.transpose();         // Shape: {4, 3} (swap last two dims)

Tensor C({2, 3, 4}, ...);
Tensor D = C.transpose(0, 2);     // Shape: {4, 3, 2} (swap dims 0 and 2)
Tensor E = C.transpose(0, 1);     // Shape: {3, 2, 4} (swap dims 0 and 1)
```

---

### 8. `is_contiguous()` - Check Memory Layout

**Location:** `src/tensor/tensor.cpp`

```cpp
bool Tensor::is_contiguous() const {
    // Check if strides match expected row-major layout
    size_t expected_stride = 1;
    for (int i = sh.size() - 1; i >= 0; --i) {
        if (st[i] != expected_stride) return false;
        expected_stride *= sh[i];
    }
    return true;
}
```

**Row-Major Check:**
- For shape `{2, 3, 4}`, expected strides are `{12, 4, 1}`
- After `permute()` or `transpose()`, strides change → non-contiguous

**Examples:**
```cpp
Tensor A({2, 3}, ...);
A.is_contiguous();                // true (newly created)

Tensor B = A.transpose();
B.is_contiguous();                // false (strides changed)

Tensor C = B.contiguous();
C.is_contiguous();                // true (made contiguous)
```

---

### 9. `contiguous()` - Fix Memory Layout

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;  // Already contiguous, no copy needed
    }
    
    // Copy data in row-major order
    // Iterate through logical indices
    // Read from non-contiguous layout using strides
    // Write to contiguous layout
    return Tensor(sh, new_data, device_type());
}
```

**Smart Copying:**
- If already contiguous → returns self (no copy)
- If non-contiguous → creates new tensor with contiguous (row-major) layout
- Uses the view's offset-aware data pointer, so sliced and raw-pointer-backed views
  materialize their logical values correctly

**Examples:**
```cpp
Tensor A({2, 3}, ...);
Tensor B = A.transpose();         // Non-contiguous

Tensor C = B.contiguous();        // New contiguous tensor
C.is_contiguous();                // true
```

---

### 10. `clone()` - Deep Copy

**Location:** `src/tensor/tensor.cpp`

```cpp
Tensor Tensor::clone() const {
    // Deep copy - create new data buffer
    return Tensor(shape(), impl_->data(), device_type());
}
```

**Simple Deep Copy:**
- Always creates independent copy
- Views and clones are completely independent

**Examples:**
```cpp
Tensor A({2, 3}, ...);
Tensor B = A.view({3, 2});        // View shares data
Tensor C = A.clone();             // Clone has independent data

A.data()[0] = 99;
// B.data()[0] == 99  (shares data)
// C.data()[0] != 99  (independent)
```

---

## Memory Layout Examples

### Original Tensor

```cpp
Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
```

```
Memory: [1, 2, 3, 4, 5, 6]
Shape:  [2, 3]
Strides: [3, 1]  (contiguous, row-major)

Logical view:
[[1, 2, 3],
 [4, 5, 6]]

Access: A[i,j] = memory[i*3 + j*1]
```

---

### View (Zero-Copy)

```cpp
Tensor B = A.view({3, 2});
```

```
Memory: [1, 2, 3, 4, 5, 6]  (SAME as A)
Shape:  [3, 2]
Strides: [2, 1]

Logical view:
[[1, 2],
 [3, 4],
 [5, 6]]

Access: B[i,j] = memory[i*2 + j*1]
Example: B[1,0] = memory[1*2 + 0*1] = memory[2] = 3
```

**Key Point:** Same memory, different interpretation via strides!

---

### Permute (Zero-Copy, Non-Contiguous)

```cpp
Tensor C = A.transpose();  // Shape: {3, 2}
```

```
Memory: [1, 2, 3, 4, 5, 6]  (SAME as A)
Shape:  [3, 2]
Strides: [1, 3]  (non-contiguous!)

Logical view:
[[1, 4],
 [2, 5],
 [3, 6]]

Access: C[i,j] = memory[i*1 + j*3]
Examples:
  C[0,0] = memory[0*1 + 0*3] = memory[0] = 1
  C[0,1] = memory[0*1 + 1*3] = memory[3] = 4
  C[1,0] = memory[1*1 + 0*3] = memory[1] = 2
  C[1,1] = memory[1*1 + 1*3] = memory[4] = 5
```

**Key Point:** Non-contiguous strides allow column-major access to row-major data!

---

### Contiguous Copy

```cpp
Tensor D = C.contiguous();  // Shape: {3, 2}
```

```
Memory: [1, 4, 2, 5, 3, 6]  (NEW buffer, reordered)
Shape:  [3, 2]
Strides: [2, 1]  (contiguous again)

Logical view:
[[1, 4],
 [2, 5],
 [3, 6]]

Access: D[i,j] = memory[i*2 + j*1]
Example: D[1,0] = memory[1*2 + 0*1] = memory[2] = 2
```

**Key Point:** Data physically reordered to match logical view!

---

## Key Design Principles

### 1. Zero-Copy When Possible

Operations like `view()`, `permute()`, and `transpose()` don't copy data:
```cpp
Tensor A({1000, 1000}, ...);  // 1M elements
Tensor B = A.transpose();      // Instant! No copy
Tensor C = B.permute({1, 0});  // Instant! No copy
```

Achieved via shared ownership (`shared_ptr<TensorImpl>`) and stride manipulation.

---

### 2. Contiguity Tracking

Operations that require contiguous memory:
- Automatically detect contiguity with `is_contiguous()`
- Call `contiguous()` when needed
- Throw errors if unsafe (e.g., `view()` on non-contiguous tensor)

```cpp
Tensor A({2, 3}, ...);
Tensor B = A.transpose();      // Non-contiguous
// B.view({6});                // Would throw error
Tensor C = B.contiguous();     // Make contiguous
Tensor D = C.view({6});        // Now safe
```

---

### 3. Smart Operations

Operations automatically choose the most efficient path:

**reshape():**
```cpp
if (is_contiguous()) {
    return view(new_shape);    // Zero-copy
} else {
    return contiguous().view(new_shape);  // Copy first
}
```

**contiguous():**
```cpp
if (is_contiguous()) {
    return *this;              // No-op
} else {
    return copy_and_reorder(); // Make contiguous
}
```

---

### 4. PyTorch Compatibility

API matches PyTorch for easy migration:

| Operation | cpptensor | PyTorch |
|-----------|-----------|---------|
| Zero-copy reshape | `view()` | `view()` |
| Smart reshape | `reshape()` | `reshape()` |
| Transpose | `transpose()` | `transpose()` |
| Permute | `permute()` | `permute()` |
| Check layout | `is_contiguous()` | `is_contiguous()` |
| Fix layout | `contiguous()` | `contiguous()` |
| Deep copy | `clone()` | `clone()` |

---

## Alternative Designs

### Why Not Raw Pointers?

```cpp
TensorImpl* base_impl_;  // ❌ Bad!
```

**Problem:** No ownership management → use-after-free when base is destroyed

**Example:**
```cpp
Tensor view;
{
    Tensor base({2, 3}, ...);
    view = base.view({3, 2});
}  // base destroyed, view.base_impl_ is dangling pointer!
view.data();  // 💥 Use-after-free
```

---

### Why Not Copy Data?

```cpp
data_ = base->data_;  // ❌ Bad!
```

**Problem:** Not zero-copy → defeats the purpose of views

**Performance:**
```cpp
Tensor A({1000, 1000}, ...);  // 1M elements = 4MB

// With views (our implementation):
Tensor B = A.transpose();      // 0 bytes copied, instant

// Without views (copying):
Tensor B = A.transpose();      // 4MB copied, slow
```

---

### Why Not Always Point to Root?

```cpp
// Skip intermediate views, point directly to root
base_impl_ = find_root(base);  // 🤔 Could work, but...
```

**Problems:**
- More complex logic
- Breaks encapsulation (view needs to know about entire chain)
- Our current solution is simpler and equally correct

**Current Solution:**
```cpp
// Simple: each view points to its immediate parent
base_impl_ = base;  // ✅ Simple and works
```

The chain naturally follows through: `view→parent→grandparent→...→root`

---

## Summary

### Key Insights

1. **`base_impl_` enables safe view semantics**
    - Keeps base data alive via `shared_ptr`
    - No manual reference counting needed
    - Automatic cleanup when all views destroyed

2. **Zero-copy operations via stride manipulation**
    - Permute/transpose just reorder strides
    - No data duplication
    - Orders of magnitude faster for large tensors

3. **Smart automatic behavior**
    - Operations choose optimal path (view vs. copy)
    - Contiguity tracking prevents errors
    - PyTorch-compatible API

4. **Minimal overhead**
    - Original tensors: no extra cost (`base_impl_` is null)
    - Views: one extra `shared_ptr` (8 bytes on 64-bit systems)

---

### The Magic One-Liner

```cpp
std::shared_ptr<TensorImpl> base_impl_;  // This enables:
// ✅ Safe view semantics
// ✅ Zero-copy operations
// ✅ Automatic memory management
// ✅ No manual reference counting
```

An elegant solution leveraging C++'s `shared_ptr` to achieve PyTorch-like view behavior with automatic memory management!

---

## References

- **Implementation files:**
    - `include/cpptensor/tensor/tensor.hpp`
    - `include/cpptensor/tensor/tensorimpl.hpp`
    - `src/tensor/tensor.cpp`
    - `src/tensor/tensorimpl.cpp`

- **Tests:**
    - `test/test_tensor_manipulation.cpp`

- **Examples:**
    - `examples/tensor_example.cpp`

- **Documentation:**
    - `TENSOR_MANIPULATION_IMPLEMENTATION.md`
