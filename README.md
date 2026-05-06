# cpptensor
Tensor library written in c++ 26

https://deepwiki.com/Research-Commons/cpptensor/10.3-license

# build
Remember to clone the submodules - `` git clone --recurse-submodules <cpptensor> ``

# runtime behavior
Public tensor ops lazily initialize the kernel registry on first use, so a fresh
process can call `A + B`, `sum()`, `matmul()`, and other registered ops without
calling `initialize_kernels()` manually. `initialize_kernels()` remains available
as an optional explicit warm-up step.

# dtype support
`Tensor` now tracks element dtype metadata (`bool`, `int32`, `float32`, `float64`).
Comparison operators produce `bool` tensors, and dtype is preserved across views,
clone/contiguous, and factory creation (`zeros`, `ones`, `full`, `randn`).

# checkpoint I/O
Tensor checkpoints are supported via `Tensor::save(path)` and `Tensor::load(path)`.
See `docs/TensorSerialization.md` for the versioned binary format and view behavior.
