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
