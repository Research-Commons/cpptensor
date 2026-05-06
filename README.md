# cpptensor
Tensor library written in c++ 26

https://deepwiki.com/Research-Commons/cpptensor/10.3-license

# build
Remember to clone the submodules - `` git clone --recurse-submodules <cpptensor> ``

## CMake build targets
Tests, examples, and benchmarks are opt-in:

- `CPPTENSOR_BUILD_TESTS` (default: `OFF`)
- `CPPTENSOR_BUILD_EXAMPLES` (default: `OFF`)
- `CPPTENSOR_BUILD_BENCHMARKS` (default: `OFF`)

Minimal library-only configure:

```bash
cmake -S . -B build
```

Development configure with all extras:

```bash
cmake -S . -B build-dev \
  -DCPPTENSOR_BUILD_TESTS=ON \
  -DCPPTENSOR_BUILD_EXAMPLES=ON \
  -DCPPTENSOR_BUILD_BENCHMARKS=ON
```

# runtime behavior
Public tensor ops lazily initialize the kernel registry on first use, so a fresh
process can call `A + B`, `sum()`, `matmul()`, and other registered ops without
calling `initialize_kernels()` manually. `initialize_kernels()` remains available
as an optional explicit warm-up step.

# checkpoint I/O
Tensor checkpoints are supported via `Tensor::save(path)` and `Tensor::load(path)`.
See `docs/TensorSerialization.md` for the versioned binary format and view behavior.
