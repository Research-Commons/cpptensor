# cpptensor
Tensor library written in c++ 26

https://deepwiki.com/Research-Commons/cpptensor/10.3-license

## Clone
Remember to clone the submodules:

```bash
git clone --recurse-submodules <cpptensor>
```

## Canonical developer build/test workflow
All local configure/build/test commands should run in the `cpptensor` conda environment.

```bash
conda env update -n cpptensor -f environment.yml --prune
conda run -n cpptensor cmake --workflow --preset dev
```

That single workflow preset performs configure + build + test in `build/dev`.

### Other common presets
- Test profile: `conda run -n cpptensor cmake --workflow --preset ci`
- Release build: `conda run -n cpptensor cmake --preset release && conda run -n cpptensor cmake --build --preset release`
- Sanitizers: `conda run -n cpptensor cmake --preset sanitizer && conda run -n cpptensor cmake --build --preset sanitizer && conda run -n cpptensor ctest --preset sanitizer`
- Benchmarks (CPU target build): `conda run -n cpptensor cmake --preset benchmark && conda run -n cpptensor cmake --build --preset benchmark`

## Runtime behavior
Public tensor ops lazily initialize the kernel registry on first use, so a fresh
process can call `A + B`, `sum()`, `matmul()`, and other registered ops without
calling `initialize_kernels()` manually. `initialize_kernels()` remains available
as an optional explicit warm-up step.

## Checkpoint I/O
Tensor checkpoints are supported via `Tensor::save(path)` and `Tensor::load(path)`.
See `docs/TensorSerialization.md` for the versioned binary format and view behavior.
