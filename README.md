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

## CMake build profiles
- `Debug`: target-scoped `-O0 -g3` (or MSVC `/Od /Zi /RTC1`).
- `Release`: target-scoped `-O3 -DNDEBUG` (or MSVC `/O2 /DNDEBUG`).
- `RelWithDebInfo`: target-scoped `-O2 -g -DNDEBUG` (or MSVC `/O2 /Zi /DNDEBUG`).
- Single-config generators default to `RelWithDebInfo` when `CMAKE_BUILD_TYPE` is omitted.

Optional toggles:
- `-DCPPTENSOR_ENABLE_PROFILING=ON`: preserve frame pointers for profiling.
- `-DCPPTENSOR_ENABLE_LTO=ON`: enable IPO/LTO for `Release` and `RelWithDebInfo` when supported.
- `-DCPPTENSOR_ENABLE_GPERFTOOLS=ON`: link examples with `libprofiler` when available.

## install + downstream CMake usage
Install cpptensor to a prefix:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix /tmp/cpptensor-install
```

Consume from another CMake project:

```cmake
find_package(cpptensor CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE cpptensor::cpptensor)
```

## Build safety modes (warnings + sanitizers)

cpptensor now provides opt-in CMake toggles for stricter development/CI builds:

- `CPPTENSOR_ENABLE_STRICT_WARNINGS=ON`: enable an elevated warning profile on cpptensor-owned targets only.
- `CPPTENSOR_WARNINGS_AS_ERRORS=ON`: promote enabled warnings to errors (`-Werror` / `/WX`).
- `CPPTENSOR_ENABLE_ASAN=ON`: AddressSanitizer
- `CPPTENSOR_ENABLE_UBSAN=ON`: UndefinedBehaviorSanitizer
- `CPPTENSOR_ENABLE_TSAN=ON`: ThreadSanitizer

Notes:
- Sanitizers currently support GCC/Clang-style toolchains.
- `CPPTENSOR_ENABLE_ASAN` and `CPPTENSOR_ENABLE_TSAN` are mutually exclusive in a single build.
- Sanitizer mode currently requires `BUILD_CUDA=OFF`.

### Documented sanitizer workflow

Run this from a shell with Conda available:

```bash
conda run -n cpptensor cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_CUDA=OFF \
  -DUSE_OPENBLAS=OFF \
  -DCPPTENSOR_ENABLE_ASAN=ON \
  -DCPPTENSOR_ENABLE_UBSAN=ON

conda run -n cpptensor cmake --build build-sanitize -j
conda run -n cpptensor ctest --test-dir build-sanitize --output-on-failure
```

To turn on warning-gate mode for local/CI hardening, add:
`-DCPPTENSOR_ENABLE_STRICT_WARNINGS=ON -DCPPTENSOR_WARNINGS_AS_ERRORS=ON`

### Warning policy by compiler

- GCC / Clang: `-Wall -Wextra -Wpedantic -Wformat=2 -Wnull-dereference -Wnon-virtual-dtor`
- MSVC: `/W4 /permissive-`

# runtime behavior
Public tensor ops lazily initialize the kernel registry on first use, so a fresh
process can call `A + B`, `sum()`, `matmul()`, and other registered ops without
calling `initialize_kernels()` manually. `initialize_kernels()` remains available
as an optional explicit warm-up step.

# numerical stability policy
`sum()`, `mean()`, and `dot()` prioritize accumulation accuracy over raw
throughput on cancellation-heavy inputs:

- CPU reductions use widened compensated accumulation before casting back to
  `float`.
- AVX runtime dispatch still uses optimized paths for pointwise and matmul-style
  kernels, but `sum`/`mean` route through the stable reduction implementation.
- `dot()` kernels accumulate products in widened precision before casting the
  scalar result back to `float`.

This trades some peak reduction throughput for better numerical robustness.

# dtype support
`Tensor` now tracks element dtype metadata (`bool`, `int32`, `float32`, `float64`).
Comparison operators produce `bool` tensors, and dtype is preserved across views,
clone/contiguous, and factory creation (`zeros`, `ones`, `full`, `randn`).

# checkpoint I/O
Tensor checkpoints are supported via `Tensor::save(path)` and `Tensor::load(path)`.
See `docs/TensorSerialization.md` for the versioned binary format and view behavior.
