# cpptensor

`cpptensor` is a C++ tensor library focused on CPU performance (generic + AVX2/AVX-512), with optional OpenBLAS acceleration and early CUDA support.

## What you get

- N-dimensional `Tensor` with views + contiguous materialization.
- Arithmetic, unary math, reductions, comparisons, reshape/transpose/concat/stack.
- Linear algebra ops including `matmul`, `dot`, `tensordot`, `svd`, and `eig`.
- Runtime CPU ISA dispatch with build-time AVX2/AVX-512 specializations.
- Lazy kernel registry initialization on first tensor op.
- Catch2 test suite + Google Benchmark targets.
- Tensor checkpoint save/load (`Tensor::save`, `Tensor::load`).

## Platform and toolchain support

| Area | Status |
|---|---|
| Linux x86_64 | **Supported / primary path** (validated workflow). |
| Linux aarch64 | **Supported for generic CPU path**; AVX2/AVX-512 auto-disable. |
| macOS | CPU builds expected; CUDA is forced `OFF`. |
| Windows | Not part of the documented conda workflow (experimental). |

### Required build tools

- CMake (minimum 3.20).
- A C++ compiler with **C++26** support.
- Ninja or Make.
- Git (for submodules).

## Quickstart (fresh clone)

> All build/test commands below use the dedicated `cpptensor` conda environment.

```bash
git clone --recurse-submodules https://github.com/Research-Commons/cpptensor.git
cd cpptensor
```

### 1) Create or update the conda environment

```bash
# First-time setup:
conda env create -f environment.yml

# Existing environment:
conda env update -n cpptensor -f environment.yml --prune
```

### 2) Configure (tests/examples/benchmarks enabled)

```bash
conda run -n cpptensor cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPPTENSOR_BUILD_TESTS=ON \
  -DCPPTENSOR_BUILD_EXAMPLES=ON \
  -DCPPTENSOR_BUILD_BENCHMARKS=ON
```

### 3) Build

```bash
conda run -n cpptensor cmake --build build -j
```

### 4) Run tests

```bash
conda run -n cpptensor ctest --test-dir build --output-on-failure
```

### 5) Run a smoke test and benchmark

```bash
conda run -n cpptensor ./build/test/cpptensor_lazy_init_smoke
conda run -n cpptensor ./build/benchmarks/cpptensor_bench_cpu --benchmark_min_time=0.01s
```

## CMake build targets

Tests, examples, and benchmarks are opt-in:

- `CPPTENSOR_BUILD_TESTS` (default: `OFF`)
- `CPPTENSOR_BUILD_EXAMPLES` (default: `OFF`)
- `CPPTENSOR_BUILD_BENCHMARKS` (default: `OFF`)

Minimal library-only configure:

```bash
conda run -n cpptensor cmake -S . -B build
```

## Build profiles and developer toggles

- Build profiles: `Debug`, `Release`, `RelWithDebInfo`.
- Single-config generators default to `RelWithDebInfo` if `CMAKE_BUILD_TYPE` is omitted.
- `CPPTENSOR_ENABLE_PROFILING=ON`: profiling-friendly frame-pointer settings.
- `CPPTENSOR_ENABLE_LTO=ON`: interprocedural optimization (LTO) for `Release`/`RelWithDebInfo` when supported.
- `CPPTENSOR_ENABLE_GPERFTOOLS=ON`: link example binaries with `libprofiler` when available.

## Build safety modes (warnings + sanitizers)

- `CPPTENSOR_ENABLE_STRICT_WARNINGS=ON`: stricter warning profile on cpptensor-owned targets.
- `CPPTENSOR_WARNINGS_AS_ERRORS=ON`: promote warnings to errors.
- `CPPTENSOR_ENABLE_ASAN=ON`: AddressSanitizer.
- `CPPTENSOR_ENABLE_UBSAN=ON`: UndefinedBehaviorSanitizer.
- `CPPTENSOR_ENABLE_TSAN=ON`: ThreadSanitizer.

Notes:

- Sanitizers currently support GCC/Clang-style toolchains.
- `CPPTENSOR_ENABLE_ASAN` and `CPPTENSOR_ENABLE_TSAN` are mutually exclusive.
- Sanitizer mode currently requires `BUILD_CUDA=OFF`.

Example sanitizer workflow:

```bash
conda run -n cpptensor cmake -S . -B build-sanitize -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCPPTENSOR_BUILD_TESTS=ON \
  -DBUILD_CUDA=OFF \
  -DUSE_OPENBLAS=OFF \
  -DCPPTENSOR_ENABLE_ASAN=ON \
  -DCPPTENSOR_ENABLE_UBSAN=ON

conda run -n cpptensor cmake --build build-sanitize -j
conda run -n cpptensor ctest --test-dir build-sanitize --output-on-failure
```

## Backend feature flags

Configure with `conda run -n cpptensor cmake ... -D<OPTION>=<VALUE>`.

- `USE_OPENBLAS` (default: `ON`)
  - If OpenBLAS is found, `matmul`/`dot` can use BLAS-backed paths.
  - `svd`/`eig` require OpenBLAS/LAPACK support.
- `BUILD_AVX2` / `BUILD_AVX512` (auto-detected defaults)
  - Enabled only when target + compiler support are detected.
  - Forcing `ON` when unsupported fails at configure time.
- `BUILD_CUDA` (default: `OFF`)
  - Requires CUDA toolkit discoverable by CMake.
  - On Apple platforms this is automatically forced `OFF`.
- `USE_STD_SIMD`
  - Declared in CMake, but not currently wired as an active backend toggle.

## Install + downstream CMake usage

Install cpptensor to a prefix:

```bash
conda run -n cpptensor cmake -S . -B build
conda run -n cpptensor cmake --build build
conda run -n cpptensor cmake --install build --prefix /tmp/cpptensor-install
```

Consume from another CMake project:

```cmake
find_package(cpptensor CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE cpptensor::cpptensor)
```

## Minimal usage example

```cpp
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/ops/arithmetic/add.hpp"

int main() {
    using namespace cpptensor;

    Tensor a = Tensor::full({2, 2}, 1.0f);
    Tensor b = Tensor::full({2, 2}, 2.0f);
    Tensor c = a + b;
    c.print();
}
```

## Runtime behavior

Public tensor ops lazily initialize the kernel registry on first use, so a fresh
process can call `A + B`, `sum()`, `matmul()`, and other registered ops without
calling `initialize_kernels()` manually. `initialize_kernels()` remains available
as an optional explicit warm-up step.


## ISA build/run guardrails (AVX2 / AVX-512)

- `BUILD_AVX2` / `BUILD_AVX512` control whether ISA-specialized object code is built.
  Configure fails fast if those targets are forced on unsupported architectures.
- Runtime CPU dispatch validates **CPU feature bits + OS XSAVE state** before selecting
  AVX2/AVX-512 kernels, and safely falls back to lower ISA levels.
- `CPPGRAD_CPU_ISA` overrides are bounded by host capability:
  - `avx512` → AVX-512 only when fully supported; otherwise AVX2/generic fallback.
  - `avx2` → AVX2 only when supported; otherwise generic fallback.
  - other values → generic.

### Benchmark safety and CI expectations

- `cpptensor_bench_avx2` and `cpptensor_bench_avx512` perform runtime capability
  checks before running and exit with code **77** when unsupported (graceful skip).
- ISA benchmark binaries force `CPPGRAD_CPU_ISA` internally (`avx2`/`avx512`) only
  after passing runtime checks, so they benchmark the intended kernels.
- To register benchmark smoke checks in CTest, configure with:
  - `-DCPPTENSOR_REGISTER_BENCHMARK_TESTS=ON`
  - CTest treats AVX benchmark exit code `77` as **skipped**.
- CI should only run ISA benchmark jobs on compatible hardware, or rely on the
  built-in skip behavior above for mixed/shared runners.

## Numerical stability policy

`sum()`, `mean()`, and `dot()` prioritize accumulation accuracy over raw
throughput on cancellation-heavy inputs:

- CPU reductions use widened compensated accumulation before casting back to `float`.
- AVX runtime dispatch still uses optimized paths for pointwise/matmul-style kernels,
  but `sum`/`mean` route through the stable reduction implementation.
- `dot()` kernels accumulate products in widened precision before casting the scalar
  result back to `float`.

## Dtype support

`Tensor` tracks element dtype metadata (`bool`, `int32`, `float32`, `float64`).
Comparison operators produce `bool` tensors, and dtype is preserved across views,
clone/contiguous, and factory creation (`zeros`, `ones`, `full`, `randn`).

## Checkpoint I/O

Tensor checkpoints are supported via `Tensor::save(path)` and `Tensor::load(path)`.
See `docs/TensorSerialization.md` for the versioned binary format and view behavior.

## More docs

- [Operator status and semantics](docs/Ops_Status.md)
- [Tensor serialization format](docs/TensorSerialization.md)
- [Tensor view behavior](docs/TensorViews.md)
- [How to add new ops/kernels](docs/HowToAddNewOpsWithKernel.md)
- [Linear algebra decomposition notes](docs/LinearAlgebraDecompositionNotes.md)
- [Performance notes and benchmark write-up](docs/PerformanceComparisonOps.md)
