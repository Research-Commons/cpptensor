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

# ISA build/run guardrails (AVX2 / AVX-512)

- `BUILD_AVX2` / `BUILD_AVX512` control whether ISA-specialized object code is built.
  Configure fails fast if those targets are forced on unsupported architectures.
- Runtime CPU dispatch now validates **CPU feature bits + OS XSAVE state** before
  selecting AVX2/AVX-512 kernels, and safely falls back to lower ISA levels.
- `CPPGRAD_CPU_ISA` overrides are bounded by host capability:
  - `avx512` → AVX-512 only when fully supported; otherwise AVX2/generic fallback.
  - `avx2` → AVX2 only when supported; otherwise generic fallback.
  - other values → generic.

## Benchmark safety and CI expectations

- `cpptensor_bench_avx2` and `cpptensor_bench_avx512` now perform runtime capability
  checks before running and exit with code **77** when unsupported (graceful skip).
- ISA benchmark binaries force `CPPGRAD_CPU_ISA` internally (`avx2`/`avx512`) only
  after passing runtime checks, so they benchmark the intended kernels.
- To register benchmark smoke checks in CTest, configure with:
  - `-DCPPTENSOR_REGISTER_BENCHMARK_TESTS=ON`
  - CTest treats AVX benchmark exit code `77` as **skipped**.
- CI should only run ISA benchmark jobs on compatible hardware, or rely on the
  built-in skip behavior above for mixed/shared runners.
