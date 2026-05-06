# Benchmark baselines

Store machine-specific baseline files here (or in your CI artifact store).

Suggested naming:

- `linux-x86_64-release.json`
- `linux-x86_64-avx2.json`
- `linux-x86_64-avx512.json`

Generate a baseline with:

```bash
conda run -n cpptensor python3 benchmarks/benchmark_harness.py \
  --build-dir cmake-build-release \
  --output-dir benchmark_results \
  --backends cpu,avx2,avx512

cp benchmark_results/latest.json benchmarks/baselines/linux-x86_64-release.json
```

Then compare candidate runs with:

```bash
conda run -n cpptensor python3 benchmarks/compare_benchmark_results.py \
  --baseline benchmarks/baselines/linux-x86_64-release.json \
  --candidate benchmark_results/latest.json \
  --max-regression-pct 5
```
