#!/bin/bash
set -euo pipefail

# Structured benchmark runner for cpptensor.
#
# Optional env vars:
#   BUILD_DIR                (default: cmake-build-release)
#   OUTPUT_DIR               (default: benchmark_results)
#   BENCH_BACKENDS           (default: cpu,avx2,avx512,cuda)
#   BENCHMARK_FILTER         (default: unset)
#   BENCHMARK_MIN_TIME       (default: unset)
#   BASELINE_JSON            (default: unset; enables regression check)
#   MAX_REGRESSION_PCT       (default: 5.0)
#   THRESHOLDS_FILE          (default: unset)
#   FAIL_ON_MISSING_BINARY   (default: 0)
#   FAIL_ON_MISSING_ROWS     (default: 0)

if [[ "${BASH_SOURCE[0]}" == */* ]]; then
  ROOT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" >/dev/null 2>&1 && pwd)"
else
  ROOT_DIR="$PWD"
fi
BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"
BENCH_BACKENDS="${BENCH_BACKENDS:-cpu,avx2,avx512,cuda}"
MAX_REGRESSION_PCT="${MAX_REGRESSION_PCT:-5.0}"

HARNESS_SCRIPT="${ROOT_DIR}/benchmarks/benchmark_harness.py"
COMPARE_SCRIPT="${ROOT_DIR}/benchmarks/compare_benchmark_results.py"

HARNESS_CMD=(
  "${HARNESS_SCRIPT}"
  --build-dir "${BUILD_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --backends "${BENCH_BACKENDS}"
)

if [[ -n "${BENCHMARK_FILTER:-}" ]]; then
  HARNESS_CMD+=(--benchmark-filter "${BENCHMARK_FILTER}")
fi

if [[ -n "${BENCHMARK_MIN_TIME:-}" ]]; then
  HARNESS_CMD+=(--benchmark-min-time "${BENCHMARK_MIN_TIME}")
fi

if [[ "${FAIL_ON_MISSING_BINARY:-0}" == "1" ]]; then
  HARNESS_CMD+=(--fail-on-missing-binary)
fi

for extra_arg in "$@"; do
  HARNESS_CMD+=(--extra-benchmark-arg "${extra_arg}")
done

echo "[cpptensor-bench] Collecting structured benchmark results..."
python3 "${HARNESS_CMD[@]}"

LATEST_JSON="${OUTPUT_DIR}/latest.json"
echo "[cpptensor-bench] Latest consolidated JSON: ${LATEST_JSON}"

if [[ -n "${BASELINE_JSON:-}" ]]; then
  echo "[cpptensor-bench] Comparing against baseline: ${BASELINE_JSON}"
  COMPARE_CMD=(
    "${COMPARE_SCRIPT}"
    --baseline "${BASELINE_JSON}"
    --candidate "${LATEST_JSON}"
    --max-regression-pct "${MAX_REGRESSION_PCT}"
    --output-json "${OUTPUT_DIR}/latest.regression-report.json"
  )

  if [[ -n "${THRESHOLDS_FILE:-}" ]]; then
    COMPARE_CMD+=(--thresholds "${THRESHOLDS_FILE}")
  fi

  if [[ "${FAIL_ON_MISSING_ROWS:-0}" == "1" ]]; then
    COMPARE_CMD+=(--fail-on-missing)
  fi

  python3 "${COMPARE_CMD[@]}"
  echo "[cpptensor-bench] Regression report: ${OUTPUT_DIR}/latest.regression-report.json"
fi

echo "[cpptensor-bench] Done."
