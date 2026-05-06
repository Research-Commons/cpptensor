#!/usr/bin/env python3
"""Run cpptensor benchmark binaries and emit structured result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_BINARIES = {
    "cpu": "cpptensor_bench_cpu",
    "avx2": "cpptensor_bench_avx2",
    "avx512": "cpptensor_bench_avx512",
    "cuda": "cpptensor_bench_cuda",
}


@dataclass
class BackendRun:
    backend: str
    binary_path: str
    skipped: bool
    skip_reason: str | None
    command: list[str]
    return_code: int | None
    stdout_path: str | None
    stderr_path: str | None
    benchmark_json_path: str | None
    parsed_result: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        default="cmake-build-release",
        help="CMake build directory that contains benchmark binaries.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory where consolidated benchmark artifacts are written.",
    )
    parser.add_argument(
        "--backends",
        default=",".join(BENCH_BINARIES.keys()),
        help="Comma-separated backend list (cpu,avx2,avx512,cuda).",
    )
    parser.add_argument(
        "--benchmark-filter",
        default=None,
        help="Optional Google Benchmark regex filter.",
    )
    parser.add_argument(
        "--benchmark-min-time",
        type=float,
        default=None,
        help="Optional Google Benchmark minimum runtime in seconds.",
    )
    parser.add_argument(
        "--extra-benchmark-arg",
        action="append",
        default=[],
        help="Extra raw argument passed to each benchmark binary. Repeatable.",
    )
    parser.add_argument(
        "--fail-on-missing-binary",
        action="store_true",
        help="Fail instead of skipping unavailable backend binaries.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def get_git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = run_command(["git", *args], cwd=repo_root)
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def get_cpu_metadata() -> dict[str, Any]:
    cpuinfo_path = Path("/proc/cpuinfo")
    model_name = None
    flags: list[str] = []

    if cpuinfo_path.exists():
        text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
        blocks = text.split("\n\n")
        first = blocks[0] if blocks else ""

        model_match = re.search(r"^model name\s*:\s*(.+)$", first, flags=re.MULTILINE)
        flags_match = re.search(r"^flags\s*:\s*(.+)$", first, flags=re.MULTILINE)
        if model_match:
            model_name = model_match.group(1).strip()
        if flags_match:
            flags = [flag.strip() for flag in flags_match.group(1).split() if flag.strip()]

    return {
        "model_name": model_name,
        "flags": flags,
        "logical_cores": os.cpu_count(),
    }


def get_cuda_metadata() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run_command(query)
    except FileNotFoundError:
        return {
            "available": False,
            "query_error": "nvidia-smi executable not found",
            "gpus": [],
        }
    if result.returncode != 0:
        return {
            "available": False,
            "query_error": result.stderr.strip() or result.stdout.strip() or "nvidia-smi unavailable",
            "gpus": [],
        }

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": int(parts[2]) if parts[2].isdigit() else parts[2],
                }
            )

    return {"available": bool(gpus), "gpus": gpus}


def parse_cmake_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}

    desired_keys = {
        "BUILD_CUDA",
        "BUILD_AVX2",
        "BUILD_AVX512",
        "CPPTENSOR_ENABLE_AVX2",
        "CPPTENSOR_ENABLE_AVX512",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "USE_OPENBLAS",
    }

    results: dict[str, Any] = {}
    for raw_line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        if "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _type = key_type.split(":", 1)
        if key in desired_keys:
            results[key] = value

    return results


def build_environment_metadata() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "platform": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": get_cpu_metadata(),
        "cuda": get_cuda_metadata(),
    }


def make_benchmark_command(
    binary_path: Path,
    output_json_path: Path,
    backend: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(binary_path),
        f"--benchmark_out={output_json_path}",
        "--benchmark_out_format=json",
        f"--benchmark_context=cpptensor_backend={backend},cpptensor_binary={binary_path.name}",
    ]

    if args.benchmark_filter:
        command.append(f"--benchmark_filter={args.benchmark_filter}")

    if args.benchmark_min_time is not None:
        command.append(f"--benchmark_min_time={args.benchmark_min_time}")

    command.extend(args.extra_benchmark_arg)
    return command


def write_csv(summary: dict[str, Any], csv_path: Path) -> None:
    fieldnames = [
        "backend",
        "benchmark_name",
        "run_name",
        "run_type",
        "aggregate_name",
        "iterations",
        "real_time",
        "cpu_time",
        "time_unit",
        "bytes_per_second",
        "items_per_second",
        "label",
        "error_occurred",
        "error_message",
    ]

    rows: list[dict[str, Any]] = []
    for suite in summary["suites"]:
        backend = suite["backend"]
        parsed = suite.get("result")
        if not parsed:
            continue
        for bench in parsed.get("benchmarks", []):
            rows.append(
                {
                    "backend": backend,
                    "benchmark_name": bench.get("name"),
                    "run_name": bench.get("run_name", ""),
                    "run_type": bench.get("run_type", ""),
                    "aggregate_name": bench.get("aggregate_name", ""),
                    "iterations": bench.get("iterations", ""),
                    "real_time": bench.get("real_time", ""),
                    "cpu_time": bench.get("cpu_time", ""),
                    "time_unit": bench.get("time_unit", ""),
                    "bytes_per_second": bench.get("bytes_per_second", ""),
                    "items_per_second": bench.get("items_per_second", ""),
                    "label": bench.get("label", ""),
                    "error_occurred": bench.get("error_occurred", False),
                    "error_message": bench.get("error_message", ""),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    build_dir = Path(args.build_dir).resolve()
    benchmark_dir = build_dir / "benchmarks"

    backends = [item.strip().lower() for item in args.backends.split(",") if item.strip()]
    unknown = sorted(set(backends) - set(BENCH_BINARIES))
    if unknown:
        print(f"Unknown backends requested: {', '.join(unknown)}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_runs: list[BackendRun] = []
    failed_backends: list[str] = []

    for backend in backends:
        binary_path = benchmark_dir / BENCH_BINARIES[backend]
        if not binary_path.exists():
            reason = f"missing binary: {binary_path}"
            suite_runs.append(
                BackendRun(
                    backend=backend,
                    binary_path=str(binary_path),
                    skipped=True,
                    skip_reason=reason,
                    command=[],
                    return_code=None,
                    stdout_path=None,
                    stderr_path=None,
                    benchmark_json_path=None,
                    parsed_result=None,
                )
            )
            if args.fail_on_missing_binary:
                failed_backends.append(backend)
            continue

        raw_json = output_dir / f"{backend}.google-benchmark.json"
        stdout_path = output_dir / f"{backend}.stdout.log"
        stderr_path = output_dir / f"{backend}.stderr.log"
        command = make_benchmark_command(binary_path, raw_json, backend, args)

        print(f"[cpptensor-bench] Running {backend}: {' '.join(shlex.quote(arg) for arg in command)}")
        result = run_command(command, cwd=benchmark_dir)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")

        parsed: dict[str, Any] | None = None
        if result.returncode == 0:
            try:
                parsed = json.loads(raw_json.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                failed_backends.append(backend)
                parsed = {
                    "parse_error": str(exc),
                }
        else:
            failed_backends.append(backend)

        suite_runs.append(
            BackendRun(
                backend=backend,
                binary_path=str(binary_path),
                skipped=False,
                skip_reason=None,
                command=command,
                return_code=result.returncode,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                benchmark_json_path=str(raw_json),
                parsed_result=parsed,
            )
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(output_dir),
        "build": {
            "build_dir": str(build_dir),
            "benchmark_dir": str(benchmark_dir),
            "cmake_cache": parse_cmake_cache(build_dir / "CMakeCache.txt"),
        },
        "source_control": get_git_metadata(repo_root),
        "environment": build_environment_metadata(),
        "run_config": {
            "backends": backends,
            "benchmark_filter": args.benchmark_filter,
            "benchmark_min_time": args.benchmark_min_time,
            "extra_benchmark_arg": args.extra_benchmark_arg,
            "fail_on_missing_binary": args.fail_on_missing_binary,
        },
        "suites": [
            {
                "backend": item.backend,
                "binary_path": item.binary_path,
                "skipped": item.skipped,
                "skip_reason": item.skip_reason,
                "command": item.command,
                "return_code": item.return_code,
                "stdout_path": item.stdout_path,
                "stderr_path": item.stderr_path,
                "benchmark_json_path": item.benchmark_json_path,
                "result": item.parsed_result,
            }
            for item in suite_runs
        ],
    }

    total_benchmarks = 0
    for suite in summary["suites"]:
        result = suite.get("result")
        if result and isinstance(result, dict):
            total_benchmarks += len(result.get("benchmarks", []))

    summary["summary"] = {
        "suite_count": len(summary["suites"]),
        "failed_backends": failed_backends,
        "total_benchmark_rows": total_benchmarks,
    }

    consolidated_json = output_dir / "benchmark_results.json"
    consolidated_csv = output_dir / "benchmark_results.csv"

    consolidated_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(summary, consolidated_csv)

    latest_json = Path(args.output_dir).resolve() / "latest.json"
    latest_csv = Path(args.output_dir).resolve() / "latest.csv"
    latest_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    latest_csv.write_text(consolidated_csv.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[cpptensor-bench] Wrote JSON summary: {consolidated_json}")
    print(f"[cpptensor-bench] Wrote CSV summary:  {consolidated_csv}")
    print(f"[cpptensor-bench] Updated pointers:   {latest_json}, {latest_csv}")

    if failed_backends:
        print(
            "[cpptensor-bench] Backends with failures: " + ", ".join(sorted(set(failed_backends))),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
