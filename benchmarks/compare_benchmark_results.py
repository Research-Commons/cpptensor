#!/usr/bin/env python3
"""Compare cpptensor benchmark runs against a baseline and detect regressions."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComparisonRow:
    benchmark_id: str
    baseline_value: float
    candidate_value: float
    delta_pct: float
    threshold_pct: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Path to baseline benchmark_results.json")
    parser.add_argument("--candidate", required=True, help="Path to candidate benchmark_results.json")
    parser.add_argument(
        "--metric",
        default="real_time",
        help="Numeric benchmark metric to compare (default: real_time).",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=5.0,
        help="Allowed slowdown percentage for unspecified benchmarks.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional JSON file with benchmark-specific thresholds.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail when baseline/candidate do not share the same benchmark IDs.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for machine-readable comparison output JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_thresholds(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"default_max_regression_pct": None, "overrides": []}
    data = load_json(path)
    default = data.get("default_max_regression_pct")
    overrides = data.get("overrides", [])
    return {
        "default_max_regression_pct": default,
        "overrides": overrides,
    }


def normalize_rows(doc: dict[str, Any], metric: str) -> dict[str, float]:
    rows: dict[str, float] = {}

    if "suites" in doc:
        for suite in doc.get("suites", []):
            backend = suite.get("backend", "unknown")
            result = suite.get("result") or {}
            for entry in result.get("benchmarks", []):
                if entry.get("run_type", "iteration") != "iteration":
                    continue
                if metric not in entry:
                    continue
                value = entry.get(metric)
                if not isinstance(value, (int, float)):
                    continue
                benchmark_name = entry.get("run_name") or entry.get("name")
                if not benchmark_name:
                    continue
                key = f"{backend}:{benchmark_name}"
                rows[key] = float(value)
    elif "benchmarks" in doc:
        for entry in doc.get("benchmarks", []):
            if entry.get("run_type", "iteration") != "iteration":
                continue
            if metric not in entry:
                continue
            value = entry.get(metric)
            if not isinstance(value, (int, float)):
                continue
            benchmark_name = entry.get("run_name") or entry.get("name")
            if not benchmark_name:
                continue
            rows[f"unknown:{benchmark_name}"] = float(value)

    return rows


def resolve_threshold(benchmark_id: str, default_threshold: float, threshold_config: dict[str, Any]) -> float:
    for override in threshold_config.get("overrides", []):
        pattern = override.get("pattern")
        threshold = override.get("max_regression_pct")
        if not isinstance(pattern, str) or not isinstance(threshold, (float, int)):
            continue
        if fnmatch.fnmatch(benchmark_id, pattern):
            return float(threshold)

    config_default = threshold_config.get("default_max_regression_pct")
    if isinstance(config_default, (int, float)):
        return float(config_default)
    return default_threshold


def compare(
    baseline_rows: dict[str, float],
    candidate_rows: dict[str, float],
    default_threshold: float,
    threshold_config: dict[str, Any],
    fail_on_missing: bool,
) -> tuple[list[ComparisonRow], list[str]]:
    diagnostics: list[str] = []

    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)

    missing_in_candidate = sorted(baseline_keys - candidate_keys)
    missing_in_baseline = sorted(candidate_keys - baseline_keys)

    if missing_in_candidate:
        diagnostics.append(f"Missing in candidate: {', '.join(missing_in_candidate)}")
    if missing_in_baseline:
        diagnostics.append(f"Missing in baseline: {', '.join(missing_in_baseline)}")

    if fail_on_missing and (missing_in_candidate or missing_in_baseline):
        diagnostics.append("Missing benchmarks are considered failures (--fail-on-missing).")

    rows: list[ComparisonRow] = []
    for benchmark_id in sorted(baseline_keys & candidate_keys):
        baseline_value = baseline_rows[benchmark_id]
        candidate_value = candidate_rows[benchmark_id]

        if baseline_value == 0:
            delta_pct = math.inf if candidate_value > 0 else 0.0
        else:
            delta_pct = ((candidate_value - baseline_value) / baseline_value) * 100.0

        threshold = resolve_threshold(benchmark_id, default_threshold, threshold_config)
        status = "ok"
        if delta_pct > threshold:
            status = "regression"
        elif delta_pct < 0:
            status = "improvement"

        rows.append(
            ComparisonRow(
                benchmark_id=benchmark_id,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                delta_pct=delta_pct,
                threshold_pct=threshold,
                status=status,
            )
        )

    return rows, diagnostics


def print_report(rows: list[ComparisonRow], diagnostics: list[str]) -> None:
    if diagnostics:
        print("\n".join(f"[diag] {item}" for item in diagnostics))

    print("benchmark_id | baseline | candidate | delta% | threshold% | status")
    print("-" * 84)
    for row in rows:
        print(
            f"{row.benchmark_id} | {row.baseline_value:.4f} | {row.candidate_value:.4f} | "
            f"{row.delta_pct:.2f} | {row.threshold_pct:.2f} | {row.status}"
        )


def main() -> int:
    args = parse_args()

    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    thresholds_path = Path(args.thresholds).resolve() if args.thresholds else None

    baseline_doc = load_json(baseline_path)
    candidate_doc = load_json(candidate_path)
    threshold_config = parse_thresholds(thresholds_path)

    baseline_rows = normalize_rows(baseline_doc, args.metric)
    candidate_rows = normalize_rows(candidate_doc, args.metric)

    rows, diagnostics = compare(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        default_threshold=args.max_regression_pct,
        threshold_config=threshold_config,
        fail_on_missing=args.fail_on_missing,
    )

    print_report(rows, diagnostics)

    regressions = [row for row in rows if row.status == "regression"]
    missing_failure = args.fail_on_missing and any("Missing" in item for item in diagnostics)

    output_payload = {
        "schema_version": 1,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "metric": args.metric,
        "default_max_regression_pct": args.max_regression_pct,
        "threshold_file": str(thresholds_path) if thresholds_path else None,
        "diagnostics": diagnostics,
        "summary": {
            "total_compared": len(rows),
            "regression_count": len(regressions),
            "missing_failure": missing_failure,
        },
        "rows": [
            {
                "benchmark_id": row.benchmark_id,
                "baseline_value": row.baseline_value,
                "candidate_value": row.candidate_value,
                "delta_pct": row.delta_pct,
                "threshold_pct": row.threshold_pct,
                "status": row.status,
            }
            for row in rows
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote comparison JSON to {output_path}")

    if regressions:
        print(f"Detected {len(regressions)} regressions above threshold.", file=sys.stderr)
        return 1

    if missing_failure:
        print("Missing benchmark rows treated as failure.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
