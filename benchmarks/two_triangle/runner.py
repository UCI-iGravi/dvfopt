"""CLI entrypoint and programmatic API for sweeping variants × cases.

Usage (CLI):
    python -m benchmarks.two_triangle.runner --variants baseline_serial \
        --cases synth2d_single_cell_flip
    python -m benchmarks.two_triangle.runner --all
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure repo root on sys.path when invoked as `python -m`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import benchmarks.two_triangle.cases  # noqa: E402  -- registers cases
import benchmarks.two_triangle.variants  # noqa: F401, E402  -- registers variants
from benchmarks.two_triangle import registry  # noqa: E402
from benchmarks.two_triangle.result import SolverResult  # noqa: E402


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_rss_mb() -> float:
    """Resident set size in MB. Falls back to 0 if psutil is unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _run_cell(
    variant_name: str,
    case_name: str,
    *,
    timeout_s: float,
    threshold: float,
    max_iterations: int,
    output_root: Path,
) -> dict:
    """Run one (variant, case). Returns a manifest entry (always — even on error).

    Caveat: timeout_s is honoured *cooperatively* by variants that own their
    loop (they check elapsed time per outer iter). The baseline variant calls
    the underlying iterative_serial directly and cannot be interrupted —
    `max_iterations` is the only bound for it.

    Memory sampling is best-effort: rss_after captures resident set size
    immediately after the variant returns. It is an upper-bound proxy for
    peak memory, not a true high-water mark — sampling continuously would
    require a thread, which complicates the runner.
    """
    variant_fn = registry.get_variant(variant_name)
    case_fn = registry.get_case(case_name)
    case_meta = registry.case_metadata(case_name)
    cell_dir = output_root / variant_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cell_dir / f"{case_name}.csv"

    entry = {
        "variant": variant_name,
        "case": case_name,
        "case_meta": case_meta,
        "started_at": _utc_stamp(),
    }
    try:
        dvf, _meta = case_fn()
        rss_before = _current_rss_mb()
        t0 = time.perf_counter()
        result: SolverResult = variant_fn(
            dvf, threshold=threshold, max_iterations=max_iterations, timeout_s=timeout_s
        )
        elapsed = time.perf_counter() - t0
        rss_after = _current_rss_mb()
        result.to_csv(csv_path)
        entry.update(
            {
                "status": "ok",
                "wall_s": elapsed,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": max(0.0, rss_after - rss_before),
                "converged": result.converged,
                "timed_out": result.timed_out,
                "final_folds_jdet": int(result.trajectory.iloc[-1]["fold_count_jdet"]),
                "final_folds_tri": int(result.trajectory.iloc[-1]["fold_count_tri"]),
                "n_traj_rows": len(result.trajectory),
                "result_csv": str(csv_path.relative_to(output_root)),
            }
        )
    except FileNotFoundError as exc:
        entry.update({"status": "skipped", "reason": f"data file missing: {exc}"})
    except Exception:
        entry.update({"status": "error", "traceback": traceback.format_exc()})
    return entry


def run_sweep(
    *,
    variants: Iterable[str],
    cases: Iterable[str],
    output_dir: Optional[Path] = None,
    timeout_s: float = 600.0,
    threshold: float = 0.01,
    max_iterations: int = 100,
) -> Path:
    """Programmatic sweep entrypoint. Returns path to the run's manifest.json."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "runs" / _utc_stamp()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate before running anything
    variants = list(variants)
    cases = list(cases)
    for v in variants:
        registry.get_variant(v)
    for c in cases:
        registry.get_case(c)

    cells = []
    for v in variants:
        for c in cases:
            print(f"[run] {v} × {c}", flush=True)
            cells.append(
                _run_cell(
                    v,
                    c,
                    timeout_s=timeout_s,
                    threshold=threshold,
                    max_iterations=max_iterations,
                    output_root=output_dir,
                )
            )

    manifest = {
        "git_sha": _git_sha(),
        "utc": _utc_stamp(),
        "host": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "n_cells": len(cells),
        "config": {
            "timeout_s": timeout_s,
            "threshold": threshold,
            "max_iterations": max_iterations,
        },
        "cells": cells,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[done] manifest -> {manifest_path}", flush=True)
    return manifest_path


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-triangle benchmark runner")
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant names. Use --all-variants for everything.",
    )
    p.add_argument(
        "--cases", nargs="+", default=None, help="Case names. Use --all-cases for everything."
    )
    p.add_argument("--all", action="store_true", help="Shortcut for --all-variants --all-cases.")
    p.add_argument("--all-variants", action="store_true")
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.all:
        args.all_variants = args.all_cases = True
    variants = registry.list_variants() if args.all_variants else args.variants
    cases = registry.list_cases() if args.all_cases else args.cases
    if not variants or not cases:
        print("Specify --variants and --cases (or --all). Available:")
        print(f"  variants: {registry.list_variants()}")
        print(f"  cases:    {registry.list_cases()}")
        return 2
    run_sweep(
        variants=variants,
        cases=cases,
        output_dir=args.output,
        timeout_s=args.timeout_s,
        threshold=args.threshold,
        max_iterations=args.max_iter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
