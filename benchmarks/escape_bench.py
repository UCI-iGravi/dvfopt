"""Benchmark the residual-escape modes (:mod:`windowed_escape`) head-to-head.

The distance-minimising windowed solve leaves a few folds behind (an objective-basin
trap, not a geometric floor — see the z=16 analysis). This benchmark compares the four
escape modes on the SAME residual fields:

- ``baseline``  — current inner; leaves the residual (the comparison floor).
- ``twophase``  — pure feasibility then re-anchor distance (reference).
- ``weighted``  — Option A: one solve, fold-localized fidelity weight.
- ``penalty``   — Option B: one solve, soft-penalty + elastic-rho continuation.

Each mode is applied to the same residual field via
:func:`windowed_escape.repair_residuals`; we record folds cleared, the L1/L2 correction
footprint (the fidelity cost), the worst enforced-constraint margin, damage (must be 0),
and wall time. Inputs are pre-solved residual fields (``.npy`` of a ``(2, H, W)``
``[dy, dx]`` slice with a handful of leftover folds), e.g. the z=16 / z=0 2-tri outputs.

Usage:
    python benchmarks/escape_bench.py --fields z16.npy z0.npy --family 2tri --objective l2
"""

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import windowed_escape as we  # noqa: E402


def run_field(path, family, objective, threshold, margin, maxiter, taper, modes, max_passes):
    """Run every mode on one residual field; return a list of report dicts (+ tag)."""
    phi = np.load(path)
    if phi.ndim == 4:  # accept a (3,1,H,W) or (2,1,H,W) array too
        phi = phi[-2:, 0]
    tag = Path(path).stem
    rows = []
    for mode in modes:
        _, rep = we.repair_residuals(
            phi,
            family=family,
            threshold=threshold,
            mode=mode,
            objective=objective,
            margin=margin,
            maxiter=maxiter,
            taper=taper,
            max_passes=max_passes,
        )
        rep["field"] = tag
        rows.append(rep)
        fa = rep["folds_after"]
        print(
            f"  {tag:>16} {mode:<9} folds {rep['folds_before']}->{fa} "
            f"damage={rep['damage']} L2={rep['l2_move']:.3f} L1={rep['l1_move']:.1f} "
            f"margin={_fmt(rep['min_margin'])} wins={rep['n_windows']} {rep['time_s']:.1f}s",
            flush=True,
        )
    return rows


def _fmt(m):
    return "na" if m is None else f"{m:.4f}"


def write_report(path, rows, cfg, modes):
    """Aggregate per mode across all fields and write a markdown comparison."""
    lines = [
        "# Residual-escape mode benchmark",
        "",
        f"family `{cfg['family']}` | objective `{cfg['objective']}` | threshold "
        f"{cfg['threshold']} | margin {cfg['margin']} | maxiter {cfg['maxiter']} | "
        f"taper {cfg['taper']} | fields: {cfg['n_fields']}",
        "",
        "Each mode repairs the SAME residual fields. `folds→` is total leftover folds "
        "before→after; `%cleared` is the fraction removed; `L1/L2 move` is the correction "
        "footprint (fidelity cost — lower = more faithful); `worst_margin` is the min "
        "enforced-constraint value (≥0 = feasible); `damage` MUST be 0 (no-damage invariant).",
        "",
        "## Per-mode totals (summed over fields)",
        "",
        "| mode | fields | folds→ | %cleared | worst_damage | avg L1 move | avg L2 move | "
        "worst_margin | total_time_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        if not mr:
            continue
        fb = sum(r["folds_before"] for r in mr)
        fa = sum(r["folds_after"] for r in mr)
        pct = 100.0 if fb == 0 else (fb - fa) / fb * 100.0
        wd = max(r["damage"] for r in mr)
        wd_cell = str(wd) if wd == 0 else f"**INVARIANT VIOLATED ({wd})**"
        avg_l1 = float(np.mean([r["l1_move"] for r in mr]))
        avg_l2 = float(np.mean([r["l2_move"] for r in mr]))
        margins = [r["min_margin"] for r in mr if r["min_margin"] is not None]
        wm = "na" if not margins else f"{min(margins):.4f}"
        tt = sum(r["time_s"] for r in mr)
        lines.append(
            f"| {mode} | {len(mr)} | {fb}→{fa} | {pct:.1f} | {wd_cell} | {avg_l1:.1f} | "
            f"{avg_l2:.3f} | {wm} | {tt:.1f} |"
        )

    lines += [
        "",
        "## Per-field detail",
        "",
        "| field | mode | folds→ | L1 move | L2 move | margin | time_s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['field']} | {r['mode']} | {r['folds_before']}→{r['folds_after']} | "
            f"{r['l1_move']:.1f} | {r['l2_move']:.3f} | {_fmt(r['min_margin'])} | {r['time_s']:.1f} |"
        )
    lines += [
        "",
        "## Reading it",
        "",
        "- **baseline** is the floor: it leaves the residual (the objective-basin trap).",
        "- **twophase / penalty (B)** reach the uniform-nearest feasible point (lowest L2).",
        "- **weighted (A)** localises the correction to the fold, so it is sparser (lowest "
        "L1) and faster, at a marginally higher L2 — the fidelity loss sits only on the "
        "cells that were non-injective in the input anyway.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", nargs="+", required=True, help="residual .npy fields to repair")
    ap.add_argument("--family", default="2tri", choices=["jdet", "finite", "2tri"])
    ap.add_argument("--objective", default="l2", choices=["l2", "l1"])
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--taper", type=float, default=6.0)
    ap.add_argument("--max-passes", type=int, default=3, help="residual re-window passes per mode")
    ap.add_argument(
        "--modes", default="baseline,twophase,weighted,penalty", help="comma list of escape modes"
    )
    ap.add_argument("--out-dir", default="benchmarks/output/escape_bench")
    a = ap.parse_args()

    modes = [m.strip() for m in a.modes.split(",")]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"escape bench | family={a.family} obj={a.objective} modes={modes} fields={len(a.fields)}"
    )

    rows = []
    for f in a.fields:
        print(f"field {f}:")
        rows += run_field(
            f, a.family, a.objective, a.threshold, a.margin, a.maxiter, a.taper, modes, a.max_passes
        )

    cfg = {
        "family": a.family,
        "objective": a.objective,
        "threshold": a.threshold,
        "margin": a.margin,
        "maxiter": a.maxiter,
        "taper": a.taper,
        "n_fields": len(a.fields),
    }
    report = out_dir / "report.md"
    write_report(report, rows, cfg, modes)
    print(f"\nreport -> {report}")


if __name__ == "__main__":
    main()
