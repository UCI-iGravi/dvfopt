"""Benchmark the optimized I-SLSQP (``isqp-osqp``) against ``scipy-slsqp`` and
``scipy-trust-constr`` on every N-th slice of the B0039 Laplacian field.

For each sampled slice with a fold, we crop a fixed patch around the worst
determinant (so every solver sees the *identical* full-grid feasibility
problem), correct it with each solver, and record folds cleared, resulting
min-jdet, feasibility, iterations, and wall time. Results stream to a CSV and a
printed per-slice line; a summary aggregates win-rate and timing.

Usage:
    python -u benchmarks/b0039_isqp_bench.py --stride 32 --size 48 --out out.csv

``isqp-proto`` (dense quadprog POC) is excluded from the sweep by default — it
is ~10x slower; add it with ``--include-proto`` to see the speedup on one slice.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slsqp_variants as sv

from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d

DEFAULT_VOL = "data/dvfs/b0039/b0039_laplacian_deformation_field.npy"
SWEEP_SOLVERS = ("scipy-slsqp", "scipy-trust-constr", "isqp-osqp")


def _folds(dy, dx, thr=0.0):
    j = _numpy_jdet_2d(dy, dx)
    return int((j < thr).sum()), float(j.min())


def run(vol_path, stride, size, maxiter, threshold, solvers, out_csv):
    vol = np.load(vol_path).astype(np.float64)  # (3, D, H, W)
    d = vol.shape[1]
    rows = []
    header = [
        "z",
        "folds0",
        "min0",
        "solver",
        "folds_after",
        "min_after",
        "success",
        "n_iter",
        "time_s",
    ]
    print(f"volume {vol_path} shape={vol.shape} | stride={stride} size={size} solvers={solvers}")
    for z in range(0, d, stride):
        sl = vol[:, z : z + 1]  # (3,1,H,W)
        n0, mn0 = _folds(sl[1, 0], sl[2, 0], thr=threshold)
        if n0 == 0:
            continue
        crop = sv.crop_fold_region(sl, size=size, threshold=threshold)
        cn0, cmn0 = _folds(crop[0], crop[1], thr=threshold)  # folds inside the crop
        line = [f"z={z:3d} slice-folds={n0:5d} crop-folds={cn0:4d} min={cmn0:8.3f}"]
        for s in solvers:
            t = time.perf_counter()
            try:
                _, info = sv.full_grid_correct(crop, s, threshold=threshold, maxiter=maxiter)
                dt = time.perf_counter() - t
                rows.append(
                    [
                        z,
                        cn0,
                        cmn0,
                        s,
                        info["folds_after"],
                        info["min_after"],
                        int(info["success"]),
                        info["n_iter"],
                        round(dt, 2),
                    ]
                )
                line.append(
                    f"{s}:{info['folds_after']}f/{info['min_after']:+.3f}/{dt:.1f}s"
                    + ("" if info["success"] else "!")
                )
            except Exception as e:
                dt = time.perf_counter() - t
                rows.append([z, cn0, cmn0, s, -1, float("nan"), 0, -1, round(dt, 2)])
                line.append(f"{s}:ERR({type(e).__name__})")
        print("  " + " | ".join(line))
        sys.stdout.flush()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ---- summary ----
    print("\n=== summary ===")
    by = {s: [r for r in rows if r[3] == s] for s in solvers}
    n_probs = len({r[0] for r in rows})
    print(f"problems (slices with folds): {n_probs}")
    for s in solvers:
        rs = by[s]
        if not rs:
            continue
        succ = sum(r[6] for r in rs)
        tt = sum(r[8] for r in rs if r[8] >= 0)
        med = float(np.median([r[8] for r in rs if r[8] >= 0])) if rs else float("nan")
        worst_min = min((r[5] for r in rs if np.isfinite(r[5])), default=float("nan"))
        print(
            f"  {s:20s} feasible {succ:3d}/{len(rs):3d} | total {tt:7.1f}s | median {med:5.2f}s | "
            f"worst min_after {worst_min:+.4f}"
        )
    print(f"\ncsv -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default=DEFAULT_VOL)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--include-proto", action="store_true", help="also run the dense quadprog POC")
    ap.add_argument("--out", default="benchmarks/output/b0039_isqp_bench.csv")
    a = ap.parse_args()
    solvers = list(SWEEP_SOLVERS) + (["isqp-proto"] if a.include_proto else [])
    solvers = [s for s in solvers if s in sv.available_solvers()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    run(a.vol, a.stride, a.size, a.maxiter, a.threshold, solvers, a.out)
