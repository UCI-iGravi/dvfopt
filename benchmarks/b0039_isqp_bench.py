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
SWEEP_SOLVERS = (
    "isqp-osqp",
    "scipy-slsqp",
    "scipy-trust-constr",
    "scipy-slsqp+trust-constr",  # escalation hybrid
)


def _folds(dy, dx, thr=0.0):
    j = _numpy_jdet_2d(dy, dx)
    return int((j < thr).sum()), float(j.min())


HEADER = [
    "z",
    "objective",
    "solver",
    "crop_folds",
    "crop_min",
    "folds_after",
    "min_after",
    "success",
    "n_iter",
    "l1_move",
    "l2_move",
    "n_moved",
    "time_s",
]


def run(vol_path, stride, size, maxiter, threshold, solvers, objectives, out_csv, eps=1e-2):
    # Keep the (3,D,H,W) volume in its native dtype (float32 ~= 0.6 GB); each tiny
    # crop is cast to float64 inside _problem. Casting the whole volume to float64
    # up front is ~1.85 GB and pointless (only the crop is solved).
    vol = np.load(vol_path)
    d = vol.shape[1]
    rows = []
    print(
        f"volume {vol_path} shape={vol.shape} | stride={stride} size={size} "
        f"objectives={objectives} solvers={solvers}"
    )
    for z in range(0, d, stride):
        sl = vol[:, z : z + 1]  # (3,1,H,W)
        n0, _ = _folds(sl[1, 0], sl[2, 0], thr=threshold)
        if n0 == 0:
            continue
        crop = sv.crop_fold_region(sl, size=size, threshold=threshold)
        cn0, cmn0 = _folds(crop[0], crop[1], thr=threshold)  # folds inside the crop
        for obj in objectives:
            line = [f"z={z:3d} {obj} crop-folds={cn0:4d} min={cmn0:8.3f}"]
            for s in solvers:
                t = time.perf_counter()
                try:
                    _, info = sv.full_grid_correct(
                        crop, s, threshold=threshold, maxiter=maxiter, objective=obj, eps=eps
                    )
                    dt = time.perf_counter() - t
                    rows.append(
                        [
                            z,
                            obj,
                            s,
                            cn0,
                            cmn0,
                            info["folds_after"],
                            info["min_after"],
                            int(info["success"]),
                            info["n_iter"],
                            round(info["l1_move"], 4),
                            round(info["l2_move"], 4),
                            info["n_moved"],
                            round(dt, 2),
                        ]
                    )
                    line.append(
                        f"{s}:{info['min_after']:+.3f}/L1={info['l1_move']:.2f}/{dt:.1f}s"
                        + ("" if info["success"] else "!")
                    )
                except Exception as e:
                    dt = time.perf_counter() - t
                    rows.append(
                        [
                            z,
                            obj,
                            s,
                            cn0,
                            cmn0,
                            -1,
                            float("nan"),
                            0,
                            -1,
                            float("nan"),
                            float("nan"),
                            -1,
                            round(dt, 2),
                        ]
                    )
                    line.append(f"{s}:ERR({type(e).__name__})")
            print("  " + " | ".join(line))
            sys.stdout.flush()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    # ---- summary: per (objective, solver) feasibility, speed, and correction footprint ----
    print("\n=== summary (feasible = no negative dets; L1/L2-move = correction footprint) ===")
    n_probs = len({r[0] for r in rows})
    print(f"problems (slices with folds): {n_probs}")
    for obj in objectives:
        print(f"[{obj} objective]")
        for s in solvers:
            rs = [r for r in rows if r[1] == obj and r[2] == s]
            if not rs:
                continue
            succ = sum(r[7] for r in rs)
            med_t = float(np.median([r[12] for r in rs if r[12] >= 0]))
            med_l1 = float(np.median([r[9] for r in rs if np.isfinite(r[9])]))
            med_l2 = float(np.median([r[10] for r in rs if np.isfinite(r[10])]))
            med_nm = float(np.median([r[11] for r in rs if r[11] >= 0]))
            worst_min = min((r[6] for r in rs if np.isfinite(r[6])), default=float("nan"))
            print(
                f"  {s:20s} feasible {succ:3d}/{len(rs):3d} | median {med_t:5.2f}s | "
                f"L1-move {med_l1:8.3f} | L2-move {med_l2:7.3f} | pixels-moved {med_nm:6.0f} | "
                f"worst min {worst_min:+.4f}"
            )
    print(f"\ncsv -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default=DEFAULT_VOL)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--objective", choices=["l1", "l2", "both"], default="both")
    ap.add_argument(
        "--eps", type=float, default=1e-2, help="smoothed-L1 eps (1e-4 collapses curvature)"
    )
    ap.add_argument("--include-proto", action="store_true", help="also run the dense quadprog POC")
    ap.add_argument("--out", default="benchmarks/output/b0039_isqp_bench.csv")
    a = ap.parse_args()
    solvers = list(SWEEP_SOLVERS) + (["isqp-proto"] if a.include_proto else [])
    solvers = [s for s in solvers if s in sv.available_solvers()]
    objectives = ["l2", "l1"] if a.objective == "both" else [a.objective]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    run(a.vol, a.stride, a.size, a.maxiter, a.threshold, solvers, objectives, a.out, eps=a.eps)
