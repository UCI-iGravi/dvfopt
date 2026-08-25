"""Graduated-inflation ladder on a residual fold cluster (resumable, instrumented).

The distance-minimising windowed solve can leave a residual that is NOT a geometric
floor but an objective-basin / area-transport trap (B0039 z=0: a ~3x compressed core
whose cells sit at the 2t self-capacity). This harness attacks it with the monotone
elastic isqp inner (no row may worsen) over one global window around the residual,
inflating the target threshold rung by rung (0 -> ... -> thr) so area can seep in from
the fat periphery across stages.

Per rung it records fold counts AND the inflation physics: cells below the 2t
self-capacity, area quantiles inside the enforced region, and the region's total
area (invariant under interior node moves - a built-in sanity check). State is saved
after every rung and the run resumes from ``<out>/state.npy`` + ``rungs.csv``.

Usage:
    python benchmarks/ladder_bench.py --input benchmarks/output/ladder/inputs/z0_feasnone_out.npy
        --out benchmarks/output/ladder/z0 --rungs 0,0.002,0.005,0.0075,0.01 --repeat-final 6
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from dvfopt.constraints import (  # noqa: E402
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    TriConstraint2D,
)
from dvfopt.core.primitives.isqp import isqp_solve  # noqa: E402
from dvfopt.core.windowed import build_subproblem, min_field  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import L2Objective, NoneObjective  # noqa: E402

FAMILY = {"jdet": JdetConstraint2D, "2tri": TriConstraint2D, "finite": FiniteJdetConstraint2D}
FIELDS = [
    "rung",
    "target",
    "exit",
    "nit",
    "wall_s",
    "below_target",
    "folds",
    "neg",
    "min",
    "cells_below_2t",
    "area_p01",
    "area_p25",
    "area_min",
    "region_area",
    "l1_move",
    "l2_move",
]


def physics(phi, box, thr):
    """Inflation physics over the cells inside ``box`` (fy0, fy1, fx0, fx1)."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    area = (T1 + T2)[box[0] : box[1] - 1, box[2] : box[3] - 1]
    return {
        "cells_below_2t": int((area < 2 * thr).sum()),
        "area_p01": float(np.percentile(area, 1)),
        "area_p25": float(np.percentile(area, 25)),
        "area_min": float(area.min()),
        "region_area": float(area.sum()),
    }


def rung(phi, phi0, ctype, box, target, thr, a, log):
    """One ladder rung: monotone pure-feasibility isqp over the global window.

    With ``--protect-healthy`` the per-row target is ``thr`` for rows already at or
    above ``thr`` and ``target`` (the rung) only for sick rows — so no healthy cell
    is ever sacrificed to a lower rung (the plain ladder lets at-capacity cells sink
    to the rung target and the final rung cannot repay the borrowed area).
    """
    c = ctype(shape=phi.shape[1:])
    # objective: pure feasibility (anchor 0) or a small L2 anchor to the RUNG-START
    # state so borrowing has a price — without it dips are free and borrowed area
    # diffuses into a halo instead of being funneled into the sick cells.
    objective = L2Objective() if a.anchor_weight > 0 else NoneObjective()
    sub = build_subproblem(c, phi, box, target, objective=objective, margin_delta=0.0)
    w = a.anchor_weight
    obj_fn = (lambda f, _o=sub.obj: w * _o(f)) if w > 0 else sub.obj
    grad_fn = (lambda f, _g=sub.obj_grad: w * _g(f)) if w > 0 else sub.obj_grad
    cons, n_healthy = sub.cons, 0
    if a.protect_healthy:
        v0 = np.asarray(sub.cons(sub.flat0)) + target  # raw enforced-row values now
        # healthy rows keep the full threshold, or — with --dip — may give up at most
        # `dip` below their current value this rung (controlled borrowing; the next
        # rung re-raises them): transport needs borrow-then-repay, strict monotone
        # never borrows and plain merit never repays.
        floor = thr if a.dip <= 0 else np.maximum(target, np.minimum(thr, v0 - a.dip))
        t_vec = np.where(v0 >= thr, floor, target)
        n_healthy = int((v0 >= thr).sum())
        shift = target - t_vec

        def cons(f, _s=shift, _c=sub.cons):
            return np.asarray(_c(f)) + _s  # vals - t_vec

    hess = (lambda f, _h=sub.hess_diag: w * _h(f)) if w > 0 else sub.hess_diag

    tr = {}
    t = time.perf_counter()
    x, nit, _ok = isqp_solve(
        sub.flat0.copy(),
        cons,
        sub.cons_jac,
        grad_fn,
        a.maxiter,
        rho=a.rho,
        obj=obj_fn,
        hess_diag=hess,
        free_idx=sub.free_idx,
        trace=tr,
        monotone=a.monotone,
        protect=a.protect,
        osqp_eps=a.osqp_eps,
        log_every=a.log_every,
    )
    wall = time.perf_counter() - t
    patch = np.asarray(sub.constraint.unflatten(x))
    py0, py1, px0, px1 = sub.patch_box
    phi[:, py0:py1, px0:px1][:, sub.free_mask] = patch[:, sub.free_mask]
    mf = min_field(c, phi)
    move = phi - phi0
    row = {
        "target": target,
        "exit": tr.get("exit"),
        "nit": nit,
        "wall_s": round(wall, 1),
        "below_target": int((mf < target).sum()) if target > 0 else int((mf < 0).sum()),
        "folds": int((mf < thr).sum()),
        "neg": int((mf < 0).sum()),
        "min": float(mf.min()),
        "l1_move": float(np.abs(move).sum()),
        "l2_move": float(np.linalg.norm(move)),
        **physics(phi, box, thr),
    }
    log(
        f"rung target={target:.4f}: exit={row['exit']:<11} nit={nit:3d} folds={row['folds']:5d} "
        f"neg={row['neg']:4d} min={row['min']:+.5f} below2t={row['cells_below_2t']:4d} "
        f"p25area={row['area_p25']:.4f} region_area={row['region_area']:.1f} "
        f"healthy_protected={n_healthy} {wall:.0f}s"
    )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="2tri", choices=list(FAMILY))
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--rungs", default="0,0.002,0.005,0.0075,0.01")
    ap.add_argument("--repeat-final", type=int, default=6, help="repeat last rung while folds drop")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--rho", type=float, default=1e3)
    ap.add_argument("--osqp-eps", type=float, default=1e-5)
    ap.add_argument("--protect", type=float, default=1.0)
    ap.add_argument("--no-monotone", dest="monotone", action="store_false")
    ap.add_argument(
        "--protect-healthy", action="store_true", help="healthy rows keep thr at every rung"
    )
    ap.add_argument(
        "--dip", type=float, default=0.0, help="max per-rung dip allowed for healthy rows"
    )
    ap.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help="L2 anchor to rung-start state (0 = pure feasibility)",
    )
    ap.add_argument("--margin", type=int, default=6, help="window margin around the residual bbox")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state, csv_path, logp = out / "state.npy", out / "rungs.csv", out / "log.txt"
    thr, ctype = a.threshold, FAMILY[a.family]

    def log(msg):
        print(msg, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    phi0 = np.load(a.input).astype(np.float64)
    done = []
    if a.resume and state.exists() and csv_path.exists():
        phi = np.load(state)
        with open(csv_path) as f:
            done = list(csv.DictReader(f))
        log(f"RESUME from {state} after {len(done)} rungs")
    else:
        phi = phi0.copy()
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    H, W = phi.shape[1:]
    c0 = ctype(shape=(H, W))
    mask = min_field(c0, phi0) < thr
    ys, xs = np.where(mask)
    box = (
        max(0, ys.min() - a.margin),
        min(H, ys.max() + 2 + a.margin),
        max(0, xs.min() - a.margin),
        min(W, xs.max() + 2 + a.margin),
    )
    box = tuple(int(v) for v in box)
    (out / "config.json").write_text(json.dumps({**vars(a), "box": box}, indent=1))
    log(
        f"input {a.input}: {int(mask.sum())} residual folds, window {box} ({a.family}, "
        f"monotone={a.monotone}, protect={a.protect}, eps={a.osqp_eps}, maxiter={a.maxiter})"
    )
    log("start physics: " + json.dumps(physics(phi, box, thr)))

    rungs = [float(r) for r in a.rungs.split(",")]
    plan = [(i, t) for i, t in enumerate(rungs)] + [
        (len(rungs) + k, rungs[-1]) for k in range(a.repeat_final)
    ]
    n_done = len(done)
    last = int(done[-1]["folds"]) if done else None
    for i, target in plan:
        if i < n_done:
            continue
        if i >= len(rungs) and last == 0:
            break
        row = rung(phi, phi0, ctype, box, target, thr, a, log)
        row["rung"] = i
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        np.save(state, phi)
        if i >= len(rungs) and last is not None and row["folds"] >= last:
            log("final-rung repeat stopped improving")
            break
        last = row["folds"]
        if i >= len(rungs) - 1 and row["folds"] == 0:
            log("CLUSTER CLEARED")
            break
    log(f"done: final folds={last} -> {state}")


if __name__ == "__main__":
    main()
