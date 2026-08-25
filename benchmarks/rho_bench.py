"""Venue A1: rho-continuation (elastic-penalty homotopy) for isqp on a global window.

Hypothesis: isqp's elastic slacks already permit infeasible iterates; what pinned
the hard residuals was a FIXED rho=1e3 (constraints effectively rigid from step
one) plus window locality. Annealing rho from soft to hard is the SQP-native
analogue of the chi untangler's eps-continuation: low rho lets the iterate travel
THROUGH folds (the disconnected-feasible-set requirement), high rho converges to
exact feasibility. One subproblem, one decision vector, continuation over rho.

Usage:
    python benchmarks/rho_bench.py --input benchmarks/output/ladder/inputs/z16_2tri_out.npy
    python benchmarks/rho_bench.py --input .../z0_feasnone_out.npy --margin 20 --out out.npy
"""

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from dvfopt.constraints import FiniteJdetConstraint2D, SimplexConstraint2D  # noqa: E402
from dvfopt.core.primitives.isqp import isqp_solve  # noqa: E402
from dvfopt.core.windowed import build_subproblem, min_field  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import L2Objective, NoneObjective  # noqa: E402


def stats(phi, thr):
    m = np.minimum(*_triangle_areas_2d(phi[0], phi[1]))
    c = FiniteJdetConstraint2D(shape=phi.shape[1:])
    f = np.asarray(c.values(c.flatten(phi))) + thr  # raw finite jdet values
    return {
        "simplex_folds": int((m < thr).sum()),
        "simplex_neg": int((m < 0).sum()),
        "simplex_min": float(m.min()),
        "finite_folds": int((f < thr).sum()),
        "finite_min": float(f.min()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--rhos", default="1,3,10,30,100,1e3,1e4,1e5")
    ap.add_argument("--objective", default="none", choices=["none", "l2"])
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--margin", type=int, default=8, help="window margin around residual bbox")
    ap.add_argument("--box", default="bbox", choices=["bbox", "full"])
    ap.add_argument("--repeat-final", type=int, default=4)
    a = ap.parse_args()

    thr = a.threshold
    phi0 = np.load(a.input).astype(np.float64)
    H, W = phi0.shape[1:]
    c = SimplexConstraint2D(shape=(H, W))
    print(f"input {a.input}: {stats(phi0, thr)}", flush=True)

    if a.box == "full":
        box = (0, H, 0, W)
    else:
        ys, xs = np.where(min_field(c, phi0) < thr)
        box = (
            max(0, int(ys.min()) - a.margin),
            min(H, int(ys.max()) + 2 + a.margin),
            max(0, int(xs.min()) - a.margin),
            min(W, int(xs.max()) + 2 + a.margin),
        )
    obj = NoneObjective() if a.objective == "none" else L2Objective()
    sub = build_subproblem(c, phi0, box, thr, objective=obj)
    x = sub.flat0.copy()
    print(
        f"window {box} ({(box[1] - box[0]) * (box[3] - box[2])} px), objective={a.objective}",
        flush=True,
    )

    rhos = [float(r) for r in a.rhos.split(",")]
    rhos += [rhos[-1]] * a.repeat_final
    phi = phi0.copy()
    last = None
    for i, rho in enumerate(rhos):
        tr = {}
        t = time.perf_counter()
        x, nit, _ok = isqp_solve(
            x,
            sub.cons,
            sub.cons_jac,
            sub.obj_grad,
            a.maxiter,
            rho=rho,
            obj=sub.obj,
            hess_diag=sub.hess_diag,
            free_idx=sub.free_idx,
            trace=tr,
        )
        patch = np.asarray(sub.constraint.unflatten(x))
        py0, py1, px0, px1 = sub.patch_box
        phi[:, py0:py1, px0:px1][:, sub.free_mask] = patch[:, sub.free_mask]
        s = stats(phi, thr)
        print(
            f"rung {i:2d} rho={rho:8.0f}: exit={tr.get('exit')!s:<11} nit={nit:3d} "
            f"simplex folds={s['simplex_folds']:5d} neg={s['simplex_neg']:4d} min={s['simplex_min']:+.5f} "
            f"| finite folds={s['finite_folds']:5d} min={s['finite_min']:+.5f} "
            f"| L2move={np.linalg.norm(phi - phi0):.1f} {time.perf_counter() - t:.0f}s",
            flush=True,
        )
        if i >= len(rhos) - a.repeat_final - 1:
            if s["simplex_folds"] == 0:
                print("0 SIMPLEX FOLDS — rho-continuation cleared the residual", flush=True)
                break
            if last is not None and s["simplex_folds"] >= last:
                print("final-rho repeats stopped improving", flush=True)
                break
            last = s["simplex_folds"]
    if a.out:
        np.save(a.out, phi)


if __name__ == "__main__":
    main()
