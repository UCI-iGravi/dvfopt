"""Venue 4: chi untangling objective INSIDE isqp (hard rows kept).

A1 showed the twisted residuals stall isqp because the LINEARIZED constraint
rows are degenerate at bow-tie cells (ratio<=0 every step, any rho). The chi
energy's gradient does not suffer this: it comes from the smooth distortion
barrier, stays informative through folds, and pulls nodes across inversions.
So: keep the elastic constraint rows (they finish the job and certify), but
replace the distance objective with the untangler energy
    f(phi) = sum (||J||_F^2 + beta)/chi_eps(det - 2t) + (w/2)||phi - anchor||^2
under a JOINT continuation: eps anneals down (objective passes through folds)
while rho anneals up (constraints soften early, harden to exact feasibility).

Usage:
    python benchmarks/chi_in_isqp_bench.py --input benchmarks/output/ladder/inputs/z16_2tri_out.npy
"""

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from chi_untangle import (  # noqa: E402  (benchmarks/ on sys.path)
    chi_stable,
    energy_grad,
    min_area_margin,
)

from dvfopt.constraints import FiniteJdetConstraint2D, SimplexConstraint2D  # noqa: E402
from dvfopt.core.primitives.isqp import isqp_solve  # noqa: E402
from dvfopt.core.windowed import build_subproblem, min_field  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402


def stats(phi, thr):
    m = np.minimum(*_triangle_areas_2d(phi[0], phi[1]))
    c = FiniteJdetConstraint2D(shape=phi.shape[1:])
    f = np.asarray(c.values(c.flatten(phi))) + thr
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
    ap.add_argument("--w-anchor", type=float, default=1e-2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--hscale", type=float, default=20.0, help="diagonal QP Hessian scale")
    ap.add_argument("--rho0", type=float, default=10.0)
    ap.add_argument("--rho-max", type=float, default=1e5)
    ap.add_argument("--maxiter", type=int, default=200, help="isqp iters per stage")
    ap.add_argument("--stages", type=int, default=20)
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument(
        "--no-tr", action="store_true", help="legacy line-search path (trust_region=False)"
    )
    ap.add_argument(
        "--hess",
        default="const",
        choices=["const", "chi"],
        help="QP diagonal: constant or exact chi-energy fro-term diagonal",
    )
    ap.add_argument(
        "--rho-hold",
        action="store_true",
        help="hold rho at rho0 until chi reports feasible, then anneal",
    )
    a = ap.parse_args()

    thr = a.threshold
    phi0 = np.load(a.input).astype(np.float64)
    H, W = phi0.shape[1:]
    c = SimplexConstraint2D(shape=(H, W))
    print(f"input {a.input}: {stats(phi0, thr)}", flush=True)

    ys, xs = np.where(min_field(c, phi0) < thr)
    box = (
        max(0, int(ys.min()) - a.margin),
        min(H, int(ys.max()) + 2 + a.margin),
        max(0, int(xs.min()) - a.margin),
        min(W, int(xs.max()) + 2 + a.margin),
    )
    sub = build_subproblem(c, phi0, box, thr, objective=NoneObjective())
    pc = sub.constraint  # patch-shaped constraint (flatten/unflatten for the patch)
    anchor = np.asarray(pc.unflatten(sub.flat0)).copy()
    x = sub.flat0.copy()
    n = x.size
    print(f"window {box} ({(box[1] - box[0]) * (box[3] - box[2])} px)", flush=True)

    eps_box = [1.0]  # mutable: closures see current eps

    def obj(xf):
        p = np.asarray(pc.unflatten(xf))
        e, _ = energy_grad(p, thr, eps_box[0], anchor, a.w_anchor, beta=a.beta)
        return e

    def obj_grad(xf):
        p = np.asarray(pc.unflatten(xf))
        _, g = energy_grad(p, thr, eps_box[0], anchor, a.w_anchor, beta=a.beta)
        return np.asarray(pc.flatten(g))

    def hess_diag(xf):
        if a.hess == "const":
            return np.full(n, a.hscale)
        # exact diagonal of the fro/chi term: corner k appears in the two J columns
        # of each incident triangle with total d2(fro)/d(node)^2 = 4 or 2; /chi + anchor w
        p = np.asarray(pc.unflatten(xf))
        h2, w2 = p.shape[1:]
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float64)
        X, Y = xx + p[1], yy + p[0]
        TL = (X[:-1, :-1], Y[:-1, :-1])
        TR = (X[:-1, 1:], Y[:-1, 1:])
        BL = (X[1:, :-1], Y[1:, :-1])
        BR = (X[1:, 1:], Y[1:, 1:])
        d1 = (BR[0] - BL[0]) * (BR[1] - TR[1]) - (BR[1] - BL[1]) * (BR[0] - TR[0])
        d2 = (TR[0] - TL[0]) * (BL[1] - TL[1]) - (TR[1] - TL[1]) * (BL[0] - TL[0])
        v1, _ = chi_stable(d1 - 2 * thr, eps_box[0])
        v2, _ = chi_stable(d2 - 2 * thr, eps_box[0])
        g = np.full((h2, w2), a.w_anchor)
        g[1:, 1:] += 4.0 / v1
        g[1:, :-1] += 2.0 / v1
        g[:-1, 1:] += 2.0 / v1  # T1: BR,BL,TR
        g[:-1, :-1] += 4.0 / v2
        g[:-1, 1:] += 2.0 / v2
        g[1:, :-1] += 2.0 / v2  # T2: TL,TR,BL
        return np.asarray(pc.flatten(np.stack([g, g])))

    phi = phi0.copy()
    t0 = time.perf_counter()
    rho = a.rho0
    smin, _, _ = min_area_margin(np.asarray(pc.unflatten(x)), thr)
    last = None
    for stage in range(a.stages):
        if smin > 0:
            if eps_box[0] <= 1e-4 * 1.01 and rho >= a.rho_max * 0.999:
                break
            eps_box[0] = max(1e-4, eps_box[0] / 4.0)
        else:
            eps_box[0] = min(max(1e-4, abs(smin) + 1e-3), 10.0)
        tr = {}
        x, nit, _ok = isqp_solve(
            x,
            sub.cons,
            sub.cons_jac,
            obj_grad,
            a.maxiter,
            rho=rho,
            obj=obj,
            hess_diag=hess_diag,
            free_idx=sub.free_idx,
            trace=tr,
            trust_region=not a.no_tr,
        )
        smin, _, _ = min_area_margin(np.asarray(pc.unflatten(x)), thr)
        patch = np.asarray(pc.unflatten(x))
        py0, py1, px0, px1 = sub.patch_box
        phi[:, py0:py1, px0:px1][:, sub.free_mask] = patch[:, sub.free_mask]
        s = stats(phi, thr)
        print(
            f"stage {stage:2d}: eps={eps_box[0]:.4g} rho={rho:8.0f} exit={tr.get('exit')!s:<11} "
            f"nit={nit:3d} simplex folds={s['simplex_folds']:5d} neg={s['simplex_neg']:4d} "
            f"min={s['simplex_min']:+.5f} | finite folds={s['finite_folds']:4d} "
            f"| L2move={np.linalg.norm(phi - phi0):.1f} {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
        if not a.rho_hold or smin > 0:
            rho = min(a.rho_max, rho * 4.0)
        if s["simplex_folds"] == 0 and smin > 0 and rho >= a.rho_max * 0.999:
            print("0 SIMPLEX FOLDS — chi-objective isqp cleared the residual", flush=True)
            break
        if last is not None and stage > 8 and s["simplex_folds"] >= last and nit <= 2:
            print("stalled (no iterations accepted and folds not improving)", flush=True)
            break
        last = s["simplex_folds"]
    s = stats(phi, thr)
    print(
        f"RESULT: {s} | L2move={np.linalg.norm(phi - phi0):.1f} {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    if a.out:
        np.save(a.out, phi)


if __name__ == "__main__":
    main()
