"""Newton-SQP prototype harness: verify the constant row Hessians, then measure.

Two modes:

``--check``  derivation + implementation self-checks (no solving):
  1. OSQP's dual sign on a one-row QP (the convention the Lagrangian term rides on).
  2. EXHAUSTIVE comparison of the analytic constant row Hessians against second
     differences of ``constraint.values`` on a small grid, for the 2-tri and the
     bilinear families (all rows, all variable pairs).
  3. The per-row PSD projection identity (``P_+ >= 0`` and ``P_+ - lam*Hc >= 0``).
  4. :class:`benchmarks.isqp_newton.NewtonHess` assembly on a real windowed
     sub-problem: restricted, convexified, PSD, and matching the reference sum.

``--run``    baseline vs Newton on the hard crops and raw B0039 z16, through the
             stock windowed engine at its defaults (coarse_to_fine, hybrid QP
             backend, tile fit) with the bilinear constraint + ``NoneObjective``.
             Baseline runs through the SAME instrumented driver with the
             Lagrangian term switched off, so wall / iterations / ADMM counts are
             apples to apples (``--parity`` proves that path is the stock one).

Usage:
    python -u benchmarks/newton_sqp_proto.py --check
    python -u benchmarks/newton_sqp_proto.py --parity
    python -u benchmarks/newton_sqp_proto.py --run --cases z16_twist,z0_cluster
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import isqp_newton as nt  # noqa: E402

from dvfopt.constraints import SimplexConstraint2D, SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.windowed import windowed_correct  # noqa: E402
from dvfopt.core.windowed._common import build_subproblem  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402

THR = 0.01


def _repo_root() -> Path:
    """The MAIN checkout (data/ and benchmarks/output/ are gitignored and live
    only there, not in a worktree)."""
    here = Path(__file__).resolve()
    parts = here.parts
    if ".claude" in parts:
        return Path(*parts[: parts.index(".claude")])
    return here.parent.parent


ROOT = _repo_root()
CROPS = ROOT / "benchmarks" / "output" / "testcases"
RAW_VOL = ROOT / "data" / "dvfs" / "b0039" / "b0039_laplacian_deformation_field.npy"


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _check_osqp_dual_sign():
    """min 1/2 x^2 s.t. x >= 1 -> x* = 1, and OSQP's stationarity Px + q + A'y = 0
    forces y = -1: a row at its LOWER bound has a NON-POSITIVE dual."""
    import osqp
    from scipy import sparse

    p = osqp.OSQP()
    p.setup(
        sparse.csc_matrix([[1.0]]),
        np.zeros(1),
        sparse.csc_matrix([[1.0]]),
        np.array([1.0]),
        np.array([np.inf]),
        verbose=False,
    )
    r = p.solve()
    assert abs(r.x[0] - 1.0) < 1e-3, r.x
    assert r.y[0] < -0.5, f"OSQP dual sign changed: y={r.y}"
    print(f"  [1] OSQP dual sign: x*={r.x[0]:.6f}  y*={r.y[0]:+.6f}  (lower bound -> y <= 0) OK")


def _analytic_hessians(enforced_idx, ph, pw, n_var):
    """Dense ``(n_rows, n_var, n_var)`` constant Hessians from the derivation."""
    abc = nt.triangle_abc(enforced_idx, ph, pw)
    n_pix = ph * pw
    out = np.zeros((abc.shape[0], n_var, n_var))
    rows = np.arange(abc.shape[0])
    for q, p, v in nt._PAIRS:
        out[rows, abc[:, q], abc[:, p] + n_pix] += v
        out[rows, abc[:, p] + n_pix, abc[:, q]] += v
    return out


def _fd_hessians(c, x0, h=1e-2):
    """Second differences of ``constraint.values`` -- every row, every pair."""
    n = x0.size
    m = np.asarray(c.values(x0)).size
    out = np.zeros((m, n, n))
    for a in range(n):
        ea = np.zeros(n)
        ea[a] = h
        for b in range(a, n):
            eb = np.zeros(n)
            eb[b] = h
            v = (
                np.asarray(c.values(x0 + ea + eb))
                - np.asarray(c.values(x0 + ea - eb))
                - np.asarray(c.values(x0 - ea + eb))
                + np.asarray(c.values(x0 - ea - eb))
            ) / (4.0 * h * h)
            out[:, a, b] = v
            out[:, b, a] = v
    return out


def _check_row_hessians():
    rng = np.random.default_rng(0)
    for cls, name in ((SimplexConstraint2D, "2tri"), (SimplexConstraint2DBilinear, "bilinear")):
        ph, pw = 4, 5
        c = cls(shape=(ph, pw))
        x0 = rng.normal(0.0, 0.4, c.n_variables)
        an = _analytic_hessians(np.arange(c.n_constraints), ph, pw, c.n_variables)
        fd = _fd_hessians(c, x0)
        scale = max(np.abs(an).max(), 1e-12)
        rel = np.abs(fd - an).max() / scale
        assert rel < 1e-6, f"{name}: analytic vs FD Hessian rel err {rel:.2e}"
        # and the analytic pattern really is x-y only, zero diagonal
        n_pix = ph * pw
        assert np.abs(np.einsum("rii->ri", an)).max() == 0.0
        assert np.abs(an[:, :n_pix, :n_pix]).max() == 0.0
        assert np.abs(an[:, n_pix:, n_pix:]).max() == 0.0
        print(
            f"  [2] {name:9s} rows={c.n_constraints:3d} vars={c.n_variables:3d}  "
            f"max|H_fd - H_analytic| / max|H| = {rel:.3e}  (< 1e-6) OK"
        )


def _check_psd_projection():
    m = np.zeros((3, 3))
    for q, p, v in nt._PAIRS:
        m[q, p] = v
    hc = np.block([[np.zeros((3, 3)), m], [m.T, np.zeros((3, 3))]])
    ev = np.linalg.eigvalsh(hc)
    assert abs(abs(ev).max() - nt.SIG) < 1e-12, ev
    q3 = np.eye(3) - np.ones((3, 3)) / 3.0
    worst = 0.0
    for lam in (-7.3, -1.0, -0.05, 0.0, 0.6):
        hp = 0.5 * np.block(
            [[abs(lam) * nt.SIG * q3, lam * m], [lam * m.T, abs(lam) * nt.SIG * q3]]
        )
        worst = max(worst, -np.linalg.eigvalsh(hp).min(), -np.linalg.eigvalsh(hp - lam * hc).min())
    assert worst < 1e-12, worst
    print(
        f"  [3] per-row spectrum = +-{nt.SIG:.6f} (x2) and 0 (x2); "
        f"PSD projection: min eig(P+) and min eig(P+ - lam*Hc) >= {-worst:.1e} OK"
    )


def _check_assembly():
    rng = np.random.default_rng(1)
    ph, pw = 9, 11
    phi = rng.normal(0.0, 0.3, (2, ph, pw))
    c = SimplexConstraint2DBilinear(shape=(ph, pw))
    sub = build_subproblem(c, phi, (3, 6, 4, 8), THR, NoneObjective())
    from dvfopt.core.windowed._locality import _locality_of

    pc = sub.constraint  # the PATCH-shaped clone (free box + frozen ring)
    ph, pw = pc.shape
    enforced_idx, _ = _locality_of(pc).influenced(pc, sub.free_mask, ph, pw, (False,) * 4)
    an = _analytic_hessians(enforced_idx, ph, pw, sub.flat0.size)
    free = np.asarray(sub.free_idx)
    nf = free.size
    lam_raw = rng.normal(0.0, 2.0, enforced_idx.size)
    hdv = np.full(nf, 2.0)
    for mode in ("gershgorin", "psd_row"):
        nh = nt.newton_for_sub(sub, mode)
        p, tau = nh.assemble(lam_raw, hdv)
        got = np.asarray(p.todense())[:nf, :nf]
        got = got + got.T - np.diag(np.diag(got))
        lam = np.minimum(lam_raw, 0.0)
        if mode == "gershgorin":
            ref = np.einsum("r,rab->ab", lam, an)[np.ix_(free, free)] + np.diag(hdv + tau)
        else:
            n_pix = ph * pw
            m3 = np.zeros((3, 3))
            for q, pp, v in nt._PAIRS:
                m3[q, pp] = v
            q3 = np.eye(3) - np.ones((3, 3)) / 3.0
            abc = nt.triangle_abc(enforced_idx, ph, pw)
            ref = np.zeros((sub.flat0.size, sub.flat0.size))
            for k, li in enumerate(lam):
                idx = np.concatenate([abc[k], abc[k] + n_pix])
                hp = 0.5 * np.block(
                    [[abs(li) * nt.SIG * q3, li * m3], [li * m3.T, abs(li) * nt.SIG * q3]]
                )
                ref[np.ix_(idx, idx)] += hp
            ref = ref[np.ix_(free, free)] + np.diag(hdv)
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12)
        assert err < 1e-10, f"{mode}: assembly mismatch {err:.2e}"
        mn = np.linalg.eigvalsh(got).min()
        assert mn > -1e-9, f"{mode}: assembled P not PSD (min eig {mn:.2e})"
        print(
            f"  [4] {mode:10s} nf={nf} m={enforced_idx.size} nnz={nh.nnz:6d} "
            f"tau={tau:8.3f}  assembly err={err:.2e}  min eig(P)={mn:+.4f} OK"
        )


def check():
    print("Newton-SQP prototype self-checks")
    _check_osqp_dual_sign()
    _check_row_hessians()
    _check_psd_projection()
    _check_assembly()
    print("all checks passed")


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def load_case(name):
    if name == "rawz16":
        vol = np.load(RAW_VOL, mmap_mode="r")
        return np.ascontiguousarray(np.asarray(vol[1:, 16], dtype=np.float64))
    return np.load(CROPS / f"{name}.npy").astype(np.float64)


def simplex_folds(phi):
    mn = np.minimum(*_triangle_areas_2d(phi[0], phi[1]))
    return int((mn < THR).sum()), float(mn.min())


VARIANTS = {
    "baseline": dict(newton=False),
    "newton": dict(newton=True, convexify="gershgorin", lam_scale=1.0),
    "newton-damped": dict(newton=True, convexify="gershgorin", lam_scale=0.5),
    "newton-psd": dict(newton=True, convexify="psd_row", lam_scale=1.0),
    "newton-psd-damped": dict(newton=True, convexify="psd_row", lam_scale=0.5),
    # capped: the elastic formulation pins a violated row's dual at exactly -rho
    # (1e3), so the raw multipliers are big-M penalty weights, not NLP multipliers.
    "newton-cap10": dict(newton=True, convexify="gershgorin", lam_cap=10.0),
    "newton-psd-cap10": dict(newton=True, convexify="psd_row", lam_cap=10.0),
    "newton-psd-cap1": dict(newton=True, convexify="psd_row", lam_cap=1.0),
    # CONTROL, no Hessian: is the step-count win available more cheaply? A rejected
    # trust-region direction currently costs a whole extra QP solve; salvage it with
    # the legacy backtracking line search instead of discarding it.
    "ls-salvage": dict(newton=False, ls_salvage=True),
    # cap sweep + the CONTROL that separates "Newton curvature" from "a local
    # Tikhonov shift": coupling=0 keeps psd_row's diagonal blocks and drops the
    # x-y coupling, i.e. all of the regularization and NONE of the second-order
    # constraint information.
    "newton-psd-cap0.1": dict(newton=True, convexify="psd_row", lam_cap=0.1),
    "newton-psd-cap3": dict(newton=True, convexify="psd_row", lam_cap=3.0),
    "psd-cap1-nocoupling": dict(newton=True, convexify="psd_row", lam_cap=1.0, coupling=0.0),
    "psd-cap1-2xcoupling": dict(newton=True, convexify="psd_row", lam_cap=1.0, coupling=2.0),
}


def run_one(phi, variant, maxiter, budget=None):
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    unbind = nt.bind(**VARIANTS[variant])
    nt.reset_stats()
    t = time.perf_counter()
    try:
        out, rep = windowed_correct(
            phi.copy(),
            "isqp",
            constraint=c,
            objective=NoneObjective(),
            threshold=THR,
            maxiter=maxiter,
            time_budget_s=budget,
            verbose=0,
        )
    finally:
        unbind()
    wall = time.perf_counter() - t
    folds, mn = simplex_folds(out)
    return out, {
        "variant": variant,
        "wall_s": round(wall, 1),
        "fine_iters": int(sum(w.inner_iters for w in rep.windows)),
        "coarse_iters": int(rep.coarse_iters),
        "qp_solves": nt.STATS["qp_solves"],
        "admm_iters": nt.STATS["admm_iters"],
        "ip_solves": nt.STATS["ip_solves"],
        "ip_iters": nt.STATS["ip_iters"],
        "admm_per_qp": round(
            nt.STATS["admm_iters"] / max(nt.STATS["qp_solves"] - nt.STATS["ip_solves"], 1), 1
        ),
        "tau_mean": round(nt.STATS["tau_sum"] / max(nt.STATS["tau_n"], 1), 2),
        "tau_max": round(nt.STATS["tau_max"], 1),
        "n_windows": rep.n_windows,
        "rounds": rep.rounds,
        "simplex_folds": folds,
        "simplex_min": round(mn, 6),
        "bilinear_folds": rep.folds_after,
        "damage": rep.damage,
        "backend_fallbacks": rep.backend_fallbacks,
        "l2_move": round(float(np.linalg.norm(out - phi)), 1),
    }


COLS = [
    "variant",
    "wall_s",
    "fine_iters",
    "coarse_iters",
    "qp_solves",
    "admm_iters",
    "admm_per_qp",
    "ip_solves",
    "tau_mean",
    "tau_max",
    "simplex_folds",
    "bilinear_folds",
    "damage",
    "l2_move",
    "backend_fallbacks",
]


def _table(rows):
    w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in COLS}
    out = [" | ".join(k.ljust(w[k]) for k in COLS)]
    out.append("-|-".join("-" * w[k] for k in COLS))
    for r in rows:
        out.append(" | ".join(str(r[k]).ljust(w[k]) for k in COLS))
    return "\n".join(out)


MICRO_COLS = [
    "variant",
    "iters",
    "feasible",
    "max_viol",
    "exit",
    "wall_s",
    "s_per_iter",
    "qp_solves",
    "admm_iters",
    "admm_per_qp",
    "tau_mean",
]


def micro(case, variants, maxiter):
    """Per-WINDOW comparison: the same frozen-ring sub-problem, same start point,
    handed to each variant's driver directly. This is the claim under test (SQP
    step count) with no engine retry cascade on top, so a losing variant costs a
    bounded amount of time instead of an unbounded one."""
    from dvfopt.core.windowed._common import find_windows
    from dvfopt.core.windowed._locality import _locality_of, pixel_fold_mask

    phi = load_case(case)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    ring = _locality_of(c).ring
    boxes = find_windows(pixel_fold_mask(c, phi, THR), max(3, ring), ring)
    print(f"=== micro {case} shape={tuple(phi.shape[1:])} windows={len(boxes)} maxiter={maxiter}")
    for wi, box in enumerate(boxes):
        sub = build_subproblem(c, phi, box, THR, NoneObjective())
        print(
            f"\n-- window {wi}: patch={sub.constraint.shape} free={sub.free_idx.size} "
            f"rows={sub.n_enforced} worst_viol={-min(sub.cons(sub.flat0).min(), 0.0):.4f}"
        )
        rows = []
        for v in variants:
            kw = dict(VARIANTS[v])
            nh = (
                nt.newton_for_sub(
                    sub,
                    kw.get("convexify", "gershgorin"),
                    kw.get("lam_scale", 1.0),
                    kw.get("lam_cap"),
                    kw.get("coupling", 1.0),
                )
                if kw.pop("newton")
                else None
            )
            trace = {}
            nt.reset_stats()
            t = time.perf_counter()
            x, nit, ok = nt.isqp_newton_solve(
                sub.flat0,
                sub.cons,
                sub.cons_jac,
                sub.obj_grad,
                maxiter,
                obj=sub.obj,
                hess_diag=sub.hess_diag,
                free_idx=sub.free_idx,
                trace=trace,
                osqp_max_iter=2000,
                qp_backend="hybrid",
                newton=nh,
                ls_salvage=kw.get("ls_salvage", False),
            )
            wall = time.perf_counter() - t
            rows.append(
                {
                    "variant": v,
                    "iters": nit,
                    "feasible": int(ok),
                    "max_viol": round(float(max(-np.asarray(sub.cons(x)).min(), 0.0)), 6),
                    "exit": trace["exit"],
                    "wall_s": round(wall, 1),
                    "s_per_iter": round(wall / max(nit, 1), 3),
                    "qp_solves": nt.STATS["qp_solves"],
                    "admm_iters": nt.STATS["admm_iters"],
                    "admm_per_qp": round(
                        nt.STATS["admm_iters"]
                        / max(nt.STATS["qp_solves"] - nt.STATS["ip_solves"], 1),
                        1,
                    ),
                    "tau_mean": round(nt.STATS["tau_sum"] / max(nt.STATS["tau_n"], 1), 1),
                }
            )
            print("   " + json.dumps(rows[-1]), flush=True)
        w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in MICRO_COLS}
        print(" | ".join(k.ljust(w[k]) for k in MICRO_COLS))
        print("-|-".join("-" * w[k] for k in MICRO_COLS))
        for r in rows:
            print(" | ".join(str(r[k]).ljust(w[k]) for k in MICRO_COLS), flush=True)


def parity():
    """The instrumented driver with the Lagrangian term OFF must reproduce the
    stock engine bit for bit (same output, same iteration count)."""
    phi = load_case("z0_cluster")
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    kw = dict(constraint=c, objective=NoneObjective(), threshold=THR, maxiter=600, verbose=0)
    out_a, rep_a = windowed_correct(phi.copy(), "isqp", **kw)
    unbind = nt.bind(newton=False)
    try:
        out_b, rep_b = windowed_correct(phi.copy(), "isqp", **kw)
    finally:
        unbind()
    ia = sum(w.inner_iters for w in rep_a.windows)
    ib = sum(w.inner_iters for w in rep_b.windows)
    same = bool(np.array_equal(out_a, out_b))
    print(f"parity z0_cluster: stock iters={ia} patched(newton=off) iters={ib} identical={same}")
    assert ia == ib and same, "the newton=off path is NOT the stock path"


def run(cases, variants, maxiter, out_json, budget=None):
    all_rows = {}
    for name in cases:
        phi = load_case(name)
        f0, m0 = simplex_folds(phi)
        print(
            f"\n=== {name}  shape={tuple(phi.shape[1:])}  simplex folds={f0} min={m0:+.4f} ===",
            flush=True,
        )
        rows = []
        for v in variants:
            _, rec = run_one(phi, v, maxiter, budget)
            rows.append(rec)
            print("   " + json.dumps(rec), flush=True)
        all_rows[name] = rows
        print(_table(rows), flush=True)
    if out_json:
        Path(out_json).write_text(json.dumps(all_rows, indent=2))
    print("\n\n===== SUMMARY =====")
    for name, rows in all_rows.items():
        print(f"\n{name}")
        print(_table(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cases", default="z16_twist,z0_cluster,z0_sliver")
    ap.add_argument("--variants", default="baseline,newton")
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=None, help="per-run time_budget_s")
    ap.add_argument("--micro", default=None, help="per-window comparison on this case")
    a = ap.parse_args()
    if a.check:
        check()
    if a.parity:
        parity()
    if a.micro:
        micro(a.micro, a.variants.split(","), a.maxiter)
    if a.run:
        run(a.cases.split(","), a.variants.split(","), a.maxiter, a.out, a.budget)
    if not (a.check or a.parity or a.run or a.micro):
        ap.print_help()


if __name__ == "__main__":
    main()
