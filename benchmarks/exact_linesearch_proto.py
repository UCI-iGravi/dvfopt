"""Exact-line-search prototype harness: verify the exact 1-D model, then measure.

Modes:

``--check``  the CORRECTNESS GATE (no measurement):
  1. The quadratic line model ``c(x + a d) = c0 + a (Jd) + a^2 q(d)`` against
     direct evaluation of ``cons(x + a d)`` at random ``a``, on a real windowed
     sub-problem (rel err must be < 1e-8).
  2. The exact merit minimiser against a dense brute-force scan of the TRUE merit
     (2001 evaluations of ``cons``): the closed-form ``a*`` must be at least as
     good as every scanned point, and its model merit must equal the true merit.
  3. The maximal feasibility-preserving step ``a_max``: no currently-satisfied row
     is below target at ``a_max``, and one is just past it.

``--parity`` the patched driver at ``step_rule='tr'`` must reproduce the stock
     engine bit for bit.

``--micro``  per-WINDOW comparison (bounded cost, no engine retry cascade).

``--run``    baseline vs the line-search variants through the stock windowed
     engine at its defaults, on the hard crops and raw B0039 z16.

Usage:
    python -u benchmarks/exact_linesearch_proto.py --check
    python -u benchmarks/exact_linesearch_proto.py --parity
    python -u benchmarks/exact_linesearch_proto.py --micro z16_twist --maxiter 150 \
        --variants baseline,ls_exact,ls_cap,ls_both
    python -u benchmarks/exact_linesearch_proto.py --run --cases z16_twist
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

import isqp_exactls as xl  # noqa: E402

from dvfopt.constraints import SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.windowed import windowed_correct  # noqa: E402
from dvfopt.core.windowed._common import build_subproblem, find_windows  # noqa: E402
from dvfopt.core.windowed._locality import _locality_of, pixel_fold_mask  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import L2Objective, NoneObjective  # noqa: E402

THR = 0.01

# variant -> isqp_exactls step_rule
VARIANTS = {
    "baseline": "tr",
    "ls_exact": "exact",
    "ls_exact_tr": "exact_tr",
    "ls_exact_bail": "exact_bail",
    # same rule, table-free quadratic term (q = c(x+d) - c - J d)
    "ls_exact_bail_eval": "exact_bail",
    "ls_cap": "cap",
    "ls_both": "both",
}


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


def load_case(name):
    """A crop by name, or a raw B0039 slice as ``rawz<N>`` (``rawz16`` = the
    reference slice every earlier measurement used)."""
    if name.startswith("rawz"):
        vol = np.load(RAW_VOL, mmap_mode="r")
        return np.ascontiguousarray(np.asarray(vol[1:, int(name[4:])], dtype=np.float64))
    return np.load(CROPS / f"{name}.npy").astype(np.float64)


def simplex_folds(phi):
    mn = np.minimum(*_triangle_areas_2d(phi[0], phi[1]))
    return int((mn < THR).sum()), float(mn.min())


def _windows(phi, c):
    ring = _locality_of(c).ring
    return find_windows(pixel_fold_mask(c, phi, THR), max(3, ring), ring)


def _first_sub(case="z0_cluster", objective=None):
    phi = load_case(case)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    box = _windows(phi, c)[0]
    return build_subproblem(c, phi, box, THR, NoneObjective() if objective is None else objective)


# ---------------------------------------------------------------------------
# 1. correctness gate
# ---------------------------------------------------------------------------


def _rand_dir(sub, rng, scale):
    d = np.zeros(sub.flat0.size)
    d[sub.free_idx] = rng.normal(0.0, scale, sub.free_idx.size)
    return d


def _model(sub, line, x, d):
    """``(c0, g, q)`` of the exact quadratic line model."""
    c0 = np.asarray(sub.cons(x))
    g = np.asarray(sub.cons_jac(x) @ d)
    return c0, g, line.quad(d)


def _check_line_model():
    rng = np.random.default_rng(0)
    for case in ("z0_cluster", "z16_twist"):
        sub = _first_sub(case)
        line = xl.line_model_for_sub(sub)
        x = sub.flat0
        worst = 0.0
        for scale in (0.05, 0.5, 2.0):
            d = _rand_dir(sub, rng, scale)
            c0, g, q = _model(sub, line, x, d)
            for a in rng.uniform(-1.0, 1.0, 6):
                got = c0 + a * g + a * a * q
                ref = np.asarray(sub.cons(x + a * d))
                worst = max(worst, np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))
        assert worst < 1e-8, f"{case}: line model rel err {worst:.3e}"
        print(
            f"  [1] {case:11s} rows={sub.n_enforced:5d} free={sub.free_idx.size:5d}  "
            f"max rel err |model - cons(x+a d)| = {worst:.3e}  (< 1e-8) OK"
        )


def _true_merit(sub, x, d, w, a):
    return sub.obj(x + a * d) + float(w @ np.clip(-np.asarray(sub.cons(x + a * d)), 0, None))


def _check_exact_min(n_grid=2001):
    rng = np.random.default_rng(1)
    for case, objective in (("z0_cluster", NoneObjective()), ("z16_twist", L2Objective())):
        sub = _first_sub(case, objective)
        line = xl.line_model_for_sub(sub)
        x = sub.flat0
        w = np.full(sub.n_enforced, 1e3)
        for scale in (0.2, 1.0):
            d = _rand_dir(sub, rng, scale)
            c0, g, q = _model(sub, line, x, d)
            fco = xl.obj_line_coeffs(sub.obj, x, d)
            a_star, m_star, m_zero, n_ev = xl.exact_line_min(c0, g, q, w, fco, 1.0)
            grid = np.linspace(0.0, 1.0, n_grid)
            vals = np.array([_true_merit(sub, x, d, w, a) for a in grid])
            m_true = _true_merit(sub, x, d, w, a_star)
            span = float(vals.max() - vals.min()) + 1e-12
            # (a) the model merit at a* IS the true merit there
            assert abs(m_star - m_true) <= 1e-8 * span, (
                f"{case}: model merit {m_star:.9g} != true {m_true:.9g}"
            )
            # (b) no scanned point beats the closed-form minimiser
            assert m_true <= vals.min() + 1e-9 * span, (
                f"{case}: brute force found {vals.min():.9g} < exact {m_true:.9g}"
            )
            # (c) m(0) matches
            assert abs(m_zero - vals[0]) <= 1e-9 * span
            back = min(_true_merit(sub, x, d, w, 0.5**k) for k in range(14))
            print(
                f"  [2] {case:11s} obj={type(objective).__name__:14s} |d|={scale:4.1f} "
                f"events={n_ev:6d} a*={a_star:.5f}  m(0)={m_zero:.6g} -> m(a*)={m_star:.6g}  "
                f"(grid min {vals.min():.6g}, backtrack min {back:.6g}) OK"
            )


def _check_max_step():
    rng = np.random.default_rng(2)
    for case in ("z0_cluster", "z16_twist"):
        sub = _first_sub(case)
        line = xl.line_model_for_sub(sub)
        x = sub.flat0
        for scale in (0.2, 1.0, 3.0):
            d = _rand_dir(sub, rng, scale)
            c0, g, q = _model(sub, line, x, d)
            a_max = xl.max_feasible_step(c0, g, q, 1.0)
            sat = c0 >= 0.0
            at = np.asarray(sub.cons(x + a_max * d))[sat]
            tolr = 1e-9 * max(np.abs(c0).max(), 1.0)
            assert at.min() >= -tolr, f"{case}: a_max={a_max} breaks a row by {at.min():.3e}"
            past = ""
            if a_max < 1.0:  # just past a_max at least one satisfied row must be below
                nxt = np.asarray(sub.cons(x + min(1.0, a_max * 1.01 + 1e-9) * d))[sat]
                assert nxt.min() < 0.0, f"{case}: a_max={a_max} is not maximal"
                past = f"  (min at 1.01*a_max = {nxt.min():+.3e} < 0)"
            print(
                f"  [3] {case:11s} |d|={scale:4.1f} a_max={a_max:.6f}  "
                f"min satisfied row at a_max = {at.min():+.3e} >= 0{past} OK"
            )


def _check_table_free_q():
    """``q = c(x + d) - c(x) - J d`` must equal the constant-Hessian table."""
    rng = np.random.default_rng(3)
    for case in ("z0_cluster", "z16_twist"):
        sub = _first_sub(case)
        line = xl.line_model_for_sub(sub)
        x = sub.flat0
        for scale in (0.2, 1.0):
            d = _rand_dir(sub, rng, scale)
            g = np.asarray(sub.cons_jac(x) @ d)
            q_tab = line.quad(d)
            q_id = np.asarray(sub.cons(x + d)) - np.asarray(sub.cons(x)) - g
            rel = np.abs(q_tab - q_id).max() / max(np.abs(q_tab).max(), 1e-12)
            assert rel < 1e-10, f"{case}: table vs identity q rel err {rel:.3e}"
            print(
                f"  [4] {case:11s} |d|={scale:4.1f}  "
                f"max rel |q_table - (c(x+d) - c - J d)| = {rel:.3e}  (< 1e-10) OK"
            )


def check():
    print("Exact-line-search prototype correctness gate")
    _check_line_model()
    _check_exact_min()
    _check_max_step()
    _check_table_free_q()
    print("all checks passed")


# ---------------------------------------------------------------------------
# 2. parity
# ---------------------------------------------------------------------------


def parity():
    phi = load_case("z0_cluster")
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    kw = dict(constraint=c, objective=NoneObjective(), threshold=THR, maxiter=600, verbose=0)
    out_a, rep_a = windowed_correct(phi.copy(), "isqp", **kw)
    unbind = xl.bind("tr")
    try:
        out_b, rep_b = windowed_correct(phi.copy(), "isqp", **kw)
    finally:
        unbind()
    ia = sum(w.inner_iters for w in rep_a.windows)
    ib = sum(w.inner_iters for w in rep_b.windows)
    same = bool(np.array_equal(out_a, out_b))
    print(
        f"parity z0_cluster: stock iters={ia} patched(step_rule='tr') iters={ib} identical={same}"
    )
    assert ia == ib and same, "the step_rule='tr' path is NOT the stock path"


# ---------------------------------------------------------------------------
# 3. micro (per-window)
# ---------------------------------------------------------------------------

MICRO_COLS = [
    "variant",
    "iters",
    "feasible",
    "max_viol",
    "exit",
    "wall_s",
    "s_per_iter",
    "qp_solves",
    "admm_per_qp",
    "alpha_mean",
    "n_no_prog",
    "n_rej",
    "cap_mean",
]


def _ls_row(variant, nit, ok, viol, exitr, wall):
    ls, st = xl.LS_STATS, xl.STATS
    return {
        "variant": variant,
        "iters": nit,
        "feasible": int(ok),
        "max_viol": round(viol, 6),
        "exit": exitr,
        "wall_s": round(wall, 1),
        "s_per_iter": round(wall / max(nit, 1), 3),
        "qp_solves": st["qp_solves"],
        "admm_iters": st["admm_iters"],
        "admm_per_qp": round(st["admm_iters"] / max(st["qp_solves"] - st["ip_solves"], 1), 1),
        "alpha_mean": round(ls["alpha_sum"] / max(ls["ls_calls"], 1), 4),
        "alpha_min": round(ls["alpha_min"], 6),
        "n_alpha_full": ls["n_alpha_full"],
        "n_no_prog": ls["n_no_progress"],
        "n_rej": ls["n_rejected"],
        "cap_mean": round(ls["cap_sum"] / max(ls["cap_calls"], 1), 4),
        "n_cap_active": ls["n_cap_active"],
        "n_cap_skip": ls["n_cap_skip"],
    }


def micro(case, variants, maxiter, max_windows=None):
    phi = load_case(case)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    boxes = _windows(phi, c)
    print(f"=== micro {case} shape={tuple(phi.shape[1:])} windows={len(boxes)} maxiter={maxiter}")
    for wi, box in enumerate(boxes):
        if max_windows is not None and wi >= max_windows:
            break
        sub = build_subproblem(c, phi, box, THR, NoneObjective())
        print(
            f"\n-- window {wi}: patch={sub.constraint.shape} free={sub.free_idx.size} "
            f"rows={sub.n_enforced} worst_viol={-min(sub.cons(sub.flat0).min(), 0.0):.4f}"
        )
        rows = []
        for v in variants:
            rule = VARIANTS[v]
            line = (
                xl.line_model_for_sub(sub) if (rule != "tr" and not v.endswith("_eval")) else None
            )
            trace = {}
            xl.reset_stats()
            t = time.perf_counter()
            x, nit, ok = xl.isqp_exactls_solve(
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
                step_rule=rule,
                line=line,
            )
            wall = time.perf_counter() - t
            viol = float(max(-np.asarray(sub.cons(x)).min(), 0.0))
            rows.append(_ls_row(v, nit, ok, viol, trace["exit"], wall))
            print("   " + json.dumps(rows[-1]), flush=True)
        w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in MICRO_COLS}
        print(" | ".join(k.ljust(w[k]) for k in MICRO_COLS))
        print("-|-".join("-" * w[k] for k in MICRO_COLS))
        for r in rows:
            print(" | ".join(str(r[k]).ljust(w[k]) for k in MICRO_COLS), flush=True)


# ---------------------------------------------------------------------------
# 4. engine runs
# ---------------------------------------------------------------------------

COLS = [
    "variant",
    "wall_s",
    "fine_iters",
    "coarse_iters",
    "qp_solves",
    "admm_per_qp",
    "n_rej",
    "n_no_prog",
    "alpha_mean",
    "simplex_folds",
    "bilinear_folds",
    "damage",
    "l2_move",
    "backend_fallbacks",
]


def run_one(phi, variant, maxiter, budget=None):
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    unbind = xl.bind(VARIANTS[variant], line_model=not variant.endswith("_eval"))
    xl.reset_stats()
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
    ls, st = xl.LS_STATS, xl.STATS
    return out, {
        "variant": variant,
        "wall_s": round(wall, 1),
        "fine_iters": int(sum(w.inner_iters for w in rep.windows)),
        "coarse_iters": int(rep.coarse_iters),
        "qp_solves": st["qp_solves"],
        "admm_iters": st["admm_iters"],
        "ip_solves": st["ip_solves"],
        "admm_per_qp": round(st["admm_iters"] / max(st["qp_solves"] - st["ip_solves"], 1), 1),
        "n_rej": ls["n_rejected"],
        "n_no_prog": ls["n_no_progress"],
        "alpha_mean": round(ls["alpha_sum"] / max(ls["ls_calls"], 1), 4),
        "n_alpha_full": ls["n_alpha_full"],
        "cap_mean": round(ls["cap_sum"] / max(ls["cap_calls"], 1), 4),
        "n_cap_active": ls["n_cap_active"],
        "n_cap_skip": ls["n_cap_skip"],
        "n_windows": rep.n_windows,
        "rounds": rep.rounds,
        "simplex_folds": folds,
        "simplex_min": round(mn, 6),
        "bilinear_folds": rep.folds_after,
        "damage": rep.damage,
        "backend_fallbacks": rep.backend_fallbacks,
        "l2_move": round(float(np.linalg.norm(out - phi)), 1),
    }


def _table(rows, cols=COLS):
    w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in cols}
    out = [" | ".join(k.ljust(w[k]) for k in cols), "-|-".join("-" * w[k] for k in cols)]
    for r in rows:
        out.append(" | ".join(str(r[k]).ljust(w[k]) for k in cols))
    return "\n".join(out)


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
        if out_json:  # write incrementally: a long sweep is still useful if cut short
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
    ap.add_argument("--micro", default=None, help="per-window comparison on this case")
    ap.add_argument("--micro-windows", type=int, default=None)
    ap.add_argument("--cases", default="z16_twist,z0_cluster,z0_sliver")
    ap.add_argument("--variants", default="baseline,ls_exact,ls_cap,ls_both")
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=None, help="per-run time_budget_s")
    a = ap.parse_args()
    if a.check:
        check()
    if a.parity:
        parity()
    if a.micro:
        micro(a.micro, a.variants.split(","), a.maxiter, a.micro_windows)
    if a.run:
        run(a.cases.split(","), a.variants.split(","), a.maxiter, a.out, a.budget)
    if not (a.check or a.parity or a.run or a.micro):
        ap.print_help()


if __name__ == "__main__":
    main()
