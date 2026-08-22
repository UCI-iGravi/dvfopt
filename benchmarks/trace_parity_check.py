"""Tracking-parity check: the vendored traced C-SLSQP driver vs real pyslsqp.

Runs the SAME full-grid fold-correction problem (a ``crop_fold_region`` patch of a
moderate B0039 slice) through

1. ``pyslsqp.optimize`` with its HDF5 per-major-iteration recorder, and
2. ``slsqp_traced.minimize_slsqp_traced`` with ``trace=`` + ``save_x=True``
   (plus ``scipy.optimize.minimize(method='SLSQP')`` as the byte-identity anchor),

then asserts field-by-field parity. Exact per-iteration equality across the Fortran
(pyslsqp) and C (scipy>=1.15) cores is impossible on hard problems; the bar (per the
migration gate) is:

- every quantity pyslsqp records has a counterpart in our trace,
- early-iteration objective/feasibility trajectories agree to engineering precision,
- both converge to the same optimum: final-x max diff <= 1e-6, same feasibility
  verdict — and the traced driver is byte-identical to scipy's SLSQP.

Usage:
    python benchmarks/trace_parity_check.py [--z 260] [--size 28] [--maxiter 200]

Writes the HDF5 to a temp dir (never committed); prints the parity table and raises
AssertionError on failure (gate for migrating windowed_isqp's scipy inners).
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slsqp_variants as sv  # noqa: E402
from slsqp_traced import minimize_slsqp_traced  # noqa: E402

DEFAULT_VOL = "data/dvfs/b0039/b0039_laplacian_deformation_field.npy"

# pyslsqp saved variable -> our counterpart (trace field or derivation)
FIELD_MAP = [
    ("majiter", "iters[i]['it']", "per-iteration major counter"),
    ("objective", "iters[i]['obj']", "objective value at major iterate"),
    ("feasibility", "iters[i]['max_viol']", "max constraint violation"),
    ("optimality", "iters[i]['opt']", "Lagrangian-gradient norm"),
    ("x", "iters[i]['x'] (save_x=True)", "iterate snapshot"),
    ("gradient", "obj_grad(iters[i]['x'])", "derivable from saved iterate"),
    ("multipliers", "trace['multipliers'] (final)", "final KKT multipliers"),
]


def build_problem(vol_path, z, size, threshold=0.01, objective="l2"):
    vol = np.load(vol_path, mmap_mode="r")
    sl = np.asarray(vol[:, z : z + 1], dtype=np.float64)  # (3,1,H,W)
    patch = sv.crop_fold_region(sl, size=size, threshold=threshold)
    c, flat0, cons, cons_jac, obj, obj_grad, _hess, _shape = sv._problem(
        patch, threshold, objective=objective
    )
    return c, flat0, cons, cons_jac, obj, obj_grad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vol", default=DEFAULT_VOL)
    ap.add_argument("--z", type=int, default=260)
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--maxiter", type=int, default=200)
    a = ap.parse_args()

    c, flat0, cons, cons_jac, obj, obj_grad = build_problem(a.vol, a.z, a.size)
    m = c.n_constraints
    print(f"problem: z={a.z} size={a.size} -> n={flat0.size} vars, m={m} rows", flush=True)

    # ---- 1) real pyslsqp with its HDF5 recorder --------------------------------
    from pyslsqp import optimize as pyslsqp_optimize
    from pyslsqp.postprocessing import load_variables

    tmp = tempfile.mkdtemp(prefix="trace_parity_")
    h5 = os.path.join(tmp, "parity.hdf5")
    t = time.time()
    r_py = pyslsqp_optimize(
        flat0.copy(),
        obj=obj,
        grad=obj_grad,
        con=cons,
        jac=cons_jac,
        meq=0,
        maxiter=a.maxiter,
        acc=1e-8,
        iprint=0,
        save_itr="major",
        save_vars=[
            "majiter",
            "objective",
            "optimality",
            "feasibility",
            "x",
            "gradient",
            "multipliers",
        ],
        save_filename=h5,
        summary_filename=os.path.join(tmp, "summary.txt"),
    )
    t_py = time.time() - t
    saved = load_variables(
        h5, ["majiter", "objective", "optimality", "feasibility", "x"], major_only=True
    )
    x_py = np.asarray(r_py["x"])
    print(
        f"pyslsqp:      {t_py:6.1f}s  nit={r_py.get('num_majiter')}  "
        f"{len(saved['majiter'])} recorded majors",
        flush=True,
    )

    # ---- 2) traced driver (+ scipy anchor) -------------------------------------
    con_dicts = [{"type": "ineq", "fun": cons, "jac": cons_jac}]
    tr: dict = {}
    t = time.time()
    r_tr = minimize_slsqp_traced(
        obj,
        flat0.copy(),
        jac=obj_grad,
        constraints=con_dicts,
        maxiter=a.maxiter,
        ftol=1e-8,
        trace=tr,
        save_x=True,
    )
    t_tr = time.time() - t
    print(
        f"slsqp-traced: {t_tr:6.1f}s  nit={r_tr.nit}  {len(tr['iters'])} trace records", flush=True
    )

    from scipy.optimize import minimize

    t = time.time()
    r_sc = minimize(
        obj,
        flat0.copy(),
        jac=obj_grad,
        method="SLSQP",
        constraints=con_dicts,
        options={"maxiter": a.maxiter, "ftol": 1e-8},
    )
    t_sc = time.time() - t
    print(f"scipy SLSQP:  {t_sc:6.1f}s  nit={r_sc.nit}", flush=True)

    # ---- parity table -----------------------------------------------------------
    print("\n== field-by-field parity (pyslsqp saved variable -> traced counterpart) ==")
    for py_name, ours, note in FIELD_MAP:
        print(f"  {py_name:<12} -> {ours:<28} ({note})")

    # Alignment: pyslsqp's major-0 record is the INITIAL point (its feasibility/
    # optimality columns hold a 99.0 placeholder there), so pyslsqp major i+1
    # corresponds to our trace record i. Its 'feasibility' column also uses a
    # different norm than our max-violation, so we recompute feasibility with ONE
    # uniform definition (max violation via cons()) on BOTH solvers' saved iterates.
    def viol_at(xv):
        return float(np.clip(-np.asarray(cons(np.asarray(xv))), 0.0, None).max(initial=0.0))

    print("\n== trajectory comparison (aligned: pyslsqp major i+1 <-> traced it i) ==")
    k = min(5, len(saved["majiter"]) - 1, len(tr["iters"]))
    print("  it |        obj(pyslsqp)        obj(traced) | feas@x(py) feas@x(tr) |  |dx|max")
    for i in range(k):
        rec = tr["iters"][i]
        x_p = np.asarray(saved["x"][i + 1])
        dx_i = float(np.abs(x_p - rec["x"]).max())
        print(
            f"  {rec['it']:3d} | {saved['objective'][i + 1]:18.10f} "
            f"{rec['obj']:18.10f} | {viol_at(x_p):9.2e}  {viol_at(rec['x']):9.2e} | {dx_i:8.1e}"
        )
    viol_py = float(np.clip(-np.asarray(cons(x_py)), 0, None).max(initial=0.0))
    viol_tr = float(np.clip(-np.asarray(cons(r_tr.x)), 0, None).max(initial=0.0))
    dx_final = float(np.abs(x_py - r_tr.x).max())
    dx_scipy = float(np.abs(r_sc.x - r_tr.x).max())
    print(
        f"\n  final: |x_pyslsqp - x_traced|_max = {dx_final:.2e} | "
        f"viol pyslsqp={viol_py:.2e} traced={viol_tr:.2e} | "
        f"|x_scipy - x_traced|_max = {dx_scipy:.2e}"
    )

    # ---- assertions (the migration gate) ---------------------------------------
    assert np.array_equal(r_sc.x, r_tr.x) and r_sc.nit == r_tr.nit, (
        f"traced driver is NOT byte-identical to scipy SLSQP (dx={dx_scipy:.2e})"
    )
    for i in range(min(3, k)):  # early-trajectory engineering agreement (aligned i+1<->i)
        o_py, o_tr = float(saved["objective"][i + 1]), float(tr["iters"][i]["obj"])
        assert abs(o_py - o_tr) <= 1e-6 * max(1.0, abs(o_py)), (i, o_py, o_tr)
        f_py, f_tr = viol_at(saved["x"][i + 1]), viol_at(tr["iters"][i]["x"])
        assert abs(f_py - f_tr) <= 1e-6 + 1e-4 * max(f_py, f_tr), (i, f_py, f_tr)
    assert dx_final <= 1e-6, f"final-x diff {dx_final:.2e} > 1e-6"
    assert (viol_py <= 1e-8) == (viol_tr <= 1e-8), "feasibility verdicts differ"
    last = tr["iters"][-1]
    for key in ("it", "obj", "max_viol", "opt", "alpha", "line", "nfev", "ngev"):
        assert key in last, f"trace record missing {key}"
    assert "multipliers" in tr and "x" in last, "trace lacks multipliers / save_x iterates"
    print(
        "\nPARITY OK — traced driver matches pyslsqp's tracking and the optimum; "
        "byte-identical to scipy SLSQP. Migration gate PASSED."
    )


if __name__ == "__main__":
    main()
