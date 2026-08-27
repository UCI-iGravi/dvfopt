"""Bench a float32-built OSQP against the stock float64 wheel on the captured
windowed-engine QPs.

The QPs in ``benchmarks/output/qp_capture/`` are real subproblems dumped from
``dvfopt.core.primitives.isqp``: ``P{i}.npz`` / ``A{i}.npz`` are scipy CSC
(``P`` upper-triangular) and ``v{i}.npz`` holds ``q, l, u`` plus the OSQP
``x`` / ``y`` that solve was warm-started to.

Run it once per interpreter (one venv has the float64 wheel, one has a
``-DOSQP_USE_FLOAT=ON`` source build), then diff the two JSON dumps::

    <f64-python> osqp_float32_bench.py --out f64.json
    <f32-python> osqp_float32_bench.py --out f32.json
    python osqp_float32_bench.py --compare f64.json f32.json

Reported per QP and mode (cold / warm): wall time, ADMM iterations, status,
max bound violation ``max(l - Ax, Ax - u, 0)`` and the objective
``0.5 x'Px + q'x``. The engine's feasibility margin is 1e-3, so the compare
view flags any violation at or above 1e-4.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CAPTURE = Path(__file__).resolve().parent / "output" / "qp_capture"
EPS = 1e-3
MAX_ITER = 8000
VIOL_FLAG = 1e-4


def load_qp(i: int, root: Path):
    v = np.load(root / f"v{i}.npz")  # plain numeric/str arrays; no pickle needed
    return (
        sp.load_npz(root / f"P{i}.npz").tocsc(),
        np.asarray(v["q"], float),
        sp.load_npz(root / f"A{i}.npz").tocsc(),
        np.asarray(v["l"], float),
        np.asarray(v["u"], float),
        np.asarray(v["x"], float),
        np.asarray(v["y"], float),
    )


def score(P, q, A, lo, up, x):
    """Max bound violation and objective, both in float64 regardless of solver."""
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return float("inf"), float("nan")
    ax = A @ x
    viol = max(0.0, float(np.max(np.concatenate([lo - ax, ax - up]))))
    # OSQP stores P upper-triangular; rebuild the symmetric quadratic form.
    pu = sp.triu(P, k=1)
    obj = 0.5 * float(x @ (P @ x) + x @ (pu.T @ x)) + float(q @ x)
    return viol, obj


def run(root: Path, out: Path, eps: float, polish: bool) -> None:
    import osqp

    dtype = str(osqp.OSQP()._dtype.__name__)
    n = len(list(root.glob("v*.npz")))
    recs = []
    for i in range(n):
        P, q, A, lo, up, x0, y0 = load_qp(i, root)
        for mode in ("cold", "warm"):
            s = osqp.OSQP()
            s.setup(
                P,
                q,
                A,
                lo,
                up,
                verbose=False,
                warm_starting=True,
                polishing=polish,
                max_iter=MAX_ITER,
                eps_abs=eps,
                eps_rel=eps,
            )
            if mode == "warm":
                s.warm_start(x=x0.astype(s._dtype), y=y0.astype(s._dtype))
            t = time.perf_counter()
            r = s.solve(raise_error=False)
            wall = time.perf_counter() - t
            viol, obj = score(P, q, A, lo, up, r.x)
            recs.append(
                {
                    "qp": i,
                    "mode": mode,
                    "n": int(P.shape[0]),
                    "m": int(A.shape[0]),
                    "wall": wall,
                    "iter": int(r.info.iter),
                    "status": str(r.info.status).strip(),
                    # OSQP v1.0.0 returns a STALE solution buffer (zeros on a fresh
                    # solver) when polish fails, while still reporting "solved".
                    "polish": str(getattr(r.info, "status_polish", "?")),
                    "viol": viol,
                    "obj": obj,
                    "x_dtype": str(np.asarray(r.x).dtype),
                }
            )
            print(
                f"qp{i:>3} {mode:<4} {wall:7.3f}s it={r.info.iter:>5} "
                f"viol={viol:.3e} obj={obj:.8g} {recs[-1]['status']} pol={recs[-1]['polish']}",
                flush=True,
            )
    tag = f"{dtype}/eps{eps:g}/{'polish' if polish else 'nopolish'}"
    out.write_text(json.dumps({"dtype": tag, "records": recs}, indent=1))
    print(f"\nwrote {out}  ({tag}, {len(recs)} solves)")


def compare(a_path: Path, b_path: Path) -> None:
    a, b = (json.loads(p.read_text()) for p in (a_path, b_path))

    def key(r):
        return (r["qp"], r["mode"])

    ra = {key(r): r for r in a["records"]}
    rb = {key(r): r for r in b["records"]}
    for mode in ("cold", "warm"):
        ks = sorted(k for k in ra if k in rb and k[1] == mode)
        if not ks:
            continue
        print(f"\n=== {mode} ({a['dtype']} -> {b['dtype']}) ===")
        print(
            f"{'qp':>3} {'n':>6} {'m':>6} | {'t_a':>8} {'t_b':>8} {'x':>5} | "
            f"{'it_a':>5} {'it_b':>5} | {'viol_a':>9} {'viol_b':>9} | {'dobj/|obj|':>10}"
        )
        tot_a = tot_b = 0.0
        for k in ks:
            x, y = ra[k], rb[k]
            tot_a += x["wall"]
            tot_b += y["wall"]
            dob = abs(y["obj"] - x["obj"]) / max(abs(x["obj"]), 1e-12)
            flag = " *" if max(x["viol"], y["viol"]) >= VIOL_FLAG else ""
            print(
                f"{k[0]:>3} {x['n']:>6} {x['m']:>6} | {x['wall']:8.3f} {y['wall']:8.3f} "
                f"{x['wall'] / max(y['wall'], 1e-9):5.2f} | {x['iter']:>5} {y['iter']:>5} | "
                f"{x['viol']:9.2e} {y['viol']:9.2e} | {dob:10.2e}{flag}"
            )
        print(f"TOTAL {tot_a:.3f}s -> {tot_b:.3f}s  speedup {tot_a / max(tot_b, 1e-9):.3f}x")
        va = max(ra[k]["viol"] for k in ks)
        vb = max(rb[k]["viol"] for k in ks)
        print(f"max viol: {a['dtype']} {va:.3e} | {b['dtype']} {vb:.3e}  (engine margin 1e-3)")


def force_no_polish() -> None:
    """Make every OSQP solve in-process skip polishing.

    Required for the float32 build: OSQP v1.0.0's polish LDL factorisation hits
    an exactly-zero pivot on these KKT systems in single precision, and the
    failure path (``goto exit`` in ``osqp_solve``) skips ``store_solution`` — so
    the caller gets the *previous* contents of the solution buffer while
    ``info.status`` still says "solved". 45 of the 80 captured solves are hit.
    """
    import osqp

    base = osqp.OSQP

    class NoPolish(base):  # type: ignore[misc, valid-type]
        def setup(self, *a, **kw):
            kw["polishing"] = False
            return super().setup(*a, **kw)

    osqp.OSQP = NoPolish


def engine(field: Path, z: int | None, out: Path) -> None:
    """One ``windowed_correct`` run at engine defaults on a raw (2, H, W) slice."""
    import osqp

    from dvfopt.constraints import SimplexConstraint2DBilinear
    from dvfopt.core.windowed import windowed_correct
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
    from dvfopt.objectives import NoneObjective

    vol = np.load(field, mmap_mode="r")
    raw = np.array(vol[1:, z] if vol.ndim == 4 else vol, dtype=float)
    assert raw.shape[0] == 2, raw.shape
    h, w = raw.shape[1:]

    t = time.perf_counter()
    out_phi, rep = windowed_correct(
        raw,
        "isqp",
        constraint=SimplexConstraint2DBilinear(shape=(h, w)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    wall = time.perf_counter() - t

    folds = int((np.minimum(*_triangle_areas_2d(out_phi[0], out_phi[1])) < 0.01).sum())
    rec = {
        "field": str(field),
        "z": z,
        "shape": [h, w],
        "dtype": osqp.OSQP()._dtype.__name__,
        "wall": wall,
        "simplex_folds": folds,
        "folds_before": rep.folds_before,
        "folds_after": rep.folds_after,
        "damage": rep.damage,
        "l2_move": float(np.linalg.norm(out_phi - raw)),
        "sqp_iters": int(sum(x.inner_iters for x in rep.windows)) + int(rep.coarse_iters),
        "n_windows": rep.n_windows,
        "giant_regions": rep.giant_regions,
        "rounds": rep.rounds,
        "backend_fallbacks": rep.backend_fallbacks,
    }
    print(json.dumps(rec, indent=1))
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("osqp_bench.json"))
    ap.add_argument("--capture", type=Path, default=CAPTURE)
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument(
        "--engine", type=Path, help="run windowed_correct on this (2,H,W) or (3,D,H,W) .npy"
    )
    ap.add_argument("--z", type=int, default=16)
    args = ap.parse_args()
    if args.no_polish:
        force_no_polish()
    if args.compare:
        compare(*args.compare)
    elif args.engine:
        engine(args.engine, args.z, args.out)
    else:
        run(args.capture, args.out, args.eps, not args.no_polish)
