"""Comprehensive 2D fold-correction benchmark over ALL B0039 slices.

For every folded slice, crop a 22x22 patch around the worst central-Jdet fold (the
identical field for every run), then for each combination of
  target metric  in {central-jdet, finite-jdet, 2tri}   (what we optimise)
  solver method  in {isqp-osqp, scipy-slsqp, scipy-trust-constr, slsqp+trust}
  objective      in {l1, l2}
solve the crop to (target >= threshold) and score the RESULT under ALL THREE metrics
(cross-metric fold counts), plus feasibility in the target metric, runtime, and the
L1/L2 correction footprint. Rows stream to a CSV; a report.md aggregates.

Full-slice windowed correction is infeasible at this breadth (2-tri z=0 alone is
~2.7 h), so this isolates the solver/metric behaviour on each slice's worst fold.
Slice tasks run in parallel; the volume is memmapped and only the small crop is
handed to each worker.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Cap BLAS/OpenMP threads BEFORE numpy/scipy/osqp import so the per-worker solvers
# don't oversubscribe cores and crash the pool (BrokenProcessPool) on Windows spawn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_jdet as fj  # noqa: E402
import slsqp_variants as sv  # noqa: E402
import windowed_isqp as wi  # noqa: E402

from dvfopt.constraints import JdetConstraint2D, TriConstraint2D  # noqa: E402
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d  # noqa: E402

DEFAULT_VOL = "data/dvfs/b0039/b0039_laplacian_deformation_field.npy"
THR = 0.01
METRICS = ("central", "finite", "2tri")
METHODS = ("isqp-osqp", "scipy-slsqp", "scipy-trust-constr", "scipy-slsqp+trust-constr")
OBJECTIVES = ("l2", "l1")


def _constraint(metric, shape):
    if metric == "central":
        return JdetConstraint2D(shape=shape)
    if metric == "finite":
        return fj.FiniteJdetConstraint2D(shape)
    if metric == "2tri":
        return TriConstraint2D(shape=shape)
    raise ValueError(metric)


def _min_metric(metric, phi):
    """Min constraint value of a (2,H,W) [dy,dx] field under *metric* (folds < thr)."""
    if metric == "central":
        return float(_numpy_jdet_2d(phi[0], phi[1]).min())
    if metric == "finite":
        return fj._min_finite_jdet(phi)
    if metric == "2tri":
        m = wi.min_field("2tri", phi)
        return float(m[: phi.shape[1] - 1, : phi.shape[2] - 1].min())
    raise ValueError(metric)


def _folds_metric(metric, phi):
    if metric == "central":
        return int((_numpy_jdet_2d(phi[0], phi[1]) < THR).sum())
    if metric == "finite":
        c = fj.FiniteJdetConstraint2D(phi.shape[1:])
        return int((np.asarray(c.values(c.flatten(phi))) < THR).sum())
    if metric == "2tri":
        m = wi.min_field("2tri", phi)
        return int((m[: phi.shape[1] - 1, : phi.shape[2] - 1] < THR).sum())
    raise ValueError(metric)


def _sparse_jac(metric, c, shape):
    """A cons_jac(f) -> sparse CSR for the given metric (coloured for jdet/2tri,
    analytic for finite)."""
    if metric == "finite":
        return lambda f: c.jacobian(f)
    fam = "jdet" if metric == "central" else "2tri"
    coloring = wi._cached_coloring(fam, c, shape)
    return lambda f: sv.colored_jacobian(c, f, *coloring).tocsr()


def solve_crop(crop, metric, method, objective, maxiter=200, eps=1e-2):
    """Solve the crop under *metric* with *method* / *objective*; return corrected
    (2,H,W) field + solve time."""
    ph, pw = crop.shape[1:]
    c = _constraint(metric, (ph, pw))
    flat0 = np.asarray(c.flatten(crop), dtype=np.float64)
    jac_sp = _sparse_jac(metric, c, (ph, pw))

    def cons(f):
        return np.asarray(c.values(f)) - THR

    obj, grad, hess = wi._objective_fns(flat0, objective, eps)

    def infeasible(x):
        return bool((np.asarray(c.values(x)) < 0.0).any())

    t = time.perf_counter()
    if method == "isqp-osqp":
        x, _, _ = sv._isqp_solve_osqp(
            flat0, cons, jac_sp, grad, maxiter, constraint=None, obj=obj, hess_diag=hess
        )
    else:
        from scipy.optimize import NonlinearConstraint, minimize

        def slsqp():
            r = minimize(
                obj,
                flat0,
                jac=grad,
                method="SLSQP",
                constraints=[{"type": "ineq", "fun": cons, "jac": lambda f: jac_sp(f).toarray()}],
                options={"maxiter": maxiter, "ftol": 1e-8},
            )
            return r.x

        def trust():
            nlc = NonlinearConstraint(cons, 0.0, np.inf, jac=jac_sp)
            r = minimize(
                obj,
                flat0,
                jac=grad,
                method="trust-constr",
                constraints=[nlc],
                options={"maxiter": maxiter, "gtol": 1e-8, "xtol": 1e-10},
            )
            return r.x

        if method == "scipy-slsqp":
            x = slsqp()
        elif method == "scipy-trust-constr":
            x = trust()
        elif method == "scipy-slsqp+trust-constr":
            x = slsqp()
            if infeasible(x):
                x2 = trust()
                if np.asarray(c.values(x2)).min() > np.asarray(c.values(x)).min():
                    x = x2
        else:
            raise ValueError(method)
    dt = time.perf_counter() - t
    return np.asarray(c.unflatten(x)), dt


CSV_FIELDS = [
    "z",
    "metric",
    "method",
    "objective",
    "central_before",
    "finite_before",
    "tri_before",
    "central_after",
    "finite_after",
    "tri_after",
    "target_min_after",
    "feasible",
    "l1_move",
    "l2_move",
    "time_s",
]


def _run_slice(z, dy, dx, size, maxiter):
    """Worker: one slice -> a list of record dicts (all metric/method/objective combos)."""
    jac = _numpy_jdet_2d(dy, dx)
    yy, xx = np.unravel_index(int(np.argmin(jac)), jac.shape)
    y0 = int(np.clip(yy - size // 2, 0, jac.shape[0] - size))
    x0 = int(np.clip(xx - size // 2, 0, jac.shape[1] - size))
    crop = np.stack([dy[y0 : y0 + size, x0 : x0 + size], dx[y0 : y0 + size, x0 : x0 + size]])
    before = {m: _folds_metric(m, crop) for m in METRICS}
    rows = []
    for metric in METRICS:
        for method in METHODS:
            for objective in OBJECTIVES:
                try:
                    out, dt = solve_crop(crop, metric, method, objective, maxiter=maxiter)
                    move = out - crop
                    rows.append(
                        {
                            "z": z,
                            "metric": metric,
                            "method": method,
                            "objective": objective,
                            "central_before": before["central"],
                            "finite_before": before["finite"],
                            "tri_before": before["2tri"],
                            "central_after": _folds_metric("central", out),
                            "finite_after": _folds_metric("finite", out),
                            "tri_after": _folds_metric("2tri", out),
                            "target_min_after": round(_min_metric(metric, out), 5),
                            "feasible": int(_min_metric(metric, out) >= 0.0),
                            "l1_move": round(float(np.abs(move).sum()), 3),
                            "l2_move": round(float(np.linalg.norm(move)), 4),
                            "time_s": round(dt, 2),
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "z": z,
                            "metric": metric,
                            "method": method,
                            "objective": objective,
                            "feasible": -1,
                            "time_s": -1,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
    return rows


def write_report(path, rows, cfg):
    ok = [r for r in rows if r.get("feasible", -1) != -1]
    n_slices = len({r["z"] for r in rows})
    lines = [
        "# Comprehensive 2D fold-correction benchmark — B0039",
        "",
        f"Volume `{cfg['vol']}` | slices with folds: {n_slices} | crop {cfg['size']}x{cfg['size']} "
        f"around worst central-Jdet fold | maxiter {cfg['maxiter']} | threshold {THR}",
        "",
        "Each result is scored under ALL THREE metrics (central-Jdet / finite-Jdet / 2-tri fold "
        "counts). `feasible` = the OPTIMISED (target) metric reached >= 0.",
        "",
        "## Feasibility and runtime, per (target metric, method, objective)",
        "",
        "| target metric | method | objective | feasible | median time | median folds after "
        "(central / finite / 2tri) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        for method in METHODS:
            for objective in OBJECTIVES:
                g = [
                    r
                    for r in ok
                    if r["metric"] == metric
                    and r["method"] == method
                    and r["objective"] == objective
                ]
                if not g:
                    continue
                feas = sum(r["feasible"] for r in g)
                med_t = float(np.median([r["time_s"] for r in g]))
                mc = int(np.median([r["central_after"] for r in g]))
                mf = int(np.median([r["finite_after"] for r in g]))
                mt = int(np.median([r["tri_after"] for r in g]))
                lines.append(
                    f"| {metric} | {method} | {objective} | {feas}/{len(g)} | "
                    f"{med_t:.2f}s | {mc} / {mf} / {mt} |"
                )
    lines += [
        "",
        "## Reading it",
        "",
        "- **feasible** counts crops where the target metric cleared. The cross-metric "
        "columns show what optimising one metric leaves under the others — e.g. optimising "
        "central-Jdet typically leaves large 2-tri residual (central-Jdet is checkerboard-"
        "blind), whereas optimising 2-tri clears the real geometry.",
        "- Compare methods within a (metric, objective) block for the speed/robustness "
        "trade-off (isqp-osqp fastest; slsqp+trust-constr most robust per second).",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vol", default=DEFAULT_VOL)
    ap.add_argument("--stride", type=int, default=1, help="1 = every slice")
    ap.add_argument("--size", type=int, default=22)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="stop after N folded slices (0 = all)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out_dir = Path(
        a.out_dir
        or Path("benchmarks/output/comprehensive_2d") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = np.load(a.vol, mmap_mode="r")
    D = vol.shape[1]
    print(
        f"volume {a.vol} shape={vol.shape} | metrics={METRICS} methods={METHODS} "
        f"objectives={OBJECTIVES} workers={a.workers} out={out_dir}",
        flush=True,
    )

    # find folded slices
    folded = []
    for z in range(0, D, a.stride):
        dy = np.asarray(vol[1, z], dtype=np.float64)
        dx = np.asarray(vol[2, z], dtype=np.float64)
        if int((_numpy_jdet_2d(dy, dx) < THR).sum()) > 0:
            folded.append((z, dy, dx))
            if a.limit and len(folded) >= a.limit:
                break
    print(
        f"{len(folded)} folded slices x {len(METRICS) * len(METHODS) * len(OBJECTIVES)} combos "
        f"= {len(folded) * len(METRICS) * len(METHODS) * len(OBJECTIVES)} solves",
        flush=True,
    )

    rows = []
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run_slice, z, dy, dx, a.size, a.maxiter): z for (z, dy, dx) in folded}
        for fut in as_completed(futs):
            z = futs[fut]
            try:
                rows.extend(fut.result())
            except Exception as e:
                print(f"  z={z} slice FAILED: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 10 == 0 or done == len(folded):
                el = time.perf_counter() - t0
                eta = el / done * (len(folded) - done)
                print(
                    f"  {done}/{len(folded)} slices | {el / 60:.1f} min elapsed | "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["z"], r["metric"], r["method"], r["objective"])):
            w.writerow(r)
    write_report(out_dir / "report.md", rows, {"vol": a.vol, "size": a.size, "maxiter": a.maxiter})
    print(f"\nrows: {len(rows)} | out dir -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
