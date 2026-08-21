"""Full-slice multi-inner fold-correction benchmark over a subset of B0039.

For every N-th folded slice of the B0039 Laplacian field, run
:func:`windowed_isqp.windowed_correct` for the full cross of **3 inner solvers x 3
constraint families x 2 objectives**, and record folds cleared, the no-damage
invariant (``damage``), giant/mop counts, wall time, AND the corrected field's fold
count re-scored under ALL three metrics (cross-metric scoring: does clearing the
central-diff Jdet also clear the forward-diff / 2-tri folds, and vice versa?).

- inners:     ``isqp-osqp`` (default OSQP-SQP), ``scipy-slsqp``, ``scipy-slsqp+trust-constr``
- families:   ``jdet`` (central-diff Jdet), ``finite`` (forward-diff Jdet), ``2tri`` (min triangle area)
- objectives: ``l2`` (min L2 move), ``l1`` (sparse move)

Slice tasks are parallelised across processes: the main process mmaps the ~0.86 GB
volume once and hands each worker only the small ``(2, H, W)`` per-slice array, so
the volume is never pickled. ``--workers 1`` runs serially for debugging.

Usage:
    python -u benchmarks/fullslice_bench.py --stride 32 --maxiter 800
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# Cap BLAS/OpenMP threads BEFORE importing numpy — this repo hit BrokenProcessPool on
# Windows spawn when the math libraries' own thread pools fought the process pool.
# setdefault so an explicit outer env setting still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import windowed_isqp as wi  # noqa: E402

DEFAULT_VOL = "data/dvfs/b0039/b0039_laplacian_deformation_field.npy"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

INNERS = ["isqp-osqp", "scipy-slsqp", "scipy-slsqp+trust-constr"]
FAMILIES = ["jdet", "finite", "2tri"]  # jdet == central-diff Jdet
OBJECTIVES = ["l2", "l1"]

# CSV column order (also the record-dict key order).
CSV_FIELDS = [
    "z",
    "family",
    "inner",
    "objective",
    "folds_before",
    "folds_after",
    "damage",
    "central_after",
    "finite_after",
    "tri_after",
    "giant_regions",
    "mop_windows",
    "rounds",
    "time_s",
]


def _slice_phi(vol, z):
    """Extract the ``(2, H, W)`` float64 ``[dy, dx]`` slice; volume stays float32."""
    return np.stack([vol[1, z], vol[2, z]]).astype(np.float64)


def _cross_metric_folds(phi_out, threshold):
    """Fold counts of a corrected field under all three metrics (cross-scoring).

    Jdet is a full ``(H, W)`` pixel map; the cell-grid metrics (finite/2tri) pad the
    last row/col with ``+inf`` (no cell there) so those are sliced off before counting.
    """
    H, W = phi_out.shape[1:]
    central = int((wi.min_field("jdet", phi_out) < threshold).sum())
    finite = int((wi.min_field("finite", phi_out)[: H - 1, : W - 1] < threshold).sum())
    tri = int((wi.min_field("2tri", phi_out)[: H - 1, : W - 1] < threshold).sum())
    return central, finite, tri


def _rec(z, family, inner, objective, rep, cross):
    """Flatten a :class:`windowed_isqp.SliceReport` + cross-scores into a record dict."""
    central, finite, tri = cross
    return {
        "z": z,
        "family": family,
        "inner": inner,
        "objective": objective,
        "folds_before": rep.folds_before,
        "folds_after": rep.folds_after,
        "damage": rep.damage,
        "central_after": central,
        "finite_after": finite,
        "tri_after": tri,
        "giant_regions": rep.giant_regions,
        "mop_windows": rep.mop_windows,
        "rounds": rep.rounds,
        "time_s": rep.time_s,
    }


def _err_rec(z, family, inner, objective, msg):
    """A sentinel record for a failed task (``folds_after == -1``)."""
    d = {k: -1 for k in CSV_FIELDS}
    d["z"], d["family"], d["inner"], d["objective"] = z, family, inner, objective
    d["error"] = msg
    return d


def _run_task(z, family, inner, objective, phi, threshold, maxiter):
    """Worker: correct one slice under one (family, inner, objective), cross-score the
    result, return a plain record dict.

    Top-level so it survives the Windows spawn pickle. Receives only the small
    ``(2, H, W)`` float64 slice, never the volume. Any solver failure is caught and
    returned as an error record so one bad task never kills the run.
    """
    try:
        phi_out, rep = wi.windowed_correct(
            phi,
            family=family,
            objective=objective,
            inner=inner,
            threshold=threshold,
            maxiter=maxiter,
            z=z,
        )
        cross = _cross_metric_folds(phi_out, threshold)
        return _rec(z, family, inner, objective, rep, cross)
    except Exception as e:  # deliberately never let one bad task crash the whole run
        return _err_rec(z, family, inner, objective, f"{type(e).__name__}: {e}")


def _line(d):
    """Human-readable per-result progress line."""
    tag = f"z={d['z']:>3} {d['family']}/{d['inner']}/{d['objective']}"
    if d["folds_after"] == -1:
        return f"{tag}  ERR {d.get('error', '')}"
    return (
        f"{tag}  folds {d['folds_before']}->{d['folds_after']} damage={d['damage']} "
        f"cross(c/f/t)={d['central_after']}/{d['finite_after']}/{d['tri_after']} "
        f"giants={d['giant_regions']} mop={d['mop_windows']} {d['time_s']:.1f}s"
    )


def build_tasks(vol, stride, threshold, maxiter, limit, slow_2tri=False):
    """Return ``(tasks, n_sampled)``: one task per (folded z, family, inner, objective).

    A sampled ``z`` is "folded" if ANY family folds it; unfolded slices are skipped.
    ``limit > 0`` caps the number of folded slices sampled (0 = all). Each folded z is
    read once and its ``(2, H, W)`` slice shared across that z's tasks.

    Unless ``slow_2tri`` is set, the ``2tri`` family runs ONLY the isqp-osqp inner:
    the scipy inners solve a dense reduced sub-problem per window and 2-tri has many
    windows + tiling + mop, so 2tri x scipy-inner on dense slices is intractable
    (days). jdet/finite (few, small windows) run all three inners.
    """
    d = vol.shape[1]
    sampled = list(range(0, d, stride))
    tasks = []
    n_folded = 0
    for z in sampled:
        phi = _slice_phi(vol, z)
        fam_folded = [f for f in FAMILIES if wi.pixel_fold_mask(f, phi, threshold).any()]
        if not fam_folded:
            print(f"z={z:>3}: 0 folds before -> skip")
            continue
        n_folded += 1
        if limit > 0 and n_folded > limit:
            break
        for family in fam_folded:
            for inner in INNERS:
                if family == "2tri" and inner != "isqp-osqp" and not slow_2tri:
                    continue
                for objective in OBJECTIVES:
                    tasks.append((z, family, inner, objective, phi, threshold, maxiter))
    return tasks, len(sampled)


def run_tasks(tasks, workers):
    """Run tasks (serial if ``workers == 1``, else a process pool), printing each
    result as it lands. Returns the collected record dicts."""
    records = []
    if workers == 1:
        for t in tasks:
            d = _run_task(*t)
            records.append(d)
            print("  " + _line(d))
            sys.stdout.flush()
        return records

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_meta = {ex.submit(_run_task, *t): t[:4] for t in tasks}
        for fut in as_completed(fut_meta):
            z, family, inner, objective = fut_meta[fut]
            try:
                d = fut.result()
            except Exception as e:  # backstop if a worker process crashed outright
                d = _err_rec(z, family, inner, objective, f"{type(e).__name__}: {e}")
            records.append(d)
            print("  " + _line(d))
            sys.stdout.flush()
    return records


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda r: (r["z"], r["family"], r["inner"], r["objective"])):
            w.writerow(r)


def _med(rows, key):
    vals = [r[key] for r in rows]
    return float(np.median(vals)) if vals else float("nan")


def _group_stats(group):
    """Aggregate one (family, inner, objective) group (error rows excluded from sums)."""
    ok = [r for r in group if r["folds_after"] != -1]
    tfb = sum(r["folds_before"] for r in ok)
    tfa = sum(r["folds_after"] for r in ok)
    return {
        "n_ok": len(ok),
        "errs": len(group) - len(ok),
        "tfb": tfb,
        "tfa": tfa,
        "worst_damage": max((r["damage"] for r in ok), default=0),
        "fully": sum(1 for r in ok if r["folds_after"] == 0),
        "med_t": _med(ok, "time_s"),
        "med_central": _med(ok, "central_after"),
        "med_finite": _med(ok, "finite_after"),
        "med_tri": _med(ok, "tri_after"),
        "pct": 100.0 if tfb == 0 else (tfb - tfa) / tfb * 100.0,
    }


METHOD = (
    "Each row solves one (family, inner, objective) over the folded slices. The "
    "**target metric** is the row's own `family`; `folds_before/after` and `%cleared` "
    "are in that metric. `med_central/finite/tri` re-score the SAME corrected field "
    "under all three metrics (cross-metric fold counts) — clearing one metric need not "
    "clear the others. `worst_damage` = the max over slices of folds created OUTSIDE "
    "every window's free region; it MUST be 0 (the no-damage invariant), and is flagged "
    "loudly otherwise."
)


def write_report(path, records, cfg):
    """Write ``report.md``: run config, method, per-group table, and verdicts."""
    n_folded = len({r["z"] for r in records})
    lines = [
        "# Full-slice multi-inner fold-correction benchmark",
        "",
        f"Volume `{cfg['vol']}` | stride {cfg['stride']} | maxiter {cfg['maxiter']} | "
        f"threshold {cfg['threshold']} | sampled z: {cfg['n_sampled']} (folded: {n_folded})",
        "",
        f"Inners: {', '.join(INNERS)} | Families: {', '.join(FAMILIES)} | "
        f"Objectives: {', '.join(OBJECTIVES)}",
        "",
        "## Method",
        "",
        METHOD,
        "",
        "## Summary",
        "",
        "| family | inner | obj | slices | before | after | %cleared | worst_damage | "
        "median_time_s | med_central | med_finite | med_tri |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    verdicts = []
    for family in FAMILIES:
        for inner in INNERS:
            for objective in OBJECTIVES:
                group = [
                    r
                    for r in records
                    if r["family"] == family and r["inner"] == inner and r["objective"] == objective
                ]
                if not group:
                    continue
                s = _group_stats(group)
                wd = s["worst_damage"]
                wd_cell = str(wd) if wd == 0 else f"**INVARIANT VIOLATED ({wd})**"
                mt = "-" if np.isnan(s["med_t"]) else f"{s['med_t']:.1f}"
                lines.append(
                    f"| {family} | {inner} | {objective} | {s['n_ok']} | {s['tfb']} | "
                    f"{s['tfa']} | {s['pct']:.1f} | {wd_cell} | {mt} | "
                    f"{s['med_central']:.0f} | {s['med_finite']:.0f} | {s['med_tri']:.0f} |"
                )
                dmg = "damage=0" if wd == 0 else f"**damage={wd} — INVARIANT VIOLATED**"
                err = f", {s['errs']} errored" if s["errs"] else ""
                verdicts.append(
                    f"- {family}/{inner}/{objective}: {s['fully']}/{s['n_ok']} slices fully "
                    f"cleared, {s['tfb']}->{s['tfa']} target folds ({s['pct']:.0f}%), {dmg}, "
                    f"median {mt}s{err}"
                )

    lines += ["", "## Verdict", "", *verdicts, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vol", default=DEFAULT_VOL)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="cap number of folded slices (0 = all)")
    ap.add_argument(
        "--slow-2tri",
        action="store_true",
        help="also run 2tri x scipy inners (intractable on dense slices; days)",
    )
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out_dir = Path(
        a.out_dir or Path("benchmarks/output/fullslice") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = np.load(a.vol, mmap_mode="r")
    print(
        f"volume {a.vol} shape={vol.shape} | stride={a.stride} maxiter={a.maxiter} "
        f"workers={a.workers} threshold={a.threshold} limit={a.limit or 'all'}"
    )
    print(f"inners={INNERS} families={FAMILIES} objectives={OBJECTIVES}")
    tasks, n_sampled = build_tasks(vol, a.stride, a.threshold, a.maxiter, a.limit, a.slow_2tri)
    print(f"{len(tasks)} tasks over {n_sampled} sampled slices")
    records = run_tasks(tasks, a.workers)

    csv_path, report_path = out_dir / "results.csv", out_dir / "report.md"
    write_csv(csv_path, records)
    cfg = {
        "vol": a.vol,
        "stride": a.stride,
        "maxiter": a.maxiter,
        "n_sampled": n_sampled,
        "threshold": a.threshold,
    }
    write_report(report_path, records, cfg)
    print(f"\nrows: {len(records)} | out dir -> {out_dir}")


if __name__ == "__main__":
    main()
