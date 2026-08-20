"""Benchmark the windowed fold-correction solver across folded slices of B0039.

For every N-th slice of the B0039 Laplacian field that actually contains a fold,
run :func:`windowed_isqp.windowed_correct` (one small window per fold cluster,
frozen context ring) for each requested constraint family / objective, and record
folds cleared, the no-damage invariant (``damage``), giant-region count, window
count, and wall time. Rows stream to a CSV; a markdown ``report.md`` aggregates
per (family, objective) with a loud flag if the ``damage`` invariant is ever
violated.

Slice tasks are parallelised across processes: the main process loads the ~0.86 GB
volume once and hands each worker only the small ``(2, H, W)`` per-slice array, so
the volume is never pickled. ``--workers 1`` runs serially for debugging.

Usage:
    python -u benchmarks/windowed_bench.py --stride 48 --families jdet,2tri
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import windowed_isqp as wi

DEFAULT_VOL = "data/dvfs/b0039/b0039_laplacian_deformation_field.npy"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

# CSV column order (also the record-dict key order).
CSV_FIELDS = [
    "z",
    "family",
    "objective",
    "folds_before",
    "folds_after",
    "damage",
    "residual_in_window",
    "giant_regions",
    "n_windows",
    "rounds",
    "min_before",
    "min_after",
    "time_s",
]


def _rec(z, family, objective, rep):
    """Flatten a :class:`windowed_isqp.SliceReport` into a picklable record dict."""
    return {
        "z": z,
        "family": family,
        "objective": objective,
        "folds_before": rep.folds_before,
        "folds_after": rep.folds_after,
        "damage": rep.damage,
        "residual_in_window": rep.residual_in_window,
        "giant_regions": rep.giant_regions,
        "n_windows": rep.n_windows,
        "rounds": rep.rounds,
        "min_before": rep.min_before,
        "min_after": rep.min_after,
        "time_s": rep.time_s,
    }


def _err_rec(z, family, objective, msg):
    """A sentinel record for a failed task (``folds_after == -1``)."""
    d = {k: -1 for k in CSV_FIELDS}
    d["z"], d["family"], d["objective"] = z, family, objective
    d["min_before"] = d["min_after"] = float("nan")
    d["error"] = msg
    return d


def _run_task(z, family, objective, phi, threshold):
    """Worker: correct one slice and return a plain record dict.

    Top-level so it survives the Windows spawn pickle. Receives only the small
    ``(2, H, W)`` float64 slice, never the volume. Any solver failure is caught and
    returned as an error record so one bad slice never kills the run.
    """
    try:
        _, rep = wi.windowed_correct(
            phi, family=family, objective=objective, threshold=threshold, z=z
        )
        return _rec(z, family, objective, rep)
    except Exception as e:  # deliberately never let one bad slice crash the whole run
        return _err_rec(z, family, objective, f"{type(e).__name__}: {e}")


def _line(d):
    """Human-readable per-result progress line."""
    tag = f"z={d['z']:>3} {d['family']}/{d['objective']}"
    if d["folds_after"] == -1:
        return f"{tag}  ERR {d.get('error', '')}"
    return (
        f"{tag}  folds {d['folds_before']}->{d['folds_after']} "
        f"damage={d['damage']} windows={d['n_windows']} giants={d['giant_regions']} "
        f"rounds={d['rounds']} {d['time_s']:.1f}s"
    )


def _slice_phi(vol, z):
    """Extract the ``(2, H, W)`` float64 ``[dy, dx]`` slice; volume stays float32."""
    return np.stack([vol[1, z], vol[2, z]]).astype(np.float64)


def build_tasks(vol, stride, families, objectives, threshold):
    """Return ``(tasks, n_sampled)``: one task per (folded z, family, objective).

    Cheaply screens each (z, family) with ``pixel_fold_mask``; a pair with no fold
    is skipped (and noted) rather than handed to the solver.
    """
    d = vol.shape[1]
    tasks = []
    sampled = list(range(0, d, stride))
    for z in sampled:
        phi = _slice_phi(vol, z)
        for family in families:
            if int(wi.pixel_fold_mask(family, phi, threshold).sum()) == 0:
                print(f"z={z:>3} {family}: 0 folds before -> skip")
                continue
            for objective in objectives:
                tasks.append((z, family, objective, phi, threshold))
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
        fut_meta = {ex.submit(_run_task, *t): t[:3] for t in tasks}
        for fut in as_completed(fut_meta):
            z, family, objective = fut_meta[fut]
            try:
                d = fut.result()
            except Exception as e:  # backstop if a worker process crashed outright
                d = _err_rec(z, family, objective, f"{type(e).__name__}: {e}")
            records.append(d)
            print("  " + _line(d))
            sys.stdout.flush()
    return records


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda r: (r["z"], r["family"], r["objective"])):
            w.writerow(r)


def _group_stats(group):
    """Aggregate one (family, objective) group (error rows excluded from sums)."""
    ok = [r for r in group if r["folds_after"] != -1]
    tfb = sum(r["folds_before"] for r in ok)
    tfa = sum(r["folds_after"] for r in ok)
    times = [r["time_s"] for r in ok]
    return {
        "n_ok": len(ok),
        "errs": len(group) - len(ok),
        "tfb": tfb,
        "tfa": tfa,
        "worst_damage": max((r["damage"] for r in ok), default=0),
        "giants": sum(r["giant_regions"] for r in ok),
        "fully": sum(1 for r in ok if r["folds_after"] == 0),
        "med": float(np.median(times)) if times else float("nan"),
        "tot": float(sum(times)),
        "pct": 100.0 if tfb == 0 else (tfb - tfa) / tfb * 100.0,
    }


METHOD = (
    "Windowing solves one small window per fold cluster and freezes a context ring "
    "so untouched area is never modified.\n\n"
    "- Jdet uses central finite differences (np.gradient), so a patch's "
    "interior-cut edge rows use one-sided differences that disagree with the global "
    "field; the windowing enforces only the central-difference-valid rows and "
    "freezes a 2px context ring (true image borders excepted, where the global field "
    "is also one-sided). The 2-tri metric uses exact triangle areas (no "
    "finite-difference subtlety) with a 1px ring.\n"
    "- The `damage` column = folds created outside every window's free region. "
    "damage=0 across all slices is the proof that windowing never creates a fold in "
    "untouched area.\n"
)


def write_report(path, records, cfg):
    """Write ``report.md``: run config, method, per-group table, and verdicts."""
    n_folded = len({r["z"] for r in records})
    lines = [
        "# Windowed fold-correction benchmark",
        "",
        f"Volume `{cfg['vol']}` | stride {cfg['stride']} | families "
        f"{','.join(cfg['families'])} | objectives {','.join(cfg['objectives'])} | "
        f"sampled z: {cfg['n_sampled']} (folded: {n_folded})",
        "",
        "## Method",
        "",
        METHOD,
        "## Summary",
        "",
        "| family | objective | slices | total_folds_before | total_folds_after | "
        "%cleared | worst_damage | total_giant_regions | median_time_s | total_time_s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    verdicts = []
    for family in cfg["families"]:
        for objective in cfg["objectives"]:
            group = [r for r in records if r["family"] == family and r["objective"] == objective]
            if not group:
                continue
            s = _group_stats(group)
            wd = s["worst_damage"]
            wd_cell = str(wd) if wd == 0 else f"**INVARIANT VIOLATED ({wd})**"
            med = "-" if np.isnan(s["med"]) else f"{s['med']:.1f}"
            lines.append(
                f"| {family} | {objective} | {s['n_ok']} | {s['tfb']} | {s['tfa']} | "
                f"{s['pct']:.1f} | {wd_cell} | {s['giants']} | {med} | {s['tot']:.1f} |"
            )
            dmg = (
                "damage=0 across all slices" if wd == 0 else f"**damage={wd} — INVARIANT VIOLATED**"
            )
            err = f", {s['errs']} errored" if s["errs"] else ""
            verdicts.append(
                f"- {family}/{objective}: {s['fully']}/{s['n_ok']} slices, "
                f"{s['tfb']}->{s['tfa']} folds cleared ({s['pct']:.0f}%), {dmg}, "
                f"median {med}s{err}"
            )

    lines += ["", "## Verdict", "", *verdicts, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vol", default=DEFAULT_VOL)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--families", default="jdet,2tri")
    ap.add_argument("--objectives", default="l2")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    families = [f.strip() for f in a.families.split(",") if f.strip()]
    objectives = [o.strip() for o in a.objectives.split(",") if o.strip()]
    out_dir = Path(
        a.out_dir or Path("benchmarks/output/windowed") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = np.load(a.vol, mmap_mode="r")
    print(
        f"volume {a.vol} shape={vol.shape} | stride={a.stride} families={families} "
        f"objectives={objectives} workers={a.workers} threshold={a.threshold}"
    )
    tasks, n_sampled = build_tasks(vol, a.stride, families, objectives, a.threshold)
    print(f"{len(tasks)} tasks over {n_sampled} sampled slices")
    records = run_tasks(tasks, a.workers)

    csv_path, report_path = out_dir / "results.csv", out_dir / "report.md"
    write_csv(csv_path, records)
    cfg = {
        "vol": a.vol,
        "stride": a.stride,
        "families": families,
        "objectives": objectives,
        "n_sampled": n_sampled,
    }
    write_report(report_path, records, cfg)
    print(f"\nrows: {len(records)} | out dir -> {out_dir}")


if __name__ == "__main__":
    main()
