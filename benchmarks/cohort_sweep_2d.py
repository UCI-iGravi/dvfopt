#!/usr/bin/env python
"""Cohort 2D sweep — does the one-call windowed/bilinear recipe generalise past B0039?

The recipe under test (every engine default on ``main``)::

    windowed_correct(phi, 'isqp', constraint=SimplexConstraint2DBilinear(shape=...),
                     objective=NoneObjective(), threshold=0.01)

is run on a deterministic sample of z-slices from every brain and every variant
of the in-repo cohort (``data/dvfs/brain25_cohort_corrected/``, gitignored).
Success criterion: **0 residual 2-triangle folds, damage 0** on every slice.

Three stages, each resumable (re-running skips finished work)::

    python benchmarks/cohort_sweep_2d.py survey     # per-z fold counts + slice cache
    python benchmarks/cohort_sweep_2d.py sweep      # the solves (process pool)
    python benchmarks/cohort_sweep_2d.py summarize  # tables -> summary.json

Everything lands in ``benchmarks/output/cohort_sweep_2d/`` (gitignored).

Slice sampling is deterministic: every ``--stride``-th z plus the single most
folded z of that field. The worst slice of each field is additionally re-run with
``step_rule='tr'`` to check whether the trust-region sliver pathology (which the
exact line search was introduced to dodge) shows up on real data.

Parallelism: a ``ProcessPoolExecutor`` of ``--workers`` (4 = this box's measured
bandwidth ceiling) module-level workers, each pinned to one compute thread per
library via :func:`dvfopt.core._pool.pin_worker_threads`. Because wall-clock is
contended, SQP iteration counts are recorded as the contention-proof metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # notebooks/CLI both import benchmark_utils from here
    sys.path.insert(0, str(HERE))

import benchmark_utils as bu  # noqa: E402

THRESHOLD = 0.01
#: The ANTs warp lives inside each laplacian_* directory; both copies are near
#: identical, so the sweep takes the one from the folding-benchmark default dir.
ANTS_SRC_DIR = "laplacian_exterior"
VARIANTS = ("laplacian_all", "laplacian_exterior", "ants")

OUT = HERE / "output" / "cohort_sweep_2d"
SLICES = OUT / "slices"
COUNTS = OUT / "zcounts"
RESULTS = OUT / "results.csv"

COLUMNS = [
    "brain",
    "variant",
    "z",
    "H",
    "W",
    "step_rule",
    "folds_before",
    "folds_after",
    "min_before",
    "min_after",
    "bilinear_folds_after",
    "bilinear_min_after",
    "finite_folds_after",
    "damage",
    "rounds",
    "windows",
    "giant_regions",
    "mop_windows",
    "backend_fallbacks",
    "coarse_solve_s",
    "sqp_iters",
    "wall_s",
    "l2_move",
    "l2_move_pct",
    "pct_pixels_moved",
    "error",
]


# ---------------------------------------------------------------------------
# Field loading
# ---------------------------------------------------------------------------


def load_ants_field(brain):
    """ANTs warp -> ``(3, D, H, W)`` ``[dz, dy, dx]`` in VOXEL index units.

    ``dvfopt.io.fields.load_dvf_sitk`` cannot be reused here: it maps sitk's
    ``[dx, dy, dz]`` *physical* components straight onto numpy axes, which is
    only correct for an identity direction cosine matrix. The cohort's ANTs
    warps are stored on a permuted LPS grid (direction
    ``[[0,0,-1],[1,0,0],[0,-1,0]]``, 0.025 mm isotropic), so the physical
    displacement is rotated into the image's own index frame first::

        d_index = Dir.T @ d_phys / spacing

    The sitk array is ``(k, j, i)``-ordered; the cohort layout is ``(i, j, k)``
    (the sizes 528/320/456 are distinct, so ``i,j,k -> D,H,W`` is pinned
    uniquely by shape against the Laplacian field). Kept float32 through the
    rotation — the full volume is ~0.9 GB per copy.
    """
    import SimpleITK as sitk

    p = bu.cohort_dir() / brain / ANTS_SRC_DIR / "ants_warp_0.nii.gz"
    if not p.is_file():
        raise FileNotFoundError(f"cohort ANTs warp not found (data is gitignored): {p}")
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)  # (k, j, i, 3), components [px, py, pz] LPS
    direction = np.asarray(img.GetDirection(), dtype=np.float32).reshape(3, 3)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float32)  # along (i, j, k)
    idx = (arr @ direction) / spacing  # (..., 3) -> components (di, dj, dk)
    return np.ascontiguousarray(np.transpose(idx, (3, 2, 1, 0)))  # (3, i, j, k)


def load_field(brain, variant):
    """``(3, D, H, W)`` cohort field for a ``variant`` of ``VARIANTS``."""
    if variant == "ants":
        return load_ants_field(brain)
    return bu.load_cohort_field(brain, variant)  # reuse the canonical loader


def available_fields(brains=None, variants=None):
    """Sorted ``(brain, variant)`` pairs present on disk."""
    have = {b for b, _ in bu.list_cohort()}
    out = []
    for brain in sorted(have):
        if brains and brain not in brains:
            continue
        for variant in variants or VARIANTS:
            if variant == "ants":
                ok = (bu.cohort_dir() / brain / ANTS_SRC_DIR / "ants_warp_0.nii.gz").is_file()
            else:
                ok = (
                    bu.cohort_dir() / brain / variant / "laplacian_deformation_field.npz"
                ).is_file()
            if ok:
                out.append((brain, variant))
    return out


# ---------------------------------------------------------------------------
# Fold metrics (one definition, shared by survey and sweep)
# ---------------------------------------------------------------------------


def fold_count(constraint, phi):
    """Cell-min fold count: pixels whose constraint minimum is below threshold.

    ``pixel_fold_mask`` collapses each cell's rows (2 for 2-triangle, 4 for
    bilinear, 1 for finite) to their minimum, so this is exactly the task's
    ``min(T1, T2) < threshold`` count and is comparable across families.
    """
    from dvfopt.core.windowed import pixel_fold_mask

    return int(pixel_fold_mask(constraint, phi, THRESHOLD).sum())


def fold_min(constraint, phi):
    from dvfopt.core.windowed import min_field

    return float(min_field(constraint, phi).min())


# ---------------------------------------------------------------------------
# Stage 1 — survey: per-z fold counts + a small cache of the sampled slices
# ---------------------------------------------------------------------------


def sample_z(zcounts, stride):
    """Deterministic slice sample: every ``stride``-th z plus the most folded z."""
    picks = set(range(0, len(zcounts), stride))
    picks.add(int(np.argmax(zcounts)))
    return sorted(picks)


def survey_field(brain, variant, stride, force=False):
    """Cache per-z 2-tri fold counts and the sampled ``(2, H, W)`` slices.

    Returns ``(zcounts, sampled_z)``. Loading a full ANTs volume costs ~1 GB, so
    the volume is touched exactly once and only the sampled slices survive.
    """
    from dvfopt.constraints import make_constraint

    COUNTS.mkdir(parents=True, exist_ok=True)
    SLICES.mkdir(parents=True, exist_ok=True)
    cpath = COUNTS / f"{brain}__{variant}.npy"

    if cpath.is_file() and not force:
        zcounts = np.load(cpath)
        sampled = sample_z(zcounts, stride)
        if all((SLICES / f"{brain}__{variant}__z{z:04d}.npy").is_file() for z in sampled):
            return zcounts, sampled

    field = load_field(brain, variant)
    D, H, W = field.shape[1:]
    tri = make_constraint("simplex_standard", (H, W))
    zcounts = np.empty(D, dtype=np.int64)
    for z in range(D):
        zcounts[z] = fold_count(tri, field[1:, z].astype(np.float64))
    np.save(cpath, zcounts)

    sampled = sample_z(zcounts, stride)
    for z in sampled:
        np.save(
            SLICES / f"{brain}__{variant}__z{z:04d}.npy",
            np.ascontiguousarray(field[1:, z].astype(np.float64)),
        )
    del field
    return zcounts, sampled


# ---------------------------------------------------------------------------
# Stage 2 — sweep: one windowed/bilinear solve per sampled slice
# ---------------------------------------------------------------------------


def solve_slice(task):
    """Pool worker: run the recipe on one cached slice, return a CSV row.

    Module level + file-based script so Windows ``spawn`` can import it.
    """
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()  # OMP/OPENBLAS/MKL/NUMEXPR/NUMBA/RAYON -> 1

    from dvfopt.constraints import make_constraint
    from dvfopt.core.windowed import windowed_correct
    from dvfopt.objectives import NoneObjective

    brain, variant, z, step_rule, path, budget = task
    phi = np.load(path)
    H, W = phi.shape[1:]
    bil = make_constraint("bilinear", (H, W))
    tri = make_constraint("simplex_standard", (H, W))
    fin = make_constraint("finite", (H, W))

    row = dict.fromkeys(COLUMNS, "")
    row.update(
        brain=brain,
        variant=variant,
        z=z,
        H=H,
        W=W,
        step_rule=step_rule,
        folds_before=fold_count(tri, phi),
        min_before=round(fold_min(tri, phi), 6),
    )

    t0 = time.perf_counter()
    try:
        phi_out, rep = windowed_correct(
            phi,
            "isqp",
            constraint=bil,
            objective=NoneObjective(),
            threshold=THRESHOLD,
            step_rule=step_rule,
            time_budget_s=budget,
            verbose=0,
        )
    except Exception as exc:  # a crash on one slice must not sink the sweep
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    wall = time.perf_counter() - t0

    diff = phi_out - phi
    moved = int(np.abs(diff).max(axis=0).astype(bool).sum())
    norm0 = float(np.sqrt((phi * phi).sum()))
    l2 = float(np.sqrt((diff * diff).sum()))
    row.update(
        folds_after=fold_count(tri, phi_out),
        min_after=round(fold_min(tri, phi_out), 6),
        bilinear_folds_after=fold_count(bil, phi_out),
        bilinear_min_after=round(fold_min(bil, phi_out), 6),
        finite_folds_after=fold_count(fin, phi_out),
        damage=rep.damage,
        rounds=rep.rounds,
        windows=rep.n_windows,
        giant_regions=rep.giant_regions,
        mop_windows=rep.mop_windows,
        backend_fallbacks=rep.backend_fallbacks,
        coarse_solve_s=round(rep.coarse_solve_s, 2),
        sqp_iters=int(sum(w.inner_iters for w in rep.windows) + rep.coarse_iters),
        wall_s=round(wall, 2),
        l2_move=round(l2, 4),
        l2_move_pct=round(100.0 * l2 / norm0, 4) if norm0 else 0.0,
        pct_pixels_moved=round(100.0 * moved / (H * W), 4),
    )
    return row


def done_keys():
    """``(brain, variant, z, step_rule)`` tuples already in ``results.csv``."""
    if not RESULTS.is_file():
        return set()
    with open(RESULTS, newline="", encoding="utf-8") as f:
        return {(r["brain"], r["variant"], int(r["z"]), r["step_rule"]) for r in csv.DictReader(f)}


def build_tasks(pairs, stride, budget, tr_check=True):
    """Task list: exact_ls on every sampled slice + a tr re-run of each worst slice."""
    tasks = []
    for brain, variant in pairs:
        zcounts, sampled = survey_field(brain, variant, stride)
        worst = int(np.argmax(zcounts))
        for z in sampled:
            path = SLICES / f"{brain}__{variant}__z{z:04d}.npy"
            tasks.append((brain, variant, z, "exact_ls", str(path), budget))
        if tr_check and zcounts[worst] > 0:
            path = SLICES / f"{brain}__{variant}__z{worst:04d}.npy"
            tasks.append((brain, variant, worst, "tr", str(path), budget))
    return tasks


def append_row(row):
    new = not RESULTS.is_file()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_sweep(pairs, stride, workers, budget, tr_check=True, hardest_first=False):
    done = done_keys()
    tasks = [
        t
        for t in build_tasks(pairs, stride, budget, tr_check)
        if (t[0], t[1], t[2], t[3]) not in done
    ]
    print(f"[sweep] {len(tasks)} slice(s) to run on {workers} worker(s)", flush=True)
    if not tasks:
        return
    # Mildest first. Hardest-first minimises makespan, but this cohort's severity
    # is so skewed (B0304's worst slice is 8x any other brain's) that it also
    # buries every ordinary slice behind hours of outliers. Mildest-first answers
    # the generalisation question early and leaves the outlier tail cuttable —
    # the run is resumable, so a cut tail costs nothing but the tail.
    counts = {(b, v): np.load(COUNTS / f"{b}__{v}.npy") for b, v in pairs}
    sign = -1 if hardest_first else 1
    tasks.sort(key=lambda t: sign * counts[(t[0], t[1])][t[2]])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(solve_slice, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            append_row(row)
            tag = "ERR" if row["error"] else ("OK" if row["folds_after"] == 0 else "RESID")
            print(
                f"[{i:>3}/{len(tasks)}] {tag:<5} {row['brain']} {row['variant']:<19} "
                f"z={row['z']:<4} {row['step_rule']:<8} "
                f"{row['folds_before']} -> {row['folds_after']} folds  "
                f"dmg={row['damage']}  it={row['sqp_iters']}  {row['wall_s']}s  "
                f"[elapsed {time.perf_counter() - t0:.0f}s] {row['error']}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Stage 3 — summary
# ---------------------------------------------------------------------------


def _median(xs):
    return float(np.median(xs)) if xs else float("nan")


def summarize(budget_s=None):
    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    main = [r for r in rows if r["step_rule"] == "exact_ls"]

    groups = {}
    for r in main:
        groups.setdefault((r["brain"], r["variant"]), []).append(r)

    table, failures = [], []
    for (brain, variant), rs in sorted(groups.items()):
        ok = [r for r in rs if not r["error"]]
        before = [int(r["folds_before"]) for r in ok]
        after = [int(r["folds_after"]) for r in ok]
        table.append(
            {
                "brain": brain,
                "variant": variant,
                "slices": len(rs),
                "init_folds_min": min(before) if before else None,
                "init_folds_max": max(before) if before else None,
                "slices_zero_folds": sum(1 for a in after if a == 0),
                "worst_residual": max(after) if after else None,
                "max_damage": max(int(r["damage"]) for r in ok) if ok else None,
                "wall_median_s": round(_median([float(r["wall_s"]) for r in ok]), 1),
                "wall_max_s": round(max(float(r["wall_s"]) for r in ok), 1) if ok else None,
                "iters_median": round(_median([float(r["sqp_iters"]) for r in ok]), 1),
                "l2_move_pct_median": round(_median([float(r["l2_move_pct"]) for r in ok]), 4),
                "pct_pixels_moved_median": round(
                    _median([float(r["pct_pixels_moved"]) for r in ok]), 4
                ),
                "errors": sum(1 for r in rs if r["error"]),
            }
        )
        failures += [
            r for r in rs if r["error"] or int(r["folds_after"] or 0) or int(r["damage"] or 0)
        ]

    # A residual is only a solver verdict when the engine stopped on its own. A row
    # whose wall exceeded the per-slice budget was CUT mid-repair, which is a
    # different (and much weaker) statement — and, when the coarse-to-fine warm
    # start is on, the only way this engine reports damage != 0 at all: the
    # prolongated coarse delta is masked to the round-1 window boxes, but only the
    # windows actually solved enter `touched`, so a cut leaves moved-but-unrepaired
    # pixels outside every solved window. Classify rather than conflate.
    def _cut(r):
        return budget_s is not None and float(r["wall_s"] or 0) > budget_s

    for r in failures:
        r["outcome"] = "error" if r["error"] else ("budget_cut" if _cut(r) else "plateau")

    # tr vs exact_ls, paired on the worst slice of each field
    tr = {(r["brain"], r["variant"], r["z"]): r for r in rows if r["step_rule"] == "tr"}
    pairs = []
    for key, t in sorted(tr.items()):
        e = next((r for r in main if (r["brain"], r["variant"], r["z"]) == key), None)
        if e:
            pairs.append(
                {
                    "brain": key[0],
                    "variant": key[1],
                    "z": int(key[2]),
                    "folds_before": int(e["folds_before"]),
                    "exact_ls_folds_after": int(e["folds_after"] or -1),
                    "tr_folds_after": int(t["folds_after"] or -1),
                    "exact_ls_wall_s": float(e["wall_s"]),
                    "tr_wall_s": float(t["wall_s"]),
                    "exact_ls_iters": int(e["sqp_iters"] or 0),
                    "tr_iters": int(t["sqp_iters"] or 0),
                    "exact_ls_l2": float(e["l2_move"] or 0),
                    "tr_l2": float(t["l2_move"] or 0),
                    "tr_error": t["error"],
                    "exact_ls_error": e["error"],
                }
            )

    summary = {
        "threshold": THRESHOLD,
        "recipe": "windowed_correct(phi,'isqp',constraint=bilinear,objective=none,threshold=0.01)",
        "n_slices": len(main),
        "n_zero_folds": sum(1 for r in main if not r["error"] and int(r["folds_after"]) == 0),
        "n_damage": sum(1 for r in main if not r["error"] and int(r["damage"]) != 0),
        "budget_s": budget_s,
        "n_plateau": sum(1 for r in failures if r.get("outcome") == "plateau"),
        "n_budget_cut": sum(1 for r in failures if r.get("outcome") == "budget_cut"),
        "per_field": table,
        "not_clean": failures,
        "tr_vs_exact_ls": pairs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\n{'brain':<7} {'variant':<19} {'n':>3} {'init folds':>17} {'0-fold':>7} "
        f"{'resid':>6} {'dmg':>4} {'wall med/max':>16} {'iters':>8} {'L2%':>8} {'px%':>7}"
    )
    for t in table:
        print(
            f"{t['brain']:<7} {t['variant']:<19} {t['slices']:>3} "
            f"{str(t['init_folds_min']) + '-' + str(t['init_folds_max']):>17} "
            f"{str(t['slices_zero_folds']) + '/' + str(t['slices']):>7} "
            f"{t['worst_residual']!s:>6} {t['max_damage']!s:>4} "
            f"{str(t['wall_median_s']) + '/' + str(t['wall_max_s']):>16} "
            f"{t['iters_median']:>8} {t['l2_move_pct_median']:>8} "
            f"{t['pct_pixels_moved_median']:>7}"
        )
    print(
        f"\n{summary['n_zero_folds']}/{summary['n_slices']} slices reached 0 two-triangle "
        f"folds; damage != 0 on {summary['n_damage']}."
    )
    if failures:
        print("\nNOT CLEAN:")
        for r in failures:
            print(
                f"  {r['brain']} {r['variant']} z={r['z']} {r['step_rule']}: "
                f"{r['folds_before']} -> {r['folds_after']} folds, damage={r['damage']}, "
                f"min_before={r['min_before']}, giants={r['giant_regions']}, "
                f"{r['wall_s']}s [{r.get('outcome', '?')}] {r['error']}"
            )
    if pairs:
        print("\ntr vs exact_ls (worst slice per field):")
        for p in pairs:
            print(
                f"  {p['brain']} {p['variant']} z={p['z']}: init {p['folds_before']} | "
                f"exact_ls {p['exact_ls_folds_after']} folds / {p['exact_ls_wall_s']}s / "
                f"{p['exact_ls_iters']} it | tr {p['tr_folds_after']} folds / "
                f"{p['tr_wall_s']}s / {p['tr_iters']} it {p['tr_error']}"
            )
    return summary


# ---------------------------------------------------------------------------


def selfcheck():
    """Assert the two pieces of non-trivial local logic. Runs without cohort data."""
    # sample_z: deterministic stride sample, worst z always included, no duplicates.
    counts = np.zeros(200, dtype=np.int64)
    counts[137] = 99
    assert sample_z(counts, 64) == [0, 64, 128, 137, 192]
    counts[128] = 500  # worst coincides with a stride pick -> no duplicate row
    assert sample_z(counts, 64) == [0, 64, 128, 192]

    # fold_count: a pure shear folds every cell it touches; identity folds none.
    from dvfopt.constraints import make_constraint

    phi = np.zeros((2, 8, 8))
    tri = make_constraint("simplex_standard", (8, 8))
    assert fold_count(tri, phi) == 0
    phi[1, :, 3] = -3.0  # pull one column left past its neighbours
    assert fold_count(tri, phi) > 0

    # load_ants_field's rotation, on the cohort's exact direction/spacing: a
    # physical +y displacement is a +i (== cohort dz) index displacement.
    direction = np.array([[-0.0, -0.0, -1.0], [1.0, -0.0, -0.0], [0.0, -1.0, 0.0]])
    spacing = np.full(3, 0.025)
    assert np.allclose(np.array([0.0, 0.025, 0.0]) @ direction / spacing, [1, 0, 0])
    assert np.allclose(np.array([0.0, 0.0, -0.025]) @ direction / spacing, [0, 1, 0])
    assert np.allclose(np.array([-0.025, 0.0, 0.0]) @ direction / spacing, [0, 0, 1])
    print("selfcheck OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("survey", "sweep", "summarize", "selfcheck"))
    ap.add_argument("--brains", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None, choices=VARIANTS)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--time-budget-s",
        type=float,
        default=1800.0,
        help="per-slice engine time budget (windowed_correct time_budget_s)",
    )
    ap.add_argument("--no-tr-check", action="store_true")
    ap.add_argument(
        "--hardest-first",
        action="store_true",
        help="run the most folded slices first (minimises makespan; buries the mild ones)",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="override the cohort directory (e.g. running from a git worktree)",
    )
    args = ap.parse_args(argv)

    if args.data_root:
        root = Path(args.data_root)
        bu.cohort_dir = lambda: root  # every cohort loader routes through this

    if args.stage == "selfcheck":
        selfcheck()
        return 0

    if args.stage == "summarize":
        summarize(args.time_budget_s)
        return 0

    pairs = available_fields(args.brains, args.variants)
    if not pairs:
        print(f"no cohort fields found under {bu.cohort_dir()} (data is gitignored)")
        return 2

    if args.stage == "survey":
        for brain, variant in pairs:
            t0 = time.perf_counter()
            zcounts, sampled = survey_field(brain, variant, args.stride)
            print(
                f"{brain} {variant:<19} D={len(zcounts)} folds: total={zcounts.sum()} "
                f"max={zcounts.max()}@z{int(np.argmax(zcounts))} "
                f"nonzero_z={(zcounts > 0).sum()} | sampled {len(sampled)} slices "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        return 0

    run_sweep(
        pairs,
        args.stride,
        args.workers,
        args.time_budget_s,
        not args.no_tr_check,
        args.hardest_first,
    )
    summarize(args.time_budget_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
