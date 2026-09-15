#!/usr/bin/env python
"""The canonical 2D benchmark: one pinned engine, one protocol, every 2D source.

Runs ``dvfopt.correct_dvf`` over a case registry spanning the repo's five 2D
sources (fold-origin artefacts, the real brain cohort, the cohort's ANTs warps,
the hard B0039 crops, synthetic cases) under a fixed set of configurations, and
writes one self-contained run directory plus the corrected DVFs.

Protocol (pre-registered — see
``docs/superpowers/notes/2026-09-11-2d-canonical-benchmark-handoff.md``):
``correct_dvf(phi, threshold=0.01, record_history=True, **config)`` at the
pinned commit, no per-case knobs, ``time_budget_s`` NOT set (the engine runs to
its own termination; ``--cap-s`` is a wall-clock the driver only *records* as
``hit_cap``). Nothing is dropped: a case that raises stays as a row with its
error text and ``feasible=False``, and every aggregate includes it. The three
hard crops are the engine's TUNING set — label them as such in every table.

Metrics per (input, output, result) — :func:`metrics`:

* the ``cohort_benchmark`` row schema, unchanged names, on the central-difference
  Jdet (``n_neg_*``, ``neg_vol_*``, ``n_clusters_*``, ``min_jdet_*``, ``l2_err``,
  ``time_s``);
* the four constraint families' per-location certificate maps
  (``dvfopt.core.windowed.min_field``) before and after, each as
  ``fold_stats``'s ``n_neg`` (``<= 0`` — the gauge at 0), ``n_below``
  (``< 0.01 - 1e-5`` — the gauge at the threshold) and ``min``. Counts are
  PER LOCATION (per cell for the simplicial families, per pixel for jdet), not
  per constraint row. The ``bilinear`` map is exactly
  ``cell_min_jdet_2d / 2`` (verified: triangle areas are half the determinant),
  i.e. the same sub-pixel certificate evaluated at the scale the solver
  constrains. ``certified`` is ``bilinear`` ``n_below == 0``;
* the registration-standard pair, before and after, over ALL pixels of the
  central Jdet: ``frac_nonpos_jdet = mean(jdet <= 0)`` and
  ``sdlogj = std(log(clip(jdet, 1e-3, None)))``. **Clipping convention:**
  Learn2Reg computes SDlogJ on the foreground; we have no masks, so this is
  whole-slice, and non-positive Jacobians (where the log is undefined) are
  clipped up to 1e-3 rather than dropped;
* move and locality (``moved_frac``, ``l1_move``, ``l2_move``, ``max_move``,
  ``mean_move_moved``);
* the IFT injectivity-radius diagnostics before and after (``ift_min_radius``,
  ``ift_frac_subpixel``) — an estimate, never a certificate;
* engine accounting from ``res.info`` on windowed rows (``damage`` — must be 0 —
  ``n_windows``, ``giant_regions``, ``mop_cleared``, ``rounds``, ``sqp_iters``
  summed over phases NOT named ``giant*``, since a giant phase is nested in its
  round and ``total_iter`` double-counts it), ``-1`` elsewhere;
* the correspondence residual on cohort slices (median / MAD before and after at
  the prescribed Laplacian boundary conditions, plus the outlier counts), ``-1``
  elsewhere.

Outputs (``<run-dir>/``): ``results.csv`` (one row per case x config),
``summary.json`` (per source x config aggregates + provenance + the gauge legend
and the sentinel / ANTs-orientation notes), ``manifest.json`` (every run with its
case, config, shape, sha256 and the INPUT's source path — inputs are not
duplicated), ``figures/`` with ``--figures``, and ``report/report.html`` from the
shared cohort writer. Corrected DVFs go to
``data/dvfs/results/<run-name>/<source>/<case>__<cfg>.npz``.

``results.csv`` and ``manifest.json`` are written INCREMENTALLY (header first,
then one flushed row per completed run), so an interrupted or crashed chain
leaves a partial but valid run directory. A run that never produced a result —
its input would not load, or its worker died — is a full ``-1`` row with
``feasible=False`` and its error text, present in the CSV, the manifest and every
denominator. ``-1`` is always a sentinel, never a measurement, and the median /
IQR aggregates skip it.

Fault tolerance (parallel pass only — the serial pass stays in-process, it is the
wall column). A worker that dies abruptly breaks the whole ``ProcessPoolExecutor``
and fails every running and queued future with ``BrokenProcessPool`` — an
infrastructure loss, not a measurement, so those pairs are never recorded as
such. Instead the first ``n_workers + 1`` unfinished pairs in submission order
(the ones a worker could have been running, plus the one pre-queued call) are
rerun ALONE, each in a fresh single-worker pool: a pair that breaks its own solo
pool gets a measured ``WorkerCrash`` row, any other gets its real result. The
rest are resubmitted to a fresh pool; past ``MAX_POOL_REBUILDS`` breaks every
remaining pair runs isolated. ``--isolate-config`` sends named configs straight
to that isolated path after the parallel pass. ``--resume RUN_DIR`` reuses every
measured row of a previous run (its DVF verified by sha256) and reruns only the
losses (missing rows, ``BrokenProcessPool`` / ``WorkerCrash`` rows, missing or
altered DVFs).

CLI::

    # smoke (no gitignored data needed)
    python benchmarks/canonical_2d.py --source synthetic --config isqp_none \\
        --sample smoke --run-dir /tmp/smoke

    # the throughput pass over a source, with figures and the markdown table
    python benchmarks/canonical_2d.py --source origins --n-workers 4 --figures --table

    # the per-case wall column the paper quotes
    python benchmarks/canonical_2d.py --source origins --serial-timing

    # rerun only what a crashed / broken-pool run lost
    python benchmarks/canonical_2d.py --source origins --n-workers 4 \\
        --resume benchmarks/output/2d_canonical_<stamp> --isolate-config slsqp_windowed

Run it from the repo root (the gitignored data resolves relative to this file).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark modules

import benchmark_utils as bu
import cohort_benchmark as cb
import correspondence_analysis as ca

import dvfopt
from dvfopt import correct_dvf
from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
)
from dvfopt.core._pool import pin_worker_threads, pinned_thread_env
from dvfopt.core.windowed import min_field
from dvfopt.io.fields import load_dvf
from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.metrics import fold_stats

REPO = Path(__file__).resolve().parents[1]
DVF_ROOT = REPO / "data" / "dvfs"
THRESHOLD = 0.01
ERR_TOL = 1e-5
SDLOGJ_CLIP = 1e-3
MOVE_EPS = 1e-9
DEFAULT_CAP_S = 7200.0

SOURCES = ("origins", "cohort", "ants", "crops", "synthetic")

#: Every value goes straight to ``correct_dvf(phi, threshold=..., record_history=True, **cfg)``.
CONFIGS = {
    "isqp_none": dict(constraint="bilinear", strategy="isqp_windowed", objective="none"),
    "isqp_l2": dict(constraint="bilinear", strategy="isqp_windowed", objective="l2"),
    "auto": dict(constraint="bilinear", strategy="auto", objective="auto"),
    "slp": dict(constraint="simplex_standard", strategy="slp", objective="l1"),
    "barrier": dict(constraint="simplex_standard", strategy="barrier", objective="l2"),
    "m14": dict(constraint="simplex_standard", strategy="m14", objective="l2"),
    "slsqp_windowed": dict(constraint="jdet", strategy="slsqp_windowed", objective="l2"),
}
#: The two engine rows run on every source; the rest of the taxonomy on the small ones.
_EVERY_SOURCE_CONFIGS = ("isqp_none", "isqp_l2")
_SMALL_SOURCES = ("origins", "crops", "synthetic")

#: ``dvf_origins`` manifest ``mechanism`` is an int; the on-disk folders carry these names.
MECHANISMS = {
    1: "m1_interpolation",
    2: "m2_dense_optimization",
    3: "m3_learned",
    4: "m4_diffeomorphic",
}

#: The cohort grid (all 7 brains). Used to reorient the ANTs warps, whose NIfTI
#: index axes are a signed permutation of it (see :func:`_ants_volume`).
COHORT_SHAPE = (528, 320, 456)
COHORT_Z_STEP = 48
#: The cohort-sweep plateau slices (CLAUDE.md) — always in the sample.
HARD_SLICES = {"B0039": (1, 2, 11, 16, 264), "B0032": (1,), "B0304": (128, 181)}
COHORT_VARIANT = "laplacian_exterior"

#: Carried in summary.json and manifest.json so nobody pairs the two sources by id.
ANTS_Z_NOTE = (
    "ANTs warps are reoriented from their own NIfTI index frame to the cohort grid; "
    "the mapping is fixed only up to a per-axis REVERSAL, so an ants_<brain>_z<k> row "
    f"may correspond to the cohort's z = {COHORT_SHAPE[0] - 1} - k. A reversal negates "
    "the axis and its displacement component together, so every Jacobian sign (and "
    "hence every metric here) is invariant; do NOT pair cohort_* and ants_* rows by z."
)
SENTINEL_NOTE = (
    "-1 is a sentinel, never a measurement: engine columns (damage, n_windows, "
    "giant_regions, mop_cleared, rounds, sqp_iters) are -1 outside the windowed "
    "engine, corr_* is -1 outside the cohort, and a run that never produced a result "
    "(load failure / dead worker) is a full -1 row with feasible=False and its error. "
    "Rates and denominators count those rows; the median/IQR aggregates skip them."
)

#: ``cohort_benchmark``'s row schema, in its existing order — these keep their names.
_SCHEMA_COLS = (
    "n_neg_init",
    "n_neg_final",
    "neg_vol_init",
    "neg_vol_final",
    "n_clusters_init",
    "n_clusters_final",
    "min_jdet_init",
    "min_jdet_final",
    "l2_err",
    "time_s",
)
_LEAD_COLS = ("label", "case", "source", "config", "mechanism", "tool", "shape", *_SCHEMA_COLS)

_FAMILIES = (
    ("simplex", SimplexConstraint2D),
    ("bilinear", SimplexConstraint2DBilinear),
    ("finite", FiniteJdetConstraint2D),
    ("jdet", JdetConstraint2D),
)

#: What each certificate column MEANS — its rows, its scale, and the two gauges.
#: Written into ``summary.json`` and above the markdown table so the numbers are
#: never read at the wrong scale (the simplicial families are triangle areas,
#: i.e. HALF the determinant: ``bilinear == cell_min_jdet_2d / 2`` exactly).
GAUGES = {
    "simplex": {
        "rows": "2 triangles per cell (fixed BL-TR diagonal)",
        "scale": "triangle area = det/2",
        "per": "cell (last row/col are +inf)",
    },
    "bilinear": {
        "rows": "4 triangles per cell (both diagonals)",
        "scale": "triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2",
        "per": "cell (last row/col are +inf)",
        "note": "the headline certificate: certified = bilinear_n_below_final == 0",
    },
    "finite": {
        "rows": "forward-difference Jdet (1 triangle per cell)",
        "scale": "determinant",
        "per": "cell (last row/col are +inf)",
    },
    "jdet": {
        "rows": "central-difference Jdet",
        "scale": "determinant",
        "per": "pixel",
    },
    "_gauges": {
        "n_neg": "count of values <= 0",
        "n_below": f"count of values < threshold - err_tol ({THRESHOLD} - {ERR_TOL})",
        "min": "minimum value, at the column's own scale",
    },
}


def _log(msg):
    print(f"[canonical2d] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One benchmark input. Picklable by construction: the loader is
    :func:`load_case` dispatching on ``source`` + ``key``, never a closure."""

    id: str
    source: str
    mechanism: str = ""
    tool: str = ""
    shape: tuple = ()
    path: str = ""  # the INPUT file, recorded in the manifest ('' for built cases)
    key: str = ""  # loader key: builder name, or 'brain:z'


def as_field(arr) -> np.ndarray:
    """Coerce a 2D deformation field to ``(3, 1, H, W)`` float64 ``[dz, dy, dx]``.

    Accepts ``(2, H, W)`` ``[dy, dx]`` (dz is set to 0), ``(3, H, W)`` and
    ``(3, 1, H, W)``. Anything else raises.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 4 and a.shape[0] == 3 and a.shape[1] == 1:
        return a.copy()
    if a.ndim == 3 and a.shape[0] == 3:
        return a[:, None].copy()
    if a.ndim == 3 and a.shape[0] == 2:
        out = np.zeros((3, 1, *a.shape[1:]), dtype=np.float64)
        out[1:, 0] = a
        return out
    raise ValueError(f"expected (2,H,W), (3,H,W) or (3,1,H,W); got {a.shape}")


def _cohort_zs(brain: str) -> list:
    """The handoff's slice sample: every 48th z, plus the named hard slices."""
    zs = set(range(0, COHORT_SHAPE[0], COHORT_Z_STEP)) | set(HARD_SLICES.get(brain, ()))
    return sorted(zs)


@lru_cache(maxsize=1)
def _cohort_volume(brain: str) -> np.ndarray:
    """The brain's Laplacian volume, cached one at a time (each is ~1 GB)."""
    return bu.load_cohort_field(brain, COHORT_VARIANT)


@lru_cache(maxsize=1)
def _ants_volume(brain: str) -> np.ndarray:
    """The brain's ANTs warp in the COHORT's index frame, cached (one at a time).

    ``load_dvf`` maps the stored physical vectors into the file's own index
    space, but not its axes: the cohort warps carry a signed-permutation
    direction, so the array comes back as ``(3, 456, 320, 528)`` while the
    Laplacian field is ``(3, 528, 320, 456)``. The three axis lengths are
    distinct, so exactly one permutation matches — apply it to the spatial axes
    and to the channels together. The mapping is determined only up to a
    per-axis reversal (a reversal negates both the axis and its component, so
    every Jacobian sign is invariant); a slice index may therefore be the
    cohort's ``D - 1 - z``, which does not affect this source's statistics (the
    ANTs warps are fold-free on every slice).
    """
    vol = load_dvf(str(DVF_ROOT / "cohort" / brain / COHORT_VARIANT / "ants_warp_0.nii.gz"))
    spatial = tuple(vol.shape[1:])
    if spatial == COHORT_SHAPE:
        return vol
    if sorted(spatial) != sorted(COHORT_SHAPE):
        raise ValueError(f"ANTs warp {spatial} is not a permutation of cohort {COHORT_SHAPE}")
    perm = [spatial.index(n) for n in COHORT_SHAPE]  # unique: the three lengths differ
    return np.transpose(vol, (0, *(p + 1 for p in perm)))[perm]


def _synthetic_cases() -> list:
    """``SYNTHETIC_CASES`` (Laplacian interpolations of correspondence pairs) +
    ``RANDOM_DVF_CASES`` (seeds in the registry) + any ``canonical_2tri_2d``
    ``.npz`` not already covered by name. All seeds are fixed in
    ``dvfopt.testdata``; the random cases carry ``seed=42``."""
    from dvfopt.testdata import RANDOM_DVF_CASES, SYNTHETIC_CASES

    out = [
        Case(
            id=k,
            source="synthetic",
            tool="Laplacian (dvfopt.laplacian)",
            shape=(3, 1, *v["resolution"]),
            key=f"syn:{k}",
        )
        for k, v in SYNTHETIC_CASES.items()
    ]
    out += [
        Case(id=k, source="synthetic", tool=f"random dvf (seed {v['seed']})", key=f"rand:{k}")
        for k, v in RANDOM_DVF_CASES.items()
    ]
    seen = {c.id for c in out}
    for p in sorted((DVF_ROOT / "canonical_2tri_2d").glob("*.npz")):
        if p.stem not in seen:
            out.append(
                Case(
                    id=p.stem,
                    source="synthetic",
                    tool="canonical_2tri_2d fixture",
                    path=str(p),
                    key=f"npz:{p}",
                )
            )
    return out


def cases(source: str, sample: str = "canonical") -> list:
    """The registry for *source*. ``sample='smoke'`` keeps the first two cases.

    Missing (gitignored) data degrades to an empty list with a warning — never a
    traceback, so a checkout without the payloads can still run the rest.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    out: list = []
    if source == "origins":
        man = DVF_ROOT / "origins" / "manifest.json"
        if not man.is_file():
            _log(f"WARNING: no origins manifest at {man} — source skipped")
        else:
            entries = json.loads(man.read_text(encoding="utf-8"))
            out = [
                Case(
                    id=cid,
                    source="origins",
                    mechanism=MECHANISMS.get(e["mechanism"], str(e["mechanism"])),
                    tool=f"{e['tool']} ({e['source']})",
                    shape=tuple(e["shape"]),
                    path=str(DVF_ROOT / "origins" / e["file"]),
                    key=f"file:{DVF_ROOT / 'origins' / e['file']}",
                )
                for cid, e in sorted(entries.items())
            ]
    elif source in ("cohort", "ants"):
        brains = sorted({b for b, v in bu.list_cohort() if v == COHORT_VARIANT})
        if not brains:
            _log(f"WARNING: no cohort brains under {bu.cohort_dir()} — source skipped")
        fname = "laplacian_deformation_field.npz" if source == "cohort" else "ants_warp_0.nii.gz"
        tool = "Laplacian (RegTools)" if source == "cohort" else "ANTs SyN"
        for b in brains:
            for z in _cohort_zs(b):
                out.append(
                    Case(
                        # source-prefixed: a cohort slice and its ANTs twin would
                        # otherwise share an id (and a manifest entry, and a label)
                        id=f"{source}_{b}_z{z}",
                        source=source,
                        tool=tool,
                        shape=(3, 1, COHORT_SHAPE[1], COHORT_SHAPE[2]),
                        path=str(DVF_ROOT / "cohort" / b / COHORT_VARIANT / fname),
                        key=f"{b}:{z}",
                    )
                )
    elif source == "crops":
        paths = sorted((DVF_ROOT / "crops").glob("*.npy"))
        if not paths:
            _log(f"WARNING: no crops under {DVF_ROOT / 'crops'} — source skipped")
        out = [
            Case(
                id=p.stem,
                source="crops",
                tool="B0039 hard crop (TUNING SET)",
                path=str(p),
                key=f"file:{p}",
            )
            for p in paths
        ]
    else:
        out = _synthetic_cases()
    if sample == "smoke":
        out = out[:2]
    elif sample != "canonical":
        raise ValueError(f"unknown sample {sample!r}; expected 'canonical' or 'smoke'")
    return out


def load_case(case: Case) -> np.ndarray:
    """Materialize a case as a ``(3, 1, H, W)`` float64 field."""
    kind, _, rest = case.key.partition(":")
    if kind == "file":
        return as_field(load_dvf(rest))
    if kind == "syn":
        from dvfopt.testdata import make_deformation

        return as_field(make_deformation(rest)[0])
    if kind == "rand":
        from dvfopt.testdata import make_random_dvf

        return as_field(make_random_dvf(rest))
    if kind == "npz":
        with np.load(rest) as z:
            return as_field(z["phi"])
    brain, z = kind, int(rest)  # 'brain:z' — cohort / ants
    vol = _cohort_volume(brain) if case.source == "cohort" else _ants_volume(brain)
    return as_field(vol[:, z])


def case_correspondences(case: Case):
    """``(mp_slice, fp_slice)`` for a cohort case, else ``None``."""
    if case.source != "cohort":
        return None
    brain, _, z = case.key.partition(":")
    mp, fp = bu.load_cohort_correspondences(brain, COHORT_VARIANT)
    if mp is None:
        return None
    mp_s, fp_s = ca.slice_correspondences(mp, fp, int(z))
    return (mp_s, fp_s) if fp_s is not None and len(fp_s) else None


# ---------------------------------------------------------------------------
# The metric block
# ---------------------------------------------------------------------------


def _jdet(phi) -> np.ndarray:
    """Central-difference Jacobian determinant of a ``(3, 1, H, W)`` field, as ``(H, W)``."""
    return np.asarray(jacobian_det2D(np.stack([phi[1, 0], phi[2, 0]]))).squeeze()


def _reg_stats(jac) -> tuple:
    """``(frac_nonpos_jdet, sdlogj)`` over all pixels — see the module docstring
    for the clipping convention."""
    return (
        float((jac <= 0).mean()),
        float(np.std(np.log(np.clip(jac, SDLOGJ_CLIP, None)))),
    )


def _certificates(phi, threshold, suffix) -> dict:
    """The four families' per-location fold statistics for one field."""
    dydx = np.ascontiguousarray(phi[1:, 0], dtype=np.float64)
    shape = dydx.shape[1:]
    out = {}
    for name, ctor in _FAMILIES:
        s = fold_stats(min_field(ctor(shape=shape), dydx), threshold, ERR_TOL)
        out[f"{name}_n_neg_{suffix}"] = s.n_neg  # values <= 0   — the gauge at 0
        out[f"{name}_n_below_{suffix}"] = s.n_below  # < thr - err — the gauge at 0.01
        out[f"{name}_min_{suffix}"] = s.min_val
    return out


#: ``SolveInfo.strategy_name`` values that mean "the ``dvfopt.core.windowed``
#: engine ran" — the only strategies whose engine columns are meaningful. An
#: ``auto`` run that resolved to it reports the resolved class name, so it counts.
WINDOWED_STRATEGY_NAMES = ("ISQPWindowedStrategy", "WindowedWrapperStrategy")

#: Engine columns: ``-1`` on every non-windowed row, and skipped by the aggregates.
ENGINE_KEYS = ("damage", "n_windows", "giant_regions", "mop_cleared", "rounds", "sqp_iters")


def _engine_stats(res) -> dict:
    """Engine accounting from ``res.info``, ``-1`` for every other strategy.

    The sentinel is unconditional outside the windowed engine: the other
    strategies DO report phases (``barrier`` logs its L-BFGS iterations,
    ``slp`` / ``m14`` log named stages with ``n_iter=0``), and summing those as
    ``sqp_iters`` / counting them as ``rounds`` would put three different
    quantities in one column.
    """
    out = dict.fromkeys(ENGINE_KEYS, -1)
    info = getattr(res, "info", None)
    if info is None or getattr(info, "strategy_name", "") not in WINDOWED_STRATEGY_NAMES:
        return out
    extras = getattr(info, "extras", {}) or {}
    for k in ("damage", "n_windows", "giant_regions", "mop_cleared"):
        if k in extras:
            out[k] = int(extras[k])
    phases = getattr(info, "phases", []) or []
    if phases:
        out["rounds"] = sum(1 for p in phases if p.name.startswith("round"))
        # a 'giant' phase is nested inside its round entry — total_iter double-counts it
        out["sqp_iters"] = int(sum(p.n_iter for p in phases if not p.name.startswith("giant")))
    return out


def _corr_stats(sec_init, sec_out, corr_pts) -> dict:
    """Correspondence-residual diagnostics for a cohort slice; ``-1`` elsewhere.

    ``analyze_slice`` reports MEAN residuals and the outlier flags; the robust
    median / MAD are recomputed from the in-grid points it returns (its own
    filtering already applied), so no filtering logic is duplicated here.
    """
    out = dict.fromkeys(
        (
            "corr_n",
            "corr_resid_med_init",
            "corr_resid_mad_init",
            "corr_resid_med_final",
            "corr_resid_mad_final",
            "corr_n_outliers",
            "corr_n_large",
            "corr_n_high_resid",
            "corr_n_incoherent",
        ),
        -1.0,
    )
    if corr_pts is None:
        return out
    info = ca.analyze_slice(sec_init, sec_out, corr_pts[0], corr_pts[1])
    if info is None:
        return out
    fy, fx = info["fy"].astype(int), info["fx"].astype(int)
    pdy = info["my"].astype(np.float64) - fy
    pdx = info["mx"].astype(np.float64) - fx
    for tag, sec in (("init", sec_init), ("final", sec_out)):
        r = np.hypot(sec[1, 0, fy, fx] - pdy, sec[2, 0, fy, fx] - pdx)
        med = float(np.median(r))
        out[f"corr_resid_med_{tag}"] = med
        out[f"corr_resid_mad_{tag}"] = float(np.median(np.abs(r - med)) * 1.4826)
    st = info["stats"]
    out["corr_n"] = float(st["n"])
    for k in ("n_outliers", "n_large", "n_high_resid", "n_incoherent"):
        out[f"corr_{k}"] = float(st[k])
    return out


def _ift_stats(phi, suffix) -> dict:
    """The quantitative-IFT injectivity-radius diagnostics (0.07 s on a 320x456
    slice). An ESTIMATE, never a certificate, and orientation-blind — read it
    beside the four certificate gauges, never instead of them. The stats' 2D
    ``cell_min_jdet`` is dropped: it is the ``bilinear`` gauge again, x2."""
    from dvfopt.metrics import injectivity_stats

    s = injectivity_stats(phi)
    return {
        f"ift_min_radius_{suffix}": s.min_radius,
        f"ift_frac_subpixel_{suffix}": s.frac_subpixel,
    }


def metrics(phi_in, phi_out, res=None, threshold: float = THRESHOLD, elapsed: float = 0.0) -> dict:
    """The full metric block for one ``(input, output, result)`` triple.

    ``res`` is the :class:`~dvfopt.solver.SolveResult` (``None`` when the solve
    raised — every metric is still computed, on the unchanged field). The
    ``cohort_benchmark`` schema keys keep their names; see the module docstring
    for the definitions.
    """
    phi_in = np.asarray(phi_in, dtype=np.float64)
    phi_out = np.asarray(phi_out, dtype=np.float64)
    ji, jf = _jdet(phi_in), _jdet(phi_out)
    diff = phi_out - phi_in
    moved = (np.abs(diff) > MOVE_EPS).any(axis=0)[0]
    frac_i, sd_i = _reg_stats(ji)
    frac_f, sd_f = _reg_stats(jf)
    l2 = float(np.linalg.norm(diff.ravel()))
    row = {
        # --- cohort_benchmark's schema, same names, central-difference Jdet ---
        "n_neg_init": int((ji < threshold).sum()),
        "n_neg_final": int((jf < threshold).sum()),
        "neg_vol_init": cb._neg_volume(ji, threshold),
        "neg_vol_final": cb._neg_volume(jf, threshold),
        "n_clusters_init": cb._n_clusters(ji, threshold),
        "n_clusters_final": cb._n_clusters(jf, threshold),
        "min_jdet_init": float(ji.min()),
        "min_jdet_final": float(jf.min()),
        "l2_err": l2,
        "time_s": float(elapsed),
        # --- registration-standard pair ---
        "frac_nonpos_jdet_init": frac_i,
        "frac_nonpos_jdet_final": frac_f,
        "sdlogj_init": sd_i,
        "sdlogj_final": sd_f,
        # --- move and locality ---
        "moved_frac": float(moved.mean()),
        "l1_move": float(np.abs(diff).sum()),
        "l2_move": l2,
        "max_move": float(np.abs(diff).max()),
        "mean_move_moved": float(np.abs(diff).sum() / max(int(moved.sum()), 1)),
        # --- solver verdict ---
        "feasible": bool(getattr(res, "feasible", False)),
    }
    row.update(_certificates(phi_in, threshold, "init"))
    row.update(_certificates(phi_out, threshold, "final"))
    row.update(_ift_stats(phi_in, "init"))
    row.update(_ift_stats(phi_out, "final"))
    row["certified"] = row["bilinear_n_below_final"] == 0
    row.update(_engine_stats(res))
    return row


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_IDENT_KEYS = ("label", "case", "source", "config", "mechanism", "tool", "shape")
_TAIL_KEYS = ("error", "timing_mode", "hit_cap", "out_path", "sha256")


def _identity(case: Case, cfg_name: str, shape: str = "") -> dict:
    return {
        "label": f"{case.id}__{cfg_name}",
        "case": case.id,
        "source": case.source,
        "config": cfg_name,
        "mechanism": case.mechanism,
        "tool": case.tool,
        "shape": shape,
    }


@lru_cache(maxsize=1)
def record_keys() -> tuple:
    """Every record key, in CSV order, derived ONCE from the real builders on a
    tiny field — so the header can be written before the first row exists and can
    never drift from what :func:`run_case` produces."""
    z = np.zeros((3, 1, 4, 4))
    seen = dict.fromkeys(_IDENT_KEYS)
    seen.update(dict.fromkeys(metrics(z, z)))
    seen.update(dict.fromkeys(_corr_stats(z, z, None)))
    seen.update(dict.fromkeys(_TAIL_KEYS))
    ordered = [c for c in _LEAD_COLS if c in seen] + [k for k in seen if k not in _LEAD_COLS]
    return tuple(ordered)


def sentinel_record(case: Case, cfg_name: str, error: str, timing_mode="throughput") -> dict:
    """A full row for a (case, config) that never produced a result — a load
    failure or a dead worker. Every metric is ``-1``, ``feasible`` is False and
    ``error`` carries the reason, so the row is in the CSV, in the manifest (with
    no DVF) and in every denominator: nothing is dropped."""
    rec = dict.fromkeys(record_keys(), -1)
    rec.update(_identity(case, cfg_name))
    rec.update(dict.fromkeys(("feasible", "certified", "hit_cap"), False))
    rec.update(error=error, timing_mode=timing_mode, out_path="", sha256="")
    return rec


def run_case(
    case: Case,
    cfg_name: str,
    phi_in=None,
    corr_pts=None,
    *,
    threshold: float = THRESHOLD,
    cap_s: float = DEFAULT_CAP_S,
    timing_mode: str = "throughput",
    out_path: str = "",
    verbose: int = 0,
) -> dict:
    """Solve one (case, config) and return its record. Module-level and picklable
    — this is the process-pool worker.

    The input is loaded HERE (``phi_in=None``, the default) so the parent never
    holds one array per queued run; pass an array to reuse a load across the
    configs of one case. Saves the corrected DVF to *out_path* (``.npz``, key
    ``arr``) when given, and never raises: a failed load or a failed solve comes
    back as a row with ``error`` set.
    """
    pin_worker_threads()
    if phi_in is None:
        try:
            phi_in = load_case(case)
            corr_pts = case_correspondences(case) if corr_pts is None else corr_pts
        except Exception as exc:
            return sentinel_record(case, cfg_name, f"{type(exc).__name__}: {exc}", timing_mode)
    phi_in = np.asarray(phi_in, dtype=np.float64)
    t0 = time.perf_counter()
    err = ""
    try:
        res = correct_dvf(
            phi_in.copy(),
            threshold=threshold,
            record_history=True,
            verbose=verbose,
            **CONFIGS[cfg_name],
        )
        phi_out = np.asarray(res.corrected, dtype=np.float64)
    except Exception as exc:  # nothing dropped — the failure is a row
        res, phi_out, err = None, phi_in.copy(), f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0

    rec = _identity(case, cfg_name, "x".join(str(n) for n in phi_in.shape))
    rec.update(metrics(phi_in, phi_out, res, threshold, elapsed))
    rec.update(_corr_stats(phi_in, phi_out, corr_pts))
    rec["error"] = err
    rec["timing_mode"] = timing_mode
    rec["hit_cap"] = bool(elapsed > cap_s)
    rec["out_path"] = ""
    rec["sha256"] = ""
    if out_path and not err:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, arr=phi_out)  # save_dvf handles .npy/sitk only
        rec["out_path"] = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
        rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return rec


#: Pool rebuilds allowed per run; past it, every remaining pair runs isolated.
MAX_POOL_REBUILDS = 5
#: TEST-ONLY: see :func:`_pool_worker`.
CRASH_ENV = "CANONICAL_2D_TEST_CRASH_CASE"
#: ``error`` prefixes that mark an infrastructure loss, not a measurement.
LOSS_PREFIXES = ("BrokenProcessPool", "WorkerCrash")


def _pool_worker(case: Case, cfg_name: str, **kw) -> dict:
    """The process-pool entry point: :func:`run_case` in a spawned worker.

    TEST-ONLY crash hook: when the env var ``CANONICAL_2D_TEST_CRASH_CASE``
    equals this pair's case id or ``<case>::<config>``, the worker dies with
    ``os._exit(1)`` before solving — the abrupt, exception-less death that
    breaks a ``ProcessPoolExecutor``. Never set it outside the tests. The serial
    path calls :func:`run_case` directly, so the hook cannot kill the parent.
    """
    if os.environ.get(CRASH_ENV) in (case.id, f"{case.id}::{cfg_name}"):
        os._exit(1)
    return run_case(case, cfg_name, **kw)


def _new_pool(n: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=n, initializer=pin_worker_threads)


def _run_isolated(case: Case, cfg_name: str, kw: dict, why: str) -> tuple:
    """Run one pair ALONE in a fresh single-worker pool -> ``(record, crashed)``.

    Alone, a pool break can only be this pair's own death, so it is recorded as
    a measured ``WorkerCrash`` row (no DVF). An ordinary exception out of the
    future keeps the dead-worker row with that error."""
    with pinned_thread_env(), _new_pool(1) as ex:
        fut = ex.submit(_pool_worker, case, cfg_name, **kw)
        try:
            return fut.result(), False
        except BrokenProcessPool:
            err = f"WorkerCrash: worker process terminated abruptly ({why})"
            return sentinel_record(case, cfg_name, err, kw["timing_mode"]), True
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            return sentinel_record(case, cfg_name, err, kw["timing_mode"]), False


def _run_parallel(work, n_workers, kw_for, record, isolate=()) -> dict:
    """The parallel pass with pool-break recovery. Returns the counters.

    ``kw_for(case, cfg)`` gives the :func:`run_case` kwargs of a pair and
    ``record(rec)`` lands a row. A ``BrokenProcessPool`` never becomes a row:
    once the pool has broken and every future has resolved, the unfinished
    pairs are sorted back into submission order and the first
    ``n_workers + 1`` are the suspects. Workers take calls FIFO, so the pairs a
    worker was running are always the earliest unfinished ones (at most
    ``n_workers``); the ``+ 1`` covers the call the executor pre-queues. Each
    suspect reruns alone (:func:`_run_isolated`) — only a pair that breaks its
    own solo pool is a ``WorkerCrash`` — and the rest go to a fresh pool. After
    ``MAX_POOL_REBUILDS`` breaks the remaining pairs all run isolated. Pairs
    whose config is in *isolate* run isolated after the parallel pass.
    """
    isolate = set(isolate)
    pending = [p for p in work if p[1] not in isolate]
    stats = {"pool_breaks": 0, "worker_crashes": 0}

    def land(rec):
        record(rec)
        _log(f"{rec['label']} done  {rec['error']}".rstrip())

    def solo(pair, why):
        rec, crashed = _run_isolated(*pair, kw_for(*pair), why)
        stats["worker_crashes"] += crashed
        land(rec)
        return rec

    while pending:
        if stats["pool_breaks"] > MAX_POOL_REBUILDS:
            _log(
                f"pool-rebuild cap ({MAX_POOL_REBUILDS}) exceeded: running the "
                f"{len(pending)} remaining pairs one at a time in isolation"
            )
            for pair in pending:
                solo(pair, "isolated run past the pool-rebuild cap")
            break
        lost = []
        with pinned_thread_env(), _new_pool(n_workers) as ex:
            futs = {ex.submit(_pool_worker, *pair, **kw_for(*pair)): pair for pair in pending}
            for fut in as_completed(futs):
                c, cfg = futs[fut]
                try:
                    rec = fut.result()
                except BrokenProcessPool:  # an infrastructure loss: rerun, never a row
                    lost.append((c, cfg))
                    continue
                except Exception as exc:  # an ordinary worker exception is a row
                    err = f"{type(exc).__name__}: {exc}"
                    rec = sentinel_record(c, cfg, err, kw_for(c, cfg)["timing_mode"])
                land(rec)
        if not lost:
            break
        stats["pool_breaks"] += 1
        order = {pair: i for i, pair in enumerate(pending)}
        lost.sort(key=order.__getitem__)
        suspects, pending = lost[: n_workers + 1], lost[n_workers + 1 :]
        labels = [f"{c.id}__{cfg}" for c, cfg in suspects]
        _log(
            f"POOL BREAK #{stats['pool_breaks']}: {len(lost)} unfinished pairs; "
            f"rerunning the {len(suspects)} suspects alone: {labels}"
        )
        outcome = [solo(p, "isolated rerun after a pool break")["error"] or "ok" for p in suspects]
        _log(
            f"pool break #{stats['pool_breaks']} suspects: "
            + ", ".join(f"{lab} -> {o.split(':')[0]}" for lab, o in zip(labels, outcome))
            + f"; resubmitting {len(pending)} pairs to a fresh pool"
        )
    for pair in work:
        if pair[1] in isolate:
            solo(pair, "isolated run, --isolate-config")
    return stats


_STR_KEYS = (*_IDENT_KEYS, "error", "timing_mode", "out_path", "sha256")


def _typed_row(row: dict) -> dict:
    """A ``results.csv`` row back to the types :func:`run_case` produced, so a
    reused row aggregates like a fresh one AND writes back byte-identically
    (``""`` <-> None, ``True``/``False``, int, float repr, else the string)."""
    out: dict = {}
    for k, v in row.items():
        if k in _STR_KEYS:
            out[k] = v
        elif v == "":
            out[k] = None
        elif v in ("True", "False"):
            out[k] = v == "True"
        else:
            for conv in (int, float, str):
                try:
                    out[k] = conv(v)
                    break
                except ValueError:
                    pass
    return out


def _dvf_ok(file: str, sha: str) -> bool:
    p = Path(file)
    p = p if p.is_absolute() else REPO / p
    return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha


RERUN_REASONS = ("missing", *LOSS_PREFIXES, "dvf_missing_or_sha_mismatch")


def _split_resume(old_dir, work, timing_mode, threshold, cap_s) -> tuple:
    """Split *work* against a previous run dir -> ``(reused, rerun, reasons)``.

    A pair's old row is reused verbatim iff it exists, its ``error`` is empty or
    a real measured failure (not a :data:`LOSS_PREFIXES` loss), and the DVF its
    manifest entry names (if any) still exists with the recorded sha256.
    ``reused`` are typed records in work-list order, ``rerun`` the pairs to run,
    ``reasons`` a count per :data:`RERUN_REASONS`.

    Refuses (``ValueError``) an old run measured under a different protocol: a
    ``threshold`` or ``cap_s`` differing from the old ``summary.json``, or a
    reused row whose ``timing_mode`` differs — a throughput wall must never land
    in a serial column (reused rows keep their ``time_s`` and ``hit_cap``)."""
    old_dir = Path(old_dir)
    summary = old_dir / "summary.json"
    if summary.is_file():
        prov = json.loads(summary.read_text(encoding="utf-8"))["provenance"]
        for name, new in (("threshold", threshold), ("cap_s", cap_s)):
            if prov.get(name) != new:
                raise ValueError(
                    f"--resume {old_dir}: {name} mismatch (old {prov.get(name)!r}, new {new!r})"
                )
    with open(old_dir / "results.csv", newline="", encoding="utf-8") as f:
        rows = {(r["source"], r["case"], r["config"]): r for r in csv.DictReader(f)}
    manifest = json.loads((old_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {(e["source"], e["case"], e["config"]): e for e in manifest["fields"]}
    reused, rerun = [], []
    reasons = dict.fromkeys(RERUN_REASONS, 0)
    for c, cfg in work:
        key = (c.source, c.id, cfg)
        row = rows.get(key)
        why = None
        if row is None:
            why = "missing"
        else:
            why = next((p for p in LOSS_PREFIXES if row["error"].startswith(p)), None)
            entry = entries.get(key, {})
            file = entry.get("file", row["out_path"])
            if why is None and file and not _dvf_ok(file, entry.get("sha256", row["sha256"])):
                why = "dvf_missing_or_sha_mismatch"
        if why is None:
            reused.append(_typed_row(row))
        else:
            reasons[why] += 1
            rerun.append((c, cfg))
    for r in reused:
        if r["timing_mode"] != timing_mode:
            raise ValueError(
                f"--resume {old_dir}: timing_mode mismatch on {r['label']} "
                f"(old {r['timing_mode']!r}, new {timing_mode!r})"
            )
    return reused, rerun, reasons


def _work_list(sources, config_names, sample, explicit_configs):
    """``[(case, cfg_name)]`` in a deterministic order, applying the protocol's
    source filter (the two engine rows everywhere, the rest of the taxonomy on
    the small sources) unless the configs were named explicitly."""
    work = []
    for src in sources:
        for case in cases(src, sample):
            for cfg in config_names:
                applies = explicit_configs or cfg in _EVERY_SOURCE_CONFIGS or src in _SMALL_SOURCES
                if applies:
                    work.append((case, cfg))
    return work


def run(
    sources=SOURCES,
    config_names=tuple(CONFIGS),
    *,
    sample="canonical",
    n_workers=1,
    run_dir=None,
    cap_s=DEFAULT_CAP_S,
    serial_timing=False,
    figures=False,
    table=False,
    hist_case=None,
    explicit_configs=False,
    threshold=THRESHOLD,
    verbose=0,
    resume=None,
    isolate_configs=(),
):
    """Run the benchmark and write the run directory. Returns its ``Path``.

    *resume* is a previous run dir whose measured rows are reused (see
    :func:`_split_resume`); *isolate_configs* run one pair at a time in their
    own single-worker pool after the parallel pass (ignored on the serial path,
    which is in-process by design)."""
    if serial_timing:
        n_workers = 1
    timing_mode = "serial" if serial_timing else "throughput"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_dir) if run_dir else REPO / "benchmarks" / "output" / f"2d_canonical_{stamp}"
    work = _work_list(sources, config_names, sample, explicit_configs)
    by_key = {(c.source, c.id): c for c, _ in work}
    _log(f"{len(work)} (case, config) runs -> {run_dir}  [{timing_mode}, n_workers={n_workers}]")
    reused: list = []
    reasons = dict.fromkeys(RERUN_REASONS, 0)
    if resume:  # may refuse: nothing has been written yet
        reused, work, reasons = _split_resume(resume, work, timing_mode, threshold, cap_s)
        _log(f"resume from {resume}: reusing {len(reused)} rows, rerunning {len(work)} {reasons}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name
    dvf_dir = DVF_ROOT / "results" / run_name
    box_load = _box_load()
    # a lone pair still goes through a pool: a resumed crash must not kill the parent
    parallel = n_workers > 1 and len(work) > 0
    if isolate_configs and not parallel:
        _log("WARNING: --isolate-config ignored: the serial path runs in-process")
    pool_stats = {"pool_breaks": 0, "worker_crashes": 0}

    # Inputs are loaded lazily, inside the worker (or just before the serial
    # solve): the parent never holds one array per queued run.
    def out_path(case, cfg):
        return str(dvf_dir / case.source / f"{case.id}__{cfg}.npz")

    t_run = time.perf_counter()
    kw = dict(threshold=threshold, cap_s=cap_s, timing_mode=timing_mode, verbose=verbose)
    records: list = []
    with results_csv(run_dir) as append_row:
        # Each row is flushed to results.csv (and the manifest rewritten) as it
        # lands, so an interrupted or crashed chain still leaves a valid run dir.
        def _record(rec):
            records.append(rec)
            append_row(rec)
            _write_manifest(run_dir, records, by_key)

        for rec in reused:  # a resumed run's measured rows land first, in work order
            _record(rec)
        if parallel:
            pool_stats = _run_parallel(
                work,
                n_workers,
                lambda c, cfg: dict(out_path=out_path(c, cfg), **kw),
                _record,
                isolate_configs,
            )
        else:
            phi = corr = None
            prev_case = load_err = None
            for i, (c, cfg) in enumerate(work, 1):
                _log(f"{c.id}__{cfg} ({i}/{len(work)}) ...")
                if c != prev_case:  # one load shared by every config of a case
                    prev_case, phi, corr, load_err = c, None, None, None
                    try:
                        phi, corr = load_case(c), case_correspondences(c)
                    except Exception as exc:
                        load_err = f"{type(exc).__name__}: {exc}"
                        _log(f"WARNING: cannot load {c.id}: {load_err}")
                if phi is None:
                    _record(sentinel_record(c, cfg, load_err or "LoadError", timing_mode))
                    continue
                _record(run_case(c, cfg, phi, corr, out_path=out_path(c, cfg), **kw))
    total_s = time.perf_counter() - t_run

    _write_summary(
        run_dir,
        records,
        sources,
        config_names,
        sample,
        n_workers,
        timing_mode,
        cap_s,
        threshold,
        total_s,
        box_load,
        extra_provenance={
            **pool_stats,
            "isolated_configs": list(isolate_configs) if parallel else [],
            "resumed_from": str(resume) if resume else None,
            "n_reused": len(reused),
            "n_rerun": len(work),
            "rerun_reasons": reasons,
        },
    )
    _write_report(run_dir, records, threshold, total_s)
    if figures:
        make_figures(run_dir / "figures", records, hist_case=hist_case)
    if table:
        md = markdown_table(records)
        (run_dir / "table.md").write_text(md, encoding="utf-8")
        print(md)
    _log(f"wrote {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


@contextmanager
def results_csv(run_dir):
    """Open ``results.csv``, write the header NOW — the schema is known before the
    first row (:func:`record_keys`) — and yield an ``append(record)`` that flushes
    after every row, so a chain that dies mid-way still leaves a readable CSV."""
    with open(Path(run_dir) / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=record_keys(), extrasaction="ignore")
        w.writeheader()
        f.flush()

        def append(rec):
            w.writerow({k: rec.get(k) for k in record_keys()})
            f.flush()

        yield append


#: Aggregated per source x config as median + IQR. Every one of these is
#: non-negative by construction, so a ``-1`` can only be a sentinel (a case that
#: never ran, or an engine column on a non-windowed strategy) and is skipped:
#: the DENOMINATORS (``n``, the rates) still count those rows, the DISTRIBUTIONS
#: do not.
_AGG_KEYS = (
    "time_s",
    "l1_move",
    "l2_move",
    "max_move",
    "moved_frac",
    "sqp_iters",
    "rounds",
    "n_windows",
    "damage",
    "sdlogj_init",
    "sdlogj_final",
    "frac_nonpos_jdet_init",
    "frac_nonpos_jdet_final",
    "n_neg_init",
    "n_neg_final",
)


def _quantiles(vals):
    """Median / IQR over the non-sentinel values, or ``None`` if there are none."""
    v = np.asarray([x for x in vals if x is not None and x >= 0], dtype=np.float64)
    if v.size == 0:
        return None
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def _group_summary(rows):
    n = len(rows)
    damages = [r["damage"] for r in rows if r["damage"] >= 0]
    out = {
        "n": n,
        "feasible_rate": sum(1 for r in rows if r["feasible"]) / n,
        "certified_rate": sum(1 for r in rows if r["certified"]) / n,
        "errors": sum(1 for r in rows if r["error"]),
        "hit_cap": sum(1 for r in rows if r["hit_cap"]),
        # None (JSON null) when no row in the group ran the windowed engine —
        # never -1, which would read as "damage minus one"
        "max_damage": max(damages) if damages else None,
        "n_windowed_rows": len(damages),
    }
    for fam, _ in _FAMILIES:  # certification rate under each gauge (after)
        out[f"rate_{fam}_zero_at_threshold"] = (
            sum(1 for r in rows if r[f"{fam}_n_below_final"] == 0) / n
        )
        out[f"rate_{fam}_zero_at_0"] = sum(1 for r in rows if r[f"{fam}_n_neg_final"] == 0) / n
    for k in _AGG_KEYS:
        out[k] = _quantiles([r.get(k) for r in rows])
    return out


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _box_load():
    """Average CPU load (%) at run start — the protocol requires recording it
    (an unrelated job once doubled every wall). ``-1`` when unavailable."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property LoadPercentage -Average).Average",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return -1.0


def _write_summary(
    run_dir,
    records,
    sources,
    config_names,
    sample,
    n_workers,
    timing_mode,
    cap_s,
    threshold,
    total_s,
    box_load,
    extra_provenance=None,
):
    groups = {}
    for r in records:
        groups.setdefault(f"{r['source']}/{r['config']}", []).append(r)
    summary = {
        "provenance": {
            "git_commit": _git_commit(),
            "dvfopt_version": dvfopt.__version__,
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "n_workers": n_workers,
            "timing_mode": timing_mode,
            "box_load_pct_at_start": box_load,
            "threshold": threshold,
            "err_tol": ERR_TOL,
            "time_budget_s": None,  # NOT set — the engine runs to its own termination
            "cap_s": cap_s,
            "sample": sample,
            "sources": list(sources),
            "configs": {k: CONFIGS[k] for k in config_names},
            "sdlogj_clip": SDLOGJ_CLIP,
            "total_time_s": total_s,
            **(extra_provenance or {}),
        },
        "gauges": GAUGES,
        "notes": [ANTS_Z_NOTE, SENTINEL_NOTE],
        "n_runs": len(records),
        "n_certified": sum(1 for r in records if r["certified"]),
        "groups": {k: _group_summary(v) for k, v in sorted(groups.items())},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_manifest(run_dir, records, case_by_key):
    """Every run, with its case, config, shape, sha256 and INPUT path.

    Keyed by ``(source, id)`` — the cohort and its ANTs twin are different
    inputs, and only the pair identifies one. A run that produced no DVF (a load
    failure, a dead worker, a failed solve) is still listed, with an empty
    ``file`` and its ``error``: nothing is dropped from the record of the run.
    """
    entries = [
        {
            "file": r["out_path"],
            "case": r["case"],
            "source": r["source"],
            "config": r["config"],
            "shape": r["shape"],
            "sha256": r["sha256"],
            "input_path": getattr(case_by_key.get((r["source"], r["case"])), "path", ""),
            "error": r["error"],
        }
        for r in records
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"run": run_dir.name, "notes": [ANTS_Z_NOTE, SENTINEL_NOTE], "fields": entries},
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_report(run_dir, records, threshold, total_s):
    """``report/report.html`` via the shared cohort writer (figures off — ours
    live in ``figures/``; its own CSV/JSON copies land beside it in ``report/``)."""
    rep = run_dir / "report"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    cb._write_run_artifacts(
        rep,
        rep / "figures",
        records,
        "canonical_2d: correct_dvf(threshold=0.01, record_history=True, **config)",
        threshold,
        total_s,
        make_figures=False,
    )


# ---------------------------------------------------------------------------
# Figures and table
# ---------------------------------------------------------------------------


def make_figures(fig_dir, records, hist_case=None):
    """Fold counts before/after per source (log axis), the wall-vs-move frontier
    per config on the origins set, and jdet histograms for one named case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dvfopt.viz.theme import apply_theme

    apply_theme("paper")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. folds before/after per source (bilinear gauge at the threshold).
    # The INPUT count is per case — summing it over configs would multiply the
    # "before" bar by the number of configs; the "after" bars are per config.
    srcs = sorted({r["source"] for r in records})
    cfgs = sorted({r["config"] for r in records})
    rows_ok = [r for r in records if r["bilinear_n_below_init"] >= 0]  # skip never-ran rows
    before = []
    for s in srcs:
        seen = {}
        for r in rows_ok:
            if r["source"] == s:
                seen.setdefault(r["case"], r["bilinear_n_below_init"])
        before.append(sum(seen.values()))
    fig, ax = plt.subplots(figsize=(1.9 * max(len(srcs), 3) + 2, 3.6))
    x = np.arange(len(srcs))
    w = 0.8 / (len(cfgs) + 1)
    ax.bar(x - 0.4 + w / 2, np.maximum(before, 0.5), w, label="before (input)", color=cb._C_BEFORE)
    for i, cfg in enumerate(cfgs, 1):
        after = [
            sum(
                r["bilinear_n_below_final"]
                for r in rows_ok
                if r["source"] == s and r["config"] == cfg
            )
            for s in srcs
        ]
        ax.bar(x - 0.4 + w * (i + 0.5), np.maximum(after, 0.5), w, label=f"after · {cfg}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(srcs)
    ax.set_ylabel("folded cells (bilinear, < 0.01)")
    ax.set_title("Folds before / after by source (0.5 = zero, log axis)")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "folds_by_source.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. wall vs L2 move, per config, on the origins set
    org = [r for r in records if r["source"] == "origins"]
    if org:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for cfg in sorted({r["config"] for r in org}):
            rs = [r for r in org if r["config"] == cfg]
            ax.scatter([r["time_s"] for r in rs], [r["l2_move"] for r in rs], s=22, label=cfg)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall (s)")
        ax.set_ylabel("L2 move")
        ax.set_title("Wall vs move frontier (origins)")
        ax.legend(fontsize=7)
        fig.savefig(fig_dir / "frontier_origins.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. jdet histograms before/after for one named case
    if hist_case:
        rs = [r for r in records if r["case"] == hist_case and r["out_path"]]
        if not rs:
            _log(f"WARNING: --hist-case {hist_case!r} has no saved output — histogram skipped")
        for r in rs:
            case = next(c for c in cases(r["source"]) if c.id == hist_case)
            ji = _jdet(load_case(case))
            with np.load(REPO / r["out_path"]) as z:
                jf = _jdet(z["arr"])
            fig = bu.plot_jdet_histograms(
                [[("before", ji), ("after", jf)]], [f"{hist_case} · {r['config']}"]
            )
            fig.savefig(
                fig_dir / f"hist_{hist_case}__{r['config']}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)


def markdown_table(records) -> str:
    """Markdown summary per source x config: medians with IQR in brackets.

    Prefixed by the gauge legend, so the certificate scales travel with the
    numbers (the simplicial columns are triangle areas = det/2).
    """
    groups = {}
    for r in records:
        groups.setdefault((r["source"], r["config"]), []).append(r)
    legend = [
        "<!-- certificate gauges: "
        + "; ".join(
            f"{k}: {v['rows']}, {v['scale']}, per {v['per']}"
            for k, v in GAUGES.items()
            if not k.startswith("_")
        )
        + ". certified = bilinear has 0 values < 0.01 - 1e-5 after."
        + " -1 is a sentinel (see summary.json notes), skipped by every median. -->"
    ]
    head = (
        "| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | "
        "L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |"
    )
    lines = [*legend, head, "|" + "---|" * 11]

    def med(rows, k, fmt=".4g"):
        q = _quantiles([r[k] for r in rows])
        return f"{q['median']:{fmt}} [{q['q1']:{fmt}}, {q['q3']:{fmt}}]" if q else "n/a"

    def med1(rows, k, fmt=".4g"):
        q = _quantiles([r[k] for r in rows])
        return f"{q['median']:{fmt}}" if q else "n/a"

    for (src, cfg), rows in sorted(groups.items()):
        n = len(rows)
        note = " (TUNING SET)" if src == "crops" else ""
        damages = [r["damage"] for r in rows if r["damage"] >= 0]
        lines.append(
            f"| {src}{note} | {cfg} | {n} | "
            f"{sum(1 for r in rows if r['certified'])}/{n} | "
            f"{sum(1 for r in rows if r['feasible'])}/{n} | "
            f"{med(rows, 'time_s')} | {med(rows, 'l1_move')} | {med(rows, 'l2_move')} | "
            f"{med1(rows, 'sdlogj_init')} -> {med1(rows, 'sdlogj_final')} | "
            f"{med1(rows, 'frac_nonpos_jdet_init', '.3g')} -> "
            f"{med1(rows, 'frac_nonpos_jdet_final', '.3g')} | "
            f"{max(damages) if damages else 'n/a'} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", nargs="+", choices=SOURCES, default=list(SOURCES))
    p.add_argument(
        "--config",
        nargs="+",
        choices=list(CONFIGS),
        default=None,
        help="default: every config, with the protocol's source filter",
    )
    p.add_argument("--sample", choices=("canonical", "smoke"), default="canonical")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument(
        "--serial-timing",
        action="store_true",
        help="pin n_workers=1 and tag the rows timing_mode='serial'",
    )
    p.add_argument("--run-dir", default=None)
    p.add_argument(
        "--cap-s",
        type=float,
        default=DEFAULT_CAP_S,
        help="soft per-run wall cap: recorded as hit_cap, never interrupts",
    )
    p.add_argument("--figures", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--hist-case", default=None, help="case id for the jdet histogram figure")
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help="reuse RUN_DIR's measured rows; rerun only missing / BrokenProcessPool / "
        "WorkerCrash rows and rows whose DVF is missing or altered",
    )
    p.add_argument(
        "--isolate-config",
        nargs="+",
        choices=list(CONFIGS),
        default=[],
        metavar="CFG",
        help="run these configs after the parallel pass, one pair per single-worker pool",
    )
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    run_dir = run(
        a.source,
        tuple(a.config) if a.config else tuple(CONFIGS),
        sample=a.sample,
        n_workers=a.n_workers,
        run_dir=a.run_dir,
        cap_s=a.cap_s,
        serial_timing=a.serial_timing,
        figures=a.figures,
        table=a.table,
        hist_case=a.hist_case,
        explicit_configs=a.config is not None,
        verbose=a.verbose,
        resume=a.resume,
        isolate_configs=tuple(a.isolate_config),
    )
    return 0 if run_dir else 1


if __name__ == "__main__":  # spawn-safe: workers re-import this module, never main()
    raise SystemExit(main())
