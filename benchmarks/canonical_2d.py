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
``summary.json`` (per source x config aggregates + provenance),
``manifest.json`` (every saved DVF with case, config, shape, sha256 and the
INPUT's source path — inputs are not duplicated), ``figures/`` with
``--figures``, and ``report/report.html`` from the shared cohort writer.
Corrected DVFs go to ``data/dvfs/results/<run-name>/<source>/<case>__<cfg>.npz``.

CLI::

    # smoke (no gitignored data needed)
    python benchmarks/canonical_2d.py --source synthetic --config isqp_none \\
        --sample smoke --run-dir /tmp/smoke

    # the throughput pass over a source, with figures and the markdown table
    python benchmarks/canonical_2d.py --source origins --n-workers 4 --figures --table

    # the per-case wall column the paper quotes
    python benchmarks/canonical_2d.py --source origins --serial-timing

Run it from the repo root (the gitignored data resolves relative to this file).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
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
                        id=f"{b}_z{z}",
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


def _engine_stats(res) -> dict:
    """Engine accounting from ``res.info``; ``-1`` for a non-windowed strategy."""
    out = dict.fromkeys(
        ("damage", "n_windows", "giant_regions", "mop_cleared", "rounds", "sqp_iters"), -1
    )
    if res is None:
        return out
    info = getattr(res, "info", None)
    if info is None:
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


def run_case(
    case: Case,
    cfg_name: str,
    phi_in,
    corr_pts=None,
    *,
    threshold: float = THRESHOLD,
    cap_s: float = DEFAULT_CAP_S,
    timing_mode: str = "throughput",
    out_path: str = "",
    verbose: int = 0,
) -> dict:
    """Solve one (case, config) and return its record. Module-level and picklable
    — this is the process-pool worker. Saves the corrected DVF to *out_path*
    (``.npz``, key ``arr``) when given, and never raises: a failed solve becomes
    a row with ``error`` set and the unchanged field as the output."""
    pin_worker_threads()
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

    rec = {
        "label": f"{case.id}__{cfg_name}",
        "case": case.id,
        "source": case.source,
        "config": cfg_name,
        "mechanism": case.mechanism,
        "tool": case.tool,
        "shape": "x".join(str(n) for n in phi_in.shape),
    }
    rec.update(metrics(phi_in, phi_out, res, threshold, elapsed))
    rec.update(_corr_stats(phi_in, phi_out, corr_pts))
    rec["error"] = err
    rec["timing_mode"] = timing_mode
    rec["hit_cap"] = bool(elapsed > cap_s)
    rec["out_path"] = ""
    rec["sha256"] = ""
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, arr=phi_out)  # save_dvf handles .npy/sitk only
        rec["out_path"] = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
        rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return rec


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
):
    """Run the benchmark and write the run directory. Returns its ``Path``."""
    if serial_timing:
        n_workers = 1
    timing_mode = "serial" if serial_timing else "throughput"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_dir) if run_dir else REPO / "benchmarks" / "output" / f"2d_canonical_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name
    dvf_dir = DVF_ROOT / "results" / run_name
    box_load = _box_load()

    work = _work_list(sources, config_names, sample, explicit_configs)
    _log(f"{len(work)} (case, config) runs -> {run_dir}  [{timing_mode}, n_workers={n_workers}]")

    # The parent loads every input (one cohort/ANTs volume cached at a time) and
    # ships the (3, 1, H, W) section to the worker — a few MB per pickle.
    tasks = []
    for case, cfg in work:
        try:
            phi = load_case(case)
        except Exception as exc:
            _log(f"WARNING: cannot load {case.id}: {type(exc).__name__}: {exc} — skipped")
            continue
        out_path = str(dvf_dir / case.source / f"{case.id}__{cfg}.npz")
        tasks.append((case, cfg, phi, case_correspondences(case), out_path))

    t_run = time.perf_counter()
    kw = dict(threshold=threshold, cap_s=cap_s, timing_mode=timing_mode, verbose=verbose)
    if n_workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with (
            pinned_thread_env(),
            ProcessPoolExecutor(max_workers=n_workers, initializer=pin_worker_threads) as ex,
        ):
            futs = [
                ex.submit(run_case, c, cfg, phi, cp, out_path=op, **kw)
                for (c, cfg, phi, cp, op) in tasks
            ]
            records = []
            for i, f in enumerate(futs, 1):
                records.append(f.result())
                _log(f"{records[-1]['label']} done ({i}/{len(futs)})")
    else:
        records = []
        for i, (c, cfg, phi, cp, op) in enumerate(tasks, 1):
            _log(f"{c.id}__{cfg} ({i}/{len(tasks)}) ...")
            records.append(run_case(c, cfg, phi, cp, out_path=op, **kw))
    total_s = time.perf_counter() - t_run

    _write_results_csv(run_dir, records)
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
    )
    _write_manifest(run_dir, records, {c.id: c for c, _ in work})
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


def _write_results_csv(run_dir, records):
    """One row per case x config. Columns: the identity + existing schema keys
    first, in their existing order, then every other key in first-seen order."""
    seen = dict.fromkeys(k for r in records for k in r)
    cols = [c for c in _LEAD_COLS if c in seen] + [k for k in seen if k not in _LEAD_COLS]
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in cols})


#: Aggregated per source x config as median + IQR.
_AGG_KEYS = (
    "time_s",
    "l1_move",
    "l2_move",
    "max_move",
    "moved_frac",
    "sqp_iters",
    "sdlogj_init",
    "sdlogj_final",
    "frac_nonpos_jdet_init",
    "frac_nonpos_jdet_final",
    "n_neg_init",
    "n_neg_final",
)


def _quantiles(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
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
    out = {
        "n": n,
        "feasible_rate": sum(1 for r in rows if r["feasible"]) / n,
        "certified_rate": sum(1 for r in rows if r["certified"]) / n,
        "errors": sum(1 for r in rows if r["error"]),
        "hit_cap": sum(1 for r in rows if r["hit_cap"]),
        "max_damage": max(r["damage"] for r in rows),
    }
    for fam, _ in _FAMILIES:  # certification rate under each gauge (after)
        out[f"rate_{fam}_zero_at_threshold"] = (
            sum(1 for r in rows if r[f"{fam}_n_below_final"] == 0) / n
        )
        out[f"rate_{fam}_zero_at_0"] = sum(1 for r in rows if r[f"{fam}_n_neg_final"] == 0) / n
    for k in _AGG_KEYS:
        q = _quantiles([r.get(k) for r in rows])
        if q is not None:
            out[k] = q
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
        },
        "n_runs": len(records),
        "n_certified": sum(1 for r in records if r["certified"]),
        "groups": {k: _group_summary(v) for k, v in sorted(groups.items())},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_manifest(run_dir, records, case_by_id):
    """Every saved DVF with its case, config, shape, sha256 and INPUT path."""
    entries = [
        {
            "file": r["out_path"],
            "case": r["case"],
            "source": r["source"],
            "config": r["config"],
            "shape": r["shape"],
            "sha256": r["sha256"],
            "input_path": case_by_id[r["case"]].path,
        }
        for r in records
        if r["out_path"]
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"run": run_dir.name, "fields": entries}, indent=2), encoding="utf-8"
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

    # 1. folds before/after per source (bilinear gauge at the threshold)
    srcs = sorted({r["source"] for r in records})
    fig, ax = plt.subplots(figsize=(1.6 * max(len(srcs), 3) + 2, 3.6))
    x = np.arange(len(srcs))
    before = [sum(r["bilinear_n_below_init"] for r in records if r["source"] == s) for s in srcs]
    after = [sum(r["bilinear_n_below_final"] for r in records if r["source"] == s) for s in srcs]
    ax.bar(x - 0.2, np.maximum(before, 0.5), 0.4, label="before", color=cb._C_BEFORE)
    ax.bar(x + 0.2, np.maximum(after, 0.5), 0.4, label="after", color=cb._C_AFTER)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(srcs)
    ax.set_ylabel("folded cells (bilinear, < 0.01)")
    ax.set_title("Folds before / after by source (0.5 = zero, log axis)")
    ax.legend()
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
    """Markdown summary per source x config: medians with IQR in brackets."""
    groups = {}
    for r in records:
        groups.setdefault((r["source"], r["config"]), []).append(r)
    head = (
        "| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | "
        "L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |"
    )
    lines = [head, "|" + "---|" * 11]

    def med(rows, k):
        q = _quantiles([r[k] for r in rows])
        return f"{q['median']:.4g} [{q['q1']:.4g}, {q['q3']:.4g}]" if q else "n/a"

    for (src, cfg), rows in sorted(groups.items()):
        n = len(rows)
        note = " (TUNING SET)" if src == "crops" else ""
        lines.append(
            f"| {src}{note} | {cfg} | {n} | "
            f"{sum(1 for r in rows if r['certified'])}/{n} | "
            f"{sum(1 for r in rows if r['feasible'])}/{n} | "
            f"{med(rows, 'time_s')} | {med(rows, 'l1_move')} | {med(rows, 'l2_move')} | "
            f"{np.median([r['sdlogj_init'] for r in rows]):.4g} -> "
            f"{np.median([r['sdlogj_final'] for r in rows]):.4g} | "
            f"{np.median([r['frac_nonpos_jdet_init'] for r in rows]):.3g} -> "
            f"{np.median([r['frac_nonpos_jdet_final'] for r in rows]):.3g} | "
            f"{max(r['damage'] for r in rows)} |"
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
    )
    return 0 if run_dir else 1


if __name__ == "__main__":  # spawn-safe: workers re-import this module, never main()
    raise SystemExit(main())
