"""Manuscript-grade L1-anchor comparison across 2D methods on the B0039
Laplacian DVF + canonical synthetic suite — Jdet-CD vs 2-tri constraint
families.

Why this benchmark
------------------
We compare every 2D method exposed by ``dvfopt`` under a uniform
:class:`L1Objective` anchor, across two constraint families:

* ``constraint='jdet'`` — central-difference Jacobian determinant
  (``jacobian_det2D``).
* ``constraint='2tri'`` — full-coverage 2-triangle decomposition
  (``TriConstraint2DFullCoverage``). Sees sub-pixel folds the Jdet
  stencil misses, so its ``n_neg`` count is consistently higher.

Cases come from two datasets:

* **B0039 Laplacian DVF** (``dataset='b0039'``) — a 528-slice 3D field
  with every slice carrying ~280–5300 folded 2-triangle cells. Slices
  are picked into equal-density buckets (easy / medium / hard / extreme)
  so the manuscript can defensibly say "evaluated across the difficulty
  range." z=0 and z=12 are always included (the canonical extreme
  cases from CLAUDE.md).
* **Canonical synthetic suite** (``dataset='canonical'``) — the 6
  ``test_cases.canonical_2tri_2d`` fixtures (10×10 and 20×20 quads with
  planted crossings / opposing displacements). Fast smoke checks that
  also pin behaviour at the small-grid limit.

For each ``(case_id, constraint_family, strategy)`` triple we record:

* feasibility metrics under BOTH views (because Jdet-CD misses
  sub-pixel folds that 2-tri catches): ``init_n_neg_jdet``,
  ``init_n_neg_2tri``, ``final_n_neg_jdet``, ``final_n_neg_2tri``,
  plus the strategy-native ``final_n_neg`` / ``final_min_T``;
* deviation metrics: ``l1_total``, ``l1_mean``, ``l2_total``, ``linf``;
* wall-clock + iteration counts;
* reproducibility metadata: ``git_sha``, ``timestamp_utc``,
  ``threshold``, ``eps_l1``, ``time_budget_s``.

Resumable design
----------------
Rows are appended to ``results.csv`` *one at a time*, flushed
immediately. On startup we read the CSV and skip any
``(case_id, constraint, strategy)`` triple already recorded. Crashes /
Ctrl-C / OS power events resume cleanly. Failures (timeout, exception)
ARE recorded with ``error_kind``/``error_msg`` populated so a re-run
won't re-attempt them. Pass ``--retry-errors`` to re-attempt error rows.

Outputs
-------
* ``benchmarks/output/b0039_l1_comparison/results.csv`` — one row per run.
* ``benchmarks/output/b0039_l1_comparison/case_scan.csv`` — initial
  per-case feasibility stats (cached; regenerated if missing).
* ``benchmarks/output/b0039_l1_comparison/log.txt`` — progress log.

Usage
-----
    python benchmarks/b0039_l1_comparison.py
    python benchmarks/b0039_l1_comparison.py --slice-count 20
    python benchmarks/b0039_l1_comparison.py --no-canonical
    python benchmarks/b0039_l1_comparison.py --retry-errors
    python benchmarks/b0039_l1_comparison.py --cases b0039_z0,canonical_01a_10x10_crossing
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
B0039_PATH = REPO_ROOT / 'data' / 'dvfs' / 'b0039' / 'b0039_laplacian_deformation_field.npy'
OUT_DIR = REPO_ROOT / 'benchmarks' / 'output' / 'b0039_l1_comparison'
RESULTS_CSV = OUT_DIR / 'results.csv'
# Two scan files: the full per-slice density scan (528 slices, used as
# the bucket-selection cache so re-runs pick the same slices) AND the
# selected-cases summary (used by downstream analysis to join init
# stats onto the per-run results).
B0039_FULL_SCAN_CSV = OUT_DIR / 'slice_scan.csv'
CASE_SCAN_CSV = OUT_DIR / 'case_scan.csv'
LOG_FILE = OUT_DIR / 'log.txt'

# Per-strategy wall-clock budget. Strategies with a ``time_budget_s`` field
# will respect this; others (SLSQP) may exceed it slightly via their own
# maxiter caps but won't run forever.
TIME_BUDGET_S = 180.0

THRESHOLD = 0.01
EPS_L1 = 1e-4
ERR_TOL = 1e-5

# Equal-density buckets for the B0039 slice picker. The volume's
# 2-tri fold count ranges roughly [280, 5300]; bucket ranges are
# chosen so each spans roughly the same density-magnitude band.
B0039_BUCKETS = [
    ('easy', 0, 600),          # 78 of 528 slices live here
    ('medium', 600, 1500),     # 296 of 528
    ('hard', 1500, 3000),      # 137 of 528
    ('extreme', 3000, 100_000),  # 17 of 528
]

# Methods to compare, grouped by constraint family. Each tuple is
# (method_id, constraint_label, strategy_label).
METHOD_SPECS: list[tuple[str, str, str]] = [
    # ---- Jdet (central-difference Jacobian) ----
    ('nmvf', 'jdet', 'NMVF'),
    ('barrier_jdet', 'jdet', 'Barrier'),
    ('slsqp_windowed_jdet', 'jdet', 'SLSQPWindowed'),
    # ---- 2-triangle full-coverage ----
    ('barrier_2tri', '2tri', 'Barrier'),
    ('slsqp_fullgrid_2tri', '2tri', 'SLSQPFullGrid'),
    ('slsqp_windowed_2tri', '2tri', 'SLSQPWindowed'),
    # Schwarz coverage — three variants so we can compare Schwarz around
    # different inner strategies (one of the manuscript's key claims).
    ('schwarz_slsqp_2tri', '2tri', 'SchwarzSLSQP'),
    ('schwarz_wrap_barrier_2tri', '2tri', 'SchwarzWrap(Barrier)'),
    ('m10_2tri', '2tri', 'HarmonicALMBarrier'),
    ('schwarz_wrap_m10_2tri', '2tri', 'SchwarzWrap(M10)'),
    ('m14_2tri', '2tri', 'HarmonicALMRefineRepair'),
    ('m14_schwarz_2tri', '2tri', 'SchwarzHarmonicALMRefineRepair'),
]


CSV_COLUMNS = [
    # Identity
    'case_id',
    'dataset',
    'slice_z',
    'case_shape',
    'constraint',
    'method_id',
    'strategy',
    'objective',
    # Initial stats under both views.
    'init_n_neg_jdet',
    'init_n_neg_2tri',
    'init_min_T_jdet',
    'init_min_T_2tri',
    # Final stats under BOTH views + the strategy-native pair.
    'final_n_neg',
    'final_min_T',
    'final_n_neg_jdet',
    'final_min_T_jdet',
    'final_n_neg_2tri',
    'final_min_T_2tri',
    'feasible',
    # Deviation metrics — all under the L1 objective.
    'l1_total',
    'l1_mean',
    'l2_total',
    'linf',
    # Solver stats
    'wall_time_s',
    'n_phases',
    'total_iter',
    'cluster_count',  # only populated for Schwarz methods; -1 otherwise
    'convergence_reason',  # 'converged' | 'timeout' | 'stalled' | 'error'
    # Error reporting
    'error_kind',
    'error_msg',
    # Reproducibility
    'threshold',
    'eps_l1',
    'time_budget_s',
    'git_sha',
    'numpy_version',
    'scipy_version',
    'dvfopt_version',
    'timestamp_utc',
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    line = f'[{datetime.now().isoformat(timespec="seconds")}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One benchmark case — a (case_id, phi, init-stats) bundle.

    ``case_id`` is the primary key in the results CSV. ``phi`` has shape
    ``(2, H, W)`` channels ``[dy, dx]``. ``meta`` carries dataset-level
    metadata that's copied verbatim onto every row of this case.
    """

    case_id: str
    dataset: str  # 'b0039' or 'canonical'
    phi: np.ndarray  # (2, H, W) float64
    slice_z: int  # -1 for synthetic
    jdet_n_neg: int
    jdet_min: float
    tri_n_neg: int
    tri_min: float


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _make_strategy(method_id: str):
    """Return a fresh Strategy instance for ``method_id``."""
    from dvfopt import (
        BarrierStrategy,
        HarmonicALMBarrierStrategy,
        HarmonicALMRefineRepairStrategy,
        NMVFStrategy,
        SchwarzHarmonicALMRefineRepairStrategy,
        SchwarzStrategy,
        SchwarzWrapperStrategy,
        SLSQPFullGridStrategy,
        SLSQPWindowedStrategy,
    )

    if method_id == 'nmvf':
        return NMVFStrategy(max_iter=300)
    if method_id == 'barrier_jdet':
        return BarrierStrategy()
    if method_id == 'slsqp_windowed_jdet':
        return SLSQPWindowedStrategy()
    if method_id == 'barrier_2tri':
        return BarrierStrategy()
    if method_id == 'slsqp_fullgrid_2tri':
        return SLSQPFullGridStrategy()
    if method_id == 'slsqp_windowed_2tri':
        return SLSQPWindowedStrategy()
    if method_id == 'schwarz_slsqp_2tri':
        return SchwarzStrategy()
    if method_id == 'schwarz_wrap_barrier_2tri':
        return SchwarzWrapperStrategy(
            inner=BarrierStrategy(),
            time_budget_s=TIME_BUDGET_S,
        )
    if method_id == 'm10_2tri':
        return HarmonicALMBarrierStrategy(time_budget_s=TIME_BUDGET_S)
    if method_id == 'schwarz_wrap_m10_2tri':
        return SchwarzWrapperStrategy(
            inner=HarmonicALMBarrierStrategy(time_budget_s=TIME_BUDGET_S),
            time_budget_s=TIME_BUDGET_S,
        )
    if method_id == 'm14_2tri':
        return HarmonicALMRefineRepairStrategy(time_budget_s=TIME_BUDGET_S)
    if method_id == 'm14_schwarz_2tri':
        return SchwarzHarmonicALMRefineRepairStrategy(time_budget_s=TIME_BUDGET_S)
    raise KeyError(f'unknown method_id={method_id!r}')


def _make_constraint(constraint_label: str, shape: tuple[int, int]):
    from dvfopt import JdetConstraint2D, TriConstraint2DFullCoverage

    if constraint_label == 'jdet':
        return JdetConstraint2D(shape=shape)
    if constraint_label == '2tri':
        return TriConstraint2DFullCoverage(shape=shape)
    raise KeyError(f'unknown constraint_label={constraint_label!r}')


# ---------------------------------------------------------------------------
# Case discovery + selection
# ---------------------------------------------------------------------------


def _init_stats(phi: np.ndarray) -> dict:
    """Return ``{jdet_n_neg, jdet_min, tri_n_neg, tri_min}`` for ``phi``."""
    from dvfopt.jacobian.numpy_jdet import jacobian_det2D
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

    J = jacobian_det2D(phi)[0]
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return dict(
        jdet_n_neg=int((J <= 0).sum()),
        jdet_min=float(J.min()),
        tri_n_neg=int((np.minimum(T1, T2) <= 0).sum()),
        tri_min=float(min(T1.min(), T2.min())),
    )


def _build_b0039_cases(slice_count_per_bucket: int) -> list[Case]:
    """Equal-density-bucket selection across the B0039 volume.

    Picks roughly ``slice_count_per_bucket`` slices from each density
    bucket (easy / medium / hard / extreme), spread uniformly within
    the bucket by tri_n_neg rank. Forces z=0 and z=12 to be included
    (the canonical extreme-density references)."""
    if not B0039_PATH.exists():
        _log(f'WARNING: B0039 DVF not found at {B0039_PATH}; skipping b0039 dataset')
        return []

    _log(f'loading B0039 DVF: {B0039_PATH}')
    phi_volume = np.load(B0039_PATH).astype(np.float64)
    _log(f'B0039 shape={phi_volume.shape} dtype={phi_volume.dtype}')

    # Per-slice initial stats — cache the FULL 528-slice scan so re-runs
    # bucket-select from the same population. (The selected-cases
    # CASE_SCAN_CSV is rewritten downstream after picking.)
    if B0039_FULL_SCAN_CSV.exists():
        _log(f'reusing existing {B0039_FULL_SCAN_CSV.name} for B0039 init stats')
        with open(B0039_FULL_SCAN_CSV, encoding='utf-8') as f:
            scan_by_z = {
                int(r['slice_z']): dict(
                    jdet_n_neg=int(r['jdet_n_neg']),
                    jdet_min=float(r['jdet_min']),
                    tri_n_neg=int(r['tri_n_neg']),
                    tri_min=float(r['tri_min']),
                )
                for r in csv.DictReader(f)
            }
    else:
        scan_by_z = {}
        _log('scanning all B0039 slices for initial fold density ...')
        t0 = time.time()
        for z in range(phi_volume.shape[1]):
            s = phi_volume[1:, z]
            scan_by_z[z] = _init_stats(s)
        _log(f'B0039 scan complete in {time.time() - t0:.1f}s')
        # Persist for next run.
        with open(B0039_FULL_SCAN_CSV, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(
                f, fieldnames=['slice_z', 'jdet_n_neg', 'jdet_min', 'tri_n_neg', 'tri_min']
            )
            w.writeheader()
            for z in sorted(scan_by_z.keys()):
                w.writerow({'slice_z': z, **scan_by_z[z]})

    # Select within each bucket: rank by tri_n_neg, take evenly-spaced ranks.
    picks: list[int] = []
    for bucket_name, lo, hi in B0039_BUCKETS:
        in_bucket = sorted(
            (z for z, s in scan_by_z.items() if lo <= s['tri_n_neg'] < hi),
            key=lambda z: scan_by_z[z]['tri_n_neg'],
        )
        if not in_bucket:
            _log(f'  bucket "{bucket_name}" empty — no slices in [{lo}, {hi})')
            continue
        n = min(slice_count_per_bucket, len(in_bucket))
        # Evenly-spaced ranks across the bucket.
        idxs = [int(round((i + 0.5) * (len(in_bucket) - 1) / max(1, n - 1))) if n > 1 else 0
                for i in range(n)]
        idxs = sorted(set(min(i, len(in_bucket) - 1) for i in idxs))
        chosen = [in_bucket[i] for i in idxs]
        picks.extend(chosen)
        _log(f'  bucket "{bucket_name}" [{lo},{hi}): {len(in_bucket)} avail, picked {chosen}')

    # Always force z=0 and z=12.
    for forced in (0, 12):
        if forced in scan_by_z and forced not in picks:
            picks.append(forced)
    picks = sorted(set(picks))

    cases: list[Case] = []
    for z in picks:
        s = scan_by_z[z]
        phi = phi_volume[1:, z].astype(np.float64).copy()
        cases.append(
            Case(
                case_id=f'b0039_z{z:03d}',
                dataset='b0039',
                phi=phi,
                slice_z=z,
                **s,
            )
        )
    return cases


def _build_canonical_cases() -> list[Case]:
    """The 6 canonical synthetic 2-tri 2D cases from :mod:`test_cases`."""
    try:
        from test_cases import canonical_2tri_2d
    except Exception as exc:
        _log(f'WARNING: failed to import test_cases.canonical_2tri_2d: {exc}')
        return []
    cases: list[Case] = []
    for name, phi, _meta in canonical_2tri_2d():
        phi = np.asarray(phi, dtype=np.float64)
        s = _init_stats(phi)
        cases.append(
            Case(
                case_id=f'canonical_{name}',
                dataset='canonical',
                phi=phi,
                slice_z=-1,
                **s,
            )
        )
    return cases


def _save_case_scan(cases: list[Case]) -> None:
    rows = [
        dict(
            case_id=c.case_id,
            dataset=c.dataset,
            slice_z=c.slice_z,
            case_shape=f'{c.phi.shape[1]}x{c.phi.shape[2]}',
            jdet_n_neg=c.jdet_n_neg,
            jdet_min=c.jdet_min,
            tri_n_neg=c.tri_n_neg,
            tri_min=c.tri_min,
        )
        for c in cases
    ]
    if not rows:
        return
    with open(CASE_SCAN_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def _lib_versions() -> dict:
    """Pin library versions onto every row for reproducibility."""
    import numpy
    import scipy

    try:
        import dvfopt

        dv = getattr(dvfopt, '__version__', 'unknown')
    except Exception:
        dv = 'unknown'
    return dict(
        numpy_version=str(numpy.__version__),
        scipy_version=str(scipy.__version__),
        dvfopt_version=str(dv),
    )


_LIB_VERSIONS = None


def _cached_lib_versions() -> dict:
    """Compute lib versions once per process (called per-row)."""
    global _LIB_VERSIONS
    if _LIB_VERSIONS is None:
        _LIB_VERSIONS = _lib_versions()
    return _LIB_VERSIONS


def _classify_convergence(*, feasible: bool, wall_s: float, budget_s: float, error_kind: str) -> str:
    """Categorise the outcome of one run."""
    if error_kind:
        return 'error'
    if feasible:
        return 'converged'
    if wall_s >= 0.9 * budget_s:
        # Within 10% of the budget — treat as a timeout.
        return 'timeout'
    return 'stalled'


def _extract_cluster_count(info) -> int:
    """Count how many per-cluster sub-runs Schwarz reported, or -1 if N/A.

    The schwarz pipelines emit a dict with a top-level ``cluster_runs``
    list. ``_build_solve_info`` turns each top-level key into a phase;
    list-valued entries land in ``PhaseInfo.extras['value']``. So the
    cluster count lives at:

        info.extras['cluster_runs']           # dict-extras path
        phase('cluster_runs').extras['value']  # stage-keyed path
    """
    extras = getattr(info, 'extras', {}) or {}
    if isinstance(extras, dict):
        runs = extras.get('cluster_runs')
        if isinstance(runs, list):
            return len(runs)
    for phase in getattr(info, 'phases', None) or []:
        if getattr(phase, 'name', '') == 'cluster_runs':
            value = (getattr(phase, 'extras', None) or {}).get('value')
            if isinstance(value, list):
                return len(value)
    return -1


def _run_one(case: Case, method_id: str, constraint_label: str) -> dict:
    """Solve one (case, constraint, method) triple. Return a CSV row dict."""
    from dvfopt import L1Objective, Solver

    shape = case.phi.shape[1:]
    row: dict = {
        'case_id': case.case_id,
        'dataset': case.dataset,
        'slice_z': case.slice_z,
        'case_shape': f'{shape[0]}x{shape[1]}',
        'constraint': constraint_label,
        'method_id': method_id,
        'strategy': '',
        'objective': 'l1',
        'init_n_neg_jdet': case.jdet_n_neg,
        'init_n_neg_2tri': case.tri_n_neg,
        'init_min_T_jdet': case.jdet_min,
        'init_min_T_2tri': case.tri_min,
        'final_n_neg': -1,
        'final_min_T': float('nan'),
        'final_n_neg_jdet': -1,
        'final_min_T_jdet': float('nan'),
        'final_n_neg_2tri': -1,
        'final_min_T_2tri': float('nan'),
        'feasible': False,
        'l1_total': float('nan'),
        'l1_mean': float('nan'),
        'l2_total': float('nan'),
        'linf': float('nan'),
        'wall_time_s': float('nan'),
        'n_phases': -1,
        'total_iter': -1,
        'cluster_count': -1,
        'convergence_reason': '',
        'error_kind': '',
        'error_msg': '',
        'threshold': THRESHOLD,
        'eps_l1': EPS_L1,
        'time_budget_s': TIME_BUDGET_S,
        'git_sha': _git_sha(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        **_cached_lib_versions(),
    }

    try:
        strategy = _make_strategy(method_id)
        row['strategy'] = type(strategy).__name__
        constraint = _make_constraint(constraint_label, shape)
        objective = L1Objective(eps=EPS_L1)
        solver = Solver(
            constraint=constraint,
            objective=objective,
            strategy=strategy,
            threshold=THRESHOLD,
            err_tol=ERR_TOL,
        )
    except Exception as exc:
        row['error_kind'] = f'setup:{type(exc).__name__}'
        row['error_msg'] = str(exc).splitlines()[0][:300]
        return row

    t0 = time.perf_counter()
    try:
        # ``record_history=True`` populates ``result.info.phases`` so
        # ``n_phases`` / ``total_iter`` / ``cluster_count`` end up with
        # real values instead of zeros.
        result = solver.fit(case.phi.copy(), record_history=True)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        row['wall_time_s'] = time.perf_counter() - t0
        row['error_kind'] = f'solve:{type(exc).__name__}'
        row['error_msg'] = (
            str(exc).splitlines()[0][:300] if str(exc) else traceback.format_exc(limit=1)[:300]
        )
        row['convergence_reason'] = _classify_convergence(
            feasible=False,
            wall_s=row['wall_time_s'],
            budget_s=TIME_BUDGET_S,
            error_kind=row['error_kind'],
        )
        return row
    row['wall_time_s'] = time.perf_counter() - t0

    diff = (result.corrected - case.phi).ravel()
    row['final_n_neg'] = int(result.final_n_neg)
    row['final_min_T'] = float(result.final_min_T)
    row['feasible'] = bool(result.feasible)
    row['l1_total'] = float(np.abs(diff).sum())
    row['l1_mean'] = float(np.abs(diff).mean())
    row['l2_total'] = float(np.sqrt((diff * diff).sum()))
    row['linf'] = float(np.abs(diff).max())
    info = result.info
    if hasattr(info, 'phases'):
        row['n_phases'] = len(info.phases)
        row['total_iter'] = int(getattr(info, 'total_iter', 0))
    row['cluster_count'] = _extract_cluster_count(info)
    row['convergence_reason'] = _classify_convergence(
        feasible=row['feasible'],
        wall_s=row['wall_time_s'],
        budget_s=TIME_BUDGET_S,
        error_kind='',
    )

    # Both-view residual stats on the corrected field.
    post = _init_stats(result.corrected)
    row['final_n_neg_jdet'] = post['jdet_n_neg']
    row['final_min_T_jdet'] = post['jdet_min']
    row['final_n_neg_2tri'] = post['tri_n_neg']
    row['final_min_T_2tri'] = post['tri_min']
    return row


# ---------------------------------------------------------------------------
# Resume + CSV append
# ---------------------------------------------------------------------------


def _completed_keys(retry_errors: bool) -> set[tuple[str, str, str]]:
    """Return ``(case_id, constraint, method_id)`` triples already recorded.

    Rows with non-empty ``error_kind`` are excluded when ``retry_errors``
    is True so they get re-run."""
    if not RESULTS_CSV.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    with open(RESULTS_CSV, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if retry_errors and r.get('error_kind'):
                continue
            keys.add((r['case_id'], r['constraint'], r['method_id']))
    return keys


def _append_row(row: dict) -> None:
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        out = {}
        for k in CSV_COLUMNS:
            v = row.get(k, '')
            if isinstance(v, float):
                out[k] = '' if np.isnan(v) else f'{v:.6g}'
            else:
                out[k] = v
        w.writerow(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        '--slice-count',
        type=int,
        default=5,
        help='Slices per B0039 difficulty bucket (default 5 — 4 buckets → 20 b0039 slices).',
    )
    p.add_argument(
        '--no-canonical',
        action='store_true',
        help='Skip the canonical synthetic suite (B0039 only).',
    )
    p.add_argument(
        '--no-b0039',
        action='store_true',
        help='Skip the B0039 slices (canonical only).',
    )
    p.add_argument(
        '--cases',
        type=str,
        default=None,
        help='Comma-separated case_id values, overrides bucket selection.',
    )
    p.add_argument(
        '--methods',
        type=str,
        default=None,
        help='Comma-separated method_id substrings to filter (e.g. "barrier,m14").',
    )
    p.add_argument(
        '--retry-errors',
        action='store_true',
        help='Re-attempt rows previously recorded with non-empty error_kind.',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the work plan without solving.',
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _log(f'b0039_l1_comparison starting  git={_git_sha()}  pid={os.getpid()}')
    _log(f'output dir: {OUT_DIR}')

    cases: list[Case] = []
    if not args.no_b0039:
        cases.extend(_build_b0039_cases(args.slice_count))
    if not args.no_canonical:
        cases.extend(_build_canonical_cases())

    if args.cases:
        wanted = set(s.strip() for s in args.cases.split(',') if s.strip())
        cases = [c for c in cases if c.case_id in wanted]

    _save_case_scan(cases)
    _log(f'benchmark cases ({len(cases)}):')
    for c in cases:
        _log(
            f'  {c.case_id:<30s}  shape={c.phi.shape[1]}x{c.phi.shape[2]}  '
            f'tri_n_neg={c.tri_n_neg:>4d}  jdet_n_neg={c.jdet_n_neg:>4d}'
        )

    method_filter = None
    if args.methods:
        method_filter = [s.strip() for s in args.methods.split(',') if s.strip()]
    methods = [m for m in METHOD_SPECS if method_filter is None or any(f in m[0] for f in method_filter)]
    _log(f'methods ({len(methods)}): {[m[0] for m in methods]}')

    done = _completed_keys(args.retry_errors)
    _log(f'resume: {len(done)} rows already in results.csv (will skip)')

    plan: list[tuple[Case, str, str]] = []
    for case in cases:
        for method_id, constraint_label, _ in methods:
            key = (case.case_id, constraint_label, method_id)
            if key in done:
                continue
            plan.append((case, constraint_label, method_id))
    _log(f'work plan: {len(plan)} runs to execute')

    if args.dry_run:
        for case, cl, mid in plan:
            print(f'  {case.case_id:<30s}  constraint={cl:<5s}  method={mid}')
        return 0

    interrupted = False
    for i, (case, constraint_label, method_id) in enumerate(plan, 1):
        if interrupted:
            break
        _log(
            f'[{i}/{len(plan)}] {case.case_id}  constraint={constraint_label}  '
            f'method={method_id}  init_n_neg_2tri={case.tri_n_neg}  '
            f'init_n_neg_jdet={case.jdet_n_neg}'
        )
        try:
            row = _run_one(case, method_id, constraint_label)
        except KeyboardInterrupt:
            _log('KeyboardInterrupt — saving partial state and exiting')
            interrupted = True
            break

        _append_row(row)
        if row['error_kind']:
            _log(
                f'   FAIL  {row["error_kind"]}  wall={row["wall_time_s"]:.2f}s  '
                f'msg={row["error_msg"][:120]}'
            )
        else:
            feas = 'feas' if row['feasible'] else 'NOT feas'
            _log(
                f'   {feas:>8}  wall={row["wall_time_s"]:.2f}s  '
                f'final_2tri={row["final_n_neg_2tri"]}  final_jdet={row["final_n_neg_jdet"]}  '
                f'L1={row["l1_total"]:.3f}  L2={row["l2_total"]:.3f}'
            )

    _log('done')
    return 130 if interrupted else 0


if __name__ == '__main__':
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
