"""Per-method dispatch + uniform metric record.

``run_method(name, phi_2hw) -> dict`` runs ``name`` on ``phi_2hw`` and
returns a dict with all metrics specified in the design spec.
"""
from __future__ import annotations

import time

import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from research.strict_feasibility_2d.algorithms.lp_direct_2tri import (
    lp_oneshot,
    slp_iter,
)

THRESHOLD = 0.01
SAFETY_TOL = 1e-5

METHOD_NAMES = (
    'harmonic_only',
    'm10',
    'm14',
    'm14_schwarz',
    'cluster_pipeline',
    'lp_oneshot',
    'slp_iter',
    'slp_iter_m14_seed',
    'slp_iter_wide_tr',
    'cluster_slp',
    'auto_slp',
)

# auto_slp dispatch threshold. One signal now matters:
#   - Pixel count: below ~5k px, LP at slice scale is cheap enough
#     that slp_iter_m14_seed beats cluster_slp on L1 (no per-cluster
#     M14-inner overhead).
#
# After commit 148049d added threshold-aware re-clustering to
# cluster_slp, the per-cluster approach now wins both L1 AND wall
# against M14 even on sparse-fold large slices (the old "polish-fires-
# expensively" failure mode no longer fires). So the fold-count
# branch is obsolete — large slices always route to cluster_slp.
_AUTO_CLUSTER_PIXEL_THRESHOLD = 5_000


def _stats(phi_2hw: np.ndarray):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    T_min = np.minimum(T1, T2)
    return {
        'n_neg_2tri': int((T_min <= 0).sum()),
        'min_T': float(T_min.min()),
    }


def _solve_via_strategy(strategy_cls, phi_2hw: np.ndarray):
    """Wrap the v0.2 ``Solver`` + Strategy API for a (2, H, W) field."""
    from dvfopt import L1Objective, Solver, TriConstraint2DFullCoverage

    H, W = phi_2hw.shape[1:]
    constraint = TriConstraint2DFullCoverage(shape=(H, W))
    objective = L1Objective(eps=1e-4)
    strategy = strategy_cls()
    solver = Solver(
        constraint=constraint, objective=objective, strategy=strategy, threshold=THRESHOLD
    )
    result = solver.fit(phi_2hw)
    return result.corrected


def _dispatch(name: str, phi_2hw: np.ndarray):
    """Return ``(phi_out, extra_info_dict)``."""
    if name == 'harmonic_only':
        from dvfopt.core.wallbreakers import harmonic_extension_2d
        phi_out = harmonic_extension_2d(phi_2hw, threshold=THRESHOLD)
        return phi_out, {}
    if name == 'm10':
        from dvfopt import HarmonicALMBarrierStrategy
        phi_out = _solve_via_strategy(HarmonicALMBarrierStrategy, phi_2hw)
        return phi_out, {}
    if name == 'm14':
        from dvfopt import HarmonicALMRefineRepairStrategy
        phi_out = _solve_via_strategy(HarmonicALMRefineRepairStrategy, phi_2hw)
        return phi_out, {}
    if name == 'm14_schwarz':
        from dvfopt import SchwarzHarmonicALMRefineRepairStrategy
        phi_out = _solve_via_strategy(SchwarzHarmonicALMRefineRepairStrategy, phi_2hw)
        return phi_out, {}
    if name == 'cluster_pipeline':
        # Not yet wired. ``notebooks/manuscript/_run_2d_clusters.py::process_one_slice``
        # takes (z, phi_full, phi_anchor_full, executor) and depends on module-level
        # globals. A clean adapter is its own follow-up task.
        raise NotImplementedError(
            'cluster_pipeline adapter not yet implemented; see Task 9 note in plan'
        )
    if name == 'lp_oneshot':
        phi_out, info = lp_oneshot(phi_2hw, threshold=THRESHOLD)
        return phi_out, info
    if name == 'slp_iter':
        phi_out, info = slp_iter(phi_2hw, threshold=THRESHOLD)
        return phi_out, info
    if name == 'slp_iter_m14_seed':
        # Seed from m14 (closest-to-phi_in feasible point we have).
        # SLP can only polish further; never worse than m14 on L1.
        phi_out, info = slp_iter(phi_2hw, threshold=THRESHOLD, seed='m14')
        return phi_out, info
    if name == 'slp_iter_wide_tr':
        # Wide initial trust region (2 cell units) lets the first LP step
        # cover ~full-displacement inputs in one shot. Same m10 seed.
        phi_out, info = slp_iter(phi_2hw, threshold=THRESHOLD, trust_radius_0=2.0)
        return phi_out, info
    if name == 'cluster_slp':
        # Per-cluster SLP with m14 seed per cluster. Makes the LP
        # tractable at B0039 scale by avoiding the 290k-var direct solve.
        from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
            cluster_slp_iter,
        )
        phi_out, info = cluster_slp_iter(
            phi_2hw, threshold=THRESHOLD, inner_seed='m14'
        )
        return phi_out, info
    if name == 'auto_slp':
        # Adaptive: route by pixel count to the empirical winner.
        # - Small slice (≤5k px): slp_iter_m14_seed (best L1; small
        #   enough that the global LP is cheap).
        # - Large slice (>5k px): cluster_slp with n_workers=8
        #   parallelism (per-cluster scaling plus threshold-aware
        #   re-clustering plus shared-pool parallelism: 3-4× wall win
        #   on B0039 z=100/z=300, identical L1, no help-no-harm on
        #   the densest z=12 slice).
        H, W = phi_2hw.shape[1:]
        pixels = H * W
        if pixels <= _AUTO_CLUSTER_PIXEL_THRESHOLD:
            phi_out, info = slp_iter(phi_2hw, threshold=THRESHOLD, seed='m14')
            info = {**info, 'auto_dispatch': 'slp_iter_m14_seed', 'pixels': pixels}
        else:
            from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
                cluster_slp_iter,
            )
            # n_workers=16 is the universal-win lever (small but consistent
            # ~5-10% over n=8 on every slice). The tempting md=4 +
            # inner_max_iter=5 combo tuned to z=300 (~1.34x) backfires on
            # sparser slices (z=450/500: +46% wall, +65% L1) — kept at
            # defaults to preserve the universal-strict-feasibility goal.
            phi_out, info = cluster_slp_iter(
                phi_2hw, threshold=THRESHOLD, max_outer_iters=6, n_workers=16,
                scheduler='continuous',
            )
            info = {**info, 'auto_dispatch': 'cluster_slp', 'pixels': pixels}
        return phi_out, info
    raise ValueError(f'unknown method: {name!r} (known: {METHOD_NAMES})')


def run_method(name: str, phi_in_2hw: np.ndarray) -> dict:
    """Run ``name`` on ``phi_in_2hw`` and return a metrics record.

    Unknown method names raise ValueError immediately. Errors during
    dispatch (e.g. NotImplementedError, solver failure) are caught and
    recorded in the ``error`` field; the row still returns with
    ``phi_out = phi_in`` so downstream batching keeps going.
    """
    if name not in METHOD_NAMES:
        raise ValueError(f'unknown method: {name!r} (known: {METHOD_NAMES})')
    init = _stats(phi_in_2hw)
    t0 = time.time()
    try:
        phi_out, extra = _dispatch(name, phi_in_2hw)
        error = None
    except Exception as exc:
        phi_out = phi_in_2hw.copy()
        extra = {}
        error = f'{type(exc).__name__}: {exc}'
    wall = time.time() - t0
    final = _stats(phi_out)
    diff = phi_out.astype(np.float64) - phi_in_2hw.astype(np.float64)
    return {
        'method': name,
        'phi_out': phi_out,
        'init_n_neg_2tri': init['n_neg_2tri'],
        'init_min_T': init['min_T'],
        'final_n_neg_2tri': final['n_neg_2tri'],
        'final_min_T': final['min_T'],
        'feasible': final['n_neg_2tri'] == 0 and final['min_T'] >= THRESHOLD - SAFETY_TOL,
        'L1_dev': float(np.abs(diff).sum()),
        'L2_dev': float(np.linalg.norm(diff)),
        'Linf_dev': float(np.max(np.abs(diff))),
        'wall_s': wall,
        'error': error,
        'extra': extra,
    }
