"""Hybrid: harmonic-extension seed + torch 2-tri barrier polish + ALM cleanup.

Strategy: 3 stages, each one only kicks in if the previous didn't reach
feasibility:

  Stage 1.  Harmonic extension over fold cores (m02).
            Should give us a feasible field at the cost of L2 distance
            in the cores.

  Stage 2.  Torch full-grid log-barrier (m05) initialised from stage-1
            output. The barrier *cannot* cross the wall, so it stays
            feasible and just polishes L2 distance back down where it can.

  Stage 3.  If anything is still infeasible (margin not reached), an
            augmented Lagrangian (m03) pass picks it up.

This is the most likely method to give both 100% 2-tri feasibility AND
acceptable L2 distance on the dense slices.
"""
from __future__ import annotations

import time
import numpy as np

from . import m02_harmonic_extension as harmonic
from . import m03_augmented_lagrangian as alm
from . import m05_torch_full_grid as torch_m
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'harmonic+polish'
DESCRIPTION = 'Harmonic-extension seed -> torch barrier polish -> ALM cleanup (best-effort hybrid)'


def _min_tri(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(np.minimum(T1, T2).min())


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          stage_budgets_s: tuple = (60.0, 240.0, 240.0),
          use_alm_cleanup: bool = True,
          verbose: int = 0) -> dict:
    t0 = time.time()
    info = {}

    # Stage 1: harmonic extension seed.
    r1 = harmonic.solve(phi_in, threshold=threshold,
                        ring_pad=2, max_grow_iters=8, margin=0.0)
    phi_cur = r1['phi_out']
    info['stage1_harmonic'] = dict(min_T=_min_tri(phi_cur),
                                   wall=time.time() - t0,
                                   patches=r1['info'].get('patches'))
    if verbose:
        print(f'  stage1 harmonic  min_T={_min_tri(phi_cur):+.5f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    # Stage 2: torch barrier polish (interior point; can't lose feasibility).
    if _min_tri(phi_cur) > threshold:  # interior point applicable
        r2 = torch_m.solve(phi_cur, threshold=threshold, margin=margin,
                           anchor='l2',
                           time_budget_s=stage_budgets_s[1],
                           verbose=0)
        # Only accept if it didn't break feasibility.
        if _min_tri(r2['phi_out']) > _min_tri(phi_cur) - 1e-5:
            phi_cur = r2['phi_out']
        info['stage2_torch_polish'] = dict(min_T=_min_tri(phi_cur),
                                           wall=time.time() - t0)
    else:
        # Not strictly feasible at threshold yet; skip barrier polish.
        info['stage2_torch_polish'] = dict(skipped='infeasible-seed',
                                           wall=time.time() - t0)

    if verbose:
        print(f'  stage2 torch     min_T={_min_tri(phi_cur):+.5f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    # Stage 3: ALM cleanup if still below target.
    if use_alm_cleanup and _min_tri(phi_cur) < threshold + margin:
        remaining = max(60.0, stage_budgets_s[2] - max(0,
                                                       (time.time() - t0) - sum(stage_budgets_s[:2])))
        r3 = alm.solve(phi_cur, threshold=threshold, margin=margin,
                       anchor='l2', outer_max=40,
                       inner_maxiter=150,
                       time_budget_s=remaining,
                       verbose=0)
        if _min_tri(r3['phi_out']) > _min_tri(phi_cur):
            phi_cur = r3['phi_out']
        info['stage3_alm'] = dict(min_T=_min_tri(phi_cur),
                                  wall=time.time() - t0)
    else:
        info['stage3_alm'] = dict(skipped='already-feasible',
                                  wall=time.time() - t0)
    if verbose:
        print(f'  stage3 alm       min_T={_min_tri(phi_cur):+.5f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    info['final_min_T'] = _min_tri(phi_cur)
    return {'phi_out': phi_cur, 'info': info}
