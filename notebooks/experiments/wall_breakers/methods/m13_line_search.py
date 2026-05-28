"""Backtracking line search from a feasible seed toward phi_in.

Diagnosis after m12: the soft-penalty L-BFGS-B drops L2 by ~90% but
fails strict feasibility (168 residual folds on z=12). The quadratic
penalty saturates: lam can grow without ever exactly zeroing the
violation in finite gradient steps. We need a method that PROJECTS
back to the feasible manifold instead of penalising it.

Simplest projection: take m10's feasible output, walk a fraction
``alpha`` along the line toward ``phi_in``, keep the largest
``alpha`` that stays feasible. Globally, this is a 1-D line search;
locally, it's a SAFE move because the L2 objective is strictly
convex along the line and the feasible set is a closed manifold.

This single-shot global step probably leaves L2 high (most slices
have many independent feasible directions toward phi_in). To exploit
those, we iterate:

  1. Try alpha in (0, 1]; do a binary-search bisection on
     feasibility; pick the largest feasible alpha.
  2. If alpha < 1 (couldn't reach phi_in), update
     ``phi <- (1-alpha) * phi + alpha * phi_in`` and repeat.
  3. Termination: alpha stops growing, or budget exhausted.

The result is the closest point to ``phi_in`` along straight-line
projections that the feasible manifold allows. It is GUARANTEED
feasibility-preserving and monotone in L2.
"""
from __future__ import annotations

import time
import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'line_search_refine'
DESCRIPTION = ('Backtracking line-search from m10 seed toward phi_in '
               '(monotone L2-reducing, feasibility-preserving)')


def _feasible(phi, threshold):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(min(T1.min(), T2.min())) >= threshold


def _min_tri(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(min(T1.min(), T2.min()))


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          seed: np.ndarray | None = None,
          margin: float = 0.0,
          alpha_init: float = 1.0,
          alpha_min: float = 1e-4,
          max_outer: int = 20,
          time_budget_s: float = 120.0,
          verbose: int = 0) -> dict:
    """Iterated bisection line search from seed toward phi_in."""
    if seed is None:
        from . import m10_harmonic_l2_polished as m10
        seed = m10.solve(phi_in, threshold=threshold,
                         margin=1e-3, time_budget_s=time_budget_s * 0.5,
                         verbose=verbose)['phi_out']

    target_thr = threshold + margin
    phi = seed.copy()
    t0 = time.time()
    log = []
    L2_seed = float(np.linalg.norm((seed - phi_in).ravel()))
    for outer in range(max_outer):
        if time.time() - t0 > time_budget_s:
            break
        # Bisect for largest alpha keeping feasibility.
        lo, hi = 0.0, alpha_init
        # Probe hi first.
        cand_hi = (1 - hi) * phi + hi * phi_in
        if _min_tri(cand_hi) >= target_thr:
            phi = cand_hi
            log.append(dict(outer=outer, alpha=hi, L2=float(np.linalg.norm(
                (phi - phi_in).ravel())), reached='full'))
            if verbose:
                print(f'  outer={outer:2d}  alpha=1.0  reached phi_in  '
                      f'L2={log[-1]["L2"]:.3f}', flush=True)
            break
        # Bisection in [lo, hi].
        for _ in range(40):  # ~1e-12 precision
            mid = 0.5 * (lo + hi)
            cand = (1 - mid) * phi + mid * phi_in
            if _min_tri(cand) >= target_thr:
                lo = mid
            else:
                hi = mid
            if hi - lo < alpha_min:
                break
        if lo < alpha_min:
            # Couldn't move at all this round.
            log.append(dict(outer=outer, alpha=lo, L2=float(np.linalg.norm(
                (phi - phi_in).ravel())), reached='stuck'))
            if verbose:
                print(f'  outer={outer:2d}  alpha={lo:.4e}  stuck  '
                      f'L2={log[-1]["L2"]:.3f}', flush=True)
            break
        phi = (1 - lo) * phi + lo * phi_in
        L2_cur = float(np.linalg.norm((phi - phi_in).ravel()))
        log.append(dict(outer=outer, alpha=lo, L2=L2_cur, reached='partial'))
        if verbose:
            print(f'  outer={outer:2d}  alpha={lo:.4f}  L2={L2_cur:.3f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
        if L2_cur < 1e-6:
            break

    return {'phi_out': phi,
            'info': {'L2_seed': L2_seed,
                     'L2_final': float(np.linalg.norm((phi - phi_in).ravel())),
                     'final_min_T': _min_tri(phi),
                     'outer_used': len(log), 'log': log[-5:]}}
