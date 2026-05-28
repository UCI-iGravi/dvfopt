"""L-BFGS-B 2-tri penalty -> log-barrier (manuscript baseline).

This is a thin wrapper around ``dvfopt.core.iterative2d_tri_barrier.
iterative_2d_tri_barrier`` so it can be benchmarked under the wall-
breaker harness with the same per-slice JSON / CSV machinery.

What it does (recap of the existing solver):

* Anchored L2 objective ``0.5 ||phi - phi_init||^2``.
* Phase 1 -- exterior quadratic penalty
      F_pen = anchor + lam * sum max(0, threshold + margin - T_k)^2
  with a growing lam schedule.
* Phase 2 -- log-barrier interior point
      F_bar = anchor - mu * sum log(T_k - threshold)
  with a shrinking mu schedule, only after every T_k > threshold.
* Both phases minimised with scipy L-BFGS-B; analytical J^T v product
  for the 2-tri Jacobian.

This is THE manuscript solver compared against in the wall-breaker
suite: it is the strongest single-stage L-BFGS-B method for the 2-tri
metric on a full slice. It is *not* the cluster-based SLSQP solver
used in ``_run_2d_clusters.py`` for the production run (that one is
SLSQP per cluster, not full-grid L-BFGS-B); see
``manuscript_slsqp_baseline.py`` for that comparison.
"""
from __future__ import annotations

import numpy as np

from dvfopt.core.iterative2d_tri_barrier import iterative_2d_tri_barrier
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'lbfgs_barrier'
DESCRIPTION = ('Full-grid 2-tri L-BFGS-B penalty -> log-barrier '
               '(dvfopt.core.iterative2d_tri_barrier)')


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          anchor: str = 'l2',
          lam_schedule: tuple = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
          mu_schedule: tuple = (1e-1, 1e-2, 1e-3, 1e-4),
          max_minimize_iter: int = 300,
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    """Run the existing 2-tri penalty/barrier L-BFGS-B solver and return
    in the wall-breaker harness's solve() contract.

    The ``time_budget_s`` parameter is accepted for the harness but the
    underlying solver doesn't expose a time limit; in practice each
    full slice finishes well within a few minutes on the dense cases.
    """
    phi_corrected = iterative_2d_tri_barrier(
        phi_in.copy(), threshold=threshold, margin=margin,
        lam_schedule=lam_schedule, mu_schedule=mu_schedule,
        max_minimize_iter=max_minimize_iter, anchor=anchor,
        verbose=verbose, record_history=False)

    T1, T2 = _triangle_areas_2d(phi_corrected[0], phi_corrected[1])
    final_min = float(np.minimum(T1, T2).min())
    return {'phi_out': phi_corrected,
            'info': {'final_min_T': final_min,
                     'anchor': anchor,
                     'lam_max': float(lam_schedule[-1]),
                     'mu_min': float(mu_schedule[-1])}}
