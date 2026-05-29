"""L1-anchored variant of m14_l2_refine_repair.

Same 4-stage pipeline (m10 seed -> soft-penalty pull -> harmonic repair
-> log-barrier polish) but every objective term uses a smoothed-L1
anchor:

.. math::

    R(\\phi) = \\sum_i \\sqrt{(\\phi_i - \\phi^{\\text{in}}_i)^2 + \\varepsilon^2}

The L1 anchor tolerates a few large local deviations cheaply (the
penalty scales linearly, not quadratically, in the deviation), and is
the natural fit when most of the field can return exactly to phi_in
and only a few corners must move far to satisfy feasibility.

We measure both L2 and L1 of the correction in this run so the
trade-off vs the L2-anchored m14 is visible side-by-side.
"""
from __future__ import annotations

import numpy as np

from . import m14_l2_refine_repair as base

NAME = 'l1_refine_repair'
DESCRIPTION = ('m14 pipeline (m10 -> soft-penalty pull -> repair -> '
               'polish) with smoothed-L1 anchor instead of L2')


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          seed: np.ndarray | None = None,
          lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8),
          inner_maxiter: int = 200,
          ring_pad: int = 2,
          polish_mu: tuple = (1e-2, 1e-4, 1e-6),
          polish_maxiter: int = 200,
          time_budget_s: float = 360.0,
          verbose: int = 0,
          eps_l1: float = 1e-4) -> dict:
    return base.solve(
        phi_in, threshold=threshold, margin=margin,
        anchor='l1',
        seed=seed,
        lam_schedule=lam_schedule, inner_maxiter=inner_maxiter,
        ring_pad=ring_pad,
        polish_mu=polish_mu, polish_maxiter=polish_maxiter,
        time_budget_s=time_budget_s, verbose=verbose, eps_l1=eps_l1)
