"""2-triangle SLP (sequential linear programming) solver — the L1-minimising
strict-feasibility champion, promoted from ``research/strict_feasibility_2d``.

Per-cluster trust-region SLP with an m14 seed, solved via HiGHS
(``scipy.optimize.linprog``), with a continuous (as-completed) parallel
cluster scheduler. Exposed to the package through
:class:`dvfopt.strategies.SLPStrategy`.

Public entry points:

* :func:`cluster_slp_iter` — per-cluster SLP over a full slice (the
  ``auto_slp`` large-slice path); ``scheduler='continuous'`` is the default
  in the strategy.
* :func:`slp_iter` — global (non-clustered) trust-region SLP.
* :func:`lp_oneshot` — single LP linearised around a feasible seed.
* :func:`linearize_T_2tri` / :func:`build_sparse_jacobian_T` — the 2-tri
  constraint linearisation.
* :func:`solve_l1_lp_step` — one L1-epigraph LP step (HiGHS backend).
"""

from dvfopt.core.slp.cluster_lp_2tri import cluster_slp_iter
from dvfopt.core.slp.highs_solver import solve_l1_lp_step
from dvfopt.core.slp.lp_direct_2tri import lp_oneshot, slp_iter
from dvfopt.core.slp.tri_linearize import (
    build_sparse_jacobian_T,
    linearize_T_2tri,
)

__all__ = [
    'build_sparse_jacobian_T',
    'cluster_slp_iter',
    'linearize_T_2tri',
    'lp_oneshot',
    'slp_iter',
    'solve_l1_lp_step',
]
