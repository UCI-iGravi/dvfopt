"""Sequential-LP (SLP) strategy for the 2D 2-triangle constraint.

The L1-minimising strict-feasibility champion (``auto_slp``), promoted from
``research/strict_feasibility_2d`` into the installable package. On the
B0039 benchmark it Pareto-dominates the wallbreaker M14 (~3–5× wall at
equal-or-better L1) and reaches strict feasibility on every slice.

Mechanism (see :mod:`dvfopt.core.slp`):

* **Seed** the field feasible with an m14-family seed (harmonic → ALM →
  L2-refine), reusing the package wallbreaker strategies.
* **Trust-region SLP**: repeatedly linearise the 2-tri areas, solve an
  L1-epigraph LP (HiGHS) inside a trust region, accept on exact-area
  feasibility.
* For large slices, **decompose into fold clusters** solved concurrently
  with a continuous (as-completed) scheduler and a frozen-ring splice
  (:func:`dvfopt.core.slp.cluster_slp_iter`); small slices use the global
  :func:`dvfopt.core.slp.slp_iter`. This auto-routing matches ``auto_slp``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvfopt.constraints import TriConstraint2D, TriConstraint2DFullCoverage
from dvfopt.strategies.base import (
    Strategy,
    _build_solve_info,
    register_strategy,
)


@register_strategy('slp')
@dataclass
class SLPStrategy(Strategy):
    """Per-cluster trust-region sequential-LP — the 2-tri L1 champion.

    Parameters
    ----------
    n_workers : int, default 16
        Process-pool size for the per-cluster solves (large-slice path).
    scheduler : {'continuous', 'subround'}, default 'continuous'
        Cluster scheduling. ``'continuous'`` (as-completed) keeps the pool
        full; ``'subround'`` uses barrier sub-rounds. Continuous is the
        shipped default (~1.01–1.16× wall, L1-identical).
    max_outer_iters : int, default 6
        Outer re-cluster / splice-cleanup rounds.
    cluster_seed : str, default 'm14_fast'
        Per-cluster seed kind (large-slice path).
    global_seed : str, default 'm14'
        Seed kind for the small-slice global ``slp_iter`` path.
    cluster_pixel_threshold : int, default 5000
        Slices with ``H*W`` above this take the cluster path; smaller
        slices use the global LP (cheap enough un-decomposed).
    """

    n_workers: int = 16
    scheduler: str = 'continuous'
    max_outer_iters: int = 6
    cluster_seed: str = 'm14_fast'
    global_seed: str = 'm14'
    cluster_pixel_threshold: int = 5000

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        **_,
    ):
        from dvfopt.core.slp import cluster_slp_iter, slp_iter

        self._check_constraint(constraint)
        # 2-tri pack is DY_FIRST: phi_in is (2, H, W) = [dy, dx], exactly
        # what the SLP solvers consume. The objective is implicit (the LP
        # minimises L1 deviation from phi_in); ``threshold`` is the per-tri
        # area lower bound.
        phi = np.asarray(phi_in, dtype=np.float64)
        H, W = phi.shape[1:]
        if H * W <= self.cluster_pixel_threshold:
            phi_out, info = slp_iter(
                phi, threshold=threshold, seed=self.global_seed,
            )
            info = {**info, 'slp_dispatch': 'global'}
        else:
            phi_out, info = cluster_slp_iter(
                phi, threshold=threshold,
                max_outer_iters=self.max_outer_iters,
                n_workers=self.n_workers, scheduler=self.scheduler,
                inner_seed=self.cluster_seed,
            )
            info = {**info, 'slp_dispatch': 'cluster'}
        return phi_out, _build_solve_info('SLPStrategy', info, threshold)


__all__ = ['SLPStrategy']
