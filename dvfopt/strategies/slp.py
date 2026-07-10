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
    accuracy : {'fast', 'max'}, default 'fast'
        ``'fast'`` runs the SLP path directly on the input (the shipped
        behaviour). ``'max'`` first runs the whole-slice GPU PHR-ALM
        untangler (:func:`dvfopt.core.slp._gpu_untangle.gpu_untangle_alm_2d`)
        to reach a low-L1, mostly-untangled basin, then feeds that field
        into the SLP solver — reaching strict feasibility at ~2x lower L1
        on dense slices. Requires PyTorch. If the untangler hits a CUDA
        out-of-memory error it retries once on the CPU (with a warning).

        Anchoring note: on the small-slice *global* path the LP still
        anchors its L1 objective to the raw input; on the large-slice
        *cluster* path the LP anchors to the GPU-untangled field (by
        design — fold-clustering must run on the near-feasible field).
        In ``'max'`` mode the solve info therefore reports
        ``l1_anchor`` (``'input'`` or ``'gpu_seed'``) and
        ``l1_from_input`` (the true L1 deviation of the output from the
        raw input) so results remain comparable across paths.
    """

    n_workers: int = 16
    scheduler: str = 'continuous'
    max_outer_iters: int = 6
    cluster_seed: str = 'm14_fast'
    global_seed: str = 'm14'
    cluster_pixel_threshold: int = 5000
    accuracy: str = 'fast'

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def __post_init__(self):
        if self.accuracy not in ('fast', 'max'):
            raise ValueError(f"accuracy must be 'fast' or 'max', got {self.accuracy!r}")

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

        # accuracy='max': untangle the whole slice on the GPU first to reach
        # a low-L1, mostly-untangled basin, then run SLP from it. The two
        # dispatch paths use the untangled field differently:
        #   * global path — the untangled field is only the SLP *starting
        #     point*; the LP's L1 objective stays anchored to the raw input.
        #   * cluster path — cluster_slp_iter's first argument is BOTH the
        #     L1 anchor and the cluster-detection field, so there the anchor
        #     is the GPU seed by design (clustering must run on the
        #     near-feasible field, not the raw input's dense fold mass).
        # In 'max' mode we therefore also report `l1_from_input` (true L1
        # vs the raw input) and `l1_anchor` in the solve info.
        phi_raw = phi
        gpu_seed = None
        if self.accuracy == 'max':
            import importlib.util

            # Probe torch explicitly: _gpu_untangle imports torch lazily
            # INSIDE the function, so importing the module itself would
            # succeed even without torch and users would get a raw
            # ModuleNotFoundError from deep inside the call.
            if importlib.util.find_spec('torch') is None:
                raise ImportError("accuracy='max' requires PyTorch (pip install torch).")
            from dvfopt.core.slp._gpu_untangle import gpu_untangle_alm_2d

            try:
                gpu_seed = gpu_untangle_alm_2d(phi, threshold=threshold)
            except Exception as exc:
                import torch

                cuda_oom = getattr(torch.cuda, 'OutOfMemoryError', ())
                if not isinstance(exc, cuda_oom):
                    raise
                import warnings

                warnings.warn(
                    "accuracy='max' GPU untangler hit CUDA out-of-memory; "
                    "retrying once on CPU (slower)."
                )
                gpu_seed = gpu_untangle_alm_2d(phi, threshold=threshold, device='cpu')

        if self.cluster_pixel_threshold >= H * W:
            # Global path: anchor L1 to the raw input `phi`; when max, start
            # the SLP from the untangled field via seed= (an array seed;
            # _build_seed accepts it) instead of recomputing an m14 seed.
            phi_out, info = slp_iter(
                phi,
                threshold=threshold,
                seed=(gpu_seed if gpu_seed is not None else self.global_seed),
            )
            info = {**info, 'slp_dispatch': 'global'}
            l1_anchor = 'input'
        else:
            # Cluster path: when max, run cluster_slp_iter ON the untangled
            # field so fold-clustering operates on the near-feasible field
            # (few residual clusters) rather than the raw input's dense fold
            # mass. cluster_slp_iter uses its first arg as both the L1 anchor
            # and the cluster-detection field, so in this path the anchor IS
            # the GPU seed by construction — meaning the L1 numbers cluster
            # solvers report measure deviation from the GPU seed, NOT the raw
            # input. This anchoring is the validated design (~2x lower L1 vs
            # the raw input than the 'fast' path on dense slices); the true
            # from-input L1 is recorded below as `l1_from_input`.
            phi_out, info = cluster_slp_iter(
                gpu_seed if gpu_seed is not None else phi,
                threshold=threshold,
                max_outer_iters=self.max_outer_iters,
                n_workers=self.n_workers,
                scheduler=self.scheduler,
                inner_seed=self.cluster_seed,
            )
            info = {**info, 'slp_dispatch': 'cluster'}
            l1_anchor = 'gpu_seed' if gpu_seed is not None else 'input'
        info = {**info, 'accuracy': self.accuracy}
        if self.accuracy == 'max':
            info['slp_seed'] = 'gpu'
            info['l1_anchor'] = l1_anchor
            # TRUE L1 vs the raw input (the anchor-relative numbers inside
            # `info` are not comparable across paths in 'max' mode).
            info['l1_from_input'] = float(np.abs(phi_out - phi_raw).sum())
        return phi_out, _build_solve_info('SLPStrategy', info, threshold)


__all__ = ['SLPStrategy']
