"""Cluster-localized m14 (refine-repair) for the 3D 6-tet constraint.

Thin closure-shim around the generic
:func:`dvfopt.core.wallbreakers._schwarz_common.cluster_schwarz_3d_tet`,
binding ``inner_solve`` to
:func:`dvfopt.core.wallbreakers._refine_repair_3d.iterative_3d_tet_refine_repair`.

The schwarz domain-decomposition logic lives once in
``_schwarz_common``; this module just constructs the per-strategy
callback and hands off.

For now no final global polish (unlike the 2D variant) — the
per-cluster m14-3D already runs its own log-barrier polish stage on
each crop. A composite global polish can be wired in here later by
passing ``final_polish_fn`` to ``cluster_schwarz_3d_tet``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core.wallbreakers._refine_repair_3d import iterative_3d_tet_refine_repair
from dvfopt.core.wallbreakers._schwarz_common import cluster_schwarz_3d_tet


def iterative_3d_tet_refine_repair_schwarz(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    anchor: str = 'l2',
    eps_l1: float = 1e-4,
    pad: int = 4,
    merge_dilation: int = 2,
    max_outer_iters: int = 3,
    fallback_size_ratio: float = 0.7,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
    step_callback=None,
    **m14_kwargs,
):
    """Cluster-localized 3D refine-repair (m14-Schwarz-3D).

    See :mod:`dvfopt.core.wallbreakers._schwarz_common` for the
    underlying algorithm. This entry point pins ``inner_solve`` to
    :func:`iterative_3d_tet_refine_repair`.

    Parameters
    ----------
    phi_in : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    threshold, margin, anchor, eps_l1
        Forwarded to per-cluster :func:`iterative_3d_tet_refine_repair`.
    pad, merge_dilation, max_outer_iters, fallback_size_ratio,
    time_budget_s, verbose, record_history
        Forwarded to :func:`cluster_schwarz_3d_tet`.
    **m14_kwargs
        Forwarded to :func:`iterative_3d_tet_refine_repair`.

    Returns
    -------
    phi_out : ndarray, shape ``(3, D, H, W)``.
    info : dict, only if ``record_history=True``.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    def inner_solve(phi_crop: np.ndarray, time_budget_s: Optional[float] = None):
        return iterative_3d_tet_refine_repair(
            phi_crop,
            threshold=threshold,
            margin=margin,
            anchor=anchor,
            eps_l1=eps_l1,
            time_budget_s=time_budget_s if time_budget_s is not None else 600.0,
            verbose=max(0, verbose - 1),
            step_callback=step_callback,
            **m14_kwargs,
        )

    return cluster_schwarz_3d_tet(
        phi_in,
        inner_solve,
        threshold=threshold,
        pad=pad,
        merge_dilation=merge_dilation,
        max_outer_iters=max_outer_iters,
        fallback_size_ratio=fallback_size_ratio,
        time_budget_s=time_budget_s,
        final_polish_fn=None,
        verbose=verbose,
        record_history=record_history,
    )


__all__ = ['iterative_3d_tet_refine_repair_schwarz']
