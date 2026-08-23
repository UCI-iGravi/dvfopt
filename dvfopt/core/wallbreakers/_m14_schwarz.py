"""Cluster-localized m14 (refine-repair) — Schwarz domain decomposition.

Thin closure-shim around the generic
:func:`dvfopt.core.schwarz._common.cluster_schwarz_2d_tri`,
binding ``inner_solve`` to
:func:`dvfopt.core.wallbreakers._refine_repair.iterative_2d_tri_refine_repair`
and ``final_polish_fn`` to
:func:`dvfopt.core.barrier.tri2d.iterative_2d_tri_barrier`.

This file used to host the full schwarz pipeline. Today it just
constructs the callbacks and hands off to ``cluster_schwarz_2d_tri``
so that the schwarz-decomposition logic exists in exactly one place.

See the generic module's docstring for the algorithm description and
the per-cluster / fallback / final-polish behaviour.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core.barrier.tri2d import iterative_2d_tri_barrier
from dvfopt.core.schwarz._common import cluster_schwarz_2d_tri
from dvfopt.core.wallbreakers._refine_repair import (
    iterative_2d_tri_refine_repair,
)


def iterative_2d_tri_refine_repair_schwarz(
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
    final_polish: bool = True,
    final_polish_max_iter: int = 200,
    verbose: int = 1,
    record_history: bool = False,
    step_callback=None,
    **m14_kwargs,
):
    """Cluster-localized refine-repair (m14-Schwarz).

    See :mod:`dvfopt.core.schwarz._common` for the
    underlying algorithm. This entry point pins ``inner_solve`` to
    :func:`iterative_2d_tri_refine_repair` and (optionally) the global
    polish to :func:`iterative_2d_tri_barrier`.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)`` or ``(3, 1, H, W)``.
    threshold, margin, anchor, eps_l1
        Forwarded to per-cluster :func:`iterative_2d_tri_refine_repair`
        (and the optional final barrier polish).
    pad, merge_dilation, max_outer_iters, fallback_size_ratio,
    time_budget_s, verbose, record_history
        Forwarded to :func:`cluster_schwarz_2d_tri`.
    final_polish : bool
        Toggle the optional global :func:`iterative_2d_tri_barrier`
        polish at the end.
    final_polish_max_iter : int
        ``max_minimize_iter`` for the global polish.
    **m14_kwargs
        Forwarded to :func:`iterative_2d_tri_refine_repair` (per-cluster
        + global fallback).

    Returns
    -------
    phi_out : ndarray, shape ``(2, H, W)`` — channels ``[dy, dx]``.
    info : dict, only if ``record_history=True``.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    def inner_solve(phi_crop: np.ndarray, time_budget_s: Optional[float] = None):
        return iterative_2d_tri_refine_repair(
            phi_crop,
            threshold=threshold,
            margin=margin,
            anchor=anchor,
            eps_l1=eps_l1,
            time_budget_s=time_budget_s if time_budget_s is not None else 600.0,
            verbose=0,
            **m14_kwargs,
        )

    final_polish_fn = None
    if final_polish:

        def final_polish_fn(phi: np.ndarray) -> np.ndarray:
            return iterative_2d_tri_barrier(
                phi,
                threshold=threshold,
                margin=margin,
                max_minimize_iter=final_polish_max_iter,
                anchor=anchor,
                eps_l1=eps_l1,
                verbose=0,
            )

    return cluster_schwarz_2d_tri(
        phi_in,
        inner_solve,
        threshold=threshold,
        pad=pad,
        merge_dilation=merge_dilation,
        max_outer_iters=max_outer_iters,
        fallback_size_ratio=fallback_size_ratio,
        time_budget_s=time_budget_s,
        final_polish_fn=final_polish_fn,
        verbose=verbose,
        record_history=record_history,
        step_callback=step_callback,
    )


__all__ = ['iterative_2d_tri_refine_repair_schwarz']
