"""Shared helpers for the wall-breaker pipelines.

Used by both ``_harmonic_polished`` (m10) and ``_refine_repair`` (m14)
so the two pipelines stay consistent in their objective math.
"""
from __future__ import annotations

import numpy as np

from dvfopt.core.tri_primitives import (
    tri_areas_flat as _tri_areas_flat,
    tri_grad_T_v as _tri_grad_T_v,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def min_tri(phi: np.ndarray) -> float:
    """``min(T1, T2).min()`` for a ``(2, H, W)`` field."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(np.minimum(T1, T2).min())


def barrier_anchored_objective(phi_flat, phi_in_flat, H, W,
                                threshold, mu, anchor, eps_l1=1e-4):
    """L-BFGS-B objective: anchor(phi - phi_in) - mu * sum log(T - threshold).

    Returns ``(value, gradient)``. Returns ``(+inf, gradient)`` when the
    iterate is infeasible (``T_k <= threshold`` for some k), so the
    L-BFGS-B line search shrinks the step instead of corrupting the
    iterate. ``anchor`` is one of ``'l2'``, ``'l1'`` (smoothed), ``'none'``.

    Math:
      F = anchor(phi - phi_in) - mu * sum log(T - threshold)
      dF/dT_i = -mu / (T_i - threshold)
      dF/dphi = d anchor / d phi - mu * J^T (1 / slack)
    """
    diff = phi_flat - phi_in_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    elif anchor == 'l1':
        s = np.sqrt(diff * diff + eps_l1 * eps_l1)
        val = float((s - eps_l1).sum())
        grad = diff / s
    elif anchor == 'none':
        val = 0.0
        grad = np.zeros_like(diff)
    else:
        raise ValueError(f"unknown anchor kind: {anchor!r}")

    T = _tri_areas_flat(phi_flat, H, W)
    slack = T - threshold
    if (slack <= 0).any():
        return np.inf, grad
    val += -mu * float(np.log(slack).sum())
    grad = grad - mu * _tri_grad_T_v(phi_flat, H, W, 1.0 / slack)
    return val, grad


def resolved_safety_margin(margin: float, floor: float = 0.005) -> float:
    """Safety margin used by m10 + m14's ALM/repair fallbacks to land
    strictly above the polish's log-barrier singularity.

    With default ``margin=1e-3`` -> ``safety_margin = 0.01``. Calling this
    from one place keeps m10 and m14 in agreement when ``margin`` is
    swept.
    """
    return max(margin * 10.0, floor)
