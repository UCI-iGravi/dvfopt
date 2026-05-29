"""Shared 2-triangle constraint primitives.

These are the flat ``T1+T2`` constraint evaluation and its analytical
adjoint, used by every CPU 2-triangle solver in the package
(``iterative2d_tri_barrier``, ``iterative2d_tri_slsqp``,
``iterative2d_tri_schwarz``, the wall-breakers, and the per-cluster
solver).

The primitives previously lived under private (underscore-prefixed)
names inside ``iterative2d_tri_barrier``; the wall-breaker subpackage
and several other modules already imported them across module
boundaries, so they have an effective package-internal contract. This
module is the stable home for that contract.

Both functions assume the **tri-barrier phi-pack convention**:
``phi[:H*W] = dy.ravel(), phi[H*W:] = dx.ravel()``. The constraint
output layout is ``[T1.ravel(), T2.ravel()]`` of length
``2 * (H-1) * (W-1)``.
"""
from __future__ import annotations

# Re-export the canonical implementations from iterative2d_tri_barrier.
# Keeping a single source of truth — only the names move.
from dvfopt.core.iterative2d_tri_barrier import (
    _tri_areas_flat as tri_areas_flat,
    _tri_grad_T_v as tri_grad_T_v,
    _tri_areas_flat_full_coverage as tri_areas_flat_full_coverage,
    _tri_grad_T_v_full_coverage as tri_grad_T_v_full_coverage,
)

__all__ = [
    'tri_areas_flat',
    'tri_grad_T_v',
    'tri_areas_flat_full_coverage',
    'tri_grad_T_v_full_coverage',
]
