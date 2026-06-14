"""Per-triangle orientation lock for the LP reformulation.

``_triangle_areas_2d`` returns half the positive shoelace determinant
under canonical winding — every triangle is positively oriented when
``phi = 0``. Locking ``T_k >= +tau`` (i.e. sign = +1 for every triangle)
makes the constraint affine in ``phi`` after one linearisation step.
"""
from __future__ import annotations

import numpy as np


def n_triangles(H: int, W: int) -> int:
    """Total number of 2-tri triangles on an H x W grid (T1 + T2 per cell)."""
    return 2 * (H - 1) * (W - 1)


def canonical_signs(H: int, W: int) -> np.ndarray:
    """All +1 — the canonical positive orientation of an undeformed grid."""
    return np.ones(n_triangles(H, W), dtype=np.float64)
