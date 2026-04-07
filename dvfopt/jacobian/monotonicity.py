"""Monotonicity (global injectivity) helpers for deformation fields."""

import numpy as np

from dvfopt._defaults import _unpack_size


def _monotonicity_diffs_2d(dy, dx):
    """Forward-difference monotonicity metrics for deformed coordinates.

    Returns ``(h_mono, v_mono)`` with shapes ``(H, W-1)`` and ``(H-1, W)``.
    """
    h_mono = 1.0 + np.diff(dx, axis=1)   # (H, W-1)
    v_mono = 1.0 + np.diff(dy, axis=0)   # (H-1, W)
    return h_mono, v_mono


def _diagonal_monotonicity_diffs_2d(dy, dx):
    """Anti-diagonal monotonicity: ensures each deformed quad cell is convex.

    For cell (r, c) with corners TL/TR/BR/BL:
        d1[r,c] = 1 + dx[r, c+1] - dx[r+1, c]   (TR.x > BL.x)
        d2[r,c] = 1 + dy[r+1, c] - dy[r, c+1]   (BL.y > TR.y)

    Together with h/v monotonicity these 4 conditions guarantee each quad
    cell is convex with positive orientation, preventing cross-row
    pinch-point self-intersections.

    Returns ``(d1, d2)`` each with shape ``(H-1, W-1)``.
    """
    d1 = 1.0 + dx[:-1, 1:] - dx[1:, :-1]   # (H-1, W-1)
    d2 = 1.0 + dy[1:, :-1] - dy[:-1, 1:]   # (H-1, W-1)
    return d1, d2


def injectivity_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened monotonicity diffs for the SLSQP injectivity constraint.

    Concatenates h_mono, v_mono, d1, and d2 (diagonal) diffs.  All four must
    be positive for the deformed grid to be globally injective and convex.

    When *exclude_boundaries* is ``True``, h/v use the standard ``[1:-1,1:-1]``
    interior slice.  Diagonal constraints are extended to all cells where at
    least one vertex is free (i.e. not on the frozen sub-window boundary).
    Only the two corners whose *both* vertices are frozen are excluded:
    cell (0, 0) and cell (sy-2, sx-2).
    """
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
    d1, d2 = _diagonal_monotonicity_diffs_2d(dy, dx)
    if exclude_boundaries:
        h_vals = h_mono[1:-1, 1:-1].flatten()
        v_vals = v_mono[1:-1, 1:-1].flatten()
        # Include all diagonal cells except the two all-frozen corners.
        # (0,0):        d1 involves dx[0,1] and dx[1,0]  — both boundary
        # (sy-2,sx-2):  d1 involves dx[sy-2,sx-1] and dx[sy-1,sx-2] — both boundary
        n_diag = (sy - 1) * (sx - 1)
        keep = np.ones(n_diag, dtype=bool)
        keep[0] = False
        if n_diag > 1:
            keep[(sy - 2) * (sx - 1) + (sx - 2)] = False
        d1_vals = d1.reshape(-1)[keep]
        d2_vals = d2.reshape(-1)[keep]
    else:
        h_vals = h_mono.flatten()
        v_vals = v_mono.flatten()
        d1_vals = d1.flatten()
        d2_vals = d2.flatten()
    return np.concatenate([h_vals, v_vals, d1_vals, d2_vals])
