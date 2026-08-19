"""Monotonicity (global injectivity) helpers for deformation fields."""

import numpy as np

from dvfopt._defaults import _unpack_size


def _monotonicity_diffs_2d(dy, dx):
    """Forward-difference monotonicity metrics for deformed coordinates.

    Returns ``(h_mono, v_mono)`` with shapes ``(H, W-1)`` and ``(H-1, W)``.
    """
    h_mono = 1.0 + np.diff(dx, axis=1)  # (H, W-1)
    v_mono = 1.0 + np.diff(dy, axis=0)  # (H-1, W)
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
    d1 = 1.0 + dx[:-1, 1:] - dx[1:, :-1]  # (H-1, W-1)
    d2 = 1.0 + dy[1:, :-1] - dy[:-1, 1:]  # (H-1, W-1)
    return d1, d2


def _monotonicity_diffs_3d(dz, dy, dx):
    """Forward-difference monotonicity of deformed coordinates in 3D.

    The deformed coordinate along each axis is ``axis_index + displacement``,
    so the gap between neighbours is ``1 + diff(displacement)``.

    Returns ``(z_mono, y_mono, x_mono)`` with shapes ``(D-1, H, W)``,
    ``(D, H-1, W)`` and ``(D, H, W-1)``.
    """
    z_mono = 1.0 + np.diff(dz, axis=0)
    y_mono = 1.0 + np.diff(dy, axis=1)
    x_mono = 1.0 + np.diff(dx, axis=2)
    return z_mono, y_mono, x_mono


def injectivity_quality_3d(phi):
    """Per-voxel minimum axial monotonicity gap, spread to both endpoints.

    3D analogue of the 2D injectivity quality spread: each axial gap value
    is assigned to both voxels it separates and the element-wise minimum
    is taken, giving a ``(D, H, W)`` map whose low entries mark voxels
    involved in a (near-)crossing.

    .. note::
        Deliberately **axial-only** — the 2D version adds anti-diagonal
        terms that make each quad cell provably convex; the corresponding
        3D closure would need the face- and space-diagonal families.
        These axial gaps are necessary separation conditions (deformed
        coordinate ordering along each axis), not a full 3D injectivity
        certificate.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)`` with channels ``[dz, dy, dx]``.
    """
    dz, dy, dx = phi[0], phi[1], phi[2]
    z_mono, y_mono, x_mono = _monotonicity_diffs_3d(dz, dy, dx)
    q = np.full(dz.shape, np.inf)
    q[:-1] = np.minimum(q[:-1], z_mono)
    q[1:] = np.minimum(q[1:], z_mono)
    q[:, :-1] = np.minimum(q[:, :-1], y_mono)
    q[:, 1:] = np.minimum(q[:, 1:], y_mono)
    q[:, :, :-1] = np.minimum(q[:, :, :-1], x_mono)
    q[:, :, 1:] = np.minimum(q[:, :, 1:], x_mono)
    return q


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
