"""Sparse Jacobian construction for the 2-triangle constraint.

The existing ``dvfopt.core.primitives.tri.tri_grad_T_v`` returns
``J^T @ v`` for arbitrary ``v`` via vectorised scatter-add — efficient
for L-BFGS adjoint products but not the explicit sparse ``J`` the LP
needs. This module emits ``J`` as a ``scipy.sparse.coo_matrix`` directly,
using the same per-triangle gradient pattern.

Each row of ``J`` corresponds to one triangle ``T_k`` and has exactly
six nonzero entries — the 3 corners x 2 displacement channels that
``T_k`` depends on. ``J.shape == (2*(H-1)*(W-1), 2*H*W)``.

Decision-vector layout: ``phi_flat[:HW] = dy``, ``phi_flat[HW:] = dx``.
Constraint vector layout: ``[T1.ravel(), T2.ravel()]``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from dvfopt.core.primitives.tri import tri_areas_flat


@lru_cache(maxsize=32)
def _ref_grid(H: int, W: int):
    """Reference (undeformed) corner coordinates.

    Shape-only — cached per ``(H, W)``. The returned arrays are marked
    read-only; callers must not mutate them (they only ever add ``dy``/``dx``
    to them, producing fresh arrays).
    """
    ref_y, ref_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ref_y = ref_y.astype(np.float64)
    ref_x = ref_x.astype(np.float64)
    ref_y.flags.writeable = False
    ref_x.flags.writeable = False
    return ref_y, ref_x


@lru_cache(maxsize=32)
def _coo_pattern(H: int, W: int):
    """Shape-only COO index pattern for :func:`build_sparse_jacobian_T`.

    The sparsity pattern (rows/cols of the 12 per-cell gradient segments)
    depends only on ``(H, W)`` — only the 12 value segments depend on
    ``phi``. Caching it skips the meshgrid + index arithmetic + two
    12-segment concatenations on every linearisation call (the SLP loop
    linearises once per accepted step).

    Returns ``(rows_arr, cols_arr)`` concatenated in EXACTLY the segment
    order the uncached implementation used, so the resulting COO matrix is
    element-for-element identical. Arrays are read-only (shared across
    calls).
    """
    HW = H * W
    n_cells = (H - 1) * (W - 1)
    # Per-cell flat indices for each corner.
    ii, jj = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing='ij')
    idx_TL = (ii * W + jj).ravel()
    idx_TR = (ii * W + (jj + 1)).ravel()
    idx_BL = ((ii + 1) * W + jj).ravel()
    idx_BR = ((ii + 1) * W + (jj + 1)).ravel()

    t1_row = np.arange(n_cells)
    t2_row = np.arange(n_cells) + n_cells

    # Segment order mirrors the vals segments in build_sparse_jacobian_T:
    # T1: TR(dx), TR(dy), BL(dx), BL(dy), BR(dx), BR(dy);
    # T2: TL(dx), TL(dy), BL(dx), BL(dy), TR(dx), TR(dy).
    rows_arr = np.concatenate([t1_row] * 6 + [t2_row] * 6)
    cols_arr = np.concatenate(
        [
            idx_TR + HW,
            idx_TR,
            idx_BL + HW,
            idx_BL,
            idx_BR + HW,
            idx_BR,
            idx_TL + HW,
            idx_TL,
            idx_BL + HW,
            idx_BL,
            idx_TR + HW,
            idx_TR,
        ]
    )
    rows_arr.flags.writeable = False
    cols_arr.flags.writeable = False
    return rows_arr, cols_arr


def build_sparse_jacobian_T(phi_flat: np.ndarray, H: int, W: int) -> sp.coo_matrix:
    """Build the sparse Jacobian ``J`` of the simplex (2D) constraint at ``phi_flat``.

    Returns
    -------
    J : scipy.sparse.coo_matrix, shape (2*(H-1)*(W-1), 2*H*W).
    """
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_y = ref_y + dy
    def_x = ref_x + dx

    n_cells = (H - 1) * (W - 1)
    # Shape-only sparsity pattern (rows/cols) — cached per (H, W).
    rows_arr, cols_arr = _coo_pattern(H, W)

    # Corner deformed coords, flattened over cells.
    y_tl = def_y[:-1, :-1].ravel()
    x_tl = def_x[:-1, :-1].ravel()
    y_tr = def_y[:-1, 1:].ravel()
    x_tr = def_x[:-1, 1:].ravel()
    y_bl = def_y[1:, :-1].ravel()
    x_bl = def_x[1:, :-1].ravel()
    y_br = def_y[1:, 1:].ravel()
    x_br = def_x[1:, 1:].ravel()

    vals = []

    # ----- T1 rows (rows 0 .. n_cells-1). T1 corners: A=TR, B=BL, C=BR.
    # Gradient formulas mirror tri_grad_T_v exactly. Segment order matches
    # the cached _coo_pattern(H, W) rows/cols exactly.
    # dT1/dx[TR] = 0.5 * (y_br - y_bl)
    vals.append(0.5 * (y_br - y_bl))
    # dT1/dy[TR] = 0.5 * (x_bl - x_br)
    vals.append(0.5 * (x_bl - x_br))
    # dT1/dx[BL] = -0.5 * (y_br - y_tr)
    vals.append(-0.5 * (y_br - y_tr))
    # dT1/dy[BL] = 0.5 * (x_br - x_tr)
    vals.append(0.5 * (x_br - x_tr))
    # dT1/dx[BR] = 0.5 * (y_bl - y_tr)
    vals.append(0.5 * (y_bl - y_tr))
    # dT1/dy[BR] = -0.5 * (x_bl - x_tr)
    vals.append(-0.5 * (x_bl - x_tr))

    # ----- T2 rows (rows n_cells .. 2*n_cells-1). T2 corners: A=TL, B=BL, C=TR.
    # dT2/dx[TL] = 0.5 * (y_tr - y_bl)
    vals.append(0.5 * (y_tr - y_bl))
    # dT2/dy[TL] = 0.5 * (x_bl - x_tr)
    vals.append(0.5 * (x_bl - x_tr))
    # dT2/dx[BL] = -0.5 * (y_tr - y_tl)
    vals.append(-0.5 * (y_tr - y_tl))
    # dT2/dy[BL] = 0.5 * (x_tr - x_tl)
    vals.append(0.5 * (x_tr - x_tl))
    # dT2/dx[TR] = 0.5 * (y_bl - y_tl)
    vals.append(0.5 * (y_bl - y_tl))
    # dT2/dy[TR] = -0.5 * (x_bl - x_tl)
    vals.append(-0.5 * (x_bl - x_tl))

    vals_arr = np.concatenate(vals)
    n_rows = 2 * n_cells
    return sp.coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_rows, 2 * HW))


def linearize_T_2tri(phi_flat: np.ndarray, H: int, W: int):
    """Return ``(T_vals, J)`` at ``phi_flat`` for the simplex (2D) constraint."""
    T_vals = tri_areas_flat(phi_flat, H, W)
    J = build_sparse_jacobian_T(phi_flat, H, W)
    return T_vals, J
