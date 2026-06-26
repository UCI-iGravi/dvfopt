"""Sparse Jacobian construction for the 2-triangle constraint.

The existing ``dvfopt.core.tri_primitives.tri_grad_T_v`` returns
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

import numpy as np
import scipy.sparse as sp

from dvfopt.core.tri_primitives import tri_areas_flat


def _ref_grid(H: int, W: int):
    """Reference (undeformed) corner coordinates."""
    ref_y, ref_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return ref_y.astype(np.float64), ref_x.astype(np.float64)


def build_sparse_jacobian_T(phi_flat: np.ndarray, H: int, W: int) -> sp.coo_matrix:
    """Build the sparse Jacobian ``J`` of the 2-tri constraint at ``phi_flat``.

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
    # Per-cell flat indices for each corner.
    ii, jj = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing='ij')
    idx_TL = (ii * W + jj).ravel()
    idx_TR = (ii * W + (jj + 1)).ravel()
    idx_BL = ((ii + 1) * W + jj).ravel()
    idx_BR = ((ii + 1) * W + (jj + 1)).ravel()

    # Corner deformed coords, flattened over cells.
    y_tl = def_y[:-1, :-1].ravel()
    x_tl = def_x[:-1, :-1].ravel()
    y_tr = def_y[:-1, 1:].ravel()
    x_tr = def_x[:-1, 1:].ravel()
    y_bl = def_y[1:, :-1].ravel()
    x_bl = def_x[1:, :-1].ravel()
    y_br = def_y[1:, 1:].ravel()
    x_br = def_x[1:, 1:].ravel()

    rows = []
    cols = []
    vals = []

    # ----- T1 rows (rows 0 .. n_cells-1). T1 corners: A=TR, B=BL, C=BR.
    # Gradient formulas mirror tri_grad_T_v exactly.
    t1_row = np.arange(n_cells)
    # dT1/dx[TR] = 0.5 * (y_br - y_bl)
    rows.append(t1_row); cols.append(idx_TR + HW); vals.append(0.5 * (y_br - y_bl))
    # dT1/dy[TR] = 0.5 * (x_bl - x_br)
    rows.append(t1_row); cols.append(idx_TR);      vals.append(0.5 * (x_bl - x_br))
    # dT1/dx[BL] = -0.5 * (y_br - y_tr)
    rows.append(t1_row); cols.append(idx_BL + HW); vals.append(-0.5 * (y_br - y_tr))
    # dT1/dy[BL] = 0.5 * (x_br - x_tr)
    rows.append(t1_row); cols.append(idx_BL);      vals.append(0.5 * (x_br - x_tr))
    # dT1/dx[BR] = 0.5 * (y_bl - y_tr)
    rows.append(t1_row); cols.append(idx_BR + HW); vals.append(0.5 * (y_bl - y_tr))
    # dT1/dy[BR] = -0.5 * (x_bl - x_tr)
    rows.append(t1_row); cols.append(idx_BR);      vals.append(-0.5 * (x_bl - x_tr))

    # ----- T2 rows (rows n_cells .. 2*n_cells-1). T2 corners: A=TL, B=BL, C=TR.
    t2_row = np.arange(n_cells) + n_cells
    # dT2/dx[TL] = 0.5 * (y_tr - y_bl)
    rows.append(t2_row); cols.append(idx_TL + HW); vals.append(0.5 * (y_tr - y_bl))
    # dT2/dy[TL] = 0.5 * (x_bl - x_tr)
    rows.append(t2_row); cols.append(idx_TL);      vals.append(0.5 * (x_bl - x_tr))
    # dT2/dx[BL] = -0.5 * (y_tr - y_tl)
    rows.append(t2_row); cols.append(idx_BL + HW); vals.append(-0.5 * (y_tr - y_tl))
    # dT2/dy[BL] = 0.5 * (x_tr - x_tl)
    rows.append(t2_row); cols.append(idx_BL);      vals.append(0.5 * (x_tr - x_tl))
    # dT2/dx[TR] = 0.5 * (y_bl - y_tl)
    rows.append(t2_row); cols.append(idx_TR + HW); vals.append(0.5 * (y_bl - y_tl))
    # dT2/dy[TR] = -0.5 * (x_bl - x_tl)
    rows.append(t2_row); cols.append(idx_TR);      vals.append(-0.5 * (x_bl - x_tl))

    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    vals_arr = np.concatenate(vals)
    n_rows = 2 * n_cells
    return sp.coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_rows, 2 * HW))


def linearize_T_2tri(phi_flat: np.ndarray, H: int, W: int):
    """Return ``(T_vals, J)`` at ``phi_flat`` for the 2-tri constraint."""
    T_vals = tri_areas_flat(phi_flat, H, W)
    J = build_sparse_jacobian_T(phi_flat, H, W)
    return T_vals, J
