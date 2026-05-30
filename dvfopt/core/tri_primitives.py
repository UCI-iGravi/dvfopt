"""Shared 2-triangle constraint primitives.

These are the flat ``T1+T2`` constraint evaluation and its analytical
adjoint, used by every CPU 2-triangle solver in the package
(``iterative2d_tri_barrier``, ``iterative2d_tri_slsqp``,
``iterative2d_tri_schwarz``, the wall-breakers, and the per-cluster
solver). This module is the source of truth for those primitives; the
underscore-prefixed names in ``iterative2d_tri_barrier`` are
back-compat aliases.

Both functions assume the **tri-barrier phi-pack convention**:
``phi[:H*W] = dy.ravel(), phi[H*W:] = dx.ravel()``. The constraint
output layout is ``[T1.ravel(), T2.ravel()]`` of length
``2 * (H-1) * (W-1)``.
"""

from __future__ import annotations

import numpy as np

from dvfopt.jacobian.shoelace import _ref_grid
from dvfopt.jacobian.triangle_sign import (
    _corner_patch_areas_2d,
    _triangle_areas_2d,
)


def tri_areas_flat(phi_flat, H, W):
    """Concatenated [T1.ravel, T2.ravel] of length 2*(H-1)*(W-1)."""
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    return np.concatenate([T1.ravel(), T2.ravel()])


def tri_grad_T_v(phi_flat, H, W, v):
    """J^T @ v for the 2-triangle constraint Jacobian, analytically via
    vectorised scatter-add. ``v`` length 2*(H-1)*(W-1) (T1 then T2).
    Returns length 2*H*W ordered [dy.ravel(), dx.ravel()]."""
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy
    n_cells = (H - 1) * (W - 1)
    v1 = v[:n_cells].reshape(H - 1, W - 1)
    v2 = v[n_cells:].reshape(H - 1, W - 1)
    x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
    x_tr, y_tr = def_x[:-1, 1:], def_y[:-1, 1:]
    x_bl, y_bl = def_x[1:, :-1], def_y[1:, :-1]
    x_br, y_br = def_x[1:, 1:], def_y[1:, 1:]

    g_dy = np.zeros((H, W))
    g_dx = np.zeros((H, W))

    # T1 (A=TR, B=BL, C=BR).
    g_dx[:-1, 1:] += v1 * 0.5 * (y_br - y_bl)
    g_dy[:-1, 1:] += v1 * 0.5 * (x_bl - x_br)
    g_dx[1:, :-1] += -v1 * 0.5 * (y_br - y_tr)
    g_dy[1:, :-1] += v1 * 0.5 * (x_br - x_tr)
    g_dx[1:, 1:] += v1 * 0.5 * (y_bl - y_tr)
    g_dy[1:, 1:] += -v1 * 0.5 * (x_bl - x_tr)
    # T2 (A=TL, B=BL, C=TR).
    g_dx[:-1, :-1] += v2 * 0.5 * (y_tr - y_bl)
    g_dy[:-1, :-1] += v2 * 0.5 * (x_bl - x_tr)
    g_dx[1:, :-1] += -v2 * 0.5 * (y_tr - y_tl)
    g_dy[1:, :-1] += v2 * 0.5 * (x_tr - x_tl)
    g_dx[:-1, 1:] += v2 * 0.5 * (y_bl - y_tl)
    g_dy[:-1, 1:] += -v2 * 0.5 * (x_bl - x_tl)
    return np.concatenate([g_dy.ravel(), g_dx.ravel()])


# --- Full-coverage variants: add two corner-patch triangles so every grid
# vertex (incl. the two diagonally-opposite corners (0,0) and (H-1, W-1))
# is enforced by at least two triangles. The standard scheme above leaves
# those two corners with only ONE constraint each.


def tri_areas_flat_full_coverage(phi_flat, H, W):
    """Standard T1, T2 stack plus two corner patches.

    Output layout: ``[T1.ravel, T2.ravel, patch_TL, patch_BR]`` — length
    ``2*(H-1)*(W-1) + 2``.
    """
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    patches = _corner_patch_areas_2d(dy, dx)
    return np.concatenate([T1.ravel(), T2.ravel(), patches])


def tri_grad_T_v_full_coverage(phi_flat, H, W, v):
    """J^T @ v for the full-coverage 2-triangle Jacobian.

    Layout of ``v``: first ``2*(H-1)*(W-1)`` entries are the standard T1/T2
    constraints, last 2 are the corner patches ``[patch_TL, patch_BR]``.
    """
    n_cells = (H - 1) * (W - 1)
    HW = H * W

    # Standard contribution.
    g = tri_grad_T_v(phi_flat, H, W, v[: 2 * n_cells])

    # Patch contributions are tiny — only 6 vertices touched total — but
    # we still write them into the dy/dx grids for a clean concat.
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy

    g_dy = g[:HW].reshape(H, W).copy()
    g_dx = g[HW:].reshape(H, W).copy()

    v_tl = v[2 * n_cells]  # patch at corner (0, 0)
    v_br = v[2 * n_cells + 1]  # patch at corner (H-1, W-1)

    # patch_TL: A=TL=(0,0), B=BR=(1,1), C=TR=(0,1).
    # Derived analytically from T = -0.5 * ((Bx-Ax)(Cy-Ay) - (By-Ay)(Cx-Ax)).
    g_dx[0, 0] += v_tl * 0.5 * (def_y[0, 1] - def_y[1, 1])  # ∂T/∂Ax
    g_dy[0, 0] += v_tl * 0.5 * (def_x[1, 1] - def_x[0, 1])  # ∂T/∂Ay
    g_dx[1, 1] += v_tl * -0.5 * (def_y[0, 1] - def_y[0, 0])  # ∂T/∂Bx
    g_dy[1, 1] += v_tl * 0.5 * (def_x[0, 1] - def_x[0, 0])  # ∂T/∂By
    g_dx[0, 1] += v_tl * 0.5 * (def_y[1, 1] - def_y[0, 0])  # ∂T/∂Cx
    g_dy[0, 1] += v_tl * -0.5 * (def_x[1, 1] - def_x[0, 0])  # ∂T/∂Cy

    # patch_BR: A=TL=(H-2, W-2), B=BL=(H-1, W-2), C=BR=(H-1, W-1).
    Hm2, Wm2 = H - 2, W - 2
    g_dx[Hm2, Wm2] += v_br * 0.5 * (def_y[H - 1, W - 1] - def_y[H - 1, Wm2])
    g_dy[Hm2, Wm2] += v_br * 0.5 * (def_x[H - 1, Wm2] - def_x[H - 1, W - 1])
    g_dx[H - 1, Wm2] += v_br * -0.5 * (def_y[H - 1, W - 1] - def_y[Hm2, Wm2])
    g_dy[H - 1, Wm2] += v_br * 0.5 * (def_x[H - 1, W - 1] - def_x[Hm2, Wm2])
    g_dx[H - 1, W - 1] += v_br * 0.5 * (def_y[H - 1, Wm2] - def_y[Hm2, Wm2])
    g_dy[H - 1, W - 1] += v_br * -0.5 * (def_x[H - 1, Wm2] - def_x[Hm2, Wm2])

    return np.concatenate([g_dy.ravel(), g_dx.ravel()])


__all__ = [
    'tri_areas_flat',
    'tri_areas_flat_full_coverage',
    'tri_grad_T_v',
    'tri_grad_T_v_full_coverage',
]
