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

# Optional Numba JIT fast path for `tri_grad_T_v`. cProfile of the
# B0039 z=300 cluster_slp run showed this function at 28 s tottime
# from 465k calls inside L-BFGS-B gradient evaluations (60 μs each)
# — at the python+numpy per-call floor. A JIT-compiled loop kernel
# folds the 12 sliced broadcast-adds into a single triple-nested
# loop with no intermediate allocations.
try:
    from numba import njit, prange  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore
    prange = range  # type: ignore


def tri_areas_flat(phi_flat, H, W):
    """Concatenated [T1.ravel, T2.ravel] of length 2*(H-1)*(W-1)."""
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    return np.concatenate([T1.ravel(), T2.ravel()])


def _tri_grad_T_v_numpy(phi_flat, H, W, v):
    """Pure-numpy reference path. Kept for clarity + fallback when
    Numba is not installed."""
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


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _tri_grad_T_v_numba_kernel(dy, dx, v1, v2, H, W):
        """Single-pass JIT kernel: walks each (i, j) cell once and
        scatter-adds T1 + T2 contributions to all four corner vertices.
        Replaces 12 sliced broadcast-adds in the numpy version with
        one fused loop, no intermediate allocations."""
        g_dy = np.zeros((H, W))
        g_dx = np.zeros((H, W))
        for i in range(H - 1):
            for j in range(W - 1):
                # Deformed positions of the four cell corners.
                # ref_y[i, j] = i, ref_x[i, j] = j (unit grid).
                x_tl = j + dx[i, j]
                y_tl = i + dy[i, j]
                x_tr = (j + 1) + dx[i, j + 1]
                y_tr = i + dy[i, j + 1]
                x_bl = j + dx[i + 1, j]
                y_bl = (i + 1) + dy[i + 1, j]
                x_br = (j + 1) + dx[i + 1, j + 1]
                y_br = (i + 1) + dy[i + 1, j + 1]
                # T1 (A=TR, B=BL, C=BR) — coefficient = v1 * 0.5.
                c1 = 0.5 * v1[i, j]
                g_dx[i,     j + 1] += c1 * (y_br - y_bl)
                g_dy[i,     j + 1] += c1 * (x_bl - x_br)
                g_dx[i + 1, j]     += -c1 * (y_br - y_tr)
                g_dy[i + 1, j]     +=  c1 * (x_br - x_tr)
                g_dx[i + 1, j + 1] +=  c1 * (y_bl - y_tr)
                g_dy[i + 1, j + 1] += -c1 * (x_bl - x_tr)
                # T2 (A=TL, B=BL, C=TR) — coefficient = v2 * 0.5.
                c2 = 0.5 * v2[i, j]
                g_dx[i,     j]     +=  c2 * (y_tr - y_bl)
                g_dy[i,     j]     +=  c2 * (x_bl - x_tr)
                g_dx[i + 1, j]     += -c2 * (y_tr - y_tl)
                g_dy[i + 1, j]     +=  c2 * (x_tr - x_tl)
                g_dx[i,     j + 1] +=  c2 * (y_bl - y_tl)
                g_dy[i,     j + 1] += -c2 * (x_bl - x_tl)
        return g_dy, g_dx


def tri_grad_T_v(phi_flat, H, W, v):
    """J^T @ v for the 2-triangle constraint Jacobian, analytically.
    ``v`` length 2*(H-1)*(W-1) (T1 then T2). Returns length 2*H*W
    ordered [dy.ravel(), dx.ravel()].

    Uses the Numba JIT kernel when available (5-10x speedup on this
    hot path inside L-BFGS-B gradient evaluations). Falls back to the
    pure-numpy implementation when Numba is not installed."""
    if not _HAVE_NUMBA:
        return _tri_grad_T_v_numpy(phi_flat, H, W, v)
    HW = H * W
    n_cells = (H - 1) * (W - 1)
    dy = np.ascontiguousarray(phi_flat[:HW].reshape(H, W))
    dx = np.ascontiguousarray(phi_flat[HW:].reshape(H, W))
    v1 = np.ascontiguousarray(v[:n_cells].reshape(H - 1, W - 1))
    v2 = np.ascontiguousarray(v[n_cells:].reshape(H - 1, W - 1))
    g_dy, g_dx = _tri_grad_T_v_numba_kernel(dy, dx, v1, v2, H, W)
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
