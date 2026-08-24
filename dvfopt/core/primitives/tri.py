"""Shared 2-triangle constraint primitives.

These are the flat ``T1+T2`` constraint evaluation and its analytical
adjoint, used by every CPU 2-triangle solver in the package
(:mod:`dvfopt.core.barrier.tri2d`, :mod:`dvfopt.core.slsqp_fullgrid.tri2d`,
:mod:`dvfopt.core.schwarz.tri2d`, the wall-breakers, and the per-cluster
solver). This module is the source of truth for those primitives; the
underscore-prefixed names in :mod:`dvfopt.core.barrier.tri2d` are
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
        one fused loop, no intermediate allocations.

        Skips cells where both v1[i,j] and v2[i,j] are zero — most
        non-violating triangles during the late lambda annealing,
        which gives a constraint-side active-set effect with no
        L-BFGS-B variable restriction needed."""
        g_dy = np.zeros((H, W))
        g_dx = np.zeros((H, W))
        for i in range(H - 1):
            for j in range(W - 1):
                v1_ij = v1[i, j]
                v2_ij = v2[i, j]
                if v1_ij == 0.0 and v2_ij == 0.0:
                    continue
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
                if v1_ij != 0.0:
                    # T1 (A=TR, B=BL, C=BR) — coefficient = v1 * 0.5.
                    c1 = 0.5 * v1_ij
                    g_dx[i, j + 1] += c1 * (y_br - y_bl)
                    g_dy[i, j + 1] += c1 * (x_bl - x_br)
                    g_dx[i + 1, j] += -c1 * (y_br - y_tr)
                    g_dy[i + 1, j] += c1 * (x_br - x_tr)
                    g_dx[i + 1, j + 1] += c1 * (y_bl - y_tr)
                    g_dy[i + 1, j + 1] += -c1 * (x_bl - x_tr)
                if v2_ij != 0.0:
                    # T2 (A=TL, B=BL, C=TR) — coefficient = v2 * 0.5.
                    c2 = 0.5 * v2_ij
                    g_dx[i, j] += c2 * (y_tr - y_bl)
                    g_dy[i, j] += c2 * (x_bl - x_tr)
                    g_dx[i + 1, j] += -c2 * (y_tr - y_tl)
                    g_dy[i + 1, j] += c2 * (x_tr - x_tl)
                    g_dx[i, j + 1] += c2 * (y_bl - y_tl)
                    g_dy[i, j + 1] += -c2 * (x_bl - x_tl)
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


# --- Bilinear (both-diagonal) variant: 4 triangles per cell, the rows behind
# :class:`dvfopt.constraints.TriConstraint2DBilinear`. The TL-BR pair is the
# TR-BL pair of the x-MIRRORED field (mirroring swaps the diagonals), so the
# kernels above serve both halves unchanged.
# ponytail: 2 kernel launches + mirror copies ~ 2.3x the 2-tri cost per call;
# fuse both diagonals into the kernels if the barrier path on bilinear gets hot.


def _mirror_flat(f, H, W):
    """x-mirror of a flat DY_FIRST vector: pixel columns reversed, dx negated.
    An involution and self-adjoint, so it also maps gradients back."""
    HW = H * W
    dy, dx = f[:HW].reshape(H, W), f[HW:].reshape(H, W)
    return np.concatenate([dy[:, ::-1].ravel(), -dx[:, ::-1].ravel()])


def tri_areas_flat_bilinear(phi_flat, H, W):
    """``[T1, T2, U1, U2]`` — both diagonal splits, length ``4*(H-1)*(W-1)``.

    ``T1``/``T2`` are :func:`tri_areas_flat`'s TR-BL pair; ``U1 = (TL, BL, BR)``
    and ``U2 = (TR, TL, BR)`` are the TL-BR pair (mirrored cell ``j'`` is
    original cell ``W-2-j'``, hence the column reversal).
    """
    U = tri_areas_flat(_mirror_flat(phi_flat, H, W), H, W).reshape(2, H - 1, W - 1)[:, :, ::-1]
    return np.concatenate([tri_areas_flat(phi_flat, H, W), U.ravel()])


def tri_grad_T_v_bilinear(phi_flat, H, W, v):
    """J^T @ v for :func:`tri_areas_flat_bilinear` (``v`` of length
    ``4*(H-1)*(W-1)``), returned as ``[g_dy.ravel(), g_dx.ravel()]``."""
    m = (H - 1) * (W - 1)
    vm = v[2 * m :].reshape(2, H - 1, W - 1)[:, :, ::-1].ravel()
    gm = tri_grad_T_v(_mirror_flat(phi_flat, H, W), H, W, vm)
    return tri_grad_T_v(phi_flat, H, W, v[: 2 * m]) + _mirror_flat(gm, H, W)


def build_full_grid_tri_jac(H, W, full_coverage):
    """Build a callable ``jac(z) -> (n_constr, n_vars) ndarray`` for the
    full-grid 2-triangle constraint.

    Variable layout: ``[dy.ravel(), dx.ravel()]`` (length ``2*H*W``).
    Constraint layout: ``[T1.ravel(), T2.ravel()]`` (length
    ``2*(H-1)*(W-1)``); optionally with two corner-patch rows appended.

    The sparsity pattern is constant — only the entries change per call —
    so we precompute the (row, col) index arrays once at build time and
    scatter the per-iteration values into ONE preallocated dense buffer.
    The buffer is large — ``(2*(H-1)*(W-1), 2*H*W)`` — but scipy's SLSQP
    constraint adapter was already materialising exactly that dense array
    internally on every jac call (``j_ineq`` allocates dense zeros and
    ``.toarray()``s sparse input, scipy/optimize/_constraints.py), so the
    reused buffer strictly reduces allocation versus the old CSR return.
    """
    Hc, Wc = H - 1, W - 1
    n_cells = Hc * Wc
    n_constr = 2 * n_cells + (2 if full_coverage else 0)
    n_vars = 2 * H * W

    HW = H * W

    cy_idx = np.arange(Hc, dtype=np.int64)[:, None]
    cx_idx = np.arange(Wc, dtype=np.int64)[None, :]
    # Pixel indices for each corner of cell (cy, cx).
    pix_TL = cy_idx * W + cx_idx
    pix_TR = cy_idx * W + (cx_idx + 1)
    pix_BL = (cy_idx + 1) * W + cx_idx
    pix_BR = (cy_idx + 1) * W + (cx_idx + 1)

    # Column = pixel for dy channel; col + HW for dx channel.
    rows_T1 = (cy_idx * Wc + cx_idx) * np.ones((Hc, Wc), dtype=np.int64)
    rows_T2 = rows_T1 + n_cells

    # 12 per-cell triplets ordered to match the partial-derivative ordering
    # used inside ``jac()`` below.
    triplets = [
        # T1 partials
        (rows_T1, pix_TR, 'dT1_TR_y'),  # dy(TR)
        (rows_T1, pix_TR + HW, 'dT1_TR_x'),  # dx(TR)
        (rows_T1, pix_BL, 'dT1_BL_y'),
        (rows_T1, pix_BL + HW, 'dT1_BL_x'),
        (rows_T1, pix_BR, 'dT1_BR_y'),
        (rows_T1, pix_BR + HW, 'dT1_BR_x'),
        # T2 partials
        (rows_T2, pix_TL, 'dT2_TL_y'),
        (rows_T2, pix_TL + HW, 'dT2_TL_x'),
        (rows_T2, pix_TR, 'dT2_TR_y'),
        (rows_T2, pix_TR + HW, 'dT2_TR_x'),
        (rows_T2, pix_BL, 'dT2_BL_y'),
        (rows_T2, pix_BL + HW, 'dT2_BL_x'),
    ]
    rows_flat = np.concatenate([t[0].ravel() for t in triplets])
    cols_flat = np.concatenate([t[1].ravel() for t in triplets])
    key_order = [t[2] for t in triplets]

    if full_coverage:
        # Patch TL: A=(0,0), B=(1,1), C=(0,1). Patch BR: A=(H-2,W-2),
        # B=(H-1,W-2), C=(H-1,W-1). Each patch contributes 6 partials
        # (dy/dx for each of its 3 vertices).
        row_p_tl = 2 * n_cells
        row_p_br = 2 * n_cells + 1
        pTL_A = 0 * W + 0
        pTL_B = 1 * W + 1
        pTL_C = 0 * W + 1
        pBR_A = (H - 2) * W + (W - 2)
        pBR_B = (H - 1) * W + (W - 2)
        pBR_C = (H - 1) * W + (W - 1)

        patch_triplets = [
            (row_p_tl, pTL_A, 'dPTL_A_y'),
            (row_p_tl, pTL_A + HW, 'dPTL_A_x'),
            (row_p_tl, pTL_B, 'dPTL_B_y'),
            (row_p_tl, pTL_B + HW, 'dPTL_B_x'),
            (row_p_tl, pTL_C, 'dPTL_C_y'),
            (row_p_tl, pTL_C + HW, 'dPTL_C_x'),
            (row_p_br, pBR_A, 'dPBR_A_y'),
            (row_p_br, pBR_A + HW, 'dPBR_A_x'),
            (row_p_br, pBR_B, 'dPBR_B_y'),
            (row_p_br, pBR_B + HW, 'dPBR_B_x'),
            (row_p_br, pBR_C, 'dPBR_C_y'),
            (row_p_br, pBR_C + HW, 'dPBR_C_x'),
        ]
        rows_flat = np.concatenate(
            [rows_flat, np.array([t[0] for t in patch_triplets], dtype=np.int64)]
        )
        cols_flat = np.concatenate(
            [cols_flat, np.array([t[1] for t in patch_triplets], dtype=np.int64)]
        )
        key_order = key_order + [t[2] for t in patch_triplets]

    ref_y, ref_x = _ref_grid(H, W)

    # Preallocated dense Jacobian, rewritten in place each call. Entries
    # off the (constant) sparsity pattern stay 0 from this allocation; the
    # (row, col) pairs are unique so plain fancy-index assignment is exact.
    J_buf = np.zeros((n_constr, n_vars), dtype=np.float64)

    def jac(z):
        dy = z[:HW].reshape(H, W)
        dx = z[HW:].reshape(H, W)
        def_x = ref_x + dx
        def_y = ref_y + dy
        x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
        x_tr, y_tr = def_x[:-1, 1:], def_y[:-1, 1:]
        x_bl, y_bl = def_x[1:, :-1], def_y[1:, :-1]
        x_br, y_br = def_x[1:, 1:], def_y[1:, 1:]

        d = {
            'dT1_TR_x': 0.5 * (y_br - y_bl),
            'dT1_TR_y': 0.5 * (x_bl - x_br),
            'dT1_BL_x': 0.5 * (y_tr - y_br),
            'dT1_BL_y': 0.5 * (x_br - x_tr),
            'dT1_BR_x': 0.5 * (y_bl - y_tr),
            'dT1_BR_y': 0.5 * (x_tr - x_bl),
            'dT2_TL_x': 0.5 * (y_tr - y_bl),
            'dT2_TL_y': 0.5 * (x_bl - x_tr),
            'dT2_BL_x': 0.5 * (y_tl - y_tr),
            'dT2_BL_y': 0.5 * (x_tr - x_tl),
            'dT2_TR_x': 0.5 * (y_bl - y_tl),
            'dT2_TR_y': 0.5 * (x_tl - x_bl),
        }

        if full_coverage:
            # Patch TL: A=(0,0), B=(1,1), C=(0,1).
            Ax = def_x[0, 0]
            Ay = def_y[0, 0]
            Bx = def_x[1, 1]
            By = def_y[1, 1]
            Cx = def_x[0, 1]
            Cy = def_y[0, 1]
            d['dPTL_A_x'] = 0.5 * (Cy - By)
            d['dPTL_A_y'] = 0.5 * (Bx - Cx)
            d['dPTL_B_x'] = 0.5 * (Ay - Cy)
            d['dPTL_B_y'] = 0.5 * (Cx - Ax)
            d['dPTL_C_x'] = 0.5 * (By - Ay)
            d['dPTL_C_y'] = 0.5 * (Ax - Bx)
            # Patch BR: A=(H-2,W-2), B=(H-1,W-2), C=(H-1,W-1).
            Ax = def_x[H - 2, W - 2]
            Ay = def_y[H - 2, W - 2]
            Bx = def_x[H - 1, W - 2]
            By = def_y[H - 1, W - 2]
            Cx = def_x[H - 1, W - 1]
            Cy = def_y[H - 1, W - 1]
            d['dPBR_A_x'] = 0.5 * (Cy - By)
            d['dPBR_A_y'] = 0.5 * (Bx - Cx)
            d['dPBR_B_x'] = 0.5 * (Ay - Cy)
            d['dPBR_B_y'] = 0.5 * (Cx - Ax)
            d['dPBR_C_x'] = 0.5 * (By - Ay)
            d['dPBR_C_y'] = 0.5 * (Ax - Bx)

        parts = []
        for key in key_order:
            arr = d[key]
            parts.append(
                np.ravel(arr) if isinstance(arr, np.ndarray) else np.array([arr], dtype=np.float64)
            )
        data_flat = np.concatenate(parts)
        J_buf[rows_flat, cols_flat] = data_flat
        return J_buf

    return jac


# Back-compat name used across the fullgrid/schwarz call sites.
_build_full_grid_tri_jac = build_full_grid_tri_jac


__all__ = [
    'build_full_grid_tri_jac',
    'tri_areas_flat',
    'tri_areas_flat_bilinear',
    'tri_areas_flat_full_coverage',
    'tri_grad_T_v',
    'tri_grad_T_v_bilinear',
    'tri_grad_T_v_full_coverage',
]
