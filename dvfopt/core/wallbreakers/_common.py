"""Shared helpers for the wall-breaker pipelines.

Used by both ``_harmonic_polished`` (m10) and ``_refine_repair`` (m14)
so the two pipelines stay consistent in their objective math.
"""

from __future__ import annotations

import numpy as np

from dvfopt.core.primitives.tri import (
    tri_areas_flat as _tri_areas_flat,
)
from dvfopt.core.primitives.tri import (
    tri_grad_T_v as _tri_grad_T_v,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

# Optional fused JIT path for the log-barrier polish objective — the
# hot inner loop of the m10/m14 polish stages. Folds the anchor, the
# forward T-area geometry, and the (1/slack)-weighted adjoint scatter
# into a single pass over cells (the legacy path walks the identical
# deformed-corner geometry twice — tri_areas_flat forward, then
# tri_grad_T_v for the adjoint — plus ~0.7 ms of diff/copy/concat
# temporaries per call, and computes the full anchor before the
# feasibility gate). Auto-detected at import, mirroring
# `_l2_refine._soft_pen_l1_fused_kernel`.
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore

_ANCHOR_FLAGS = {'l2': 2, 'l1': 1, 'none': 0}


def min_tri(phi: np.ndarray) -> float:
    """``min(T1, T2).min()`` for a ``(2, H, W)`` field."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(np.minimum(T1, T2).min())


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True, boundscheck=False)
    def _barrier_anchored_fused_kernel(
        dy, dx, dy_in, dx_in, H, W, threshold, mu, anchor_flag, eps_l1
    ):
        """Fused anchor + T-areas + log-barrier + adjoint scatter.

        Returns ``(val, g_dy, g_dx)`` with ``val = +inf`` when the
        iterate is infeasible. Feasibility is tested FIRST — a
        geometry-only scan that bails on the FIRST ``slack <= 0``
        encountered, so infeasible line-search probes never pay for the
        barrier/adjoint work (the legacy path computed the full anchor
        and full T array before its feasibility gate).

        Infeasible-return convention: the gradient is the ANCHOR-ONLY
        gradient, matching the existing `barrier_anchored_objective`
        contract. NOTE the package has two conventions here —
        `_barrier_core` returns a zero gradient on infeasible iterates —
        we deliberately keep this module's anchor-gradient convention.
        (The anchor gradient is therefore still computed on the
        infeasible path; only the barrier/adjoint work is skipped.)

        Gradient (feasible): ``grad = anchor_grad - mu * J^T (1/slack)``;
        the scatter uses coefficient ``-0.5 * mu / slack`` with the same
        corner/sign pattern as ``tri_primitives._tri_grad_T_v_numba_kernel``
        (whose J^T v coefficient is ``+0.5 * v``).
        ``anchor_flag``: 2 = l2, 1 = smoothed l1, 0 = none."""
        g_dy = np.zeros((H, W))
        g_dx = np.zeros((H, W))
        # Pass 1: feasibility scan, geometry only, early exit.
        feasible = True
        for i in range(H - 1):
            for j in range(W - 1):
                x_tl = j + dx[i, j]
                y_tl = i + dy[i, j]
                x_tr = (j + 1) + dx[i, j + 1]
                y_tr = i + dy[i, j + 1]
                x_bl = j + dx[i + 1, j]
                y_bl = (i + 1) + dy[i + 1, j]
                x_br = (j + 1) + dx[i + 1, j + 1]
                y_br = (i + 1) + dy[i + 1, j + 1]
                T1 = -0.5 * ((x_bl - x_tr) * (y_br - y_tr) - (y_bl - y_tr) * (x_br - x_tr))
                T2 = -0.5 * ((x_bl - x_tl) * (y_tr - y_tl) - (y_bl - y_tl) * (x_tr - x_tl))
                if T1 - threshold <= 0.0 or T2 - threshold <= 0.0:
                    feasible = False
                    break
            if not feasible:
                break
        # Anchor (gradient needed on BOTH branches per the convention).
        val = 0.0
        if anchor_flag == 2:
            for i in range(H):
                for j in range(W):
                    diff_y = dy[i, j] - dy_in[i, j]
                    diff_x = dx[i, j] - dx_in[i, j]
                    val += 0.5 * (diff_y * diff_y + diff_x * diff_x)
                    g_dy[i, j] = diff_y
                    g_dx[i, j] = diff_x
        elif anchor_flag == 1:
            for i in range(H):
                for j in range(W):
                    diff_y = dy[i, j] - dy_in[i, j]
                    diff_x = dx[i, j] - dx_in[i, j]
                    s_y = np.sqrt(diff_y * diff_y + eps_l1 * eps_l1)
                    s_x = np.sqrt(diff_x * diff_x + eps_l1 * eps_l1)
                    val += (s_y - eps_l1) + (s_x - eps_l1)
                    g_dy[i, j] = diff_y / s_y
                    g_dx[i, j] = diff_x / s_x
        if not feasible:
            return np.inf, g_dy, g_dx
        # Pass 2: barrier value + adjoint scatter (feasible only).
        for i in range(H - 1):
            for j in range(W - 1):
                x_tl = j + dx[i, j]
                y_tl = i + dy[i, j]
                x_tr = (j + 1) + dx[i, j + 1]
                y_tr = i + dy[i, j + 1]
                x_bl = j + dx[i + 1, j]
                y_bl = (i + 1) + dy[i + 1, j]
                x_br = (j + 1) + dx[i + 1, j + 1]
                y_br = (i + 1) + dy[i + 1, j + 1]
                T1 = -0.5 * ((x_bl - x_tr) * (y_br - y_tr) - (y_bl - y_tr) * (x_br - x_tr))
                T2 = -0.5 * ((x_bl - x_tl) * (y_tr - y_tl) - (y_bl - y_tl) * (x_tr - x_tl))
                s1 = T1 - threshold
                s2 = T2 - threshold
                val += -mu * (np.log(s1) + np.log(s2))
                # T1 (A=TR, B=BL, C=BR) — coefficient -0.5 * mu / s1.
                c1 = -0.5 * mu / s1
                g_dx[i, j + 1] += c1 * (y_br - y_bl)
                g_dy[i, j + 1] += c1 * (x_bl - x_br)
                g_dx[i + 1, j] += -c1 * (y_br - y_tr)
                g_dy[i + 1, j] += c1 * (x_br - x_tr)
                g_dx[i + 1, j + 1] += c1 * (y_bl - y_tr)
                g_dy[i + 1, j + 1] += -c1 * (x_bl - x_tr)
                # T2 (A=TL, B=BL, C=TR) — coefficient -0.5 * mu / s2.
                c2 = -0.5 * mu / s2
                g_dx[i, j] += c2 * (y_tr - y_bl)
                g_dy[i, j] += c2 * (x_bl - x_tr)
                g_dx[i + 1, j] += -c2 * (y_tr - y_tl)
                g_dy[i + 1, j] += c2 * (x_tr - x_tl)
                g_dx[i, j + 1] += c2 * (y_bl - y_tl)
                g_dy[i, j + 1] += -c2 * (x_bl - x_tl)
        return val, g_dy, g_dx


def _barrier_anchored_objective_ref(
    phi_flat, phi_in_flat, H, W, threshold, mu, anchor, eps_l1=1e-4
):
    """Legacy two-pass numpy objective. Kept as the no-numba fallback
    and as the equivalence reference for the fused kernel tests."""
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


def barrier_anchored_objective(phi_flat, phi_in_flat, H, W, threshold, mu, anchor, eps_l1=1e-4):
    """L-BFGS-B objective: anchor(phi - phi_in) - mu * sum log(T - threshold).

    Returns ``(value, gradient)``. Returns ``(+inf, gradient)`` when the
    iterate is infeasible (``T_k <= threshold`` for some k), so the
    L-BFGS-B line search shrinks the step instead of corrupting the
    iterate; the returned gradient is then the anchor-only gradient.
    ``anchor`` is one of ``'l2'``, ``'l1'`` (smoothed), ``'none'``.

    Math:
      F = anchor(phi - phi_in) - mu * sum log(T - threshold)
      dF/dT_i = -mu / (T_i - threshold)
      dF/dphi = d anchor / d phi - mu * J^T (1 / slack)

    Uses a fused Numba JIT kernel when available (single-pass geometry,
    zero temporaries, early feasibility exit); falls back to the
    two-pass numpy reference otherwise.
    """
    if anchor not in _ANCHOR_FLAGS:
        raise ValueError(f"unknown anchor kind: {anchor!r}")
    if not _HAVE_NUMBA:
        return _barrier_anchored_objective_ref(
            phi_flat, phi_in_flat, H, W, threshold, mu, anchor, eps_l1
        )
    HW = H * W
    dy = np.ascontiguousarray(phi_flat[:HW].reshape(H, W))
    dx = np.ascontiguousarray(phi_flat[HW:].reshape(H, W))
    dy_in = np.ascontiguousarray(phi_in_flat[:HW].reshape(H, W))
    dx_in = np.ascontiguousarray(phi_in_flat[HW:].reshape(H, W))
    val, g_dy, g_dx = _barrier_anchored_fused_kernel(
        dy, dx, dy_in, dx_in, H, W, threshold, mu, _ANCHOR_FLAGS[anchor], eps_l1
    )
    return val, np.concatenate([g_dy.ravel(), g_dx.ravel()])


def resolved_safety_margin(margin: float, floor: float = 0.005) -> float:
    """Safety margin used by m10 + m14's ALM/repair fallbacks to land
    strictly above the polish's log-barrier singularity.

    With default ``margin=1e-3`` -> ``safety_margin = 0.01``. Calling this
    from one place keeps m10 and m14 in agreement when ``margin`` is
    swept.
    """
    return max(margin * 10.0, floor)
