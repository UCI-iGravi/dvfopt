"""Test fused JIT'd version of _soft_pen_objective on z=300.

Hypothesis: fusing the anchor + violation + gradient computation into
one JIT kernel saves additional Python/numpy overhead vs the current
two-separate-JITs approach. The current per-call cost of
_soft_pen_objective was ~4.5 us tottime; this would reduce that.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from numba import njit
from scipy.optimize import minimize

from dvfopt.core.wallbreakers._l2_refine import _soft_pen_objective as _ref_obj


@njit(cache=True, fastmath=True, boundscheck=False)
def _soft_pen_fused_kernel(dy, dx, dy_in, dx_in, H, W, threshold, lam, eps_l1):
    """Fused l1-anchor + T-areas + viol + grad in one kernel.

    Returns (val, g_dy, g_dx) where the gradient is split per-channel
    matching the dy_first phi pack convention.
    """
    n_cells = (H - 1) * (W - 1)
    HW = H * W
    g_dy = np.zeros((H, W))
    g_dx = np.zeros((H, W))
    val = 0.0
    # Anchor term (l1 smooth).
    for i in range(H):
        for j in range(W):
            diff_y = dy[i, j] - dy_in[i, j]
            diff_x = dx[i, j] - dx_in[i, j]
            s_y = np.sqrt(diff_y * diff_y + eps_l1 * eps_l1)
            s_x = np.sqrt(diff_x * diff_x + eps_l1 * eps_l1)
            val += (s_y - eps_l1) + (s_x - eps_l1)
            g_dy[i, j] = diff_y / s_y
            g_dx[i, j] = diff_x / s_x
    # Constraint violation + grad in one pass.
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
            # T1 area at (i, j+1): A=TR, B=BL, C=BR.
            ABx_1 = x_bl - x_tr
            ABy_1 = y_bl - y_tr
            ACx_1 = x_br - x_tr
            ACy_1 = y_br - y_tr
            T1 = -0.5 * (ABx_1 * ACy_1 - ABy_1 * ACx_1)
            # T2 area at (i, j): A=TL, B=BL, C=TR.
            ABx_2 = x_bl - x_tl
            ABy_2 = y_bl - y_tl
            ACx_2 = x_tr - x_tl
            ACy_2 = y_tr - y_tl
            T2 = -0.5 * (ABx_2 * ACy_2 - ABy_2 * ACx_2)
            v1 = max(0.0, threshold - T1)
            v2 = max(0.0, threshold - T2)
            if v1 == 0.0 and v2 == 0.0:
                continue
            val += lam * (v1 * v1 + v2 * v2)
            # Gradient: -2*lam * J^T @ viol
            # T1 contribution.
            if v1 != 0.0:
                c1 = -2.0 * lam * 0.5 * v1
                g_dx[i, j + 1] += c1 * (y_br - y_bl)
                g_dy[i, j + 1] += c1 * (x_bl - x_br)
                g_dx[i + 1, j] += -c1 * (y_br - y_tr)
                g_dy[i + 1, j] += c1 * (x_br - x_tr)
                g_dx[i + 1, j + 1] += c1 * (y_bl - y_tr)
                g_dy[i + 1, j + 1] += -c1 * (x_bl - x_tr)
            if v2 != 0.0:
                c2 = -2.0 * lam * 0.5 * v2
                g_dx[i, j] += c2 * (y_tr - y_bl)
                g_dy[i, j] += c2 * (x_bl - x_tr)
                g_dx[i + 1, j] += -c2 * (y_tr - y_tl)
                g_dy[i + 1, j] += c2 * (x_tr - x_tl)
                g_dx[i, j + 1] += c2 * (y_bl - y_tl)
                g_dy[i, j + 1] += -c2 * (x_bl - x_tl)
    return val, g_dy, g_dx


def soft_pen_fused(phi_flat, phi_in_flat, H, W, threshold, lam, anchor, eps_l1):
    """Fused implementation; only supports 'l1' anchor (the m14_fast default)."""
    if anchor != 'l1':
        return _ref_obj(phi_flat, phi_in_flat, H, W, threshold, lam, anchor, eps_l1)
    HW = H * W
    dy = np.ascontiguousarray(phi_flat[:HW].reshape(H, W))
    dx = np.ascontiguousarray(phi_flat[HW:].reshape(H, W))
    dy_in = np.ascontiguousarray(phi_in_flat[:HW].reshape(H, W))
    dx_in = np.ascontiguousarray(phi_in_flat[HW:].reshape(H, W))
    val, g_dy, g_dx = _soft_pen_fused_kernel(dy, dx, dy_in, dx_in, H, W, threshold, lam, eps_l1)
    grad = np.concatenate([g_dy.ravel(), g_dx.ravel()])
    return val, grad


def _make_inputs(H, W, rng, sparsity=0.99):
    phi_in = 0.05 * rng.standard_normal(2 * H * W)
    phi = phi_in + 0.001 * rng.standard_normal(2 * H * W)
    return phi, phi_in


def main():
    rng = np.random.default_rng(0)
    shapes = [
        ('Full slice ', 320, 456, 100),
        ('Med cluster', 30, 40, 2000),
        ('Big cluster', 80, 100, 500),
    ]
    threshold = 0.01
    lam = 1e6
    eps_l1 = 1e-4
    for label, H, W, n_iter in shapes:
        phi, phi_in = _make_inputs(H, W, rng)
        # Warmup JIT.
        soft_pen_fused(phi, phi_in, H, W, threshold, lam, 'l1', eps_l1)
        # Check equivalence.
        v_ref, g_ref = _ref_obj(phi, phi_in, H, W, threshold, lam, 'l1', eps_l1)
        v_jit, g_jit = soft_pen_fused(phi, phi_in, H, W, threshold, lam, 'l1', eps_l1)
        v_err = abs(v_ref - v_jit) / (abs(v_ref) + 1e-30)
        g_err = float(np.max(np.abs(g_ref - g_jit)))
        print(f'\n=== {label}  shape=({H}, {W})  iters={n_iter} ===')
        print(f'  equivalence  val_rel_err={v_err:.2e}  grad_max_abs_err={g_err:.2e}')
        # Timing.
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ref_obj(phi, phi_in, H, W, threshold, lam, 'l1', eps_l1)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n_iter):
            soft_pen_fused(phi, phi_in, H, W, threshold, lam, 'l1', eps_l1)
        t_jit = time.perf_counter() - t0
        print(f'  ref(2 JITs):  {1e6 * t_ref / n_iter:8.1f} us/call')
        print(f'  fused JIT  :  {1e6 * t_jit / n_iter:8.1f} us/call')
        print(f'  speedup    :  {t_ref / t_jit:.2f}x')


if __name__ == '__main__':
    main()
