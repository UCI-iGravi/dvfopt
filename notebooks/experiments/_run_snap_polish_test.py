"""Geometric snap-polish test.

After the barrier brings z=12 essentially to the T=0 wall (e.g. with
L2/thr=0 the worst is ~-0.002 with ~240 cells in the (-0.002, 0] band),
we cannot push the last sliver across with optimisation. This test asks
the dual question directly:

  given the current near-feasible field, find the *minimum-norm linear
  step* delta such that the linearised triangle areas T_k + (dT_k/dphi)
  @ delta >= epsilon for every folded k.

That's an underdetermined sparse linear system A @ delta = rhs whose
minimum-norm solution is obtained by LSQR. Each constraint row has 6
nonzeros (3 corners x 2 coords); rhs_k = epsilon - T_k > 0 for folded k.
After applying delta, T_k for non-folded cells changes too (only through
shared corners, so locally); we iterate.

Pipeline: barrier (L2/thr=0) -> snap-polish -> report.
Usage: python _run_snap_polish_test.py [z]   (default z=12)
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from dvfopt.jacobian.shoelace import _ref_grid
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from _run_tri_barrier_test import solve_2d_tri_barrier, phi_full

EPS = 1e-3                 # target margin above 0 for folded triangles
MAX_SNAP_ITERS = 12
DAMP = 1e-10


def _stats(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return (int((T1 <= 0).sum() + (T2 <= 0).sum()),
            float(min(T1.min(), T2.min())))


def build_folded_jacobian(phi, eps=EPS):
    """Sparse Jacobian (rows = folded triangles only, cols = flat phi)
    plus rhs = eps - T_folded. Returns None if no folds."""
    H, W = phi.shape[1], phi.shape[2]
    HW = H * W
    dy, dx = phi[0], phi[1]
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy
    T1, T2 = _triangle_areas_2d(dy, dx)
    yy1, xx1 = np.where(T1 <= 0)
    yy2, xx2 = np.where(T2 <= 0)
    n1, n2 = len(yy1), len(yy2)
    if n1 + n2 == 0:
        return None, None

    rows_all, cols_all, vals_all = [], [], []

    # T1: A=TR (y, x+1), B=BL (y+1, x), C=BR (y+1, x+1).
    if n1:
        y, x = yy1, xx1
        x_TR = def_x[y, x + 1];     y_TR = def_y[y, x + 1]
        x_BL = def_x[y + 1, x];     y_BL = def_y[y + 1, x]
        x_BR = def_x[y + 1, x + 1]; y_BR = def_y[y + 1, x + 1]
        rng = np.arange(n1)
        triples = [
            (y * W + (x + 1),               0.5 * (x_BL - x_BR)),   # dT1/dy_TR
            (HW + y * W + (x + 1),          0.5 * (y_BR - y_BL)),   # dT1/dx_TR
            ((y + 1) * W + x,               0.5 * (x_BR - x_TR)),   # dT1/dy_BL
            (HW + (y + 1) * W + x,         -0.5 * (y_BR - y_TR)),   # dT1/dx_BL
            ((y + 1) * W + (x + 1),        -0.5 * (x_BL - x_TR)),   # dT1/dy_BR
            (HW + (y + 1) * W + (x + 1),    0.5 * (y_BL - y_TR)),   # dT1/dx_BR
        ]
        for col_arr, val_arr in triples:
            rows_all.append(rng)
            cols_all.append(col_arr)
            vals_all.append(val_arr)

    # T2: A=TL (y, x), B=BL (y+1, x), C=TR (y, x+1).
    if n2:
        y, x = yy2, xx2
        x_TL = def_x[y, x];         y_TL = def_y[y, x]
        x_BL = def_x[y + 1, x];     y_BL = def_y[y + 1, x]
        x_TR = def_x[y, x + 1];     y_TR = def_y[y, x + 1]
        rng = n1 + np.arange(n2)
        triples = [
            (y * W + x,                     0.5 * (x_BL - x_TR)),   # dT2/dy_TL
            (HW + y * W + x,                0.5 * (y_TR - y_BL)),   # dT2/dx_TL
            ((y + 1) * W + x,               0.5 * (x_TR - x_TL)),   # dT2/dy_BL
            (HW + (y + 1) * W + x,         -0.5 * (y_TR - y_TL)),   # dT2/dx_BL
            (y * W + (x + 1),              -0.5 * (x_BL - x_TL)),   # dT2/dy_TR
            (HW + y * W + (x + 1),          0.5 * (y_BL - y_TL)),   # dT2/dx_TR
        ]
        for col_arr, val_arr in triples:
            rows_all.append(rng)
            cols_all.append(col_arr)
            vals_all.append(val_arr)

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n1 + n2, 2 * HW))
    T_folded = np.concatenate([T1[yy1, xx1], T2[yy2, xx2]])
    rhs = eps - T_folded
    return A, rhs


def snap_polish(phi, *, eps=EPS, max_iters=MAX_SNAP_ITERS, damp=DAMP,
                anchor=None, verbose=True):
    """Iteratively snap folded triangles toward T >= eps via the
    minimum-norm linearised step. ``anchor`` (optional) is the original
    field; the final L2 distortion vs it is reported."""
    phi = phi.copy()
    H, W = phi.shape[1], phi.shape[2]
    HW = H * W
    for it in range(max_iters):
        AB = build_folded_jacobian(phi, eps=eps)
        if AB is None:
            if verbose:
                print(f'  snap iter {it}: no folds, done', flush=True)
            return phi, True, it
        A, rhs = AB
        n0, m0 = _stats(phi)
        if verbose:
            print(f'  snap iter {it:2d}: folded={A.shape[0]:4d}  '
                  f'min_tri={m0:+.6f}  max_rhs={rhs.max():.5f}', flush=True)
        out = lsqr(A, rhs, damp=damp, show=False, iter_lim=4000)
        delta = out[0]
        flat = np.concatenate([phi[0].ravel(), phi[1].ravel()]) + delta
        phi = np.stack([flat[:HW].reshape(H, W), flat[HW:].reshape(H, W)])
    n, m = _stats(phi)
    success = (n == 0)
    if verbose:
        print(f'  snap final: folded={n}  min_tri={m:+.6f}  '
              f'success={success}', flush=True)
    return phi, success, max_iters


def main():
    z = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    phi = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
    anchor = phi.copy()

    print(f'====================== z={z} ======================', flush=True)
    n0, m0 = _stats(phi)
    print(f'[init]  n_neg={n0}  min_tri={m0:+.5f}', flush=True)

    # --- Phase A: barrier (L2 anchor, relaxed thr=0) -------------------
    print('\n[phase A: barrier (L2 anchor, threshold=0)]', flush=True)
    t0 = time.time()
    phi_b, _stats_b = solve_2d_tri_barrier(phi, threshold=0.0, anchor='l2',
                                           verbose=True)
    t_barrier = time.time() - t0
    n_b, m_b = _stats(phi_b)
    l2_b = float(np.linalg.norm(phi_b - anchor))
    print(f'\n[barrier done]  n_neg={n_b}  min_tri={m_b:+.6f}  '
          f'L2={l2_b:.3f}  ({t_barrier:.0f}s)', flush=True)

    # --- Phase B: snap-polish ------------------------------------------
    print('\n[phase B: snap-polish, eps={:.3g}]'.format(EPS), flush=True)
    t0 = time.time()
    phi_s, success, n_iters = snap_polish(phi_b, eps=EPS,
                                          max_iters=MAX_SNAP_ITERS,
                                          anchor=anchor)
    t_snap = time.time() - t0
    n_s, m_s = _stats(phi_s)
    l2_s = float(np.linalg.norm(phi_s - anchor))
    extra_l2 = float(np.linalg.norm(phi_s - phi_b))
    print(f'\n[snap done]    n_neg={n_s}  min_tri={m_s:+.6f}  '
          f'L2_total={l2_s:.3f}  delta_L2={extra_l2:.4f}  '
          f'iters={n_iters}  ({t_snap:.1f}s)', flush=True)

    # --- Summary -------------------------------------------------------
    print(f'\n{"=" * 60}', flush=True)
    print('SUMMARY  (z={})'.format(z), flush=True)
    print(f'  init      : n_neg={n0:5d}  min_tri={m0:+.5f}', flush=True)
    print(f'  barrier   : n_neg={n_b:5d}  min_tri={m_b:+.6f}  L2={l2_b:.3f}',
          flush=True)
    print(f'  +snap     : n_neg={n_s:5d}  min_tri={m_s:+.6f}  L2={l2_s:.3f}  '
          f'(+{extra_l2:.4f})', flush=True)
    print(f'  result    : {"CONVERGED (fold-free)" if success else "still folded"}',
          flush=True)


if __name__ == '__main__':
    main()
