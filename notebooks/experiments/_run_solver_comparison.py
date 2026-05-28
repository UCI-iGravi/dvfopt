"""Solver comparison on z=12's dense fold core.

Windowed SLSQP degenerates here (diagnostic: status 8, "positive
directional derivative for linesearch", once ~1472 triangle constraints
crowd the boundary). This module tests the two solver families that
handle a coupled region without that degeneracy:

  trust-constr : scipy's large-scale trust-region interior-point method.
                 Solved per fold component with the 2-triangle
                 constraint (the manuscript's feasibility check) and the
                 analytical Jacobian. Frozen 1-ring crop (the pad keeps
                 the ring clean -- the degeneracy was shown NOT to be a
                 frozen-edge effect).

  barrier      : dvfopt.core.iterative_2d_barrier, full-grid mode
                 (windowed=False) -- the whole slice optimised at once,
                 no windowing, no frozen edges. Penalty -> log-barrier
                 under L-BFGS-B. NOTE: it enforces Jdet >= threshold, not
                 the 2-triangle constraint, so its 2-triangle result is
                 reported for honest comparison.

Reports, per solver: final 2-triangle n_neg and min_tri, and wall time.
Usage: python _run_solver_comparison.py [z ...]   (default z=12)
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
_MANU = os.path.join(_REPO, 'notebooks', 'manuscript')
sys.path.insert(0, _REPO)
sys.path.insert(0, _MANU)

import numpy as np
from scipy.ndimage import label as cc_label, binary_dilation, find_objects
from scipy.optimize import minimize, NonlinearConstraint
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.core.iterative2d_barrier import iterative_2d_barrier
from _bench_worker import _make_2tri_jac_2d, _interior_pack_unpack_2d

THRESHOLD = 0.01
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

phi_full = np.load(DATA_PATH)
H, W = phi_full.shape[2], phi_full.shape[3]


def slice_phi(z):
    return np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])


def tri_stats(phi2):
    """2-triangle feasibility: (n_neg, min_tri) for a (2,H,W) field."""
    t1, t2 = _triangle_areas_2d(phi2[0], phi2[1])
    return int((t1 <= 0).sum() + (t2 <= 0).sum()), float(min(t1.min(), t2.min()))


def fold_components(phi, merge_dilation=1):
    t1, t2 = _triangle_areas_2d(phi[0], phi[1])
    fold = np.minimum(t1, t2) <= 0
    if not fold.any():
        return []
    mask = (binary_dilation(fold, iterations=merge_dilation)
            if merge_dilation > 0 else fold)
    labels, _ = cc_label(mask)
    out = []
    for sl in find_objects(labels):
        if sl is not None:
            out.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return out


# ---------------------------------------------------------------- trust-constr
def _solve_component_tc(phi, anchor, bbox, *, pad=3, maxiter=400):
    cy0, cy1, cx0, cx1 = bbox
    y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
    x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
    sy, sx = y1 - y0, x1 - x0
    if sy < 4 or sx < 4:
        return
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    anc_win = anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
    pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, im)
    if n_int == 0:
        return
    z_anchor = pack(anc_win)
    jac = _make_2tri_jac_2d(phi_win, im)

    def obj(z):
        d = z - z_anchor
        return 0.5 * float(d @ d), d

    def con(z):
        ph = unpack(z, phi_win)
        t1, t2 = _triangle_areas_2d(ph[0], ph[1])
        return np.concatenate([t1.ravel(), t2.ravel()])

    nl = NonlinearConstraint(con, THRESHOLD, np.inf, jac=jac)
    res = minimize(obj, pack(phi_win), jac=True, method='trust-constr',
                   constraints=[nl],
                   options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10,
                            'verbose': 0})
    phi_new = unpack(res.x, phi_win)
    yy, xx = np.where(im)
    phi[:, y0 + yy, x0 + xx] = phi_new[:, yy, xx]


def run_trustconstr(z, *, max_outer=10):
    phi = slice_phi(z)
    anchor = phi.copy()
    n0, m0 = tri_stats(phi)
    print(f'[trust-constr] z={z}: init n_neg={n0}  min_tri={m0:+.4f}',
          flush=True)
    t0 = time.time()
    for outer in range(1, max_outer + 1):
        comps = fold_components(phi, merge_dilation=1)
        if not comps:
            break
        for bbox in comps:
            _solve_component_tc(phi, anchor, bbox)
        n, m = tri_stats(phi)
        print(f'  outer {outer:2d}: n_neg={n:5d}  min_tri={m:+.5f}  '
              f'comps={len(comps):3d}  ({time.time()-t0:.0f}s)', flush=True)
        if n == 0:
            break
    n, m = tri_stats(phi)
    return n, m, time.time() - t0


# --------------------------------------------------------------------- barrier
def run_barrier(z):
    deformation = phi_full[:, z:z + 1, :, :].copy()   # (3, 1, H, W)
    t0 = time.time()
    phi_c = iterative_2d_barrier(deformation, threshold=THRESHOLD,
                                 windowed=False, verbose=1)
    wall = time.time() - t0
    phi_c = np.asarray(phi_c)
    if phi_c.ndim == 4:                # (3,1,H,W) -> (2,H,W) [dy,dx]
        phi_c = np.stack([phi_c[1, 0], phi_c[2, 0]])
    elif phi_c.shape[0] == 3:          # (3,H,W)
        phi_c = phi_c[1:]
    n, m = tri_stats(phi_c)
    return n, m, wall


def main():
    targets = [int(z) for z in sys.argv[1:]] or [12]
    rows = []
    for z in targets:
        print(f'\n{"="*22} z={z} {"="*22}', flush=True)
        n0, m0 = tri_stats(slice_phi(z))
        tn, tm, tt = run_trustconstr(z)
        print(f'\n[barrier] z={z}: full-grid penalty->log-barrier L-BFGS-B',
              flush=True)
        bn, bm, bt = run_barrier(z)
        rows.append((z, n0, m0, tn, tm, tt, bn, bm, bt))

    print(f'\n{"="*60}', flush=True)
    print('COMPARISON  (2-triangle metric: n_neg, min_tri)', flush=True)
    print(f'{"z":>4s} {"init_neg":>9s} | {"tc_neg":>7s} {"tc_min":>9s} '
          f'{"tc_s":>7s} | {"bar_neg":>8s} {"bar_min":>9s} {"bar_s":>7s}',
          flush=True)
    for (z, n0, m0, tn, tm, tt, bn, bm, bt) in rows:
        print(f'{z:4d} {n0:9d} | {tn:7d} {tm:+9.4f} {tt:7.0f} | '
              f'{bn:8d} {bm:+9.4f} {bt:7.0f}', flush=True)
    print(f'\ntrust-constr CONVERGED: '
          f'{sum(1 for r in rows if r[3] == 0)}/{len(rows)}', flush=True)
    print(f'barrier CONVERGED (2-tri): '
          f'{sum(1 for r in rows if r[6] == 0)}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
