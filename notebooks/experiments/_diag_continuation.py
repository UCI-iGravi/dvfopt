"""Diagnostic: WHY does continuation stall crossing T=0 on z=12?

Takes z=12's largest fold component, runs threshold-continuation on just
that one crop, and prints, for every threshold step:
  - the target threshold
  - the min triangle area SLSQP actually achieved
  - SLSQP's status code + message  (the key signal)
  - iteration count and wall time

SLSQP status decode:
  0  = converged
  4  = inequality constraints incompatible  (SLSQP thinks infeasible)
  8  = positive directional derivative in line search  (degeneracy)
  9  = iteration limit reached
A status-0 result whose achieved min is BELOW the target = SLSQP
"converged" onto an infeasible point = boundary degeneracy.

Usage: python _diag_continuation.py [z]   (default z=12)
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
from _bench_worker import _make_2tri_jac_2d, _interior_pack_unpack_2d

DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')
FINAL_THRESHOLD = 0.01

phi_full = np.load(DATA_PATH)
H, W = phi_full.shape[2], phi_full.shape[3]


def main():
    z = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    phi = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
    phi_anchor = phi.copy()

    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    fold = np.minimum(T1, T2) <= 0
    mask = binary_dilation(fold, iterations=1)
    labels, _ = cc_label(mask)
    comps = []
    for sl in find_objects(labels):
        if sl is None:
            continue
        comps.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    comps.sort(key=lambda c: (c[1] - c[0]) * (c[3] - c[2]), reverse=True)
    cy0, cy1, cx0, cx1 = comps[0]
    print(f'z={z}: {len(comps)} fold components', flush=True)
    print(f'largest component bbox: y[{cy0}:{cy1}] x[{cx0}:{cx1}]  '
          f'= {cy1-cy0}x{cx1-cx0} cells', flush=True)

    pad = 4
    y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
    x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
    sy, sx = y1 - y0, x1 - x0
    phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    anc_win = phi_anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, im)
    n_constr = 2 * sy * sx
    print(f'crop: {sy}x{sx} cells, {n_int} movable corner-vars, '
          f'{n_constr} triangle constraints', flush=True)

    z_anchor = pack(anc_win)
    jac_func = _make_2tri_jac_2d(phi_win, im)

    def obj(zz):
        d = zz - z_anchor
        return 0.5 * float(d @ d), d

    def constr(zz):
        ph = unpack(zz, phi_win)
        t1, t2 = _triangle_areas_2d(ph[0], ph[1])
        return np.concatenate([t1.ravel(), t2.ravel()])

    t1, t2 = _triangle_areas_2d(phi_win[0], phi_win[1])
    cur_min = float(min(t1.min(), t2.min()))
    print(f'crop init min_tri = {cur_min:+.5f}\n', flush=True)

    # Instrumented continuation: tight success tolerance (1e-6) so a step
    # only "succeeds" when SLSQP genuinely reaches the threshold.
    sched = list(np.linspace(cur_min - 1e-6, FINAL_THRESHOLD, 15))[1:]
    phi_cur = phi_win.copy()
    thr_prev = cur_min
    print(f'{"thr":>11s} {"got_min":>11s} {"status":>7s} {"nit":>5s} '
          f'{"t_s":>6s}  message', flush=True)
    for _ in range(80):
        if not sched:
            break
        thr = sched[0]
        t0 = time.time()
        nl = NonlinearConstraint(constr, lb=thr, ub=np.inf, jac=jac_func)
        res = minimize(obj, pack(phi_cur), jac=True, method='SLSQP',
                       constraints=[nl],
                       options={'maxiter': 150, 'ftol': 1e-10, 'disp': False})
        phi_try = unpack(res.x, phi_win)
        t1, t2 = _triangle_areas_2d(phi_try[0], phi_try[1])
        got = float(min(t1.min(), t2.min()))
        print(f'{thr:+11.5f} {got:+11.6f} {res.status:7d} {res.nit:5d} '
              f'{time.time()-t0:6.1f}  {res.message}', flush=True)
        if got >= thr - 1e-6:
            phi_cur = phi_try
            thr_prev = thr
            sched.pop(0)
        else:
            thr_mid = 0.5 * (thr_prev + thr)
            if thr - thr_mid < 1e-5:
                print(f'  -> step gap {thr-thr_mid:.2e} too small to '
                      f'subdivide; STOP', flush=True)
                break
            sched.insert(0, thr_mid)

    t1, t2 = _triangle_areas_2d(phi_cur[0], phi_cur[1])
    print(f'\nfinal crop min_tri = {float(min(t1.min(), t2.min())):+.6f}',
          flush=True)


if __name__ == '__main__':
    main()
