"""Headless driver: SLSQP threshold-continuation experiment.

The dense fold cores plateau because windowed SLSQP is asked to traverse
from a deeply-infeasible state (min triangle area ~ -59) to feasible
(>= +0.01) in one solve -- a global search an active-set local method is
bad at. Continuation turns that into a chain of easy local solves:

  for an increasing schedule of thresholds thr_0 < thr_1 < ... < +0.01,
  solve "all triangles >= thr_k" warm-started from the thr_{k-1} result.

thr_0 is set below the crop's current minimum, so the starting state is
already feasible; each step raises the bar slightly and the solution
moves only slightly -- exactly what SLSQP is good at. Steps that fail are
subdivided (adaptive continuation).

Mirrors threshold_continuation.ipynb. Usage: python _run_continuation_test.py [z ...]
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

THRESHOLD = 0.01
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')
# The dense slices the windowed runner could not crack.
DENSE_Z = [11, 12, 13, 14, 16, 17]

N_STEPS = 10            # base continuation steps per crop
STEP_MAX_ITER = 80      # SLSQP iters per continuation step
MAX_SUBSTEPS = 40       # cap on total (incl. subdivided) steps per crop
CLEAN_RING_MAX_PAD = 4  # grow the crop up to this for a feasible ring;
                        # capped low -- a bad frozen edge just stalls the
                        # schedule, and the outer loop re-crops next iter

phi_full = np.load(DATA_PATH)
H, W = phi_full.shape[2], phi_full.shape[3]


def fold_stats(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum()), float(min(T1.min(), T2.min()))


def slice_phi(z):
    return np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])


def fold_components(phi, merge_dilation=1):
    """Connected components of the cell-fold mask. Returns (cy0,cy1,cx0,cx1)."""
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    fold = np.minimum(T1, T2) <= 0
    if not fold.any():
        return []
    mask = (binary_dilation(fold, iterations=merge_dilation)
            if merge_dilation > 0 else fold)
    labels, _ = cc_label(mask)
    comps = []
    for sl in find_objects(labels):
        if sl is None:
            continue
        comps.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return comps


def grow_to_clean_ring(phi, cy0, cy1, cx0, cx1, max_pad=CLEAN_RING_MAX_PAD):
    """Grow the crop pad until the outer ring of cells is all feasible
    (T >= THRESHOLD), so the frozen-edge corners do not lock in folds.
    Returns (y0, y1, x0, x1) cell coords."""
    y0, y1, x0, x1 = cy0, cy1, cx0, cx1
    for pad in range(1, max_pad + 1):
        y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
        x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
        T1, T2 = _triangle_areas_2d(phi[0, y0:y1 + 1, x0:x1 + 1],
                                    phi[1, y0:y1 + 1, x0:x1 + 1])
        cmin = np.minimum(T1, T2)
        if cmin.shape[0] < 3 or cmin.shape[1] < 3:
            continue
        ring = np.concatenate([cmin[0, :], cmin[-1, :],
                               cmin[:, 0], cmin[:, -1]])
        full = (y0 == 0 and y1 == H - 1 and x0 == 0 and x1 == W - 1)
        if ring.min() >= THRESHOLD or full:
            break
    return y0, y1, x0, x1


def _run_continuation(phi_win, anc_win, im, *, final_threshold=THRESHOLD,
                      n_steps=N_STEPS, max_iter=STEP_MAX_ITER):
    """Threshold-continuation on one crop with movable corners given by
    ``im`` (True = movable). Returns (phi_result, feasible)."""
    pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, im)
    if n_int == 0:
        t1, t2 = _triangle_areas_2d(phi_win[0], phi_win[1])
        return phi_win.copy(), bool(min(t1.min(), t2.min())
                                    >= final_threshold - 1e-4)
    z_anchor = pack(anc_win)
    jac_func = _make_2tri_jac_2d(phi_win, im)

    def obj(z):
        d = z - z_anchor
        return 0.5 * float(d @ d), d

    def constr(z):
        ph = unpack(z, phi_win)
        T1, T2 = _triangle_areas_2d(ph[0], ph[1])
        return np.concatenate([T1.ravel(), T2.ravel()])

    T1, T2 = _triangle_areas_2d(phi_win[0], phi_win[1])
    cur_min = float(min(T1.min(), T2.min()))
    if cur_min >= final_threshold:
        return phi_win.copy(), True

    # Continuation schedule: from just below the current minimum up to
    # the final threshold. Processed as a stack so failed steps subdivide.
    sched = list(np.linspace(cur_min - 1e-6, final_threshold, n_steps + 1))[1:]
    phi_cur = phi_win.copy()
    thr_prev = cur_min
    substeps = 0
    while sched and substeps < MAX_SUBSTEPS:
        thr = sched[0]
        z_init = pack(phi_cur)
        nl = NonlinearConstraint(constr, lb=thr, ub=np.inf, jac=jac_func)
        res = minimize(obj, z_init, jac=True, method='SLSQP',
                       constraints=[nl],
                       options={'maxiter': max_iter, 'ftol': 1e-9,
                                'disp': False})
        substeps += 1
        phi_try = unpack(res.x, phi_win)
        t1, t2 = _triangle_areas_2d(phi_try[0], phi_try[1])
        got = float(min(t1.min(), t2.min()))
        if got >= thr - 1e-4:
            phi_cur = phi_try            # step succeeded -> advance
            thr_prev = thr
            sched.pop(0)
        else:
            thr_mid = 0.5 * (thr_prev + thr)   # step failed -> subdivide
            if thr - thr_mid < 1e-4:
                break                    # cannot subdivide further
            sched.insert(0, thr_mid)

    t1, t2 = _triangle_areas_2d(phi_cur[0], phi_cur[1])
    feasible = bool(min(t1.min(), t2.min()) >= final_threshold - 1e-4)
    return phi_cur, feasible


def solve_crop_continuation(phi, phi_anchor, cy0, cy1, cx0, cx1, *,
                            final_threshold=THRESHOLD, n_steps=N_STEPS,
                            max_iter=STEP_MAX_ITER,
                            clean_ring_pad=CLEAN_RING_MAX_PAD,
                            allow_loosen=True):
    """Threshold-continuation on one fold component.

    Phase 1: frozen 1-corner ring (the normal windowed solve).
    Phase 2 (only if phase 1 stalls and allow_loosen): re-solve with the
    ring **loosened** -- every corner movable except those on the genuine
    image boundary. That lets a frozen-edge-pinned core fully unfold; the
    cost is a thin seam of new shallow folds just outside the crop, which
    the outer loop picks up as small clusters and the normal frozen solve
    cleans up. Splices the solved corners back into ``phi``.
    Returns True iff the crop ends feasible."""
    y0, y1, x0, x1 = grow_to_clean_ring(phi, cy0, cy1, cx0, cx1,
                                        max_pad=clean_ring_pad)
    sy, sx = y1 - y0, x1 - x0
    if sy < 4 or sx < 4:
        return False
    phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    anc_win = phi_anchor[:, y0:y1 + 1, x0:x1 + 1].copy()

    # --- Phase 1: frozen 1-ring -----------------------------------------
    im_frozen = np.zeros((sy + 1, sx + 1), dtype=bool)
    im_frozen[1:-1, 1:-1] = True
    phi_res, feasible = _run_continuation(
        phi_win, anc_win, im_frozen, final_threshold=final_threshold,
        n_steps=n_steps, max_iter=max_iter)
    if feasible or not allow_loosen:
        yy, xx = np.where(im_frozen)
        phi[:, y0 + yy, x0 + xx] = phi_res[:, yy, xx]
        return feasible

    # --- Phase 2: stuck -> loosen the ring ------------------------------
    # Free every corner except those on the genuine image boundary.
    im_loose = np.ones((sy + 1, sx + 1), dtype=bool)
    if y0 == 0:
        im_loose[0, :] = False
    if y1 == H - 1:
        im_loose[-1, :] = False
    if x0 == 0:
        im_loose[:, 0] = False
    if x1 == W - 1:
        im_loose[:, -1] = False
    phi_res2, feasible2 = _run_continuation(
        phi_res, anc_win, im_loose, final_threshold=final_threshold,
        n_steps=n_steps, max_iter=max_iter)
    yy, xx = np.where(im_loose)
    phi[:, y0 + yy, x0 + xx] = phi_res2[:, yy, xx]
    return feasible2


def correct_slice_continuation(phi0, phi_anchor, *, max_outer=12, verbose=True):
    """Outer loop: detect fold components, solve each by threshold
    continuation. Re-detect and repeat until 0 folds."""
    phi = phi0.copy()
    history = []
    n, m = fold_stats(phi)
    history.append(dict(outer=0, n_neg=n))
    if verbose:
        print(f'  init      : n_neg={n:5d}  min_tri={m:+.4f}', flush=True)
    for outer in range(1, max_outer + 1):
        comps = fold_components(phi, merge_dilation=1)
        if not comps:
            break
        t0 = time.time()
        n_ok = 0
        # Small clean-ring pad (fast). A frozen-edge-pinned core is no
        # longer fixed by growing the crop -- the loosen-on-stuck phase
        # in solve_crop_continuation handles it instead.
        for (cy0, cy1, cx0, cx1) in comps:
            if solve_crop_continuation(phi, phi_anchor, cy0, cy1, cx0, cx1,
                                       clean_ring_pad=4):
                n_ok += 1
        n, m = fold_stats(phi)
        history.append(dict(outer=outer, n_neg=n))
        if verbose:
            print(f'  outer {outer:2d}  : n_neg={n:5d}  min_tri={m:+.4f}  '
                  f'comps={len(comps):3d}  feas={n_ok:3d}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if n == 0:
            break
    return phi, history


def main():
    targets = [int(z) for z in sys.argv[1:]] or DENSE_Z
    results = []
    for z in targets:
        phi_z = slice_phi(z)
        n0 = fold_stats(phi_z)[0]
        print(f'=== z={z}  (init n_neg={n0}) ===', flush=True)
        t0 = time.time()
        phi_c, hist = correct_slice_continuation(phi_z, phi_z, max_outer=12)
        wall = time.time() - t0
        nf, mf = fold_stats(phi_c)
        results.append((z, n0, nf, mf, hist[-1]['outer'], wall))
        print(f'    -> final n_neg={nf}  min_tri={mf:+.4f}  '
              f'outer={hist[-1]["outer"]}  wall={wall:.0f}s\n', flush=True)
    print(f'{"z":>5s} {"init":>6s} {"final":>6s} {"min_tri":>9s} '
          f'{"outer":>6s} {"wall_s":>8s}  result', flush=True)
    for z, n0, nf, mf, it, wall in results:
        print(f'{z:5d} {n0:6d} {nf:6d} {mf:+9.4f} {it:6d} {wall:8.0f}  '
              f'{"CONVERGED" if nf == 0 else "still folded"}', flush=True)
    print(f'\nconverged: {sum(1 for r in results if r[2]==0)}/{len(results)}',
          flush=True)


if __name__ == '__main__':
    main()
