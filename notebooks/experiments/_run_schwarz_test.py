"""Headless driver for the hybrid overlapping-tile Schwarz experiment.

Each outer iteration: detect connected fold components and route by size.
  - LARGE component  -> overlapping-tile Schwarz (multiplicative sweeps)
  - small components -> normal process: single frozen-edge crop, the crop
                        grows on stall (per-cell pad boost)
The outer loop re-detects after every pass, so once Schwarz has knocked a
big dense component down to sparse residuals those residuals are picked up
as small components and finished by the normal solver.

Mirrors overlapping_tiles_schwarz.ipynb. Usage: python _run_schwarz_test.py [z ...]
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
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from _bench_worker import solve_cluster_inline

THRESHOLD = 0.01
EPS_L1 = 1e-4
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')
STUCK_Z = [30, 54, 57, 58, 61, 150, 290]

# A fold component is "large" (-> Schwarz) if its bbox exceeds either of
# these; otherwise it is solved as one normal frozen-edge crop.
# LARGE_SPAN=20 was tested and produced WORSE results (plateau at 307
# vs 17 with LARGE_SPAN=40) -- forcing small dense components through
# tile-Schwarz over-constrains them. Keep LARGE_SPAN=40.
LARGE_SPAN = 40       # longer bbox axis, in cells
LARGE_AREA = 1500     # bbox area, in cells

# After this many outer iters of failure on a small dense component,
# switch from the normal single solve to multi-restart: K SLSQP attempts
# from differently-perturbed starting points, keep the first feasible.
# Whole-interior perturbation (not just fold-adjacent corners) -- gives
# meaningful basin diversity rather than nearby local-minimum samples.
MULTI_RESTART_AFTER = 2     # outer iters of failure before switching
MULTI_RESTART_K = 5         # restart attempts per call
MULTI_RESTART_SIGMA = 0.1   # base perturbation sigma (pixel-scale)
MULTI_RESTART_SIGMA_GROW = 1.5

phi_full = np.load(DATA_PATH)
H, W = phi_full.shape[2], phi_full.shape[3]


def fold_stats(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum()), float(min(T1.min(), T2.min()))


def slice_phi(z):
    return np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])


def fold_components(phi, merge_dilation=1):
    """Connected components of the cell-fold mask (after a small dilation
    that merges near-touching folds). Returns list of (cy0, cy1, cx0, cx1)
    cell-coord bboxes."""
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


def make_tiles(bbox, tile, overlap):
    """Overlapping tiles (cell coords) covering bbox=(cy0,cy1,cx0,cx1)."""
    cy0, cy1, cx0, cx1 = bbox
    stride = max(1, tile - overlap)
    out = set()
    for y0 in range(cy0, max(cy0 + 1, cy1), stride):
        for x0 in range(cx0, max(cx0 + 1, cx1), stride):
            y1 = min(y0 + tile, H - 1)
            x1 = min(x0 + tile, W - 1)
            y0c = max(0, y1 - tile)
            x0c = max(0, x1 - tile)
            if (y1 - y0c) >= 4 and (x1 - x0c) >= 4:
                out.add((y0c, y1, x0c, x1))
    return sorted(out)


def solve_crop(phi, phi_anchor, y0, y1, x0, x1, *,
               l2_passes, l2_iter, l1_iter):
    """Solve one crop [y0:y1, x0:x1] (cell coords) with a frozen-edge,
    rectangular interior mask. Splices interior corners back into phi.
    Returns the cluster row dict from solve_cluster_inline."""
    sy, sx = y1 - y0, x1 - x0
    if sy < 4 or sx < 4:
        return {'feasible': False}
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    t1w, t2w = _triangle_areas_2d(phi_win[0], phi_win[1])
    c = dict(cluster_id=0, z=-1, y0=y0, y1=y1, x0=x0, x1=x1,
             crop_cells_y=sy, crop_cells_x=sx,
             component_cells=int((np.minimum(t1w, t2w) <= 0).sum()),
             interior_mask=im, skipped_too_large=False)
    anc = phi_anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
    row, phi_l1 = solve_cluster_inline(c, phi_win, anc, THRESHOLD, EPS_L1,
                                       l2_passes, l2_iter, l1_iter)
    if phi_l1 is not None:
        yy, xx = np.where(im)
        phi[:, y0 + yy, x0 + xx] = phi_l1[:, yy, xx]
    return row


def solve_region_schwarz(phi, phi_anchor, bbox, *, tile=16, overlap=4,
                         max_sweeps=6):
    """Overlapping-tile multiplicative Schwarz on one large component's
    bbox. Light per-tile budget -- Schwarz relies on repeated sweeps, not
    on each tile fully converging."""
    cy0, cy1, cx0, cx1 = bbox
    for sweep in range(max_sweeps):
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        fold = np.minimum(T1, T2) <= 0
        sub = np.zeros_like(fold)
        sub[cy0:cy1, cx0:cx1] = fold[cy0:cy1, cx0:cx1]
        if not sub.any():
            return
        ys, xs = np.where(sub)
        rbox = (int(ys.min()), int(ys.max()) + 1,
                int(xs.min()), int(xs.max()) + 1)
        tiles = make_tiles(rbox, tile, overlap)
        if sweep % 2 == 1:
            tiles = tiles[::-1]
        for (y0, y1, x0, x1) in tiles:
            phi_win = phi[:, y0:y1 + 1, x0:x1 + 1]
            t1w, t2w = _triangle_areas_2d(phi_win[0], phi_win[1])
            if not (np.minimum(t1w, t2w) <= 0).any():
                continue
            solve_crop(phi, phi_anchor, y0, y1, x0, x1,
                       l2_passes=4, l2_iter=30, l1_iter=40)


def _crop_n_neg(phi_win):
    t1, t2 = _triangle_areas_2d(phi_win[0], phi_win[1])
    return int((t1 <= 0).sum() + (t2 <= 0).sum())


def try_multi_restart(phi, phi_anchor, y0, y1, x0, x1, *,
                      n_restarts=None, base_sigma=None,
                      sigma_growth=None, seed_hint=0):
    """Multi-restart SLSQP on the crop [y0:y1, x0:x1] (cell coords).

    Attempt 0 is the unperturbed solve (the plain normal-solve baseline);
    attempts 1..K-1 perturb the WHOLE crop interior with progressively
    widening Gaussian noise. The attempt with the FEWEST residual folds
    is committed -- so this is never worse than the plain solve, and a
    perturbed restart can only help. Whole-interior perturbation gives
    meaningful basin diversity vs. only jittering fold-adjacent corners.

    Splices the best attempt's interior corners back into ``phi``.
    Returns True iff the committed attempt is fully feasible.
    """
    if n_restarts is None:
        n_restarts = MULTI_RESTART_K
    if base_sigma is None:
        base_sigma = MULTI_RESTART_SIGMA
    if sigma_growth is None:
        sigma_growth = MULTI_RESTART_SIGMA_GROW
    sy, sx = y1 - y0, x1 - x0
    if sy < 4 or sx < 4:
        return False
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    yy, xx = np.where(im)

    anc_win = phi_anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
    phi_base = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    best_phi = None
    best_n = None
    for r in range(n_restarts):
        phi_try = phi_base.copy()
        if r > 0:                       # attempt 0 = unperturbed baseline
            sigma = base_sigma * (sigma_growth ** (r - 1))
            rng = np.random.default_rng(seed_hint * 1000003 + r * 31337 + 1)
            phi_try[:, yy, xx] = phi_try[:, yy, xx] + rng.normal(
                scale=sigma, size=(2, len(yy)))
        t1w, t2w = _triangle_areas_2d(phi_try[0], phi_try[1])
        c = dict(cluster_id=0, z=-1, y0=y0, y1=y1, x0=x0, x1=x1,
                 crop_cells_y=sy, crop_cells_x=sx,
                 component_cells=int((np.minimum(t1w, t2w) <= 0).sum()),
                 interior_mask=im, skipped_too_large=False)
        row, phi_l1 = solve_cluster_inline(
            c, phi_try, anc_win, THRESHOLD, EPS_L1, 8, 60, 80)
        if phi_l1 is None:
            continue
        n_res = _crop_n_neg(phi_l1)
        if best_n is None or n_res < best_n:
            best_n = n_res
            best_phi = phi_l1
        if n_res == 0:
            break                       # fully feasible -- stop early
    if best_phi is not None:
        phi[:, y0 + yy, x0 + xx] = best_phi[:, yy, xx]
    return bool(best_n == 0)


def correct_slice_hybrid(phi0, phi_anchor, *, max_outer=30, verbose=True):
    """Outer loop: detect fold components, route LARGE -> Schwarz tiles,
    small -> normal frozen-edge crop with a per-cell pad boost on stall."""
    phi = phi0.copy()
    pad_boost = np.zeros((H - 1, W - 1), dtype=int)
    history = []
    n, m = fold_stats(phi)
    history.append(dict(outer=0, n_neg=n))
    if verbose:
        print(f'  init       : n_neg={n:5d}  min_tri={m:+.4f}', flush=True)
    for outer in range(1, max_outer + 1):
        comps = fold_components(phi, merge_dilation=1)
        if not comps:
            break
        t0 = time.time()
        n_large = n_small = 0
        for (cy0, cy1, cx0, cx1) in comps:
            span = max(cy1 - cy0, cx1 - cx0)
            area = (cy1 - cy0) * (cx1 - cx0)
            if span > LARGE_SPAN or area > LARGE_AREA:
                solve_region_schwarz(phi, phi_anchor, (cy0, cy1, cx0, cx1))
                n_large += 1
            else:
                boost = int(pad_boost[cy0:cy1, cx0:cx1].max())
                pad = 1 + boost
                y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
                x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
                if boost < MULTI_RESTART_AFTER:
                    row = solve_crop(phi, phi_anchor, y0, y1, x0, x1,
                                      l2_passes=12, l2_iter=80, l1_iter=120)
                    success = bool(row.get('feasible'))
                else:
                    # Stalled for >=2 outer iters: switch to multi-restart
                    # with whole-interior perturbation (basin hopping).
                    success = try_multi_restart(
                        phi, phi_anchor, y0, y1, x0, x1,
                        seed_hint=outer * 10007 + cy0 * 31 + cx0 * 17)
                if success:
                    pad_boost[cy0:cy1, cx0:cx1] = 0
                else:
                    pad_boost[cy0:cy1, cx0:cx1] += 1
                n_small += 1
        n, m = fold_stats(phi)
        history.append(dict(outer=outer, n_neg=n))
        if verbose:
            print(f'  outer {outer:2d}   : n_neg={n:5d}  min_tri={m:+.4f}  '
                  f'comps={len(comps):3d} (large={n_large} small={n_small})  '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if n == 0:
            break
    return phi, history


def main():
    targets = [int(z) for z in sys.argv[1:]] or STUCK_Z
    results = []
    for z in targets:
        phi_z = slice_phi(z)
        n0 = fold_stats(phi_z)[0]
        print(f'=== z={z}  (init n_neg={n0}) ===', flush=True)
        t0 = time.time()
        phi_c, hist = correct_slice_hybrid(phi_z, phi_z, max_outer=30)
        wall = time.time() - t0
        nf, mf = fold_stats(phi_c)
        results.append((z, n0, nf, mf, hist[-1]['outer'], wall))
        print(f'    -> final n_neg={nf}  min_tri={mf:+.4f}  '
              f'outer_iters={hist[-1]["outer"]}  wall={wall:.0f}s\n', flush=True)
    print(f'{"z":>5s} {"init":>6s} {"final":>6s} {"min_tri":>9s} '
          f'{"outer":>6s} {"wall_s":>8s}  result', flush=True)
    for z, n0, nf, mf, it, wall in results:
        print(f'{z:5d} {n0:6d} {nf:6d} {mf:+9.4f} {it:6d} {wall:8.0f}  '
              f'{"CONVERGED" if nf == 0 else "still folded"}', flush=True)
    print(f'\nconverged: {sum(1 for r in results if r[2]==0)}/{len(results)}',
          flush=True)


if __name__ == '__main__':
    main()
