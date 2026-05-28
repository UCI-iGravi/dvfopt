"""Test: does decomposing z=12's dense fold core into SMALL blocks let
SLSQP cross the boundary the monolithic solve cannot?

The continuation diagnostic showed SLSQP degenerates (status 8, "positive
directional derivative for linesearch") when ~1472 triangle constraints
crowd the boundary at once -- but the negative-threshold range solves
cleanly. So:

  Phase A: continuation on the whole component crop up to thr = -0.05.
           Deep global unfolding; SLSQP's good regime. Small blocks can't
           do this (deep folds need wide displacement budget).
  Phase B: tile the crop into small B-cell blocks, no frozen *interior*
           edges (only the component-crop's outer ring stays fixed),
           block-Gauss-Seidel sweeps, each block run by continuation from
           its current min up to +0.01. Small block = few constraints =
           SLSQP stays out of the degenerate regime.

Runs on z=12's largest fold component. Usage: python _run_blocks_test.py [z]
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
from _run_continuation_test import _run_continuation, DATA_PATH

THRESHOLD = 0.01
PHASE_A_STOP = -0.05    # continuation target for phase A
BLOCK = 8               # phase-B block size in cells
OVERLAP = 3             # phase-B block overlap in cells
MAX_SWEEPS = 15         # phase-B block-Gauss-Seidel sweeps

phi_full = np.load(DATA_PATH)
H, W = phi_full.shape[2], phi_full.shape[3]


def crop_min(phi_win):
    t1, t2 = _triangle_areas_2d(phi_win[0], phi_win[1])
    return float(min(t1.min(), t2.min()))


def crop_nneg(phi_win):
    t1, t2 = _triangle_areas_2d(phi_win[0], phi_win[1])
    return int((t1 <= 0).sum() + (t2 <= 0).sum())


def phase_b_blocks(phi_win, anc_win, *, block=BLOCK, overlap=OVERLAP,
                   max_sweeps=MAX_SWEEPS):
    """Block-Gauss-Seidel sweeps. Each block is a small sub-crop solved by
    continuation; its movable corners are all block corners except those
    on the component-crop's outer ring (the slice interface stays fixed).
    Splices each block immediately so the next sees the update."""
    SY, SX = phi_win.shape[1], phi_win.shape[2]      # corner dims
    syc, sxc = SY - 1, SX - 1                        # cell dims
    stride = max(1, block - overlap)
    for sweep in range(1, max_sweeps + 1):
        if crop_min(phi_win) >= THRESHOLD:
            return True, sweep - 1
        t0 = time.time()
        n_blocks = 0
        rev = (sweep % 2 == 0)
        y_starts = list(range(0, syc, stride))
        x_starts = list(range(0, sxc, stride))
        if rev:
            y_starts = y_starts[::-1]
            x_starts = x_starts[::-1]
        for by0 in y_starts:
            for bx0 in x_starts:
                by1 = min(by0 + block, syc)
                bx1 = min(bx0 + block, sxc)
                by0c = max(0, by1 - block)
                bx0c = max(0, bx1 - block)
                bh, bw = by1 - by0c + 1, bx1 - bx0c + 1
                if bh < 4 or bw < 4:
                    continue
                blk = phi_win[:, by0c:by1 + 1, bx0c:bx1 + 1].copy()
                if crop_nneg(blk) == 0:
                    continue
                blk_anc = anc_win[:, by0c:by1 + 1, bx0c:bx1 + 1].copy()
                # movable = all block corners except component-ring corners
                mask = np.ones((bh, bw), dtype=bool)
                for r in range(bh):
                    if by0c + r == 0 or by0c + r == SY - 1:
                        mask[r, :] = False
                for c in range(bw):
                    if bx0c + c == 0 or bx0c + c == SX - 1:
                        mask[:, c] = False
                if not mask.any():
                    continue
                res, _feas = _run_continuation(
                    blk, blk_anc, mask, final_threshold=THRESHOLD,
                    n_steps=10, max_iter=80)
                yy, xx = np.where(mask)
                phi_win[:, by0c + yy, bx0c + xx] = res[:, yy, xx]
                n_blocks += 1
        print(f'    sweep {sweep:2d}: min_tri={crop_min(phi_win):+.6f}  '
              f'n_neg={crop_nneg(phi_win):4d}  blocks={n_blocks:3d}  '
              f'({time.time()-t0:.0f}s)', flush=True)
    return crop_min(phi_win) >= THRESHOLD, max_sweeps


def main():
    z = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    phi = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
    phi_anchor = phi.copy()

    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    fold = np.minimum(T1, T2) <= 0
    labels, _ = cc_label(binary_dilation(fold, iterations=1))
    comps = []
    for sl in find_objects(labels):
        if sl is None:
            continue
        comps.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    comps.sort(key=lambda c: (c[1] - c[0]) * (c[3] - c[2]), reverse=True)
    cy0, cy1, cx0, cx1 = comps[0]
    print(f'z={z}: largest component y[{cy0}:{cy1}] x[{cx0}:{cx1}] '
          f'= {cy1-cy0}x{cx1-cx0} cells', flush=True)

    pad = 4
    y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
    x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
    phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
    anc_win = phi_anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
    sy, sx = y1 - y0, x1 - x0
    print(f'crop {sy}x{sx} cells, init min_tri={crop_min(phi_win):+.4f}\n',
          flush=True)

    # --- Phase A: whole-crop continuation into the negative regime ------
    im = np.zeros((sy + 1, sx + 1), dtype=bool)
    im[1:-1, 1:-1] = True
    t0 = time.time()
    phi_a, _ = _run_continuation(phi_win, anc_win, im,
                                 final_threshold=PHASE_A_STOP,
                                 n_steps=12, max_iter=120)
    print(f'phase A (-> thr {PHASE_A_STOP}): min_tri={crop_min(phi_a):+.5f}  '
          f'n_neg={crop_nneg(phi_a)}  ({time.time()-t0:.0f}s)\n', flush=True)

    # --- Phase B: small-block Gauss-Seidel across the boundary ----------
    t0 = time.time()
    feas, sweeps = phase_b_blocks(phi_a, anc_win)
    print(f'\nphase B: {"CONVERGED" if feas else "still folded"}  '
          f'final min_tri={crop_min(phi_a):+.6f}  sweeps={sweeps}  '
          f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
