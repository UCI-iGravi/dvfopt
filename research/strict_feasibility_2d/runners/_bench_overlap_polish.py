"""Part XXI option D: feasibility-checked overlapping L1 polish sweeps.

Goal: close the ~13.5% sparse-slice L1 gap of frozen-ring clustering
without the 18x global solve. The earlier naive global polish broke
feasibility (no acceptance check). This version:

  * windows = connected regions where the cluster solution differs from
    the input (padded, so they cover the frozen rings whose L1 the
    decomposition left on the table), plus an offset second sweep;
  * per window: hard L1-LP steps anchored to the ORIGINAL input,
    linearized at the current iterate, trust-region;
  * EXACT-feasibility acceptance: candidate accepted only if the crop's
    exact areas stay >= threshold; else shrink trust and retry;
  * frozen-ring interior splice.

Reference (z=450): clustered L1=2371.7, global L1=2089.6 (+13.5%).
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

sys.path.insert(0, str(Path(__file__).parents[3]))

from dvfopt.core.slp.highs_solver import solve_l1_lp_step  # noqa: E402
from dvfopt.core.slp.tri_linearize import linearize_T_2tri  # noqa: E402
from dvfopt.core.tri_primitives import tri_areas_flat  # noqa: E402

THR = 0.01


def _areas(c):
    H, W = c.shape[1:]
    return tri_areas_flat(np.concatenate([c[0].ravel(), c[1].ravel()]), H, W)


def _polish_window(cur_c, orig_c, n_steps=3, trust0=0.25):
    """Feasibility-checked L1 polish of one window crop."""
    Hc, Wc = cur_c.shape[1:]
    anchor = np.concatenate([orig_c[0].ravel(), orig_c[1].ravel()])
    cur = np.concatenate([cur_c[0].ravel(), cur_c[1].ravel()])
    trust = trust0
    for _ in range(n_steps):
        T_lin, J = linearize_T_2tri(cur, Hc, Wc)
        accepted = False
        for _try in range(4):
            cand, st = solve_l1_lp_step(
                phi_in_flat=anchor, phi_lin_flat=cur, T_lin=T_lin,
                J_sparse=J, threshold=THR + 1e-4, trust_radius=trust,
            )
            if st['success']:
                a = tri_areas_flat(cand, Hc, Wc)
                if float(a.min()) >= THR - 1e-9:      # exact-feasibility gate
                    l1_new = float(np.abs(cand - anchor).sum())
                    l1_cur = float(np.abs(cur - anchor).sum())
                    if l1_new < l1_cur - 1e-9:
                        cur = cand
                        accepted = True
                        break
            trust *= 0.5
        if not accepted:
            break
    return np.stack([cur[: Hc * Wc].reshape(Hc, Wc), cur[Hc * Wc:].reshape(Hc, Wc)])


def polish_sweeps(out, orig, sweeps=2, pad=6, verbose=1):
    """Overlapping window sweeps over the changed regions."""
    H, W = out.shape[1:]
    cur = out.copy()
    for s in range(sweeps):
        changed = (np.abs(cur - orig).sum(axis=0) > 1e-12)
        merged = binary_dilation(changed, iterations=pad + s * 3)  # offset growth
        labels, _ = cc_label(merged)
        n_win = 0
        for bbox in find_objects(labels):
            if bbox is None:
                continue
            y0, y1 = max(0, bbox[0].start - 1), min(H, bbox[0].stop + 1)
            x0, x1 = max(0, bbox[1].start - 1), min(W, bbox[1].stop + 1)
            if (y1 - y0) < 4 or (x1 - x0) < 4:
                continue
            win = (slice(y0, y1), slice(x0, x1))
            fixed = _polish_window(cur[:, win[0], win[1]].copy(),
                                   orig[:, win[0], win[1]].copy())
            cur[:, y0 + 1:y1 - 1, x0 + 1:x1 - 1] = fixed[:, 1:-1, 1:-1]
            n_win += 1
        if verbose:
            a = _areas(cur)
            print(f'  sweep {s + 1}: {n_win} windows  '
                  f'L1={float(np.abs(cur - orig).sum()):.1f}  '
                  f'min_T={float(a.min()):+.5f}  n_neg={int((a <= 0).sum())}',
                  flush=True)
    return cur


def main():
    from dvfopt.core.slp import cluster_slp_iter

    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    for z, ref_global in ((450, 2089.6), (300, None)):
        sl = raw[1:3, z].astype(np.float64)
        print(f'\n=== z={z} ===', flush=True)
        t0 = time.time()
        base, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                   n_workers=16, scheduler='continuous')
        t_base = time.time() - t0
        l1b = float(np.abs(base - sl).sum())
        a = _areas(base)
        print(f'  cluster baseline: L1={l1b:.1f}  n_neg={int((a <= 0).sum())} '
              f' ({t_base:.1f}s)', flush=True)
        t0 = time.time()
        pol = polish_sweeps(base, sl, sweeps=2)
        t_pol = time.time() - t0
        l1p = float(np.abs(pol - sl).sum())
        a = _areas(pol)
        gap = f'  gap vs global={((l1p - ref_global) / ref_global * 100):+.1f}%' \
            if ref_global else ''
        print(f'  polished: L1={l1p:.1f} ({(l1b - l1p) / l1b * 100:.1f}% recovered)'
              f'  n_neg={int((a <= 0).sum())}  min_T={float(a.min()):+.5f}'
              f'  +{t_pol:.1f}s{gap}', flush=True)


if __name__ == '__main__':
    main()
