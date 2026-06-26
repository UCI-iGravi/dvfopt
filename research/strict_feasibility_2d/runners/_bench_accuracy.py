"""Recover the 13.5% L1 gap that cluster decomposition leaves vs a global
solve — at much less than the global solve's 18x cost.

(A) merge_dilation sweep: bigger dilation merges nearby folds into fewer,
    larger clusters => fewer frozen-ring boundaries constraining the
    solution => lower L1 (toward the global optimum), at the cost of
    larger per-cluster LPs. Find the L1/speed knee.

(B) warm-started global L1 polish: after the (already feasible) cluster
    solution, run a few GLOBAL trust-region L1-LP steps linearised at the
    current iterate and anchored to the ORIGINAL input. Since it starts
    feasible (no seed needed), it should close much of the gap in a few
    global LP solves, far cheaper than the from-scratch global slp_iter.

Reference (z=450): clustered L1=2371.7, global L1=2089.6 (gap +13.5%).
Guarded for spawn.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))


def _n_neg(out):
    from dvfopt.core.tri_primitives import tri_areas_flat
    H, W = out.shape[1:]
    return int((tri_areas_flat(
        np.concatenate([out[0].ravel(), out[1].ravel()]), H, W) <= 0).sum())


def _global_l1_polish(phi_2hw, phi_in_2hw, threshold, n_steps=3, trust=0.25):
    """A few global trust-region L1-LP steps from a feasible iterate,
    anchored to the original input. Returns (phi_out, info)."""
    from research.strict_feasibility_2d.algorithms.tri_linearize import (
        linearize_T_2tri,
    )
    from research.strict_feasibility_2d.algorithms.highs_solver import (
        solve_l1_lp_step,
    )
    H, W = phi_2hw.shape[1:]
    anchor = np.concatenate([phi_in_2hw[0].ravel(), phi_in_2hw[1].ravel()])
    cur = np.concatenate([phi_2hw[0].ravel(), phi_2hw[1].ravel()])
    inner_thr = threshold + 1e-4
    for _ in range(n_steps):
        T_lin, J = linearize_T_2tri(cur, H, W)
        nxt, st = solve_l1_lp_step(
            phi_in_flat=anchor, phi_lin_flat=cur, T_lin=T_lin, J_sparse=J,
            threshold=inner_thr, trust_radius=trust,
        )
        if not st['success']:
            break
        cur = nxt
    out = np.stack([cur[:H * W].reshape(H, W), cur[H * W:].reshape(H, W)])
    return out


def main():
    from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
        cluster_slp_iter,
    )
    THR = 0.01
    raw = np.load('data/dvfs/archive/new_b0039_laplacian_deformation_field.npz')['arr']
    z = 450
    sl = raw[1:3, z].astype(np.float64)
    GLOBAL_L1 = 2089.6  # reference from _bench_experimental

    print(f'=== (A) merge_dilation sweep (z={z}, target global L1={GLOBAL_L1}) ===',
          flush=True)
    print(f'{"merge_dil":>9} | {"wall(s)":>8} | {"n_neg":>5} | {"L1":>9} | {"gap%":>7}',
          flush=True)
    for md in [1, 2, 4, 8]:
        t0 = time.time()
        out, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                                  n_workers=16, scheduler='continuous',
                                  merge_dilation=md)
        dt = time.time() - t0
        l1 = float(np.abs(out - sl).sum())
        print(f'{md:>9} | {dt:>8.1f} | {_n_neg(out):>5} | {l1:>9.1f} | '
              f'{(l1-GLOBAL_L1)/GLOBAL_L1*100:>+6.1f}%', flush=True)

    print(f'\n=== (B) warm-started global L1 polish (z={z}) ===', flush=True)
    t0 = time.time()
    base, _ = cluster_slp_iter(sl, threshold=THR, max_outer_iters=6,
                               n_workers=16, scheduler='continuous')
    t_cluster = time.time() - t0
    l1_base = float(np.abs(base - sl).sum())
    print(f'  cluster: L1={l1_base:.1f} ({t_cluster:.1f}s, n_neg={_n_neg(base)})',
          flush=True)
    for nsteps in [1, 3]:
        t0 = time.time()
        pol = _global_l1_polish(base, sl, THR, n_steps=nsteps)
        dt = time.time() - t0
        l1 = float(np.abs(pol - sl).sum())
        print(f'  +{nsteps} global polish step(s): L1={l1:.1f} '
              f'(+{dt:.1f}s, n_neg={_n_neg(pol)}) '
              f'gap={(l1-GLOBAL_L1)/GLOBAL_L1*100:+.1f}%', flush=True)


if __name__ == '__main__':
    main()
