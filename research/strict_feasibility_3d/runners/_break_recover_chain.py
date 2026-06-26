"""Break-and-recover chain: chain together (perturb -> recover)
cycles starting from the 2-fold ms_v2_tight checkpoint, aiming
to reach n_neg=0.

Observation from v2: M14Tet's aggressive repair shattered the
9-fold MS_V1 result into 497 folds, but a subsequent M10Tet @
0.012 recovered to a strictly better 2-fold basin. The
"deliberate perturbation + barrier recovery" pattern escapes
local minima.

This script tries multiple perturbation strategies:
  1. M14Tet break + M10Tet @ 0.012 recover
  2. Random Gaussian noise on residual cells + M10Tet recover
  3. M10Tet @ very tight threshold (0.018) + M10Tet @ 0.012 recover
  4. Pure M10Tet @ 0.012 iteration (no break)

Each cycle is logged; if any reaches n_neg=0, we save and stop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def report(phi, label, phi_input=None):
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    mn = float(V.min())
    L1 = '' if phi_input is None else f'  L1={float(np.abs(phi - phi_input).sum()):.1f}'
    print(f'{label}: n_neg={n_neg}  n<0.01={n_below}  min_T={mn:+.6f}{L1}',
          flush=True)
    return n_neg, n_below, mn


def m10tet(phi, thr):
    """Single M10Tet pass at given threshold."""
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=thr,
    )
    return solver.fit(phi).corrected


def m14tet(phi, thr):
    """Single M14Tet pass (refine + repair)."""
    from dvfopt import (
        HarmonicALMRefineRepair3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMRefineRepair3DStrategy(),
        threshold=thr,
    )
    return solver.fit(phi).corrected


def perturb_near_folds(phi, sigma=0.05, ring=3):
    """Add small Gaussian perturbation to corners near current fold zones."""
    V = six_tet_volumes_3d(phi)
    fold_cells = (V.min(axis=0) < THRESHOLD)
    # Inflate to corners.
    from scipy.ndimage import binary_dilation
    inflated = binary_dilation(fold_cells, iterations=ring)
    # Apply perturbation to the corner field within inflated zone.
    out = phi.copy()
    rng = np.random.default_rng(42)
    D, H, W = phi.shape[1:]
    Dc, Hc, Wc = inflated.shape
    # Map cell mask to corner mask: each cell mask cell touches 8 corners.
    corner_mask = np.zeros((D, H, W), dtype=bool)
    cz, cy, cx = np.where(inflated)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                corner_mask[cz + dz, cy + dy, cx + dx] = True
    for c in range(3):
        noise = rng.standard_normal((D, H, W)) * sigma
        out[c] += noise * corner_mask
    return out


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    start = np.load(OUTPUT / 'b0039_z0_15_ms_v2_tight.npy').astype(np.float64)
    report(phi_input, 'Original input')
    n_neg, n_below, _ = report(start, 'START (saved 2-fold)', phi_input)

    if n_neg == 0 and n_below == 0:
        print('*** Already STRICT 100% feasible ***', flush=True)
        return

    cur = start
    best_n_neg = n_neg
    best_state = cur.copy()

    # Cycle 1: pure M10Tet @ 0.012 iteration (the cheapest path).
    print('\n=== Cycle 1: pure M10Tet @ 0.012 (3 iters) ===', flush=True)
    for it in range(3):
        t0 = time.time()
        new = m10tet(cur, 0.012)
        n_neg, n_below, _ = report(new, f'  iter {it+1}/3', phi_input)
        print(f'    wall={time.time()-t0:.1f}s', flush=True)
        if n_neg < best_n_neg:
            best_n_neg = n_neg; best_state = new.copy()
            print(f'    *** new best: n_neg={n_neg} ***', flush=True)
        cur = new
        if n_neg == 0 and n_below == 0:
            print(f'\n*** STRICT 100% feas at cycle 1 iter {it+1} ***',
                  flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_chain.npy', new)
            return

    # Cycle 2: small-perturb then M10Tet @ 0.012.
    print('\n=== Cycle 2: small Gaussian perturbation + M10Tet @ 0.012 ===',
          flush=True)
    for sigma in (0.02, 0.05, 0.10):
        cur_pre = best_state.copy()
        pert = perturb_near_folds(cur_pre, sigma=sigma, ring=3)
        n_pert, _, _ = report(pert, f'  perturbed sigma={sigma}', phi_input)
        t0 = time.time()
        rec = m10tet(pert, 0.012)
        n_neg, n_below, _ = report(rec, f'  recovered sigma={sigma}', phi_input)
        print(f'    wall={time.time()-t0:.1f}s', flush=True)
        if n_neg < best_n_neg:
            best_n_neg = n_neg; best_state = rec.copy()
            print(f'    *** new best: n_neg={n_neg} ***', flush=True)
        if n_neg == 0 and n_below == 0:
            print(f'\n*** STRICT 100% feas at cycle 2 sigma={sigma} ***',
                  flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_chain.npy', rec)
            return

    # Cycle 3: M14Tet break + M10Tet @ 0.012 recover.
    print('\n=== Cycle 3: M14Tet break + M10Tet @ 0.012 recover ===',
          flush=True)
    t0 = time.time()
    broken = m14tet(best_state.copy(), 0.015)
    n_b, _, _ = report(broken, '  m14-broken', phi_input)
    print(f'    m14 wall={time.time()-t0:.1f}s', flush=True)
    t1 = time.time()
    rec3 = m10tet(broken, 0.012)
    n_neg, n_below, _ = report(rec3, '  recovered', phi_input)
    print(f'    recover wall={time.time()-t1:.1f}s', flush=True)
    if n_neg < best_n_neg:
        best_n_neg = n_neg; best_state = rec3.copy()
        print(f'    *** new best: n_neg={n_neg} ***', flush=True)
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% feas at cycle 3 ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_chain.npy', rec3)
        return

    # Cycle 4: tighter threshold (0.018) + recover @ 0.012.
    print('\n=== Cycle 4: M10Tet @ 0.018 + M10Tet @ 0.012 ===', flush=True)
    t0 = time.time()
    over = m10tet(best_state.copy(), 0.018)
    n_o, _, _ = report(over, '  over-tightened', phi_input)
    print(f'    wall={time.time()-t0:.1f}s', flush=True)
    t1 = time.time()
    rec4 = m10tet(over, 0.012)
    n_neg, n_below, _ = report(rec4, '  recovered', phi_input)
    print(f'    recover wall={time.time()-t1:.1f}s', flush=True)
    if n_neg < best_n_neg:
        best_n_neg = n_neg; best_state = rec4.copy()
        print(f'    *** new best: n_neg={n_neg} ***', flush=True)
    if n_neg == 0 and n_below == 0:
        print('\n*** STRICT 100% feas at cycle 4 ***', flush=True)
        np.save(OUTPUT / 'b0039_z0_15_strict_via_chain.npy', rec4)
        return

    print(f'\n=== Final ===\n  best across all cycles: n_neg={best_n_neg}',
          flush=True)
    np.save(OUTPUT / 'b0039_z0_15_chain_best.npy', best_state)
    print('  saved best state to b0039_z0_15_chain_best.npy', flush=True)


if __name__ == '__main__':
    main()
