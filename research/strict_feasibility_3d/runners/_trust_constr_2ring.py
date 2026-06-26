"""Trust-constr cluster NLP with 2-ring dilation (no freeze).

Iterations=2 dilation: target + 2 layers of neighbour cubes all
constrained. No explicit freezing — the outermost cubes' 6-tet
constraints implicitly limit how much their corners can move (they
need to stay feasible).

Hypothesis: the wider ring acts as a natural buffer. The outermost
ring layer absorbs perturbations from the inner layers, and since
it has plenty of feasibility margin (positive V_k initially), it
can flex without breaking.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as cc_label
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_volumes_3d,
)
from research.strict_feasibility_3d.runners._trust_constr_cluster import (
    _cube_six_tet_signed,
    solve_cluster_nlp,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    D, H, W = phi.shape[1:]
    cube_shape = (D - 1, H - 1, W - 1)
    print(f'Loaded, shape={phi.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi)
    best_min0 = _best_min_per_cell(phi)
    print(
        f'Start: n_neg={int((V0 <= 0).sum())}  '
        f'unfixable={int((best_min0 <= 0).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    unfix_mask = (best_min0 <= 0)
    # Group unfix cells by 1-cell connectivity.
    grid = binary_dilation(unfix_mask, iterations=1)
    labels, n_comp = cc_label(grid)
    clusters = []
    for cid in range(1, n_comp + 1):
        in_comp = (labels == cid) & unfix_mask
        cz, cy, cx = np.where(in_comp)
        cells = list(zip(cz.tolist(), cy.tolist(), cx.tolist()))
        if cells:
            clusters.append(cells)
    clusters.sort(key=lambda c: -len(c))
    print(f'{len(clusters)} clusters, sizes: {[len(c) for c in clusters[:10]]}', flush=True)

    phi_new = phi.astype(np.float64).copy()
    total_L1 = 0.0
    for i, target_cells in enumerate(clusters):
        # Build 2-ring (target + 2 dilation steps).
        target_mask = np.zeros(cube_shape, dtype=bool)
        for (z, y, x) in target_cells:
            target_mask[z, y, x] = True
        ring2_mask = binary_dilation(target_mask, iterations=2)
        ring2_cells = list(zip(*np.where(ring2_mask)))
        ring2_cells = [(int(z), int(y), int(x)) for z, y, x in ring2_cells]
        print(
            f'\n--- Cluster {i+1}/{len(clusters)}: {len(target_cells)} target + '
            f'{len(ring2_cells) - len(target_cells)} 2-ring ---',
            flush=True,
        )
        phi_new, info = solve_cluster_nlp(
            phi_new, ring2_cells, threshold=THRESHOLD,
            max_iter=1000, verbose=True,
        )
        total_L1 += info['L1_added']

    V_final = six_tet_volumes_3d(phi_new)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1_final = float(np.abs(phi_new - phi).sum())
    print(
        f'\n=== Final ===\n'
        f'  total intra-cluster L1: {total_L1:.1f}\n'
        f'  global n_neg: {n_neg}\n'
        f'  global n<0.01: {n_below}\n'
        f'  global min_T: {float(V_final.min()):+.6f}\n'
        f'  global L1 from input: {L1_final:.1f}\n'
        f'  STRICT 100% feas: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_trust_constr_2ring.npy', phi_new)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
