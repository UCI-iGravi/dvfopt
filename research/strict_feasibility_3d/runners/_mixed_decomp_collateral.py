"""Follow-up to _mixed_decomp_floor33.py: collateral-damage check.

The per-cell probes show each of the 14 floor cells becomes feasible with
tiny (0.007-0.11 px) in-plane corner moves. But cube corners are shared
by up to 8 cells. This script APPLIES each cell's minimal all-corner
in-plane displacement to a padded crop copy and counts the folds it
creates in the surrounding cells — quantifying whether the fixes are
locally compatible or mutually destructive (the coupled floor).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE))

from _mixed_decomp_floor33 import (
    CORNER_OFFSETS,
    DEFAULT_INPUT,
    FALLBACK_INPUT,
    cell_corner_positions,
    probe_min_displacement,
    tet_volumes_from_positions,
)

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_all_diagonals


def crop_fold_count(vol, cell, rad=2):
    """(n_cells_nonpos_fixed, n_cells_nonpos_best) in the (2*rad+1)^3 cell
    block centred on `cell` (clipped to the volume)."""
    cz, cy, cx = cell
    D, H, W = vol.shape[1:]
    z0, z1 = max(cz - rad, 0), min(cz + rad + 1, D - 1)
    y0, y1 = max(cy - rad, 0), min(cy + rad + 1, H - 1)
    x0, x1 = max(cx - rad, 0), min(cx + rad + 1, W - 1)
    sub = np.ascontiguousarray(vol[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1])
    ad = six_tet_volumes_all_diagonals(sub)
    return int((ad[0] <= 0).sum()), int((ad.max(axis=0) <= 0).sum())


def main():
    inp = DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT
    vol = np.load(inp)
    data = json.loads((HERE / 'output' / 'mixed_decomp_floor33.json').read_text())
    floor_cells = [tuple(c['cell']) for c in data['cells'] if not c['fixable']]
    best_diag = {tuple(c['cell']): int(c['best_diag']) for c in data['cells']}
    print(f'{len(floor_cells)} floor cells', flush=True)

    print('| cell (z,y,x) | best d | move (px, all-corner in-plane) | '
          'crop folds fixed-diag before -> after | crop folds best-diag '
          'before -> after |')
    print('|---|---|---:|---|---|')
    for cell in floor_cells:
        d = best_diag[cell]
        P0 = cell_corner_positions(vol, *cell)
        r = probe_min_displacement(P0, d, list(range(8)), in_plane=True)
        assert r is not None
        # rebuild the per-dof delta by re-solving (probe returns norms only),
        # so redo the solve here inline to get the raw delta:
        from scipy.optimize import minimize
        dof = [(v, c) for v in range(8) for c in (1, 2)]

        def positions(delta):
            P = P0.copy()
            for k, (v, c) in enumerate(dof):
                P[v, c] += delta[k]
            return P

        def cons(delta):
            V = tet_volumes_from_positions(positions(delta))
            return V[d] - 1e-3

        starts = [np.zeros(len(dof))]
        ident = np.zeros(len(dof))
        for k, (v, c) in enumerate(dof):
            ident[k] = (P0[0, c] + CORNER_OFFSETS[v][c]) - P0[v, c]
        starts.append(ident)
        best = None
        for x0 in starts:
            res = minimize(lambda x: float((x ** 2).sum()), x0, method='SLSQP',
                           constraints=[dict(type='ineq', fun=cons)],
                           options=dict(maxiter=300, ftol=1e-12))
            if res.success and tet_volumes_from_positions(
                    positions(res.x))[d].min() >= 1e-3 - 1e-9:
                if best is None or (res.x ** 2).sum() < (best.x ** 2).sum():
                    best = res
        assert best is not None
        delta = best.x

        before_fix, before_best = crop_fold_count(vol, cell)
        # apply delta to a working copy of the crop region (corner moves
        # touch grid nodes cell..cell+1 in each axis)
        vol2 = vol.copy()  # full copy is ~1.7 GB but simplest & safe
        cz, cy, cx = cell
        for k, (v, c) in enumerate(dof):
            if delta[k] == 0.0:
                continue
            oz, oy, ox = CORNER_OFFSETS[v]
            vol2[c, cz + oz, cy + oy, cx + ox] += delta[k]
        after_fix, after_best = crop_fold_count(vol2, cell)
        del vol2
        mv = r['max_vertex_move']
        print(f'| {cell} | d{d} | {mv:.4f} | {before_fix} -> {after_fix} '
              f'| {before_best} -> {after_best} |', flush=True)


if __name__ == '__main__':
    main()
