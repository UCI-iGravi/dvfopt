"""Strategy D: Per-cell constrained NLP with SLSQP.

For each unfixable cell, set up a 24-variable, 6-constraint NLP:
  minimize ||phi_cell - phi_cell_in||^2
  subject to: V_k(phi_cell) >= threshold, k=0..5

Solve with scipy.optimize.minimize(method='SLSQP'). Apply update
to the global field. Repeat in a Gauss-Seidel sweep: after each
cell update, re-find unfixable cells and continue.

This is the last untested local-fix strategy. Unlike Strategy A
(corner averaging, which broke neighbours due to forcing 8 corners
to a single mean value), Strategy D moves the 8 corners minimally
— ideally just enough to make this cube non-collapsed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import minimize

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    _tet_volume_from_vertices,
    _voxel_corner_positions,
    six_tet_volumes_3d,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01


_DIAGONALS = [(0, 7), (1, 6), (2, 5), (3, 4)]


def _six_tets_for_diagonal(start, end):
    all_edges = [(v, w) for v in range(8) for w in range(v + 1, 8) if (v ^ w) in (1, 2, 4)]
    perimeter = [e for e in all_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def _best_min_per_cell(phi):
    V_default = six_tet_volumes_3d(phi)
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos_all = _voxel_corner_positions(dz, dy, dx)
    min_per_diag = np.empty((4, *V_default.shape[1:]))
    min_per_diag[0] = V_default.min(axis=0)
    D, H, W = dz.shape
    z0 = np.zeros_like(dz)
    pos_id = _voxel_corner_positions(z0, z0, z0)
    for di in range(1, 4):
        s, e = _DIAGONALS[di]
        tets = _six_tets_for_diagonal(s, e)
        V_d = np.empty((6, *V_default.shape[1:]))
        for k, (i0, i1, i2, i3) in enumerate(tets):
            v_id = float(
                _tet_volume_from_vertices(pos_id[i0], pos_id[i1], pos_id[i2], pos_id[i3])[0, 0, 0]
            )
            sgn = +1.0 if v_id > 0 else -1.0
            V_d[k] = sgn * _tet_volume_from_vertices(
                pos_all[i0], pos_all[i1], pos_all[i2], pos_all[i3]
            )
        min_per_diag[di] = V_d.min(axis=0)
    return min_per_diag.max(axis=0)


def _cell_corner_identity(z, y, x):
    """Identity (undeformed) positions of the 8 corners."""
    pos = np.zeros((8, 3))
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        pos[i] = (z + oz, y + oy, x + ox)
    return pos


def _cell_six_tet_volumes(disp_24, id_pos):
    """Compute 6 tet volumes given 24-dim displacement vector
    (packed [dz_0..7, dy_0..7, dx_0..7]) and identity corner positions."""
    # Deformed positions: pos[i] = id_pos[i] + (dz_i, dy_i, dx_i).
    dz = disp_24[:8]
    dy = disp_24[8:16]
    dx_ = disp_24[16:24]
    pos = id_pos.copy()
    pos[:, 0] += dz
    pos[:, 1] += dy
    pos[:, 2] += dx_
    out = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
        A, B, C, Dv = pos[i0], pos[i1], pos[i2], pos[i3]
        AB = B - A
        AC = C - A
        AD = Dv - A
        det = (
            AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
            - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
            + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
        )
        out[k] = float(_TET_SIGN[k]) * det / 6.0
    return out


def fix_cell(phi, z, y, x, threshold=THRESHOLD):
    """Solve SLSQP for the 8 corners of cell (z, y, x). Returns
    new (3, 2, 2, 2) array of phi for these 8 corners (in the same
    order as the 8 corners) or None on failure."""
    # Gather current state (24-dim vector).
    id_pos = _cell_corner_identity(z, y, x)
    cur = np.zeros(24)
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        cur[i] = float(phi[0, z + oz, y + oy, x + ox])  # dz
        cur[i + 8] = float(phi[1, z + oz, y + oy, x + ox])  # dy
        cur[i + 16] = float(phi[2, z + oz, y + oy, x + ox])  # dx

    # Objective: 0.5 * ||x - x_in||^2.
    def obj(x):
        d = x - cur
        return 0.5 * float(d @ d), d

    # Constraints: V_k(x) - threshold >= 0 for each of 6 tets.
    def con_factory(k):
        def fun(x):
            V = _cell_six_tet_volumes(x, id_pos)
            return V[k] - threshold

        return fun

    constraints = [{'type': 'ineq', 'fun': con_factory(k)} for k in range(6)]

    # SLSQP doesn't natively use the analytic objective gradient; pass
    # jac=True to obj wrapper.
    res = minimize(
        obj,
        cur,
        method='SLSQP',
        jac=True,
        constraints=constraints,
        options={'maxiter': 100, 'ftol': 1e-9, 'disp': False},
    )
    if not res.success:
        return None, res
    # Check exact feasibility post-SLSQP.
    V_new = _cell_six_tet_volumes(res.x, id_pos)
    if V_new.min() < threshold - 1e-7:
        return None, res
    return res.x, res


def strategy_d(phi, target_threshold=THRESHOLD, max_passes=8, verbose=True):
    """Gauss-Seidel sweep of per-cell SLSQP on unfixable cells."""
    phi_out = phi.astype(np.float64).copy()
    phi_orig = phi.astype(np.float64)
    for pass_idx in range(max_passes):
        # Identify cells to fix: anything with best-diag min_T <= 0
        # (genuinely fold) OR default-diag min_T < threshold (below
        # safety margin).
        V_default = six_tet_volumes_3d(phi_out)
        default_min = V_default.min(axis=0)
        target_mask = default_min < target_threshold - 1e-5
        n_target = int(target_mask.sum())
        if n_target == 0:
            if verbose:
                print(f'  pass {pass_idx}: 0 target cells — done.', flush=True)
            break
        if verbose:
            print(f'  pass {pass_idx}: {n_target} target cells', flush=True)

        # Process cells one at a time in order of increasing min_T.
        nz, ny, nx = np.where(target_mask)
        mins = default_min[nz, ny, nx]
        order = np.argsort(mins)
        n_fixed = 0
        n_failed = 0
        for k in order[: min(500, len(order))]:  # cap to top-500 per pass
            z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
            sol, res = fix_cell(phi_out, z, y, x, threshold=target_threshold)
            if sol is None:
                n_failed += 1
                continue
            # Apply update: 8 corners.
            for i in range(8):
                oz = (i >> 2) & 1
                oy = (i >> 1) & 1
                ox = i & 1
                phi_out[0, z + oz, y + oy, x + ox] = sol[i]
                phi_out[1, z + oz, y + oy, x + ox] = sol[i + 8]
                phi_out[2, z + oz, y + oy, x + ox] = sol[i + 16]
            n_fixed += 1
        V_new = six_tet_volumes_3d(phi_out)
        n_neg = int((V_new <= 0).sum())
        n_below = int((V_new < target_threshold - 1e-5).sum())
        L1 = float(np.abs(phi_out - phi_orig).sum())
        if verbose:
            print(
                f'    pass {pass_idx} done: fixed={n_fixed}  failed={n_failed}  '
                f'n_neg={n_neg}  n<thresh={n_below}  '
                f'min_T={float(V_new.min()):+.6f}  L1_from_orig={L1:.1f}',
                flush=True,
            )
        if n_neg == 0 and n_below == 0:
            if verbose:
                print('    *** STRICT FEASIBLE ***', flush=True)
            break
    return phi_out


def main():
    cache = OUTPUT / 'b0039_FULL_stage3_z000_016.npy'
    phi = np.load(cache)
    print(f'Loaded {cache}  shape={phi.shape}', flush=True)
    V = six_tet_volumes_3d(phi)
    best_min = _best_min_per_cell(phi)
    print(
        f'Start:  default n_neg={int((V <= 0).sum())}  '
        f'unfixable={int((best_min <= 0).sum())}  '
        f'min_T={float(V.min()):+.6f}',
        flush=True,
    )

    print('\n=== Strategy D: per-cell SLSQP Gauss-Seidel sweep ===', flush=True)
    t0 = time.time()
    phi_out = strategy_d(phi, target_threshold=THRESHOLD, max_passes=8)
    wall = time.time() - t0
    V_out = six_tet_volumes_3d(phi_out)
    n_neg = int((V_out <= 0).sum())
    n_below = int((V_out < THRESHOLD - 1e-5).sum())
    print(
        f'\nFinal:\n'
        f'  n_neg={n_neg}\n'
        f'  n<0.01={n_below}\n'
        f'  min_T={float(V_out.min()):+.6f}\n'
        f'  L1_from_orig={float(np.abs(phi_out - phi).sum()):.1f}\n'
        f'  wall: {wall:.1f}s\n'
        f'  STRICT 100% feasible: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_D.npy', phi_out)
        print('  Saved strict-feasible result.', flush=True)


if __name__ == '__main__':
    main()
