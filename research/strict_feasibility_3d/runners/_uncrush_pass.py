"""Targeted geometric uncrush of continuously-folded cubes.

For each unfixable cell (continuous det(J) < 0 somewhere inside):
  1. Compute the 8 deformed corner positions
  2. Center them and find principal axes via SVD
  3. The smallest singular value σ_3 measures how "crushed" the
     cube is along its narrowest direction
  4. Push each corner along the σ_3 direction by an amount
     proportional to its identity offset along that axis, scaling
     up the cube extent in that direction

After the uncrush pass, run a final M10Tet polish to clean up
neighbour perturbations.

Committing to pay L1 cost upfront — unlike Strategy A/D which try
to minimize L1 and plateau, this commits to the perturbation
needed to make det(J) positive.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import (
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


def _identity_offsets():
    """8 corners' identity offsets (relative to cell origin)."""
    return np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], dtype=np.float64)


def _cube_corners_deformed(phi, z, y, x):
    """Return (8, 3) array of deformed (z, y, x) positions for cell (z, y, x)."""
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        pos[i, 0] = (z + iz) + phi[0, z + iz, y + iy, x + ix]
        pos[i, 1] = (y + iy) + phi[1, z + iz, y + iy, x + ix]
        pos[i, 2] = (x + ix) + phi[2, z + iz, y + iy, x + ix]
    return pos


def _cube_corners_disp(phi, z, y, x):
    """Return (8, 3) displacement (dz, dy, dx) for the 8 corners."""
    out = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        out[i, 0] = phi[0, z + iz, y + iy, x + ix]
        out[i, 1] = phi[1, z + iz, y + iy, x + ix]
        out[i, 2] = phi[2, z + iz, y + iy, x + ix]
    return out


def uncrush_cell(phi, z, y, x, target_extent=1.2):
    """Modify 8 corners of cell (z, y, x) to uncrush the cube.

    Strategy: compute SVD of corner positions (8, 3) → cube's
    principal axes. Identify the smallest singular value σ_3 and
    its direction v_3. Apply per-corner displacement along v_3:

        delta_corner[i] = sign(<id_off[i] - centroid_id, v_3>) *
                          (target_extent - σ_3) / 2 * v_3

    This spreads the corners along v_3 to achieve `target_extent`
    extent in that direction. Returns new (8, 3) displacements.
    """
    pos = _cube_corners_deformed(phi, z, y, x)
    centroid = pos.mean(axis=0)
    P = pos - centroid  # (8, 3) centered
    # SVD: U (8x8) S (3,) V (3x3).
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    # S has 3 singular values. The smallest is the cube's narrowest axis.
    sigma_min = float(S[-1])
    v_min = Vt[-1, :]  # (3,) direction of smallest sv
    # Identity offsets (centered on identity centroid 0.5).
    id_off = _identity_offsets()  # (8, 3) values in {0, 1}
    id_centroid = id_off.mean(axis=0)  # (0.5, 0.5, 0.5)
    id_off_centered = id_off - id_centroid  # values in {-0.5, +0.5}
    # Project identity offsets onto v_min to get sign per corner.
    proj = id_off_centered @ v_min  # (8,) values around -0.866 to +0.866
    # Target extent: each corner moves to +/- (target_extent/2) along v_min.
    # Currently extent = sigma_min along v_min. Need to add
    # (target_extent - sigma_min) / max(extent_per_corner) to push.
    add = (target_extent - sigma_min) / 2.0
    if add <= 0:
        # Already non-crushed in this direction; no-op.
        return _cube_corners_disp(phi, z, y, x), 0.0
    # Per-corner delta: (sign-of-projection) * add * v_min.
    delta_pos = np.outer(np.sign(proj), v_min) * add  # (8, 3)
    # The new deformed position is pos + delta_pos. The new
    # displacement is (new_pos - identity_pos) = (pos + delta_pos) - id_pos.
    # Since pos = id_pos + disp, new_disp = disp + delta_pos.
    cur_disp = _cube_corners_disp(phi, z, y, x)
    new_disp = cur_disp + delta_pos
    l1_cost = float(np.abs(delta_pos).sum())
    return new_disp, l1_cost


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'Loaded, shape={phi.shape}', flush=True)
    V0 = six_tet_volumes_3d(phi)
    best_min0 = _best_min_per_cell(phi)
    print(
        f'Start:  n_neg={int((V0 <= 0).sum())}  '
        f'n<0.01={int((V0 < THRESHOLD - 1e-5).sum())}  '
        f'unfixable={int((best_min0 <= 0).sum())}  '
        f'min_T={float(V0.min()):+.6f}',
        flush=True,
    )

    # Uncrush all 94 unfixable cells in one pass.
    print('\n=== Uncrush pass: target_extent=1.2 ===', flush=True)
    nz, ny, nx = np.where(best_min0 <= 0)
    phi_new = phi.astype(np.float64).copy()
    total_l1 = 0.0
    n_processed = 0
    n_skipped = 0
    # Accumulate updates and apply at end (to avoid mid-pass conflicts).
    update_sum = np.zeros_like(phi_new)
    update_count = np.zeros(phi_new.shape[1:], dtype=np.int32)
    for k in range(len(nz)):
        z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
        new_disp, cost = uncrush_cell(phi_new, z, y, x, target_extent=1.2)
        if cost == 0:
            n_skipped += 1
            continue
        total_l1 += cost
        n_processed += 1
        # Scatter-add the new displacements to the accumulator.
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            update_sum[:, z + iz, y + iy, x + ix] += new_disp[i]
            update_count[z + iz, y + iy, x + ix] += 1
    # Apply averaged update where count > 0.
    mask = update_count > 0
    avg_update = np.zeros((3, *update_count.shape))
    for c in range(3):
        avg_update[c][mask] = update_sum[c][mask] / update_count[mask]
    phi_new[:, mask] = avg_update[:, mask]
    print(
        f'  processed: {n_processed}  skipped (no-op): {n_skipped}  '
        f'total ideal L1 cost: {total_l1:.1f}',
        flush=True,
    )
    V_after = six_tet_volumes_3d(phi_new)
    L1 = float(np.abs(phi_new - phi).sum())
    print(
        f'\nAfter uncrush:  n_neg={int((V_after <= 0).sum())}  '
        f'n<0.01={int((V_after < THRESHOLD - 1e-5).sum())}  '
        f'min_T={float(V_after.min()):+.6f}  L1_from_orig={L1:.1f}',
        flush=True,
    )

    # Polish with M10Tet @ threshold=0.015 to fix neighbour breakage.
    print('\n=== M10Tet polish @ threshold=0.015 ===', flush=True)
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    t0 = time.time()
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi_new.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=0.015,
    )
    phi_polished = solver.fit(phi_new).corrected
    wall = time.time() - t0
    V_final = six_tet_volumes_3d(phi_polished)
    n_neg = int((V_final <= 0).sum())
    n_below = int((V_final < THRESHOLD - 1e-5).sum())
    L1_final = float(np.abs(phi_polished - phi).sum())
    print(
        f'\nFinal:  n_neg={n_neg}  n<0.01={n_below}  '
        f'min_T={float(V_final.min()):+.6f}  L1_from_orig={L1_final:.1f}  '
        f'polish_wall={wall:.1f}s\n'
        f'  STRICT 100% feasible: {n_neg == 0 and n_below == 0}',
        flush=True,
    )
    if n_neg == 0 and n_below == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_uncrush.npy', phi_polished)
        print('  *** Saved strict-feasible result. ***', flush=True)


if __name__ == '__main__':
    main()
