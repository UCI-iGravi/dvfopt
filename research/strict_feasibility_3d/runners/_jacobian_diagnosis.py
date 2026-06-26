"""Critical diagnostic: are the 94 "unfixable" cells actually
continuously folded, or just discretely pathological?

For each unfixable cube cell, sample the trilinear-interpolated
deformation field at an internal 5x5x5 grid and compute the
continuous Jacobian det(J) at each sample. Two outcomes:

  - det(J) > 0 everywhere inside  →  the cube is CONTINUOUSLY
    fold-free. The 6-tet check is too coarse; this is a
    discretization artifact, fixable by subdivision.

  - det(J) < 0 at some point  →  the cube has a genuine
    self-intersection. Registration ambiguity zone; no
    discretization can fix it without modifying input.

If most of the 94 cells are case 1, we have a clear path to
strict feasibility via local upsampling.
"""

from __future__ import annotations

import sys
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


def _trilinear_jacobian_det(disp_corners, ref_corners, samples_per_dim=5):
    """Sample det(J) inside a single cube on a samples_per_dim^3 grid.

    The cube's 8 corners have displacements `disp_corners[i]` = (dz, dy, dx)
    and reference positions `ref_corners[i]` = (z, y, x). Trilinear basis
    functions:
        N_i(u, v, w) = w_z(u, i_z) * w_y(v, i_y) * w_x(w, i_x)
        where w_b(t, b) = (1-t) if b=0 else t
        and i_z = (i>>2)&1, i_y = (i>>1)&1, i_x = i&1.
    The interpolated position is:
        P(u, v, w) = sum_i N_i * (ref_corners[i] + disp_corners[i])
    The Jacobian at (u, v, w) is the 3x3 matrix of partials.

    Returns array of shape (samples_per_dim, samples_per_dim, samples_per_dim)
    with det(J) at each sample.
    """
    t = np.linspace(0.05, 0.95, samples_per_dim)  # avoid corner singularities
    u, v, w = np.meshgrid(t, t, t, indexing='ij')  # each (S, S, S)
    n_samples = u.size
    det_grid = np.empty_like(u)
    # Combined positions (ref + disp).
    pos_corners = ref_corners + disp_corners  # (8, 3)
    # Precompute derivatives of basis functions w.r.t. (u, v, w).
    # For corner i with (i_z, i_y, i_x), N = w_z*w_y*w_x where
    # w_b(t, 0) = 1-t, w_b(t, 1) = t.
    # ∂N/∂u = (dw_z/du)*w_y*w_x where dw_z/du = (2*i_z - 1)  (constant)
    for s in range(n_samples):
        ui, vi, wi = u.flat[s], v.flat[s], w.flat[s]
        # Position derivative w.r.t. u, v, w.
        dP_du = np.zeros(3)
        dP_dv = np.zeros(3)
        dP_dw = np.zeros(3)
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            wz = ui if iz else (1 - ui)
            wy = vi if iy else (1 - vi)
            wx = wi if ix else (1 - wi)
            d_wz = 1 if iz else -1
            d_wy = 1 if iy else -1
            d_wx = 1 if ix else -1
            # Note: u corresponds to z direction.
            dP_du += d_wz * wy * wx * pos_corners[i]
            dP_dv += wz * d_wy * wx * pos_corners[i]
            dP_dw += wz * wy * d_wx * pos_corners[i]
        # J = [dP_du, dP_dv, dP_dw] as columns.
        J = np.stack([dP_du, dP_dv, dP_dw], axis=1)  # (3, 3)
        det_grid.flat[s] = float(np.linalg.det(J))
    return det_grid


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    best_min = _best_min_per_cell(phi)
    unfix_mask = best_min <= 0
    nz, ny, nx = np.where(unfix_mask)
    print(f'Loaded; {len(nz)} unfixable cells found.', flush=True)

    n_continuously_folded = 0  # det(J) < 0 somewhere inside
    n_continuously_feasible = 0  # det(J) > 0 everywhere inside
    n_near_singular = 0  # det(J) is barely positive
    min_det_per_cell = []
    for k in range(len(nz)):
        z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
        # Build 8 corners' ref + disp.
        ref_corners = np.zeros((8, 3))
        disp_corners = np.zeros((8, 3))
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            ref_corners[i] = (z + iz, y + iy, x + ix)
            disp_corners[i, 0] = phi[0, z + iz, y + iy, x + ix]
            disp_corners[i, 1] = phi[1, z + iz, y + iy, x + ix]
            disp_corners[i, 2] = phi[2, z + iz, y + iy, x + ix]
        det_grid = _trilinear_jacobian_det(disp_corners, ref_corners, samples_per_dim=5)
        min_det = float(det_grid.min())
        min_det_per_cell.append(min_det)
        if min_det <= 0:
            n_continuously_folded += 1
        elif min_det < THRESHOLD:
            n_near_singular += 1
        else:
            n_continuously_feasible += 1

    arr = np.array(min_det_per_cell)
    print(
        f'\nResults across all {len(nz)} unfixable cells:\n'
        f'  continuously fold-free everywhere (det(J) > 0 always):   {n_continuously_feasible} ({n_continuously_feasible / len(nz) * 100:.1f}%)\n'
        f'  near-singular (0 < det(J) < threshold somewhere):         {n_near_singular} ({n_near_singular / len(nz) * 100:.1f}%)\n'
        f'  continuously folded (det(J) < 0 somewhere):               {n_continuously_folded} ({n_continuously_folded / len(nz) * 100:.1f}%)\n',
        flush=True,
    )
    print(
        f'  min(det(J)) statistics: min={arr.min():.6f}  max={arr.max():.6f}  '
        f'mean={arr.mean():.6f}  median={float(np.median(arr)):.6f}',
        flush=True,
    )

    # Histogram bins.
    print('\n  Distribution:', flush=True)
    for lo, hi in [
        (-np.inf, -0.1),
        (-0.1, -0.01),
        (-0.01, 0),
        (0, 0.01),
        (0.01, 0.1),
        (0.1, np.inf),
    ]:
        n = int(((arr > lo) & (arr <= hi)).sum())
        print(f'    {lo:+.3f} < min(det(J)) <= {hi:+.3f}:  {n}', flush=True)


if __name__ == '__main__':
    main()
