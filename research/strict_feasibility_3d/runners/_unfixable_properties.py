"""Deep investigation of the 94 unfixable cube properties.

Measures and reports for each unfixable cell:
  - SVD spectrum of 8 deformed corner positions (rank-deficiency)
  - Continuous det(J) statistics inside (depth and extent of fold)
  - Spatial location and clustering
  - Eigenvalues of the local Jacobian (which directions flip)
  - Shape statistics (extents along principal axes, aspect ratios)
  - Comparison to nearby non-unfixable cubes (gradient/discontinuity)

Outputs a summary table + per-property distributions + figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np

from dvfopt.jacobian.tetrahedron_sign import (
    _tet_volume_from_vertices,
    _voxel_corner_positions,
    six_tet_volumes_3d,
)
from research.strict_feasibility_3d.runners._jacobian_diagnosis import (
    _trilinear_jacobian_det,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell

OUTPUT = _HERE / 'output'


def _corner_positions_for_cell(phi, z, y, x):
    """Return (8, 3) deformed corner positions for cube (z, y, x)."""
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        pos[i, 0] = (z + iz) + phi[0, z + iz, y + iy, x + ix]
        pos[i, 1] = (y + iy) + phi[1, z + iz, y + iy, x + ix]
        pos[i, 2] = (x + ix) + phi[2, z + iz, y + iy, x + ix]
    return pos


def _jacobian_at(disp, ref, u, v, w):
    """3x3 Jacobian of trilinear interpolation at internal point (u, v, w)."""
    dP_du = np.zeros(3)
    dP_dv = np.zeros(3)
    dP_dw = np.zeros(3)
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        wz = u if iz else (1 - u)
        wy = v if iy else (1 - v)
        wx = w if ix else (1 - w)
        d_wz = (1 if iz else -1)
        d_wy = (1 if iy else -1)
        d_wx = (1 if ix else -1)
        pos_i = ref[i] + disp[i]
        dP_du += d_wz * wy * wx * pos_i
        dP_dv += wz * d_wy * wx * pos_i
        dP_dw += wz * wy * d_wx * pos_i
    return np.stack([dP_du, dP_dv, dP_dw], axis=1)


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    best_min = _best_min_per_cell(phi)
    unfix_mask = (best_min <= 0)
    nz, ny, nx = np.where(unfix_mask)
    n_unfix = len(nz)
    print(f'{n_unfix} unfixable cubes found\n', flush=True)

    # Collect per-cell properties.
    svd_specs = []        # (3,) per cell -- sorted singular values of centered 8-corner positions
    cube_extents = []     # (3,) per cell -- bbox extent in each axis
    det_J_centroid = []   # continuous det(J) at cube center
    det_J_min = []        # min continuous det(J) on 5x5x5 grid
    det_J_frac_neg = []   # fraction of 5x5x5 internal points with det(J) < 0
    jac_eigvals_min = []  # min real part of Jacobian eigenvalues at center (3 vals)
    aspect_ratio = []     # sigma_max / sigma_min
    rank_collapse = []    # 1 if sigma_3 < 0.1 (rank-2 deficient), 2 if sigma_2 < 0.1 (rank-1), 0 otherwise
    spatial_loc = []      # (z, y, x) integer cube indices

    for ci in range(n_unfix):
        z, y, x = int(nz[ci]), int(ny[ci]), int(nx[ci])
        spatial_loc.append((z, y, x))
        pos = _corner_positions_for_cell(phi, z, y, x)
        # SVD on centered positions.
        cen = pos.mean(axis=0)
        P = pos - cen
        U, S, Vt = np.linalg.svd(P, full_matrices=False)
        svd_specs.append(S.copy())
        aspect_ratio.append(float(S[0] / max(S[2], 1e-12)))
        if S[2] < 0.1:
            if S[1] < 0.1:
                rank_collapse.append(1)  # rank-1 (colinear)
            else:
                rank_collapse.append(2)  # rank-2 (coplanar)
        else:
            rank_collapse.append(0)
        # Cube extents (bbox).
        cube_extents.append(pos.max(axis=0) - pos.min(axis=0))
        # Continuous det(J) on 5x5x5 grid inside.
        ref = np.zeros((8, 3))
        disp = np.zeros((8, 3))
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            ref[i] = (z + iz, y + iy, x + ix)
            disp[i, 0] = phi[0, z + iz, y + iy, x + ix]
            disp[i, 1] = phi[1, z + iz, y + iy, x + ix]
            disp[i, 2] = phi[2, z + iz, y + iy, x + ix]
        det_grid = _trilinear_jacobian_det(disp, ref, samples_per_dim=5)
        det_J_min.append(float(det_grid.min()))
        det_J_frac_neg.append(float((det_grid < 0).sum()) / det_grid.size)
        # det(J) at center.
        J = _jacobian_at(disp, ref, 0.5, 0.5, 0.5)
        det_J_centroid.append(float(np.linalg.det(J)))
        # Eigenvalues at center.
        eigs = np.linalg.eigvals(J)
        jac_eigvals_min.append(float(min(eigs.real)))

    svd_specs = np.array(svd_specs)
    cube_extents = np.array(cube_extents)
    det_J_centroid = np.array(det_J_centroid)
    det_J_min = np.array(det_J_min)
    det_J_frac_neg = np.array(det_J_frac_neg)
    jac_eigvals_min = np.array(jac_eigvals_min)
    aspect_ratio = np.array(aspect_ratio)
    rank_collapse = np.array(rank_collapse)

    print('=== SVD Spectrum (sorted descending) of 8 deformed corners ===', flush=True)
    print(f'  largest sv:    min={svd_specs[:, 0].min():.3f}  median={float(np.median(svd_specs[:, 0])):.3f}  max={svd_specs[:, 0].max():.3f}', flush=True)
    print(f'  middle sv:     min={svd_specs[:, 1].min():.3f}  median={float(np.median(svd_specs[:, 1])):.3f}  max={svd_specs[:, 1].max():.3f}', flush=True)
    print(f'  smallest sv:   min={svd_specs[:, 2].min():.3f}  median={float(np.median(svd_specs[:, 2])):.3f}  max={svd_specs[:, 2].max():.3f}', flush=True)
    print(f'  aspect ratio (sigmamax/sigmamin): min={aspect_ratio.min():.2f}  median={float(np.median(aspect_ratio)):.2f}  max={aspect_ratio.max():.2f}', flush=True)
    print('  rank collapse classification:', flush=True)
    print(f'    rank-1 (nearly colinear):  {int((rank_collapse == 1).sum())}', flush=True)
    print(f'    rank-2 (nearly coplanar):  {int((rank_collapse == 2).sum())}', flush=True)
    print(f'    full rank (genuine 3D):    {int((rank_collapse == 0).sum())}', flush=True)

    print('\n=== Continuous Jacobian inside ===', flush=True)
    print(f'  det(J) at center:  min={det_J_centroid.min():.3f}  median={float(np.median(det_J_centroid)):.3f}  max={det_J_centroid.max():.3f}', flush=True)
    print(f'  det(J) min (5x5x5 grid):  min={det_J_min.min():.3f}  median={float(np.median(det_J_min)):.3f}  max={det_J_min.max():.3f}', flush=True)
    print('  fraction of internal grid with det(J)<0:', flush=True)
    print(f'    min={det_J_frac_neg.min():.2%}  median={float(np.median(det_J_frac_neg)):.2%}  max={det_J_frac_neg.max():.2%}', flush=True)

    print('\n=== Jacobian eigenvalues at center ===', flush=True)
    print(f'  smallest real eigenvalue:  min={jac_eigvals_min.min():.3f}  median={float(np.median(jac_eigvals_min)):.3f}  max={jac_eigvals_min.max():.3f}', flush=True)

    print('\n=== Spatial distribution ===', flush=True)
    z_arr = np.array([s[0] for s in spatial_loc])
    y_arr = np.array([s[1] for s in spatial_loc])
    x_arr = np.array([s[2] for s in spatial_loc])
    print(f'  z range: [{z_arr.min()}, {z_arr.max()}], unique: {len(np.unique(z_arr))}', flush=True)
    print(f'  y range: [{y_arr.min()}, {y_arr.max()}], unique: {len(np.unique(y_arr))}', flush=True)
    print(f'  x range: [{x_arr.min()}, {x_arr.max()}], unique: {len(np.unique(x_arr))}', flush=True)
    print(f'  centroid:  ({float(z_arr.mean()):.1f}, {float(y_arr.mean()):.1f}, {float(x_arr.mean()):.1f})', flush=True)

    # Comparison: nearby non-unfixable cells.
    print('\n=== Comparison: unfixable cells vs random non-unfixable cells ===', flush=True)
    # Random sample of 200 non-unfixable cells.
    fixable_mask = ~unfix_mask
    fz, fy, fx = np.where(fixable_mask)
    sel = np.random.default_rng(0).choice(len(fz), size=200, replace=False)
    fix_svd_min = []
    fix_det_min = []
    for s in sel:
        z, y, x = int(fz[s]), int(fy[s]), int(fx[s])
        pos = _corner_positions_for_cell(phi, z, y, x)
        cen = pos.mean(axis=0)
        P = pos - cen
        S = np.linalg.svd(P, full_matrices=False, compute_uv=False)
        fix_svd_min.append(float(S[2]))
        ref = np.zeros((8, 3))
        disp = np.zeros((8, 3))
        for i in range(8):
            iz = (i >> 2) & 1
            iy = (i >> 1) & 1
            ix = i & 1
            ref[i] = (z + iz, y + iy, x + ix)
            disp[i, 0] = phi[0, z + iz, y + iy, x + ix]
            disp[i, 1] = phi[1, z + iz, y + iy, x + ix]
            disp[i, 2] = phi[2, z + iz, y + iy, x + ix]
        det_grid = _trilinear_jacobian_det(disp, ref, samples_per_dim=3)
        fix_det_min.append(float(det_grid.min()))
    fix_svd_min = np.array(fix_svd_min)
    fix_det_min = np.array(fix_det_min)
    print(f'  Unfixable: sigma_min median={float(np.median(svd_specs[:, 2])):.3f}  det(J)_min median={float(np.median(det_J_min)):.3f}', flush=True)
    print(f'  Fixable:   sigma_min median={float(np.median(fix_svd_min)):.3f}  det(J)_min median={float(np.median(fix_det_min)):.3f}', flush=True)

    # Histogram figures.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Properties of 94 unfixable cubes (B0039 z=0..15 dense band)', fontsize=12)
    axes[0, 0].hist(svd_specs[:, 0], bins=30, color='C0', alpha=0.7); axes[0, 0].set_title('Largest singular value (sigma_1)')
    axes[0, 1].hist(svd_specs[:, 1], bins=30, color='C1', alpha=0.7); axes[0, 1].set_title('Middle singular value (sigma_2)')
    axes[0, 2].hist(svd_specs[:, 2], bins=30, color='C2', alpha=0.7); axes[0, 2].set_title('Smallest singular value (sigma_3)')
    axes[1, 0].hist(det_J_min, bins=30, color='C3', alpha=0.7); axes[1, 0].set_title('min det(J) inside cube (continuous)')
    axes[1, 0].axvline(0, color='k', linewidth=0.5)
    axes[1, 1].hist(det_J_frac_neg, bins=30, color='C4', alpha=0.7); axes[1, 1].set_title('Fraction of cube interior with det(J)<0')
    axes[1, 2].hist(aspect_ratio, bins=30, color='C5', alpha=0.7); axes[1, 2].set_title('Aspect ratio (sigma_1/sigma_3)')
    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT / 'unfixable_properties.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\nSaved figure: {out}', flush=True)


if __name__ == '__main__':
    main()
