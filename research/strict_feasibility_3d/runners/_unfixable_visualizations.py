"""Comprehensive visualizations of the 94 unfixable cubes:

1. Spatial distribution: 3D scatter + 2D projections showing where
   the unfixable cells are in the z=0..15 chunk.
2. SVD-vs-detJ scatter: singular value vs continuous Jacobian
   negativity — see if there's a structural relationship.
3. Per-cell Jacobian distribution: sample 5x5x5 internal points
   per cell, plot the spectrum of det(J) values.
4. Comparison: unfixable vs fixable spectra side-by-side.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._jacobian_diagnosis import (
    _trilinear_jacobian_det,
)
from research.strict_feasibility_3d.runners._uncrush_v2 import _best_min_per_cell
from research.strict_feasibility_3d.runners._unfixable_properties import (
    _corner_positions_for_cell,
)

OUTPUT = _HERE / 'output'


def main():
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    best_min = _best_min_per_cell(phi)
    unfix_mask = best_min <= 0
    nz, ny, nx = np.where(unfix_mask)
    n_unfix = len(nz)
    print(f'{n_unfix} unfixable cubes', flush=True)

    # ===== Spatial distribution figure =====
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Spatial distribution of {n_unfix} unfixable cubes in B0039 z=0..15\n'
        f'(all 94 cells lie in one localized fold region)',
        fontsize=12,
    )
    # 3D scatter.
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.scatter(nx, ny, nz, c=nz, cmap='viridis', s=40, edgecolors='k', linewidth=0.4)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_title('3D scatter (colored by z)')

    # xy projection with z as color.
    ax_xy = fig.add_subplot(2, 3, 2)
    sc_xy = ax_xy.scatter(nx, ny, c=nz, cmap='viridis', s=60, edgecolors='k')
    ax_xy.set_xlabel('x cell index')
    ax_xy.set_ylabel('y cell index')
    ax_xy.set_title(f'(x, y) projection, colored by z\n{n_unfix} cubes')
    ax_xy.set_aspect('equal')
    plt.colorbar(sc_xy, ax=ax_xy, label='z')
    ax_xy.invert_yaxis()
    ax_xy.grid(True, alpha=0.3)

    # xz projection.
    ax_xz = fig.add_subplot(2, 3, 3)
    sc_xz = ax_xz.scatter(nx, nz, c=ny, cmap='plasma', s=60, edgecolors='k')
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_xz.set_title('(x, z) projection, colored by y')
    plt.colorbar(sc_xz, ax=ax_xz, label='y')
    ax_xz.grid(True, alpha=0.3)

    # Cells per z-layer histogram.
    ax_zhist = fig.add_subplot(2, 3, 4)
    z_counts = np.bincount(nz, minlength=16)
    ax_zhist.bar(range(16), z_counts, color='C0', alpha=0.7)
    ax_zhist.set_xlabel('z cell index')
    ax_zhist.set_ylabel('# unfixable cubes')
    ax_zhist.set_title('Unfixable cubes per z-layer')
    ax_zhist.set_xticks(range(0, 16, 2))
    ax_zhist.grid(True, alpha=0.3)

    # SVD-vs-detJmin scatter.
    print('Computing SVD + det(J) per cube...', flush=True)
    svd_min = []
    detJ_min = []
    rank_classification = []
    for ci in range(n_unfix):
        z, y, x = int(nz[ci]), int(ny[ci]), int(nx[ci])
        pos = _corner_positions_for_cell(phi, z, y, x)
        S = np.linalg.svd(pos - pos.mean(axis=0), full_matrices=False, compute_uv=False)
        svd_min.append(float(S[2]))
        if S[2] < 0.1:
            rank_classification.append('rank-2' if S[1] >= 0.1 else 'rank-1')
        else:
            rank_classification.append('full rank')
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
        detJ_min.append(float(det_grid.min()))
    svd_min = np.array(svd_min)
    detJ_min = np.array(detJ_min)
    rank_classification = np.array(rank_classification)

    ax_sv = fig.add_subplot(2, 3, 5)
    colors = {'rank-1': 'red', 'rank-2': 'orange', 'full rank': 'C0'}
    for tag, col in colors.items():
        mask = rank_classification == tag
        if mask.any():
            ax_sv.scatter(
                svd_min[mask],
                detJ_min[mask],
                c=col,
                s=40,
                edgecolors='k',
                linewidth=0.4,
                alpha=0.7,
                label=f'{tag} ({int(mask.sum())})',
            )
    ax_sv.set_xlabel('smallest singular value (sigma_3)')
    ax_sv.set_ylabel('min det(J) inside cube')
    ax_sv.set_title('SVD-vs-Jacobian structural map')
    ax_sv.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_sv.set_yscale('symlog', linthresh=0.01)
    ax_sv.legend(loc='lower right', fontsize=8)
    ax_sv.grid(True, alpha=0.3)

    # Comparison: unfixable vs random fixable cells.
    print('Computing random fixable cell stats...', flush=True)
    fixable_mask = ~unfix_mask
    fz, fy, fx = np.where(fixable_mask)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(fz), size=200, replace=False)
    fix_svd_min = []
    fix_detJ_min = []
    for s in sel:
        z, y, x = int(fz[s]), int(fy[s]), int(fx[s])
        pos = _corner_positions_for_cell(phi, z, y, x)
        S = np.linalg.svd(pos - pos.mean(axis=0), full_matrices=False, compute_uv=False)
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
        fix_detJ_min.append(float(det_grid.min()))
    fix_svd_min = np.array(fix_svd_min)
    fix_detJ_min = np.array(fix_detJ_min)

    ax_cmp = fig.add_subplot(2, 3, 6)
    ax_cmp.scatter(
        fix_svd_min,
        fix_detJ_min,
        c='lightgray',
        s=20,
        alpha=0.5,
        label=f'random fixable ({len(fix_svd_min)})',
    )
    ax_cmp.scatter(svd_min, detJ_min, c='red', s=30, alpha=0.7, label=f'unfixable ({len(svd_min)})')
    ax_cmp.set_xlabel('smallest singular value (sigma_3)')
    ax_cmp.set_ylabel('min det(J) inside cube')
    ax_cmp.set_title('Unfixable vs fixable cells')
    ax_cmp.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_cmp.set_yscale('symlog', linthresh=0.01)
    ax_cmp.legend(loc='lower right', fontsize=8)
    ax_cmp.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT / 'unfixable_visualizations.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved figure: {out}', flush=True)


if __name__ == '__main__':
    main()
