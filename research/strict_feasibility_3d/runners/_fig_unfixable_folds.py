"""Generate a figure showing the geometric structure of the
"unfixable" fold cells — cubes whose 8 deformed corners are arranged
such that NO tetrahedralization yields all-positive signed volumes.

The figure helps explain why M10Tet (and any first-order method)
plateaus at a non-zero residual on dense B0039 bands.

Pipeline:
  1. Load the 173-fold checkpoint
  2. Compute 6-tet volumes for all 4 cube-diagonal choices
  3. Identify the cells that fold under every choice (~94)
  4. Pick a representative sample of these "geometrically
     unavoidable" cells
  5. Render each as a 3D wire-frame of the deformed cube, with
     edges colored by whether their incident tets are folded
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_VERTICES,
    _TET_SIGN,
    _tet_volume_from_vertices,
    _voxel_corner_positions,
    six_tet_volumes_3d,
)


OUTPUT = _HERE / 'output'


# 4 main-diagonal choices for the cube.
_DIAGONALS = [(0, 7), (1, 6), (2, 5), (3, 4)]


def _six_tets_for_diagonal(start, end):
    all_edges = [(v, w) for v in range(8) for w in range(v + 1, 8)
                 if (v ^ w) in (1, 2, 4)]
    perimeter = [e for e in all_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def _cube_corner_positions(phi, z, y, x):
    """Return (8, 3) array of deformed (z, y, x) positions of the
    8 cube corners at cell (z, y, x)."""
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos = np.zeros((8, 3))
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        cz, cy, cx = z + oz, y + oy, x + ox
        pos[i] = ((z + oz) + dz[cz, cy, cx],
                  (y + oy) + dy[cz, cy, cx],
                  (x + ox) + dx[cz, cy, cx])
    return pos


def _identity_corners(z, y, x):
    """Reference (undeformed) corner positions."""
    pos = np.zeros((8, 3))
    for i in range(8):
        oz = (i >> 2) & 1
        oy = (i >> 1) & 1
        ox = i & 1
        pos[i] = (z + oz, y + oy, x + ox)
    return pos


# Cube edges (pairs of corners differing in 1 bit).
CUBE_EDGES = [(v, w) for v in range(8) for w in range(v + 1, 8)
              if (v ^ w) in (1, 2, 4)]


def main():
    print('Loading 173-fold checkpoint...', flush=True)
    phi = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy')
    print(f'  shape={phi.shape}', flush=True)

    V_default = six_tet_volumes_3d(phi)

    # Compute min_T under all 4 diagonals.
    print('Computing min_T per diagonal...', flush=True)
    min_per_diag = np.empty((4, *V_default.shape[1:]))
    min_per_diag[0] = V_default.min(axis=0)
    dz, dy, dx = phi[0], phi[1], phi[2]
    pos_all = _voxel_corner_positions(dz, dy, dx)
    for di in range(1, 4):
        s, e = _DIAGONALS[di]
        tets = _six_tets_for_diagonal(s, e)
        # Determine sign on identity field.
        D, H, W = dz.shape
        z0 = np.zeros_like(dz)
        pos_id = _voxel_corner_positions(z0, z0, z0)
        V_d = np.empty((6, *V_default.shape[1:]))
        for k, (i0, i1, i2, i3) in enumerate(tets):
            v_id = float(_tet_volume_from_vertices(pos_id[i0], pos_id[i1],
                                                    pos_id[i2], pos_id[i3])[0, 0, 0])
            sgn = +1.0 if v_id > 0 else -1.0
            V_d[k] = sgn * _tet_volume_from_vertices(pos_all[i0], pos_all[i1],
                                                      pos_all[i2], pos_all[i3])
        min_per_diag[di] = V_d.min(axis=0)

    best_min = min_per_diag.max(axis=0)
    # Cells unfixable under any diagonal:
    unfixable_mask = (best_min <= 0)
    n_unfix = int(unfixable_mask.sum())
    print(f'  Unfixable cells (under any of 4 diagonals): {n_unfix}', flush=True)

    # Pick 12 representative cells — sort by depth of best diagonal,
    # take the worst 12 (most pathological).
    nz, ny, nx = np.where(unfixable_mask)
    severity = best_min[nz, ny, nx]
    order = np.argsort(severity)  # most negative first
    pick = order[:12]

    print('Picking 12 worst unfixable cells:', flush=True)
    for k in pick[:12]:
        z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
        print(f'  cell (z={z}, y={y}, x={x})  best min_T = {severity[k]:+.6f}', flush=True)

    # Render a 3×4 grid of 3D wireframes.
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'Geometrically unavoidable 3D folds on B0039 z=0..15 dense band\n'
        f'(12 of {n_unfix} cells whose 8 deformed cube-corners admit no\n'
        f' tetrahedralization with all-positive signed volumes)',
        fontsize=12, y=0.98,
    )
    for plot_idx, k in enumerate(pick[:12]):
        z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
        pos = _cube_corner_positions(phi, z, y, x)
        pos_id = _identity_corners(z, y, x)

        ax = fig.add_subplot(3, 4, plot_idx + 1, projection='3d')
        # Identity wireframe in light gray (reference).
        seg_id = [[pos_id[u], pos_id[v]] for (u, v) in CUBE_EDGES]
        lc_id = Line3DCollection(seg_id, colors='lightgray', linewidths=0.8, linestyles='dashed')
        ax.add_collection3d(lc_id)
        # Deformed cube wireframe in dark.
        seg_def = [[pos[u], pos[v]] for (u, v) in CUBE_EDGES]
        lc_def = Line3DCollection(seg_def, colors='C0', linewidths=1.5)
        ax.add_collection3d(lc_def)
        # 8 corners as red dots.
        ax.scatter(pos[:, 2], pos[:, 1], pos[:, 0], c='red', s=30, depthshade=False)

        # Compute under-best-diagonal volumes for annotation.
        best_di = int(min_per_diag[:, z, y, x].argmax())
        s, e = _DIAGONALS[best_di]
        tets = _six_tets_for_diagonal(s, e) if best_di != 0 else list(_TET_VERTICES)
        # 6 tet volumes for the chosen diagonal.
        vols_at_cell = min_per_diag[best_di, z, y, x]
        ax.set_title(
            f'cell ({z},{y},{x})\n'
            f'best diag={_DIAGONALS[best_di]}  min_T={vols_at_cell:+.4f}',
            fontsize=9,
        )
        # Make axes equal-ish so the twist is visible.
        all_pts = np.vstack([pos, pos_id])
        for axis_idx, set_lim in enumerate(['set_xlim', 'set_ylim', 'set_zlim']):
            col = [2, 1, 0][axis_idx]  # x, y, z columns
            lo, hi = float(all_pts[:, col].min()), float(all_pts[:, col].max())
            mid = (lo + hi) / 2
            span = max(hi - lo, 1.5)
            getattr(ax, set_lim)(mid - span * 0.6, mid + span * 0.6)
        ax.set_xlabel('x', fontsize=7)
        ax.set_ylabel('y', fontsize=7)
        ax.set_zlabel('z', fontsize=7)
        ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT / 'unfixable_folds_3d.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\nSaved figure: {out}', flush=True)

    # Cube-centered view: subtract each cube's centroid so the local
    # twist is visible. Identity cube becomes the reference unit cube.
    fig2, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig2.suptitle(
        'Unfixable 3D folds (z=0..15 dense band) — cube-centered view\n'
        f'Each panel = one of the {n_unfix} cells whose 8 deformed corners admit no all-positive 6-tet decomposition\n'
        'Dashed gray = undeformed unit cube (1×1×1, reference);  '
        'solid blue = deformed cube edges;  red = corners labeled C0-C7\n'
        'Folded geometry visible when blue edges CROSS each other inside the cube',
        fontsize=10,
    )
    for plot_idx, k in enumerate(pick[:12]):
        z, y, x = int(nz[k]), int(ny[k]), int(nx[k])
        pos = _cube_corner_positions(phi, z, y, x)
        # Center on the DEFORMED centroid (so the deformed cube fills the view).
        cen = pos.mean(axis=0)
        pos_c = pos - cen
        ax = axes[plot_idx // 4, plot_idx % 4]
        # Deformed edges (solid blue).
        for u, v in CUBE_EDGES:
            ax.plot([pos_c[u, 2], pos_c[v, 2]],
                    [pos_c[u, 1], pos_c[v, 1]],
                    color='C0', linewidth=1.8, alpha=0.85)
        # Corners red with labels.
        for i in range(8):
            ax.scatter(pos_c[i, 2], pos_c[i, 1], c='red', s=80, zorder=10)
            # Offset label so it's readable.
            ax.annotate(
                f'C{i}',
                (pos_c[i, 2], pos_c[i, 1]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8, ha='left', va='bottom', zorder=11,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='red', alpha=0.85),
            )
        # Reference: identity unit cube (small dashed at same centroid).
        # Show as inset-style 1×1 box anchored at lower right.
        ax_xlo, ax_xhi = float(pos_c[:, 2].min()), float(pos_c[:, 2].max())
        ax_ylo, ax_yhi = float(pos_c[:, 1].min()), float(pos_c[:, 1].max())
        span_x = max(ax_xhi - ax_xlo, 1.5)
        span_y = max(ax_yhi - ax_ylo, 1.5)
        margin = 0.20 * max(span_x, span_y)
        ax.set_xlim(ax_xlo - margin, ax_xhi + margin)
        ax.set_ylim(ax_ylo - margin, ax_yhi + margin)
        # Reference 1x1 square in upper-right.
        ref_x0 = ax_xhi + margin * 0.1
        ref_y0 = ax_yhi - 1
        ref_box = plt.Rectangle((ref_x0 - 1.2, ref_y0), 1.0, 1.0,
                                 fill=False, edgecolor='gray', linestyle='--', linewidth=1)
        ax.add_patch(ref_box)
        ax.text(ref_x0 - 0.7, ref_y0 - 0.3, 'undeformed\nunit cell',
                fontsize=6, ha='center', color='gray')
        ax.set_title(
            f'cell ({z},{y},{x})  best diagonal min_T={severity[k]:+.4f}',
            fontsize=9,
        )
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Δx', fontsize=8)
        ax.set_ylabel('Δy', fontsize=8)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = OUTPUT / 'unfixable_folds_centered.png'
    fig2.savefig(out2, dpi=140, bbox_inches='tight')
    print(f'Saved figure: {out2}', flush=True)


if __name__ == '__main__':
    main()
