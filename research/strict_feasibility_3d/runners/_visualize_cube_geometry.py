"""Clean visualization of the deformed fold cube — geometry, topology,
and rank-deficiency interpretation.

Outputs:
  figures/last_fold_geometry.png — 3 panels: ideal | input | BEST_1fold
  figures/last_fold_zoom.png — zoom into the crushed edge
  figures/last_fold_rank_deficiency.png — local plate geometry at corners 2 & 6
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


OUTPUT = _HERE / 'output'
FIG = _HERE / 'figures'


def cube_corners(phi, cz, cy, cx):
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1; iy = (i >> 1) & 1; ix = i & 1
        pos[i, 0] = (cz + iz) + phi[0, cz + iz, cy + iy, cx + ix]
        pos[i, 1] = (cy + iy) + phi[1, cz + iz, cy + iy, cx + ix]
        pos[i, 2] = (cx + ix) + phi[2, cz + iz, cy + iy, cx + ix]
    return pos


EDGES = [
    (0,1), (2,3), (4,5), (6,7),  # x-edges (i&1 toggles)
    (0,2), (1,3), (4,6), (5,7),  # y-edges
    (0,4), (1,5), (2,6), (3,7),  # z-edges
]
FACES = [
    [0, 1, 3, 2],  # z=cz face
    [4, 5, 7, 6],  # z=cz+1 face
    [0, 1, 5, 4],  # y=cy face
    [2, 3, 7, 6],  # y=cy+1 face
    [0, 2, 6, 4],  # x=cx face
    [1, 3, 7, 5],  # x=cx+1 face
]


def ideal_unit_cube():
    pos = np.zeros((8, 3))
    for i in range(8):
        pos[i, 0] = (i >> 2) & 1
        pos[i, 1] = (i >> 1) & 1
        pos[i, 2] = i & 1
    return pos


def plot_cube_3d(ax, pos, title, highlight_edges=None, color_edges_by_length=True):
    """Plot one cube with edges + faces + corner labels."""
    # Faces (translucent, no fill colour).
    polys = [pos[f] for f in FACES]
    coll = Poly3DCollection(polys, alpha=0.06, facecolor='steelblue',
                            edgecolor='none')
    ax.add_collection3d(coll)
    # Edges, coloured by length.
    edge_lens = [np.linalg.norm(pos[b] - pos[a]) for (a, b) in EDGES]
    if color_edges_by_length:
        # Map length to colour: 0 → red, ~1 → green, large → blue.
        edge_segs = [(pos[a], pos[b]) for (a, b) in EDGES]
        norm = plt.Normalize(vmin=0, vmax=max(3.0, max(edge_lens)))
        cmap = plt.cm.viridis
        for k, ((a, b), L) in enumerate(zip(EDGES, edge_lens)):
            color = cmap(norm(L))
            lw = 1.5 if L > 0.5 else 4.5  # thicken collapsed edges
            ax.plot(*zip(pos[a], pos[b]), color=color, lw=lw, alpha=0.95)
    else:
        for (a, b) in EDGES:
            ax.plot(*zip(pos[a], pos[b]), 'k-', lw=1)
    # Highlight specific edges.
    if highlight_edges:
        for (a, b) in highlight_edges:
            ax.plot(*zip(pos[a], pos[b]), color='red', lw=6, alpha=1.0)
    # Corner scatter.
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=80,
               edgecolors='black', linewidth=1, zorder=20)
    for i in range(8):
        ax.text(pos[i, 0], pos[i, 1], pos[i, 2], f' {i}',
                fontsize=10, fontweight='bold')
    ax.set_title(title)
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')


def fig_geometry_comparison():
    """3-panel side-by-side: ideal | input | BEST_1fold."""
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    phi_best = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)

    pos_ideal = ideal_unit_cube()
    pos_input = cube_corners(phi_input, 1, 215, 220)
    pos_best = cube_corners(phi_best, 1, 215, 220)

    fig = plt.figure(figsize=(20, 7))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_cube_3d(ax1, pos_ideal,
                 'IDEAL unit cube\n(undeformed reference)',
                 color_edges_by_length=False)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_cube_3d(ax2, pos_input,
                 'INPUT cube at (1, 215, 220)\nbefore any optimisation\n'
                 '(huge, chaotic, but 6-tet feasible under default diag)',
                 highlight_edges=None)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_cube_3d(ax3, pos_best,
                 'BEST_1fold cube\nafter all optimisation cycles\n'
                 'edge (2,6) CRUSHED to 0.03 voxels (red)',
                 highlight_edges=[(2, 6)])

    plt.suptitle('Same cube at lattice (z=1, y=215, x=220) — three states',
                 fontsize=13)
    plt.tight_layout()
    out = FIG / 'last_fold_geometry.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}', flush=True)


def fig_zoom_crushed_edge():
    """Zoom into the BEST_1fold cube around the crushed edge."""
    phi_best = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    pos = cube_corners(phi_best, 1, 215, 220)

    fig = plt.figure(figsize=(16, 7))

    # Left: full cube with crushed edge highlighted.
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_cube_3d(ax1, pos,
                 'BEST_1fold cube\n(2,6) edge collapsed to 0.03 voxels',
                 highlight_edges=[(2, 6)])

    # Right: zoom into the corners 2 and 6 region.
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Show only corners 2 and 6 with their connections.
    edges_to_show = [
        (2, 6),  # the crushed edge itself
        (2, 0), (2, 3), (2, 7),  # other connections from 2 (diagonals included)
        (6, 4), (6, 7), (6, 0),  # other connections from 6
        (2, 1), (2, 5), (6, 1), (6, 3), (6, 5),  # extra diagonals
    ]
    # Filter to only valid (those in any FACE).
    plot_cube_3d(ax2, pos,
                 f'Zoom: corners 2 and 6 nearly coincide\n'
                 f'pos(2) = ({pos[2,0]:.3f}, {pos[2,1]:.3f}, {pos[2,2]:.3f})\n'
                 f'pos(6) = ({pos[6,0]:.3f}, {pos[6,1]:.3f}, {pos[6,2]:.3f})\n'
                 f'|edge(2,6)| = 0.0305',
                 highlight_edges=[(2, 6)])
    # Tighten the zoom around corners 2 and 6.
    mid = (pos[2] + pos[6]) / 2
    r = 2.0
    ax2.set_xlim(mid[0] - r, mid[0] + r)
    ax2.set_ylim(mid[1] - r, mid[1] + r)
    ax2.set_zlim(mid[2] - r, mid[2] + r)

    plt.suptitle('The crushed edge — geometric source of the topological deadlock',
                 fontsize=13)
    plt.tight_layout()
    out = FIG / 'last_fold_zoom.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}', flush=True)


def fig_rank_deficiency_analysis():
    """Visual of the rank deficiency — show σ₃ across the cube + a
    plate-vs-bulk illustration."""
    phi_best = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    pos = cube_corners(phi_best, 1, 215, 220)

    # Compute σ₃ on 11×11×11 inside the cube.
    N = 17
    sigma3 = np.zeros((N, N, N))
    det = np.zeros((N, N, N))
    for iz in range(N):
        for iy in range(N):
            for ix in range(N):
                u = ix / (N - 1); v = iy / (N - 1); w = iz / (N - 1)
                z_p, y_p, x_p = w, v, u
                J = np.zeros((3, 3))
                for i in range(8):
                    iiz = (i >> 2) & 1; iiy = (i >> 1) & 1; iix = i & 1
                    bz = z_p if iiz else (1 - z_p)
                    by = y_p if iiy else (1 - y_p)
                    bx = x_p if iix else (1 - x_p)
                    d_z = (+1 if iiz else -1) * by * bx
                    d_y = bz * (+1 if iiy else -1) * bx
                    d_x = bz * by * (+1 if iix else -1)
                    for c in range(3):
                        J[0, c] += d_z * pos[i, c]
                        J[1, c] += d_y * pos[i, c]
                        J[2, c] += d_x * pos[i, c]
                sv = np.linalg.svd(J, compute_uv=False)
                sigma3[iz, iy, ix] = sv[2]
                det[iz, iy, ix] = np.linalg.det(J)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: sigma_3 (smallest singular value) cross-sections.
    titles = [f'sigma_3 at u=0 (x-min)', 'sigma_3 at u=0.5 (mid)',
              'sigma_3 at u=1 (x-max)']
    slices = [sigma3[:, :, 0], sigma3[:, :, N//2], sigma3[:, :, -1]]
    for ax, t, s in zip(axes[0], titles, slices):
        im = ax.imshow(s, cmap='magma', vmin=0, vmax=sigma3.max(),
                        origin='lower', extent=[0, 1, 0, 1])
        ax.set_title(t + f'\nmin={s.min():.3f} (full cube: {sigma3.min():.3f})')
        ax.set_xlabel('v (y-param)'); ax.set_ylabel('w (z-param)')
        plt.colorbar(im, ax=ax, label='sigma_3')

    # Row 2: det(J) cross-sections at same u-slices.
    titles = ['det(J) at u=0', 'det(J) at u=0.5', 'det(J) at u=1']
    slices = [det[:, :, 0], det[:, :, N//2], det[:, :, -1]]
    vmax = max(abs(det.min()), abs(det.max()))
    for ax, t, s in zip(axes[1], titles, slices):
        im = ax.imshow(s, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        origin='lower', extent=[0, 1, 0, 1])
        ax.set_title(t + f'\nmin={s.min():+.3f} max={s.max():+.3f}'
                       f'  #neg={int((s < 0).sum())}/{s.size}')
        ax.set_xlabel('v (y-param)'); ax.set_ylabel('w (z-param)')
        plt.colorbar(im, ax=ax, label='det(J)')

    # Annotate corners 2 and 6.
    for ax in axes.flat:
        # corner 2: (u, v, w) = (0, 1, 0) so v=1, w=0; appears at u=0 slice.
        ax.plot(1.0, 0.0, 'o', color='cyan', mew=2, mec='black', mfc='none',
                markersize=14, zorder=10)
        # corner 6: (u, v, w) = (0, 1, 1) so v=1, w=1; appears at u=0 slice.
        ax.plot(1.0, 1.0, 'o', color='cyan', mew=2, mec='black', mfc='none',
                markersize=14, zorder=10)

    plt.suptitle('Rank-deficiency analysis: sigma_3 (top) and det(J) (bottom)\n'
                 'inside the BEST_1fold cube — cyan circles mark corners 2 and 6 in (v,w) projection',
                 fontsize=13)
    plt.tight_layout()
    out = FIG / 'last_fold_rank_deficiency.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}', flush=True)


def fig_feasibility_landscape():
    """Show that the cube IS feasible if corners 2 and 6 separate.
    Sweep d_z(corner 2) and d_z(corner 6) over a small range and
    plot n_neg of the cube under default diagonal."""
    phi_best = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    cz, cy, cx = 1, 215, 220

    # Anchor: corner 2 at lattice (cz, cy+1, cx); corner 6 at (cz+1, cy+1, cx).
    # We vary their dz channels by +-shift around their current values.
    base = phi_best.copy()

    from dvfopt.jacobian.tetrahedron_sign import _TET_VERTICES, _TET_SIGN
    def cube_vol(phi):
        pos = cube_corners(phi, cz, cy, cx)
        vols = np.empty(6)
        for k, (i0, i1, i2, i3) in enumerate(_TET_VERTICES):
            A, B, C, D = pos[i0], pos[i1], pos[i2], pos[i3]
            AB = B - A; AC = C - A; AD = D - A
            det = (AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
                   - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
                   + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])) / 6.0
            vols[k] = float(_TET_SIGN[k]) * det
        return vols.min(), vols

    # Sweep dz(corner 2) and dz(corner 6) over [-2, 2] (independent).
    N = 51
    shifts = np.linspace(-2.0, 2.0, N)
    min_v = np.zeros((N, N))
    for i, ds2 in enumerate(shifts):
        for j, ds6 in enumerate(shifts):
            phi_tmp = base.copy()
            phi_tmp[0, cz, cy+1, cx] = base[0, cz, cy+1, cx] + ds2
            phi_tmp[0, cz+1, cy+1, cx] = base[0, cz+1, cy+1, cx] + ds6
            mn, _ = cube_vol(phi_tmp)
            min_v[i, j] = mn

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap: min tet volume.
    vmax = max(abs(min_v.min()), abs(min_v.max()), 0.5)
    im = axes[0].imshow(min_v, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                         origin='lower',
                         extent=[shifts[0], shifts[-1], shifts[0], shifts[-1]])
    axes[0].contour(shifts, shifts, min_v, levels=[0], colors='black', linewidths=2)
    axes[0].contour(shifts, shifts, min_v, levels=[0.01], colors='gray',
                     linewidths=1, linestyles='--')
    axes[0].plot(0, 0, 'o', color='yellow', mec='black', mew=2,
                  markersize=14, label='current (0, 0)')
    # Best feasible point (max of min_v).
    i_best, j_best = np.unravel_index(min_v.argmax(), min_v.shape)
    axes[0].plot(shifts[j_best], shifts[i_best], '*', color='cyan',
                  mec='black', mew=2, markersize=20,
                  label=f'best (Δ2={shifts[i_best]:+.2f}, Δ6={shifts[j_best]:+.2f}, '
                        f'min={min_v[i_best,j_best]:+.3f})')
    axes[0].set_xlabel('shift in dz(corner 6)  [voxels]')
    axes[0].set_ylabel('shift in dz(corner 2)  [voxels]')
    axes[0].set_title('Min tet volume vs. shifts in dz of corners 2 and 6\n'
                       '(this cube only — neighbours ignored)')
    axes[0].legend()
    plt.colorbar(im, ax=axes[0], label='min(6-tet volume)')

    # Show that this cube IS feasible somewhere in this 2D slice.
    feasible_mask = min_v >= 0.0
    feasible_pct = 100 * feasible_mask.mean()
    axes[1].imshow(feasible_mask.astype(int), cmap='RdYlGn', origin='lower',
                    extent=[shifts[0], shifts[-1], shifts[0], shifts[-1]],
                    vmin=0, vmax=1)
    axes[1].contour(shifts, shifts, min_v, levels=[0], colors='black', linewidths=2)
    axes[1].plot(0, 0, 'o', color='yellow', mec='black', mew=2,
                  markersize=14, label='current')
    axes[1].set_xlabel('shift in dz(corner 6)')
    axes[1].set_ylabel('shift in dz(corner 2)')
    axes[1].set_title(f'Cube-feasibility region in (dz(2), dz(6)) space\n'
                       f'{feasible_pct:.1f}% of this slice is locally feasible')
    axes[1].legend()

    plt.suptitle('The cube IS locally feasible if corners 2 and 6 separate along z\n'
                 '(But moving them breaks ~5 neighbour cubes — Part X)', fontsize=13)
    plt.tight_layout()
    out = FIG / 'last_fold_feasibility_landscape.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}', flush=True)


def main():
    fig_geometry_comparison()
    fig_zoom_crushed_edge()
    fig_rank_deficiency_analysis()
    fig_feasibility_landscape()


if __name__ == '__main__':
    main()
