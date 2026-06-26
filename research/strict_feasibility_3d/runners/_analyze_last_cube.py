"""Deep analysis of the single irreducible fold cube at lattice
(z=1, y=215, x=220). Goal: understand WHY this cube cannot be
resolved.

Investigations:
  1. Cube geometry in deformed space — corners, edges, faces.
  2. Per-tet signed volumes across all 4 main diagonals.
  3. Continuous-trilinear Jacobian det sampled densely inside cube.
  4. SVD of Jacobian (rank, sigma_1/sigma_3 ratio, principal directions).
  5. Compare to neighbouring cubes (geometry, fold status).
  6. Distance to a feasible configuration in corner-space.
  7. Visualize: 3D corner scatter with edges + tet faces coloured
     by sign; neighbour topology; per-diagonal tet-sign chart.

Outputs PNGs to research/strict_feasibility_3d/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_volumes_3d,
)

OUTPUT = _HERE / 'output'
FIG = _HERE / 'figures'
FIG.mkdir(exist_ok=True)
THRESHOLD = 0.01

_DIAGONALS = [(0, 7), (1, 6), (2, 5), (3, 4)]


def _six_tets_for_diagonal(start, end):
    all_edges = [(v, w) for v in range(8) for w in range(v + 1, 8) if (v ^ w) in (1, 2, 4)]
    perimeter = [e for e in all_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def _signed_vol(A, B, C, D):
    AB = B - A
    AC = C - A
    AD = D - A
    return (
        AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
        - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
        + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
    ) / 6.0


def cube_corners(phi, cz, cy, cx):
    """Return (8, 3) array of deformed positions."""
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        pos[i, 0] = (cz + iz) + phi[0, cz + iz, cy + iy, cx + ix]
        pos[i, 1] = (cy + iy) + phi[1, cz + iz, cy + iy, cx + ix]
        pos[i, 2] = (cx + ix) + phi[2, cz + iz, cy + iy, cx + ix]
    return pos


def vols_for_diag(pos, di):
    s, e = _DIAGONALS[di]
    if di == 0:
        tets = _TET_VERTICES
        signs = _TET_SIGN
    else:
        tets = _six_tets_for_diagonal(s, e)
        pos_id = np.zeros((8, 3))
        for i in range(8):
            pos_id[i, 0] = (i >> 2) & 1
            pos_id[i, 1] = (i >> 1) & 1
            pos_id[i, 2] = i & 1
        signs = [
            (+1.0 if _signed_vol(pos_id[i0], pos_id[i1], pos_id[i2], pos_id[i3]) > 0 else -1.0)
            for (i0, i1, i2, i3) in tets
        ]
    vols = np.empty(6)
    for k, (i0, i1, i2, i3) in enumerate(tets):
        vols[k] = signs[k] * _signed_vol(pos[i0], pos[i1], pos[i2], pos[i3])
    return vols, tets, signs


def trilinear_jacobian_det(pos, u, v, w):
    """Evaluate det(d phi / dx) of the trilinear interpolant at
    parametric coord (u, v, w) in [0,1]^3 within the cube.

    phi(u,v,w) interpolated trilinearly between the 8 corners.
    Returns scalar Jacobian determinant.
    """
    # Trilinear weights: 8 weights, partial derivatives wrt u, v, w.
    # Position[i] index: corner i has (iz, iy, ix) where iz=(i>>2)&1, etc.
    # So corner local coord is (iz*u + (1-iz)*(1-u)) etc. — wait, simpler:
    # weight[i] = (iz==1 ? w : 1-w) * (iy==1 ? v : 1-v) * (ix==1 ? u : 1-u)
    # where (u, v, w) parameterizes (x, y, z) channels.
    # Actually let me re-derive: corner index i has bits (iz, iy, ix).
    # Trilinear: phi = sum_i w_i * pos_i where
    #   w_i(z,y,x) = (z if iz else 1-z) * (y if iy else 1-y) * (x if ix else 1-x)
    # And we compute d phi / d(z,y,x).
    z, y, x = w, v, u  # match parametric to (z, y, x) for clarity.
    J = np.zeros((3, 3))  # rows: dz dy dx of phi, cols: phi components (z, y, x)
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        bz = z if iz else (1 - z)
        by = y if iy else (1 - y)
        bx = x if ix else (1 - x)
        d_z = (+1 if iz else -1) * by * bx
        d_y = bz * (+1 if iy else -1) * bx
        d_x = bz * by * (+1 if ix else -1)
        for c in range(3):
            J[0, c] += d_z * pos[i, c]
            J[1, c] += d_y * pos[i, c]
            J[2, c] += d_x * pos[i, c]
    return float(np.linalg.det(J)), J


def analyse_cube(phi, label, cz, cy, cx, phi_input=None):
    pos = cube_corners(phi, cz, cy, cx)
    print(f'\n{"=" * 70}\n[{label}] Cube at ({cz}, {cy}, {cx})\n{"=" * 70}', flush=True)
    print('\nLattice corner displacements:', flush=True)
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        z, y, x = cz + iz, cy + iy, cx + ix
        d = phi[:, z, y, x]
        print(
            f'  corner {i} (lattice {z},{y},{x}): d=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})',
            flush=True,
        )

    print('\nDeformed positions:', flush=True)
    for i in range(8):
        print(f'  corner {i}: pos=({pos[i, 0]:+.4f},{pos[i, 1]:+.4f},{pos[i, 2]:+.4f})', flush=True)

    # Edge lengths (deformed cube edges).
    edges_lattice = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),  # x-axis edges
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),  # y-axis edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # z-axis edges
    ]
    print('\nDeformed edge lengths:', flush=True)
    for a, b in edges_lattice:
        d_vec = pos[b] - pos[a]
        d_len = np.linalg.norm(d_vec)
        print(
            f'  edge ({a}, {b}): |pos[b]-pos[a]|={d_len:.4f}  '
            f'vec=({d_vec[0]:+.4f},{d_vec[1]:+.4f},{d_vec[2]:+.4f})',
            flush=True,
        )

    # 6-tet under all 4 diagonals.
    print('\n6-tet volumes under all 4 main diagonals:', flush=True)
    all_diag_vols = []
    for di in range(4):
        vols, _, _ = vols_for_diag(pos, di)
        all_diag_vols.append(vols)
        s, e = _DIAGONALS[di]
        neg = (vols < 0).sum()
        print(
            f'  diag {(s, e)}: vols={[f"{v:+.5f}" for v in vols]}  '
            f'min={vols.min():+.5f}  #neg={neg}',
            flush=True,
        )
    all_diag_vols = np.array(all_diag_vols)

    # Continuous trilinear Jacobian det on a 9x9x9 grid inside cube.
    print('\nTrilinear Jacobian det on 9x9x9 interior grid:', flush=True)
    N = 9
    grid_vals = np.empty((N, N, N))
    for iz in range(N):
        for iy in range(N):
            for ix in range(N):
                u = ix / (N - 1)
                v = iy / (N - 1)
                w = iz / (N - 1)
                det, _ = trilinear_jacobian_det(pos, u, v, w)
                grid_vals[iz, iy, ix] = det
    print(
        f'  min={grid_vals.min():+.5f}  max={grid_vals.max():+.5f}  '
        f'mean={grid_vals.mean():+.5f}  #neg={int((grid_vals < 0).sum())}/{N**3}',
        flush=True,
    )

    # SVD of Jacobian at cube center.
    _, J_center = trilinear_jacobian_det(pos, 0.5, 0.5, 0.5)
    sv = np.linalg.svd(J_center, compute_uv=False)
    print('\nJacobian at cube center (u=v=w=0.5):', flush=True)
    print(f'  J = \n{J_center}', flush=True)
    print(
        f'  singular values: sigma_1={sv[0]:.4f}  sigma_2={sv[1]:.4f}  sigma_3={sv[2]:.4f}',
        flush=True,
    )
    print(
        f'  det(J)={np.linalg.det(J_center):+.5f}  '
        f'condition number sigma_1/sigma_3={sv[0] / max(sv[2], 1e-12):.2f}',
        flush=True,
    )

    # SVDs at all 8 sample positions (corners in parametric space).
    print('\nSVD at parametric corners (sigma_1, sigma_2, sigma_3):', flush=True)
    for i in range(8):
        iz = (i >> 2) & 1
        iy = (i >> 1) & 1
        ix = i & 1
        det, J = trilinear_jacobian_det(pos, float(ix), float(iy), float(iz))
        sv_i = np.linalg.svd(J, compute_uv=False)
        print(
            f'  corner {i} (u,v,w)=({ix},{iy},{iz}): '
            f'sigma=({sv_i[0]:.3f},{sv_i[1]:.3f},{sv_i[2]:.3f})  '
            f'det={det:+.5f}',
            flush=True,
        )

    return pos, all_diag_vols, grid_vals, J_center, sv


def plot_cube_3d(pos, all_diag_vols, label, fname):
    """3D scatter of corners + edges + faces colored by tet sign."""
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: corners + edges (lattice cube shape).
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    edges_lattice = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges_lattice:
        ax1.plot(*zip(pos[a], pos[b]), 'b-', lw=2)
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=80, zorder=10)
    for i in range(8):
        ax1.text(pos[i, 0], pos[i, 1], pos[i, 2], f'  {i}', fontsize=10)
    ax1.set_title(f'{label}\nDeformed cube wireframe')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('X')

    # Subplot 2-5: tet volumes per diagonal.
    for di in range(4):
        ax = fig.add_subplot(2, 3, 2 + di, projection='3d')
        s, e = _DIAGONALS[di]
        if di == 0:
            tets = _TET_VERTICES
            signs = _TET_SIGN
        else:
            tets = _six_tets_for_diagonal(s, e)
            pos_id = np.zeros((8, 3))
            for i in range(8):
                pos_id[i, 0] = (i >> 2) & 1
                pos_id[i, 1] = (i >> 1) & 1
                pos_id[i, 2] = i & 1
            signs = [
                (+1.0 if _signed_vol(pos_id[i0], pos_id[i1], pos_id[i2], pos_id[i3]) > 0 else -1.0)
                for (i0, i1, i2, i3) in tets
            ]
        for k, (i0, i1, i2, i3) in enumerate(tets):
            tetra_pos = pos[[i0, i1, i2, i3]]
            vol = all_diag_vols[di][k]
            faces = [
                [tetra_pos[0], tetra_pos[1], tetra_pos[2]],
                [tetra_pos[0], tetra_pos[1], tetra_pos[3]],
                [tetra_pos[0], tetra_pos[2], tetra_pos[3]],
                [tetra_pos[1], tetra_pos[2], tetra_pos[3]],
            ]
            color = 'red' if vol < 0 else ('orange' if vol < THRESHOLD else 'green')
            poly = Poly3DCollection(
                faces, alpha=0.15, facecolor=color, edgecolor='black', linewidth=0.5
            )
            ax.add_collection3d(poly)
        for a, b in edges_lattice:
            ax.plot(*zip(pos[a], pos[b]), 'b-', lw=1, alpha=0.5)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=40, zorder=10)
        # Highlight diagonal endpoints.
        ax.scatter(
            pos[[s, e], 0],
            pos[[s, e], 1],
            pos[[s, e], 2],
            c='yellow',
            s=120,
            edgecolor='black',
            zorder=20,
        )
        n_neg = (all_diag_vols[di] < 0).sum()
        ax.set_title(f'Diagonal {(s, e)}  min={all_diag_vols[di].min():+.5f}  #neg={n_neg}')
        ax.set_xlabel('Z')
        ax.set_ylabel('Y')
        ax.set_zlabel('X')

    # Subplot 6: tet-volume bars under each diagonal.
    ax6 = fig.add_subplot(2, 3, 6)
    width = 0.2
    xs = np.arange(6)
    for di in range(4):
        s, e = _DIAGONALS[di]
        colors = [
            'red' if v < 0 else ('orange' if v < THRESHOLD else 'green') for v in all_diag_vols[di]
        ]
        ax6.bar(
            xs + di * width,
            all_diag_vols[di],
            width=width,
            label=f'diag {(s, e)}',
            edgecolor='black',
            linewidth=0.5,
        )
    ax6.axhline(0, color='k', lw=1)
    ax6.axhline(THRESHOLD, color='gray', lw=0.5, linestyle='--', label=f'thr={THRESHOLD}')
    ax6.set_xlabel('Tet index k (0..5)')
    ax6.set_ylabel('Signed volume')
    ax6.set_title('6-tet volumes per diagonal triangulation')
    ax6.legend(fontsize=8)
    ax6.set_xticks(xs + 1.5 * width)
    ax6.set_xticklabels([str(k) for k in xs])

    plt.suptitle(f'Cube ({label}) topology analysis', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved {FIG / fname}', flush=True)


def plot_neighbourhood(phi, cz, cy, cx, fname):
    """3-pane figure: fold cube + 1-cell ring + 2-cell ring, each
    showing how nearby cubes' min tet volume varies."""
    V = six_tet_volumes_3d(phi)
    min_V = V.min(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, ring in zip(axes, [0, 2, 5]):
        z0 = max(0, cz - ring)
        z1 = min(min_V.shape[0], cz + ring + 1)
        y0 = max(0, cy - ring)
        y1 = min(min_V.shape[1], cy + ring + 1)
        x0 = max(0, cx - ring)
        x1 = min(min_V.shape[2], cx + ring + 1)
        sub = min_V[z0:z1, y0:y1, x0:x1]
        # Project to y-x plane by taking min over z.
        proj = sub.min(axis=0)
        vmax = max(abs(proj.min()), abs(proj.max()), 1e-3)
        im = ax.imshow(
            proj, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower', extent=[x0, x1, y0, y1]
        )
        ax.scatter(
            [cx + 0.5], [cy + 0.5], c='yellow', s=120, edgecolor='black', zorder=10, label='target'
        )
        ax.set_title(f'min(tet_vol) projection, ring={ring}\n(z-min over z in [{z0}, {z1}])')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        plt.colorbar(im, ax=ax, label='min tet vol')

    plt.suptitle(f'Fold cube neighbourhood at ({cz},{cy},{cx})', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved {FIG / fname}', flush=True)


def plot_jacobian_field(pos, fname):
    """Trilinear det(J) sampled on a 31^3 grid inside the cube."""
    N = 31
    vals = np.empty((N, N, N))
    for iz in range(N):
        for iy in range(N):
            for ix in range(N):
                u = ix / (N - 1)
                v = iy / (N - 1)
                w = iz / (N - 1)
                vals[iz, iy, ix], _ = trilinear_jacobian_det(pos, u, v, w)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Mid-Z slice (w=0.5)', 'Mid-Y slice (v=0.5)', 'Mid-X slice (u=0.5)']
    slices = [vals[N // 2, :, :], vals[:, N // 2, :], vals[:, :, N // 2]]
    vmax = max(abs(vals.min()), abs(vals.max()))
    for ax, t, s in zip(axes, titles, slices):
        im = ax.imshow(s, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_title(t + f'  (min={s.min():+.4f}, max={s.max():+.4f})')
        plt.colorbar(im, ax=ax, label='det(J)')
    plt.suptitle('Trilinear Jacobian det inside the fold cube', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved {FIG / fname}', flush=True)


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    cur = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    V = six_tet_volumes_3d(cur)
    print(
        f'BEST_1fold loaded: n_neg={int((V <= 0).sum())}  '
        f'n<0.01={int((V < 0.01 - 1e-5).sum())}  '
        f'min_T={float(V.min()):+.6f}',
        flush=True,
    )

    # Analyse the fold cube.
    pos, diag_vols, grid_vals, J_c, sv = analyse_cube(cur, 'BEST_1fold', 1, 215, 220, phi_input)

    plot_cube_3d(pos, diag_vols, 'BEST_1fold cube (1,215,220)', 'last_fold_cube_3d.png')

    plot_jacobian_field(pos, 'last_fold_jacobian_interior.png')

    plot_neighbourhood(cur, 1, 215, 220, 'last_fold_neighbourhood.png')

    # Compare to the same cube in the original input (before any optimisation).
    print('\n\n' + '#' * 70, flush=True)
    print('COMPARISON: same cube in the original (unoptimised) input', flush=True)
    print('#' * 70, flush=True)
    pos_in, diag_vols_in, _, _, sv_in = analyse_cube(phi_input, 'INPUT', 1, 215, 220)

    plot_cube_3d(pos_in, diag_vols_in, 'INPUT cube (1,215,220)', 'last_fold_cube_input.png')

    # Compare delta of corners between INPUT and BEST_1fold.
    pos_delta = pos - pos_in
    print('\nCorner shifts (BEST_1fold - INPUT):', flush=True)
    for i in range(8):
        d = pos_delta[i]
        m = np.linalg.norm(d)
        print(
            f'  corner {i}: delta=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})  |delta|={m:.4f}',
            flush=True,
        )
    print(
        f'\nTotal corner movement: sum|delta|={float(np.linalg.norm(pos_delta).sum()):.4f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
