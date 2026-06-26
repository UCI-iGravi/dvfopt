"""Autonomous fold-cube atlas: analyse the geometry, topology, and
spatial distribution of fold cubes across multiple saved checkpoints
to understand what makes folds resolvable vs not, and design simpler
pipelines accordingly.

For each saved checkpoint:
  - List all fold cubes (n_neg, n<0.01)
  - Compute spatial distribution (clustering, k-ring overlap)
  - For each fold cube: SVD profile at center, crushed-edge analysis,
    best-diagonal feasibility, distance to other fold cubes
  - Suggest k-ring sizes per cube for SLSQP coverage

This is OFFLINE analysis — no optimisation. Used to design Stage 4
algorithms (the SLSQP step) more intelligently.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as cc_label

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


def _signed_vol(A, B, C, D):
    AB = B - A; AC = C - A; AD = D - A
    return (AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
            - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
            + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])) / 6.0


def _six_tets_for_diagonal(start, end):
    all_edges = [(v, w) for v in range(8) for w in range(v + 1, 8)
                 if (v ^ w) in (1, 2, 4)]
    perimeter = [e for e in all_edges if start not in e and end not in e]
    return [(start, a, b, end) for (a, b) in perimeter]


def cube_corners(phi, cz, cy, cx):
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1; iy = (i >> 1) & 1; ix = i & 1
        pos[i, 0] = (cz + iz) + phi[0, cz + iz, cy + iy, cx + ix]
        pos[i, 1] = (cy + iy) + phi[1, cz + iz, cy + iy, cx + ix]
        pos[i, 2] = (cx + ix) + phi[2, cz + iz, cy + iy, cx + ix]
    return pos


def cube_edges(pos):
    """Return list of (a, b, length) for the 12 cube edges."""
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(a, b, float(np.linalg.norm(pos[b] - pos[a]))) for (a, b) in edges]


def best_diag_min_vol(phi, cz, cy, cx):
    pos = cube_corners(phi, cz, cy, cx)
    best_min = -float('inf')
    best_di = 0
    for di in range(4):
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
            signs = [(+1.0 if _signed_vol(pos_id[i0], pos_id[i1], pos_id[i2], pos_id[i3]) > 0 else -1.0)
                     for (i0, i1, i2, i3) in tets]
        vols = np.empty(6)
        for k, (i0, i1, i2, i3) in enumerate(tets):
            vols[k] = signs[k] * _signed_vol(pos[i0], pos[i1], pos[i2], pos[i3])
        if vols.min() > best_min:
            best_min = float(vols.min())
            best_di = di
    return best_min, best_di


def svd_at_center(phi, cz, cy, cx):
    pos = cube_corners(phi, cz, cy, cx)
    J = np.zeros((3, 3))
    u, v, w = 0.5, 0.5, 0.5
    z_p, y_p, x_p = w, v, u
    for i in range(8):
        iz = (i >> 2) & 1; iy = (i >> 1) & 1; ix = i & 1
        bz = z_p if iz else (1 - z_p)
        by = y_p if iy else (1 - y_p)
        bx = x_p if ix else (1 - x_p)
        d_z = (+1 if iz else -1) * by * bx
        d_y = bz * (+1 if iy else -1) * bx
        d_x = bz * by * (+1 if ix else -1)
        for c in range(3):
            J[0, c] += d_z * pos[i, c]
            J[1, c] += d_y * pos[i, c]
            J[2, c] += d_x * pos[i, c]
    sv = np.linalg.svd(J, compute_uv=False)
    return float(sv[0]), float(sv[1]), float(sv[2]), float(np.linalg.det(J))


def cluster_fold_cubes(fold_cells, radius=3):
    """Cluster fold cubes via spatial proximity (k-ring overlap).

    Two cubes are in the same cluster if their (z, y, x) distance <= radius.
    Returns labels per fold cube and cluster centroids.
    """
    if not fold_cells:
        return [], []
    n = len(fold_cells)
    pts = np.array(fold_cells, dtype=int)
    # Build adjacency.
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            d = np.abs(pts[i] - pts[j]).max()
            if d <= radius:
                adj[i, j] = True
                adj[j, i] = True
    # Connected components via BFS.
    visited = [False] * n
    labels = [-1] * n
    cl = 0
    for i in range(n):
        if visited[i]: continue
        # BFS.
        q = [i]
        visited[i] = True
        labels[i] = cl
        while q:
            v = q.pop()
            for j in range(n):
                if adj[v, j] and not visited[j]:
                    visited[j] = True
                    labels[j] = cl
                    q.append(j)
        cl += 1
    centroids = []
    cluster_members = [[] for _ in range(cl)]
    for i, lbl in enumerate(labels):
        cluster_members[lbl].append(fold_cells[i])
    for members in cluster_members:
        pts = np.array(members)
        centroids.append(tuple(pts.mean(axis=0).astype(int)))
    return labels, centroids, cluster_members


def analyse_checkpoint(phi_path, phi_input, label):
    print(f'\n{"="*70}\n[{label}] {phi_path.name}\n{"="*70}', flush=True)
    phi = np.load(phi_path).astype(np.float64)
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    mn = float(V.min())
    print(f'n_neg={n_neg}  n<0.01={n_below}  min_T={mn:+.6f}', flush=True)

    min_per_cube = V.min(axis=0)
    fold_mask = (min_per_cube <= 0)
    fold_cells = list(zip(*np.where(fold_mask)))
    fold_cells = [tuple(int(x) for x in cell) for cell in fold_cells]
    if not fold_cells:
        print('  No fold cubes.', flush=True)
        return phi, []

    # Per-cube analysis.
    rows = []
    for (cz, cy, cx) in fold_cells:
        pos = cube_corners(phi, cz, cy, cx)
        edges = cube_edges(pos)
        min_edge = min(e[2] for e in edges)
        max_edge = max(e[2] for e in edges)
        crush_ratio = min_edge / max_edge
        s1, s2, s3, det = svd_at_center(phi, cz, cy, cx)
        best_min, best_di = best_diag_min_vol(phi, cz, cy, cx)
        my_min = min_per_cube[cz, cy, cx]
        rows.append({
            'pos': (cz, cy, cx),
            'min_T': float(my_min),
            'edges': edges,
            'min_edge': min_edge,
            'max_edge': max_edge,
            'crush_ratio': crush_ratio,
            'svd': (s1, s2, s3),
            'det_center': det,
            'best_min': best_min,
            'best_di': best_di,
        })
        print(
            f'  cube ({cz:3d},{cy:3d},{cx:3d}): min_T={my_min:+.5f}  '
            f'edges=[{min_edge:.3f}, {max_edge:.3f}] (crush={crush_ratio:.4f})  '
            f'svd=({s1:.2f},{s2:.2f},{s3:.4f})  '
            f'best_diag={_DIAGONALS[best_di]} min={best_min:+.5f}',
            flush=True,
        )

    # Clustering analysis.
    labels, centroids, members = cluster_fold_cubes(fold_cells, radius=3)
    n_clusters = len(centroids)
    print(f'\n  Spatial clusters (radius=3): {n_clusters}', flush=True)
    for i, (centroid, mems) in enumerate(zip(centroids, members)):
        print(f'    cluster {i}: centroid={centroid}, {len(mems)} cube(s)',
              flush=True)

    # Aggregate stats.
    print('\n  Aggregate cube properties:', flush=True)
    print(f'    crush_ratio   median={np.median([r["crush_ratio"] for r in rows]):.4f}  '
          f'min={min(r["crush_ratio"] for r in rows):.4f}  '
          f'max={max(r["crush_ratio"] for r in rows):.4f}',
          flush=True)
    print(f'    sigma_3       median={np.median([r["svd"][2] for r in rows]):.4f}  '
          f'min={min(r["svd"][2] for r in rows):.4f}  '
          f'max={max(r["svd"][2] for r in rows):.4f}',
          flush=True)
    print(f'    best_diag_min median={np.median([r["best_min"] for r in rows]):+.5f}  '
          f'#cubes-feasible-under-some-diag={sum(r["best_min"] > 0 for r in rows)}/{len(rows)}',
          flush=True)
    print(f'    best_diag distribution: '
          f'{[sum(r["best_di"] == d for r in rows) for d in range(4)]}',
          flush=True)

    return phi, rows


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)

    # Atlas of saved checkpoints.
    checkpoints = [
        ('INPUT (raw, 173)', OUTPUT / 'b0039_FULL_stage3_z000_016.npy'),
        ('MS_V1 (~6-9 folds)', OUTPUT / 'b0039_z0_15_ms_v1.npy'),
        ('MS_V2_TIGHT (2 folds)', OUTPUT / 'b0039_z0_15_ms_v2_tight.npy'),
        ('CHAIN_BEST (1 fold)', OUTPUT / 'b0039_z0_15_chain_best.npy'),
        ('STRICT_VIA_D (0 folds)', OUTPUT / 'b0039_z0_15_strict_via_D.npy'),
    ]
    all_results = {}
    for (label, p) in checkpoints:
        if not p.exists():
            print(f'\n[{label}] file {p} not found, skipping', flush=True)
            continue
        phi, rows = analyse_checkpoint(p, phi_input, label)
        all_results[label] = (phi, rows)

    # Cross-checkpoint insight: how do fold properties evolve through pipeline?
    print('\n\n' + '='*70, flush=True)
    print('CROSS-CHECKPOINT INSIGHT: fold property evolution', flush=True)
    print('='*70, flush=True)
    print(f'{"Checkpoint":<25} {"#folds":>7} {"med crush":>10} {"med sigma_3":>12} {"any-diag-feas":>14}',
          flush=True)
    print('-' * 70, flush=True)
    for label, (_, rows) in all_results.items():
        if not rows:
            print(f'{label:<25} {0:>7} {"-":>10} {"-":>12} {"-":>14}', flush=True)
            continue
        med_crush = np.median([r['crush_ratio'] for r in rows])
        med_sig3 = np.median([r['svd'][2] for r in rows])
        any_feas = sum(r['best_min'] > 0 for r in rows)
        print(f'{label:<25} {len(rows):>7} {med_crush:>10.4f} {med_sig3:>12.4f} '
              f'{any_feas:>9}/{len(rows):>4}',
              flush=True)


if __name__ == '__main__':
    main()
