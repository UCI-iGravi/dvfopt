"""Cluster-aware simple pipeline (Variant C): M10Tet -> cluster-
aware SLSQP -> M10Tet recovery.

Atlas finding: fold cubes are CLUSTERED (e.g., raw input has 13
clusters; one cluster has 42 cubes around (1, 218, 273)). Per-cube
SLSQP iterates over each fold cube; cluster-aware SLSQP centers ONE
SLSQP on each cluster's centroid with k-ring sized to cover the
cluster. Should be faster and more effective for multi-cube clusters.

Pipeline:
  1. M10Tet on raw input (~30 min)              → ~10-20 folds in ~4 clusters
  2. Identify fold clusters (radius=3)
  3. For each cluster:
     a. Compute centroid and max-radius in cells
     b. Choose k_ring = max_radius + 2 (covers cluster + 2-ring halo)
     c. Run coupled SLSQP @ thr=1e-3 (Method D config)
  4. After all clusters, check global n_neg
  5. M10Tet @ 0.012 recovery to tighten

Total estimated wall: ~30 min (M10Tet) + ~1 min (cluster SLSQP) + ~30 min (recovery) = ~1 hour.
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

import research.strict_feasibility_3d.runners._coupled_kring as ck
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
from research.strict_feasibility_3d.runners._coupled_kring import (
    build_coupled_problem,
    make_apply_x,
    make_constraint_fn,
    make_objective,
    report,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
SLSQP_THR = 1e-3


def cluster_fold_cubes(fold_cells, radius=3):
    """Cluster fold cubes via spatial proximity (Chebyshev distance <= radius)."""
    if not fold_cells:
        return [], []
    n = len(fold_cells)
    pts = np.array(fold_cells, dtype=int)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.abs(pts[i] - pts[j]).max()
            if d <= radius:
                adj[i, j] = True
                adj[j, i] = True
    visited = [False] * n
    labels = [-1] * n
    cl = 0
    for i in range(n):
        if visited[i]:
            continue
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
    cluster_members = [[] for _ in range(cl)]
    for i, lbl in enumerate(labels):
        cluster_members[lbl].append(fold_cells[i])
    centroids = []
    radii = []
    for members in cluster_members:
        pts = np.array(members)
        c = pts.mean(axis=0).astype(int)
        r = int(np.max(np.abs(pts - c)))
        centroids.append(tuple(int(x) for x in c))
        radii.append(r)
    return labels, centroids, cluster_members, radii


def run_slsqp_around(phi, cz, cy, cx, k_ring):
    cubes, free_corners, x0 = build_coupled_problem(phi, cz, cy, cx, k_ring)
    apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi)
    ck.FEASIBILITY_THR = SLSQP_THR
    constraint_fn, _ = make_constraint_fn(cubes, corner_idx_map)
    obj, obj_grad = make_objective(x0.copy())
    cons = [{'type': 'ineq', 'fun': constraint_fn}]
    t0 = time.time()
    res = minimize(
        obj,
        x0,
        jac=obj_grad,
        constraints=cons,
        method='SLSQP',
        options={'maxiter': 300, 'ftol': 1e-9, 'disp': False},
    )
    wall = time.time() - t0
    return apply_x(res.x), res, wall, len(cubes), 3 * len(free_corners)


def m10tet(phi, thr=0.015):
    from dvfopt import (
        HarmonicALMBarrier3DStrategy,
        L1Objective,
        Solver,
        Tet6Constraint3D,
    )

    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrier3DStrategy(),
        threshold=thr,
    )
    return solver.fit(phi).corrected


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    report(phi_input, 'INPUT (raw)', None)

    # === Step 1: M10Tet on raw input. ===
    print('\n=== Step 1: M10Tet @ 0.015 on raw input ===', flush=True)
    t0 = time.time()
    phi_after_m10 = m10tet(phi_input, 0.015)
    wall1 = time.time() - t0
    print(f'  M10Tet wall={wall1:.1f}s', flush=True)
    report(phi_after_m10, '  after M10Tet', phi_input)

    # === Step 2: Identify clusters. ===
    print('\n=== Step 2: Identify fold clusters ===', flush=True)
    V = six_tet_volumes_3d(phi_after_m10)
    min_per_cube = V.min(axis=0)
    fold_mask = min_per_cube <= 0
    fold_cells = [tuple(int(c) for c in p) for p in zip(*np.where(fold_mask))]
    print(f'  # fold cubes: {len(fold_cells)}', flush=True)
    _, centroids, members, radii = cluster_fold_cubes(fold_cells, radius=3)
    n_clusters = len(centroids)
    print(f'  # clusters (radius=3): {n_clusters}', flush=True)
    for i, (c, mem, r) in enumerate(zip(centroids, members, radii)):
        print(f'    cluster {i}: centroid={c}, size={len(mem)}, radius={r}', flush=True)

    # === Step 3: Per-cluster SLSQP. ===
    print('\n=== Step 3: Per-cluster coupled SLSQP @ thr=1e-3 ===', flush=True)
    cur = phi_after_m10.copy()
    total_slsqp_wall = 0
    for i, (c, mem, r) in enumerate(zip(centroids, members, radii)):
        # Choose k_ring = r + 2 to ensure boundary containment.
        k_ring = max(2, r + 2)
        cz, cy, cx = c
        # Check boundary safety.
        D, H, W = cur.shape[1:]
        if not (
            k_ring <= cz < D - 1 - k_ring
            and k_ring <= cy < H - 1 - k_ring
            and k_ring <= cx < W - 1 - k_ring
        ):
            # Clip to safe k_ring.
            safe_k = min(cz, cy, cx, D - 1 - cz - 1, H - 1 - cy - 1, W - 1 - cx - 1)
            k_ring = max(1, min(k_ring, safe_k))
            print(f'  cluster {i}: k_ring clipped to {k_ring} for boundary safety', flush=True)
        new, res, wall, n_cubes, n_dof = run_slsqp_around(cur, cz, cy, cx, k_ring)
        total_slsqp_wall += wall
        if res is None or not res.success:
            print(
                f'  cluster {i}: SLSQP did not converge (success={res.success if res else None})',
                flush=True,
            )
            continue
        V_new = six_tet_volumes_3d(new)
        n_neg_new = int((V_new <= 0).sum())
        V_cur = six_tet_volumes_3d(cur)
        n_neg_cur = int((V_cur <= 0).sum())
        print(
            f'  cluster {i}: k={k_ring} cubes={n_cubes} DOF={n_dof} '
            f'wall={wall:.1f}s  n_neg={n_neg_cur}->{n_neg_new}',
            flush=True,
        )
        # Accept if global n_neg didn't increase substantially.
        if n_neg_new <= n_neg_cur + 5:
            cur = new
        else:
            print('    rejected (n_neg increased)', flush=True)
    print(f'\n  total cluster-SLSQP wall={total_slsqp_wall:.1f}s', flush=True)
    n_after_slsqp, b_after_slsqp, _ = report(cur, '  after cluster SLSQP', phi_input)

    # === Step 4: M10Tet @ 0.012 recovery. ===
    if n_after_slsqp > 0 or b_after_slsqp > 0:
        print('\n=== Step 4: M10Tet @ 0.012 recovery ===', flush=True)
        t_rec = time.time()
        final = m10tet(cur, 0.012)
        wall_rec = time.time() - t_rec
        print(f'  recovery wall={wall_rec:.1f}s', flush=True)
        n, b, mn = report(final, '  FINAL', phi_input)
    else:
        final = cur
        wall_rec = 0.0
        n, b = n_after_slsqp, b_after_slsqp

    total_wall = wall1 + total_slsqp_wall + wall_rec
    print(
        f'\n=== VARIANT C FINAL ===\n'
        f'  n_neg={n}, n<0.01={b}\n'
        f'  total wall = {total_wall:.1f}s = {total_wall / 60:.1f} min',
        flush=True,
    )
    if n == 0 and b == 0:
        np.save(OUTPUT / 'b0039_z0_15_strict_via_simple_C.npy', final)
        print('  *** STRICT FEASIBLE via VARIANT C ***', flush=True)


if __name__ == '__main__':
    main()
