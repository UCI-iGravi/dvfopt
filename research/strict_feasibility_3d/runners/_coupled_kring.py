"""Coupled k-ring joint solve to break the topological deadlock.

Findings from Part XI (cube anatomy):
  - Fold cube at (z=1, y=215, x=220) is locally feasible.
  - 14 neighbour cubes (sharing corners 2 and 6) are at active
    barrier — moving any corner of the fold cube pushes a
    neighbour negative.

Strategy: jointly optimise the corner displacements of the fold
cube AND all neighbours within k-ring distance, with EXPLICIT
constraints requiring every involved cube to have all 6 tets
above threshold. Outer-boundary corners (k-ring +1) are frozen
to anchor the system.

This formulation:
  - Treats neighbour feasibility as a CONSTRAINT (not a
    static-position requirement).
  - Allows the joint system to negotiate small shifts in
    many corners to free up the fold cube.
  - Uses SLSQP for k=1 (~100-200 DOF) or trust-constr for
    larger k.

Methods 1 (coupled k-ring) AND 2 (rank-deficient direction
push) from REPORT Part XI.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt.jacobian.tetrahedron_sign import (
    _TET_SIGN,
    _TET_VERTICES,
    six_tet_volumes_3d,
)

OUTPUT = _HERE / 'output'
THRESHOLD = 0.01
FEASIBILITY_THR = 0.005  # Coupled cubes must be > this.


def _corner_indices_of_cube(cz, cy, cx):
    """Return list of 8 (z, y, x) lattice corners of cube (cz, cy, cx)."""
    corners = []
    for i in range(8):
        iz = (i >> 2) & 1; iy = (i >> 1) & 1; ix = i & 1
        corners.append((cz + iz, cy + iy, cx + ix))
    return corners


def _signed_vol_np(A, B, C, D):
    AB = B - A; AC = C - A; AD = D - A
    return (AB[..., 0] * (AC[..., 1] * AD[..., 2] - AC[..., 2] * AD[..., 1])
            - AB[..., 1] * (AC[..., 0] * AD[..., 2] - AC[..., 2] * AD[..., 0])
            + AB[..., 2] * (AC[..., 0] * AD[..., 1] - AC[..., 1] * AD[..., 0])) / 6.0


def report(phi, label, phi_input=None):
    V = six_tet_volumes_3d(phi)
    n_neg = int((V <= 0).sum())
    n_below = int((V < THRESHOLD - 1e-5).sum())
    mn = float(V.min())
    L1 = '' if phi_input is None else f'  L1={float(np.abs(phi - phi_input).sum()):.1f}'
    print(f'{label}: n_neg={n_neg}  n<0.01={n_below}  min_T={mn:+.6f}{L1}',
          flush=True)
    return n_neg, n_below, mn


def build_coupled_problem(phi, fold_cz, fold_cy, fold_cx, k_ring=1):
    """Build the coupled k-ring problem.

    Returns:
      cubes : list of (cz, cy, cx) — every cube whose corners must
              be feasible (fold cube + k-ring neighbours).
      free_corners : sorted list of (z, y, x) corners that are
                     decision variables (corners touched by `cubes`).
      x0 : (3 * n_free,) initial values [dz_i, dy_i, dx_i for i in free_corners].
      phi_init : phi snapshot at start (for restoring).
    """
    D, H, W = phi.shape[1:]
    cells_max = (D - 1, H - 1, W - 1)
    # All cubes in k-ring around fold cube.
    cubes = []
    for dz in range(-k_ring, k_ring + 1):
        for dy in range(-k_ring, k_ring + 1):
            for dx in range(-k_ring, k_ring + 1):
                cz, cy, cx = fold_cz + dz, fold_cy + dy, fold_cx + dx
                if (0 <= cz < cells_max[0]
                        and 0 <= cy < cells_max[1]
                        and 0 <= cx < cells_max[2]):
                    cubes.append((cz, cy, cx))

    # Corners involved (union of cubes' 8 corners).
    all_corners = set()
    for (cz, cy, cx) in cubes:
        for c in _corner_indices_of_cube(cz, cy, cx):
            all_corners.add(c)
    free_corners = sorted(all_corners)

    # Initial x: 3 channels (dz, dy, dx) per corner.
    x0 = np.zeros(3 * len(free_corners))
    for ci, (z, y, x) in enumerate(free_corners):
        x0[3*ci+0] = phi[0, z, y, x]
        x0[3*ci+1] = phi[1, z, y, x]
        x0[3*ci+2] = phi[2, z, y, x]
    return cubes, free_corners, x0


def make_apply_x(cubes, free_corners, phi_base):
    """Return a function (x -> phi_modified) that applies decision-vector x
    to the corners of phi_base."""
    corner_idx_map = {c: i for i, c in enumerate(free_corners)}

    def apply_x(x):
        phi_out = phi_base.copy()
        for (z, y, x_lat), ci in corner_idx_map.items():
            phi_out[0, z, y, x_lat] = x[3*ci+0]
            phi_out[1, z, y, x_lat] = x[3*ci+1]
            phi_out[2, z, y, x_lat] = x[3*ci+2]
        return phi_out

    return apply_x, corner_idx_map


def make_constraint_fn(cubes, corner_idx_map):
    """Return a function (x -> g) where g is a 6 * len(cubes) array of
    (tet_vol - threshold). For feasibility we need g >= 0.

    Vectorised: builds all 8-corner positions for all cubes at once.
    """
    n_cubes = len(cubes)
    # For each cube, store the indices into x (24 entries) and base coords (z, y, x).
    cube_corner_x_idx = np.zeros((n_cubes, 8, 3), dtype=np.int64)
    cube_corner_base = np.zeros((n_cubes, 8, 3))
    for ci, (cz, cy, cx) in enumerate(cubes):
        for k in range(8):
            iz = (k >> 2) & 1; iy = (k >> 1) & 1; ix = k & 1
            corner = (cz + iz, cy + iy, cx + ix)
            corner_i = corner_idx_map[corner]
            cube_corner_x_idx[ci, k, 0] = 3 * corner_i + 0
            cube_corner_x_idx[ci, k, 1] = 3 * corner_i + 1
            cube_corner_x_idx[ci, k, 2] = 3 * corner_i + 2
            cube_corner_base[ci, k] = [cz + iz, cy + iy, cx + ix]

    # Tet topology: 6 tets per cube under default diagonal.
    tets = np.array(_TET_VERTICES, dtype=np.int64)  # (6, 4)
    signs = np.array(_TET_SIGN)

    def constraint(x):
        # Build positions (n_cubes, 8, 3).
        ds = x[cube_corner_x_idx]  # (n_cubes, 8, 3)
        pos = cube_corner_base + ds  # (n_cubes, 8, 3)
        # Compute 6 tet volumes per cube.
        A = pos[:, tets[:, 0], :]  # (n_cubes, 6, 3)
        B = pos[:, tets[:, 1], :]
        C = pos[:, tets[:, 2], :]
        D = pos[:, tets[:, 3], :]
        vols = _signed_vol_np(A, B, C, D) * signs[None, :]  # (n_cubes, 6)
        return (vols - FEASIBILITY_THR).reshape(-1)  # (n_cubes * 6,)

    return constraint, n_cubes


def make_objective(x0_anchor):
    """Quadratic objective: minimise sum of squared shifts from anchor."""
    def obj(x):
        d = x - x0_anchor
        return 0.5 * float(np.dot(d, d))
    def grad(x):
        return x - x0_anchor
    return obj, grad


def run_coupled_solve(phi, fold_cz, fold_cy, fold_cx, k_ring=1, max_iter=200):
    print(f'\n=== Coupled k-ring solve: k={k_ring} around ({fold_cz},'
          f'{fold_cy},{fold_cx}) ===', flush=True)
    cubes, free_corners, x0 = build_coupled_problem(
        phi, fold_cz, fold_cy, fold_cx, k_ring)
    n_dof = len(x0)
    n_cubes = len(cubes)
    n_constraints = 6 * n_cubes
    print(f'  cubes involved: {n_cubes}', flush=True)
    print(f'  free corners:  {len(free_corners)}', flush=True)
    print(f'  decision DOF:  {n_dof}', flush=True)
    print(f'  constraints:   {n_constraints} (6 tets x {n_cubes} cubes >= {FEASIBILITY_THR})',
          flush=True)

    apply_x, corner_idx_map = make_apply_x(cubes, free_corners, phi)
    constraint_fn, _ = make_constraint_fn(cubes, corner_idx_map)
    obj, obj_grad = make_objective(x0.copy())

    # Initial constraint check.
    g0 = constraint_fn(x0)
    n_violated = int((g0 < 0).sum())
    print(f'  initial constraint check: #violated={n_violated} '
          f'(min_g={g0.min():+.6f}, max_g={g0.max():+.6f})', flush=True)

    # SLSQP.
    cons = [{'type': 'ineq', 'fun': constraint_fn}]
    t0 = time.time()
    res = minimize(
        obj, x0, jac=obj_grad,
        constraints=cons, method='SLSQP',
        options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': True},
    )
    wall = time.time() - t0
    print(f'  SLSQP result: success={res.success}, '
          f'fun={res.fun:.4f}, iter={res.nit}, wall={wall:.1f}s', flush=True)
    print(f'  message: {res.message}', flush=True)

    # Apply result.
    phi_out = apply_x(res.x)
    g_final = constraint_fn(res.x)
    n_viol_final = int((g_final < 0).sum())
    print(f'  final local constraint check: #violated={n_viol_final} '
          f'(min_g={g_final.min():+.6f})', flush=True)

    return phi_out, res, n_violated, n_viol_final


def rank_deficient_push(phi, fold_cz, fold_cy, fold_cx, eps=0.5):
    """Push corners 2 and 6 along their respective sigma_3 right-singular
    vectors and re-run M10Tet to see if it lands in a better basin."""
    print(f'\n=== Rank-deficient direction push at ({fold_cz},{fold_cy},'
          f'{fold_cx}) ===', flush=True)
    pos = np.zeros((8, 3))
    for i in range(8):
        iz = (i >> 2) & 1; iy = (i >> 1) & 1; ix = i & 1
        pos[i, 0] = (fold_cz + iz) + phi[0, fold_cz + iz, fold_cy + iy, fold_cx + ix]
        pos[i, 1] = (fold_cy + iy) + phi[1, fold_cz + iz, fold_cy + iy, fold_cx + ix]
        pos[i, 2] = (fold_cx + ix) + phi[2, fold_cz + iz, fold_cy + iy, fold_cx + ix]

    # SVD at corners 2 and 6 (locally).
    # Use trilinear-Jacobian at parametric (0,1,0) and (0,1,1) respectively.
    def jac_at(u, v, w):
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
        return J

    J2 = jac_at(0.0, 1.0, 0.0)
    J6 = jac_at(0.0, 1.0, 1.0)
    # v3 = last column of V in SVD (right singular vec for smallest sv).
    U2, s2, Vt2 = np.linalg.svd(J2)
    U6, s6, Vt6 = np.linalg.svd(J6)
    v3_2 = Vt2[-1]  # right singular vector at corner 2
    v3_6 = Vt6[-1]  # right singular vector at corner 6
    print(f'  Corner 2 SVD: sigma=({s2[0]:.3f},{s2[1]:.3f},{s2[2]:.3f})',
          flush=True)
    print(f'  Corner 6 SVD: sigma=({s6[0]:.3f},{s6[1]:.3f},{s6[2]:.3f})',
          flush=True)
    print(f'  v3 at corner 2: ({v3_2[0]:+.3f},{v3_2[1]:+.3f},{v3_2[2]:+.3f})',
          flush=True)
    print(f'  v3 at corner 6: ({v3_6[0]:+.3f},{v3_6[1]:+.3f},{v3_6[2]:+.3f})',
          flush=True)

    # Push corners 2 and 6 by eps in their +v3 directions (and try -v3 too).
    best_phi = None
    best_n_neg = None
    for sign_2 in (+1, -1):
        for sign_6 in (+1, -1):
            phi_p = phi.copy()
            # Corner 2: lattice (cz, cy+1, cx). Channels are (dz, dy, dx).
            # v3 is in (dz, dy, dx) direction since J's first index is dz/dz etc.
            # Wait — J rows are (d/dz, d/dy, d/dx) of phi. The right singular
            # vec v3 is a unit vector in the INPUT space of J (z, y, x) — i.e.,
            # the direction of the smallest stretching. The OUTPUT direction is
            # u3 (left singular). For corner translation, we want to displace
            # the OUTPUT side. Use u3 instead.
            u3_2 = U2[:, -1]
            u3_6 = U6[:, -1]
            phi_p[0, fold_cz, fold_cy+1, fold_cx] += sign_2 * eps * u3_2[0]
            phi_p[1, fold_cz, fold_cy+1, fold_cx] += sign_2 * eps * u3_2[1]
            phi_p[2, fold_cz, fold_cy+1, fold_cx] += sign_2 * eps * u3_2[2]
            phi_p[0, fold_cz+1, fold_cy+1, fold_cx] += sign_6 * eps * u3_6[0]
            phi_p[1, fold_cz+1, fold_cy+1, fold_cx] += sign_6 * eps * u3_6[1]
            phi_p[2, fold_cz+1, fold_cy+1, fold_cx] += sign_6 * eps * u3_6[2]
            n_neg, _, mn = report(phi_p, f'  pushed sign=({sign_2:+d},{sign_6:+d}) eps={eps:.2f}')
            if best_phi is None or n_neg < best_n_neg:
                best_phi = phi_p; best_n_neg = n_neg
    return best_phi, best_n_neg


def main():
    phi_input = np.load(OUTPUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
    cur = np.load(OUTPUT / 'b0039_z0_15_BEST_1fold.npy').astype(np.float64)
    n_neg, n_below, _ = report(cur, 'START (BEST_1fold)', phi_input)

    if n_neg == 0 and n_below == 0:
        print('Already strict feasible.', flush=True)
        return

    FOLD_CZ, FOLD_CY, FOLD_CX = 1, 215, 220

    # ============================================================
    # METHOD 1: Coupled k-ring SLSQP at k=1 (3^3 = 27 cubes minus boundary trim).
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD 1: Coupled k-ring SLSQP', flush=True)
    print('='*70, flush=True)
    for k in (1, 2):
        phi_after, res, n_v0, n_vf = run_coupled_solve(
            cur, FOLD_CZ, FOLD_CY, FOLD_CX, k_ring=k, max_iter=300)
        n_neg, n_below, _ = report(phi_after,
                                    f'  GLOBAL after k={k} coupled SLSQP',
                                    phi_input)
        if n_neg == 0 and n_below == 0:
            print(f'\n*** STRICT 100% FEASIBLE via k={k} coupled solve ***',
                  flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_coupled.npy', phi_after)
            return
        # Save for chaining.
        if k == 1:
            phi_after_k1 = phi_after.copy()

    # ============================================================
    # METHOD 2: Rank-deficient direction push followed by M10Tet recover.
    # ============================================================
    print('\n' + '='*70, flush=True)
    print('METHOD 2: Rank-deficient v3 push + M10Tet recovery', flush=True)
    print('='*70, flush=True)
    for eps in (0.2, 0.5, 1.0):
        pushed, n_pushed = rank_deficient_push(
            cur.copy(), FOLD_CZ, FOLD_CY, FOLD_CX, eps=eps)
        print(f'\n  best after push (eps={eps}): n_neg={n_pushed}', flush=True)
        # Recover with M10Tet @ 0.012.
        print('  M10Tet @ 0.012 recovery ...', flush=True)
        from dvfopt import (
            HarmonicALMBarrier3DStrategy,
            L1Objective,
            Solver,
            Tet6Constraint3D,
        )
        solver = Solver(
            constraint=Tet6Constraint3D(shape=pushed.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMBarrier3DStrategy(),
            threshold=0.012,
        )
        t0 = time.time()
        recovered = solver.fit(pushed).corrected
        print(f'  recovery wall={time.time()-t0:.1f}s', flush=True)
        n_neg, n_below, _ = report(recovered,
                                    f'  after recovery eps={eps}',
                                    phi_input)
        if n_neg == 0 and n_below == 0:
            print(f'\n*** STRICT 100% FEASIBLE via rank-push eps={eps} ***',
                  flush=True)
            np.save(OUTPUT / 'b0039_z0_15_strict_via_rank_push.npy', recovered)
            return

    print('\n=== Final ===\n  Could not reach n_neg=0 via coupled / rank-push',
          flush=True)


if __name__ == '__main__':
    main()
