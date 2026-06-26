"""Faster coupled k-ring SLSQP with analytical constraint Jacobian.

The default _coupled_kring.py uses scipy's finite-difference for the
constraint Jacobian — at 500-1000 DOF this means 500-1000 constraint
evaluations per SLSQP iteration. This script provides the analytical
Jacobian, which has 24 nonzeros per row (one tet volume depends only
on its 4 corners × 3 coords = 12 DOF; we cover 24 because we pack
8 corners × 3 = 24 per cube).

Speed-up target: 10-100x for the SLSQP step. Enables larger k_ring
(k=4, k=5) within reasonable wall-time.

Each tet volume:
  V(A, B, C, D) = sign * (1/6) * det(B-A, C-A, D-A)

where each of A, B, C, D is a point in R^3.

The gradient of V w.r.t. each component of each corner is computed
in closed form (cross-product structure).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.jacobian.tetrahedron_sign import _TET_VERTICES, _TET_SIGN


def signed_vol_grad(A, B, C, D):
    """Compute (V, dV/dA, dV/dB, dV/dC, dV/dD) where V = (1/6)*det(B-A, C-A, D-A).

    Inputs: (3,) arrays for A, B, C, D.
    Returns: scalar V, and four (3,) gradient vectors.
    """
    AB = B - A
    AC = C - A
    AD = D - A
    # V = (1/6) * AB . (AC x AD)
    cross_CD = np.cross(AC, AD)
    V = AB.dot(cross_CD) / 6.0
    # dV/dB = (1/6) * (AC x AD).
    grad_B = cross_CD / 6.0
    # dV/dC = (1/6) * (AD x AB).
    grad_C = np.cross(AD, AB) / 6.0
    # dV/dD = (1/6) * (AB x AC).
    grad_D = np.cross(AB, AC) / 6.0
    # dV/dA = - dV/dB - dV/dC - dV/dD.
    grad_A = -(grad_B + grad_C + grad_D)
    return V, grad_A, grad_B, grad_C, grad_D


def make_constraint_fn_with_jacobian(cubes, corner_idx_map, feasibility_thr,
                                      n_dof):
    """Return (fn(x) -> g, jac(x) -> J) for the coupled cube-feasibility
    constraint set.

    g shape: (n_cubes * 6,) — each value is tet_vol - feasibility_thr.
    J shape: (n_cubes * 6, n_dof) — sparse but stored dense for SLSQP.
    """
    n_cubes = len(cubes)
    # Pre-compute index mappings.
    # cube_corner_x_idx[ci, k, c] -> index into x for cube ci, corner k, channel c.
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
    tets = np.array(_TET_VERTICES, dtype=np.int64)  # (6, 4)
    signs = np.array(_TET_SIGN)

    def fn(x):
        ds = x[cube_corner_x_idx]  # (n_cubes, 8, 3)
        pos = cube_corner_base + ds
        A = pos[:, tets[:, 0], :]
        B = pos[:, tets[:, 1], :]
        C = pos[:, tets[:, 2], :]
        D = pos[:, tets[:, 3], :]
        AB = B - A; AC = C - A; AD = D - A
        vols = (AB[..., 0] * (AC[..., 1] * AD[..., 2] - AC[..., 2] * AD[..., 1])
                - AB[..., 1] * (AC[..., 0] * AD[..., 2] - AC[..., 2] * AD[..., 0])
                + AB[..., 2] * (AC[..., 0] * AD[..., 1] - AC[..., 1] * AD[..., 0])) / 6.0
        vols = vols * signs[None, :]
        return (vols - feasibility_thr).reshape(-1)

    def jac(x):
        """Compute (n_cubes*6, n_dof) Jacobian via the analytic formula."""
        ds = x[cube_corner_x_idx]  # (n_cubes, 8, 3)
        pos = cube_corner_base + ds  # (n_cubes, 8, 3)
        J = np.zeros((n_cubes * 6, n_dof))
        # For each cube ci and tet k, compute gradient w.r.t. each of the 4 tet corners.
        for ci in range(n_cubes):
            for k in range(6):
                i0, i1, i2, i3 = tets[k]
                A = pos[ci, i0]; B = pos[ci, i1]; C = pos[ci, i2]; D = pos[ci, i3]
                AB = B - A; AC = C - A; AD = D - A
                # V = (1/6) * AB . (AC x AD)
                # dV/dB = (1/6) * (AC x AD)
                cross_CD = np.array([
                    AC[1] * AD[2] - AC[2] * AD[1],
                    AC[2] * AD[0] - AC[0] * AD[2],
                    AC[0] * AD[1] - AC[1] * AD[0],
                ])
                cross_DB = np.array([
                    AD[1] * AB[2] - AD[2] * AB[1],
                    AD[2] * AB[0] - AD[0] * AB[2],
                    AD[0] * AB[1] - AD[1] * AB[0],
                ])
                cross_BC = np.array([
                    AB[1] * AC[2] - AB[2] * AC[1],
                    AB[2] * AC[0] - AB[0] * AC[2],
                    AB[0] * AC[1] - AB[1] * AC[0],
                ])
                grad_B = cross_CD / 6.0
                grad_C = cross_DB / 6.0
                grad_D = cross_BC / 6.0
                grad_A = -(grad_B + grad_C + grad_D)
                sgn = signs[k]
                row = ci * 6 + k
                # Scatter into J.
                for c in range(3):
                    J[row, cube_corner_x_idx[ci, i0, c]] += sgn * grad_A[c]
                    J[row, cube_corner_x_idx[ci, i1, c]] += sgn * grad_B[c]
                    J[row, cube_corner_x_idx[ci, i2, c]] += sgn * grad_C[c]
                    J[row, cube_corner_x_idx[ci, i3, c]] += sgn * grad_D[c]
        return J

    return fn, jac, n_cubes
