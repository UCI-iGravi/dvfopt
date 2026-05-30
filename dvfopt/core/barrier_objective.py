"""Building blocks for the 3D Jdet barrier objectives.

phi_flat packing: ``[dx_flat, dy_flat, dz_flat]``.

This module provides:

* :func:`jdet_full` — flat 3D Jacobian determinant.
* :func:`_jdet_grad_T_v` — analytical ``J^T @ v`` (vectorised; no per-voxel loop).
* :func:`_adjoint_central_diff` — adjoint of ``np.gradient`` along one axis,
  also reused by the 2D barrier solver.

The actual penalty / log-barrier objective functions and their L-BFGS-B
homotopy live in :mod:`dvfopt.core._barrier_core` — shared by every CPU
barrier solver in the package.
"""

import numpy as np

from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d


def _split_phi(phi_flat, grid_size):
    D, H, W = grid_size
    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    return dx, dy, dz, n


def jdet_full(phi_flat, grid_size):
    """Return flat J(u) for full grid (length D*H*W)."""
    dx, dy, dz, _ = _split_phi(phi_flat, grid_size)
    return _numpy_jdet_3d(dz, dy, dx).flatten()


def _adjoint_central_diff(w, axis):
    """Adjoint of ``np.gradient(_, axis=axis)`` applied to w.

    np.gradient uses central differences (±0.5) interior, one-sided (±1)
    at the two endpoints. Returns G.T @ w as an array shaped like w.
    """
    n = w.shape[axis]
    if n == 1:
        return np.zeros_like(w)

    w_m = np.moveaxis(w, axis, 0)
    out_m = np.zeros_like(w_m)

    # Forward stencil rows i: G[i, i+1] = c_next[i], G[i, i-1] = -c_prev[i]
    # G[0,0] = -1, G[n-1, n-1] = +1 (one-sided endpoint diagonals).
    c_next = np.full(n, 0.5)
    c_next[0] = 1.0
    c_prev = np.full(n, 0.5)
    c_prev[n - 1] = 1.0

    bshape = [1] * w_m.ndim
    bshape[0] = n - 1
    # Off-diagonal contributions
    out_m[1:n] += c_next[: n - 1].reshape(bshape) * w_m[: n - 1]
    out_m[0 : n - 1] -= c_prev[1:n].reshape(bshape) * w_m[1:n]
    # Diagonal endpoint terms
    out_m[0] -= w_m[0]
    out_m[n - 1] += w_m[n - 1]

    return np.moveaxis(out_m, 0, axis)


def _jdet_grad_T_v(phi_flat, grid_size, v):
    """Compute (dJ/du)^T @ v vectorised, returning a flat 3N gradient.

    Equivalent to ``jdet_constraint_jacobian_3d(phi_flat, grid_size).T @ v``
    but avoids the per-voxel Python loop.
    """
    D, H, W = grid_size
    N = D * H * W
    dx, dy, dz, _ = _split_phi(phi_flat, grid_size)
    v3 = v.reshape(D, H, W)

    # Deformation gradient components
    ddx_dx = np.gradient(dx, axis=2)
    ddx_dy = np.gradient(dx, axis=1)
    ddx_dz = np.gradient(dx, axis=0)
    ddy_dx = np.gradient(dy, axis=2)
    ddy_dy = np.gradient(dy, axis=1)
    ddy_dz = np.gradient(dy, axis=0)
    ddz_dx = np.gradient(dz, axis=2)
    ddz_dy = np.gradient(dz, axis=1)
    ddz_dz = np.gradient(dz, axis=0)

    a11 = 1 + ddx_dx
    a12 = ddx_dy
    a13 = ddx_dz
    a21 = ddy_dx
    a22 = 1 + ddy_dy
    a23 = ddy_dz
    a31 = ddz_dx
    a32 = ddz_dy
    a33 = 1 + ddz_dz

    # Cofactors
    C11 = a22 * a33 - a23 * a32
    C12 = -(a21 * a33 - a23 * a31)
    C13 = a21 * a32 - a22 * a31
    C21 = -(a12 * a33 - a13 * a32)
    C22 = a11 * a33 - a13 * a31
    C23 = -(a11 * a32 - a12 * a31)
    C31 = a12 * a23 - a13 * a22
    C32 = -(a11 * a23 - a13 * a21)
    C33 = a11 * a22 - a12 * a21

    # For each component k of u, dJ(q)/du_k(p) = C_{k,axis}(q) * stencil_axis(p,q)
    # so (dJ/du_k)^T @ v at p = sum_axis adjoint_grad_axis(C_{k,axis} * v)[p].
    # Component ordering: u = [dx, dy, dz] → use (C11,C12,C13) for dx etc.

    # dx block uses cofactors (C11 at axis=2, C12 at axis=1, C13 at axis=0)
    g_dx = (
        _adjoint_central_diff(C11 * v3, axis=2)
        + _adjoint_central_diff(C12 * v3, axis=1)
        + _adjoint_central_diff(C13 * v3, axis=0)
    )
    g_dy = (
        _adjoint_central_diff(C21 * v3, axis=2)
        + _adjoint_central_diff(C22 * v3, axis=1)
        + _adjoint_central_diff(C23 * v3, axis=0)
    )
    g_dz = (
        _adjoint_central_diff(C31 * v3, axis=2)
        + _adjoint_central_diff(C32 * v3, axis=1)
        + _adjoint_central_diff(C33 * v3, axis=0)
    )

    out = np.empty(3 * N)
    out[:N] = g_dx.ravel()
    out[N : 2 * N] = g_dy.ravel()
    out[2 * N :] = g_dz.ravel()
    return out
