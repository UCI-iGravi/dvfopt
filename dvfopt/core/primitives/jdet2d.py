"""Flat 2D Jacobian-determinant forward form + adjoint (phi pack: [dx, dy])."""

import numpy as np

from dvfopt.core.primitives.jdet3d import _adjoint_central_diff
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d


def _split_phi_2d(phi_flat, grid_size):
    H, W = grid_size
    n = H * W
    dx = phi_flat[:n].reshape(H, W)
    dy = phi_flat[n:].reshape(H, W)
    return dx, dy, n


def jdet_2d_flat(phi_flat, grid_size):
    dx, dy, _ = _split_phi_2d(phi_flat, grid_size)
    return _numpy_jdet_2d(dy, dx).flatten()


def jdet_grad_T_v_2d(phi_flat, grid_size, v):
    H, W = grid_size
    n = H * W
    dx, dy, _ = _split_phi_2d(phi_flat, grid_size)
    v2 = v.reshape(H, W)

    ddx_dx = np.gradient(dx, axis=1)
    ddy_dy = np.gradient(dy, axis=0)
    ddx_dy = np.gradient(dx, axis=0)
    ddy_dx = np.gradient(dy, axis=1)
    a11 = 1 + ddx_dx
    a22 = 1 + ddy_dy
    # J = a11*a22 - ddx_dy*ddy_dx
    # Cofactors: dJ/da11 = a22; dJ/da22 = a11; dJ/d(ddx_dy) = -ddy_dx; dJ/d(ddy_dx) = -ddx_dy
    # dx column: contributes via a11 (∂/∂x) and ddx_dy (∂/∂y).
    g_dx = _adjoint_central_diff(a22 * v2, axis=1) + _adjoint_central_diff(-ddy_dx * v2, axis=0)
    g_dy = _adjoint_central_diff(a11 * v2, axis=0) + _adjoint_central_diff(-ddx_dy * v2, axis=1)
    out = np.empty(2 * n)
    out[:n] = g_dx.ravel()
    out[n:] = g_dy.ravel()
    return out


_jdet_2d_flat = jdet_2d_flat
_jdet_grad_T_v_2d = jdet_grad_T_v_2d

__all__ = ['jdet_2d_flat', 'jdet_grad_T_v_2d']
