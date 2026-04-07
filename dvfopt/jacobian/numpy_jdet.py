"""Numpy-based Jacobian determinant computation for 2D and 3D fields."""

import numpy as np


def _numpy_jdet_2d(dy, dx):
    """Compute 2D Jacobian determinant from displacement components (numpy).

    Uses central differences for interior pixels, matching SimpleITK's
    ``DisplacementFieldJacobianDeterminant`` for interior values.

    Parameters
    ----------
    dy, dx : ndarray, shape ``(H, W)``

    Returns
    -------
    ndarray, shape ``(H, W)``
    """
    ddx_dx = np.gradient(dx, axis=1)  # ∂dx/∂x
    ddy_dy = np.gradient(dy, axis=0)  # ∂dy/∂y
    ddx_dy = np.gradient(dx, axis=0)  # ∂dx/∂y
    ddy_dx = np.gradient(dy, axis=1)  # ∂dy/∂x
    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx


def jacobian_det2D(phi_xy):
    """Compute the Jacobian determinant from a ``(2, H, W)`` phi array.

    Also accepts ``(2, 1, H, W)`` (the extra unit dimension is squeezed).
    """
    H, W = phi_xy.shape[-2:]
    dy = phi_xy[0].reshape(H, W)
    dx = phi_xy[1].reshape(H, W)
    jdet = _numpy_jdet_2d(dy, dx)
    return jdet[np.newaxis, :, :]  # (1, H, W) to match existing API


def _numpy_jdet_3d(dz, dy, dx):
    """Compute 3D Jacobian determinant from displacement components (numpy).

    Uses central differences via ``np.gradient`` for all 9 partial
    derivatives of the deformation gradient tensor.

    Parameters
    ----------
    dz, dy, dx : ndarray, shape ``(D, H, W)``

    Returns
    -------
    ndarray, shape ``(D, H, W)``
    """
    ddx_dx = np.gradient(dx, axis=2)
    ddx_dy = np.gradient(dx, axis=1)
    ddx_dz = np.gradient(dx, axis=0)

    ddy_dx = np.gradient(dy, axis=2)
    ddy_dy = np.gradient(dy, axis=1)
    ddy_dz = np.gradient(dy, axis=0)

    ddz_dx = np.gradient(dz, axis=2)
    ddz_dy = np.gradient(dz, axis=1)
    ddz_dz = np.gradient(dz, axis=0)

    a11 = 1 + ddx_dx;  a12 = ddx_dy;      a13 = ddx_dz
    a21 = ddy_dx;       a22 = 1 + ddy_dy;  a23 = ddy_dz
    a31 = ddz_dx;       a32 = ddz_dy;      a33 = 1 + ddz_dz

    return (a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31))


def jacobian_det3D(phi):
    """Compute the Jacobian determinant from a ``(3, D, H, W)`` phi array.

    Returns shape ``(D, H, W)``.
    """
    dz = phi[0]
    dy = phi[1]
    dx = phi[2]
    return _numpy_jdet_3d(dz, dy, dx)
