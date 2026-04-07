"""Laplacian interpolation solvers for deformation field construction."""

import numpy as np
from scipy.sparse.linalg import lgmres

from laplacian_interp.matrix import laplacian_a_3d


def _load_volume(fixed_image):
    """Return a 3D numpy array from *fixed_image*.

    Accepts either a numpy array (returned as-is) or a file path string
    (loaded via nibabel).
    """
    if isinstance(fixed_image, np.ndarray):
        return fixed_image
    # File path — lazy-import nibabel so users who only pass arrays
    # don't need it installed.
    import nibabel as nib
    return nib.load(fixed_image).get_fdata().copy()


def _prepare_correspondence_data(shape, mpoints, fpoints):
    """Compute displacement arrays and boundary indices from correspondences.

    Returns ``(Xd, Yd, Zd, boundary_indices)``.
    """
    nx, ny, nz = shape
    flen = nx * ny * nz

    Xd = np.zeros(flen)
    Yd = np.zeros(flen)
    Zd = np.zeros(flen)
    Ycount = np.zeros(flen)

    f_indices = (fpoints[:, 0] * ny * nz + fpoints[:, 1] * nz + fpoints[:, 2]).astype(int)

    Ycount[f_indices] += 1
    Xd[f_indices] += mpoints[:, 0] - fpoints[:, 0]
    Yd[f_indices] += mpoints[:, 1] - fpoints[:, 1]
    Zd[f_indices] += mpoints[:, 2] - fpoints[:, 2]

    boundary_indices = Ycount.nonzero()[0]
    return Xd, Yd, Zd, boundary_indices


def slice_to_slice_3d_laplacian(fixed_image, mpoints, fpoints):
    """End-to-end Laplacian interpolation from a volume and correspondences.

    Parameters
    ----------
    fixed_image : ndarray or str
        A 3D numpy array giving the volume shape, **or** a file path to
        a NIfTI image (loaded via nibabel).
    mpoints : ndarray, shape ``(N, 3)``
        Moving correspondences ``[z, y, x]``.
    fpoints : ndarray, shape ``(N, 3)``
        Fixed correspondences ``[z, y, x]``.

    Returns
    -------
    deformation_field : ndarray, shape ``(3, nz, ny, nx)``
    A : sparse matrix
    Xd, Yd, Zd : ndarray
    """
    fdata = _load_volume(fixed_image)

    nx, ny, nz = fdata.shape
    nd = len(fdata.shape)

    deformation_field = np.zeros((nd, nx, ny, nz))

    Xd, Yd, Zd, boundary_indices = _prepare_correspondence_data(fdata.shape, mpoints, fpoints)

    A = laplacian_a_3d(fdata.shape, boundary_indices)

    dx = lgmres(A, Xd, rtol=1e-2)[0]
    dy = lgmres(A, Yd, rtol=1e-2)[0]
    dz = lgmres(A, Zd, rtol=1e-2)[0]

    deformation_field[0] = np.zeros(fdata.shape)
    deformation_field[1] = dy.reshape(fdata.shape)
    deformation_field[2] = dz.reshape(fdata.shape)

    return deformation_field, A, Xd, Yd, Zd
