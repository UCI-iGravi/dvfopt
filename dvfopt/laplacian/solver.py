"""Laplacian interpolation solvers for deformation field construction."""

import numpy as np
from scipy.sparse.linalg import lgmres

from dvfopt.io.nifti import loadNiiImages
from dvfopt.laplacian.matrix import laplacianA3D


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

    fIndices = (fpoints[:, 0] * ny * nz + fpoints[:, 1] * nz + fpoints[:, 2]).astype(int)

    Ycount[fIndices] += 1
    Xd[fIndices] += mpoints[:, 0] - fpoints[:, 0]
    Yd[fIndices] += mpoints[:, 1] - fpoints[:, 1]
    Zd[fIndices] += mpoints[:, 2] - fpoints[:, 2]

    boundary_indices = Ycount.nonzero()[0]
    return Xd, Yd, Zd, boundary_indices


def compute3DLaplacianFromShape(shape, mpoints, fpoints):
    """Solve the Laplacian system for a given shape and correspondences.

    Returns ``(phi_xy, A_, b_)`` where *phi_xy* is the concatenated
    ``[Xd, Yd]`` solution, *A_* is the block Laplacian, and *b_* is
    the concatenated RHS.
    """
    Xd, Yd, Zd, boundary_indices = _prepare_correspondence_data(shape, mpoints, fpoints)

    A = laplacianA3D(shape, boundary_indices)

    A0 = np.zeros((A.shape[0], A.shape[1]))
    A_ = np.block([
        [A.todense(), A0],
        [A0, A.todense()]
    ])
    b_ = np.concatenate([Xd, Yd])
    phi_xy = lgmres(A_, b_, rtol=1e-2)[0]

    return phi_xy, A_, b_


def sliceToSlice3DLaplacian(fixedImage, mpoints, fpoints):
    """End-to-end Laplacian interpolation from a NIfTI image and correspondences.

    Returns ``(deformationField, A, Xd, Yd, Zd)``.
    """
    fdata = loadNiiImages([fixedImage])

    nx, ny, nz = fdata.shape
    nd = len(fdata.shape)

    deformationField = np.zeros((nd, nx, ny, nz))

    Xd, Yd, Zd, boundary_indices = _prepare_correspondence_data(fdata.shape, mpoints, fpoints)

    A = laplacianA3D(fdata.shape, boundary_indices)

    dx = lgmres(A, Xd, rtol=1e-2)[0]
    dy = lgmres(A, Yd, rtol=1e-2)[0]
    dz = lgmres(A, Zd, rtol=1e-2)[0]

    deformationField[0] = np.zeros(fdata.shape)
    deformationField[1] = dy.reshape(fdata.shape)
    deformationField[2] = dz.reshape(fdata.shape)

    return deformationField, A, Xd, Yd, Zd


def createA(fixedImage, mpoints, fpoints, use_correspondences=True):
    """Build the Laplacian matrix *A* from an image array and correspondences.

    Parameters
    ----------
    use_correspondences : bool
        When ``False``, build the matrix without Dirichlet correspondence BCs.
    """
    _, _, _, boundary_indices = _prepare_correspondence_data(
        fixedImage.shape, mpoints, fpoints)
    return laplacianA3D(fixedImage.shape, boundary_indices,
                        use_correspondences=use_correspondences)
