"""DVF rescaling utilities."""

import numpy as np
from scipy.ndimage import zoom


def scale_dvf(dvf, new_size):
    """Rescale a ``(3, 1, H, W)`` deformation field to *new_size* ``(new_H, new_W)``.

    Spatial interpolation is bilinear (``order=1``) and displacement
    magnitudes are scaled proportionally.
    """
    C, _, H, W = dvf.shape
    new_H, new_W = new_size
    scale_y = new_H / H
    scale_x = new_W / W

    dvf_resized = np.zeros((C, 1, new_H, new_W), dtype=dvf.dtype)
    for c in range(C):
        dvf_resized[c, 0] = zoom(dvf[c, 0], (scale_y, scale_x), order=1)

    dvf_resized[2, 0] *= scale_x  # dx
    dvf_resized[1, 0] *= scale_y  # dy
    return dvf_resized


def scale_dvf_3d(dvf, new_size):
    """Rescale a ``(3, D, H, W)`` deformation field to *new_size* ``(new_D, new_H, new_W)``.

    Spatial interpolation is trilinear (``order=1``) and displacement
    magnitudes are scaled proportionally.
    """
    C, D, H, W = dvf.shape
    new_D, new_H, new_W = new_size
    scale_z = new_D / D
    scale_y = new_H / H
    scale_x = new_W / W

    dvf_resized = np.zeros((C, new_D, new_H, new_W), dtype=dvf.dtype)
    for c in range(C):
        dvf_resized[c] = zoom(dvf[c], (scale_z, scale_y, scale_x), order=1)

    dvf_resized[0] *= scale_z  # dz
    dvf_resized[1] *= scale_y  # dy
    dvf_resized[2] *= scale_x  # dx
    return dvf_resized
