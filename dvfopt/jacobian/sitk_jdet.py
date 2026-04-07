"""SimpleITK-based Jacobian determinant computation."""

import numpy as np
import SimpleITK as sitk


def sitk_jacobian_determinant(deformation: np.ndarray, reverse_channels=True):
    """Compute Jacobian determinant using SimpleITK.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Displacement field with channels ``[dz, dy, dx]``.
    reverse_channels : bool
        If ``True`` (default), reverses the channel order to SimpleITK's
        expected ``[dx, dy, dz]`` ordering before computing the Jacobian.

    Returns
    -------
    ndarray, shape ``(D, H, W)``
    """
    deformation = np.transpose(deformation, [1, 2, 3, 0])  # (3,D,H,W) -> (D,H,W,3)
    if reverse_channels:
        deformation = deformation[:, :, :, [2, 1, 0]]  # [dz,dy,dx] -> [dx,dy,dz]
    sitk_displacement_field = sitk.GetImageFromArray(deformation, isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian_det_np_arr = sitk.GetArrayFromImage(jacobian_det_volume)
    return jacobian_det_np_arr
