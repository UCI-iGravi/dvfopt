"""SimpleITK-based Jacobian determinant computation."""

import numpy as np
import SimpleITK as sitk


def sitk_jacobian_determinant(deformation: np.ndarray, transpose_displacements=True):
    '''
    deformation - 3, X, Y, Z, 3
    '''
    deformation = np.transpose(deformation, [1, 2, 3, 0])
    if transpose_displacements:
        deformation = deformation[:, :, :, [2, 1, 0]]
    sitk_displacement_field = sitk.GetImageFromArray(deformation, isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian_det_np_arr = sitk.GetArrayFromImage(jacobian_det_volume)
    return jacobian_det_np_arr
