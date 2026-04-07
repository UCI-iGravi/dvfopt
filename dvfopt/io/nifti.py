"""NIfTI image loading utilities."""

import nibabel as nib
import numpy as np
import scipy.ndimage


def load_nii_images(image_list, scale=False):
    """
    image_list can contain both paths to .nii images or loaded nii images.
    Loads nii images from the paths provided in image_list and returns a list
    of 3D numpy arrays representing image data. If numpy data is present in
    image_list, the same will be returned.
    """
    if scale:
        if isinstance(image_list[0], str):
            f_image = nib.load(image_list[0])
        else:
            scale = False

    images = []
    for image in image_list:
        if isinstance(image, str):
            nii_image = nib.load(image)
            imdata = nii_image.get_fdata()

            # Execution is faster on copied data
            if scale:
                scales = tuple(
                    np.array(nii_image.header.get_zooms())
                    / np.array(f_image.header.get_zooms())
                )
                imdata = scipy.ndimage.zoom(imdata.copy(), scales, order=1)
            images.append(imdata.copy())
        else:
            images.append(image.copy())

    if len(image_list) == 1:
        return images[0]
    return images
