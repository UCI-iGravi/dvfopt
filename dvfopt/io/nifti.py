"""NIfTI image loading utilities."""

import nibabel as nib
import numpy as np
import scipy.ndimage


def loadNiiImages(imageList, scale=False):
    """
    imageList can contain both paths to .nii images or loaded nii images.
    Loads nii images from the paths provided in imageList and returns a list
    of 3D numpy arrays representing image data. If numpy data is present in
    imageList, the same will be returned.
    """
    if scale:
        if isinstance(imageList[0], str):
            fImage = nib.load(imageList[0])
        else:
            scale = False

    images = []
    for image in imageList:
        if isinstance(image, str):
            niiImage = nib.load(image)
            imdata = niiImage.get_fdata()

            # Execution is faster on copied data
            if scale:
                scales = tuple(
                    np.array(niiImage.header.get_zooms())
                    / np.array(fImage.header.get_zooms())
                )
                imdata = scipy.ndimage.zoom(imdata.copy(), scales, order=1)
            images.append(imdata.copy())
        else:
            images.append(image.copy())

    if len(imageList) == 1:
        return images[0]
    return images
