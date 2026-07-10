"""Displacement-field import/export via SimpleITK (NIfTI/MetaImage/NRRD).

Channel/axis convention mirrors :mod:`dvfopt.jacobian.sitk_jdet` (the
package's single source of truth): the canonical numpy layout is
``(3, D, H, W)`` with channels ``[dz, dy, dx]``; sitk vector images store
``(D, H, W, 3)`` arrays with components ``[dx, dy, dz]``. Conversion is a
``(1, 2, 3, 0)`` transpose plus a ``[2, 1, 0]`` component reorder.

SimpleITK is optional at runtime: :func:`sitk_available` gates the GUI's
file-dialog filters; the load/save functions import lazily.
"""

from __future__ import annotations

import numpy as np

SITK_EXTENSIONS = ('.nii', '.nii.gz', '.mha', '.mhd', '.nrrd')


def sitk_available() -> bool:
    """True when SimpleITK can be imported."""
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        return False
    return True


def is_sitk_path(path) -> bool:
    """True when ``path`` has a SimpleITK-handled extension."""
    lower = str(path).lower()
    return any(lower.endswith(ext) for ext in SITK_EXTENSIONS)


def load_dvf_sitk(path) -> np.ndarray:
    """Load a displacement field into the canonical ``(3, D, H, W)``
    ``[dz, dy, dx]`` float64 layout.

    Accepts 3-component 3D vector images and 2-component 2D vector images
    (mapped to a single-slice volume with ``dz = 0``). Raises ``ValueError``
    for anything else (e.g. scalar images).
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    ncomp = img.GetNumberOfComponentsPerPixel()
    if ncomp == 3 and arr.ndim == 4:
        arr = arr[..., [2, 1, 0]]  # components [dx,dy,dz] -> [dz,dy,dx]
        return np.ascontiguousarray(np.transpose(arr, (3, 0, 1, 2))).astype(np.float64)
    if ncomp == 2 and arr.ndim == 3:
        H, W = arr.shape[:2]
        vol = np.zeros((3, 1, H, W), dtype=np.float64)
        vol[1, 0] = arr[..., 1]  # dy
        vol[2, 0] = arr[..., 0]  # dx
        return vol
    raise ValueError(
        f'not a 2/3-component displacement field: array shape {arr.shape}, '
        f'{ncomp} component(s) per pixel'
    )


def save_dvf_sitk(path, vol) -> None:
    """Write ``(3, D, H, W)`` ``[dz, dy, dx]`` as a 3-component vector image."""
    import SimpleITK as sitk

    vol = np.asarray(vol, dtype=np.float64)
    if vol.ndim != 4 or vol.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W); got {vol.shape}')
    arr = np.transpose(vol, (1, 2, 3, 0))[..., [2, 1, 0]]  # -> (D,H,W,3) [dx,dy,dz]
    sitk.WriteImage(sitk.GetImageFromArray(arr, isVector=True), str(path))
