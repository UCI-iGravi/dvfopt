"""Displacement-field import/export (``.npy``/``.npz`` + SimpleITK).

Library-level field I/O usable without the ``[gui]`` extra. The
extension-dispatching :func:`load_dvf` / :func:`save_dvf` are the entry
points the CLI and any non-GUI caller use; the ``*_sitk`` helpers handle
NIfTI/MetaImage/NRRD specifically.

Channel/axis convention mirrors :mod:`dvfopt.jacobian.sitk_jdet` (the
package's single source of truth): the canonical numpy layout is
``(3, D, H, W)`` with channels ``[dz, dy, dx]``; sitk vector images store
``(D, H, W, 3)`` arrays with components ``[dx, dy, dz]``. Conversion is a
``(1, 2, 3, 0)`` transpose plus a ``[2, 1, 0]`` component reorder — after
mapping the stored PHYSICAL vectors into index space through the image's
direction matrix and spacing (:func:`dvf_from_sitk_image`), which is the
identity for files this module writes and is NOT for real registration warps.

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


def dvf_from_sitk_image(img) -> np.ndarray:
    """Convert a SimpleITK 2-/3-component vector image to the canonical
    ``(3, D, H, W)`` ``[dz, dy, dx]`` float64 layout, in INDEX space.

    ITK stores displacement vectors in physical space (mm, LPS) on an image
    with per-axis ``spacing`` and a ``direction`` matrix ``D`` (index axes ->
    physical axes), so the index-space displacement is ``D^-1 . v / spacing``.
    An image with identity direction and unit spacing — what
    :func:`save_dvf_sitk` writes — therefore loads exactly as its raw array.
    Real registration warps are not that: the cohort ANTs SyN warps carry a
    signed-permutation direction, and ignoring it reads 4667 spurious 3D
    folds on a warp that is diffeomorphic by construction (0 with it).

    Accepts 3-component 3D and 2-component 2D vector images (the latter map
    to a single-slice volume with ``dz = 0``); raises ``ValueError`` for
    anything else (e.g. scalar images).
    """
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img)  # (..., ncomp); array axes = index axes reversed
    ncomp = img.GetNumberOfComponentsPerPixel()
    dim = img.GetDimension()
    if ncomp not in (2, 3) or ncomp != dim or arr.ndim != dim + 1:
        raise ValueError(
            f'not a 2/3-component displacement field: array shape {arr.shape}, '
            f'{ncomp} component(s) per pixel'
        )
    D = np.asarray(img.GetDirection(), dtype=np.float64).reshape(dim, dim)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    idx = arr.astype(np.float64)
    if not (np.array_equal(D, np.eye(dim)) and np.all(spacing == 1.0)):
        idx = np.einsum('ab,...b->...a', np.linalg.inv(D), idx) / spacing
    # components are per ITK index axis (x, y[, z]); canonical channels are the reverse
    if dim == 3:
        return np.ascontiguousarray(np.moveaxis(idx[..., ::-1], -1, 0))
    H, W = arr.shape[:2]
    vol = np.zeros((3, 1, H, W), dtype=np.float64)
    vol[1, 0] = idx[..., 1]  # dy
    vol[2, 0] = idx[..., 0]  # dx
    return vol


def load_dvf_sitk(path) -> np.ndarray:
    """Load a displacement field into the canonical ``(3, D, H, W)``
    ``[dz, dy, dx]`` float64 layout (index space — see
    :func:`dvf_from_sitk_image` for the geometry handling and what it accepts).
    """
    import SimpleITK as sitk

    return dvf_from_sitk_image(sitk.ReadImage(str(path)))


def save_dvf_sitk(path, vol) -> None:
    """Write ``(3, D, H, W)`` ``[dz, dy, dx]`` as a 3-component vector image."""
    import SimpleITK as sitk

    vol = np.asarray(vol, dtype=np.float64)
    if vol.ndim != 4 or vol.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W); got {vol.shape}')
    arr = np.transpose(vol, (1, 2, 3, 0))[..., [2, 1, 0]]  # -> (D,H,W,3) [dx,dy,dz]
    sitk.WriteImage(sitk.GetImageFromArray(arr, isVector=True), str(path))


def load_dvf(path) -> np.ndarray:
    """Load a displacement field from ``.npy``/``.npz`` or a SimpleITK format.

    ``.npz`` must hold exactly one array. SimpleITK formats return the
    canonical ``(3, D, H, W)`` ``[dz, dy, dx]`` layout via
    :func:`load_dvf_sitk`; numpy files are returned as stored (any layout
    ``dvfopt.validation.validate_dvf`` accepts), as float64.
    """
    lower = str(path).lower()
    if lower.endswith('.npy'):
        return np.asarray(np.load(path), dtype=np.float64)
    if lower.endswith('.npz'):
        with np.load(path) as z:
            if len(z.files) != 1:
                raise ValueError(f'{path}: .npz holds {len(z.files)} arrays; expected exactly 1')
            return np.asarray(z[z.files[0]], dtype=np.float64)
    if is_sitk_path(lower):
        return load_dvf_sitk(path)
    raise ValueError(f'unsupported DVF format: {path} (want .npy/.npz or one of {SITK_EXTENSIONS})')


def save_dvf(path, vol) -> None:
    """Save a displacement field to ``.npy`` or a SimpleITK format."""
    lower = str(path).lower()
    if lower.endswith('.npy'):
        np.save(path, np.asarray(vol))
        return
    if is_sitk_path(lower):
        save_dvf_sitk(path, vol)
        return
    raise ValueError(f'unsupported DVF format: {path} (want .npy or one of {SITK_EXTENSIONS})')
