"""Loaders for real fields already on disk (all gitignored data).

* cohort Laplacian slice (mechanism 1, real) and ANTs SyN warp slice
  (mechanism 4, real) from ``data/dvfs/brain25_cohort_corrected/``;
* any saved field (e.g. a VoxelMorph / TransMorph notebook output) for
  mechanism 3.
"""

from pathlib import Path

import numpy as np

from dvf_origins._common import ROOT, pack2d

COHORT = ROOT / 'data' / 'dvfs' / 'brain25_cohort_corrected'


def _slice2d(vol, z):
    """``(3, D, H, W)`` volume -> ``(3, 1, H, W)`` slice with ``dz`` zeroed;
    returns ``(phi, max |dz| dropped)``."""
    vol = np.asarray(vol)
    if vol.ndim != 4 or vol.shape[0] != 3:
        raise ValueError(f'expected a (3, D, H, W) field, got shape {vol.shape}')
    if not 0 <= z < vol.shape[1]:
        raise ValueError(f'z={z} out of range for D={vol.shape[1]}')
    phi = vol[:, z : z + 1].astype(np.float64)
    dz_max = float(np.abs(phi[0]).max())
    phi[0] = 0.0
    return phi, dz_max


def _cohort_file(brain, variant, name):
    p = COHORT / brain / variant / name
    if not p.is_file():
        raise FileNotFoundError(f'cohort file not found (data is gitignored): {p}')
    return p


def laplacian_slice(brain, z, variant='laplacian_exterior'):
    """Cohort Laplacian-refinement field, one z-slice (already dz == 0)."""
    vol = np.load(_cohort_file(brain, variant, 'laplacian_deformation_field.npz'))['arr']
    phi, dz_max = _slice2d(vol, z)
    meta = dict(
        source='real',
        tool='Laplacian (RegTools)',
        brain=brain,
        z=z,
        variant=variant,
        dz_max_dropped=dz_max,
    )
    return phi, meta


def ants_slice(brain, z, variant='laplacian_exterior'):
    """Cohort ANTs SyN warp, one z-slice on the SAME grid and layout as the
    cohort Laplacian field, in the warp image's voxel units.

    The physical -> index conversion (direction matrix and spacing) is
    ``dvfopt.io.fields.dvf_from_sitk_image``'s. What is harness-specific is
    the layout: the cohort Laplacian field is ``(3, i, j, k)`` (ANTsPy index
    order; ``(528, 320, 456)`` for B0039) while SimpleITK's array is
    ``(k, j, i)``, so the plane ``i = z`` is extracted (a ``RegionOfInterest``,
    not a full-volume conversion) and its axes reversed to ``(j, k) = (H, W)``.
    The through-plane component is dropped (``meta['dz_max_dropped']``).
    The warp must live on the template grid for ``z`` to mean the same plane
    as in the Laplacian file — the self-check asserts the shapes agree.
    """
    import SimpleITK as sitk

    from dvfopt.io.fields import dvf_from_sitk_image

    img = sitk.ReadImage(str(_cohort_file(brain, variant, 'ants_warp_0.nii.gz')))
    n_i, n_j, n_k = img.GetSize()
    if not 0 <= z < n_i:
        raise ValueError(f'z={z} out of range for D={n_i}')
    plane = sitk.RegionOfInterest(img, [1, n_j, n_k], [z, 0, 0])
    vol = dvf_from_sitk_image(plane)  # (3, k, j, 1) layout, channels [d_k, d_j, d_i]
    dy, dx = vol[1, :, :, 0].T, vol[0, :, :, 0].T  # onto the (j, k) = (H, W) grid
    meta = dict(
        source='real',
        tool='ANTs SyN (RegTools)',
        brain=brain,
        z=z,
        variant=variant,
        size_ijk=(n_i, n_j, n_k),
        spacing_ijk=list(img.GetSpacing()),
        direction=np.round(np.reshape(img.GetDirection(), (3, 3)), 6).tolist(),
        dz_max_dropped=float(np.abs(vol[2]).max()),
    )
    return pack2d(dy, dx), meta


def saved_field(path, z=0):
    """Any ``.npy`` / ``.npz`` / NIfTI / MetaImage / NRRD field (``dvfopt.io.fields.load_dvf``),
    one z-slice. Path is relative to the repo root unless absolute."""
    from dvfopt.io.fields import load_dvf

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f'saved field not found: {p}')
    phi, dz_max = _slice2d(load_dvf(p), z)
    return phi, dict(
        source='real', tool=f'saved: {p.name}', path=str(p), z=z, dz_max_dropped=dz_max
    )
