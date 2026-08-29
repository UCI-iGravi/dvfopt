"""Loaders for real fields already on disk (all gitignored data).

* cohort Laplacian slice (mechanism 1, real) and ANTs SyN warp slice
  (mechanism 4, real) from ``data/dvfs/brain25_cohort_corrected/``;
* any saved field (e.g. a VoxelMorph / TransMorph notebook output) for
  mechanism 3.
"""

from pathlib import Path

import numpy as np

from dvf_origins import ROOT, pack2d

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
    cohort Laplacian field, in voxel units.

    ITK stores the warp as physical (mm, LPS) displacement vectors on an image
    whose direction matrix ``D`` maps index axes ``(i, j, k)`` to physical
    axes. The cohort files have ``D = [[0,0,-1],[1,0,0],[0,-1,0]]`` — a signed
    permutation — so dividing by spacing alone mixes the components. The
    index-space displacement is ``D^-1 · phys / spacing``. The cohort Laplacian
    field is laid out ``(3, i, j, k)`` (ANTsPy index order; ``(528, 320, 456)``
    for B0039) while SimpleITK's array is ``(k, j, i, comp)``, so the plane
    ``i = z`` is extracted and its axes reversed to ``(j, k) = (H, W)``.

    Verified on B0039: the naive conversion gives 4667 voxels with 3D Jdet
    <= 0 on a warp that is diffeomorphic by construction; this one gives 0
    (min 0.18). The through-plane component is dropped
    (``meta['dz_max_dropped']``). A large in-plane fold count here means a
    convention mismatch, not folds.
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(_cohort_file(brain, variant, 'ants_warp_0.nii.gz')))
    n_i, n_j, n_k = img.GetSize()
    if not 0 <= z < n_i:
        raise ValueError(f'z={z} out of range for D={n_i}')
    D = np.array(img.GetDirection()).reshape(3, 3)
    spacing = np.array(img.GetSpacing())  # per index axis (i, j, k)
    plane = sitk.RegionOfInterest(img, [1, n_j, n_k], [z, 0, 0])
    phys = sitk.GetArrayFromImage(plane)[:, :, 0, :]  # (k, j, comp) physical x,y,z
    phys = phys.transpose(1, 0, 2).astype(np.float64)  # (j, k, comp) = (H, W, comp)
    idx = np.einsum('ab,hwb->ahw', np.linalg.inv(D), phys) / spacing[:, None, None]  # [d_i,d_j,d_k]
    meta = dict(
        source='real',
        tool='ANTs SyN (RegTools)',
        brain=brain,
        z=z,
        variant=variant,
        direction=np.round(D, 6).tolist(),
        spacing_ijk=spacing.tolist(),
        dz_max_dropped=float(np.abs(idx[0]).max()),
    )
    return pack2d(idx[1], idx[2]), meta


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
