"""Loaders for real fields already on disk (all gitignored data).

* cohort Laplacian slice (mechanism 1, real) and ANTs SyN warp slice
  (mechanism 4, real) from ``data/dvfs/brain25_cohort_corrected/``;
* any saved field (e.g. a VoxelMorph / TransMorph notebook output) for
  mechanism 3.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'benchmarks') not in sys.path:  # reuse the cohort loader as the tests do
    sys.path.insert(0, str(ROOT / 'benchmarks'))


def _slice2d(vol, z):
    phi = np.asarray(vol, dtype=np.float64)[:, z : z + 1].copy()
    dz_max = float(np.abs(phi[0]).max())
    phi[0] = 0.0
    return phi, dz_max


def laplacian_slice(brain, z, variant='laplacian_exterior'):
    """Cohort Laplacian-refinement field, one z-slice (already dz == 0)."""
    import benchmark_utils as bu

    phi = bu.load_cohort_section(brain, z, variant)
    meta = dict(source='real', tool='Laplacian (RegTools)', brain=brain, z=z, variant=variant)
    return np.asarray(phi, dtype=np.float64), meta


def ants_slice(brain, z, variant='laplacian_exterior'):
    """Cohort ANTs SyN warp, one z-slice, converted from physical (mm) to voxel
    units per axis. The in-plane ``(dy, dx)`` is kept; ``dz`` is dropped
    (recorded as ``meta['dz_max_dropped']``).

    Convention caveat: ITK stores warps in physical LPS space; the axis
    reordering is ``dvfopt.io.fields.load_dvf_sitk``'s. Zero in-plane folds
    is the expected outcome for a SyN warp — a large fold count here means a
    sign/axis convention mismatch, not folds.
    """
    import benchmark_utils as bu
    import SimpleITK as sitk

    from dvfopt.io.fields import load_dvf_sitk

    p = bu.cohort_dir() / brain / variant / 'ants_warp_0.nii.gz'
    if not p.is_file():
        raise FileNotFoundError(f'ANTs warp not found (data is gitignored): {p}')
    vol = load_dvf_sitk(p)  # (3, D, H, W) [dz, dy, dx], physical units
    sx, sy, sz = sitk.ReadImage(str(p)).GetSpacing()
    vol /= np.array([sz, sy, sx], dtype=np.float64)[:, None, None, None]
    phi, dz_max = _slice2d(vol, z)
    meta = dict(
        source='real',
        tool='ANTs SyN (RegTools)',
        brain=brain,
        z=z,
        variant=variant,
        spacing_zyx=(sz, sy, sx),
        dz_max_dropped=dz_max,
    )
    return phi, meta


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
