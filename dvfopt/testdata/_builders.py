"""Deformation field builders — functions that assemble test inputs.

These depend on both ``dvfopt`` (DVF generation, Jacobian computation) and
``dvfopt.laplacian`` (correspondence-based interpolation).
"""

import contextlib
import io
import warnings

import numpy as np

from dvfopt.dvf import generate_random_dvf, scale_dvf
from dvfopt.laplacian import solveLaplacianFromCorrespondences
from dvfopt.testdata._cases import (
    CANONICAL_2TRI_2D_KEYS,
    RANDOM_DVF_CASES,
    SYNTHETIC_CASES,
)


def make_deformation(case_key):
    """Build a ``(3, 1, H, W)`` deformation field from a synthetic test case.

    Uses Laplacian interpolation from correspondences.  Returns
    ``(deformation, msample, fsample)``.
    """
    case = SYNTHETIC_CASES[case_key]
    # SYNTHETIC_CASES holds shared numpy arrays at module scope; copy so
    # downstream code can mutate freely without poisoning the cache.
    ms = np.asarray(case["msample"]).copy()
    fs = np.asarray(case["fsample"]).copy()
    H, W = case["resolution"]
    deformation = solveLaplacianFromCorrespondences((1, H, W), ms, fs)
    return deformation, ms, fs


def make_random_dvf(case_key):
    """Build a ``(3, 1, H, W)`` random DVF from a :data:`RANDOM_DVF_CASES` entry.

    Returns the deformation array.
    """
    case = RANDOM_DVF_CASES[case_key]
    dvf = generate_random_dvf(case["original_shape"], case["max_magnitude"], case["seed"])
    if case["new_size"] is not None:
        dvf = scale_dvf(dvf, case["new_size"])
    return dvf


def make_patch_folded_dvf(shape=(128, 128), *, n_patches=5, patch=8, amp=2.0, base_amp=0.3, seed=0):
    """Smooth random 2D field (``base_amp`` px) with ``n_patches`` square patches
    folded by a slope-``amp`` flip along a random direction (det = 1 - amp inside).
    Returns ``(3, 1, H, W)`` float64 with ``dz == 0``; seeded. The GUI's
    "New random folded field" source.
    """
    from scipy.ndimage import gaussian_filter

    H, W = shape
    patch = min(patch, H, W)
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.standard_normal((2, H, W)), sigma=(0, 8, 8))
    base *= base_amp / max(float(np.abs(base).max()), 1e-12)
    yy, xx = np.mgrid[0:patch, 0:patch] - (patch - 1) / 2
    for _ in range(n_patches):
        y0 = int(rng.integers(0, H - patch + 1))
        x0 = int(rng.integers(0, W - patch + 1))
        th = rng.uniform(0, 2 * np.pi)
        dy, dx = np.sin(th), np.cos(th)
        proj = yy * dy + xx * dx
        base[0, y0 : y0 + patch, x0 : x0 + patch] -= amp * proj * dy
        base[1, y0 : y0 + patch, x0 : x0 + patch] -= amp * proj * dx
    out = np.zeros((3, 1, H, W))
    out[1:, 0] = base
    return out


def load_slice(
    slice_idx,
    scale_factor=1.0,
    mpoints_path="data/corrected_correspondences_count_touching/mpoints.npy",
    fpoints_path="data/corrected_correspondences_count_touching/fpoints.npy",
):
    """Load a real-data slice and compute its deformation field.

    Parameters
    ----------
    slice_idx : int
    scale_factor : float
    mpoints_path, fpoints_path : str

    Returns
    -------
    deformation : ndarray, shape ``(3, 1, H, W)``
    mpoints : ndarray, shape ``(N, 3)``
    fpoints : ndarray, shape ``(N, 3)``
    """
    msample = np.load(mpoints_path)
    fsample = np.load(fpoints_path)

    mask_m = msample[:, 0] == slice_idx
    mask_f = fsample[:, 0] == slice_idx

    mpoints = msample[mask_m].copy()
    fpoints = fsample[mask_f].copy()
    mpoints[:, 0] = 0
    fpoints[:, 0] = 0

    H_full, W_full = 320, 456
    H_new = int(H_full * scale_factor)
    W_new = int(W_full * scale_factor)

    scaled_m = np.round(mpoints * scale_factor).astype(int)
    scaled_f = np.round(fpoints * scale_factor).astype(int)

    deformation = solveLaplacianFromCorrespondences((1, H_new, W_new), scaled_m, scaled_f)

    return deformation, scaled_m, scaled_f


def _silent_make_deformation(case_key):
    """Build a synthetic deformation with stdout redirected — used by the
    canonical-suite loader so callers can call it from notebooks without
    the laplacian solver's progress chatter."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        deformation, _, _ = make_deformation(case_key)
    return deformation


def canonical_2tri_2d(*, with_meta=True):
    """Yield the canonical 2D 2-triangle benchmark suite.

    Six synthetic correspondence-based cases promoted from notebook 14
    (`14_l1-warmstart-2d-cases.ipynb`) — see
    :data:`dvfopt.testdata._cases.CANONICAL_2TRI_2D_KEYS` for the curated list.

    Each iteration yields ``(name, phi_2hw, meta)`` where:

    * ``name`` is the case key (e.g. ``'01a_10x10_crossing'``).
    * ``phi_2hw`` is a ``(2, H, W)`` ``float64`` array with channels
      ``[dy, dx]`` — the convention used by every 2-triangle solver.
    * ``meta`` is a dict with the initial-fold stats and provenance:
      ``init_n_neg``, ``init_min_T``, ``shape``, ``title``,
      ``msample``, ``fsample``.

    Use ``with_meta=False`` to get just ``(name, phi_2hw)`` pairs.

    Examples
    --------
    >>> from dvfopt.testdata import canonical_2tri_2d
    >>> for name, phi, meta in canonical_2tri_2d():
    ...     print(name, phi.shape, meta['init_n_neg'])
    """
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

    out = []
    for key in CANONICAL_2TRI_2D_KEYS:
        deformation = _silent_make_deformation(key)
        # (3, 1, H, W) -> (2, H, W) with channels [dy, dx]
        phi = deformation[1:, 0].astype(np.float64).copy()
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        min_T = float(min(T1.min(), T2.min()))
        case = SYNTHETIC_CASES[key]
        meta = dict(
            shape=phi.shape[1:],
            title=case['title'],
            msample=np.asarray(case['msample']).copy(),
            fsample=np.asarray(case['fsample']).copy(),
            init_n_neg=n_neg,
            init_min_T=min_T,
        )
        out.append((key, phi, meta) if with_meta else (key, phi))
    return out


def save_and_summarize(deformation, save_path):
    """Save a deformation field and print a one-line summary.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, 1, H, W)``
    save_path : str
    """
    from dvfopt.jacobian.sitk_jdet import sitk_jacobian_determinant

    np.save(save_path, deformation)
    J = sitk_jacobian_determinant(deformation)
    neg = int(np.sum(J <= 0))
    H, W = deformation.shape[2], deformation.shape[3]
    print(f"  {save_path}  |  {H}\u00d7{W}  |  neg Jdet: {neg}  |  min Jdet: {np.min(J):.4f}")
