"""Deformation field builders — functions that assemble test inputs.

These depend on both ``dvfopt`` (DVF generation, Jacobian computation) and
``laplacian_interp`` (correspondence-based interpolation).
"""

import numpy as np

from laplacian_interp import slice_to_slice_3d_laplacian
from dvfopt.dvf import generate_random_dvf, scale_dvf

from test_cases._cases import SYNTHETIC_CASES, RANDOM_DVF_CASES


def make_deformation(case_key):
    """Build a ``(3, 1, H, W)`` deformation field from a synthetic test case.

    Uses Laplacian interpolation from correspondences.  Returns
    ``(deformation, msample, fsample)``.
    """
    case = SYNTHETIC_CASES[case_key]
    ms, fs = case["msample"], case["fsample"]
    H, W = case["resolution"]
    fixed_sample = np.zeros((1, H, W))
    deformation, _, _, _, _ = slice_to_slice_3d_laplacian(fixed_sample, ms, fs)
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


def load_slice(slice_idx, scale_factor=1.0,
               mpoints_path="data/corrected_correspondences_count_touching/mpoints.npy",
               fpoints_path="data/corrected_correspondences_count_touching/fpoints.npy"):
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

    fixed_sample = np.zeros((1, H_new, W_new))
    deformation, _, _, _, _ = slice_to_slice_3d_laplacian(fixed_sample, scaled_m, scaled_f)

    return deformation, scaled_m, scaled_f


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
