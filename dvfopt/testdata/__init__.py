"""Test case definitions and data-loading utilities for deformation field experiments.

Each synthetic test case is a dict with keys:

* ``title`` — human-readable name
* ``msample`` — ``(N, 3)`` moving correspondences ``[z, y, x]``
* ``fsample`` — ``(N, 3)`` fixed correspondences ``[z, y, x]``
* ``resolution`` — ``(H, W)`` grid size

Real-data slices are loaded via :func:`load_slice`.

Usage::

    from dvfopt.testdata import SYNTHETIC_CASES, load_slice, make_deformation
"""

from dvfopt.testdata._builders import (
    canonical_2tri_2d,
    load_slice,
    make_deformation,
    make_patch_folded_dvf,
    make_random_dvf,
    save_and_summarize,
)
from dvfopt.testdata._cases import (
    CANONICAL_2TRI_2D_KEYS,
    RANDOM_DVF_CASES,
    REAL_DATA_SLICES,
    SYNTHETIC_CASES,
)

__all__ = [
    "CANONICAL_2TRI_2D_KEYS",
    "RANDOM_DVF_CASES",
    "REAL_DATA_SLICES",
    "SYNTHETIC_CASES",
    "canonical_2tri_2d",
    "load_slice",
    "make_deformation",
    "make_patch_folded_dvf",
    "make_random_dvf",
    "save_and_summarize",
]
