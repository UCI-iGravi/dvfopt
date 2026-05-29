"""Test case definitions and data-loading utilities for deformation field experiments.

Each synthetic test case is a dict with keys:

* ``title`` — human-readable name
* ``msample`` — ``(N, 3)`` moving correspondences ``[z, y, x]``
* ``fsample`` — ``(N, 3)`` fixed correspondences ``[z, y, x]``
* ``resolution`` — ``(H, W)`` grid size

Real-data slices are loaded via :func:`load_slice`.

Usage::

    from test_cases import SYNTHETIC_CASES, load_slice, make_deformation
"""

from test_cases._cases import (
    SYNTHETIC_CASES,
    RANDOM_DVF_CASES,
    REAL_DATA_SLICES,
    CANONICAL_2TRI_2D_KEYS,
)
from test_cases._builders import (
    make_deformation,
    make_random_dvf,
    load_slice,
    save_and_summarize,
    canonical_2tri_2d,
)

__all__ = [
    "SYNTHETIC_CASES",
    "RANDOM_DVF_CASES",
    "REAL_DATA_SLICES",
    "CANONICAL_2TRI_2D_KEYS",
    "make_deformation",
    "make_random_dvf",
    "load_slice",
    "save_and_summarize",
    "canonical_2tri_2d",
]
