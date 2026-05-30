"""Synthetic 2D bowtie + single-cell-flip cases.

Each case is constructed from raw correspondence pairs via Laplacian
interpolation (mirroring the patterns shown in notebooks 09 and 13). All
return (3, 1, H, W) deformation arrays so they're drop-in compatible with
the existing iterative_serial signature.
"""

import numpy as np

from benchmarks.two_triangle.registry import register_case
from laplacian import solveLaplacianFromCorrespondences


def _build(H: int, W: int, msample: list, fsample: list) -> np.ndarray:
    ms = np.array(msample, dtype=int)
    fs = np.array(fsample, dtype=int)
    return solveLaplacianFromCorrespondences((1, H, W), ms, fs)


@register_case("synth2d_single_cell_flip", category="synthetic_2d", dim=2)
def synth2d_single_cell_flip():
    """Tiny 5x5 grid: two vertically-adjacent points swapped — single-cell flip."""
    dvf = _build(
        5,
        5,
        msample=[[0, 1, 2], [0, 3, 2]],
        fsample=[[0, 3, 2], [0, 1, 2]],
    )
    return dvf, {"title": "Single-cell flip (5x5)", "expected_folds": "small"}


@register_case("synth2d_horizontal_bowtie", category="synthetic_2d", dim=2)
def synth2d_horizontal_bowtie():
    """Two horizontally-displaced points crossing — classic bowtie pattern."""
    dvf = _build(
        20,
        20,
        msample=[[0, 8, 5], [0, 12, 5]],
        fsample=[[0, 12, 5], [0, 8, 5]],
    )
    return dvf, {"title": "Horizontal bowtie (20x20)", "expected_folds": "moderate"}


@register_case("synth2d_diagonal_bowtie", category="synthetic_2d", dim=2)
def synth2d_diagonal_bowtie():
    """Diagonal point swap — harder to detect with central-diff Jdet alone."""
    dvf = _build(
        20,
        20,
        msample=[[0, 8, 8], [0, 12, 12]],
        fsample=[[0, 12, 12], [0, 8, 8]],
    )
    return dvf, {"title": "Diagonal bowtie (20x20)", "expected_folds": "moderate"}


@register_case("synth2d_layered_bowtie_stack", category="synthetic_2d", dim=2)
def synth2d_layered_bowtie_stack():
    """Three stacked bowtie pairs — exercises multi-region windowing."""
    dvf = _build(
        30,
        30,
        msample=[
            [0, 5, 8],
            [0, 9, 8],
            [0, 13, 8],
            [0, 17, 8],
            [0, 21, 8],
            [0, 25, 8],
        ],
        fsample=[
            [0, 9, 8],
            [0, 5, 8],
            [0, 17, 8],
            [0, 13, 8],
            [0, 25, 8],
            [0, 21, 8],
        ],
    )
    return dvf, {"title": "Layered bowtie stack (30x30)", "expected_folds": "many"}
