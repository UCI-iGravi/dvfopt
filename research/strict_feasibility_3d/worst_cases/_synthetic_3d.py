"""Small synthetic 3D fold cases for ``research/strict_feasibility_3d/``.

The 2D equivalent is ``research/strict_feasibility_2d/worst_cases/
_build_adversarial.py``. These cases are deliberately small so the LP
problem is tractable (full LP at 20x20x20 has ~24k decision vars,
which HiGHS solves in seconds).
"""
from __future__ import annotations

import numpy as np


def bowtie_3d_cube(size: int = 10, magnitude: float = 1.5):
    """3D bowtie analog of the canonical 2D shoelace bowtie.

    A single voxel and its x-neighbour swap displacement values, which
    crashes 6 tetrahedra in the cube around the central cell.
    """
    phi = np.zeros((3, size, size, size), dtype=np.float64)
    cz, cy, cx = size // 2, size // 2, size // 2
    # Swap dx values between (cz, cy, cx) and (cz, cy, cx+1).
    phi[2, cz, cy, cx] = +magnitude
    phi[2, cz, cy, cx + 1] = -magnitude
    return phi


def dense_random_3d(size: int = 12, magnitude: float = 0.6, seed: int = 0):
    """Random small-displacement field with enough magnitude to fold
    a few cells. Useful for unit-test-level smoke checks."""
    rng = np.random.default_rng(seed)
    return magnitude * rng.standard_normal((3, size, size, size)).astype(np.float64)
