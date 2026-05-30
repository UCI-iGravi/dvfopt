"""Synthetic 3D single-tet-flip and 6-tet-bowtie cases.

Built directly as displacement arrays rather than via correspondences —
the 3D Laplacian solver is heavier and the failure modes we want to
exercise (single-vertex folds, 6-tet bowties) are easier to specify
explicitly. Mirrors the patterns from notebooks 12a and 12c.
"""

import numpy as np

from benchmarks.two_triangle.registry import register_case


@register_case("synth3d_single_tet_flip", category="synthetic_3d", dim=3)
def synth3d_single_tet_flip():
    """5x5x5 cube where one interior vertex is shoved across its neighbour."""
    D, H, W = 5, 5, 5
    dvf = np.zeros((3, D, H, W), dtype=np.float64)
    # Shove vertex (2, 2, 2) by (-2, -2, -2) — large enough to flip
    # several incident tets.
    dvf[0, 2, 2, 2] = -2.0  # dz
    dvf[1, 2, 2, 2] = -2.0  # dy
    dvf[2, 2, 2, 2] = -2.0  # dx
    return dvf, {"title": "Single-tet vertex flip (5x5x5)", "expected_folds": "small", "dim": 3}


@register_case("synth3d_6tet_bowtie", category="synthetic_3d", dim=3)
def synth3d_6tet_bowtie():
    """Two adjacent vertices swapped — exercises the 6-tet decomposition.

    Pattern: vertex (2, 2, 2) and (2, 2, 3) swap their displacements,
    creating a 3D analog of the 2D bowtie that fools central-diff Jacobians
    in some triangulations but is caught by the strict 6-tet check.
    """
    D, H, W = 6, 6, 6
    dvf = np.zeros((3, D, H, W), dtype=np.float64)
    dvf[2, 2, 2, 2] = +2.0  # vertex A: dx +2 -> lands on B
    dvf[2, 2, 2, 3] = -2.0  # vertex B: dx -2 -> lands on A
    return dvf, {"title": "6-tet bowtie (6x6x6)", "expected_folds": "moderate", "dim": 3}
