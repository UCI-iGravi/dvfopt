import numpy as np

from research.strict_feasibility_2d.algorithms.orientation_fix import (
    canonical_signs,
    n_triangles,
)


def test_n_triangles_count():
    # 2 triangles per (H-1)*(W-1) cell
    assert n_triangles(7, 7) == 2 * 6 * 6
    assert n_triangles(10, 10) == 2 * 9 * 9
    assert n_triangles(20, 20) == 2 * 19 * 19


def test_canonical_signs_all_positive():
    s = canonical_signs(10, 10)
    assert s.shape == (n_triangles(10, 10),)
    assert np.all(s == 1.0)


def test_canonical_signs_dtype():
    s = canonical_signs(7, 7)
    assert s.dtype == np.float64
