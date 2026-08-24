"""Tests for dvfopt.jacobian.injectivity_radius + dvfopt.metrics.injectivity_stats.

The bowtie field (notebooks 13/15) is the canonical adversarial case: the
central-difference pixel Jdet stays positive (+0.40) while the cell is
geometrically folded — the diagnostics here must flag what CD misses.
The Nyquist and untracked-diagonal fields pin the two historical blind
spots (spacing-2 stencils; 2-tri tracking only one diagonal per cell).
"""

import numpy as np
import pytest

from dvfopt import injectivity_stats, jacobian_det2D
from dvfopt.jacobian import (
    cell_min_jdet_2d,
    cell_to_pixel_min,
    ift_radius_2d,
    ift_radius_3d,
)
from dvfopt.metrics import constraint_fold_stats


def bowtie(a=1.2, H=7, W=7):
    """dx[3,3] = +a, dx[3,4] = -a — folds two cells for a > 1."""
    phi = np.zeros((2, H, W))
    phi[1, 3, 3] = +a
    phi[1, 3, 4] = -a
    return phi


def test_bowtie_cd_positive_but_cell_certificate_fires():
    phi = bowtie(1.2)
    assert np.squeeze(jacobian_det2D(phi)).min() > 0  # CD misses the fold
    cm = cell_min_jdet_2d(phi)
    assert cm.min() < 0  # bilinear corner certificate catches it
    assert (cm <= 0).sum() == 2


def test_ift_radius_collapses_at_fold_pixels_only():
    r = ift_radius_2d(bowtie(1.2))
    # Tight 3-point stencils give L ~= 4.33 at the spike, so r ~= 0.046 —
    # the same scale as the true bilinear fold distance (~0.036).
    assert r[3, 3] < 0.1 and r[3, 4] < 0.1
    assert r[0, 0] >= 1.0  # calm corner keeps a >= 1 px estimate


def test_nyquist_field_is_flagged_not_saturated():
    # Period-2 fields have identically zero central differences; the tight
    # second-difference stencils must still flag them (this was a real
    # blind spot: the old double-np.gradient Hessian returned L = 0 here
    # and the radius saturated at the cap on a maximally folded field).
    phi = np.zeros((2, 21, 21))
    phi[1] = (-1.0) ** np.arange(21)
    assert cell_min_jdet_2d(phi).min() < 0  # genuinely folded
    assert (ift_radius_2d(phi) < 1.0).all()  # flagged everywhere


def test_certified_never_exceeds_pointwise_estimate_or_cap():
    phi = bowtie(0.4)
    r_ladder = ift_radius_2d(phi, max_window=4)
    r_point = ift_radius_2d(phi, max_window=0)
    assert (r_ladder <= r_point + 1e-12).all()
    assert r_ladder.max() <= 4.0


def test_identity_field_saturates_at_cap():
    assert (ift_radius_2d(np.zeros((2, 9, 9)), max_window=5) == 5.0).all()


def test_invalid_max_window_raises():
    with pytest.raises(ValueError, match='max_window'):
        ift_radius_2d(np.zeros((2, 5, 5)), max_window=-1)
    with pytest.raises(ValueError, match='max_window'):
        ift_radius_2d(np.zeros((2, 5, 5)), max_window=0.5)


def test_cell_to_pixel_min_projects_fold_to_corners():
    phi = bowtie(1.2)
    px = cell_to_pixel_min(cell_min_jdet_2d(phi), 7, 7)
    assert px.shape == (7, 7)
    assert px[3, 3] < 0  # corner of a folded cell
    assert px[0, 0] > 0


def test_cell_certificate_catches_untracked_diagonal():
    # Single-pixel push along the anti-diagonal folds only the corner
    # triangles the standard 2-tri split does not track: the field is
    # strictly 2tri-feasible yet bilinear-folded.
    phi = np.zeros((2, 5, 5))
    phi[0, 2, 2] = -0.7
    phi[1, 2, 2] = +0.7
    _, st = constraint_fold_stats(phi, constraint='2tri')
    assert st.feasible
    assert cell_min_jdet_2d(phi).min() < 0


def test_ift_radius_3d_identity_and_spike():
    assert (ift_radius_3d(np.zeros((3, 5, 7, 7)), max_window=3) == 3.0).all()
    phi = np.zeros((3, 5, 7, 7))
    phi[2, 2, 3, 3] = +1.2
    phi[2, 2, 3, 4] = -1.2
    r = ift_radius_3d(phi)
    assert r[2, 3, 3] < 0.1
    assert r[0, 0, 0] >= 1.0


def test_ift_radius_3d_rejects_single_slice():
    with pytest.raises(ValueError, match='ift_radius_2d'):
        ift_radius_3d(np.zeros((3, 1, 8, 8)))


def test_injectivity_stats_2d_and_3d():
    st = injectivity_stats(bowtie(1.2))
    assert st.min_radius < 0.1 and 0 < st.frac_subpixel < 1
    assert st.cell_min_jdet < 0 and st.n_cells_nonpos == 2
    assert st.max_window == 8

    st3 = injectivity_stats(np.zeros((3, 4, 6, 6)))
    assert st3.min_radius == 8.0 and st3.frac_subpixel == 0.0
    assert st3.cell_min_jdet is None and st3.n_cells_nonpos is None


def test_injectivity_stats_accepts_canonical_single_slice_layout():
    phi = np.zeros((3, 1, 7, 7))
    phi[2, 0, 3, 3] = +1.2
    phi[2, 0, 3, 4] = -1.2
    st = injectivity_stats(phi)  # (3, 1, H, W) routes to the 2D path
    assert st.n_cells_nonpos == 2


def test_injectivity_stats_rejects_bad_input():
    with pytest.raises(ValueError):  # non-finite
        injectivity_stats(np.full((2, 6, 6), np.nan))
    with pytest.raises(ValueError):  # (2, D>1, H, W) is no known layout
        injectivity_stats(np.zeros((2, 4, 8, 8)))
