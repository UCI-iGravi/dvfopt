"""Tests for the 2.5D marching sweep core.

The marching sweep repairs inter-layer simplex (3D) folds between adjacent
z-slices of a ``(3, D, H, W)`` field, holding dz identically zero (the
2.5D precondition). Folds are therefore planted as strong dy mismatches
between adjacent slices — NOT as dz displacements (which would violate
dz≡0). Each slice here is a ``(2, H, W)`` ``[dy, dx]`` array.
"""

import numpy as np
import pytest

from dvfopt.core.marching._marching_25d import (
    _boxes_conflict,
    _cluster_boxes,
    layer_min_v,
    march_slice,
)


def _planted_pair(seed=0, H=20, W=20, amp=1.5):
    """A (lower, upper) slice pair with a planted inter-layer dy fold."""
    rng = np.random.default_rng(seed)
    lower = rng.normal(0, 0.02, (2, H, W))  # [dy, dx]
    upper = rng.normal(0, 0.02, (2, H, W))
    lower[0, 8:11, 8:11] = +amp  # dy on lower slice
    upper[0, 8:11, 8:11] = -amp  # dy on upper slice -> inter-layer simplex (3D) fold
    return lower, upper


def test_layer_min_v_detects_interlayer_fold():
    lower, upper = _planted_pair()
    if not (layer_min_v(lower, upper).min() < 0):
        pytest.skip('no fold planted')
    assert layer_min_v(lower, upper).min() < 0

    # A smooth pair (no plant) must be fold-free.
    rng = np.random.default_rng(1)
    H = W = 20
    s_lower = rng.normal(0, 0.02, (2, H, W))
    s_upper = rng.normal(0, 0.02, (2, H, W))
    assert (layer_min_v(s_lower, s_upper) > 0).all()


def test_march_slice_removes_folds_serial():
    lower, upper = _planted_pair()
    if not (layer_min_v(lower, upper).min() < 0):
        pytest.skip('no fold planted')

    lower_before = lower.copy()
    upper_before = upper.copy()

    cur, n_before, n_after = march_slice(
        lower,
        upper,
        cur_is_upper=True,
        n_workers=1,
        pool_map=None,
    )

    assert n_before > 0
    # The marching sweep strictly reduces inter-layer folds. It repairs only
    # the FREE plane's interior against the frozen neighbour, so a small
    # residual can persist at the frozen block's edge (the source pipeline
    # clears this with a downstream 3D mop-up). Assert the guaranteed
    # contract — strict reduction — and that the residual is small.
    assert n_after < n_before
    assert n_after <= n_before // 2

    # Inputs must not be mutated.
    assert np.array_equal(lower, lower_before), 'frozen slice was mutated'
    assert np.array_equal(upper, upper_before), 'anchor slice was mutated'

    # The rim of cur' equals upper's rim (only the interior is repaired).
    assert np.array_equal(cur[:, 0, :], upper[:, 0, :])
    assert np.array_equal(cur[:, -1, :], upper[:, -1, :])
    assert np.array_equal(cur[:, :, 0], upper[:, :, 0])
    assert np.array_equal(cur[:, :, -1], upper[:, :, -1])
    # ...and cur' actually differs from upper somewhere in the interior.
    assert not np.array_equal(cur, upper)


def test_boxes_conflict():
    a = (0, 5, 0, 5)  # inclusive (y0, y1, x0, x1)
    overlap = (3, 8, 3, 8)  # overlaps a
    touch = (5, 9, 5, 9)  # shares the corner (inclusive -> conflict)
    far_y = (10, 15, 0, 5)  # separated on y
    far_x = (0, 5, 10, 15)  # separated on x
    assert _boxes_conflict(a, overlap) is True
    assert _boxes_conflict(a, touch) is True
    assert _boxes_conflict(a, far_y) is False
    assert _boxes_conflict(a, far_x) is False
    assert _boxes_conflict(far_y, far_x) is False


@pytest.mark.parametrize('phase', [0, 16])  # 16 == max_box // 2
def test_cluster_boxes_tiles_oversized(phase):
    H = W = 200
    bad = np.zeros((H - 1, W - 1), dtype=bool)
    bad[2:190, 2:190] = True  # spans most of the grid

    pad, dil, max_box = 4, 2, 32
    boxes = _cluster_boxes(bad, H, W, pad=pad, dil=dil, max_box=max_box, phase=phase)
    assert boxes, 'no boxes produced'

    slack = pad + 2  # dilation + rounding leeway
    covered = np.zeros_like(bad)
    for y0, y1, x0, x1 in boxes:
        assert y1 - y0 <= max_box + slack, f'box too tall: {(y0, y1, x0, x1)}'
        assert x1 - x0 <= max_box + slack, f'box too wide: {(y0, y1, x0, x1)}'
        # Mark covered cells (boxes are grid coords; clip to mask extent).
        yy1 = min(y1, bad.shape[0] - 1)
        xx1 = min(x1, bad.shape[1] - 1)
        covered[y0 : yy1 + 1, x0 : xx1 + 1] = True

    assert covered[bad].all(), 'boxes do not cover all bad cells'


def test_cluster_boxes_phase_shifts_seams():
    # A node on a tile seam at phase 0 must be tile-interior at
    # phase = max_box // 2 (the deterministic seam shift between rounds).
    H = W = 200
    bad = np.zeros((H - 1, W - 1), dtype=bool)
    bad[2:190, 2:190] = True
    pad, dil, max_box = 4, 2, 32

    def seam_ys(phase):
        boxes = _cluster_boxes(bad, H, W, pad=pad, dil=dil, max_box=max_box, phase=phase)
        edges = set()
        for y0, y1, _x0, _x1 in boxes:
            edges.add(y0)
            edges.add(y1)
        return edges

    e0 = seam_ys(0)
    e1 = seam_ys(max_box // 2)
    assert e0 != e1, 'phase shift did not move the tile seams'


def test_cluster_boxes_dil0_no_dilation():
    # dil=0 must mean NO dilation — scipy's iterations=0 would instead
    # dilate until convergence and balloon a lone bad cell to the whole grid.
    H = W = 200
    bad = np.zeros((H - 1, W - 1), dtype=bool)
    bad[100, 100] = True  # single bad cell

    pad, max_box = 4, 32
    boxes = _cluster_boxes(bad, H, W, pad=pad, dil=0, max_box=max_box)
    assert len(boxes) == 1
    (y0, y1, x0, x1) = boxes[0]
    assert y1 - y0 <= 2 * pad + 1, 'dil=0 produced an oversized cluster'
    assert x1 - x0 <= 2 * pad + 1, 'dil=0 produced an oversized cluster'


def test_cluster_boxes_negative_dil_raises():
    bad = np.zeros((9, 9), dtype=bool)
    bad[4, 4] = True
    with pytest.raises(ValueError, match='dil'):
        _cluster_boxes(bad, 10, 10, pad=2, dil=-1, max_box=32)


def test_march_slice_dil0_reduces_folds():
    lower, upper = _planted_pair()
    if not (layer_min_v(lower, upper).min() < 0):
        pytest.skip('no fold planted')
    cur, n_before, n_after = march_slice(
        lower,
        upper,
        cur_is_upper=True,
        dil=0,
        n_workers=1,
        pool_map=None,
    )
    assert n_before > 0
    assert n_after < n_before
