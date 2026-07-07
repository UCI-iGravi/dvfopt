"""Tests for the whole-slice GPU PHR-ALM 2D untangler.

The whole file skips cleanly when torch is absent (torch is only in the
``benchmarks`` extra). Guards the critical triangle-area sign convention
(must match ``tri_areas_flat`` exactly) and that the untangler reduces
the planted fold count (it is a seed, so it need not reach 0).
"""

import pytest

torch = pytest.importorskip('torch')

import numpy as np

from dvfopt.core.slp._gpu_untangle import _areas_torch, gpu_untangle_alm_2d
from dvfopt.core.tri_primitives import tri_areas_flat


def _n_neg(p):
    H, W = p.shape[1:]
    a = tri_areas_flat(np.concatenate([p[0].ravel(), p[1].ravel()]), H, W)
    return int((a <= 0).sum())


def _planted(H, W, spots):
    phi = np.zeros((2, H, W))
    for r, c in spots:
        phi[0, r, c] = 1.3
        phi[0, r, c + 1] = -1.3
    return phi


def test_areas_match_reference():
    """Torch triangle areas must match tri_areas_flat to 1e-10 (guards the
    -0.5 sign / vertex convention)."""
    H, W = 20, 24
    f = np.random.default_rng(0).normal(0, 0.3, (2, H, W))
    a_np = tri_areas_flat(np.concatenate([f[0].ravel(), f[1].ravel()]), H, W)
    dy = torch.tensor(f[0], dtype=torch.float64)
    dx = torch.tensor(f[1], dtype=torch.float64)
    t1, t2 = _areas_torch(dy, dx, torch)
    a_t = torch.cat([t1.reshape(-1), t2.reshape(-1)]).numpy()
    assert np.abs(a_t - a_np).max() < 1e-10


def test_untangle_reduces_folds():
    """The PHR-ALM untangler must strictly reduce the planted fold count
    (it is a seed, not a feasibility solver — do not assert 0)."""
    rng = np.random.default_rng(1)
    H, W = 24, 24
    phi = rng.normal(0, 0.01, (2, H, W))
    for r, c in [(8, 8), (16, 15)]:
        phi[0, r, c] = 1.3
        phi[0, r, c + 1] = -1.3
    n_in = _n_neg(phi)
    if n_in == 0:
        pytest.skip('no fold planted')
    out = gpu_untangle_alm_2d(phi, threshold=0.01, n_outer=40)
    assert out.shape == (2, H, W)
    assert _n_neg(out) < n_in
