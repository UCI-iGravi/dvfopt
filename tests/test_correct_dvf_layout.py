"""``correct_dvf`` / ``dvfopt correct`` on the canonical ``(3, 1, H, W)`` layout
(regression for the ``phi.shape[1:]`` shape inference — see the CHANGELOG)."""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt import correct_dvf
from dvfopt.cli import main as cli_main
from dvfopt.constraints import infer_shape
from tests.conftest import planted_fold


def _canonical(phi2):  # (2, H, W) -> (3, 1, H, W) [dz, dy, dx], dz = 0
    return np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])[:, None]


def test_infer_shape_follows_the_family_dimensionality():
    assert infer_shape('simplex', np.zeros((3, 1, 12, 14))) == (12, 14)
    assert infer_shape('bilinear', np.zeros((2, 12, 14))) == (12, 14)
    assert infer_shape('simplex_3d', np.zeros((3, 5, 12, 14))) == (5, 12, 14)
    assert infer_shape('jdet', np.zeros((3, 1, 12, 14)).tolist()) == (12, 14)  # array-likes too
    with pytest.raises(ValueError, match='unknown constraint'):
        infer_shape('nope', np.zeros((2, 12, 14)))


@pytest.mark.parametrize(
    'constraint,strategy,objective',
    [('simplex', 'auto', 'l1'), ('bilinear', 'barrier', 'l2'), ('jdet', 'barrier', 'l2')],
)
def test_correct_dvf_accepts_canonical_layout(constraint, strategy, objective):
    phi = _canonical(planted_fold(12, 14, seed=1, scale=0.4))
    res = correct_dvf(phi, constraint=constraint, strategy=strategy, objective=objective)
    assert res.init_n_neg > 0 and res.final_n_neg < res.init_n_neg
    out = np.asarray(res.corrected)
    assert out.shape == phi.shape  # the input layout comes back, dz untouched
    np.testing.assert_array_equal(out[0], phi[0])


def test_bare_canonical_and_explicit_shape_agree():
    phi2 = planted_fold(12, 14, seed=1, scale=0.4)
    kw = dict(constraint='simplex', strategy='slsqp', objective='l2')
    a = correct_dvf(phi2, **kw)
    b = correct_dvf(_canonical(phi2), **kw)
    c = correct_dvf(_canonical(phi2), shape=(12, 14), **kw)  # what cli --pipeline slices passes
    assert a.corrected.shape == (2, 12, 14) and b.corrected.shape == (3, 1, 12, 14)
    np.testing.assert_allclose(a.corrected, b.corrected[1:, 0])
    np.testing.assert_array_equal(b.corrected, c.corrected)


def test_cli_correct_accepts_canonical_layout(tmp_path):
    src, dst = tmp_path / 'in.npy', tmp_path / 'out.npy'
    phi = _canonical(planted_fold(12, 14, seed=1, scale=0.4))
    np.save(src, phi)
    args = ['correct', str(src), str(dst), '--constraint', 'bilinear']
    assert cli_main([*args, '--strategy', 'barrier', '--objective', 'l2']) == 0  # feasible
    out = np.load(dst)
    assert out.shape == phi.shape and not np.array_equal(out, phi)
