"""Tests for the packaged 2-tri SLP strategy (dvfopt.SLPStrategy).

The auto_slp champion, promoted from research into the dvfopt package.
Covers both dispatch paths (global small-slice, cluster large-slice) and
constraint compatibility.
"""

import numpy as np
import pytest

from dvfopt import L1Objective, SLPStrategy, Solver, TriConstraint2D
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


def test_slp_global_path_small_slice():
    """Small slice (<=5000 px) routes to the global slp_iter; reaches 0."""
    phi = _planted(16, 16, [(7, 7)])
    if _n_neg(phi) == 0:
        pytest.skip('no fold planted')
    res = Solver(
        constraint=TriConstraint2D(shape=(16, 16)),
        objective=L1Objective(eps=1e-4),
        strategy=SLPStrategy(),
        threshold=0.01,
    ).fit(phi)
    assert _n_neg(res.corrected) == 0


def test_slp_cluster_path_serial():
    """Large slice (>5000 px) routes to the cluster path; n_workers=1 keeps
    it serial (no pool spawn) for a fast test."""
    phi = _planted(80, 80, [(20, 20), (60, 60)])
    if _n_neg(phi) == 0:
        pytest.skip('no fold planted')
    res = Solver(
        constraint=TriConstraint2D(shape=(80, 80)),
        objective=L1Objective(eps=1e-4),
        strategy=SLPStrategy(n_workers=1),
        threshold=0.01,
    ).fit(phi)
    assert _n_neg(res.corrected) == 0


def test_slp_is_2d_only():
    """SLPStrategy must reject a 3D constraint (supports_3d=False)."""
    from dvfopt import Tet6Constraint3D
    from dvfopt.exceptions import IncompatibleConstraintError

    with pytest.raises((IncompatibleConstraintError, Exception)):
        Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L1Objective(eps=1e-4),
            strategy=SLPStrategy(),
            threshold=0.01,
        )


def test_slp_registered_label():
    """The strategy is registered under 'slp' (string-based construction)."""
    from dvfopt.strategies import make_strategy

    s = make_strategy('slp')
    assert isinstance(s, SLPStrategy)


def _accuracy_tag(info):
    """Best-effort extraction of the 'accuracy' tag from a SolveInfo."""
    extras = getattr(info, 'extras', None)
    if isinstance(extras, dict) and 'accuracy' in extras:
        return extras['accuracy']
    for p in getattr(info, 'phases', []) or []:
        pe = getattr(p, 'extras', None)
        if isinstance(pe, dict) and 'accuracy' in pe:
            return pe['accuracy']
    return None


def test_slp_accuracy_max_reaches_feasibility():
    """accuracy='max' prepends the GPU untangler and still reaches 0 folds."""
    pytest.importorskip('torch')
    phi = _planted(24, 24, [(11, 11)])
    if _n_neg(phi) == 0:
        pytest.skip('no fold planted')
    res = Solver(
        constraint=TriConstraint2D(shape=(24, 24)),
        objective=L1Objective(eps=1e-4),
        strategy=SLPStrategy(accuracy='max'),
        threshold=0.01,
    ).fit(phi)
    assert _n_neg(res.corrected) == 0
    tag = _accuracy_tag(res.info)
    if tag is not None:
        assert tag == 'max'


def test_slp_accuracy_max_via_correct_dvf():
    """correct_dvf(strategy='slp', accuracy='max') flows through kwargs."""
    pytest.importorskip('torch')
    from dvfopt import correct_dvf

    H, W = 24, 24
    phi = _planted(H, W, [(11, 11)])
    if _n_neg(phi) == 0:
        pytest.skip('no fold planted')
    res = correct_dvf(
        phi,
        constraint='2tri',
        shape=(H, W),
        strategy='slp',
        accuracy='max',
        threshold=0.01,
    )
    assert _n_neg(res.corrected) == 0


def test_slp_accuracy_invalid_raises():
    """An unknown accuracy mode is rejected (pure validation, no torch)."""
    with pytest.raises(ValueError):
        SLPStrategy(accuracy='nonsense')
