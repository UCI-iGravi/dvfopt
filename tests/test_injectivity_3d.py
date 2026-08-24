"""Tests for the 3D injectivity (axial monotonicity) constraint mode.

Covers the new pieces: the monotonicity diff/quality helpers, the sparse
LinearConstraint builder, and ``iterative_3d(enforce_injectivity=True)``
end-to-end (including via ``SLSQPWindowedStrategy``).
"""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt.jacobian.monotonicity import (
    _monotonicity_diffs_3d,
    injectivity_quality_3d,
)
from dvfopt.jacobian.numpy_jdet import jacobian_det3D

THRESHOLD = 0.01


class TestMonotonicityDiffs3D:
    def test_identity_gaps_are_unity(self):
        dz = np.zeros((4, 5, 6))
        dy = np.zeros((4, 5, 6))
        dx = np.zeros((4, 5, 6))
        gz, gy, gx = _monotonicity_diffs_3d(dz, dy, dx)
        assert gz.shape == (3, 5, 6) and gy.shape == (4, 4, 6) and gx.shape == (4, 5, 5)
        assert np.allclose(gz, 1.0) and np.allclose(gy, 1.0) and np.allclose(gx, 1.0)

    def test_quality_flags_crossing(self):
        phi = np.zeros((3, 4, 5, 6))
        phi[2, :, :, 1] = -1.5  # deformed X of column 1 crosses column 0
        q = injectivity_quality_3d(phi)
        assert q.shape == (4, 5, 6)
        # Both endpoint columns of the violated gap are flagged.
        assert (q[:, :, 0] < 0).all() and (q[:, :, 1] < 0).all()
        # Untouched columns keep unit separation quality.
        assert np.allclose(q[:, :, 3], 1.0)

    def test_quality_identity_is_unity(self):
        q = injectivity_quality_3d(np.zeros((3, 3, 4, 5)))
        assert np.allclose(q, 1.0)


class TestLinearConstraintBuilder:
    def test_row_count_and_values(self):
        from dvfopt.core.slsqp_windowed.constraints3d import _injectivity_linear_constraint_3d

        sz = sy = sx = 3
        lc = _injectivity_linear_constraint_3d((sz, sy, sx), inj_lb=THRESHOLD)
        # 3 axes x (2 gaps x 3 x 3 per axis) = 54 rows over 81 variables.
        assert lc.A.shape == (54, 3 * 27)
        # A @ phi + 1 must equal the concatenated axial gaps.
        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.1, (3, sz, sy, sx))  # [dz, dy, dx]
        phi_flat = np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])
        gz, gy, gx = _monotonicity_diffs_3d(phi[0], phi[1], phi[2])
        expected = np.concatenate([gx.ravel(), gy.ravel(), gz.ravel()])
        np.testing.assert_allclose(lc.A @ phi_flat + 1.0, expected, atol=1e-12)

    def test_fully_frozen_boundary_returns_none(self):
        from dvfopt.core.slsqp_windowed.constraints3d import _injectivity_linear_constraint_3d

        freeze = np.ones((3, 3, 3), dtype=bool)
        freeze[1, 1, 1] = False  # single free voxel — no free pair exists
        assert _injectivity_linear_constraint_3d((3, 3, 3), THRESHOLD, freeze) is None


def _folded_volume():
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, 4, 6, 6))
    phi[1, 1:3, 2:4, 2:4] -= 1.4
    phi[2, 1:3, 2:4, 2:4] -= 1.4
    assert (jacobian_det3D(phi) <= 0).any()
    return phi


class TestIterative3DInjectivity:
    def test_end_to_end_both_metrics_feasible(self):
        from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d

        phi = _folded_volume()
        out = iterative_3d(
            phi,
            threshold=THRESHOLD,
            verbose=0,
            enforce_injectivity=True,
            max_iterations=200,
        )
        jdet = jacobian_det3D(out)
        q = injectivity_quality_3d(out)
        assert float(jdet.min()) >= THRESHOLD - 1e-4, f'jdet min {jdet.min():.5f}'
        assert float(q.min()) >= THRESHOLD - 1e-4, f'gap min {q.min():.5f}'

    def test_custom_injectivity_threshold(self):
        from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d

        phi = _folded_volume()
        inj_lb = 0.15
        out = iterative_3d(
            phi,
            threshold=THRESHOLD,
            verbose=0,
            enforce_injectivity=True,
            injectivity_threshold=inj_lb,
            max_iterations=200,
        )
        q = injectivity_quality_3d(out)
        assert float(q.min()) >= inj_lb - 1e-4, f'gap min {q.min():.5f} < {inj_lb}'


class TestStrategySurface:
    def test_windowed_strategy_injectivity_3d(self):
        from dvfopt import JdetConstraint3D, L2Objective, SLSQPWindowedStrategy, Solver

        phi = _folded_volume()
        result = Solver(
            constraint=JdetConstraint3D(shape=phi.shape[1:]),
            objective=L2Objective(),
            strategy=SLSQPWindowedStrategy(enforce_injectivity=True),
            threshold=THRESHOLD,
        ).fit(phi)
        q = injectivity_quality_3d(result.corrected)
        assert float(q.min()) >= THRESHOLD - 1e-4

    def test_shoelace_3d_raises(self):
        from dvfopt import JdetConstraint3D, L2Objective, SLSQPWindowedStrategy, Solver

        phi = _folded_volume()
        with pytest.raises(ValueError, match='simplex_3d'):
            Solver(
                constraint=JdetConstraint3D(shape=phi.shape[1:]),
                objective=L2Objective(),
                strategy=SLSQPWindowedStrategy(enforce_shoelace=True),
                threshold=THRESHOLD,
            ).fit(phi)


class TestInjectivityOnlyRepair:
    """Regression: an injectivity-only violation in a Jdet-healthy region
    must be repairable. The first implementation's accept/rollback gate
    measured only the local Jdet, so the fix was reverted whenever it
    nudged a still-healthy Jdet down by any epsilon — livelocking the
    outer loop until max_iterations with the crossing intact."""

    def test_no_livelock_on_jdet_feasible_crossing(self):
        from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d

        phi = np.zeros((3, 5, 8, 8))
        phi[2, :, :, 3] = -1.05  # x-gap col 2->3 = -0.05; Jdet stays feasible
        phi[1, 2, 4:7, 4] = [-0.95, -1.90, -0.95]  # healthy-but-low Jdet spot
        assert (jacobian_det3D(phi) > 0).all(), 'fixture must be Jdet-feasible'
        out = iterative_3d(
            phi,
            threshold=THRESHOLD,
            verbose=0,
            enforce_injectivity=True,
            max_iterations=100,
        )
        q = injectivity_quality_3d(out)
        assert float(q.min()) >= THRESHOLD - 1e-4, f'gap min {q.min():.5f} — livelock regression'
