"""Regression tests for the SLSQP/windowed-solver review findings.

* F2 — NaN livelock: non-finite inputs must raise at solver entry
  instead of livelocking ``np.argmin`` on a NaN window.
* F3 — 3D frozen edges must be released at max window (mirroring 2D
  ``exclude_bounds`` semantics), so fold components larger than
  ``max_window`` can still make progress.
* F7 — window-local quality-map patch must match the full-grid map
  exactly for every constraint mode.
* F8 — failed/worse SLSQP results must be rolled back, not applied
  unconditionally.
* F9 — ``SLSQPWindowedStrategy`` must plumb the composed objective all
  the way down to the per-window SLSQP solve instead of silently
  hard-coding an L2 anchor.
"""

import warnings

import numpy as np
import pytest

from dvfopt.core.slsqp_windowed.iterative import iterative_serial
from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d
from dvfopt.core.slsqp_windowed.parallel import iterative_parallel
from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.objectives import L2Objective, Objective


def _folded_deformation_2d(H=10, W=10, spike=5.0):
    """(3, 1, H, W) deformation with a genuine fold near the centre."""
    d = np.zeros((3, 1, H, W), dtype=np.float64)
    d[2, 0, H // 2, W // 2] = spike  # dx spike -> negative Jdet
    return d


# ---------------------------------------------------------------------------
# F2 — NaN entry validation
# ---------------------------------------------------------------------------


class TestNonFiniteEntryGuard:
    def test_serial_nan_plus_real_fold_raises_immediately(self):
        d = _folded_deformation_2d()
        d[2, 0, 1, 1] = np.nan  # NaN pixel alongside a real fold
        with pytest.raises(ValueError, match='non-finite'):
            iterative_serial(d, verbose=0)

    def test_serial_inf_raises(self):
        d = _folded_deformation_2d()
        d[1, 0, 2, 2] = np.inf
        with pytest.raises(ValueError, match='non-finite'):
            iterative_serial(d, verbose=0)

    def test_parallel_nan_raises(self):
        d = _folded_deformation_2d()
        d[2, 0, 1, 1] = np.nan
        with pytest.raises(ValueError, match='non-finite'):
            iterative_parallel(d, verbose=0, max_workers=1)

    def test_3d_nan_raises(self):
        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 3.0  # real fold
        d[1, 1, 1, 1] = np.nan
        with pytest.raises(ValueError, match='non-finite'):
            iterative_3d(d, verbose=0)

    def test_finite_input_still_solves(self):
        d = _folded_deformation_2d(spike=1.2)
        phi = iterative_serial(d, verbose=0, max_iterations=50)
        assert np.isfinite(phi).all()


# ---------------------------------------------------------------------------
# F3 — 3D frozen edges released at max window
# ---------------------------------------------------------------------------


class TestFrozenEdgeReleaseAtMaxWindow3D:
    def test_builder_drops_freeze_at_max_window(self):
        from scipy.optimize import LinearConstraint

        from dvfopt.core.slsqp_windowed.constraints3d import _build_constraints_3d

        sz, sy, sx = 4, 4, 4
        phi_flat = np.zeros(3 * sz * sy * sx)
        mask = np.zeros((sz, sy, sx), dtype=bool)
        mask[0, :, :] = True

        # Default: freeze applies (Jdet excludes frozen rim + equality rows).
        cs = _build_constraints_3d(phi_flat, (sz, sy, sx), mask, 0.01)
        assert any(isinstance(c, LinearConstraint) for c in cs)
        nlc = next(c for c in cs if not isinstance(c, LinearConstraint))
        assert nlc.fun(phi_flat).size == sz * sy * sx - int(mask.sum())

        # At max window: no equality rows, Jdet covers ALL voxels
        # (mirrors 2D exclude_bounds=False semantics).
        cs_max = _build_constraints_3d(phi_flat, (sz, sy, sx), mask, 0.01, window_reached_max=True)
        assert not any(isinstance(c, LinearConstraint) for c in cs_max)
        nlc_max = next(c for c in cs_max if not isinstance(c, LinearConstraint))
        assert nlc_max.fun(phi_flat).size == sz * sy * sx

    def test_fold_larger_than_max_window_makes_progress(self):
        """A fold component bigger than max_window must not spin.

        Pre-fix, the 3D path pinned the (negative) window rim with
        equality constraints even at max window — an infeasible SLSQP
        problem that never made progress.
        """
        D = H = W = 8
        d = np.zeros((3, D, H, W), dtype=np.float64)
        # dx displacement plane -> negative-Jdet sheet at x=4 spanning the
        # whole (z, y) extent: component far larger than a 3x3x3 window.
        d[2, :, :, 3] = 3.0

        init_neg = int((jacobian_det3D(d) <= 0.01 - 1e-5).sum())
        assert init_neg > 27  # sanity: component bigger than max window

        phi = iterative_3d(
            d,
            verbose=0,
            max_window=(3, 3, 3),
            max_iterations=40,
        )
        final_neg = int((jacobian_det3D(phi) <= 0.01 - 1e-5).sum())
        assert final_neg < init_neg, (
            f'no progress on fold larger than max_window: {init_neg} -> {final_neg}'
        )


# ---------------------------------------------------------------------------
# F7 — window-local quality patch is exact
# ---------------------------------------------------------------------------


class TestPatchQuality2D:
    @pytest.mark.parametrize(
        'shoe,inj,tri',
        [
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, True),
        ],
    )
    def test_matches_full_quality_map(self, shoe, inj, tri):
        from dvfopt.core.slsqp_windowed._metrics import _patch_jacobian_2d, _patch_quality_2d
        from dvfopt.core.slsqp_windowed.constraints import _quality_map

        rng = np.random.default_rng(7)
        phi = rng.standard_normal((2, 14, 14)) * 0.3
        jac = jacobian_det2D(phi)
        qual = _quality_map(phi, shoe, inj, enforce_triangles=tri, jacobian_matrix=jac)

        # Perturb a window, patch Jdet + quality locally.
        center, sub_size = (7, 6), (5, 5)
        phi[:, 5:10, 4:9] += rng.standard_normal((2, 5, 5)) * 0.4
        _patch_jacobian_2d(jac, phi, center, sub_size)
        _patch_quality_2d(qual, phi, jac, center, sub_size, shoe, inj, enforce_triangles=tri)

        jac_full = jacobian_det2D(phi)
        qual_full = _quality_map(phi, shoe, inj, enforce_triangles=tri, jacobian_matrix=jac_full)
        # The Jdet patch itself is exact only on the write-back region;
        # outside it phi did not change, so both maps must agree everywhere.
        np.testing.assert_allclose(qual, qual_full, atol=1e-12)

    def test_matches_at_grid_corner(self):
        from dvfopt.core.slsqp_windowed._metrics import _patch_jacobian_2d, _patch_quality_2d
        from dvfopt.core.slsqp_windowed.constraints import _quality_map

        rng = np.random.default_rng(11)
        phi = rng.standard_normal((2, 9, 9)) * 0.25
        jac = jacobian_det2D(phi)
        qual = _quality_map(phi, True, True, enforce_triangles=True, jacobian_matrix=jac)

        center, sub_size = (1, 1), (3, 3)
        phi[:, 0:3, 0:3] += rng.standard_normal((2, 3, 3)) * 0.4
        _patch_jacobian_2d(jac, phi, center, sub_size)
        _patch_quality_2d(qual, phi, jac, center, sub_size, True, True, enforce_triangles=True)

        jac_full = jacobian_det2D(phi)
        qual_full = _quality_map(phi, True, True, enforce_triangles=True, jacobian_matrix=jac_full)
        np.testing.assert_allclose(qual, qual_full, atol=1e-12)

    def test_update_metrics_quality_patch_path(self):
        """_update_metrics with quality_matrix= must equal the legacy path."""
        from dvfopt.core.slsqp_windowed._metrics import _update_metrics
        from dvfopt.core.slsqp_windowed.constraints import _quality_map

        rng = np.random.default_rng(3)
        phi = rng.standard_normal((2, 12, 12)) * 0.3
        phi_init = phi.copy()
        jac = jacobian_det2D(phi)
        qual = _quality_map(phi, True, False, enforce_triangles=False, jacobian_matrix=jac)

        phi[:, 4:9, 4:9] += rng.standard_normal((2, 5, 5)) * 0.3

        jac_new, qual_new, _, _ = _update_metrics(
            phi,
            phi_init,
            True,
            False,
            [],
            [],
            jacobian_matrix=jac,
            patch_center=(6, 6),
            patch_size=(5, 5),
            quality_matrix=qual,
        )
        qual_ref = _quality_map(
            phi, True, False, enforce_triangles=False, jacobian_matrix=jacobian_det2D(phi)
        )
        np.testing.assert_allclose(qual_new, qual_ref, atol=1e-12)
        assert qual_new is qual  # patched in place, not reallocated


# ---------------------------------------------------------------------------
# F8 — compare-and-rollback of failed/worse SLSQP results
# ---------------------------------------------------------------------------


class TestRollbackWorseResults:
    def test_serial_garbage_result_is_rolled_back(self, monkeypatch):
        """When every sub-solve returns garbage, phi must come back unchanged."""
        import dvfopt.core.slsqp_windowed.coordinator as solver_mod

        rng = np.random.default_rng(0)

        def garbage_optimize(phi_sub_flat, phi_init_sub_flat, *args, **kwargs):
            # Deterministically catastrophic: huge non-smooth noise makes
            # the local Jdet far worse than any starting fold.
            return phi_sub_flat + rng.normal(0.0, 10.0, phi_sub_flat.size), 0.0, False

        monkeypatch.setattr(solver_mod, '_optimize_single_window', garbage_optimize)

        d = _folded_deformation_2d(spike=2.0)
        phi = iterative_serial(d, verbose=0, max_iterations=3, max_per_index_iter=2)

        # Every application was strictly worse locally -> all rolled back.
        np.testing.assert_array_equal(phi[0], d[1, 0])
        np.testing.assert_array_equal(phi[1], d[2, 0])

    def test_serial_garbage_result_3d_is_rolled_back(self, monkeypatch):
        import dvfopt.core.slsqp_windowed.coordinator3d as solver3d_mod

        rng = np.random.default_rng(1)

        def garbage_optimize(phi_sub_flat, phi_init_sub_flat, *args, **kwargs):
            return phi_sub_flat + rng.normal(0.0, 10.0, phi_sub_flat.size), 0.0, False

        monkeypatch.setattr(solver3d_mod, '_optimize_single_window_3d', garbage_optimize)

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 2.0  # fold
        phi = iterative_3d(d, verbose=0, max_iterations=3, max_per_index_iter=2)
        np.testing.assert_array_equal(phi, d)

    def test_improving_results_still_applied(self):
        """Sanity: the rollback guard must not block genuine progress."""
        d = _folded_deformation_2d(spike=2.5)
        init_neg = int((jacobian_det2D(np.stack([d[1, 0], d[2, 0]])) <= 0.01 - 1e-5).sum())
        assert init_neg > 0
        phi = iterative_serial(d, verbose=0, max_iterations=60)
        final_neg = int((jacobian_det2D(phi) <= 0.01 - 1e-5).sum())
        assert final_neg < init_neg


# ---------------------------------------------------------------------------
# F9 — SLSQPWindowedStrategy objective plumbing (no silent L2 fallback)
# ---------------------------------------------------------------------------


class _CountingObjective(Objective):
    """L2 anchor that records how many times the solver evaluated it."""

    label = 'l2'

    def __init__(self):
        self.calls = 0

    def __call__(self, diff):
        self.calls += 1
        return L2Objective()(diff)


class TestWindowedStrategyObjective:
    @staticmethod
    def _solve(objective, phi=None):
        from dvfopt.constraints import JdetConstraint2D
        from dvfopt.strategies.slsqp import SLSQPWindowedStrategy

        if phi is None:
            phi = np.zeros((2, 6, 6))  # identity field -> returns immediately
        return SLSQPWindowedStrategy(max_iterations=2).solve(
            phi,
            constraint=JdetConstraint2D(shape=(6, 6)),
            objective=objective,
            threshold=0.01,
            verbose=0,
        )

    def test_objective_reaches_the_window_solve(self):
        """The composed objective is plumbed down to the per-window
        ``scipy.optimize.minimize`` call (it used to be ignored there in
        favour of a hard-coded L2 anchor)."""
        phi = np.zeros((2, 6, 6))
        phi[0, 3, 3] = 2.0
        phi[1, 3, 3] = 2.0  # plants a 2-pixel Jdet fold
        obj = _CountingObjective()
        self._solve(obj, phi=phi)
        assert obj.calls > 0, 'windowed solver never evaluated the composed objective'

    def test_l2_objective_does_not_warn(self):
        from dvfopt.objectives import L2Objective

        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self._solve(L2Objective())

    def test_none_objective_does_not_warn(self):
        from dvfopt.objectives import NoneObjective

        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self._solve(NoneObjective())
