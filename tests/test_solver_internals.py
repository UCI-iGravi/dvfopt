"""Tests for solver internals.

Covers the ``dvfopt.core.slsqp_windowed.coordinator`` helper functions (windowed SLSQP
plumbing) plus ``dvfopt.solver`` internals: the ``Solver.fit`` input
coercion / layout-restore contract, the ``SolveInfo`` legacy-history
stash, and the strategy/constraint registry overwrite guards.
"""

import numpy as np
import pytest

from dvfopt.core.slsqp_windowed.constraints import _build_constraints
from dvfopt.core.slsqp_windowed.coordinator import (
    _apply_result,
    _init_phi,
    _patch_jacobian_2d,
    _setup_accumulators,
    _update_metrics,
)
from dvfopt.jacobian.numpy_jdet import jacobian_det2D


class TestInitPhi:
    def test_shape(self):
        d = np.zeros((3, 1, 8, 12), dtype=np.float64)
        phi, phi_init, H, W = _init_phi(d)
        assert phi.shape == (2, 8, 12)
        assert phi_init.shape == (2, 8, 12)
        assert H == 8
        assert W == 12

    def test_channel_mapping(self):
        """phi[0] = dy (deformation[-2]), phi[1] = dx (deformation[-1])."""
        d = np.zeros((3, 1, 5, 5), dtype=np.float64)
        d[1, 0] = 1.0  # dy channel
        d[2, 0] = 2.0  # dx channel
        phi, _phi_init, _, _ = _init_phi(d)
        np.testing.assert_array_equal(phi[0], 1.0)  # dy
        np.testing.assert_array_equal(phi[1], 2.0)  # dx

    def test_is_independent_copy(self):
        d = np.ones((3, 1, 5, 5), dtype=np.float64)
        phi, phi_init, _, _ = _init_phi(d)
        phi[0, 0, 0] = 999.0
        assert phi_init[0, 0, 0] != 999.0, "phi_init should be independent copy"

    def test_dz_channel_ignored(self):
        """The z-channel (d[0]) is not used in 2D phi."""
        d = np.zeros((3, 1, 5, 5), dtype=np.float64)
        d[0, 0] = 99.0  # dz — should be ignored
        phi, _, _, _ = _init_phi(d)
        np.testing.assert_array_equal(phi, 0.0)


class TestApplyResult:
    def test_writes_correct_region(self):
        phi = np.zeros((2, 10, 10))
        # result_x packing: [dx_flat, dy_flat]
        result_x = np.concatenate([np.ones(9), np.full(9, 2.0)])  # 3x3 window
        _apply_result(phi, result_x, cy=5, cx=5, sub_size=(3, 3))
        # dx (phi[1]) should be 1.0 in the 3x3 region
        np.testing.assert_array_equal(phi[1, 4:7, 4:7], 1.0)
        # dy (phi[0]) should be 2.0 in the 3x3 region
        np.testing.assert_array_equal(phi[0, 4:7, 4:7], 2.0)

    def test_does_not_modify_outside(self):
        phi = np.zeros((2, 10, 10))
        result_x = np.ones(2 * 3 * 3)
        _apply_result(phi, result_x, cy=5, cx=5, sub_size=(3, 3))
        # Outside the 3x3 region should still be zero
        phi[:, 4:7, 4:7] = 0.0
        np.testing.assert_array_equal(phi, 0.0)

    def test_rectangular_window(self):
        phi = np.zeros((2, 10, 10))
        sy, sx = 3, 5
        result_x = np.ones(2 * sy * sx) * 7.0
        _apply_result(phi, result_x, cy=5, cx=5, sub_size=(sy, sx))
        assert phi[1, 4:7, 3:8].sum() == 7.0 * sy * sx


class TestPatchJacobian2D:
    def test_matches_full_recomputation(self):
        """Patched Jacobian should match full recomputation."""
        rng = np.random.default_rng(42)
        phi = rng.standard_normal((2, 12, 12)) * 0.3
        jac_full = jacobian_det2D(phi)

        # Start with stale Jacobian (all ones), patch around center
        jac_patched = np.ones((1, 12, 12))
        _patch_jacobian_2d(jac_patched, phi, center=(6, 6), sub_size=(5, 5))

        # The patched region (and its border) should match
        np.testing.assert_allclose(jac_patched[0, 3:10, 3:10], jac_full[0, 3:10, 3:10], atol=1e-12)

    def test_patch_at_corner(self):
        """Patching near grid corner should not crash."""
        rng = np.random.default_rng(99)
        phi = rng.standard_normal((2, 8, 8)) * 0.2
        jac_full = jacobian_det2D(phi)

        jac_patched = np.ones((1, 8, 8))
        _patch_jacobian_2d(jac_patched, phi, center=(1, 1), sub_size=(3, 3))
        # Region around (1,1) should match full
        np.testing.assert_allclose(jac_patched[0, 0:4, 0:4], jac_full[0, 0:4, 0:4], atol=1e-12)

    def test_mutates_in_place(self):
        phi = np.zeros((2, 8, 8))
        jac = np.zeros((1, 8, 8))
        result = _patch_jacobian_2d(jac, phi, center=(4, 4), sub_size=(3, 3))
        assert result is jac


class TestUpdateMetrics:
    def test_appends_to_accumulators(self):
        phi = np.zeros((2, 6, 6))
        phi_init = np.zeros((2, 6, 6))
        num_neg = []
        min_jdet = []
        error_list = []

        _update_metrics(phi, phi_init, False, False, num_neg, min_jdet, error_list)

        assert len(num_neg) == 1
        assert len(min_jdet) == 1
        assert len(error_list) == 1
        assert num_neg[0] == 0
        np.testing.assert_allclose(min_jdet[0], 1.0)
        np.testing.assert_allclose(error_list[0], 0.0)

    def test_counts_negatives(self):
        phi = np.zeros((2, 6, 6))
        phi[1, 3, 3] = 5.0  # spike creates negative Jdet
        num_neg = []
        min_jdet = []

        _update_metrics(phi, phi.copy(), False, False, num_neg, min_jdet)
        assert num_neg[0] > 0
        assert min_jdet[0] < 1.0

    def test_returns_jacobian_matrix(self):
        phi = np.zeros((2, 6, 6))
        phi_init = phi.copy()
        num_neg = []
        min_jdet = []

        jac, _qm, _neg, _mn = _update_metrics(phi, phi_init, False, False, num_neg, min_jdet)
        assert jac.shape == (1, 6, 6)


class TestBuildConstraints:
    def test_identity_field_constraints_satisfied(self):
        """Identity field should satisfy all constraints."""
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=False, window_reached_max=False, threshold=0.01
        )
        # Should have Jdet constraint + boundary freeze constraint
        assert len(constraints) >= 2

    def test_boundary_freezing(self):
        """When not at edge, boundary pixels should be frozen via LinearConstraint."""
        sy, sx = 5, 5
        rng = np.random.default_rng(42)
        phi_flat = rng.standard_normal(2 * sy * sx) * 0.1
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=False, window_reached_max=False, threshold=0.01
        )

        # Find LinearConstraint (boundary freeze)
        from scipy.optimize import LinearConstraint

        linear_cs = [c for c in constraints if isinstance(c, LinearConstraint)]
        assert len(linear_cs) == 1

        lc = linear_cs[0]
        # Verify frozen values match original
        A_dense = lc.A.toarray()
        frozen_vals = A_dense @ phi_flat
        np.testing.assert_allclose(frozen_vals, lc.lb)
        np.testing.assert_allclose(frozen_vals, lc.ub)

    def test_at_edge_no_boundary_freeze(self):
        """When at grid edge, no boundary freeze constraint is added."""
        sy, sx = 5, 5
        phi_flat = np.zeros(2 * sy * sx)
        constraints = _build_constraints(
            phi_flat, (sy, sx), is_at_edge=True, window_reached_max=False, threshold=0.01
        )

        from scipy.optimize import LinearConstraint

        linear_cs = [c for c in constraints if isinstance(c, LinearConstraint)]
        assert len(linear_cs) == 0

    def test_shoelace_adds_constraint(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        c_no = _build_constraints(phi_flat, (sy, sx), False, False, 0.01, enforce_shoelace=False)
        c_yes = _build_constraints(phi_flat, (sy, sx), False, False, 0.01, enforce_shoelace=True)
        assert len(c_yes) > len(c_no)

    def test_injectivity_adds_constraint(self):
        sy, sx = 6, 6
        phi_flat = np.zeros(2 * sy * sx)
        c_no = _build_constraints(phi_flat, (sy, sx), False, False, 0.01, enforce_injectivity=False)
        c_yes = _build_constraints(phi_flat, (sy, sx), False, False, 0.01, enforce_injectivity=True)
        assert len(c_yes) > len(c_no)


class TestSetupAccumulators:
    def test_returns_five_structures(self):
        result = _setup_accumulators()
        assert len(result) == 5
        for i in range(4):
            assert isinstance(result[i], list)


# ---------------------------------------------------------------------------
# Solver.fit input coercion / layout restore (dvfopt.solver)
# ---------------------------------------------------------------------------


def _folded_2d(H=10, W=10, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, 0.3, (H, W)), rng.normal(0, 0.3, (H, W))])


class TestSolverFitLayoutCoercion:
    """``Solver.fit`` must coerce loose input layouts to the constraint's
    canonical form ONCE before the strategy sees them, then restore the
    original layout on the corrected output (dz passthrough for
    3-channel 2D inputs).

    Regressions reproduced here (pre-fix probe results):

    * wallbreakers crashed deep inside on a raw ``(3, H, W)`` input;
    * ``SLPStrategy`` crashed with 'too many values to unpack' on a
      ``(3, 1, H, W)`` input;
    * ``BarrierStrategy`` silently returned ``(2, H, W)`` for a
      ``(3, H, W)`` input, dropping the dz channel.
    """

    @staticmethod
    def _phi_3hw(H=10, W=10, dz_value=7.0, seed=0):
        phi2 = _folded_2d(H, W, seed)
        dz = np.full((H, W), dz_value)
        return np.stack([dz, phi2[0], phi2[1]])

    def test_wallbreaker_accepts_3hw_and_preserves_dz(self):
        from dvfopt import (
            HarmonicALMRefineRepairStrategy,
            L1Objective,
            Solver,
            TriConstraint2DFullCoverage,
        )

        phi = self._phi_3hw()
        res = Solver(
            constraint=TriConstraint2DFullCoverage(shape=(10, 10)),
            objective=L1Objective(),
            strategy=HarmonicALMRefineRepairStrategy(),
        ).fit(phi)
        assert res.corrected.shape == (3, 10, 10)
        np.testing.assert_array_equal(res.corrected[0], 7.0)  # dz untouched
        # dy/dx must actually have been corrected (not just copied back).
        assert not np.array_equal(res.corrected[1:], phi[1:])

    def test_slp_accepts_31hw_singleton_d(self):
        from dvfopt import L1Objective, SLPStrategy, Solver, TriConstraint2DFullCoverage

        phi2 = _folded_2d()
        phi = np.zeros((3, 1, 10, 10))
        phi[0, 0] = 3.0  # dz sentinel
        phi[1, 0] = phi2[0]
        phi[2, 0] = phi2[1]
        res = Solver(
            constraint=TriConstraint2DFullCoverage(shape=(10, 10)),
            objective=L1Objective(),
            strategy=SLPStrategy(n_workers=1),
        ).fit(phi)
        assert res.corrected.shape == (3, 1, 10, 10)
        np.testing.assert_array_equal(res.corrected[0], 3.0)  # dz untouched

    def test_barrier_3hw_returns_same_shape_with_dz(self):
        from dvfopt import BarrierStrategy, L1Objective, Solver, TriConstraint2DFullCoverage

        phi = self._phi_3hw(dz_value=-2.5)
        res = Solver(
            constraint=TriConstraint2DFullCoverage(shape=(10, 10)),
            objective=L1Objective(),
            strategy=BarrierStrategy(),
        ).fit(phi)
        # Pre-fix: barrier silently returned (2, 10, 10), dropping dz.
        assert res.corrected.shape == phi.shape
        np.testing.assert_array_equal(res.corrected[0], -2.5)

    def test_canonical_2hw_input_passes_through(self):
        from dvfopt import BarrierStrategy, L1Objective, Solver, TriConstraint2DFullCoverage

        phi = _folded_2d()
        res = Solver(
            constraint=TriConstraint2DFullCoverage(shape=(10, 10)),
            objective=L1Objective(),
            strategy=BarrierStrategy(),
        ).fit(phi)
        assert res.corrected.shape == (2, 10, 10)


# ---------------------------------------------------------------------------
# SolveInfo legacy-history stash (dvfopt.solver)
# ---------------------------------------------------------------------------


class TestLegacyHistoryStash:
    """``SolveInfo.from_legacy_history`` must not retain the raw history
    verbatim in ``extras['_legacy_history']`` when the phases were
    successfully extracted — that duplicated the data (once as
    PhaseInfo, once raw) on every slice."""

    def test_no_stash_when_phases_extracted(self):
        from dvfopt.solver import SolveInfo

        legacy = [
            dict(phase='penalty', nit=10, n_neg=5, min_T=-0.2, wall_s=0.1),
            dict(phase='barrier', nit=15, n_neg=0, min_T=0.011, wall_s=0.2),
        ]
        info = SolveInfo.from_legacy_history('S', legacy, threshold=0.01)
        assert len(info.phases) == 2
        assert '_legacy_history' not in info.extras

    def test_stash_kept_when_extraction_fails(self):
        from dvfopt.solver import SolveInfo

        legacy = ['not-a-dict', 42]  # nothing extractable
        info = SolveInfo.from_legacy_history('S', legacy, threshold=0.01)
        assert info.phases == []
        assert info.extras['_legacy_history'] == legacy


# ---------------------------------------------------------------------------
# Registry overwrite guards
# ---------------------------------------------------------------------------


class TestStrategyRegistryOverwriteGuard:
    _LABEL = '__test_registry_guard_strategy__'

    def test_same_class_reregistration_is_silent(self):
        from dvfopt.strategies.base import _STRATEGY_REGISTRY, Strategy, register_strategy

        class _Dummy(Strategy):
            def solve(self, phi_in, *, constraint, objective, threshold, **kwargs):
                return phi_in, {}

        try:
            register_strategy(self._LABEL)(_Dummy)
            # Same class object again — idempotent, no raise.
            register_strategy(self._LABEL)(_Dummy)
            assert _STRATEGY_REGISTRY[self._LABEL] is _Dummy
        finally:
            _STRATEGY_REGISTRY.pop(self._LABEL, None)

    def test_different_class_raises(self):
        from dvfopt.strategies.base import _STRATEGY_REGISTRY, Strategy, register_strategy

        class _DummyA(Strategy):
            def solve(self, phi_in, *, constraint, objective, threshold, **kwargs):
                return phi_in, {}

        class _DummyB(Strategy):
            def solve(self, phi_in, *, constraint, objective, threshold, **kwargs):
                return phi_in, {}

        try:
            register_strategy(self._LABEL)(_DummyA)
            with pytest.raises(ValueError, match='already registered'):
                register_strategy(self._LABEL)(_DummyB)
            # Both class names appear in the message.
            with pytest.raises(ValueError, match='_DummyA'):
                register_strategy(self._LABEL)(_DummyB)
            with pytest.raises(ValueError, match='_DummyB'):
                register_strategy(self._LABEL)(_DummyB)
            # Original registration is intact.
            assert _STRATEGY_REGISTRY[self._LABEL] is _DummyA
        finally:
            _STRATEGY_REGISTRY.pop(self._LABEL, None)


class TestConstraintRegistryOverwriteGuard:
    _LABEL = '__test_registry_guard_constraint__'

    def test_same_class_reregistration_is_silent(self):
        from dvfopt.constraints import (
            _CONSTRAINT_REGISTRY,
            TriConstraint2D,
            register_constraint,
        )

        try:
            register_constraint(self._LABEL)(TriConstraint2D)
            register_constraint(self._LABEL)(TriConstraint2D)  # no raise
            assert _CONSTRAINT_REGISTRY[self._LABEL] is TriConstraint2D
        finally:
            _CONSTRAINT_REGISTRY.pop(self._LABEL, None)

    def test_different_class_raises(self):
        from dvfopt.constraints import (
            _CONSTRAINT_REGISTRY,
            JdetConstraint2D,
            TriConstraint2D,
            register_constraint,
        )

        try:
            register_constraint(self._LABEL)(TriConstraint2D)
            with pytest.raises(ValueError, match=r'TriConstraint2D.*JdetConstraint2D'):
                register_constraint(self._LABEL)(JdetConstraint2D)
            assert _CONSTRAINT_REGISTRY[self._LABEL] is TriConstraint2D
        finally:
            _CONSTRAINT_REGISTRY.pop(self._LABEL, None)
