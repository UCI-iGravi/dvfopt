"""Direct unit tests for ``dvfopt.core.barrier._core``.

This is the single homotopy engine that routes all four CPU barrier
solvers, so it deserves localized tests. Failures here are far easier to
locate than via the multi-layer integration tests.
"""

import numpy as np
import pytest

from dvfopt.core.barrier._core import (
    DEFAULT_LAM_SCHEDULE,
    DEFAULT_MU_SCHEDULE,
    anchor_term,
    run_penalty_barrier_lbfgs,
)

# ---------------------------------------------------------------------------
# anchor_term: per-mode value + gradient verification
# ---------------------------------------------------------------------------


class TestAnchorTerm:
    def test_l2_zero_diff(self):
        diff = np.zeros(5)
        val, grad = anchor_term(diff, 'l2')
        assert val == 0.0
        np.testing.assert_array_equal(grad, np.zeros(5))

    def test_l2_value_and_grad(self):
        diff = np.array([1.0, 2.0, -3.0])
        val, grad = anchor_term(diff, 'l2')
        # F = 0.5 * (1 + 4 + 9) = 7.0; dF/d diff = diff
        assert val == pytest.approx(7.0)
        np.testing.assert_array_equal(grad, diff)

    def test_l1_zero_diff_returns_zero(self):
        # Smoothed L1: sqrt(0 + eps^2) - eps = 0
        diff = np.zeros(4)
        val, grad = anchor_term(diff, 'l1', eps_l1=1e-3)
        assert val == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(grad, 0.0, atol=1e-9)

    def test_l1_large_diff_approaches_abs(self):
        # When |diff| >> eps, F ≈ sum(|diff|) and grad ≈ sign(diff).
        diff = np.array([10.0, -5.0, 3.0])
        val, grad = anchor_term(diff, 'l1', eps_l1=1e-4)
        assert val == pytest.approx(np.abs(diff).sum(), rel=1e-3)
        np.testing.assert_allclose(grad, np.sign(diff), atol=1e-3)

    def test_none_returns_zero(self):
        diff = np.array([1.0, 2.0, 3.0])
        val, grad = anchor_term(diff, 'none')
        assert val == 0.0
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            anchor_term(np.zeros(3), 'l3')


# ---------------------------------------------------------------------------
# Toy 1-D problem: T(phi) = phi - 0.5, identity Jacobian.
# Lets us check the homotopy mechanics without any constraint coupling.
# ---------------------------------------------------------------------------


def _toy_constraint():
    return (lambda p: p - 0.5, lambda p, v: v)


class TestHomotopyMechanics:
    def test_already_feasible_skips_penalty(self):
        # Start above target: penalty phase shouldn't iterate.
        phi = np.array([1.0])
        vals, adj = _toy_constraint()
        _phi_out, info = run_penalty_barrier_lbfgs(
            phi,
            phi.copy(),
            constraint_values=vals,
            constraint_adjoint=adj,
            threshold=0.0,
            margin=0.01,
            max_iter=20,
            verbose=0,
        )
        assert info['lam_steps'] == 0
        assert info['feasible'] is True
        # Barrier phase still runs over the full mu_schedule.
        assert info['mu_steps'] == len(DEFAULT_MU_SCHEDULE)

    def test_infeasible_runs_penalty(self):
        # Start below target: must escalate lam.
        phi = np.array([0.0])
        vals, adj = _toy_constraint()
        _, info = run_penalty_barrier_lbfgs(
            phi,
            phi.copy(),
            constraint_values=vals,
            constraint_adjoint=adj,
            threshold=0.0,
            margin=0.01,
            max_iter=20,
            verbose=0,
        )
        assert info['lam_steps'] >= 1

    def test_zero_mu_steps_when_infeasibility_persists(self):
        # Use a tiny lam_schedule so the penalty phase never reaches feasibility.
        phi = np.array([-10.0])  # very far below target
        vals, adj = _toy_constraint()
        _, info = run_penalty_barrier_lbfgs(
            phi,
            phi.copy(),
            constraint_values=vals,
            constraint_adjoint=adj,
            threshold=0.0,
            margin=0.01,
            lam_schedule=(0.001,),  # one tiny step
            max_iter=5,
            verbose=0,
        )
        # Penalty phase did one step but didn't reach feasibility.
        assert info['lam_steps'] == 1
        # Barrier phase never runs when infeasible.
        assert info['mu_steps'] == 0
        assert info['feasible'] is False


class TestRecordHistorySchema:
    def test_history_records_per_step(self):
        phi = np.array([0.0])
        vals, adj = _toy_constraint()
        _, info = run_penalty_barrier_lbfgs(
            phi,
            phi.copy(),
            constraint_values=vals,
            constraint_adjoint=adj,
            threshold=0.0,
            margin=0.01,
            max_iter=10,
            verbose=0,
            record_history=True,
        )
        # At least one penalty step (probably one barrier step too).
        assert len(info['history']) >= 1
        for h in info['history']:
            assert 'phase' in h
            assert h['phase'] in ('penalty', 'barrier')
            assert 'step' in h
            assert 'n_neg' in h
            assert 'min_T' in h
            assert 'wall_s' in h
            if h['phase'] == 'penalty':
                assert 'lam' in h
            else:
                assert 'mu' in h

    def test_history_empty_by_default(self):
        phi = np.array([0.0])
        vals, adj = _toy_constraint()
        _, info = run_penalty_barrier_lbfgs(
            phi,
            phi.copy(),
            constraint_values=vals,
            constraint_adjoint=adj,
            threshold=0.0,
            margin=0.01,
            max_iter=5,
            verbose=0,
            record_history=False,
        )
        assert info['history'] == []


# ---------------------------------------------------------------------------
# active_mask: only listed constraint cells participate
# ---------------------------------------------------------------------------


class TestActiveMask:
    def test_mask_excludes_cells_from_penalty(self):
        """A cell masked out should not drive the penalty even if T < threshold."""
        # T(phi) = phi, dim 3. Mask out cell 2 (the one that's most negative).
        # If the mask is honoured, cell 2's value is ignored and the solver
        # only fixes cells 0, 1.
        phi = np.array([-0.5, -0.5, -2.0])
        anchor = phi.copy()
        active = np.array([True, True, False])
        phi_out, _info = run_penalty_barrier_lbfgs(
            phi,
            anchor,
            constraint_values=lambda p: p,
            constraint_adjoint=lambda p, v: v,
            threshold=0.0,
            margin=0.01,
            active_mask=active,
            max_iter=30,
            verbose=0,
        )
        # Cells 0, 1 should be lifted above target; cell 2 stays roughly where it was.
        assert phi_out[0] >= 0.01 - 1e-3
        assert phi_out[1] >= 0.01 - 1e-3
        # Cell 2 should NOT have been corrected by the penalty (it's masked out).
        # Some L2 anchor drift is OK, but not the +2 correction.
        assert phi_out[2] < 0.0


# ---------------------------------------------------------------------------
# bounds: frozen variables stay fixed
# ---------------------------------------------------------------------------


class TestBounds:
    def test_pinned_variable_does_not_move(self):
        # Two variables; the first is pinned by bounds=(value, value).
        phi = np.array([0.0, -0.5])
        anchor = phi.copy()
        phi_out, _info = run_penalty_barrier_lbfgs(
            phi,
            anchor,
            constraint_values=lambda p: p,
            constraint_adjoint=lambda p, v: v,
            threshold=0.0,
            margin=0.01,
            bounds=[(0.0, 0.0), (None, None)],
            max_iter=20,
            verbose=0,
        )
        # First var stays pinned.
        assert phi_out[0] == pytest.approx(0.0, abs=1e-8)
        # Second var should be corrected.
        assert phi_out[1] >= 0.0


# ---------------------------------------------------------------------------
# Infeasible iterate guard in barrier phase
# ---------------------------------------------------------------------------


class TestBarrierInfeasibleGuard:
    """The barrier objective returns inf for infeasible iterates; phi_flat
    must not be silently updated when that happens."""

    def test_barrier_rejects_infinite_iterate(self):
        # Hand-craft: start above target. The L-BFGS-B inner loop may
        # try to step into the infeasible region; if it does, the
        # guard `if np.isfinite(res.fun)` prevents corrupting phi.
        # We can't easily force that line search, but we can at least
        # verify the result is feasible after a full run.
        phi = np.array([1.0])
        vals, adj = _toy_constraint()
        phi_out, _info = run_penalty_barrier_lbfgs(
            phi,
            phi.copy(),
            constraint_values=vals,
            constraint_adjoint=adj,
            threshold=0.0,
            margin=0.01,
            max_iter=30,
            verbose=0,
        )
        # After full run, must still be feasible (T > threshold).
        T = vals(phi_out)
        assert float(T.min()) > 0.0
