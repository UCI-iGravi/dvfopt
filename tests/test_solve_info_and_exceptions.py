"""Tests for the SolveInfo / PhaseInfo contract and exception hierarchy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dvfopt import (
    BarrierStrategy,
    DVFopt,
    DVFoptConfig,
    DVFoptError,
    IncompatibleConstraintError,
    JdetConstraint2D,
    L1Objective,
    L2Objective,
    M10Strategy,
    PhaseInfo,
    SimplexConstraint2D,
    SolveInfo,
    Solver,
    SolverConfigError,
)

# ---------------------------------------------------------------------------
# SolveInfo / PhaseInfo
# ---------------------------------------------------------------------------


class TestSolveInfo:
    def test_barrier_run_produces_populated_solve_info(self):
        """Barrier strategy should produce a SolveInfo with at least one
        PhaseInfo per λ-step (penalty phase) + μ-step (barrier phase)."""
        rng = np.random.default_rng(7)
        phi = np.stack([rng.normal(0, 0.4, (8, 8)), rng.normal(0, 0.4, (8, 8))])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = Solver(
                constraint=SimplexConstraint2D((8, 8)),
                objective=L1Objective(),
                strategy=BarrierStrategy(),
            ).fit(phi, record_history=True)
        assert isinstance(res.info, SolveInfo)
        assert res.info.strategy_name == 'BarrierStrategy'
        assert len(res.info.phases) > 0
        # Every phase has a name + the canonical fields.
        for p in res.info.phases:
            assert isinstance(p, PhaseInfo)
            assert p.name
            # PhaseInfo fields exist even if empty.
            assert p.n_neg >= -1
            assert p.wall_s >= 0
        # When the strategy reached feasibility, the index should be set.
        if res.feasible:
            assert res.info.feasible_after_phase >= 0

    def test_empty_info_normalizes_to_empty_solve_info(self):
        """Strategies that don't track history still produce a SolveInfo
        (empty phases) — visualization code can always rely on the type."""
        rng = np.random.default_rng(7)
        phi = np.stack([rng.normal(0, 0.4, (8, 8)), rng.normal(0, 0.4, (8, 8))])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = Solver(
                constraint=SimplexConstraint2D((8, 8)),
                objective=L1Objective(),
                strategy=BarrierStrategy(),
            ).fit(phi)  # record_history=False
        assert isinstance(res.info, SolveInfo)
        # Phases may be empty but the container type is consistent.
        assert isinstance(res.info.phases, list)

    def test_solve_info_from_legacy_history(self):
        """Adapter handles the legacy ``history`` list-of-dicts shape."""
        legacy = [
            dict(phase='penalty', nit=10, n_neg=5, min_T=-0.2, wall_s=0.1, lam=1.0),
            dict(phase='penalty', nit=8, n_neg=0, min_T=0.012, wall_s=0.1, lam=100.0),
            dict(phase='barrier', nit=15, n_neg=0, min_T=0.0110, wall_s=0.2, mu=0.01),
        ]
        info = SolveInfo.from_legacy_history('TestStrategy', legacy, threshold=0.01)
        assert info.strategy_name == 'TestStrategy'
        assert len(info.phases) == 3
        # feasible_after_phase = first index where n_neg == 0 AND min_T >= threshold.
        assert info.feasible_after_phase == 1
        # Free-form keys (lam, mu) end up in extras.
        assert info.phases[0].extras['lam'] == 1.0
        assert info.phases[2].extras['mu'] == 0.01

    def test_history_df_still_works_through_dvfopt(self):
        """The DVFopt.Result.history_df() helper should still produce a
        sensible dataframe even after the SolveInfo wiring."""
        pytest.importorskip('pandas')  # history_df() lazily imports pandas (optional dep)
        rng = np.random.default_rng(7)
        phi = np.stack([rng.normal(0, 0.4, (8, 8)), rng.normal(0, 0.4, (8, 8))])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = DVFopt(
                DVFoptConfig(
                    constraint='simplex',
                    solver='barrier',
                    objective='l1',
                    verbose=0,
                    record_history=True,
                )
            ).fit(phi)
        df = res.history_df()
        assert len(df) > 0
        # Required columns from the SolveInfo flattening.
        for col in ('z', 'phase', 'nit', 'n_neg', 'min_T'):
            assert col in df.columns


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_incompatible_constraint_raises_specific_exception(self):
        """``M10Strategy`` requires a 2-triangle constraint; constructing
        a Solver with a Jdet constraint should raise the new
        ``IncompatibleConstraintError``."""
        with pytest.raises(IncompatibleConstraintError) as ei:
            Solver(
                constraint=JdetConstraint2D((8, 8)),
                objective=L2Objective(),
                strategy=M10Strategy(),
            )
        # Subclasses both DVFoptError and TypeError — existing code that
        # catches either still works.
        assert isinstance(ei.value, DVFoptError)
        assert isinstance(ei.value, TypeError)

    def test_solver_config_error_for_bad_solver_label(self):
        with pytest.raises(SolverConfigError) as ei:
            DVFopt(DVFoptConfig(solver='not-a-real-solver'))
        assert isinstance(ei.value, DVFoptError)
        assert isinstance(ei.value, ValueError)

    def test_solver_config_error_for_bad_constraint(self):
        with pytest.raises(SolverConfigError):
            DVFopt(DVFoptConfig(constraint='not-a-real-constraint'))

    def test_solver_config_error_for_bad_objective(self):
        with pytest.raises(SolverConfigError):
            DVFopt(DVFoptConfig(objective='not-a-real-objective'))

    def test_solver_config_error_for_bad_solver_type(self):
        """``solver`` must be str or Strategy; integers should raise
        SolverConfigError."""
        with pytest.raises(SolverConfigError):
            DVFopt(DVFoptConfig(solver=42))
