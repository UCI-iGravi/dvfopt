"""Tests for the NMVF heuristic strategy and the legacy → descriptive
strategy class renames.

Two concerns covered here:

* NMVF runs through ``Solver`` / ``correct_dvf`` and clears at least
  one sparse, isolated planted Jdet fold.
* Every renamed wallbreaker is also exported under its old
  ``M*Strategy`` tag and the two names refer to the same class — so
  existing notebooks / external code keeps importing whatever it
  imported before.
"""

from __future__ import annotations

import numpy as np
import pytest

import dvfopt
from dvfopt import (
    BarrierStrategy,
    HarmonicALMBarrier3DStrategy,
    HarmonicALMBarrierStrategy,
    HarmonicALMRefineRepair3DStrategy,
    HarmonicALMRefineRepairStrategy,
    JdetConstraint2D,
    M10Strategy,
    M10TetStrategy,
    M14Schwarz3DStrategy,
    M14SchwarzStrategy,
    M14Strategy,
    M14TetStrategy,
    NMVFStrategy,
    NoneObjective,
    SchwarzHarmonicALMRefineRepair3DStrategy,
    SchwarzHarmonicALMRefineRepairStrategy,
    Solver,
    correct_dvf,
)
from dvfopt.jacobian.numpy_jdet import jacobian_det2D

# ---------------------------------------------------------------------------
# Back-compat aliases: old name `is` new name (same class object).
# ---------------------------------------------------------------------------


class TestBackCompatAliases:
    """The old ``M*Strategy`` exports must remain valid after the rename."""

    def test_m10_alias(self):
        assert M10Strategy is HarmonicALMBarrierStrategy

    def test_m14_alias(self):
        assert M14Strategy is HarmonicALMRefineRepairStrategy

    def test_m14_schwarz_alias(self):
        assert M14SchwarzStrategy is SchwarzHarmonicALMRefineRepairStrategy

    def test_m10_tet_alias(self):
        assert M10TetStrategy is HarmonicALMBarrier3DStrategy

    def test_m14_tet_alias(self):
        assert M14TetStrategy is HarmonicALMRefineRepair3DStrategy

    def test_m14_schwarz_3d_alias(self):
        assert M14Schwarz3DStrategy is SchwarzHarmonicALMRefineRepair3DStrategy

    def test_old_string_keys_still_register(self):
        """The original ``'m10'`` / ``'m14'`` / ``'m14_schwarz'`` registry
        strings (and the 3D variants) must still resolve."""
        from dvfopt.strategies import make_strategy

        assert isinstance(make_strategy('m10'), HarmonicALMBarrierStrategy)
        assert isinstance(make_strategy('m14'), HarmonicALMRefineRepairStrategy)
        assert isinstance(make_strategy('m14_schwarz'), SchwarzHarmonicALMRefineRepairStrategy)
        assert isinstance(make_strategy('m10_3d'), HarmonicALMBarrier3DStrategy)
        assert isinstance(make_strategy('m14_3d'), HarmonicALMRefineRepair3DStrategy)
        assert isinstance(
            make_strategy('m14_schwarz_3d'),
            SchwarzHarmonicALMRefineRepair3DStrategy,
        )

    def test_new_string_keys_register(self):
        """The new descriptive registry strings resolve too."""
        from dvfopt.strategies import make_strategy

        assert isinstance(make_strategy('harmonic_alm_barrier'), HarmonicALMBarrierStrategy)
        assert isinstance(
            make_strategy('harmonic_alm_refine_repair'),
            HarmonicALMRefineRepairStrategy,
        )
        assert isinstance(
            make_strategy('schwarz_harmonic_alm_refine_repair'),
            SchwarzHarmonicALMRefineRepairStrategy,
        )
        assert isinstance(make_strategy('harmonic_alm_barrier_3d'), HarmonicALMBarrier3DStrategy)
        assert isinstance(
            make_strategy('harmonic_alm_refine_repair_3d'),
            HarmonicALMRefineRepair3DStrategy,
        )
        assert isinstance(
            make_strategy('schwarz_harmonic_alm_refine_repair_3d'),
            SchwarzHarmonicALMRefineRepair3DStrategy,
        )
        assert isinstance(make_strategy('nmvf'), NMVFStrategy)


# ---------------------------------------------------------------------------
# NMVF — heuristic neighborhood-mean-vector filter.
# ---------------------------------------------------------------------------


def _planted_jdet_fold_2d(H: int = 7, W: int = 7) -> np.ndarray:
    """Build a ``(2, H, W)`` field with at least one Jdet < 0 pixel.

    The numpy Jdet stencil uses a one-sided diff (``phi[i+1] - phi[i]``),
    so we plant a sharply asymmetric step at (cy-1, cx-1) → (cy+1, cx+1)
    that pushes the cross term well past 1 at the centre. With
    ``v = 3.0`` the centre Jdet drops to ``1 - v*v = -8``.
    """
    phi = np.zeros((2, H, W), dtype=np.float64)
    cy, cx = H // 2, W // 2
    v = 3.0
    phi[0, cy, cx - 1] = +v
    phi[0, cy, cx + 1] = -v
    phi[1, cy - 1, cx] = +v
    phi[1, cy + 1, cx] = -v
    return phi


class TestNMVFStrategy:
    def test_clears_isolated_fold(self):
        phi = _planted_jdet_fold_2d(7, 7)
        J0 = jacobian_det2D(phi)[0]
        assert (J0 <= 0).any(), 'test fixture must plant at least one fold'

        result = Solver(
            constraint=JdetConstraint2D(shape=phi.shape[1:]),
            objective=NoneObjective(),
            strategy=NMVFStrategy(max_iter=200),
        ).fit(phi)

        # NMVF should drive every Jdet ≤ 0 pixel positive on a sparse fold.
        J = jacobian_det2D(result.corrected)[0]
        assert (J <= 0).sum() == 0, (
            f'NMVF left {(J <= 0).sum()} folded pixels (min_J={J.min():.4f})'
        )

    def test_correct_dvf_string_path(self):
        """``correct_dvf(..., strategy='nmvf')`` is the user-facing path."""
        phi = _planted_jdet_fold_2d(7, 7)
        result = correct_dvf(
            phi,
            constraint='jdet',
            objective='none',
            strategy='nmvf',
        )
        J = jacobian_det2D(result.corrected)[0]
        assert (J <= 0).sum() == 0

    def test_no_op_on_feasible_input(self):
        """Zero displacement is already feasible — NMVF should return
        early without modifying the field."""
        phi = np.zeros((2, 6, 6), dtype=np.float64)
        result = Solver(
            constraint=JdetConstraint2D(shape=(6, 6)),
            objective=NoneObjective(),
            strategy=NMVFStrategy(max_iter=10),
        ).fit(phi)
        np.testing.assert_array_equal(result.corrected, phi)

    def test_exact_zero_jdet_does_not_spin(self):
        """Regression: the loop gate counts ``J <= 0`` but the work list
        used ``J < 0`` — an exactly-zero determinant kept the loop alive
        with an EMPTY work list, burning ``max_iter`` no-op full-slice
        iterations and reporting converged=False. The work list must
        treat J == 0 cells as folds so the smoother can act on them."""
        from dvfopt.core.nmvf import nmvf_correct_2d

        H, W = 9, 9
        cy, cx = 4, 4
        phi = np.zeros((2, H, W), dtype=np.float64)
        # Central-difference stencil: ddy_dy(cy,cx) = (dy[cy+1]-dy[cy-1])/2
        # = -1 exactly -> J(cy,cx) = (1-1)*(1+0) - 0 = 0, no negative J.
        phi[0, cy - 1, cx] = +1.0
        phi[0, cy + 1, cx] = -1.0
        J0 = jacobian_det2D(phi)[0]
        assert (J0 == 0).any(), 'fixture must plant an exactly-zero Jdet'
        assert not (J0 < 0).any(), 'fixture must have no strictly-negative Jdet'

        max_iter = 100
        _out, info = nmvf_correct_2d(phi, max_iter=max_iter, record_history=True)
        # No spin: the zero cell is on the work list, the smoother resolves
        # it in a handful of iterations instead of exhausting max_iter.
        assert info['n_iter'] < max_iter, (
            f'NMVF spun for all {max_iter} iterations on a J == 0 cell'
        )
        assert info['n_iter'] <= 10
        assert info['converged'] is True
        assert info['final_min_J'] > 0

    def test_rejects_3d_constraint(self):
        """NMVF is 2D-only; pairing with a 3D constraint must surface
        as a clear error at Solver construction."""
        from dvfopt import JdetConstraint3D
        from dvfopt.exceptions import IncompatibleConstraintError

        with pytest.raises((IncompatibleConstraintError, TypeError)):
            Solver(
                constraint=JdetConstraint3D(shape=(3, 6, 6)),
                objective=NoneObjective(),
                strategy=NMVFStrategy(),
            )


# ---------------------------------------------------------------------------
# Public-API smoke: the new names are importable from the top-level package.
# ---------------------------------------------------------------------------


class TestPublicAPISmoke:
    def test_all_new_names_on_dvfopt(self):
        for name in (
            'NMVFStrategy',
            'HarmonicALMBarrierStrategy',
            'HarmonicALMRefineRepairStrategy',
            'SchwarzHarmonicALMRefineRepairStrategy',
            'HarmonicALMBarrier3DStrategy',
            'HarmonicALMRefineRepair3DStrategy',
            'SchwarzHarmonicALMRefineRepair3DStrategy',
        ):
            assert hasattr(dvfopt, name), f'dvfopt.{name} missing'
            assert name in dvfopt.__all__, f'dvfopt.__all__ missing {name!r}'

    def test_all_old_names_still_on_dvfopt(self):
        for name in (
            'M10Strategy',
            'M14Strategy',
            'M14SchwarzStrategy',
            'M10TetStrategy',
            'M14TetStrategy',
            'M14Schwarz3DStrategy',
        ):
            assert hasattr(dvfopt, name), f'dvfopt.{name} missing'
            assert name in dvfopt.__all__, f'dvfopt.__all__ missing {name!r}'


# Reference imports so tools don't strip them as unused (BarrierStrategy
# is here as a sanity check that the strategy module still imports
# cleanly after the rename — if NMVF / new names broke the strategies
# package, this import would have raised at collection time already).
_ = BarrierStrategy
