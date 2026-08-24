"""Tests for the input-validation layer.

Covers:

* :func:`dvfopt.validation.validate_dvf` directly, on each failure mode.
* End-to-end behavior via :meth:`DVFopt.fit` and :meth:`Solver.fit`
  on the same set of inputs — ensures the validator is actually
  reached from both entry points.

The stress matrix mirrors the manual sweep from the
"does DVFopt handle inputs gracefully" audit: list-of-lists,
missing channel dim, zero-size, NaN/Inf, singleton-D layouts, etc.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dvfopt import (
    DVFopt,
    DVFoptConfig,
    L2Objective,
    SimplexConstraint2D,
    Solver,
    SolverConfigError,
    coerce_to_ndarray,
    validate_dvf,
    validate_finite,
    validate_spatial_min_size,
)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TestCoerceToNdarray:
    def test_converts_list_of_lists(self):
        out = coerce_to_ndarray([[[0.0] * 4] * 4] * 2)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == (2, 4, 4)

    def test_upcasts_int_to_float64(self):
        out = coerce_to_ndarray(np.zeros((2, 4, 4), dtype=np.int16))
        assert out.dtype == np.float64

    def test_returns_writeable_copy_of_readonly_input(self):
        arr = np.zeros((2, 4, 4))
        arr.setflags(write=False)
        out = coerce_to_ndarray(arr)
        assert out.flags.writeable

    def test_rejects_non_array_like(self):
        with pytest.raises(TypeError, match='convertible to a numpy array'):
            coerce_to_ndarray('not an array')


class TestValidateFinite:
    def test_passes_finite(self):
        validate_finite(np.zeros((2, 4, 4)))  # no raise

    def test_reports_nan_count(self):
        arr = np.zeros((2, 4, 4))
        arr[0, 0, 0] = np.nan
        arr[1, 1, 1] = np.nan
        with pytest.raises(ValueError, match='2 NaN'):
            validate_finite(arr)

    def test_reports_inf_count(self):
        arr = np.zeros((2, 4, 4))
        arr[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match='1 Inf'):
            validate_finite(arr)


class TestValidateSpatialMinSize:
    def test_passes_above_min(self):
        validate_spatial_min_size((10, 10), min_size=3)
        validate_spatial_min_size((5, 10, 10), min_size=3)

    def test_rejects_below_min(self):
        with pytest.raises(SolverConfigError, match='H=1 is below the minimum'):
            validate_spatial_min_size((1, 10), min_size=3)
        with pytest.raises(SolverConfigError, match='W=0 is below the minimum'):
            validate_spatial_min_size((10, 0), min_size=3)


# ---------------------------------------------------------------------------
# Top-level validate_dvf
# ---------------------------------------------------------------------------


class TestValidateDvf2D:
    def test_canonical_passes(self):
        out = validate_dvf(np.zeros((2, 4, 4)), dim=2)
        assert out.shape == (2, 4, 4)
        assert out.dtype == np.float64

    def test_3channel_drops_dz(self):
        arr = np.zeros((3, 4, 4))
        arr[0] = 99.0  # dz — should be dropped
        out = validate_dvf(arr, dim=2)
        assert out.shape == (2, 4, 4)
        assert (out == 0.0).all()

    def test_singleton_d_promoted(self):
        out = validate_dvf(np.zeros((2, 1, 4, 4)), dim=2)
        assert out.shape == (2, 4, 4)

    def test_3channel_singleton_d_promoted_drops_dz(self):
        arr = np.zeros((3, 1, 4, 4))
        arr[0, 0] = 99.0
        out = validate_dvf(arr, dim=2)
        assert out.shape == (2, 4, 4)
        assert (out == 0.0).all()

    def test_strict_mode_rejects_3channel(self):
        with pytest.raises(SolverConfigError):
            validate_dvf(np.zeros((3, 4, 4)), dim=2, accept_3channel_2d=False)

    def test_rejects_bad_shape_with_message(self):
        with pytest.raises(SolverConfigError, match='not a valid 2D'):
            validate_dvf(np.zeros((5, 4, 4)), dim=2)

    def test_rejects_too_small(self):
        with pytest.raises(SolverConfigError, match='below the minimum'):
            validate_dvf(np.zeros((2, 2, 2)), dim=2)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match='NaN'):
            validate_dvf(np.full((2, 4, 4), np.nan), dim=2)


class TestValidateDvf3D:
    def test_canonical_passes(self):
        out = validate_dvf(np.zeros((3, 4, 5, 5)), dim=3)
        assert out.shape == (3, 4, 5, 5)

    def test_rejects_2d_shape_for_3d_dim(self):
        with pytest.raises(SolverConfigError, match='not a valid 3D'):
            validate_dvf(np.zeros((2, 4, 4)), dim=3)


# ---------------------------------------------------------------------------
# End-to-end: DVFopt.fit + Solver.fit
# ---------------------------------------------------------------------------


def _silent_opt():
    return DVFopt(DVFoptConfig(constraint='simplex', solver='barrier', objective='l1', verbose=0))


class TestDvfoptFitGracefulRejection:
    """Every input mode that previously failed cryptically should now
    raise a typed, actionable error."""

    def test_list_of_lists_works(self):
        out = _silent_opt().fit([[[0.0] * 8] * 8] * 2)
        assert out.corrected.shape == (2, 8, 8)

    def test_missing_channel_dim_rejected(self):
        with pytest.raises(SolverConfigError, match='Accepted'):
            _silent_opt().fit(np.zeros((8, 8)))

    def test_zero_size_rejected(self):
        with pytest.raises(SolverConfigError, match='below the minimum'):
            _silent_opt().fit(np.zeros((2, 0, 8)))

    def test_too_small_rejected(self):
        with pytest.raises(SolverConfigError, match='below the minimum'):
            _silent_opt().fit(np.zeros((2, 1, 1)))

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match='NaN'):
            _silent_opt().fit(np.full((2, 8, 8), np.nan))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match='Inf'):
            _silent_opt().fit(np.full((2, 8, 8), np.inf))

    def test_wrong_channel_count_rejected(self):
        with pytest.raises(SolverConfigError, match='Accepted'):
            _silent_opt().fit(np.zeros((4, 8, 8)))

    def test_float32_accepted(self):
        out = _silent_opt().fit(np.zeros((2, 8, 8), dtype=np.float32))
        assert out.corrected.shape == (2, 8, 8)

    def test_readonly_accepted(self):
        arr = np.zeros((2, 8, 8))
        arr.setflags(write=False)
        out = _silent_opt().fit(arr)
        assert out.corrected.shape == (2, 8, 8)

    def test_3channel_singleton_d_canonical_accepted(self):
        out = _silent_opt().fit(np.zeros((3, 1, 8, 8)))
        assert out.corrected.shape == (3, 1, 8, 8)

    def test_3channel_2d_accepted_and_returned_as_3channel(self):
        out = _silent_opt().fit(np.zeros((3, 8, 8)))
        assert out.corrected.shape == (3, 8, 8)

    def test_input_not_mutated(self):
        """``fit`` defensively copies — the input array is untouched
        even on a successful run."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rng = np.random.default_rng(7)
            phi = np.stack([rng.normal(0, 0.4, (10, 10)), rng.normal(0, 0.4, (10, 10))])
            phi_copy = phi.copy()
            _silent_opt().fit(phi)
            np.testing.assert_array_equal(phi, phi_copy)


class TestSolverFitGracefulRejection:
    """Same matrix via the Solver-level entry point."""

    def _solver(self):
        return Solver(
            constraint=SimplexConstraint2D((8, 8)),
            objective=L2Objective(),
            strategy=__import__('dvfopt').BarrierStrategy(),
        )

    def test_constraint_coerce_handles_3channel(self):
        """A (3, H, W) input is coerced to the canonical (2, H, W) for
        the strategy, then restored: corrected has the INPUT's shape
        with the dz channel passed through unchanged."""
        phi = np.zeros((3, 8, 8))
        phi[0] = 4.0  # dz sentinel
        out = self._solver().fit(phi)
        assert out.corrected.shape == (3, 8, 8)
        np.testing.assert_array_equal(out.corrected[0], 4.0)

    def test_constraint_coerce_rejects_nan(self):
        with pytest.raises(ValueError, match='NaN'):
            self._solver().fit(np.full((2, 8, 8), np.nan))

    def test_constraint_coerce_rejects_wrong_spatial_size(self):
        with pytest.raises(SolverConfigError, match='does not match'):
            self._solver().fit(np.zeros((2, 10, 10)))  # constraint built for (8, 8)
