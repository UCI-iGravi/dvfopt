"""Tests for solver convergence properties.

Verifies that the iterative solver:
- Monotonically reduces the number of negative Jdet pixels (or stays flat)
- Produces bounded L2 error
- Converges within a reasonable iteration count
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.dvf.generation import generate_random_dvf
from dvfopt.core.solver import _init_phi, _update_metrics, _setup_accumulators


class TestConvergenceProperties:
    @staticmethod
    def _run_and_collect_metrics(deformation, **kwargs):
        """Run solver and return per-iteration metrics."""
        from dvfopt.core.iterative import iterative_with_jacobians2

        # Collect Jdet snapshots by running manually
        phi_init_snap = np.stack([deformation[1, 0], deformation[2, 0]])
        jdet_before = jacobian_det2D(phi_init_snap)
        neg_before = int((jdet_before <= 0).sum())
        min_before = float(jdet_before.min())

        phi = iterative_with_jacobians2(deformation, verbose=0, **kwargs)

        jdet_after = jacobian_det2D(phi)
        neg_after = int((jdet_after <= 0).sum())
        min_after = float(jdet_after.min())
        l2_error = float(np.sqrt(np.sum((phi - phi_init_snap) ** 2)))

        return {
            "neg_before": neg_before,
            "neg_after": neg_after,
            "min_before": min_before,
            "min_after": min_after,
            "l2_error": l2_error,
            "phi": phi,
        }

    def test_neg_count_decreases(self):
        """After correction, neg count should be <= before."""
        d = generate_random_dvf((3, 1, 10, 10), max_magnitude=3.0, seed=42).astype(np.float64)
        m = self._run_and_collect_metrics(d, threshold=0.01, max_iterations=500)
        assert m["neg_after"] <= m["neg_before"]

    def test_min_jdet_improves(self):
        """After correction, min Jdet should be >= before."""
        d = generate_random_dvf((3, 1, 10, 10), max_magnitude=3.0, seed=42).astype(np.float64)
        m = self._run_and_collect_metrics(d, threshold=0.01, max_iterations=500)
        assert m["min_after"] >= m["min_before"]

    def test_l2_error_bounded(self):
        """L2 error should be finite and reasonable."""
        d = generate_random_dvf((3, 1, 10, 10), max_magnitude=3.0, seed=42).astype(np.float64)
        m = self._run_and_collect_metrics(d, threshold=0.01, max_iterations=500)
        assert np.isfinite(m["l2_error"])
        # L2 error should be small relative to grid size
        H, W = 10, 10
        max_possible_l2 = np.sqrt(2 * H * W) * 10  # generous upper bound
        assert m["l2_error"] < max_possible_l2

    def test_identity_zero_error(self):
        """Identity field should have zero L2 error (no correction needed)."""
        d = np.zeros((3, 1, 8, 8), dtype=np.float64)
        m = self._run_and_collect_metrics(d, threshold=0.01, max_iterations=10)
        np.testing.assert_allclose(m["l2_error"], 0.0, atol=1e-10)
        assert m["neg_before"] == 0
        assert m["neg_after"] == 0

    def test_multiple_seeds_all_converge(self):
        """Multiple random seeds should all converge to zero negative Jdet."""
        for seed in [10, 20, 30, 40, 50]:
            d = generate_random_dvf((3, 1, 8, 8), max_magnitude=2.0, seed=seed).astype(np.float64)
            jdet_before = jacobian_det2D(d[[1, 2], 0])
            if jdet_before.min() >= 0.01:
                continue  # skip if already clean
            m = self._run_and_collect_metrics(d, threshold=0.01, max_iterations=500)
            assert m["neg_after"] == 0, f"seed={seed}: {m['neg_after']} negative Jdet remain"


class TestUpdateMetricsAccumulation:
    """Verify that _update_metrics correctly tracks per-iteration state."""

    def test_multiple_calls_append(self):
        phi = np.zeros((2, 6, 6))
        phi_init = phi.copy()
        num_neg, _, _, min_jdet, _ = _setup_accumulators()
        error_list = []

        for _ in range(5):
            _update_metrics(phi, phi_init, False, False,
                            num_neg, min_jdet, error_list)

        assert len(num_neg) == 5
        assert len(min_jdet) == 5
        assert len(error_list) == 5

    def test_error_increases_with_displacement(self):
        phi = np.zeros((2, 6, 6))
        phi_init = phi.copy()
        error_list = []
        num_neg = []
        min_jdet = []

        _update_metrics(phi, phi_init, False, False,
                        num_neg, min_jdet, error_list)
        e0 = error_list[-1]

        phi[0, 3, 3] = 1.0  # introduce displacement
        _update_metrics(phi, phi_init, False, False,
                        num_neg, min_jdet, error_list)
        e1 = error_list[-1]

        assert e1 > e0
