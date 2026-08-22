"""Integration tests for the 3D penalty -> log-barrier solvers.

Mirrors test_integration_3d.py but exercises the full-grid L-BFGS-B
(numpy/scipy) and torch L-BFGS backends.
"""

import numpy as np
import pytest

from dvfopt.dvf.generation import generate_random_dvf_3d
from dvfopt.jacobian.numpy_jdet import jacobian_det3D

THRESHOLD = 0.01


def _assert_no_neg_jdet_3d(phi, threshold=THRESHOLD):
    jdet = jacobian_det3D(phi)
    n_neg = int((jdet <= 0).sum())
    min_j = float(jdet.min())
    assert n_neg == 0, f"Expected 0 negative Jdet voxels, got {n_neg} (min={min_j:.6f})"
    assert min_j >= threshold - 1e-5, f"Expected min Jdet >= {threshold}, got {min_j:.6f}"


class TestBarrier3DNumpy:
    def test_identity_unchanged(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = np.zeros((3, 4, 4, 4), dtype=np.float64)
        phi = iterative_3d_barrier(d, verbose=0)
        assert phi.shape == (3, 4, 4, 4)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_output_shape(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = np.zeros((3, 4, 5, 6), dtype=np.float64)
        phi = iterative_3d_barrier(d, verbose=0)
        assert phi.shape == (3, 4, 5, 6)

    def test_corrects_single_spike(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier(
            d, verbose=0, threshold=THRESHOLD, lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
        )
        _assert_no_neg_jdet_3d(phi)

    def test_corrects_random_field(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = generate_random_dvf_3d((3, 5, 5, 5), max_magnitude=2.0, seed=42)
        if jacobian_det3D(d).min() >= THRESHOLD:
            pytest.skip("Random DVF has no negative Jacobians")

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_3d(phi)

    def test_opposing_spikes(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 2] = 3.0
        d[2, 3, 3, 3] = -3.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_3d(phi)

    def test_displacement_stays_close(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        assert np.abs(phi - d).max() < 10.0

    def test_non_cubic_grid(self):
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        d = np.zeros((3, 3, 5, 7), dtype=np.float64)
        d[2, 1, 2, 3] = 4.0
        d[2, 1, 2, 4] = -4.0
        if jacobian_det3D(d).min() >= THRESHOLD:
            pytest.skip("No negative Jdet in non-cubic field")

        phi = iterative_3d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_3d(phi)

    def test_full_grid_mode_returns(self):
        """F1 regression: windowed=False used to raise RuntimeError after
        the optimisation because it demanded 'l2'/'phi_flat' keys that
        run_penalty_barrier_lbfgs never records. It must return a field."""
        from dvfopt.core.barrier.jdet3d import iterative_3d_barrier

        rng = np.random.default_rng(0)
        d = rng.standard_normal((3, 4, 5, 5)) * 0.01
        phi = iterative_3d_barrier(d, windowed=False, verbose=0)
        assert phi.shape == (3, 4, 5, 5)
        assert np.all(np.isfinite(phi))


class TestBarrier3DTorch:
    def setup_method(self):
        pytest.importorskip('torch')

    def test_identity_unchanged(self):
        from dvfopt.core.barrier.jdet3d_torch import iterative_3d_barrier_torch

        d = np.zeros((3, 4, 4, 4), dtype=np.float64)
        phi = iterative_3d_barrier_torch(d, verbose=0, device="cpu")
        assert phi.shape == (3, 4, 4, 4)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_corrects_single_spike(self):
        from dvfopt.core.barrier.jdet3d_torch import iterative_3d_barrier_torch

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier_torch(
            d,
            verbose=0,
            threshold=THRESHOLD,
            device="cpu",
            lam_schedule=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6),
        )
        _assert_no_neg_jdet_3d(phi)

    def test_opposing_spikes(self):
        from dvfopt.core.barrier.jdet3d_torch import iterative_3d_barrier_torch

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 2] = 3.0
        d[2, 3, 3, 3] = -3.0
        assert jacobian_det3D(d).min() < THRESHOLD

        phi = iterative_3d_barrier_torch(d, verbose=0, threshold=THRESHOLD, device="cpu")
        _assert_no_neg_jdet_3d(phi)

    def test_no_torch_raises_clear_error(self):
        """F10(c) regression: the module used to hard-import torch at the
        top level. It must be importable without torch, with the public
        entry raising a friendly ImportError at call time."""
        from dvfopt.core import iterative3d_barrier_torch as mod

        original = mod.torch
        mod.torch = None
        try:
            with pytest.raises(ImportError, match="torch"):
                mod.iterative_3d_barrier_torch(np.zeros((3, 4, 4, 4)))
        finally:
            mod.torch = original


class TestBatchedBarrier3DTorchInfeasiblePatch:
    """F4 regression (3D sibling of the 2D test): the batched barrier
    phase must leave patches that never reached feasibility at their best
    penalty-phase state instead of dragging them back to init via the
    shared (previously unmasked) data term."""

    HARD = (2, 8, 2, 8, 2, 8)
    EASY = (10, 14, 10, 14, 10, 14)  # fold-free -> feasible from the start
    D = 16
    H = 16
    W = 16
    LAM_SCHEDULE = (0.1, 0.5)  # too weak to clear a magnitude-5 spike

    def setup_method(self):
        pytest.importorskip('torch')

    def _build(self):
        import torch

        phi = torch.zeros((3, self.D, self.H, self.W), dtype=torch.float64)
        phi[2, 5, 5, 5] = 5.0  # deep dx spike inside HARD
        return phi

    def _hard_interior_min_j(self, phi):
        from dvfopt.core.barrier.jdet3d_torch import _jdet_3d_torch

        z0, z1, y0, y1, x0, x1 = self.HARD
        j = _jdet_3d_torch(phi)
        return float(j[z0 + 1 : z1, y0 + 1 : y1, x0 + 1 : x1].min().item())

    def _run(self, mu_schedule):
        import torch

        from dvfopt.core.barrier.jdet3d_torch import _optimize_batch_3d_torch

        phi_full = self._build()
        _optimize_batch_3d_torch(
            phi_full,
            [self.HARD, self.EASY],
            (self.D, self.H, self.W),
            THRESHOLD,
            1e-3,
            self.LAM_SCHEDULE,
            mu_schedule,
            50,
            torch.device('cpu'),
            torch.float64,
        )
        return phi_full

    def test_infeasible_patch_keeps_penalty_progress(self):
        import torch

        phi_init = self._build()
        phi_pen = self._run(mu_schedule=())  # penalty phase only
        phi_bar = self._run(mu_schedule=(1e-1, 1e-2))

        j_init = self._hard_interior_min_j(phi_init)
        j_pen = self._hard_interior_min_j(phi_pen)
        j_bar = self._hard_interior_min_j(phi_bar)

        assert j_pen > j_init + 0.1, f"penalty made no progress (init={j_init}, pen={j_pen})"
        assert j_pen < THRESHOLD, f"hard patch unexpectedly feasible after penalty (j={j_pen})"

        z0, z1, y0, y1, x0, x1 = self.HARD
        sl = (slice(None), slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
        assert float((phi_bar[sl] - phi_init[sl]).abs().max().item()) > 1e-3, (
            "infeasible patch was reverted to init by the barrier phase"
        )
        assert j_bar >= j_pen - 1e-9, (
            f"barrier phase degraded the infeasible patch: pen={j_pen}, bar={j_bar}"
        )
        torch.testing.assert_close(phi_bar[sl], phi_pen[sl])
