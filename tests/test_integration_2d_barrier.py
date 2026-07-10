"""Integration tests for the 2D penalty -> log-barrier solvers (numpy + torch)."""

import numpy as np
import pytest

from dvfopt.dvf.generation import generate_random_dvf
from dvfopt.jacobian.numpy_jdet import jacobian_det2D

THRESHOLD = 0.01


def _make_folded_field(H=10, W=10):
    return generate_random_dvf((3, 1, H, W), max_magnitude=3.0, seed=42).astype(np.float64)


def _assert_no_neg_jdet_2d(phi, threshold=THRESHOLD):
    jdet = jacobian_det2D(phi)
    n_neg = int((jdet <= 0).sum())
    min_j = float(jdet.min())
    assert n_neg == 0, f"Expected 0 negative Jdet, got {n_neg} (min={min_j:.6f})"
    assert min_j >= threshold - 1e-5, f"Expected min Jdet >= {threshold}, got {min_j:.6f}"


class TestBarrier2DNumpy:
    def test_identity_unchanged(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = np.zeros((3, 1, 8, 8), dtype=np.float64)
        phi = iterative_2d_barrier(d, verbose=0)
        assert phi.shape == (2, 8, 8)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_output_shape(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(8, 12)
        phi = iterative_2d_barrier(d, verbose=0)
        assert phi.shape == (2, 8, 12)

    def test_corrects_negative_jacobians(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(10, 10)
        if jacobian_det2D(d[[1, 2], 0]).min() >= THRESHOLD:
            pytest.skip("Test field already feasible")
        phi = iterative_2d_barrier(d, verbose=0, threshold=THRESHOLD)
        _assert_no_neg_jdet_2d(phi)

    def test_displacement_stays_close(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(10, 10)
        phi_init = np.stack([d[1, 0], d[2, 0]])
        phi = iterative_2d_barrier(d, verbose=0, threshold=THRESHOLD)
        assert np.linalg.norm(phi - phi_init) < 50.0

    def test_max_iterations_zero_does_not_crash(self):
        """F10(b) regression: windowed mode with max_iterations<=0 used to
        hit the not-converged log with unbound cur_neg/cur_min
        (UnboundLocalError). It must return the (uncorrected) field."""
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(8, 8)
        phi = iterative_2d_barrier(d, verbose=1, max_iterations=0, windowed=True)
        assert phi.shape == (2, 8, 8)

    def test_full_grid_mode_runs(self):
        """Regression: the full-grid (windowed=False) path was migrated to
        the unified _barrier_core homotopy. Confirm it still produces a
        valid output and corrects folds."""
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier

        d = _make_folded_field(8, 8)
        phi = iterative_2d_barrier(
            d, verbose=0, threshold=THRESHOLD, windowed=False, max_minimize_iter=100
        )
        assert phi.shape == (2, 8, 8)
        # Result should be at least as good as the input.
        init_min = float(jacobian_det2D(d[[1, 2], 0]).min())
        final_min = float(jacobian_det2D(phi).min())
        assert final_min >= init_min - 1e-6, (
            f"full-grid mode made the field worse: init_min={init_min}, final_min={final_min}"
        )


class TestBarrier2DTorch:
    def setup_method(self):
        pytest.importorskip('torch')

    def test_identity_unchanged(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier_torch

        d = np.zeros((3, 1, 8, 8), dtype=np.float64)
        phi = iterative_2d_barrier_torch(d, verbose=0, device="cpu")
        assert phi.shape == (2, 8, 8)
        np.testing.assert_allclose(phi, 0.0, atol=1e-3)

    def test_corrects_negative_jacobians(self):
        from dvfopt.core.iterative2d_barrier import iterative_2d_barrier_torch

        d = _make_folded_field(10, 10)
        if jacobian_det2D(d[[1, 2], 0]).min() >= THRESHOLD:
            pytest.skip("Test field already feasible")
        phi = iterative_2d_barrier_torch(d, verbose=0, threshold=THRESHOLD, device="cpu")
        _assert_no_neg_jdet_2d(phi)


class TestBatchedBarrier2DTorchInfeasiblePatch:
    """F4 regression: in the batched torch solver's barrier (mu) phase the
    shared data term used to cover ALL K patches' free DOFs while the
    barrier term only covered feasible patches — LBFGS dragged infeasible
    patches' variables back to init, undoing the penalty phase's progress.

    Construction: a 2-patch batch where patch 0 holds a deep fold that a
    deliberately tiny lam schedule cannot clear, and patch 1 is fold-free
    (feasible from the start, so the barrier phase runs). The infeasible
    patch's output must equal its best penalty-phase state, not init.
    """

    HARD = (2, 8, 2, 8)  # bbox containing the deep fold
    EASY = (12, 17, 12, 17)  # fold-free bbox — feasible_k[1] is True throughout
    H = 20
    W = 20
    LAM_SCHEDULE = (0.1, 0.5)  # too weak to clear a magnitude-5 spike

    def setup_method(self):
        pytest.importorskip('torch')

    def _build(self):
        import torch

        phi = torch.zeros((2, self.H, self.W), dtype=torch.float64)
        phi[1, 5, 5] = 5.0  # deep dx spike -> deep fold inside HARD
        return phi

    def _hard_interior_min_j(self, phi):
        import torch

        from dvfopt.core.iterative2d_barrier import _jdet_2d_torch

        y0, y1, x0, x1 = self.HARD
        j = _jdet_2d_torch(phi)
        return float(j[y0 + 1 : y1, x0 + 1 : x1].min().item())

    def _run(self, mu_schedule):
        import torch

        from dvfopt.core.iterative2d_barrier import _optimize_batch_2d_torch

        phi_full = self._build()
        _optimize_batch_2d_torch(
            phi_full,
            [self.HARD, self.EASY],
            (self.H, self.W),
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
        # mu_schedule=() isolates the penalty phase (barrier loop body never runs).
        phi_pen = self._run(mu_schedule=())
        phi_bar = self._run(mu_schedule=(1e-1, 1e-2))

        j_init = self._hard_interior_min_j(phi_init)
        j_pen = self._hard_interior_min_j(phi_pen)
        j_bar = self._hard_interior_min_j(phi_bar)

        # Scenario preconditions: penalty made real progress on the hard
        # patch but did NOT reach feasibility (so the patch is excluded
        # from the barrier term).
        assert j_pen > j_init + 0.1, f"penalty made no progress (init={j_init}, pen={j_pen})"
        assert j_pen < THRESHOLD, f"hard patch unexpectedly feasible after penalty (j={j_pen})"

        # The barrier phase must not touch the infeasible patch: output
        # differs from init and min_J is no worse than after the penalty
        # phase (pre-fix it reverted exactly to init, j_bar == j_init).
        y0, y1, x0, x1 = self.HARD
        hard_bar = phi_bar[:, y0 : y1 + 1, x0 : x1 + 1]
        hard_init = phi_init[:, y0 : y1 + 1, x0 : x1 + 1]
        hard_pen = phi_pen[:, y0 : y1 + 1, x0 : x1 + 1]
        assert float((hard_bar - hard_init).abs().max().item()) > 1e-3, (
            "infeasible patch was reverted to init by the barrier phase"
        )
        assert j_bar >= j_pen - 1e-9, (
            f"barrier phase degraded the infeasible patch: pen={j_pen}, bar={j_bar}"
        )
        torch.testing.assert_close(hard_bar, hard_pen)
