"""Integration tests for the 3D iterative solver.

These tests verify the full 3D pipeline works end-to-end:
after correction, zero negative Jacobian determinants remain.
"""

import numpy as np
import pytest

from dvfopt.jacobian.numpy_jdet import jacobian_det3D
from dvfopt.dvf.generation import generate_random_dvf_3d


THRESHOLD = 0.01


def _assert_no_neg_jdet_3d(phi, threshold=THRESHOLD):
    jdet = jacobian_det3D(phi)
    n_neg = int((jdet <= 0).sum())
    min_j = float(jdet.min())
    assert n_neg == 0, f"Expected 0 negative Jdet voxels, got {n_neg} (min={min_j:.6f})"
    assert min_j >= threshold - 1e-5, f"Expected min Jdet >= {threshold}, got {min_j:.6f}"


class TestIterative3D:
    def test_identity_unchanged(self):
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 4, 4, 4), dtype=np.float64)
        phi = iterative_3d(d, verbose=0, max_iterations=5)
        assert phi.shape == (3, 4, 4, 4)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_output_shape(self):
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 4, 5, 6), dtype=np.float64)
        phi = iterative_3d(d, verbose=0, max_iterations=5)
        assert phi.shape == (3, 4, 5, 6)

    def test_corrects_single_spike(self):
        """A single large displacement spike should be correctable."""
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0  # large dx spike
        jdet_before = jacobian_det3D(d)
        assert jdet_before.min() < THRESHOLD

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        _assert_no_neg_jdet_3d(phi)

    def test_corrects_random_field(self):
        """Random 3D DVF with negative Jacobians should be fully corrected."""
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = generate_random_dvf_3d((3, 5, 5, 5), max_magnitude=2.0, seed=42)
        jdet_before = jacobian_det3D(d)
        if jdet_before.min() >= THRESHOLD:
            pytest.skip("Random 3D DVF has no negative Jacobians")

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=1000)
        _assert_no_neg_jdet_3d(phi)

    def test_opposing_spikes(self):
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 2] = 3.0
        d[2, 3, 3, 3] = -3.0
        jdet_before = jacobian_det3D(d)
        assert jdet_before.min() < THRESHOLD

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        _assert_no_neg_jdet_3d(phi)

    def test_displacement_stays_close(self):
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 4.0
        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        max_change = np.abs(phi - d).max()
        assert max_change < 10.0

    def test_non_cubic_grid(self):
        """Non-cubic 3D grid triggers full-grid fallback path."""
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 3, 5, 7), dtype=np.float64)
        d[2, 1, 2, 3] = 4.0
        d[2, 1, 2, 4] = -4.0
        jdet_before = jacobian_det3D(d)
        if jdet_before.min() >= THRESHOLD:
            pytest.skip("No negative Jdet in non-cubic field")

        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD, max_iterations=500)
        _assert_no_neg_jdet_3d(phi)


class TestMaxWindowVoxels:
    """The ``max_window_voxels`` cap must be respected end-to-end."""

    def test_cap_respected_and_corrects(self, monkeypatch):
        """With a tight voxel cap the solver should still fix a small fold,
        and no window passed to the SLSQP inner call may exceed the cap."""
        from dvfopt.core.slsqp import iterative3d as it3
        from dvfopt.core import solver3d as s3

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 3.5
        assert jacobian_det3D(d).min() < THRESHOLD

        max_obs = {"v": 0}
        orig_opt = s3._optimize_single_window_3d

        def spy(phi_sub_flat, phi_init_sub_flat, subvolume_size, *a, **kw):
            sz, sy, sx = subvolume_size
            v = int(sz) * int(sy) * int(sx)
            max_obs["v"] = max(max_obs["v"], v)
            return orig_opt(phi_sub_flat, phi_init_sub_flat,
                           subvolume_size, *a, **kw)

        monkeypatch.setattr(s3, "_optimize_single_window_3d", spy)

        CAP = 80
        phi = it3.iterative_3d(d, verbose=0, threshold=THRESHOLD,
                               max_iterations=200, max_window_voxels=CAP)
        _assert_no_neg_jdet_3d(phi)
        assert max_obs["v"] <= CAP, \
            f"A window of {max_obs['v']} voxels exceeded cap {CAP}"

    def test_cap_with_aspect_preserved(self):
        """Budget shrink should preserve aspect ratio (no axis collapse)."""
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        # Elongated grid: any window will prefer matching aspect.
        d = np.zeros((3, 4, 4, 20), dtype=np.float64)
        d[2, 2, 2, 10] = 3.5
        assert jacobian_det3D(d).min() < THRESHOLD
        phi = iterative_3d(d, verbose=0, threshold=THRESHOLD,
                           max_iterations=200, max_window_voxels=60)
        _assert_no_neg_jdet_3d(phi)

    def test_no_cap_equivalent_to_default(self):
        """``max_window_voxels=None`` should match unset default behaviour."""
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = np.zeros((3, 5, 5, 5), dtype=np.float64)
        d[2, 2, 2, 2] = 3.0
        phi_a = iterative_3d(d, verbose=0, threshold=THRESHOLD,
                             max_iterations=200)
        phi_b = iterative_3d(d, verbose=0, threshold=THRESHOLD,
                             max_iterations=200, max_window_voxels=None)
        np.testing.assert_allclose(phi_a, phi_b, atol=1e-12)


class TestVoxelCapEscalation:
    """Livelock escalation should raise the effective voxel cap."""

    def test_ceiling_is_respected(self, monkeypatch):
        """The escalated cap must never exceed the ceiling."""
        from dvfopt.core.slsqp import iterative3d as it3
        from dvfopt.core import solver3d as s3

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 3.5
        assert jacobian_det3D(d).min() < THRESHOLD

        max_obs = {"v": 0}
        orig = s3._optimize_single_window_3d

        def spy(pf, pif, sub, *a, **kw):
            v = int(sub[0]) * int(sub[1]) * int(sub[2])
            max_obs["v"] = max(max_obs["v"], v)
            return orig(pf, pif, sub, *a, **kw)

        monkeypatch.setattr(s3, "_optimize_single_window_3d", spy)

        CEIL = 120
        phi = it3.iterative_3d(
            d, verbose=0, threshold=THRESHOLD, max_iterations=200,
            max_window_voxels=40, max_window_voxels_ceiling=CEIL,
            voxel_cap_stall_threshold=2,
        )
        # Correctness first.
        _assert_no_neg_jdet_3d(phi)
        # Never exceed the hard ceiling.
        assert max_obs["v"] <= CEIL, \
            f"Max window {max_obs['v']} > ceiling {CEIL}"

    def test_no_escalation_when_ceiling_none(self, monkeypatch):
        """Without a ceiling the initial cap must stay in force."""
        from dvfopt.core.slsqp import iterative3d as it3
        from dvfopt.core import solver3d as s3

        d = np.zeros((3, 6, 6, 6), dtype=np.float64)
        d[2, 3, 3, 3] = 3.5

        max_obs = {"v": 0}
        orig = s3._optimize_single_window_3d

        def spy(pf, pif, sub, *a, **kw):
            v = int(sub[0]) * int(sub[1]) * int(sub[2])
            max_obs["v"] = max(max_obs["v"], v)
            return orig(pf, pif, sub, *a, **kw)

        monkeypatch.setattr(s3, "_optimize_single_window_3d", spy)

        CAP = 60
        it3.iterative_3d(
            d, verbose=0, threshold=THRESHOLD, max_iterations=50,
            max_window_voxels=CAP, max_window_voxels_ceiling=None,
            voxel_cap_stall_threshold=2,
        )
        assert max_obs["v"] <= CAP, \
            f"Max window {max_obs['v']} exceeded static cap {CAP}"
