"""Tests for the promoted 3D (simplex (3D)) SLP solvers and SLPStrategy 3D path.

The 3D SLP family (lp_direct_6tet / cluster_lp_6tet / _gpu_untangle_3d)
was promoted from ``research/strict_feasibility_3d`` into
``dvfopt.core.slp``; ``SLPStrategy`` gained a simplex (3D) dispatch. These tests
pin the promoted API and end-to-end feasibility on a small synthetic
fold. The research paths remain importable via back-compat shims.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

THRESHOLD = 0.01


def _folded_volume():
    """Small (3, 4, 6, 6) volume with a punched central fold."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.03, (3, 4, 6, 6))
    phi[1, 1:3, 2:4, 2:4] -= 1.2
    phi[2, 1:3, 2:4, 2:4] -= 1.2
    assert (six_tet_volumes_3d(phi) <= 0).any(), 'fixture must actually fold'
    return phi


class TestSlpIter3D:
    def test_reaches_feasibility_from_m10_seed(self):
        from dvfopt.core.slp import slp_iter_3d

        phi = _folded_volume()
        phi_out, info = slp_iter_3d(phi, threshold=THRESHOLD, seed='m10')
        V = six_tet_volumes_3d(phi_out)
        assert float(V.min()) >= THRESHOLD - 1e-5, f'min_T={V.min():.6f}'
        assert info['L1_dev'] > 0.0

    def test_array_seed_accepted(self):
        from dvfopt.core.slp import slp_iter_3d

        phi = _folded_volume()
        # Zero field is trivially feasible (identity map) — valid seed array.
        seed = np.zeros_like(phi)
        phi_out, info = slp_iter_3d(phi, threshold=THRESHOLD, seed=seed)
        V = six_tet_volumes_3d(phi_out)
        assert float(V.min()) >= THRESHOLD - 1e-5


class TestClusterSlpIter3D:
    def test_reaches_feasibility(self):
        from dvfopt.core.slp import cluster_slp_iter_3d

        phi = _folded_volume()
        phi_out, info = cluster_slp_iter_3d(phi, threshold=THRESHOLD, inner_seed='m10')
        V = six_tet_volumes_3d(phi_out)
        assert float(V.min()) >= THRESHOLD - 1e-5, f'min_T={V.min():.6f}'
        assert info['total_cluster_solves'] >= 1


class TestSLPStrategy3D:
    def test_solver_composition_reaches_feasibility(self):
        from dvfopt import L1Objective, SimplexConstraint3D, SLPStrategy, Solver

        phi = _folded_volume()
        result = Solver(
            constraint=SimplexConstraint3D(shape=phi.shape[1:]),
            objective=L1Objective(),
            strategy=SLPStrategy(),
            threshold=THRESHOLD,
        ).fit(phi)
        V = six_tet_volumes_3d(result.corrected)
        assert float(V.min()) >= THRESHOLD - 1e-5

    def test_from_spec_string_composition(self):
        from dvfopt import Solver

        phi = _folded_volume()
        result = Solver.from_spec(
            constraint='simplex_3d',
            objective='l1',
            strategy='slp',
            shape=phi.shape[1:],
            threshold=THRESHOLD,
        ).fit(phi)
        V = six_tet_volumes_3d(result.corrected)
        assert float(V.min()) >= THRESHOLD - 1e-5

    def test_large_volume_routes_to_cluster_path(self):
        """Volumes above cluster_pixel_threshold dispatch to the 3D
        cluster path (checked via the solve info, no big solve needed —
        we shrink the threshold instead of growing the volume)."""
        from dvfopt import L1Objective, SimplexConstraint3D, SLPStrategy, Solver

        phi = _folded_volume()
        result = Solver(
            constraint=SimplexConstraint3D(shape=phi.shape[1:]),
            objective=L1Objective(),
            strategy=SLPStrategy(cluster_pixel_threshold=1),  # force cluster path
            threshold=THRESHOLD,
        ).fit(phi, record_history=True)
        V = six_tet_volumes_3d(result.corrected)
        assert float(V.min()) >= THRESHOLD - 1e-5


class TestResearchShims:
    def test_shims_reexport_promoted_impl(self):
        from dvfopt.core.slp import lp_direct_6tet as promoted
        from research.strict_feasibility_3d.algorithms import lp_direct_6tet as shim

        assert shim.slp_iter is promoted.slp_iter
        assert shim.lp_oneshot is promoted.lp_oneshot

    def test_gpu_shim(self):
        from dvfopt.core.slp import _gpu_untangle_3d as promoted
        from research.strict_feasibility_3d.algorithms import (
            _gpu_untangle_3d as shim,
        )

        assert shim.gpu_untangle_alm_3d is promoted.gpu_untangle_alm_3d


class TestGpuUntangle3D:
    """Torch-gated: the GPU untangler must clear (most) folds and be a
    no-op on already-feasible fields."""

    def setup_method(self):
        pytest.importorskip('torch')

    def test_noop_on_feasible_field(self):
        from dvfopt.core.slp._gpu_untangle_3d import gpu_untangle_alm_3d

        rng = np.random.default_rng(1)
        phi = rng.normal(0, 0.02, (3, 4, 5, 5))
        out = gpu_untangle_alm_3d(phi, threshold=THRESHOLD)
        np.testing.assert_array_equal(out, phi)

    def test_reduces_folds(self):
        from dvfopt.core.slp._gpu_untangle_3d import gpu_untangle_alm_3d

        phi = _folded_volume()
        n_before = int((six_tet_volumes_3d(phi) <= 0).sum())
        out = gpu_untangle_alm_3d(phi, threshold=THRESHOLD, n_outer=10, n_inner=100)
        n_after = int((six_tet_volumes_3d(out) <= 0).sum())
        assert n_after < n_before
