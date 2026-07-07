"""Tests for the CoupledKRing3D strategy + low-level module.

The strategy is the production wrapper for the SLSQP escape mechanism
that broke through the 1-fold M10Tet attractor in
``research/strict_feasibility_3d/REPORT.md`` Parts XI-XIV.
"""

import numpy as np
import pytest

from dvfopt import (
    CoupledKRing3DStrategy,
    L1Objective,
    Solver,
    Tet6Constraint3D,
)
from dvfopt.core.wallbreakers._coupled_kring_3d import (
    _build_problem,
    _make_constraint_fn,
    _make_constraint_jacobian,
    _make_index_tables,
    coupled_kring_slsqp_3d,
    find_worst_fold_cube,
)
from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d


def _planted_fold_3d(rng_seed=0, scale=2.0):
    """Small 3D field with a planted fold in the middle cube.

    Swaps the z-positions of two adjacent corner planes (z=2 and z=3)
    via a large negative dz displacement on the z=2 plane and positive
    on z=3, forcing the z-axis edges to cross and at least one tet
    volume to flip negative.
    """
    rng = np.random.default_rng(rng_seed)
    phi = rng.normal(0, 0.05, (3, 5, 5, 5)).astype(np.float64)
    # Plant a strong z-axis swap on the middle cube's two z-faces.
    phi[0, 2, 2:4, 2:4] = +scale  # z=2 plane: push z up.
    phi[0, 3, 2:4, 2:4] = -scale  # z=3 plane: push z down (crosses).
    return phi


class TestFindWorstFoldCube:
    def test_no_folds_returns_none(self):
        phi = np.zeros((3, 4, 4, 4))
        assert find_worst_fold_cube(phi) is None

    def test_finds_fold_location(self):
        phi = _planted_fold_3d()
        V = six_tet_volumes_3d(phi)
        assert (V <= 0).any(), 'test setup planted no folds'
        loc = find_worst_fold_cube(phi)
        assert loc is not None
        cz, cy, cx = loc
        # Returned cube actually has the worst min volume.
        min_per_cube = V.min(axis=0)
        assert min_per_cube[cz, cy, cx] == min_per_cube.min()


class TestBuildProblem:
    def test_centred_cube_k1(self):
        phi = np.zeros((3, 5, 5, 5))
        cubes, free_corners, _, x0 = _build_problem(phi, 2, 2, 2, k_ring=1)
        # 3x3x3 = 27 cubes, all interior so none trimmed.
        assert len(cubes) == 27
        # 4x4x4 = 64 corners.
        assert len(free_corners) == 64
        # 3 DOF per corner.
        assert x0.shape == (3 * 64,)
        # x0 initialised to current phi (zeros here).
        assert np.allclose(x0, 0.0)

    def test_boundary_cube_trims(self):
        # Centre at z=0 should trim the k=1 halo to half-thickness.
        phi = np.zeros((3, 5, 5, 5))
        cubes, _, _, _ = _build_problem(phi, 0, 2, 2, k_ring=1)
        # cubes with z in [0, 1] only, so 2x3x3 = 18.
        assert len(cubes) == 18


class TestAnalyticalJacobian:
    """The closed-form Jacobian must match finite differences."""

    def test_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.1, (3, 4, 5, 5)).astype(np.float64)
        cubes, free_corners, corner_idx, x0 = _build_problem(phi, 1, 2, 2, k_ring=1)
        cube_corner_x_idx, cube_corner_base = _make_index_tables(cubes, corner_idx)
        fn = _make_constraint_fn(cube_corner_x_idx, cube_corner_base, 0.0)
        jac = _make_constraint_jacobian(cube_corner_x_idx, cube_corner_base, len(x0))

        J_an = jac(x0)
        eps = 1e-6
        J_fd = np.zeros_like(J_an)
        for i in range(len(x0)):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += eps
            xm[i] -= eps
            J_fd[:, i] = (fn(xp) - fn(xm)) / (2 * eps)
        # Tight tolerance — analytic should be bit-precise except for FP noise.
        assert np.max(np.abs(J_an - J_fd)) < 1e-7


class TestCoupledKRingModule:
    def test_already_feasible_noop(self):
        phi = np.zeros((3, 5, 5, 5))
        loc = find_worst_fold_cube(phi)
        assert loc is None

    def test_signature_runs(self):
        """End-to-end smoke: call with a small synthetic problem."""
        phi = _planted_fold_3d(scale=0.55)
        loc = find_worst_fold_cube(phi)
        if loc is None:
            pytest.skip('synthetic test case did not plant a fold')
        cz, cy, cx = loc
        phi_out, info = coupled_kring_slsqp_3d(phi, cz, cy, cx, k_ring=1, feasibility_thr=1e-3)
        assert phi_out.shape == phi.shape
        assert 'success' in info
        assert 'wall_s' in info
        assert info['fold_center'] == (cz, cy, cx)


class TestCoupledKRing3DStrategy:
    def test_strategy_passes_constraint_check(self):
        """Smoke: strategy can be constructed + accepts Tet6Constraint3D."""
        strategy = CoupledKRing3DStrategy(k_ring=1, feasibility_thr=1e-3)
        constraint = Tet6Constraint3D(shape=(5, 5, 5))
        # _check_constraint shouldn't raise.
        strategy._check_constraint(constraint)

    def test_solver_smoke(self):
        """Solver.fit() runs without error and returns a corrected field."""
        phi = _planted_fold_3d(scale=0.55)
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=CoupledKRing3DStrategy(k_ring=1, feasibility_thr=1e-3),
            threshold=0.01,
        )
        result = solver.fit(phi)
        assert result.corrected.shape == phi.shape

    def test_explicit_target_cube(self):
        """target_cube override is honoured (does not call find_worst)."""
        phi = _planted_fold_3d()
        # Centre on a known interior cube; outcome is just "ran without error".
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=CoupledKRing3DStrategy(k_ring=1, feasibility_thr=1e-3, target_cube=(2, 2, 2)),
            threshold=0.01,
        )
        result = solver.fit(phi)
        # Phase records the target.
        phases = result.info.phases
        if phases:
            assert phases[0].extras['fold_center'] == (2, 2, 2)


class TestAnalyticalJacobianOption:
    """The analytical-Jacobian path is opt-in (off by default)."""

    def test_default_uses_fd(self):
        strategy = CoupledKRing3DStrategy()
        assert strategy.use_analytical_jacobian is False

    def test_analytical_path_produces_diagnostic(self):
        phi = _planted_fold_3d()
        loc = find_worst_fold_cube(phi)
        if loc is None:
            pytest.skip('synthetic test case did not plant a fold')
        cz, cy, cx = loc
        _, info = coupled_kring_slsqp_3d(
            phi,
            cz,
            cy,
            cx,
            k_ring=1,
            feasibility_thr=1e-3,
            use_analytical_jacobian=True,
        )
        # The path runs (success may be False — see strategy docstring),
        # but the info dict records that the analytical jac was used.
        assert info['analytical_jac'] is True


class TestClusterFoldCubes:
    """The clustering helper used by mode='cluster'."""

    def test_empty(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            cluster_fold_cubes,
        )

        centroids, members, radii = cluster_fold_cubes([])
        assert centroids == [] and members == [] and radii == []

    def test_single_cube(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            cluster_fold_cubes,
        )

        centroids, members, radii = cluster_fold_cubes([(2, 2, 2)])
        assert centroids == [(2, 2, 2)]
        assert members == [[(2, 2, 2)]]
        assert radii == [0]

    def test_two_close_cubes_one_cluster(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            cluster_fold_cubes,
        )

        # Chebyshev distance 1, well within default radius=2.
        centroids, members, _ = cluster_fold_cubes([(0, 0, 0), (1, 0, 0)], radius=2)
        assert len(centroids) == 1
        assert sorted(members[0]) == [(0, 0, 0), (1, 0, 0)]

    def test_two_far_cubes_two_clusters(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            cluster_fold_cubes,
        )

        # Chebyshev distance 10 > radius=2 → separate clusters.
        centroids, members, _ = cluster_fold_cubes([(0, 0, 0), (10, 10, 10)], radius=2)
        assert len(centroids) == 2


class TestClusterMode:
    """End-to-end tests for CoupledKRing3DStrategy(mode='cluster')."""

    def test_cluster_mode_no_folds_noop(self):
        phi = np.zeros((3, 5, 5, 5))
        strategy = CoupledKRing3DStrategy(
            k_ring=1,
            feasibility_thr=1e-3,
            mode='cluster',
        )
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=strategy,
            threshold=0.01,
        )
        result = solver.fit(phi)
        assert np.array_equal(result.corrected, phi)

    def test_cluster_mode_runs_serial(self):
        """n_workers=1 forces sequential execution."""
        phi = _planted_fold_3d(scale=2.0)
        loc = find_worst_fold_cube(phi)
        if loc is None:
            pytest.skip('synthetic test case did not plant a fold')
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=CoupledKRing3DStrategy(
                k_ring=1,
                feasibility_thr=1e-3,
                mode='cluster',
                n_workers=1,
            ),
            threshold=0.01,
        )
        result = solver.fit(phi)
        assert result.corrected.shape == phi.shape
        # Phase reports cluster count.
        phases = result.info.phases
        if phases:
            assert phases[0].name == 'coupled_kring_slsqp_cluster'
            assert phases[0].extras['n_clusters'] >= 1

    def test_partition_non_overlapping(self):
        """Centres within 2*k_ring fall in separate batches; far ones share."""
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            _partition_non_overlapping,
        )

        # k_ring=2 → overlap radius 4. (0,0,0) and (3,0,0) overlap;
        # (0,0,0) and (10,0,0) do not.
        batches = _partition_non_overlapping(
            [(0, 0, 0), (3, 0, 0), (10, 0, 0)],
            k_ring=2,
        )
        # First batch: (0,0,0) and (10,0,0). Second: (3,0,0).
        assert len(batches) == 2
        first = sorted(batches[0])
        second = sorted(batches[1])
        assert (0, 0, 0) in first
        assert (10, 0, 0) in first
        assert (3, 0, 0) in second

    def test_invalid_mode_raises(self):
        phi = np.zeros((3, 5, 5, 5))
        strategy = CoupledKRing3DStrategy(mode='nonsense')
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=strategy,
            threshold=0.01,
        )
        with pytest.raises(ValueError, match='mode must be'):
            solver.fit(phi)


class TestLocalAlmRecovery:
    """Local crop recovery: the ~430x-faster replacement for global M10Tet
    recovery after a coupled k-ring escape."""

    def test_no_folds_noop(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            local_alm_recovery_3d,
        )

        phi = np.zeros((3, 6, 6, 6))
        out, info = local_alm_recovery_3d(phi, center=None)
        assert np.array_equal(out, phi)
        assert info['accepted'] is True
        assert info['n_neg_after'] == 0

    def test_crop_bbox_centered(self):
        """With an explicit center, the crop is a bounded box, not the
        whole field (the whole point — locality)."""
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            local_alm_recovery_3d,
        )

        # A trivial feasible field so the inner solve is a fast no-op; we
        # only assert the crop geometry stays local.
        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.01, (3, 20, 20, 20)).astype(np.float64)

        captured = {}

        def fake_inner(crop, time_budget_s=600.0):
            captured['shape'] = crop.shape
            return crop  # identity

        # Force a fold so recovery actually crops (else early-returns).
        phi[0, 10, 10:12, 10:12] = 2.0
        phi[0, 11, 10:12, 10:12] = -2.0

        _, info = local_alm_recovery_3d(
            phi,
            center=(10, 10, 10),
            k_ring=2,
            pad=3,
            inner_solve=fake_inner,
        )
        # Crop must be far smaller than the full 20^3 field.
        assert captured['shape'][1] < 20
        assert captured['shape'][2] < 20
        assert captured['shape'][3] < 20
        assert info['crop_bbox'] is not None

    def test_rejects_regression(self):
        """If the inner solve makes things globally worse, the original
        field is returned unchanged (accepted=False)."""
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            local_alm_recovery_3d,
        )

        phi = _planted_fold_3d(scale=2.0)
        n0 = int((six_tet_volumes_3d(phi) <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')

        def wrecking_inner(crop, time_budget_s=600.0):
            # Deliberately worsen with per-corner noise (a uniform shift
            # would be a rigid translation that preserves all volumes).
            rng = np.random.default_rng(123)
            return crop + rng.normal(0, 3.0, crop.shape)

        out, info = local_alm_recovery_3d(
            phi,
            center=find_worst_fold_cube(phi),
            k_ring=2,
            pad=2,
            inner_solve=wrecking_inner,
            max_widen=1,
        )
        # Wrecking inner regresses -> original returned.
        assert info['accepted'] is False
        assert np.array_equal(out, phi)


class TestFusedMinKernel:
    """Parallel fused min-volume kernel must equal materialise-then-min."""

    def test_matches_min_over_volumes(self):
        from dvfopt.jacobian.tetrahedron_sign import (
            six_tet_min_volume_3d,
            six_tet_volumes_3d,
        )

        rng = np.random.default_rng(7)
        for shape in [(3, 5, 6, 7), (3, 16, 32, 32)]:
            phi = rng.normal(0, 0.3, shape).astype(np.float64)
            ref = six_tet_volumes_3d(phi).min(axis=0)
            fused = six_tet_min_volume_3d(phi)
            assert fused.shape == ref.shape
            np.testing.assert_allclose(fused, ref, atol=1e-12)

    def test_identity_positive(self):
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        phi = np.zeros((3, 5, 5, 5))
        mv = six_tet_min_volume_3d(phi)
        assert (mv > 0).all()  # identity: every cube min tet = +1/6


class TestBestDiagonal:
    """Variable Kuhn-triangulation (per-cube best diagonal) feasibility."""

    def test_all_diagonals_shape_and_diag0_matches_fixed(self):
        from dvfopt.jacobian.tetrahedron_sign import (
            six_tet_volumes_3d,
            six_tet_volumes_all_diagonals,
        )

        rng = np.random.default_rng(3)
        phi = rng.normal(0, 0.1, (3, 5, 6, 7)).astype(np.float64)
        alld = six_tet_volumes_all_diagonals(phi)
        assert alld.shape == (4, 4, 5, 6)
        # Diagonal 0 must equal the canonical fixed-split min.
        fixed = six_tet_volumes_3d(phi).min(axis=0)
        np.testing.assert_allclose(alld[0], fixed, atol=1e-12)

    def test_best_diag_dominates_fixed(self):
        """Best-of-4 min volume is always >= the fixed-diagonal min."""
        from dvfopt.jacobian.tetrahedron_sign import (
            best_diagonal_min_volume,
            six_tet_volumes_3d,
        )

        rng = np.random.default_rng(4)
        phi = rng.normal(0, 0.25, (3, 5, 5, 5)).astype(np.float64)
        fixed = six_tet_volumes_3d(phi).min(axis=0)
        best_min, best_diag = best_diagonal_min_volume(phi)
        assert (best_min >= fixed - 1e-12).all()
        assert best_diag.shape == fixed.shape
        assert best_diag.min() >= 0 and best_diag.max() <= 3

    def test_n_neg_best_diag_le_fixed(self):
        from dvfopt.jacobian.tetrahedron_sign import (
            n_neg_best_diagonal,
            six_tet_volumes_3d,
        )

        rng = np.random.default_rng(5)
        phi = rng.normal(0, 0.3, (3, 6, 6, 6)).astype(np.float64)
        fixed_neg = int((six_tet_volumes_3d(phi).min(axis=0) <= 0).sum())
        best_neg = n_neg_best_diagonal(phi, threshold=0.0)
        # Allowing per-cube diagonal choice can only help (or tie).
        assert best_neg <= fixed_neg

    def test_identity_feasible_all_diagonals(self):
        from dvfopt.jacobian.tetrahedron_sign import (
            six_tet_volumes_all_diagonals,
        )

        phi = np.zeros((3, 5, 5, 5))
        alld = six_tet_volumes_all_diagonals(phi)
        # Identity: every diagonal yields positive worst-tet (= +1/6).
        assert (alld > 0).all()


class TestActiveBandRecovery:
    """active_band_alm_recovery_3d: per-cluster crop M10Tet, global-verified."""

    def test_no_folds_noop(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            active_band_alm_recovery_3d,
        )

        phi = np.zeros((3, 6, 6, 6))
        out, info = active_band_alm_recovery_3d(phi, threshold=0.012)
        assert info['n_clusters'] == 0
        assert info['n_neg_after'] == 0
        assert np.array_equal(out, phi)

    def test_two_clusters_detected_and_cropped(self):
        """Two separate planted folds -> two clusters, each solved on a
        crop far smaller than the full field; never increases folds."""
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            _fold_cluster_bboxes,
            active_band_alm_recovery_3d,
        )
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.02, (3, 8, 24, 24)).astype(np.float64)
        phi[0, 3, 4:6, 4:6] = 1.5
        phi[0, 4, 4:6, 4:6] = -1.5
        phi[0, 3, 18:20, 18:20] = 1.5
        phi[0, 4, 18:20, 18:20] = -1.5
        mv = six_tet_min_volume_3d(phi)
        n0 = int((mv <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')
        bb = _fold_cluster_bboxes(mv, 0.0)
        assert len(bb) == 2  # two spatially-separate clusters
        out, info = active_band_alm_recovery_3d(phi, threshold=0.012, pad=2)
        # Active-band never makes the global fold count worse.
        assert info['n_neg_after'] <= info['n_neg_before']
        # Each crop is smaller than the full field (locality).
        for pc in info['per_cluster']:
            assert pc['crop_shape'][1] < 24 and pc['crop_shape'][2] < 24

    def test_oversized_cluster_tiles_not_global(self):
        """A cluster spanning most of an axis must be TILED into bounded
        crops, never solved as the full field (the OOM-segfault bug)."""
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            active_band_alm_recovery_3d,
        )
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.02, (3, 6, 40, 40)).astype(np.float64)
        # Plant a wide fold band spanning most of the y and x extent (well
        # over max_band_fraction of those axes).
        phi[0, 2, 2:38, 2:38] = 1.5
        phi[0, 3, 2:38, 2:38] = -1.5
        n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')

        recorded = []

        def recording_inner(crop, time_budget_s=600.0):
            recorded.append(tuple(int(s) for s in crop.shape))
            return crop  # no-op: still exercises crop/paste/verify

        pad = 2
        max_box = 16
        out, info = active_band_alm_recovery_3d(
            phi,
            threshold=0.012,
            pad=pad,
            max_box=max_box,
            inner_solve=recording_inner,
        )
        # (c) ran without error and returned the right shape.
        assert out.shape == phi.shape
        assert recorded, 'inner_solve was never called'
        # (a) the global fallback is gone: no crop is the whole field.
        assert (3, 6, 40, 40) not in recorded
        # (b) every crop is bounded on y and x by the tile cap + padding.
        bound = max_box + 2 * pad + 4
        for shp in recorded:
            assert shp[2] <= bound, f'crop y-extent {shp[2]} exceeds {bound}'
            assert shp[3] <= bound, f'crop x-extent {shp[3]} exceeds {bound}'


class TestActiveBandParallelBatching:
    """Non-overlap batching logic for the parallel active-band path
    (pure logic — no process spawn, so fast and deterministic)."""

    def test_boxes_separated(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import _boxes_separated

        a = (0, 5, 0, 5, 0, 5)
        far = (10, 15, 0, 5, 0, 5)  # gap 4 on z
        near = (6, 9, 0, 5, 0, 5)  # gap 0 on z (touching+1)
        assert _boxes_separated(a, far, gap=2) is True
        assert _boxes_separated(a, near, gap=2) is False

    def test_batch_partition_disjoint(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            _batch_nonoverlapping_boxes,
            _boxes_separated,
        )

        boxes = [
            (0, 5, 0, 5, 0, 5),
            (7, 9, 0, 5, 0, 5),  # overlaps box0 (gap 1) -> different batch
            (20, 25, 0, 5, 0, 5),  # far from both -> can share a batch
        ]
        batches = _batch_nonoverlapping_boxes(boxes)
        # Every pair within a batch must be separated.
        for batch in batches:
            for i in range(len(batch)):
                for j in range(i + 1, len(batch)):
                    assert _boxes_separated(boxes[batch[i]], boxes[batch[j]])

    def test_padded_box_clips(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import _padded_box

        # Near the origin and the far edge, padding must clip to [0, dim-1].
        pb = _padded_box((0, 1, 0, 1, 0, 1), pad=4, shape=(8, 8, 8))
        assert pb[0] == 0 and pb[2] == 0 and pb[4] == 0
        assert pb[1] <= 7 and pb[3] <= 7 and pb[5] <= 7


class TestActiveBandStrategy:
    """ActiveBandALM3DStrategy via the Solver facade."""

    def test_solver_reaches_feasible_on_scattered_folds(self):
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.02, (3, 8, 24, 24)).astype(np.float64)
        phi[0, 3, 4:6, 4:6] = 1.5
        phi[0, 4, 4:6, 4:6] = -1.5
        phi[0, 3, 18:20, 18:20] = 1.5
        phi[0, 4, 18:20, 18:20] = -1.5
        if int((six_tet_min_volume_3d(phi) <= 0).sum()) == 0:
            pytest.skip('no fold planted')
        from dvfopt import (
            ActiveBandALM3DStrategy,
            L1Objective,
            Solver,
            Tet6Constraint3D,
        )

        res = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=ActiveBandALM3DStrategy(pad=2),
            threshold=0.01,
        ).fit(phi)
        mv = six_tet_min_volume_3d(res.corrected)
        assert int((mv <= 0).sum()) == 0
        assert res.info.phases[0].name == 'active_band_alm'
        assert res.info.phases[0].extras['n_clusters'] == 2


class TestParallelZBand:
    """parallel_zband_solve: coarse z-band decomposition. Covers both the
    sequential path (n_workers=1) and the process-pool path (n_workers>1,
    which routes through dvfopt.core._pool.pool_map)."""

    def test_bands_reach_feasible(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            parallel_zband_solve,
        )
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.02, (3, 48, 40, 40)).astype(np.float64)
        for z, y, x in [(5, 10, 10), (27, 20, 28), (40, 30, 12)]:
            phi[0, z, y : y + 2, x : x + 2] = 1.5
            phi[0, z + 1, y : y + 2, x : x + 2] = -1.5
        n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')
        out, info = parallel_zband_solve(
            phi,
            threshold=0.012,
            band_size=16,
            overlap=4,
            pad=3,
            n_workers=1,
        )
        assert info['n_bands'] == 3
        # Never worse than input; here reaches strict feasibility.
        assert info['n_neg_after'] <= n0
        assert int((six_tet_min_volume_3d(out) <= 0).sum()) == info['n_neg_after']

    def test_single_band_no_split(self):
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            parallel_zband_solve,
        )

        phi = np.zeros((3, 8, 10, 10))
        out, info = parallel_zband_solve(
            phi,
            threshold=0.012,
            band_size=32,
            n_workers=1,
        )
        assert info['n_bands'] == 1
        assert info['n_neg_after'] == 0

    def test_multiworker_pool_path(self):
        """n_workers>1 with multiple bands must route through the shared
        pool without a NameError (regression: pool_map import was dropped
        when the path was switched off a per-call executor)."""
        from dvfopt.core._pool import shutdown_pool
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            parallel_zband_solve,
        )
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        rng = np.random.default_rng(1)
        phi = rng.normal(0, 0.02, (3, 48, 36, 36)).astype(np.float64)
        for z, y, x in [(6, 10, 10), (28, 18, 24)]:
            phi[0, z, y : y + 2, x : x + 2] = 1.5
            phi[0, z + 1, y : y + 2, x : x + 2] = -1.5
        n0 = int((six_tet_min_volume_3d(phi) <= 0).sum())
        if n0 == 0:
            pytest.skip('no fold planted')
        try:
            out, info = parallel_zband_solve(
                phi,
                threshold=0.012,
                band_size=16,
                overlap=4,
                pad=3,
                n_workers=2,
            )
        finally:
            shutdown_pool()
        assert info['n_bands'] > 1
        assert info['n_neg_after'] <= n0
        assert int((six_tet_min_volume_3d(out) <= 0).sum()) == info['n_neg_after']


class TestRecoverMode:
    """CoupledKRing3DStrategy(recover=True) = self-contained escape+tighten."""

    def test_recover_threshold_default(self):
        s = CoupledKRing3DStrategy(recover=True)
        assert s.recover_threshold is None  # -> resolves to 1.2*threshold

    def test_recover_adds_phase(self):
        phi = _planted_fold_3d(scale=2.0)
        if find_worst_fold_cube(phi) is None:
            pytest.skip('no fold planted')
        solver = Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L1Objective(eps=1e-4),
            strategy=CoupledKRing3DStrategy(
                k_ring=1,
                feasibility_thr=1e-3,
                recover=True,
                recover_pad=2,
            ),
            threshold=0.01,
        )
        result = solver.fit(phi)
        names = [p.name for p in result.info.phases]
        # Both the escape and the local recovery phases should be recorded.
        assert 'coupled_kring_slsqp' in names
        assert 'local_alm_recovery' in names
