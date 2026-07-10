"""Tests for the 6-tet per-voxel sign helper.

The winding signs in ``_TET_SIGN`` are easy to get wrong (six tets, two
plausible windings each, plus a global ±1 from the +y-down image
convention). These tests pin the convention to:

1. **Identity field** → every tet has signed volume exactly ``+1/6``.
2. **Volume sum** → for any field, the 6 tet volumes per cell sum to
   the cell's signed volume under the chosen decomposition. On the
   identity field this sum is exactly ``+1`` per cell.
3. **Single-voxel fold** → punching a large negative displacement at
   one voxel flips at least one tet in the surrounding cells.
4. **Small random fields stay feasible.**
"""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt.jacobian.tetrahedron_sign import (
    six_tet_fold_classification,
    six_tet_volumes_3d,
    tet_grad_T_v,
    tet_volumes_flat,
)


class TestIdentity:
    def test_all_tets_positive_one_sixth(self):
        phi = np.zeros((3, 4, 5, 6))
        V = six_tet_volumes_3d(phi)
        assert V.shape == (6, 3, 4, 5)
        assert np.allclose(V, 1 / 6)

    def test_volume_sum_per_cell_is_unity(self):
        """6 tets covering a unit cube → total volume 1."""
        phi = np.zeros((3, 4, 5, 6))
        V = six_tet_volumes_3d(phi)
        total = V.sum(axis=0)
        assert np.allclose(total, 1.0)

    def test_classification_returns_zero(self):
        phi = np.zeros((3, 4, 5, 6))
        n = six_tet_fold_classification(phi)
        assert n.shape == (3, 4, 5)
        assert (n == 0).all()


class TestSmallRandomFeasible:
    def test_no_flips_on_small_jitter(self):
        rng = np.random.default_rng(0)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            phi = rng.normal(0, 0.05, (3, 4, 5, 5))
            V = six_tet_volumes_3d(phi)
            assert (V > 0).all(), f'seed={seed}: unexpected flip on tiny jitter'


class TestFoldedField:
    def test_punched_fold_produces_flips(self):
        phi = np.zeros((3, 5, 6, 6))
        # Deep fold at the center: collapse via large negative dy/dx
        phi[1, 1:4, 2:4, 2:4] -= 1.8
        phi[2, 1:4, 2:4, 2:4] -= 1.8
        V = six_tet_volumes_3d(phi)
        assert (V <= 0).any()

    def test_classification_max_is_six(self):
        """A whole-cell collapse (all 8 corners pulled inward) should
        give that cell all 6 tets flipped.

        A single-corner displacement only affects tets that touch that
        corner — max 4 of the 6 — so this test deliberately collapses
        every corner of the central cell.
        """
        phi = np.zeros((3, 5, 6, 6))
        # Collapse the cell at (z=2, y=3, x=3) by pulling its 8 corners
        # heavily toward each other.
        for oz in (0, 1):
            for oy in (0, 1):
                for ox in (0, 1):
                    sgn_y = 1 if oy == 0 else -1
                    sgn_x = 1 if ox == 0 else -1
                    phi[1, 2 + oz, 3 + oy, 3 + ox] = sgn_y * -3.0
                    phi[2, 2 + oz, 3 + oy, 3 + ox] = sgn_x * -3.0
        n = six_tet_fold_classification(phi)
        assert n.max() == 6, f'max-tet-flips per cell was {n.max()}, expected 6'

    def test_isolated_perturbation_partial_flip(self):
        """A small single-corner perturbation should flip only a few
        tets (not all 6 of any one cell)."""
        phi = np.zeros((3, 4, 5, 5))
        phi[1, 1, 1, 1] = -0.3  # tiny single-vertex tweak
        n = six_tet_fold_classification(phi)
        # The neighboring cells should NOT have any tet flipped (jitter
        # too small to break feasibility).
        assert (n == 0).all() or n.max() < 6


class TestFusedMinVolumeEquivalence:
    """The wallbreaker hot paths (_harmonic_3d, _schwarz_common,
    _refine_repair_3d) replaced ``six_tet_volumes_3d(phi).min(axis=0)``
    with the fused :func:`six_tet_min_volume_3d` kernel. Pin the exact
    equivalence those swaps rely on."""

    def test_fused_matches_materialised_min(self):
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        rng = np.random.default_rng(42)
        phi = rng.normal(0, 0.4, (3, 6, 7, 8))
        ref = six_tet_volumes_3d(phi).min(axis=0)
        fused = six_tet_min_volume_3d(phi)
        assert fused.shape == ref.shape
        np.testing.assert_allclose(fused, ref, rtol=0, atol=1e-12)


class TestTorchBackend:
    """Verify the torch forward matches the numpy forward and that
    autograd through it agrees with the analytical adjoint.

    Skipped if torch isn't installed (torch is in the ``benchmarks``
    extra, not the core deps)."""

    def setup_method(self):
        pytest.importorskip('torch')

    def test_identity_matches_numpy(self):
        import torch

        from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch

        phi_pt = torch.zeros((3, 4, 5, 5), dtype=torch.float64)
        V = six_tet_volumes_3d_torch(phi_pt).numpy()
        assert np.allclose(V, 1 / 6)

    def test_random_parity_with_numpy(self):
        import torch

        from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch

        rng = np.random.default_rng(0)
        phi_np = rng.normal(0, 0.05, (3, 4, 5, 5))
        phi_pt = torch.from_numpy(phi_np.copy()).double()
        V_np = six_tet_volumes_3d(phi_np)
        V_pt = six_tet_volumes_3d_torch(phi_pt).numpy()
        assert float(np.abs(V_np - V_pt).max()) < 1e-13

    def test_autograd_matches_analytical_adjoint(self):
        """Autograd through the torch forward should match the closed-form
        analytical adjoint from the numpy path."""
        import torch

        from dvfopt.jacobian.tetrahedron_sign_torch import six_tet_volumes_3d_torch

        rng = np.random.default_rng(0)
        D, H, W = 3, 4, 5
        phi_np = rng.normal(0, 0.05, (3, D, H, W))
        v = rng.normal(size=(6 * (D - 1) * (H - 1) * (W - 1)))

        phi_pt = torch.from_numpy(phi_np.copy()).double().requires_grad_(True)
        V_pt = six_tet_volumes_3d_torch(phi_pt)
        v_pt = torch.from_numpy(v.reshape(V_pt.shape).copy())
        autograd_grad = torch.autograd.grad(V_pt, phi_pt, grad_outputs=v_pt)[0].numpy()
        # autograd returns (3, D, H, W) in [dz, dy, dx]; convert to flat [dx, dy, dz]
        autograd_flat = np.concatenate(
            [autograd_grad[2].ravel(), autograd_grad[1].ravel(), autograd_grad[0].ravel()]
        )
        phi_flat = np.concatenate([phi_np[2].ravel(), phi_np[1].ravel(), phi_np[0].ravel()])
        ana = tet_grad_T_v(phi_flat, D, H, W, v)
        assert float(np.abs(autograd_flat - ana).max()) < 1e-10


class TestShape:
    @pytest.mark.parametrize('shape', [(3, 3, 3), (3, 4, 5), (5, 4, 4)])
    def test_output_shape(self, shape):
        D, H, W = shape
        phi = np.zeros((3, D, H, W))
        V = six_tet_volumes_3d(phi)
        assert V.shape == (6, D - 1, H - 1, W - 1)
        n = six_tet_fold_classification(phi)
        assert n.shape == (D - 1, H - 1, W - 1)


# ---------------------------------------------------------------------------
# Flat-pack primitives (the constraint-system entry points)
# ---------------------------------------------------------------------------


class TestFlatPack:
    """Verify the flat-pack ``[dx, dy, dz]`` versions agree with the
    array versions, and that the analytical adjoint matches FD."""

    def test_tet_volumes_flat_matches_six_tet_volumes_3d(self):
        rng = np.random.default_rng(0)
        D, H, W = 3, 4, 5
        phi = rng.normal(0, 0.05, (3, D, H, W))  # (3, D, H, W) [dz, dy, dx]
        # Flat-pack is [dx, dy, dz].
        phi_flat = np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])
        V_flat = tet_volumes_flat(phi_flat, D, H, W)
        V_array = six_tet_volumes_3d(phi)
        np.testing.assert_allclose(V_flat, V_array.ravel())

    def test_tet_volumes_flat_identity(self):
        D, H, W = 3, 4, 5
        phi_flat = np.zeros(3 * D * H * W)
        V = tet_volumes_flat(phi_flat, D, H, W)
        assert V.shape == (6 * (D - 1) * (H - 1) * (W - 1),)
        assert np.allclose(V, 1 / 6)

    @pytest.mark.parametrize('seed', [0, 1, 2, 7])
    def test_adjoint_matches_finite_differences(self, seed):
        """Gradcheck: analytical J^T @ v vs central-difference J^T @ v."""
        rng = np.random.default_rng(seed)
        D, H, W = 3, 4, 5
        phi_flat = rng.normal(0, 0.05, 3 * D * H * W)
        n_v = phi_flat.size
        n_c = 6 * (D - 1) * (H - 1) * (W - 1)
        v = rng.normal(size=n_c)

        ana = tet_grad_T_v(phi_flat, D, H, W, v)

        eps = 1e-6
        num = np.zeros(n_v)
        for i in range(n_v):
            p = phi_flat.copy()
            p[i] += eps
            m = phi_flat.copy()
            m[i] -= eps
            num[i] = float(
                np.dot((tet_volumes_flat(p, D, H, W) - tet_volumes_flat(m, D, H, W)) / (2 * eps), v)
            )

        err = float(np.abs(ana - num).max())
        assert err < 1e-6, f'seed={seed}: gradcheck err={err:.2e}'


# ---------------------------------------------------------------------------
# End-to-end via Tet6Constraint3D
# ---------------------------------------------------------------------------


class TestTet6SparseJacobian:
    """The sparse forward Jacobian (used by SLSQP) should match the
    analytical adjoint (``J.T @ v``) and the finite-difference dense
    Jacobian. This is the analogue of
    ``test_tri_constraint_2d_sparse_jacobian_matches_dense``."""

    def test_shape_and_nnz(self):
        from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac

        D, H, W = 3, 4, 5
        J = build_tet_sparse_jac(D, H, W)(np.zeros(3 * D * H * W))
        n_constraints = 6 * (D - 1) * (H - 1) * (W - 1)
        n_variables = 3 * D * H * W
        assert J.shape == (n_constraints, n_variables)
        assert J.nnz == 72 * (D - 1) * (H - 1) * (W - 1)

    def test_transpose_matches_analytical_adjoint(self):
        from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac

        rng = np.random.default_rng(0)
        D, H, W = 3, 4, 5
        phi = rng.normal(0, 0.05, 3 * D * H * W)
        v = rng.normal(size=6 * (D - 1) * (H - 1) * (W - 1))
        J = build_tet_sparse_jac(D, H, W)(phi)
        sparse_adjoint = J.T @ v
        ana = tet_grad_T_v(phi, D, H, W, v)
        assert float(np.abs(sparse_adjoint - ana).max()) < 1e-14

    def test_matches_finite_difference_dense(self):
        from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac

        rng = np.random.default_rng(0)
        D, H, W = 3, 4, 4
        phi = rng.normal(0, 0.05, 3 * D * H * W)
        J = build_tet_sparse_jac(D, H, W)(phi).toarray()

        eps = 1e-6
        n_vars = 3 * D * H * W
        n_constr = 6 * (D - 1) * (H - 1) * (W - 1)
        J_fd = np.zeros((n_constr, n_vars))
        for i in range(n_vars):
            p = phi.copy()
            p[i] += eps
            m = phi.copy()
            m[i] -= eps
            J_fd[:, i] = (tet_volumes_flat(p, D, H, W) - tet_volumes_flat(m, D, H, W)) / (2 * eps)
        assert float(np.abs(J - J_fd).max()) < 1e-6

    def test_constraint_jacobian_method(self):
        """``Tet6Constraint3D.jacobian()`` should return the same thing
        as the underlying ``build_tet_sparse_jac``."""
        from dvfopt import Tet6Constraint3D

        rng = np.random.default_rng(0)
        c = Tet6Constraint3D(shape=(3, 4, 5))
        phi = rng.normal(0, 0.05, c.n_variables)
        J = c.jacobian(phi)
        assert J.shape == (c.n_constraints, c.n_variables)
        # Round-trip via the analytical adjoint.
        v = rng.normal(size=c.n_constraints)
        assert float(np.abs(J.T @ v - c.adjoint(phi, v)).max()) < 1e-14


class TestALM3DStrategy:
    """PHR augmented Lagrangian for 6-tet — standalone Phase C wallbreaker."""

    @staticmethod
    def _phi_planted_fold_3d():
        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        return phi

    def test_reaches_feasibility_at_threshold(self):
        from dvfopt import ALM3DStrategy, L2Objective, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=ALM3DStrategy(),
        )
        result = solver.fit(self._phi_planted_fold_3d())
        V = six_tet_volumes_3d(result.corrected)
        assert (V <= 0).sum() == 0
        assert V.min() >= 0.01 - 1e-5

    def test_l2_better_than_harmonic_alone(self):
        """ALM should produce a smaller L2 than the harmonic-alone seed
        on this single-corner case — the harmonic patch is L2 = sqrt(4.5)
        ≈ 2.12 (smoothest), ALM tightens to ~0.59."""
        from dvfopt import ALM3DStrategy, L2Objective, Solver, Tet6Constraint3D

        phi = self._phi_planted_fold_3d()
        result = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=ALM3DStrategy(),
        ).fit(phi)
        l2 = float(np.linalg.norm(result.corrected - phi))
        assert l2 < 1.0, f'expected ALM L2 < 1.0; got {l2}'

    def test_registry(self):
        from dvfopt import correct_dvf

        result = correct_dvf(
            self._phi_planted_fold_3d(),
            constraint='6tet',
            objective='l2',
            strategy='alm_3d',
        )
        assert result.feasible
        assert result.info.strategy_name == 'ALM3DStrategy'

    def test_phase_n_neg_populated(self):
        """PR #16 review (g)+(h): per-phase ``n_neg`` must flow from
        ALM's per-outer log entries into :class:`PhaseInfo`, and
        :attr:`SolveInfo.feasible_after_phase` must mark the round in
        which the iterate first reached feasibility. Before this fix,
        ALM history entries had no ``n_neg`` key, so all PhaseInfo
        entries had ``n_neg=-1`` and ``feasible_after_phase=-1``."""
        from dvfopt import ALM3DStrategy, L2Objective, Solver, Tet6Constraint3D

        result = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=ALM3DStrategy(),
        ).fit(self._phi_planted_fold_3d(), record_history=True)

        assert result.feasible
        assert result.info.phases, 'ALM3DStrategy should record per-outer phases'
        # Every recorded phase must have a real n_neg (not the -1 sentinel).
        for ph in result.info.phases:
            assert ph.n_neg >= 0, (
                f'phase {ph.name!r} has n_neg={ph.n_neg} (PhaseInfo default); '
                f'ALM log did not include n_neg'
            )
        # The final phase should have n_neg == 0 (we reached feasibility).
        assert result.info.phases[-1].n_neg == 0
        # And feasible_after_phase should be set to the first feasible round.
        assert result.info.feasible_after_phase >= 0


class TestM10TetStrategy:
    """Full m10-3D pipeline (harmonic → ALM → barrier polish).

    Verifies the chained pipeline: harmonic seed brings the field to
    feasibility, ALM pulls it back toward the input via the anchor
    (passing phi_anchor explicitly so the anchor reference is the
    ORIGINAL input, not the harmonic seed), and optional barrier
    polish tightens further.
    """

    @staticmethod
    def _phi_planted_fold_3d():
        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        return phi

    def test_polish_off_emits_two_phases(self):
        """With polish=False, the SolveInfo history has exactly two
        phases (harmonic + alm); ALM's anchor pulls toward the
        original input, not the harmonic seed."""
        from dvfopt import L2Objective, M10TetStrategy, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=M10TetStrategy(polish=False),
        )
        result = solver.fit(self._phi_planted_fold_3d(), record_history=True)
        phase_names = [p.name for p in result.info.phases]
        assert phase_names == ['harmonic', 'alm'], (
            f'polish=False should produce [harmonic, alm]; got {phase_names}'
        )
        # ALM should pull L2 well below the harmonic-alone result (~2.12).
        l2 = float(np.linalg.norm(result.corrected - self._phi_planted_fold_3d()))
        assert l2 < 1.5, f'M10 (harmonic+ALM) L2 should be < 1.5; got {l2}'

    def test_polish_on_emits_pipeline(self):
        from dvfopt import L2Objective, M10TetStrategy, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=M10TetStrategy(polish=True),
        )
        result = solver.fit(self._phi_planted_fold_3d(), record_history=True)
        phase_names = [p.name for p in result.info.phases]
        assert phase_names[:2] == ['harmonic', 'alm'], (
            f'expected harmonic + alm first; got {phase_names}'
        )
        assert any(n.startswith('polish_') for n in phase_names), (
            f'expected at least one polish_* phase; got {phase_names}'
        )

    def test_registry(self):
        from dvfopt import correct_dvf

        result = correct_dvf(
            self._phi_planted_fold_3d(),
            constraint='6tet',
            objective='l2',
            strategy='m10_3d',
        )
        assert result.feasible
        assert result.info.strategy_name == 'HarmonicALMBarrier3DStrategy'


class TestM14Schwarz3DStrategy:
    """Cluster-localized m14-3D (Phase E)."""

    def test_two_clusters_separately_processed(self):
        """Two well-separated planted folds should be detected as two
        clusters and processed independently (no fallback)."""
        from dvfopt import L2Objective, M14Schwarz3DStrategy, Solver, Tet6Constraint3D

        phi = np.zeros((3, 10, 10, 10))
        phi[1, 2, 2, 2] = 1.5
        phi[2, 2, 2, 2] = 1.5
        phi[1, 7, 7, 7] = 1.5
        phi[2, 7, 7, 7] = 1.5

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(10, 10, 10)),
            objective=L2Objective(),
            strategy=M14Schwarz3DStrategy(pad=2),
        )
        result = solver.fit(phi, record_history=True)
        V = six_tet_volumes_3d(result.corrected)
        assert (V <= 0).sum() == 0
        assert V.min() >= 0.01 - 1e-5
        # The stage-keyed info dict from iterative_3d_tet_refine_repair_schwarz
        # gets converted to SolveInfo.phases via _build_solve_info — each
        # top-level key becomes a phase. Expect both 'init' and 'final'.
        phase_names = [p.name for p in result.info.phases]
        assert 'init' in phase_names, f'expected init phase; got {phase_names}'
        assert 'final' in phase_names, f'expected final phase; got {phase_names}'

    def test_dense_field_falls_back_to_global(self):
        """A small dense field where one cluster spans the whole volume
        should fall back to global m14-3D."""
        from dvfopt import L2Objective, M14Schwarz3DStrategy, Solver, Tet6Constraint3D

        # Dense field — fold cells touch the global boundary so a
        # merge_dilation of 2 grows to cover the whole volume.
        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        phi[1, 2, 2, 2] = 1.5
        phi[2, 2, 2, 2] = 1.5

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=M14Schwarz3DStrategy(pad=1, fallback_size_ratio=0.5),
        )
        result = solver.fit(phi, record_history=True)
        V = six_tet_volumes_3d(result.corrected)
        # Still expect feasibility (fallback should still solve it).
        assert (V <= 0).sum() == 0

    def test_registry(self):
        from dvfopt import correct_dvf

        phi = np.zeros((3, 10, 10, 10))
        phi[1, 2, 2, 2] = 1.5
        phi[2, 2, 2, 2] = 1.5
        result = correct_dvf(phi, constraint='6tet', objective='l2', strategy='m14_schwarz_3d')
        assert result.feasible
        assert result.info.strategy_name == 'SchwarzHarmonicALMRefineRepair3DStrategy'


class TestM14TetStrategy:
    """Full m14-3D refine-repair pipeline (Phase D)."""

    @staticmethod
    def _phi_planted_fold_3d():
        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        return phi

    def test_reaches_feasibility(self):
        from dvfopt import L2Objective, M14TetStrategy, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=M14TetStrategy(),
        )
        result = solver.fit(self._phi_planted_fold_3d())
        V = six_tet_volumes_3d(result.corrected)
        assert (V <= 0).sum() == 0
        assert V.min() >= 0.01 - 1e-5

    def test_history_records_pipeline_stages(self):
        from dvfopt import L2Objective, M14TetStrategy, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=M14TetStrategy(),
        )
        result = solver.fit(self._phi_planted_fold_3d(), record_history=True)
        phase_names = [p.name for p in result.info.phases]
        # The stage-keyed dict from iterative_3d_tet_refine_repair becomes
        # SolveInfo.phases via _build_solve_info — each top-level key is
        # a phase. We expect at least stage1/2/3 to be present.
        assert any('stage1' in n for n in phase_names), f'no stage1; got {phase_names}'
        assert any('stage2' in n for n in phase_names), f'no stage2; got {phase_names}'
        assert any('stage3' in n for n in phase_names), f'no stage3; got {phase_names}'

    def test_registry(self):
        from dvfopt import correct_dvf

        result = correct_dvf(
            self._phi_planted_fold_3d(),
            constraint='6tet',
            objective='l2',
            strategy='m14_3d',
        )
        assert result.feasible
        assert result.info.strategy_name == 'HarmonicALMRefineRepair3DStrategy'

    def test_rejects_2tri(self):
        from dvfopt import (
            IncompatibleConstraintError,
            L2Objective,
            M14TetStrategy,
            Solver,
            TriConstraint2DFullCoverage,
        )

        with pytest.raises(IncompatibleConstraintError):
            Solver(
                constraint=TriConstraint2DFullCoverage(shape=(8, 8)),
                objective=L2Objective(),
                strategy=M14TetStrategy(),
            )


class TestHarmonic3DWallbreaker:
    """3D harmonic-extension wallbreaker — Phase 1 first cut.

    Validates the core algorithm directly (numpy) and via the
    `Harmonic3DStrategy` wrapper.
    """

    def test_harmonic_extension_clears_planted_fold(self):
        """Direct call to harmonic_extension_3d on the same planted fold
        that other tet solvers handle. Verifies feasibility + reports a
        patch record."""
        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d

        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        V_init = six_tet_volumes_3d(phi)
        assert (V_init <= 0).any()

        phi_out, info = harmonic_extension_3d(phi, record_history=True)
        V_out = six_tet_volumes_3d(phi_out)
        assert V_out.min() > 0
        assert info['patches'] >= 1

    def test_harmonic_identity_already_feasible(self):
        """Identity field is already feasible — the algorithm should
        return immediately with reason='already-feasible'."""
        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d

        phi = np.zeros((3, 4, 4, 4))
        phi_out, info = harmonic_extension_3d(phi, record_history=True)
        assert info['reason'] == 'already-feasible'
        np.testing.assert_array_equal(phi_out, phi)

    def test_merge_dilation_zero_keeps_patches_separate(self):
        """merge_dilation=0 must mean "no grouping dilation" — NOT scipy's
        binary_dilation(iterations=0) repeat-until-convergence, which
        fills the volume and collapses everything into one whole-volume
        patch."""
        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d

        phi = np.zeros((3, 5, 16, 16))
        phi[1, 2, 3, 3] = 1.5
        phi[2, 2, 3, 3] = 1.5
        phi[1, 2, 12, 12] = 1.5
        phi[2, 2, 12, 12] = 1.5
        assert (six_tet_volumes_3d(phi) <= 0).any()

        _phi_out, info = harmonic_extension_3d(phi, merge_dilation=0, record_history=True)
        assert info['patches'] == 2
        # Each patch is local, never the whole cell grid.
        n_cells_total = 4 * 15 * 15
        for rec in info['records']:
            assert rec['n_cells'] < n_cells_total

    def test_ring_pad_zero_stays_local(self):
        """ring_pad=0 with grow=0 used to call binary_dilation(iterations=0),
        which fills the whole volume (near-full-volume spsolve). The
        correct semantic is "no dilation this round" — far-away corners
        must be untouched."""
        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d

        phi = np.zeros((3, 5, 16, 16))
        phi[1, 2, 3, 3] = 1.5
        phi[2, 2, 3, 3] = 1.5
        phi_out = harmonic_extension_3d(phi, ring_pad=0, max_grow_iters=2)
        np.testing.assert_array_equal(phi_out[:, :, 10:, 10:], phi[:, :, 10:, 10:])

    def test_negative_dilation_params_raise(self):
        from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d

        phi = np.zeros((3, 4, 4, 4))
        with pytest.raises(ValueError, match='merge_dilation'):
            harmonic_extension_3d(phi, merge_dilation=-1)
        with pytest.raises(ValueError, match='ring_pad'):
            harmonic_extension_3d(phi, ring_pad=-1)

    def test_strategy_polish_off(self):
        """``Harmonic3DStrategy(polish=False)`` should reach feasibility
        without going through barrier."""
        from dvfopt import (
            Harmonic3DStrategy,
            L2Objective,
            Solver,
            Tet6Constraint3D,
        )

        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=Harmonic3DStrategy(polish=False),
        )
        result = solver.fit(phi)
        assert result.feasible
        assert result.info.strategy_name == 'Harmonic3DStrategy'

    def test_strategy_registry(self):
        """``strategy='harmonic_3d'`` should resolve correctly."""
        from dvfopt import correct_dvf

        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        result = correct_dvf(phi, constraint='6tet', objective='l2', strategy='harmonic_3d')
        assert result.feasible
        assert result.info.strategy_name == 'Harmonic3DStrategy'

    def test_rejects_2tri_constraint(self):
        """Strategy is 3D-tet-only — composing with a 2-tri constraint
        must fail at Solver init."""
        from dvfopt import (
            Harmonic3DStrategy,
            IncompatibleConstraintError,
            L2Objective,
            Solver,
            TriConstraint2DFullCoverage,
        )

        with pytest.raises(IncompatibleConstraintError):
            Solver(
                constraint=TriConstraint2DFullCoverage(shape=(8, 8)),
                objective=L2Objective(),
                strategy=Harmonic3DStrategy(),
            )


class TestTetBarrierTorch:
    """The GPU tet barrier should reach feasibility on the same planted
    fold the numpy barrier path solves, with comparable final L2."""

    def setup_method(self):
        pytest.importorskip('torch')

    def test_clears_planted_fold(self):
        from dvfopt.core.iterative3d_tet_barrier_torch import iterative_3d_tet_barrier_torch
        from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5

        phi_out = iterative_3d_tet_barrier_torch(phi, verbose=0, device='cpu', anchor='l2')
        V = six_tet_volumes_3d(phi_out)
        assert (V <= 0).sum() == 0
        assert V.min() >= 0.01 - 1e-4  # threshold (default 0.01), slack for float32

    def test_windowed_reaches_feasibility(self):
        """Windowed mode on an 8^3 volume with a single 2-magnitude
        corner fold — verify it converges + parity with full-grid."""
        from dvfopt.core.iterative3d_tet_barrier_torch import iterative_3d_tet_barrier_torch

        phi = np.zeros((3, 8, 8, 8))
        phi[1, 3, 3, 3] = 2.0
        phi[2, 3, 3, 3] = 2.0

        phi_w = iterative_3d_tet_barrier_torch(
            phi, windowed=True, pad=2, verbose=0, device='cpu', anchor='l2'
        )
        V_w = six_tet_volumes_3d(phi_w)
        assert (V_w <= 0).sum() == 0
        assert V_w.min() >= 0.01 - 1e-4

    def test_windowed_record_history_uses_min_T(self):
        """Windowed history must follow the canonical schema (min_T)."""
        from dvfopt.core.iterative3d_tet_barrier_torch import iterative_3d_tet_barrier_torch

        phi = np.zeros((3, 6, 6, 6))
        phi[1, 2, 2, 2] = 1.8
        phi[2, 2, 2, 2] = 1.8
        phi_w, history = iterative_3d_tet_barrier_torch(
            phi, windowed=True, pad=2, verbose=0, device='cpu', record_history=True
        )
        assert history, 'history should be non-empty'
        for h in history:
            assert 'min_T' in h, f'phase {h.get("phase")!r} missing min_T'

    def test_windowed_n_neg_uses_canonical_tet_count(self):
        """PR #15 review (e): ``n_neg`` in windowed-mode history must
        mean "tets with V <= 0" (canonical schema), NOT "cells below
        threshold". Before the fix the per-outer pre-record entries
        used the cells count while the final-record used the tets
        count, breaking downstream
        :meth:`SolveInfo.from_legacy_history` consumers."""
        from dvfopt.core.iterative3d_tet_barrier_torch import iterative_3d_tet_barrier_torch
        from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

        phi = np.zeros((3, 6, 6, 6))
        phi[1, 2, 2, 2] = 1.8
        phi[2, 2, 2, 2] = 1.8

        V0 = six_tet_volumes_3d(phi)
        expected_tets_neg = int((V0 <= 0).sum())
        # On this fixture the cell count is strictly less than the tet
        # count (each folded cell contributes multiple folded tets), so
        # the bug — if it returned — would surface as a mismatch here.
        expected_cells_below = int((V0.min(axis=0) < 0.01).sum())
        assert expected_tets_neg > expected_cells_below, (
            'fixture should plant more folded tets than folded cells '
            f'(tets={expected_tets_neg}, cells={expected_cells_below}) '
            'so the schema-mismatch bug has something to surface'
        )

        _, history = iterative_3d_tet_barrier_torch(
            phi, windowed=True, pad=2, verbose=0, device='cpu', record_history=True
        )
        pre_entry = next(h for h in history if h.get('phase', '').endswith('_pre'))
        assert pre_entry['n_neg'] == expected_tets_neg, (
            f'windowed pre-record n_neg={pre_entry["n_neg"]} should match '
            f'tet-count {expected_tets_neg}, not cell-count {expected_cells_below}'
        )
        # The cell-count is still preserved under a different key.
        assert pre_entry.get('n_fold_cells') == expected_cells_below

    def test_windowed_matches_full_grid_feasibility(self):
        """Both modes should reach feasibility on the same input. L2
        distances may differ slightly (windowed locks more boundary
        corners) but the feasibility verdict should be identical."""
        from dvfopt.core.iterative3d_tet_barrier_torch import iterative_3d_tet_barrier_torch

        phi = np.zeros((3, 6, 6, 6))
        phi[1, 2, 2, 2] = 1.5
        phi[2, 2, 2, 2] = 1.5

        phi_w = iterative_3d_tet_barrier_torch(phi, windowed=True, verbose=0, device='cpu')
        phi_f = iterative_3d_tet_barrier_torch(phi, windowed=False, verbose=0, device='cpu')
        V_w = six_tet_volumes_3d(phi_w)
        V_f = six_tet_volumes_3d(phi_f)
        assert (V_w <= 0).sum() == 0, 'windowed should be feasible'
        assert (V_f <= 0).sum() == 0, 'full-grid should be feasible'

    def test_no_torch_raises_clear_error(self):
        """If torch is unavailable, the public entry should raise
        ImportError with a clear message. Simulate via a temporary
        monkeypatch (we already know torch is installed if this test
        runs, but we can still test the guard at the module level)."""
        from dvfopt.core import iterative3d_tet_barrier_torch as mod

        # Save then null-out the module's torch reference; restore after.
        original = mod.torch
        mod.torch = None
        try:
            with pytest.raises(ImportError, match='torch'):
                mod.iterative_3d_tet_barrier_torch(np.zeros((3, 4, 4, 4)))
        finally:
            mod.torch = original


class TestSLSQPFullGrid3DStrategy:
    """End-to-end via the new Strategy wrapper, exercising both direct
    composition and the registry (``correct_dvf(strategy='slsqp_3d_tet')``)."""

    def test_direct_composition_clears_fold(self):
        from dvfopt import L2Objective, SLSQPFullGrid3DStrategy, Solver, Tet6Constraint3D

        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5

        c = Tet6Constraint3D(shape=(4, 4, 4))
        assert (c.values(c.flatten(phi)) <= 0).any()

        solver = Solver(
            constraint=c,
            objective=L2Objective(),
            strategy=SLSQPFullGrid3DStrategy(max_iter=200),
        )
        result = solver.fit(phi)
        assert result.feasible
        assert result.info.strategy_name == 'SLSQPFullGrid3DStrategy'

    def test_registry_via_correct_dvf(self):
        from dvfopt import correct_dvf

        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5

        result = correct_dvf(phi, constraint='6tet', objective='l2', strategy='slsqp_3d_tet')
        assert result.feasible
        assert result.info.strategy_name == 'SLSQPFullGrid3DStrategy'

    def test_rejects_wrong_constraint(self):
        """Strategy declares accepts_constraints = (Tet6Constraint3D,) —
        composing with a 2-tri constraint must fail at Solver init."""
        from dvfopt import (
            IncompatibleConstraintError,
            L2Objective,
            SLSQPFullGrid3DStrategy,
            Solver,
            TriConstraint2DFullCoverage,
        )

        with pytest.raises(IncompatibleConstraintError):
            Solver(
                constraint=TriConstraint2DFullCoverage(shape=(8, 8)),
                objective=L2Objective(),
                strategy=SLSQPFullGrid3DStrategy(),
            )


class TestSLSQPOnTet:
    """End-to-end: direct scipy SLSQP using ``Tet6Constraint3D.jacobian()``
    should clear a planted 3D fold. Today this is the only way to drive
    SLSQP on tet — no Strategy is wired (3D SLSQP doesn't scale, so
    there's no `SLSQPFullGrid3DStrategy`). When such a Strategy is
    eventually added, this test guards the math underneath it."""

    def test_clears_planted_fold(self):
        from scipy.optimize import NonlinearConstraint, minimize

        from dvfopt import Tet6Constraint3D

        # Single-corner push that flips multiple tets meeting at that corner.
        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5

        c = Tet6Constraint3D(shape=(4, 4, 4))
        flat_init = c.flatten(phi)
        V_init = c.values(flat_init)
        assert (V_init <= 0).any(), 'precondition: planted field should have folded tets'

        threshold = 0.01
        nlc = NonlinearConstraint(
            lambda z: c.values(z),
            lb=threshold,
            ub=np.inf,
            jac=lambda z: c.jacobian(z),
        )

        def obj(z):
            diff = z - flat_init
            return 0.5 * float(diff @ diff), diff

        res = minimize(
            obj,
            flat_init,
            method='SLSQP',
            jac=True,
            constraints=[nlc],
            options={'maxiter': 300, 'ftol': 1e-8},
        )
        assert res.success, f'SLSQP did not converge: status={res.status}'
        V_after = c.values(res.x)
        assert V_after.min() >= threshold - 1e-6, f'final min V = {V_after.min()} below threshold'


class TestTet6Constraint3D:
    """Verify the public Constraint surface (registry, shape, packing,
    barrier-strategy round-trip)."""

    def test_registry(self):
        from dvfopt import make_constraint
        from dvfopt.constraints import Tet6Constraint3D

        c = make_constraint('6tet', (3, 4, 5))
        assert isinstance(c, Tet6Constraint3D)
        c2 = make_constraint('6tet_3d', (3, 4, 5))
        assert isinstance(c2, Tet6Constraint3D)

    def test_shape_consistency(self):
        from dvfopt import Tet6Constraint3D

        D, H, W = 3, 4, 5
        c = Tet6Constraint3D(shape=(D, H, W))
        assert c.n_variables == 3 * D * H * W
        assert c.n_constraints == 6 * (D - 1) * (H - 1) * (W - 1)

    def test_flatten_unflatten_round_trip(self):
        from dvfopt import Tet6Constraint3D

        rng = np.random.default_rng(0)
        c = Tet6Constraint3D(shape=(3, 4, 5))
        phi = rng.normal(0, 0.1, (3, 3, 4, 5))
        np.testing.assert_array_equal(c.unflatten(c.flatten(phi)), phi)

    def test_identity_feasible(self):
        from dvfopt import Tet6Constraint3D

        c = Tet6Constraint3D(shape=(3, 4, 5))
        phi = np.zeros((3, 3, 4, 5))
        flat = c.flatten(phi)
        V = c.values(flat)
        assert np.allclose(V, 1 / 6)

    def test_auto_strategy_routes_to_barrier(self):
        """``auto_strategy`` must route ``Tet6Constraint3D`` to barrier
        (it's the only strategy that supports tet today). Regression:
        early versions fell through to ``slsqp_windowed`` which doesn't
        accept tet constraints."""
        from dvfopt import Tet6Constraint3D
        from dvfopt.solver import auto_strategy

        c = Tet6Constraint3D(shape=(4, 5, 5))
        # Across a range of fold densities, tet should always go barrier.
        for n_neg, init_min in [(1, -0.05), (200, -0.5), (10000, -5.0)]:
            label = auto_strategy(c, init_n_neg=n_neg, init_min=init_min, objective_label='l2')
            assert label == 'barrier', f'n_neg={n_neg}: got {label!r}, expected barrier'

    def test_auto_dispatch_via_correct_dvf(self):
        """End-to-end: ``correct_dvf(constraint='6tet', strategy='auto')``
        should reach feasibility, not crash."""
        from dvfopt import correct_dvf

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.05, (3, 4, 5, 5))
        phi[1, 2, 2:4, 2:4] -= 1.0
        phi[2, 2, 2:4, 2:4] -= 1.0
        result = correct_dvf(phi, constraint='6tet', objective='l2', strategy='auto')
        assert result.feasible
        assert result.info.strategy_name == 'BarrierStrategy'

    def test_end_to_end_barrier_reaches_feasibility(self):
        """A small folded 3D field should be feasibilised by barrier
        through the Tet6Constraint3D pipeline."""
        from dvfopt import BarrierStrategy, L2Objective, Solver, Tet6Constraint3D

        rng = np.random.default_rng(0)
        phi = rng.normal(0, 0.05, (3, 4, 5, 5))
        # Inject a modest fold.
        phi[1, 2, 2:4, 2:4] -= 1.0
        phi[2, 2, 2:4, 2:4] -= 1.0

        c = Tet6Constraint3D(shape=(4, 5, 5))
        assert c.values(c.flatten(phi)).min() <= 0, 'precondition: should start folded'

        solver = Solver(constraint=c, objective=L2Objective(), strategy=BarrierStrategy())
        result = solver.fit(phi)
        assert result.feasible, 'barrier should reach feasibility on this small case'
        # And the corrected field should produce strictly positive volumes.
        V_after = c.values(c.flatten(result.corrected))
        assert V_after.min() > 0.0


# ---------------------------------------------------------------------------
# Regression: history-schema parity across the new 3D-tet solvers.
#
# All three new solvers (SLSQP-3D-tet, GPU tet barrier, Harmonic3DStrategy +
# polish) emit ``history`` dicts that ``SolveInfo.from_legacy_history`` /
# ``_build_solve_info`` consume. The shared adapter reads ``min_T`` (or its
# legacy alias ``min_tri``); the original implementations used ``min_V``,
# which silently fell back to NaN and broke ``feasible_after_phase``
# detection. Pin the schema below so it doesn't regress.
# ---------------------------------------------------------------------------


class TestHistorySchemaParity:
    """The three new 3D-tet solvers must emit history dicts compatible
    with ``SolveInfo.from_legacy_history`` — i.e. use the canonical
    ``min_T`` key for the minimum constraint value. Regressions caught
    by the PR #13 code review."""

    @staticmethod
    def _phi_planted_fold_3d():
        phi = np.zeros((3, 4, 4, 4))
        phi[1, 1, 1, 1] = 1.5
        phi[2, 1, 1, 1] = 1.5
        return phi

    def test_slsqp_3d_tet_history_uses_min_T(self):
        from dvfopt.core.iterative3d_tet_slsqp import iterative_3d_tet_slsqp

        phi_out, history = iterative_3d_tet_slsqp(
            self._phi_planted_fold_3d(), verbose=0, record_history=True
        )
        assert isinstance(history, list) and history, 'history must be a non-empty list'
        for h in history:
            assert 'min_T' in h, f'phase {h.get("phase")!r} missing min_T (got keys={list(h)})'
            assert 'min_V' not in h, 'min_V is the wrong key — use min_T (canonical schema)'

    def test_slsqp_3d_tet_solve_info_populates_min_T(self):
        """End-to-end: ``Solver.fit`` should produce a SolveInfo whose
        phases have a finite ``min_T``."""
        from dvfopt import L2Objective, SLSQPFullGrid3DStrategy, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=SLSQPFullGrid3DStrategy(max_iter=200),
        )
        result = solver.fit(self._phi_planted_fold_3d(), record_history=True)
        info = result.info
        assert info.phases, 'expected at least one phase in SolveInfo'
        for p in info.phases:
            assert not np.isnan(p.min_T), f'phase {p.name!r} has NaN min_T — schema mismatch'

    def test_tet_barrier_torch_history_uses_min_T(self):
        pytest.importorskip('torch')
        from dvfopt.core.iterative3d_tet_barrier_torch import iterative_3d_tet_barrier_torch

        phi_out, history = iterative_3d_tet_barrier_torch(
            self._phi_planted_fold_3d(), verbose=0, device='cpu', record_history=True
        )
        assert isinstance(history, list) and history
        for h in history:
            assert 'min_T' in h, f'phase {h.get("phase")!r} missing min_T (got keys={list(h)})'
            assert 'min_V' not in h, 'min_V is the wrong key — use min_T (canonical schema)'

    def test_harmonic_3d_polish_emits_chained_phases(self):
        """``Harmonic3DStrategy(polish=True)`` chains the harmonic step
        + each barrier-polish phase as separate entries in the SolveInfo
        history. The prior bug (``polish_info.to_dict()`` guard always
        falling through to ``{}``) silently dropped the polish phases.

        After the fix, ``info.phases`` has 1 harmonic + N polish entries,
        each with a finite ``min_T``.
        """
        from dvfopt import Harmonic3DStrategy, L2Objective, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=Harmonic3DStrategy(polish=True),
        )
        result = solver.fit(self._phi_planted_fold_3d(), record_history=True)
        info = result.info

        phase_names = [p.name for p in info.phases]
        assert any(n == 'harmonic' for n in phase_names), (
            f'expected a harmonic phase; got {phase_names}'
        )
        assert any(n.startswith('polish_') for n in phase_names), (
            f'expected at least one polish_* phase (polish=True); got {phase_names}'
        )
        for p in info.phases:
            assert not np.isnan(p.min_T), f'phase {p.name!r} has NaN min_T — schema mismatch'

    def test_harmonic_3d_polish_off_emits_just_harmonic_phase(self):
        from dvfopt import Harmonic3DStrategy, L2Objective, Solver, Tet6Constraint3D

        solver = Solver(
            constraint=Tet6Constraint3D(shape=(4, 4, 4)),
            objective=L2Objective(),
            strategy=Harmonic3DStrategy(polish=False),
        )
        result = solver.fit(self._phi_planted_fold_3d(), record_history=True)
        info = result.info
        phase_names = [p.name for p in info.phases]
        assert phase_names == ['harmonic'], (
            f'polish=False should produce exactly one harmonic phase; got {phase_names}'
        )
        assert not np.isnan(info.phases[0].min_T)
