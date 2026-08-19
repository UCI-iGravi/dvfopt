"""Smoke tests for the rest of dvfopt.viz.

These don't verify pixel correctness — they verify each plotter renders
without exceptions and produces a figure with the expected number of
axes. The goal is a coverage safety net: if a refactor breaks the API
of a plotting function (renamed argument, dropped parameter), this
catches it before a user discovers it.

Theme + overview already have dedicated tests in
``test_viz_theme.py``; this file covers everything else
(fields, fields3d, grids, closeups, snapshots, triangle_debug).
"""

from __future__ import annotations

import warnings

import matplotlib

# Must select the headless backend *before* importing pyplot — these
# two lines genuinely belong between imports, hence the noqa.
matplotlib.use('Agg')

# Several legacy viz functions call plt.show() on a non-interactive
# Agg backend — harmless but noisy. Silence at file level.
warnings.filterwarnings(
    'ignore',
    message='FigureCanvasAgg is non-interactive',
    category=UserWarning,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D  # noqa: E402
from dvfopt.viz import (  # noqa: E402
    DebugTracer,
    find_problematic_pixels,
    plot_2d_deformation_grid,
    plot_checkerboard_before_after,
    plot_deformation_field,
    plot_deformation_grid_3d,
    plot_deformations,
    plot_deformed_quads,
    plot_deformed_quads_colored,
    plot_grid,
    plot_grid_before_after,
    plot_grid_before_after_3d,
    plot_initial_deformation,
    plot_jacobians_iteratively,
    plot_jdet_3d,
    plot_jdet_3d_before_after,
    plot_jdet_slices,
    plot_neg_jdet_neighborhoods,
    plot_neg_voxels_before_after,
    plot_problematic_triangles,
    plot_step_snapshot,
    plot_triangle_debug,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def folded_2d():
    """``(2, H, W)`` folded synthetic field."""
    rng = np.random.default_rng(0)
    H, W = 12, 12
    phi = np.stack([rng.normal(0, 0.2, (H, W)), rng.normal(0, 0.2, (H, W))])
    phi[0, 5, 5] = -1.8
    phi[1, 5, 5] = -1.8
    return phi


@pytest.fixture
def folded_3channel_singleton(folded_2d):
    """``(3, 1, H, W)`` form expected by some of the older fields.py
    functions (laplacian-pipeline output convention)."""
    H, W = folded_2d.shape[1:]
    return np.stack([np.zeros((H, W)), folded_2d[0], folded_2d[1]])[:, None]


@pytest.fixture
def folded_3d():
    rng = np.random.default_rng(0)
    D, H, W = 5, 8, 8
    phi = np.stack([rng.normal(0, 0.1, (D, H, W)) for _ in range(3)])
    phi[1, D // 2, H // 2, W // 2] -= 1.5
    phi[2, D // 2, H // 2, W // 2] -= 1.5
    return phi


# ---------------------------------------------------------------------------
# fields.py
# ---------------------------------------------------------------------------


def _close_all():
    """Close every open matplotlib figure. Some viz functions in this
    codebase pre-date the 'return the figure' convention — they call
    ``plt.show()`` internally and return ``None``. Closing by handle
    isn't always possible, so we close globally between tests."""
    plt.close('all')


class TestFields:
    def teardown_method(self):
        _close_all()

    def test_initial_deformation(self, folded_3channel_singleton):
        # plot_initial_deformation may return None (calls plt.show);
        # smoke test only — no exception, figures exist after call.
        plot_initial_deformation(folded_3channel_singleton)
        assert plt.get_fignums(), 'expected at least one figure created'

    def test_initial_deformation_with_correspondences(self, folded_3channel_singleton):
        msample = np.array([[0, 3, 3], [0, 5, 5]], dtype=float)
        fsample = np.array([[0, 4, 4], [0, 6, 6]], dtype=float)
        plot_initial_deformation(folded_3channel_singleton, msample=msample, fsample=fsample)

    def test_deformations(self, folded_2d, folded_3channel_singleton):
        msample = np.array([[0, 3, 3]], dtype=float)
        fsample = np.array([[0, 4, 4]], dtype=float)
        # plot_deformations expects deformation_i as (3, 1, H, W) and
        # phi_corrected as (2, H, W).
        plot_deformations(
            msample,
            fsample,
            folded_3channel_singleton,
            folded_2d,
        )

    def test_deformation_field(self, folded_3channel_singleton):
        plot_deformation_field(folded_3channel_singleton)

    def test_jacobians_iteratively(self, folded_3channel_singleton):
        jdets = [
            jacobian_det2D(folded_3channel_singleton[1:][:, 0]),
            jacobian_det2D(folded_3channel_singleton[1:][:, 0]) * 0.8,
        ]
        plot_jacobians_iteratively(jdets)


# ---------------------------------------------------------------------------
# fields3d.py
# ---------------------------------------------------------------------------


class TestFields3d:
    def teardown_method(self):
        _close_all()

    def test_jdet_slices(self, folded_3d):
        jb = jacobian_det3D(folded_3d)
        ja = jb * 0.5
        fig = plot_jdet_slices(jb, ja)
        assert len(fig.axes) >= 2
        if fig is not None:
            plt.close(fig)

    def test_jdet_slices_max_slices(self, folded_3d):
        jb = jacobian_det3D(folded_3d)
        fig = plot_jdet_slices(jb, jb * 0.5, max_slices=2)
        if fig is not None:
            plt.close(fig)

    def test_jdet_3d(self, folded_3d):
        jb = jacobian_det3D(folded_3d)
        fig = plot_jdet_3d(jb)
        if fig is not None:
            plt.close(fig)

    def test_jdet_3d_before_after(self, folded_3d):
        jb = jacobian_det3D(folded_3d)
        fig = plot_jdet_3d_before_after(jb, jb * 0.5)
        if fig is not None:
            plt.close(fig)

    def test_neg_voxels_before_after(self, folded_3d):
        jb = jacobian_det3D(folded_3d)
        fig = plot_neg_voxels_before_after(jb, jb * 0.5)
        if fig is not None:
            plt.close(fig)

    def test_deformation_grid_3d(self, folded_3d):
        fig = plot_deformation_grid_3d(folded_3d, spacing=2)
        if fig is not None:
            plt.close(fig)

    def test_grid_before_after_3d(self, folded_3d):
        fig = plot_grid_before_after_3d(folded_3d, folded_3d * 0.5, spacing=2)
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# grids.py
# ---------------------------------------------------------------------------


class TestGrids:
    def teardown_method(self):
        _close_all()

    def test_2d_deformation_grid(self, folded_3channel_singleton):
        fig = plot_2d_deformation_grid(folded_3channel_singleton)
        if fig is not None:
            plt.close(fig)

    def test_deformed_quads(self, folded_3channel_singleton):
        H, W = folded_3channel_singleton.shape[-2:]
        fig = plot_deformed_quads(folded_3channel_singleton, H // 2, W // 2, patch_size=4)
        if fig is not None:
            plt.close(fig)

    def test_deformed_quads_colored(self, folded_3channel_singleton):
        H, W = folded_3channel_singleton.shape[-2:]
        plot_deformed_quads_colored(folded_3channel_singleton, H // 2, W // 2, patch_size=4)

    def test_grid(self, folded_3channel_singleton):
        fig = plot_grid(folded_3channel_singleton)
        if fig is not None:
            plt.close(fig)

    def test_grid_before_after(self, folded_2d, folded_3channel_singleton):
        # Signature: (deformation_i (3,1,H,W), phi_corrected (2,H,W)).
        plot_grid_before_after(folded_3channel_singleton, folded_2d)


# ---------------------------------------------------------------------------
# closeups.py
# ---------------------------------------------------------------------------


class TestCloseups:
    def teardown_method(self):
        _close_all()

    def test_checkerboard_before_after(self, folded_2d, folded_3channel_singleton):
        # Signature: (deformation_i (3,1,H,W), phi_corrected (2,H,W)).
        plot_checkerboard_before_after(folded_3channel_singleton, folded_2d, max_panels=2)

    def test_neg_jdet_neighborhoods(self, folded_2d, folded_3channel_singleton):
        plot_neg_jdet_neighborhoods(folded_3channel_singleton, folded_2d, max_panels=2, half_win=2)


# ---------------------------------------------------------------------------
# snapshots.py
# ---------------------------------------------------------------------------


class TestSnapshots:
    def teardown_method(self):
        _close_all()

    def test_step_snapshot_single(self, folded_3channel_singleton):
        jdet = jacobian_det2D(folded_3channel_singleton[1:][:, 0])
        plot_step_snapshot(jdet, iteration=0, neg_count=1, min_val=float(jdet.min()))

    def test_step_snapshot_before_after(self, folded_3channel_singleton):
        jdet = jacobian_det2D(folded_3channel_singleton[1:][:, 0])
        plot_step_snapshot(
            jdet,
            iteration=0,
            neg_count=1,
            min_val=float(jdet.min()),
            jacobian_before=jdet,
        )


# ---------------------------------------------------------------------------
# triangle_debug.py
# ---------------------------------------------------------------------------


class TestTriangleDebug:
    def teardown_method(self):
        _close_all()

    def test_triangle_debug(self, folded_2d):
        H, W = folded_2d.shape[1:]
        fig = plot_triangle_debug(folded_2d, x=W // 2, y=H // 2)
        if fig is not None:
            plt.close(fig)

    def test_problematic_triangles(self, folded_2d):
        # plot_problematic_triangles returns a LIST of figures (one
        # per bad pixel up to max_plots). Close them all.
        result = plot_problematic_triangles(folded_2d, max_plots=2)
        if isinstance(result, list):
            for f in result:
                plt.close(f)
        elif result is not None:
            plt.close(result)

    def test_find_problematic_pixels(self, folded_2d):
        bad = find_problematic_pixels(folded_2d)
        assert hasattr(bad, '__len__'), 'find_problematic_pixels should return a sequence'


# ---------------------------------------------------------------------------
# debug.py
# ---------------------------------------------------------------------------


class TestDebugTracer:
    def test_constructs(self):
        tracer = DebugTracer()
        assert tracer is not None


# ---------------------------------------------------------------------------
# solveinfo.py
# ---------------------------------------------------------------------------


class TestPlotSolveInfo:
    def test_renders_real_solve_history(self):
        from dvfopt import BarrierStrategy, L2Objective, Solver, TriConstraint2D
        from dvfopt.viz import plot_solve_info

        rng = np.random.default_rng(3)
        H, W = 10, 10
        phi = np.stack([rng.normal(0, 0.2, (H, W)), rng.normal(0, 0.2, (H, W))])
        phi[:, 4:6, 4:6] -= 1.2  # punch a fold
        result = Solver(
            constraint=TriConstraint2D(shape=(H, W)),
            objective=L2Objective(),
            strategy=BarrierStrategy(max_iter=50),
        ).fit(phi, record_history=True)
        fig = plot_solve_info(result.info, threshold=0.01)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_renders_empty_info(self):
        from dvfopt.solver import SolveInfo
        from dvfopt.viz import plot_solve_info

        fig = plot_solve_info(SolveInfo(strategy_name='EmptyStrategy'))
        assert len(fig.axes) == 2
        plt.close(fig)
