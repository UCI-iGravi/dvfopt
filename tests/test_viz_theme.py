"""Tests for the visualization theme + overview plots.

We don't pixel-compare — these tests check that the figures render
without crashing, the theme is applied + idempotent, and the
:class:`Palette` constants are referenceable. Pixel-accurate baselines
add maintenance overhead far beyond their value here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dvfopt.viz import (
    PALETTE,
    apply_theme,
    jdet_norm,
    plot_before_after,
    plot_before_after_3d,
    plot_fold_overview,
    plot_fold_overview_3d,
    plot_solver_comparison,
    reset_theme,
)


@pytest.fixture
def folded_phi():
    rng = np.random.default_rng(0)
    H, W = 10, 10
    phi = np.stack([rng.normal(0, 0.2, (H, W)), rng.normal(0, 0.2, (H, W))])
    # punch a fold at the center
    phi[0, 4:6, 4:6] -= 1.5
    phi[1, 4:6, 4:6] -= 1.5
    return phi


@pytest.fixture
def feasible_phi():
    rng = np.random.default_rng(1)
    H, W = 10, 10
    return np.stack([rng.normal(0, 0.05, (H, W)), rng.normal(0, 0.05, (H, W))])


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------


class TestApplyTheme:
    def test_apply_then_reset_round_trip(self):
        reset_theme()
        default_dpi = plt.rcParams['figure.dpi']
        apply_theme()
        assert plt.rcParams['figure.dpi'] == 130
        assert plt.rcParams['axes.spines.top'] is False
        assert plt.rcParams['axes.spines.right'] is False
        assert plt.rcParams['image.cmap'] == PALETTE.cmap_jdet
        reset_theme()
        # rcdefaults restores the matplotlib defaults — dpi back to 100.
        assert plt.rcParams['figure.dpi'] == default_dpi or plt.rcParams['figure.dpi'] == 100

    def test_apply_is_idempotent(self):
        reset_theme()
        apply_theme()
        dpi_after_first = plt.rcParams['figure.dpi']
        # Mutate, then call again — without force=True it should NOT re-apply
        plt.rcParams['figure.dpi'] = 999
        apply_theme()
        assert plt.rcParams['figure.dpi'] == 999, 'idempotent apply should not overwrite'
        # With force=True, it does re-apply.
        apply_theme(force=True)
        assert plt.rcParams['figure.dpi'] == dpi_after_first

    def test_context_levels(self):
        reset_theme()
        apply_theme(context='paper', force=True)
        paper_font = plt.rcParams['font.size']
        apply_theme(context='talk', force=True)
        talk_font = plt.rcParams['font.size']
        # 'talk' context = larger fonts than 'paper'
        assert talk_font >= paper_font
        reset_theme()

    def test_apply_theme_does_not_leak_layout(self, folded_phi):
        # Regression: apply_theme must NOT set constrained_layout in GLOBAL
        # rcParams. A global default leaks into non-dvfopt figures and makes
        # their fig.tight_layout() (with a colorbar) raise — this broke
        # cohort_benchmark / interactive_report and caused a Qt abort via
        # test ordering. dvfopt applies constrained_layout PER FIGURE instead.
        reset_theme()
        apply_theme(force=True)
        assert plt.rcParams['figure.constrained_layout.use'] is False
        # A dvfopt plot (applies theme, per-figure constrained_layout) ...
        dfig = plot_fold_overview(folded_phi)
        plt.close(dfig)
        # ... must not have poisoned the global default for foreign code:
        fig, ax = plt.subplots()  # no constrained_layout kwarg → matplotlib default
        im = ax.imshow(np.zeros((4, 4)))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()  # must NOT raise
        plt.close(fig)
        reset_theme()


class TestPalette:
    def test_palette_constants_are_strings(self):
        for attr in (
            'blue',
            'orange',
            'green',
            'red',
            'purple',
            'fold',
            'feasible',
            'anchor',
            'grid_warp',
            'grid_ref',
            'cmap_jdet',
            'cmap_severity',
            'cmap_magnitude',
        ):
            v = getattr(PALETTE, attr)
            assert isinstance(v, str) and v, f'PALETTE.{attr} should be a non-empty string'

    def test_palette_is_frozen(self):
        # frozen dataclass raises FrozenInstanceError (subclass of AttributeError)
        # when assignment is attempted.
        from dataclasses import FrozenInstanceError

        with pytest.raises((FrozenInstanceError, AttributeError)):
            PALETTE.blue = '#000000'


# ---------------------------------------------------------------------------
# jdet_norm
# ---------------------------------------------------------------------------


class TestJdetNorm:
    def test_centered_on_zero(self):
        jdet = np.array([[-0.5, 0.0], [0.5, 1.2]])
        n = jdet_norm([jdet])
        assert n.vcenter == 0.0
        assert n.vmin < 0
        assert n.vmax > 0

    def test_handles_multiple_arrays(self):
        a = np.array([[-2.0, 0.0]])
        b = np.array([[0.0, 3.0]])
        n = jdet_norm([a, b])
        assert n.vmin <= -2.0
        assert n.vmax >= 3.0


# ---------------------------------------------------------------------------
# Overview plots
# ---------------------------------------------------------------------------


class TestPlotFoldOverview:
    def test_renders_with_folds(self, folded_phi):
        fig = plot_fold_overview(folded_phi)
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_renders_feasible_field_no_folds(self, feasible_phi):
        fig = plot_fold_overview(feasible_phi)
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_accepts_3channel_input(self, folded_phi):
        phi3 = np.stack([np.zeros_like(folded_phi[0]), folded_phi[0], folded_phi[1]])
        fig = plot_fold_overview(phi3)
        plt.close(fig)

    def test_accepts_3channel_singleton_d(self, folded_phi):
        phi31hw = np.stack([np.zeros_like(folded_phi[0]), folded_phi[0], folded_phi[1]])[:, None]
        fig = plot_fold_overview(phi31hw)
        plt.close(fig)

    def test_save_path(self, folded_phi, tmp_path):
        out = tmp_path / 'overview.png'
        fig = plot_fold_overview(folded_phi, save_path=str(out))
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 0

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match='cannot interpret'):
            plot_fold_overview(np.zeros((5, 5)))

    def test_triangle_layer_drawn_for_folded_cells(self, folded_phi):
        """The warped-grid panel should add at least one filled Polygon
        patch (the triangle-fill layer) when folds exist."""
        from matplotlib.patches import Polygon

        fig = plot_fold_overview(folded_phi)
        warped_grid_ax = fig.axes[1]
        polys = [p for p in warped_grid_ax.patches if isinstance(p, Polygon)]
        assert polys, 'expected at least one triangle Polygon patch on a folded field'
        plt.close(fig)

    def test_no_triangle_layer_for_feasible_field(self, feasible_phi):
        from matplotlib.patches import Polygon

        fig = plot_fold_overview(feasible_phi)
        warped_grid_ax = fig.axes[1]
        polys = [p for p in warped_grid_ax.patches if isinstance(p, Polygon)]
        assert not polys, 'no folds → no Polygon fills expected'
        plt.close(fig)


class TestPlotBeforeAfter:
    def test_renders(self, folded_phi):
        fig = plot_before_after(folded_phi, folded_phi * 0.5)
        # 2 jdet panels + magnitude panel + 1 colorbar
        assert len(fig.axes) >= 3
        plt.close(fig)


@pytest.fixture
def folded_phi_3d():
    D, H, W = 6, 10, 10
    z, y, x = np.mgrid[:D, :H, :W]
    phi = np.stack(
        [
            0.2 * np.sin(2 * np.pi * z / D),
            0.25 * np.sin(2 * np.pi * x / W),
            0.25 * np.cos(2 * np.pi * y / H),
        ]
    )
    phi[1, D // 2 - 1 : D // 2 + 2, H // 2 - 1 : H // 2 + 2, W // 2 - 1 : W // 2 + 2] -= 1.5
    phi[2, D // 2 - 1 : D // 2 + 2, H // 2 - 1 : H // 2 + 2, W // 2 - 1 : W // 2 + 2] -= 1.5
    return phi


@pytest.fixture
def feasible_phi_3d():
    rng = np.random.default_rng(1)
    return rng.normal(0, 0.05, (3, 6, 8, 8))


class TestPlotFoldOverview3d:
    def test_renders_with_folds(self, folded_phi_3d):
        fig = plot_fold_overview_3d(folded_phi_3d)
        # 4 panels (1 is a 3D scatter, others are 2D) + 1 colorbar = 5
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_renders_feasible_field(self, feasible_phi_3d):
        fig = plot_fold_overview_3d(feasible_phi_3d)
        plt.close(fig)

    def test_save_path(self, folded_phi_3d, tmp_path):
        out = tmp_path / 'overview_3d.png'
        fig = plot_fold_overview_3d(folded_phi_3d, save_path=str(out))
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 0

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError, match='cannot interpret'):
            plot_fold_overview_3d(np.zeros((2, 4, 4)))


class TestPlotBeforeAfter3d:
    def test_renders(self, folded_phi_3d):
        fig = plot_before_after_3d(folded_phi_3d, folded_phi_3d * 0.3)
        assert len(fig.axes) >= 2
        plt.close(fig)


class TestPlotSolverComparison:
    def test_renders_n_solvers(self, folded_phi):
        results = {
            'slsqp': folded_phi * 0.7,
            'barrier': folded_phi * 0.4,
            'm14': folded_phi * 0.2,
        }
        fig = plot_solver_comparison(folded_phi, results)
        # 1 input + 3 outputs = 4 jdet panels + colorbar
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_empty_results_dict(self, folded_phi):
        fig = plot_solver_comparison(folded_phi, {})
        plt.close(fig)
