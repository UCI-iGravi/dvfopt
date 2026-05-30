"""Centralized matplotlib + seaborn styling for DVFopt visualizations.

The package previously had no global theme — every plot function set its
own font sizes, colors, layout. This module is the single source of
truth: every visualization in :mod:`dvfopt.viz` and :mod:`dvfopt._plots`
calls :func:`apply_theme` once (idempotent), then uses the named palette
constants in :class:`Palette`.

Usage
-----

In a script or notebook::

    import dvfopt.viz as viz
    viz.apply_theme()                    # one-time, idempotent
    viz.plot_fold_overview(phi, title='B0039 z=12')

In package-internal plots: each viz function calls ``apply_theme()``
at the top (idempotent overhead is negligible). Users who don't want
the theme can pass ``style='matplotlib'`` to skip it.

Design notes
------------

* **Theme = seaborn 'ticks' + paper context.** Clean spines, subtle
  grid, publication-friendly font sizes. Reads well on screen AND
  prints well in PDFs.
* **Colormaps**: ``cmocean`` is gorgeous but adds a dependency we
  don't need. Stick with matplotlib defaults but pick deliberately:
  ``RdBu_r`` for diverging (centered on 0), ``magma`` for sequential
  positive, ``Spectral_r`` for fold-severity green→red.
* **High-DPI by default.** ``rcParams['figure.dpi'] = 130`` —
  reasonable on most modern screens, doesn't blow up file sizes.
* **DataFrame-ready palette**: the :class:`Palette` colors are
  consistent with seaborn's default categorical cycle so any
  ``sns.lineplot()`` you stack on top harmonizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl

if TYPE_CHECKING:
    import matplotlib.colors as mcolors
    import numpy as np

_THEME_APPLIED = False


@dataclass(frozen=True)
class Palette:
    """Curated palette for DVFopt plots. Frozen — assign to module-level
    constants below for stable references."""

    # Categorical (used for solver labels in convergence plots, etc.).
    # First three match seaborn's deep palette for harmony.
    blue: str = '#4878d0'  # primary — barrier, m10
    orange: str = '#ee854a'  # secondary — slsqp
    green: str = '#6acc64'  # tertiary — m14
    red: str = '#d65f5f'  # warning — folds, errors
    purple: str = '#956cb4'  # m14-schwarz
    brown: str = '#8c613c'
    gray: str = '#797979'

    # Semantic.
    fold: str = '#d62728'  # folded-cell highlight
    feasible: str = '#2ca02c'  # feasibility threshold line
    anchor: str = '#1f77b4'  # anchor-iterate trace
    grid_warp: str = '#5b7fb5'  # warped grid lines (subtle)
    grid_ref: str = '#e0e0e0'  # reference grid (very subtle)

    # Colormaps.
    cmap_jdet: str = 'RdBu_r'  # diverging, centered on 0
    cmap_severity: str = 'YlOrRd'  # fold severity (white → yellow → red)
    cmap_magnitude: str = 'magma'  # sequential positive (displacement mag)


PALETTE = Palette()


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------


def apply_theme(context: str = 'paper', force: bool = False) -> None:
    """Apply the DVFopt visual theme. Idempotent.

    Parameters
    ----------
    context : str
        ``'paper'`` (default, smaller text, for figures),
        ``'notebook'`` (medium), ``'talk'`` (larger).
    force : bool
        Re-apply even if already applied. Default ``False`` — the
        common case is "apply once on first import, ignore subsequent
        calls."
    """
    global _THEME_APPLIED
    if _THEME_APPLIED and not force:
        return

    # Import here so the package doesn't require seaborn unless a plot
    # is actually drawn.
    import seaborn as sns

    sns.set_theme(
        context=context,
        style='ticks',
        font='DejaVu Sans',
        font_scale=1.0,
        rc={
            # ---- Figure ----
            'figure.dpi': 130,
            'figure.facecolor': 'white',
            'figure.constrained_layout.use': True,
            # ---- Axes ----
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'axes.edgecolor': '#222222',
            'axes.titlesize': 'medium',
            'axes.titleweight': 'normal',
            'axes.titlepad': 8,
            'axes.labelsize': 'small',
            'axes.labelweight': 'normal',
            'axes.labelpad': 4,
            'axes.grid': False,  # off by default; viz fns enable per-axis
            'axes.grid.which': 'major',
            'grid.color': '#dddddd',
            'grid.linewidth': 0.6,
            'grid.alpha': 0.6,
            # ---- Ticks ----
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            # ---- Lines / markers ----
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            # ---- Legend ----
            'legend.fontsize': 'x-small',
            'legend.frameon': False,
            'legend.borderaxespad': 0.3,
            # ---- Images ----
            'image.interpolation': 'nearest',
            'image.cmap': PALETTE.cmap_jdet,
            # ---- Saving ----
            'savefig.dpi': 200,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.transparent': False,
            # ---- Fonts ----
            'font.size': 9,
        },
    )
    _THEME_APPLIED = True


def reset_theme() -> None:
    """Restore matplotlib defaults. Useful in tests."""
    global _THEME_APPLIED
    mpl.rcdefaults()
    _THEME_APPLIED = False


# ---------------------------------------------------------------------------
# Norms (diverging-centered, with sensible defaults for Jdet ranges)
# ---------------------------------------------------------------------------


def jdet_norm(jdet_arrays: list[np.ndarray], threshold: float = 0.01) -> mcolors.TwoSlopeNorm:
    """Build a :class:`matplotlib.colors.TwoSlopeNorm` centered on 0 that
    spans every supplied Jdet array. Useful for before/after panels
    sharing a colorbar."""
    import matplotlib.colors as mcolors

    vmin = min(float(j.min()) for j in jdet_arrays)
    vmax = max(float(j.max()) for j in jdet_arrays)
    return mcolors.TwoSlopeNorm(
        vmin=min(vmin, -max(0.5, threshold * 50)),
        vcenter=0.0,
        vmax=max(vmax, 1.0),
    )


__all__ = [
    'PALETTE',
    'Palette',
    'apply_theme',
    'jdet_norm',
    'reset_theme',
]
