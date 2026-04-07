"""Per-step snapshot plotting (used by iterative solvers via ``plot_every``)."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dvfopt.viz._style import (
    CMAP, INTERP,
    _make_jdet_norm, _annotate_jdet_values, _annotate_neg_contour,
)


def _draw_windows(ax, windows):
    """Draw sub-window rectangles on *ax*."""
    for entry in windows:
        cy, cx, sz = entry[0], entry[1], entry[2]
        at_edge = entry[3] if len(entry) > 3 else False
        # Support rectangular (sy, sx) or scalar sizes
        if isinstance(sz, (tuple, list)):
            sy, sx = int(sz[0]), int(sz[1])
        else:
            sy = sx = int(sz)
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx
        # Outer rectangle — full window boundary (dashed cyan)
        rect_outer = Rectangle(
            (cx - hx - 0.5, cy - hy - 0.5), sx, sy,
            linewidth=2, edgecolor='cyan', facecolor='none',
            linestyle='--')
        ax.add_patch(rect_outer)
        # Frozen border ring — only when edges are truly frozen
        if sy > 2 and sx > 2 and not at_edge:
            outer = Rectangle(
                (cx - hx - 0.5, cy - hy - 0.5), sx, sy,
                linewidth=0, edgecolor='none', facecolor='cyan',
                alpha=0.25, hatch='///')
            ax.add_patch(outer)
            inner_cover = Rectangle(
                (cx - hx + 0.5, cy - hy + 0.5), sx - 2, sy - 2,
                linewidth=0, edgecolor='none', facecolor='white',
                alpha=0.25)
            ax.add_patch(inner_cover)
            rect_inner = Rectangle(
                (cx - hx + 0.5, cy - hy + 0.5), sx - 2, sy - 2,
                linewidth=1, edgecolor='cyan', facecolor='none',
                linestyle='-')
            ax.add_patch(rect_inner)


def plot_step_snapshot(jacobian_matrix, iteration, neg_count, min_val,
                       windows=None, label=None, jacobian_before=None):
    """Show a Jacobian heatmap snapshot during iteration.

    When *jacobian_before* is provided the output is a side-by-side
    1\u00d72 figure with the BEFORE state (left, with window rectangles) and
    the AFTER state (right, ``jacobian_matrix``).  Otherwise a single
    panel is shown.

    Parameters
    ----------
    jacobian_matrix : ndarray, shape ``(1, H, W)``
        Current (after) Jacobian determinant field.
    iteration : int
        Current outer iteration number.
    neg_count : int
        Number of non-positive Jdet pixels (after).
    min_val : float
        Minimum Jdet value (after).
    windows : list of (cy, cx, size, at_edge) or None
        Sub-windows optimised during this iteration.
    label : str or None
        Optional label prepended to the title.
    jacobian_before : ndarray or None
        If given, the BEFORE Jacobian field for side-by-side display.
    """
    if jacobian_before is not None:
        # ---- side-by-side before / after ----
        all_data = [jacobian_before[0], jacobian_matrix[0]]
        norm = _make_jdet_norm(all_data)

        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(12, 4),
                                          constrained_layout=True)

        # BEFORE panel
        im_b = ax_b.imshow(jacobian_before[0], cmap=CMAP, norm=norm,
                           interpolation=INTERP)
        _annotate_neg_contour(ax_b, jacobian_before[0])
        _annotate_jdet_values(ax_b, jacobian_before[0])
        if windows:
            _draw_windows(ax_b, windows)
        neg_b = int((jacobian_before <= 0).sum())
        min_b = float(jacobian_before.min())
        ax_b.set_title(f"BEFORE \u2014 Iter {iteration}  |  neg={neg_b}  "
                       f"min={min_b:+.4f}", fontsize=9)

        # AFTER panel
        im_a = ax_a.imshow(jacobian_matrix[0], cmap=CMAP, norm=norm,
                           interpolation=INTERP)
        _annotate_neg_contour(ax_a, jacobian_matrix[0])
        _annotate_jdet_values(ax_a, jacobian_matrix[0])
        ax_a.set_title(f"AFTER \u2014 Iter {iteration}  |  neg={neg_count}  "
                       f"min={min_val:+.4f}", fontsize=9)

        fig.colorbar(im_a, ax=[ax_b, ax_a], shrink=0.9)
        plt.show()
        return

    # ---- single panel (no before/after) ----
    fig, ax = plt.subplots(figsize=(5, 4))
    norm = _make_jdet_norm([jacobian_matrix[0]])
    im = ax.imshow(jacobian_matrix[0], cmap=CMAP, norm=norm, interpolation=INTERP)
    _annotate_neg_contour(ax, jacobian_matrix[0])
    _annotate_jdet_values(ax, jacobian_matrix[0])

    if windows:
        _draw_windows(ax, windows)

    prefix = f"{label} \u2014 " if label else ""
    ax.set_title(f"{prefix}Iter {iteration}  |  neg={neg_count}  min={min_val:+.4f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
