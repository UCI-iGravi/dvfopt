"""
Visualisation and result-saving utilities for deformation field correction.

Separated from ``dvfopt.py`` so the core optimiser has no matplotlib/pandas
dependency. All plotting functions operate on the same data conventions:

* Deformation fields: ``(3, 1, H, W)`` with channels ``[dz, dy, dx]``
* Corrected phi: ``(2, H, W)`` with channels ``[dy, dx]``
* Jacobian determinant arrays: ``(1, H, W)``

Usage::

    from modules.dvfviz import (
        plot_deformations,
        plot_grid_before_after,
        plot_checkerboard_before_after,
        plot_neg_jdet_neighborhoods,
        plot_initial_deformation,
        plot_jacobians_iteratively,
        plot_step_snapshot,
        run_lapl_and_correction,
    )
"""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import map_coordinates

from modules.dvfopt import iterative_with_jacobians2, jacobian_det2D
import modules.jacobian as jacobian
import modules.laplacian as laplacian


# ---------------------------------------------------------------------------
# Shared styling constants
# ---------------------------------------------------------------------------
CMAP = "RdBu_r"
INTERP = "nearest"
QUIVER_COLOR = "#333333"
NEG_CONTOUR_COLOR = "lime"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _make_jdet_norm(jac_arrays):
    """Build a ``TwoSlopeNorm`` centred on 0 that spans all supplied Jacobian arrays."""
    vmin = min(j.min() for j in jac_arrays)
    vmax = max(j.max() for j in jac_arrays)
    return mcolors.TwoSlopeNorm(
        vmin=min(vmin, -0.5),
        vcenter=0,
        vmax=max(vmax, 1),
    )


def _annotate_jdet_values(ax, jac_2d, max_dim=25):
    """Print Jacobian determinant values on each pixel for small grids.

    Skipped when either dimension exceeds *max_dim* so large plots stay clean.
    Font size scales with the grid size.
    """
    h, w = jac_2d.shape
    if max(h, w) > max_dim:
        return
    base_fontsize = min(6.5, max(3.5, 110 / max(h, w)))
    for row in range(h):
        for col in range(w):
            val = jac_2d[row, col]
            # Use white text on very dark (negative) cells, dark gray otherwise
            color = "white" if val < -0.2 else "#444444"
            weight = "bold" if val <= 0 else "normal"
            # Gradually increase opacity and size as value approaches 0:
            # val >= 1.0 → faint/small, val ~ 0 → full/larger
            t = float(np.clip(val, 0, 1))  # 1 = safe, 0 = threshold
            alpha = 1.0 - 0.5 * t * t
            fontsize = base_fontsize * (1.0 + 0.25 * (1.0 - t * t))
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize, color=color, alpha=alpha,
                    fontweight=weight)


def _annotate_neg_contour(ax, jac_2d, threshold=0.0):
    """Overlay a contour line at *threshold* to highlight negative-Jdet regions.

    The mask is padded with a ring of zeros and upsampled 2x with
    nearest-neighbour so the contour follows pixel edges (not diagonals)
    and never clips at image boundaries.
    """
    mask = (jac_2d <= threshold).astype(float)
    if not mask.any():
        return
    # Save axis limits so the padded contour grid doesn't expand them
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Pad with one pixel of zeros so boundary contours are fully enclosed
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    ph, pw = padded.shape  # h+2, w+2
    # Upsample 2x so contour traces pixel edges
    mask_up = np.repeat(np.repeat(padded, 2, axis=0), 2, axis=1)
    # Coordinate grids: pad offset is -1 in original coords, then each
    # original pixel spans 0.5 in the upsampled grid
    ys = np.linspace(-1.25, ph - 1.75, 2 * ph)
    xs = np.linspace(-1.25, pw - 1.75, 2 * pw)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, mask_up, levels=[0.5], colors=NEG_CONTOUR_COLOR,
               linewidths=2.0, linestyles="-")
    # Restore axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# Per-step snapshot (used by iterative_with_jacobians2 via plot_every)
# ---------------------------------------------------------------------------
def _draw_windows(ax, windows):
    """Draw sub-window rectangles on *ax*."""
    from matplotlib.patches import Rectangle
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
        ax_b.set_title(f"BEFORE — Iter {iteration}  |  neg={neg_b}  "
                       f"min={min_b:+.4f}", fontsize=9)

        # AFTER panel
        im_a = ax_a.imshow(jacobian_matrix[0], cmap=CMAP, norm=norm,
                           interpolation=INTERP)
        _annotate_neg_contour(ax_a, jacobian_matrix[0])
        _annotate_jdet_values(ax_a, jacobian_matrix[0])
        ax_a.set_title(f"AFTER — Iter {iteration}  |  neg={neg_count}  "
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

    prefix = f"{label} — " if label else ""
    ax.set_title(f"{prefix}Iter {iteration}  |  neg={neg_count}  min={min_val:+.4f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Initial-state preview (shown before correction runs)
# ---------------------------------------------------------------------------
def plot_initial_deformation(deformation_i, msample=None, fsample=None,
                             figsize=(14, 6), quiver_scale=1):
    """Quick preview of the initial deformation field before correction.

    Shows a 1×2 layout: Jacobian determinant heatmap (left) and
    displacement quiver (right). Displayed immediately so the user can
    inspect the layout without waiting for the optimiser.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
        Original deformation field.
    msample, fsample : ndarray or None
        Moving / fixed correspondences ``(N, 3)`` with ``[z, y, x]``.
    figsize : tuple
        Figure size.
    quiver_scale : float
        Quiver arrow scale factor.
    """
    jacobian_initial = jacobian_det2D(deformation_i[1:])
    H, W = deformation_i.shape[-2:]
    neg = int((jacobian_initial <= 0).sum())

    norm = _make_jdet_norm([jacobian_initial[0]])

    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            gridspec_kw={"wspace": 0.25})

    # Left: Jacobian heatmap
    im = axs[0].imshow(jacobian_initial[0], cmap=CMAP, norm=norm, interpolation=INTERP)
    _annotate_neg_contour(axs[0], jacobian_initial[0])
    _annotate_jdet_values(axs[0], jacobian_initial[0])
    axs[0].set_title(f"Initial Jdet  (neg={neg})", fontsize=11)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, shrink=0.9)

    # Right: Quiver
    x, y = np.meshgrid(range(W), range(H), indexing="xy")
    axs[1].set_title("Initial displacement", fontsize=11)
    axs[1].quiver(x, y, deformation_i[2, 0], -deformation_i[1, 0],
                  scale=quiver_scale, scale_units="xy", color=QUIVER_COLOR,
                  width=0.003)
    axs[1].invert_yaxis()
    axs[1].set_aspect("equal")

    if msample is not None and fsample is not None:
        axs[1].scatter(msample[:, 2], msample[:, 1], c="lime",
                       edgecolors="k", s=50, zorder=6, label="Moving",
                       marker="o", linewidths=1.2)
        axs[1].scatter(fsample[:, 2], fsample[:, 1], c="magenta",
                       edgecolors="k", s=50, zorder=6, label="Fixed",
                       marker="X", linewidths=1.2)
        axs[1].legend(fontsize=8, loc="upper right")
        for i in range(len(msample)):
            axs[1].annotate(
                "", xy=(fsample[i][2], fsample[i][1]),
                xytext=(msample[i][2], msample[i][1]),
                arrowprops=dict(facecolor="#ff4500", edgecolor="#ff4500",
                                shrink=0.05, headwidth=6, headlength=6,
                                width=1.8, alpha=0.85),
                zorder=5,
            )

    fig.suptitle("Initial deformation (preview)", fontsize=12, fontweight="bold")
    plt.show()


# ---------------------------------------------------------------------------
# Main comparison plot
# ---------------------------------------------------------------------------
def plot_deformations(
    msample, fsample, deformation_i, phi_corrected,
    figsize=(14, 12), save_path=None, title="", quiver_scale=1,
):
    """Plot initial vs corrected Jacobian determinants and deformation quiver fields.

    Layout (2 x 2):

    * Top-left:  Initial Jacobian determinant heatmap
    * Top-right: Corrected Jacobian determinant heatmap
    * Bottom-left:  Initial displacement quiver
    * Bottom-right: Corrected displacement quiver

    Parameters
    ----------
    msample, fsample : ndarray or None
        Moving / fixed correspondences ``(N, 3)`` with ``[z, y, x]``.
        Pass ``None`` to skip correspondence overlay.
    deformation_i : ndarray, shape ``(3, 1, H, W)``
        Original deformation field.
    phi_corrected : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
    figsize : tuple
        Figure size.
    save_path : str or None
        Directory to save ``plot_final.png``.
    title : str
        Figure suptitle.
    quiver_scale : float
        Quiver arrow scale factor.
    """
    jacobian_initial = jacobian_det2D(deformation_i[1:])
    jacobian_final = jacobian_det2D(phi_corrected)

    H, W = deformation_i.shape[-2:]
    init_neg = int((jacobian_initial <= 0).sum())
    final_neg = int((jacobian_final <= 0).sum())

    # Summary table
    rows = [
        ("initial",   np.min(deformation_i[2, 0]), np.max(deformation_i[2, 0]),
                      np.min(deformation_i[1, 0]), np.max(deformation_i[1, 0]),
                      np.min(jacobian_initial),     np.max(jacobian_initial),  init_neg),
        ("corrected", np.min(phi_corrected[1]),     np.max(phi_corrected[1]),
                      np.min(phi_corrected[0]),     np.max(phi_corrected[0]),
                      np.min(jacobian_final),       np.max(jacobian_final),    final_neg),
    ]
    header = f"{'':>10s}  {'x-disp min':>10s}  {'x-disp max':>10s}  {'y-disp min':>10s}  {'y-disp max':>10s}  {'Jdet min':>10s}  {'Jdet max':>10s}  {'neg Jdet':>8s}"
    print(header)
    print("-" * len(header))
    for label, *vals in rows:
        nums = "  ".join(f"{v:>10.4f}" for v in vals[:-1])
        print(f"{label:>10s}  {nums}  {vals[-1]:>8d}")

    norm = _make_jdet_norm([jacobian_initial[0], jacobian_final[0]])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                            gridspec_kw={"hspace": 0.30, "wspace": 0.25})

    # ---- Row 0: Jacobian heatmaps ----
    im0 = axs[0, 0].imshow(jacobian_initial[0], cmap=CMAP, norm=norm, interpolation=INTERP)
    im1 = axs[0, 1].imshow(jacobian_final[0], cmap=CMAP, norm=norm, interpolation=INTERP)

    _annotate_neg_contour(axs[0, 0], jacobian_initial[0])
    _annotate_neg_contour(axs[0, 1], jacobian_final[0])

    _annotate_jdet_values(axs[0, 0], jacobian_initial[0])
    _annotate_jdet_values(axs[0, 1], jacobian_final[0])

    axs[0, 0].set_title(f"Initial Jdet  (neg={init_neg})", fontsize=11)
    axs[0, 1].set_title(f"Corrected Jdet  (neg={final_neg})", fontsize=11)

    # Shared colorbar for Jacobian row
    cbar = fig.colorbar(im1, ax=axs[0, :].tolist(), fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label("Jacobian det", fontsize=9)

    # ---- Row 1: Quiver plots ----
    x, y = np.meshgrid(range(W), range(H), indexing="xy")

    axs[1, 0].set_title("Initial displacement", fontsize=11)
    axs[1, 0].quiver(x, y, deformation_i[2, 0], -deformation_i[1, 0],
                      scale=quiver_scale, scale_units="xy", color=QUIVER_COLOR,
                      width=0.003)

    axs[1, 1].set_title("Corrected displacement", fontsize=11)
    axs[1, 1].quiver(x, y, phi_corrected[1], -phi_corrected[0],
                      scale=quiver_scale, scale_units="xy", color=QUIVER_COLOR,
                      width=0.003)

    for i in range(2):
        axs[1, i].invert_yaxis()
        axs[1, i].set_aspect("equal")

    # Overlay correspondences if provided
    if msample is not None and fsample is not None:
        for ax_idx in [0, 1]:
            axs[1, ax_idx].scatter(msample[:, 2], msample[:, 1], c="lime",
                                   edgecolors="k", s=50, zorder=6, label="Moving",
                                   marker="o", linewidths=1.2)
            axs[1, ax_idx].scatter(fsample[:, 2], fsample[:, 1], c="magenta",
                                   edgecolors="k", s=50, zorder=6, label="Fixed",
                                   marker="X", linewidths=1.2)
        axs[1, 0].legend(fontsize=8, loc="upper right")

        # Arrows from moving → fixed on initial quiver
        for i in range(len(msample)):
            axs[1, 0].annotate(
                "", xy=(fsample[i][2], fsample[i][1]),
                xytext=(msample[i][2], msample[i][1]),
                arrowprops=dict(facecolor="#ff4500", edgecolor="#ff4500",
                                shrink=0.05, headwidth=6, headlength=6,
                                width=1.8, alpha=0.85),
                zorder=5,
            )

        # Arrows from moving point → corrected destination on corrected quiver
        for i in range(len(msample)):
            my, mx = int(round(msample[i][1])), int(round(msample[i][2]))
            my = np.clip(my, 0, H - 1)
            mx = np.clip(mx, 0, W - 1)
            dest_x = msample[i][2] + phi_corrected[1, my, mx]
            dest_y = msample[i][1] + phi_corrected[0, my, mx]
            axs[1, 1].annotate(
                "", xy=(dest_x, dest_y),
                xytext=(msample[i][2], msample[i][1]),
                arrowprops=dict(facecolor="#1e90ff", edgecolor="#1e90ff",
                                shrink=0.05, headwidth=6, headlength=6,
                                width=1.8, alpha=0.85),
                zorder=5,
            )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + "/plot_final.png", bbox_inches="tight", dpi=150)

    plt.show()


# ---------------------------------------------------------------------------
# Jacobian progression grid
# ---------------------------------------------------------------------------
def plot_jacobians_iteratively(jacobians, msample=None, fsample=None, methodName="SLSQP"):
    """Plot a sequence of Jacobian determinant maps in a grid.

    Parameters
    ----------
    jacobians : list of ndarray
        Each entry is a ``(1, H, W)`` Jacobian determinant array.
    msample, fsample : ndarray or None
        Correspondences to overlay on the first panel.
    methodName : str
        Label for the suptitle.
    """
    num = len(jacobians)
    ncols = min(3, num)
    nrows = (num + ncols - 1) // ncols

    all_2d = [j[0] for j in jacobians]
    norm = _make_jdet_norm(all_2d)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(4.5 * ncols, 4 * nrows),
                            squeeze=False)
    axs_flat = axs.flatten()

    for i, jac in enumerate(jacobians):
        ax = axs_flat[i]
        im = ax.imshow(jac[0], cmap=CMAP, norm=norm, interpolation=INTERP)
        _annotate_neg_contour(ax, jac[0])
        _annotate_jdet_values(ax, jac[0])
        neg = int(np.sum(jac <= 0))
        if i == 0:
            ax.set_title(f"Initial  (neg={neg})", fontsize=10)
            if msample is not None and fsample is not None:
                ax.scatter(msample[:, 2], msample[:, 1], c="lime",
                           edgecolors="k", s=18, zorder=5, label="Moving")
                ax.scatter(fsample[:, 2], fsample[:, 1], c="magenta",
                           edgecolors="k", s=18, zorder=5, label="Fixed")
                ax.legend(fontsize=7, loc="upper right")
                for k in range(len(msample)):
                    ax.annotate(
                        "", xy=(fsample[k][2], fsample[k][1]),
                        xytext=(msample[k][2], msample[k][1]),
                        arrowprops=dict(facecolor="black", shrink=0.05,
                                        headwidth=3, headlength=4, width=0.8),
                    )
        else:
            ax.set_title(f"Step {i}  (neg={neg})", fontsize=10)

    # Hide empty axes
    for j in range(num, len(axs_flat)):
        axs_flat[j].axis("off")

    fig.colorbar(im, ax=axs_flat[:num].tolist(), orientation="vertical",
                 fraction=0.03, pad=0.04, shrink=0.8)
    fig.suptitle(f"Jacobian progression — {methodName}", fontsize=13, fontweight="bold")
    plt.show()


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------
def run_lapl_and_correction(fixed_sample, msample, fsample, methodName="SLSQP",
                            save_path=None, title="", **kwargs):
    """End-to-end: Laplacian interpolation -> iterative SLSQP correction -> plot.

    Extra ``**kwargs`` are forwarded to :func:`dvfopt.iterative_with_jacobians2`.
    """
    deformation_i, A, Zd, Yd, Xd = laplacian.sliceToSlice3DLaplacian(fixed_sample, msample, fsample)
    print(f"[Laplacian] deformation shape: {deformation_i.shape}")
    plot_initial_deformation(deformation_i, msample, fsample)
    phi_corrected = iterative_with_jacobians2(deformation_i, methodName, save_path=save_path, **kwargs)
    plot_deformations(msample, fsample, deformation_i, phi_corrected,
                      figsize=(14, 12), save_path=save_path, title=title)
    plot_grid_before_after(deformation_i, phi_corrected, title=title)
    return deformation_i, phi_corrected


# ---------------------------------------------------------------------------
# Single-field deformation preview (Jacobian heatmap + quiver)
# ---------------------------------------------------------------------------
def plot_deformation_field(deformation, msample=None, fsample=None,
                           title="", figsize=(20, 10), show_values=False,
                           show_points=True, save_path=None, quiver_scale=None):
    """Plot a single deformation field: Jacobian heatmap + displacement quiver.

    Intended for previewing test-case data before running corrections.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, 1, H, W)``
    msample, fsample : ndarray ``(N, 3)`` or None
    title : str
    save_path : str or None
        If given, saves ``.png`` next to this path.
    quiver_scale : float or None
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    J = np.squeeze(jacobian.sitk_jacobian_determinant(deformation))
    neg = int(np.sum(J <= 0))
    norm = mcolors.TwoSlopeNorm(
        vmin=min(J.min(), -3), vcenter=0, vmax=max(J.max(), 3)
    )

    # Left: Jacobian heatmap with arrows
    if msample is not None and fsample is not None:
        for i in range(len(msample)):
            axs[0].annotate(
                "", xy=(fsample[i][2], fsample[i][1]),
                xytext=(msample[i][2], msample[i][1]),
                arrowprops=dict(facecolor="black", shrink=0.045,
                                headwidth=8, headlength=10, width=3),
            )

    im = axs[0].imshow(J, cmap=CMAP, norm=norm)
    axs[0].set_title(f"Jacobian determinant ({neg} negative)")

    if show_values:
        _annotate_jdet_values(axs[0], J)
        _annotate_jdet_values(axs[1], J)

    # Right: quiver plot
    x, y = np.meshgrid(range(deformation.shape[3]), range(deformation.shape[2]), indexing="xy")
    axs[1].set_title("Deformation vector field")
    axs[1].imshow(J, cmap=CMAP, norm=norm)
    if quiver_scale is None:
        axs[1].quiver(x, y, deformation[2, 0], -deformation[1, 0])
    else:
        axs[1].quiver(x, y, deformation[2, 0], -deformation[1, 0],
                       scale=quiver_scale, scale_units="xy")

    if show_points and msample is not None and fsample is not None:
        for ax in axs:
            ax.scatter(msample[:, 2], msample[:, 1], c="g", label="Moving")
            ax.scatter(fsample[:, 2], fsample[:, 1], c="violet", label="Fixed")
            ax.legend()

    fig.suptitle(title, fontsize=16)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Jacobian determinant")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace(".npy", ".png"), bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Deformed-grid visualisations
# ---------------------------------------------------------------------------
def plot_2d_deformation_grid(deformation, spacing=1, xlim=None, ylim=None,
                             title="2D Deformation Grid", highlight_point=None):
    """Visualise a ``(3, 1, Y, X)`` deformation as a deformed grid.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, 1, Y, X)``
    spacing : int
        Grid line spacing (pixels).
    highlight_point : tuple ``(y, x)`` or None
        If given, draws deformed neighbours as a closed polygon.
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    y_coords, x_coords = np.meshgrid(
        np.arange(0, H, spacing), np.arange(0, W, spacing), indexing="ij"
    )
    new_y = y_coords + dy[y_coords, x_coords]
    new_x = x_coords + dx[y_coords, x_coords]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(y_coords.shape[0]):
        ax.plot(new_x[i, :], new_y[i, :], "r-")
    for j in range(x_coords.shape[1]):
        ax.plot(new_x[:, j], new_y[:, j], "r-")

    if highlight_point:
        cy, cx = highlight_point
        offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        pts = []
        for dy_off, dx_off in offsets:
            ny, nx = cy + dy_off, cx + dx_off
            if 0 <= ny < H and 0 <= nx < W:
                pts.append((nx + dx[ny, nx], ny + dy[ny, nx]))
                ax.scatter(pts[-1][0], pts[-1][1], color="green", zorder=5)
        ax.scatter(cx + dx[cy, cx], cy + dy[cy, cx], color="blue", zorder=5)
        if len(pts) == 4:
            xs, ys = zip(*pts)
            ax.plot(xs + (xs[0],), ys + (ys[0],), color="black", linewidth=1.5)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_deformed_quads(deformation, center_y, center_x, spacing=1,
                        patch_size=20, title="Deformed Quadrilateral Mesh"):
    """Plot a zoomed-in deformed quadrilateral mesh.

    Parameters
    ----------
    deformation : ndarray ``(3, 1, Y, X)``
    center_y, center_x : int
        Centre of the zoom window.
    spacing : int
    patch_size : int
        Width/height of the zoom window in pixels.
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    y0 = max(center_y - patch_size // 2, 0)
    y1 = min(center_y + patch_size // 2, H - spacing - 1)
    x0 = max(center_x - patch_size // 2, 0)
    x1 = min(center_x + patch_size // 2, W - spacing - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(y0, y1, spacing):
        for j in range(x0, x1, spacing):
            corners = [(j, i), (j + spacing, i),
                       (j + spacing, i + spacing), (j, i + spacing)]
            deformed = [(x + dx[y, x], y + dy[y, x]) for x, y in corners]
            poly = Polygon(deformed, closed=True, edgecolor="red",
                           facecolor="lightgray", linewidth=0.8)
            ax.add_patch(poly)

    ax.set_xlim(x0, x1 + spacing)
    ax.set_ylim(y1 + spacing, y0)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_deformed_quads_colored(deformation, center_y, center_x, spacing=1,
                                patch_size=20, cmap="bwr"):
    """Plot a zoomed quad mesh coloured by Jacobian determinant.

    Negative-Jdet quads are outlined in yellow.
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    J = np.squeeze(jacobian.sitk_jacobian_determinant(deformation))
    norm = mcolors.TwoSlopeNorm(
        vmin=min(J.min(), -3), vcenter=0, vmax=max(J.max(), 3)
    )
    colormap = plt.get_cmap(cmap)

    y0 = max(center_y - patch_size // 2, 0)
    y1 = min(center_y + patch_size // 2, H - spacing - 1)
    x0 = max(center_x - patch_size // 2, 0)
    x1 = min(center_x + patch_size // 2, W - spacing - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(y0, y1, spacing):
        for j in range(x0, x1, spacing):
            corners = [(j, i), (j + spacing, i),
                       (j + spacing, i + spacing), (j, i + spacing)]
            deformed = [(x + dx[y, x], y + dy[y, x]) for x, y in corners]

            ec = "yellow" if J[i, j] < 0 else "black"
            lw = 2.0 if J[i, j] < 0 else 0.5
            poly = Polygon(deformed, closed=True, edgecolor=ec,
                           facecolor=colormap(norm(J[i, j])), linewidth=lw)
            ax.add_patch(poly)

    ax.set_xlim(x0, x1 + spacing)
    ax.set_ylim(y1 + spacing, y0)
    ax.set_title("Deformed Mesh (coloured by Jacobian)")
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Jacobian Determinant")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Before / after deformation grid comparison
# ---------------------------------------------------------------------------
def _invert_displacement(dy, dx, iterations=50):
    """Compute inverse displacement field via fixed-point iteration.

    Given pull-back displacement *d* where ``T(x) = x + d(x)`` maps
    fixed → moving, compute ``d_inv`` where ``T^{-1}(y) = y + d_inv(y)``
    maps moving → fixed.

    Uses the iteration ``d_inv^{k+1}(y) = -d(y + d_inv^k(y))``,
    starting from the naive ``d_inv^0 = -d``.
    """
    H, W = dy.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    inv_dy = -dy.copy()
    inv_dx = -dx.copy()

    for _ in range(iterations):
        sample_y = np.clip(yy + inv_dy, 0, H - 1)
        sample_x = np.clip(xx + inv_dx, 0, W - 1)

        sampled_dy = map_coordinates(dy, [sample_y, sample_x],
                                     order=1, mode='nearest')
        sampled_dx = map_coordinates(dx, [sample_y, sample_x],
                                     order=1, mode='nearest')

        inv_dy = -sampled_dy
        inv_dx = -sampled_dx

    return inv_dy, inv_dx


def plot_grid_before_after(deformation_i, phi_corrected, figsize=(14, 6),
                           title="", spacing=1, linewidth=0.5,
                           inverse=False):
    """Side-by-side deformation grids coloured by Jacobian determinant.

    The displacement field uses the **pull-back** convention: for each
    pixel in the fixed image, the displacement vector points to where
    in the moving image that pixel samples from.  The deformed grid
    therefore shows how the regular fixed-image grid maps into
    moving-image space (the sampling pattern).

    When ``inverse=True`` the inverse displacement is computed via
    fixed-point iteration (see ``_invert_displacement``), giving the
    true push-forward view: a regular grid in moving-image space
    mapped back into fixed-image space.

    Rendering: light Jdet-coloured quad fills + black wireframe grid
    lines drawn on top.  Negative-Jdet cells are filled opaque with a
    yellow outline.  Folding is visible as grid-line crossings.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
        Original deformation field with channels ``[dz, dy, dx]``.
    phi_corrected : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.
    figsize : tuple
        Figure size.
    title : str
        Optional suptitle.
    spacing : int
        Grid cell size in pixels (1 = every pixel).
    linewidth : float
        Grid line width.
    inverse : bool
        If False (default), show fixed→moving (pull-back).
        If True, show moving→fixed via proper fixed-point inversion.
    """
    _, _, H, W = deformation_i.shape

    phi_init = np.stack([deformation_i[1, 0], deformation_i[2, 0]])  # [dy, dx]

    if inverse:
        inv_init_dy, inv_init_dx = _invert_displacement(phi_init[0], phi_init[1])
        phi_init_view = np.stack([inv_init_dy, inv_init_dx])
        inv_corr_dy, inv_corr_dx = _invert_displacement(phi_corrected[0], phi_corrected[1])
        phi_corr_view = np.stack([inv_corr_dy, inv_corr_dx])

        # Inverse Jacobian: J_{T^{-1}}(y) = 1 / J_T(T^{-1}(y))
        jac_fwd_init = np.squeeze(jacobian_det2D(phi_init))
        jac_fwd_corr = np.squeeze(jacobian_det2D(phi_corrected))

        yy, xx = np.mgrid[0:H, 0:W].astype(float)

        preimg_y = np.clip(yy + inv_init_dy, 0, H - 1)
        preimg_x = np.clip(xx + inv_init_dx, 0, W - 1)
        jac_fwd_at_preimg = map_coordinates(
            jac_fwd_init, [preimg_y, preimg_x], order=1, mode='nearest')
        with np.errstate(divide='ignore', invalid='ignore'):
            jac_init = np.where(
                np.abs(jac_fwd_at_preimg) > 1e-10,
                1.0 / jac_fwd_at_preimg, 0.0)

        preimg_y = np.clip(yy + inv_corr_dy, 0, H - 1)
        preimg_x = np.clip(xx + inv_corr_dx, 0, W - 1)
        jac_fwd_at_preimg = map_coordinates(
            jac_fwd_corr, [preimg_y, preimg_x], order=1, mode='nearest')
        with np.errstate(divide='ignore', invalid='ignore'):
            jac_corr = np.where(
                np.abs(jac_fwd_at_preimg) > 1e-10,
                1.0 / jac_fwd_at_preimg, 0.0)
    else:
        phi_init_view = phi_init
        phi_corr_view = phi_corrected
        jac_init = np.squeeze(jacobian_det2D(phi_init))
        jac_corr = np.squeeze(jacobian_det2D(phi_corrected))

    # Shared colour normalisation across both panels
    J_all = np.concatenate([jac_init.ravel(), jac_corr.ravel()])
    vmin = min(float(J_all.min()), -0.5)
    vmax = max(float(J_all.max()), 1.5)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("bwr")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    init_neg = int((jac_init <= 0).sum())
    corr_neg = int((jac_corr <= 0).sum())

    arrow = "moving\u2192fixed" if inverse else "fixed\u2192moving"

    for ax, phi, jac, label in [
        (axes[0], phi_init_view, jac_init, f"Initial — {arrow} (neg Jdet = {init_neg})"),
        (axes[1], phi_corr_view, jac_corr, f"Corrected — {arrow} (neg Jdet = {corr_neg})"),
    ]:
        dy = phi[0]
        dx = phi[1]

        # Deformed vertex positions for the full grid
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        def_x = xx + dx
        def_y = yy + dy

        # Layer 1: light Jdet-coloured cell fills
        for i in range(0, H - spacing, spacing):
            for j in range(0, W - spacing, spacing):
                ci = [i, i, i + spacing, i + spacing]
                cj = [j, j + spacing, j + spacing, j]
                corners = [(def_x[np.clip(r, 0, H-1), np.clip(c, 0, W-1)],
                            def_y[np.clip(r, 0, H-1), np.clip(c, 0, W-1)])
                           for r, c in zip(ci, cj)]
                jval = jac[i, j]
                if jval <= 0:
                    # Negative cell: opaque fill + yellow outline
                    fc = cmap(norm(jval))
                    poly = Polygon(corners, closed=True,
                                   facecolor=fc, edgecolor="yellow",
                                   linewidth=max(linewidth * 3, 1.5),
                                   zorder=2)
                else:
                    # Positive cell: light fill, no outline (grid lines go on top)
                    fc = cmap(norm(jval))
                    poly = Polygon(corners, closed=True,
                                   facecolor=(*fc[:3], 0.25),
                                   edgecolor="none", zorder=0)
                ax.add_patch(poly)

        # Layer 2: wireframe grid lines at sampled positions (drawn on top)
        # Row indices and column indices at spacing intervals
        row_indices = list(range(0, H, spacing))
        if (H - 1) not in row_indices:
            row_indices.append(H - 1)
        col_indices = list(range(0, W, spacing))
        if (W - 1) not in col_indices:
            col_indices.append(W - 1)

        for i in row_indices:
            ax.plot(def_x[i, col_indices], def_y[i, col_indices], 'k-',
                    linewidth=linewidth, zorder=1)
        for j in col_indices:
            ax.plot(def_x[row_indices, j], def_y[row_indices, j], 'k-',
                    linewidth=linewidth, zorder=1)

        # Axis limits from deformed extent
        pad = max(W, H) * 0.03
        ax.set_xlim(def_x.min() - pad, def_x.max() + pad)
        ax.set_ylim(def_y.max() + pad, def_y.min() - pad)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.87)
    fig.colorbar(sm, ax=axes.tolist(), label="Jacobian determinant",
                 fraction=0.03, pad=0.06, shrink=0.85)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.show()


# ---------------------------------------------------------------------------
# Folding close-up: reference grid vs deformed grid
# ---------------------------------------------------------------------------
def _quad_signed_areas(def_x, def_y):
    """Shoelace-formula signed area for each quad cell.

    Returns shape ``(rows-1, cols-1)``.  Positive = non-folded (consistent
    winding with the TL→TR→BR→BL vertex order used by ``_draw_grid_patch``).
    """
    x0, y0 = def_x[:-1, :-1], def_y[:-1, :-1]
    x1, y1 = def_x[:-1, 1:],  def_y[:-1, 1:]
    x2, y2 = def_x[1:, 1:],   def_y[1:, 1:]
    x3, y3 = def_x[1:, :-1],  def_y[1:, :-1]
    return 0.5 * ((x0*y1 - x1*y0) + (x1*y2 - x2*y1)
                + (x2*y3 - x3*y2) + (x3*y0 - x0*y3))


def _find_neg_jdet_centers(jac, H, W, margin, max_panels, dedup_dist):
    """Find deduplicated neg-Jdet pixel centres, worst-first."""
    neg_mask = jac <= 0
    if not neg_mask.any():
        return []
    neg_ys, neg_xs = np.where(neg_mask)
    order = np.argsort(jac[neg_ys, neg_xs])
    neg_ys, neg_xs = neg_ys[order], neg_xs[order]
    shown = []
    for cy, cx in zip(neg_ys, neg_xs):
        if len(shown) >= max_panels:
            break
        if cy < margin or cy >= H - margin or cx < margin or cx >= W - margin:
            continue
        if any(abs(cy - sy) <= dedup_dist and abs(cx - sx) <= dedup_dist
               for sy, sx in shown):
            continue
        shown.append((cy, cx))
    return shown


def _draw_grid_patch(ax, def_x, def_y, jac_patch, norm, cmap,
                     ref_x=None, ref_y=None, label_vertices=False,
                     y0=0, x0=0, color_by_direction=False):
    """Draw a deformed grid patch on *ax*.

    Parameters
    ----------
    def_x, def_y : ndarray, shape ``(rows, cols)``
        Deformed vertex positions.
    jac_patch : ndarray, shape ``(rows-1, cols-1)`` or None
        Per-cell value (Jdet or signed area).  If None, cells are not filled.
    ref_x, ref_y : ndarray or None
        If given, draw an undeformed reference grid (thin gray dashes).
    label_vertices : bool
        If True, label each vertex with its ``(row, col)`` grid index.
    color_by_direction : bool
        If True, draw row-lines in blue and column-lines in red so that
        line crossings (= folding) are immediately visible.  Only
        negative-area cells are filled (yellow highlight).
    """
    rows, cols = def_x.shape

    ROW_COLOR = '#2166ac'   # steel blue — row (horizontal) lines
    COL_COLOR = '#b2182b'   # crimson    — column (vertical) lines

    # --- reference grid (undeformed) ---
    if ref_x is not None:
        for i in range(rows):
            ax.plot(ref_x[i, :], ref_y[i, :], color='#cccccc',
                    linestyle='--', linewidth=0.7, zorder=0)
        for j in range(cols):
            ax.plot(ref_x[:, j], ref_y[:, j], color='#cccccc',
                    linestyle='--', linewidth=0.7, zorder=0)

    # --- filled quads ---
    if jac_patch is not None:
        nr, nc = jac_patch.shape
        for i in range(nr):
            for j in range(nc):
                corners = [
                    (def_x[i, j],     def_y[i, j]),
                    (def_x[i, j+1],   def_y[i, j+1]),
                    (def_x[i+1, j+1], def_y[i+1, j+1]),
                    (def_x[i+1, j],   def_y[i+1, j]),
                ]
                jval = jac_patch[i, j]
                is_neg = jval <= 0
                if color_by_direction:
                    # Only highlight folded cells
                    if is_neg:
                        poly = Polygon(
                            corners, closed=True,
                            facecolor=(1, 0.85, 0, 0.30),
                            edgecolor='orange', linewidth=1.5,
                            zorder=1)
                        ax.add_patch(poly)
                else:
                    fc = cmap(norm(jval))
                    poly = Polygon(
                        corners, closed=True,
                        facecolor=(*fc[:3], 0.55 if is_neg else 0.18),
                        edgecolor='yellow' if is_neg else 'none',
                        linewidth=2.0 if is_neg else 0,
                        zorder=1)
                    ax.add_patch(poly)

    # --- grid lines ---
    if color_by_direction:
        lw = 2.2
        for i in range(rows):
            ax.plot(def_x[i, :], def_y[i, :], color=ROW_COLOR,
                    linewidth=lw, zorder=2, solid_capstyle='round')
        for j in range(cols):
            ax.plot(def_x[:, j], def_y[:, j], color=COL_COLOR,
                    linewidth=lw, zorder=2, solid_capstyle='round')
    else:
        for i in range(rows):
            ax.plot(def_x[i, :], def_y[i, :], 'k-', linewidth=1.5, zorder=2)
        for j in range(cols):
            ax.plot(def_x[:, j], def_y[:, j], 'k-', linewidth=1.5, zorder=2)

    # --- vertex dots + optional labels ---
    ax.plot(def_x.ravel(), def_y.ravel(), 'ko', markersize=4, zorder=3)

    if label_vertices:
        for i in range(rows):
            for j in range(cols):
                ax.annotate(
                    f"{y0+i},{x0+j}", (def_x[i, j], def_y[i, j]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color='#555555', zorder=4)


def plot_checkerboard_before_after(deformation_i, phi_corrected, figsize=None,
                                   title="", max_panels=4, half_win=1):
    """Three-column folding close-up: Reference | Initial | Corrected.

    For each negative-Jdet region, extracts a ``(2*half_win+1)``-vertex
    neighbourhood and draws:

    * **Reference** — the regular (undeformed) grid.
    * **Initial** — deformed grid with Jdet-coloured fills.  Negative cells
      are blue with yellow outline.  The reference grid is drawn underneath
      in light gray dashes so the viewer can see how vertices have moved.
      Vertices are labelled with their ``(row,col)`` grid index so
      crossings are obvious (vertex order reversal = folding).
    * **Corrected** — same layout after correction.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    phi_corrected : ndarray, shape ``(2, H, W)``
    half_win : int
        Half-window in vertices around the centre pixel.  ``2`` →
        5×5 vertices (4×4 cells).
    """
    _, _, H, W = deformation_i.shape
    phi_init = np.stack([deformation_i[1, 0], deformation_i[2, 0]])
    jac_init = np.squeeze(jacobian_det2D(phi_init))

    shown = _find_neg_jdet_centers(jac_init, H, W, margin=half_win,
                                   max_panels=max_panels,
                                   dedup_dist=2 * half_win)
    if not shown:
        return

    n = len(shown)
    if figsize is None:
        figsize = (14, 4.2 * n)
    fig, axes = plt.subplots(n, 3, figsize=figsize, squeeze=False)

    vmin = min(float(jac_init.min()), -0.5)
    vmax = max(float(jac_init.max()), 1.5)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap('bwr')

    for idx, (cy, cx) in enumerate(shown):
        y0 = cy - half_win
        y1 = cy + half_win + 1
        x0 = cx - half_win
        x1 = cx + half_win + 1

        # Reference (undeformed) vertex positions
        ref_yy, ref_xx = np.mgrid[y0:y1, x0:x1].astype(float)

        def _deformed(phi):
            dy = phi[0][y0:y1, x0:x1]
            dx = phi[1][y0:y1, x0:x1]
            return ref_xx + dx, ref_yy + dy

        ix, iy = _deformed(phi_init)
        cx_, cy_ = _deformed(phi_corrected)

        # Geometric quad signed areas (matches what the viewer sees)
        area_init = _quad_signed_areas(ix, iy)
        area_corr = _quad_signed_areas(cx_, cy_)

        # Shared axis limits across all 3 panels
        all_x = np.concatenate([ref_xx.ravel(), ix.ravel(), cx_.ravel()])
        all_y = np.concatenate([ref_yy.ravel(), iy.ravel(), cy_.ravel()])
        pad = 0.8
        xlim = (all_x.min() - pad, all_x.max() + pad)
        ylim = (all_y.max() + pad, all_y.min() - pad)

        # --- Column 0: Reference grid ---
        ax0 = axes[idx, 0]
        _draw_grid_patch(ax0, ref_xx, ref_yy, jac_patch=None,
                         norm=norm, cmap=cmap,
                         color_by_direction=True)
        ax0.set_xlim(xlim); ax0.set_ylim(ylim)
        ax0.set_aspect('equal')
        n_neg = int((area_init <= 0).sum())
        ax0.set_title(f"Reference grid  ({y0}:{y1-1}, {x0}:{x1-1})",
                      fontsize=9)

        # --- Column 1: Initial (folded) ---
        ax1 = axes[idx, 1]
        _draw_grid_patch(ax1, ix, iy, jac_patch=area_init,
                         norm=norm, cmap=cmap,
                         ref_x=ref_xx, ref_y=ref_yy,
                         color_by_direction=True)
        ax1.set_xlim(xlim); ax1.set_ylim(ylim)
        ax1.set_aspect('equal')
        ax1.set_title(
            f"Initial — {n_neg} folded  (min area={area_init.min():.2f})",
            fontsize=9)

        # --- Column 2: Corrected ---
        ax2 = axes[idx, 2]
        _draw_grid_patch(ax2, cx_, cy_, jac_patch=area_corr,
                         norm=norm, cmap=cmap,
                         ref_x=ref_xx, ref_y=ref_yy,
                         color_by_direction=True)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim)
        ax2.set_aspect('equal')
        n_neg_c = int((area_corr <= 0).sum())
        ax2.set_title(
            f"Corrected — {n_neg_c} folded  (min area={area_corr.min():.2f})",
            fontsize=9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='#cccccc', linestyle='--', linewidth=1,
               label='Reference (undeformed)'),
        Line2D([0], [0], color='#2166ac', linestyle='-', linewidth=2.2,
               label='Row lines (horizontal)'),
        Line2D([0], [0], color='#b2182b', linestyle='-', linewidth=2.2,
               label='Column lines (vertical)'),
        Polygon([(0,0)], closed=True,
                facecolor=(1, 0.85, 0, 0.30), edgecolor='orange',
                linewidth=1.5, label='Folded cell'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4,
               fontsize=9, frameon=True)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


# ---------------------------------------------------------------------------
# Correction displacement arrows: where did each vertex move?
# ---------------------------------------------------------------------------
def plot_neg_jdet_neighborhoods(deformation_i, phi_corrected, figsize=None,
                                title="", max_panels=4, half_win=2):
    """Per-region displacement arrows showing the correction applied.

    For each negative-Jdet neighbourhood, draws two panels:

    * **Left** — the initial deformed grid (thick black, Jdet-coloured fills)
      with **green arrows** from each initial vertex position to its corrected
      position.  Arrow length = magnitude of correction applied.
    * **Right** — the corrected deformed grid (same style) with reference
      grid and vertex labels.

    This makes it easy to see *which vertices were moved and by how much*
    to eliminate folding.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    phi_corrected : ndarray, shape ``(2, H, W)``
    half_win : int
        Half-window in vertices.  ``2`` → 5×5 vertices (4×4 cells).
    """
    _, _, H, W = deformation_i.shape
    phi_init = np.stack([deformation_i[1, 0], deformation_i[2, 0]])
    jac_init = np.squeeze(jacobian_det2D(phi_init))

    shown = _find_neg_jdet_centers(jac_init, H, W, margin=half_win,
                                   max_panels=max_panels,
                                   dedup_dist=2 * half_win)
    if not shown:
        return

    n = len(shown)
    if figsize is None:
        figsize = (12, 5 * n)
    fig, axes = plt.subplots(n, 2, figsize=figsize, squeeze=False)

    vmin = min(float(jac_init.min()), -0.5)
    vmax = max(float(jac_init.max()), 1.5)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap('bwr')

    for idx, (cy, cx) in enumerate(shown):
        y0 = cy - half_win
        y1 = cy + half_win + 1
        x0 = cx - half_win
        x1 = cx + half_win + 1

        ref_yy, ref_xx = np.mgrid[y0:y1, x0:x1].astype(float)

        def _deformed(phi):
            dy = phi[0][y0:y1, x0:x1]
            dx = phi[1][y0:y1, x0:x1]
            return ref_xx + dx, ref_yy + dy

        ix, iy = _deformed(phi_init)
        cx_, cy_ = _deformed(phi_corrected)

        # Geometric quad signed areas (matches what the viewer sees)
        area_init = _quad_signed_areas(ix, iy)
        area_corr = _quad_signed_areas(cx_, cy_)

        # Shared limits
        all_x = np.concatenate([ref_xx.ravel(), ix.ravel(), cx_.ravel()])
        all_y = np.concatenate([ref_yy.ravel(), iy.ravel(), cy_.ravel()])
        pad = 0.8
        xlim = (all_x.min() - pad, all_x.max() + pad)
        ylim = (all_y.max() + pad, all_y.min() - pad)

        # --- Left: initial grid + correction arrows ---
        ax0 = axes[idx, 0]
        _draw_grid_patch(ax0, ix, iy, jac_patch=area_init,
                         norm=norm, cmap=cmap,
                         ref_x=ref_xx, ref_y=ref_yy,
                         color_by_direction=True)

        # Green arrows: initial → corrected vertex positions
        dx_arrow = cx_ - ix
        dy_arrow = cy_ - iy
        mag = np.sqrt(dx_arrow**2 + dy_arrow**2)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[1]):
                if mag[i, j] > 0.01:
                    ax0.annotate(
                        "", xy=(cx_[i, j], cy_[i, j]),
                        xytext=(ix[i, j], iy[i, j]),
                        arrowprops=dict(arrowstyle='->', color='green',
                                        lw=2.0, shrinkA=2, shrinkB=2),
                        zorder=5)

        ax0.set_xlim(xlim); ax0.set_ylim(ylim)
        ax0.set_aspect('equal')
        n_neg = int((area_init <= 0).sum())
        ax0.set_title(
            f"Initial + correction arrows  ({n_neg} folded)",
            fontsize=9)

        # --- Right: corrected grid ---
        ax1 = axes[idx, 1]
        _draw_grid_patch(ax1, cx_, cy_, jac_patch=area_corr,
                         norm=norm, cmap=cmap,
                         ref_x=ref_xx, ref_y=ref_yy,
                         color_by_direction=True)
        ax1.set_xlim(xlim); ax1.set_ylim(ylim)
        ax1.set_aspect('equal')
        n_neg_c = int((area_corr <= 0).sum())
        ax1.set_title(
            f"Corrected  ({n_neg_c} folded, min area={area_corr.min():.2f})",
            fontsize=9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='#cccccc', linestyle='--', linewidth=1,
               label='Reference (undeformed)'),
        Line2D([0], [0], color='#2166ac', linestyle='-', linewidth=2.2,
               label='Row lines (horizontal)'),
        Line2D([0], [0], color='#b2182b', linestyle='-', linewidth=2.2,
               label='Column lines (vertical)'),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2,
               marker='>', markersize=5,
               label='Correction applied'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4,
               fontsize=9, frameon=True)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
