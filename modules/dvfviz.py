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
        plot_initial_deformation,
        plot_jacobians_iteratively,
        plot_step_snapshot,
        run_lapl_and_correction,
    )
"""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from modules.dvfopt import iterative_with_jacobians2, jacobian_det2D
import modules.laplacian as laplacian


# ---------------------------------------------------------------------------
# Shared styling constants
# ---------------------------------------------------------------------------
CMAP = "seismic"
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
def plot_step_snapshot(jacobian_matrix, iteration, neg_count, min_val):
    """Show a single-panel Jacobian heatmap snapshot during iteration.

    Parameters
    ----------
    jacobian_matrix : ndarray, shape ``(1, H, W)``
        Current Jacobian determinant field.
    iteration : int
        Current outer iteration number.
    neg_count : int
        Number of non-positive Jdet pixels.
    min_val : float
        Minimum Jdet value in the field.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    norm = _make_jdet_norm([jacobian_matrix[0]])
    im = ax.imshow(jacobian_matrix[0], cmap=CMAP, norm=norm, interpolation=INTERP)
    _annotate_neg_contour(ax, jacobian_matrix[0])
    _annotate_jdet_values(ax, jacobian_matrix[0])
    ax.set_title(f"Iter {iteration}  |  neg={neg_count}  min={min_val:+.4f}", fontsize=10)
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
