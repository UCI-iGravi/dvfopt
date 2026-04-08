"""DVF / Jacobian field visualisation functions."""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from dvfopt.jacobian import jacobian_det2D
import dvfopt.jacobian.sitk_jdet as _sitk
from dvfopt.viz._style import (
    CMAP, INTERP, QUIVER_COLOR,
    _make_jdet_norm, _annotate_jdet_values, _annotate_neg_contour,
)


# ---------------------------------------------------------------------------
# Initial-state preview (shown before correction runs)
# ---------------------------------------------------------------------------
def plot_initial_deformation(deformation_i, msample=None, fsample=None,
                             figsize=(14, 6), quiver_scale=1):
    """Quick preview of the initial deformation field before correction.

    Shows a 1\u00d72 layout: Jacobian determinant heatmap (left) and
    displacement quiver (right).

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    msample, fsample : ndarray or None
        Moving / fixed correspondences ``(N, 3)`` with ``[z, y, x]``.
    figsize : tuple
    quiver_scale : float
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

    Layout (2\u00d72):
    * Top-left:  Initial Jacobian determinant heatmap
    * Top-right: Corrected Jacobian determinant heatmap
    * Bottom-left:  Initial displacement quiver
    * Bottom-right: Corrected displacement quiver

    Parameters
    ----------
    msample, fsample : ndarray or None
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    phi_corrected : ndarray, shape ``(2, H, W)``
    figsize : tuple
    save_path : str or None
    title : str
    quiver_scale : float
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
def plot_jacobians_iteratively(jacobians, msample=None, fsample=None, method_name="SLSQP"):
    """Plot a sequence of Jacobian determinant maps in a grid.

    Parameters
    ----------
    jacobians : list of ndarray
        Each entry is a ``(1, H, W)`` Jacobian determinant array.
    msample, fsample : ndarray or None
    method_name : str
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
    fig.suptitle(f"Jacobian progression \u2014 {method_name}", fontsize=13, fontweight="bold")
    plt.show()


# ---------------------------------------------------------------------------
# Single-field deformation preview (Jacobian heatmap + quiver)
# ---------------------------------------------------------------------------
def plot_deformation_field(deformation, msample=None, fsample=None,
                           title="", figsize=(20, 10), show_values=False,
                           show_points=True, save_path=None, quiver_scale=None):
    """Plot a single deformation field: Jacobian heatmap + displacement quiver.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, 1, H, W)``
    msample, fsample : ndarray ``(N, 3)`` or None
    title : str
    save_path : str or None
    quiver_scale : float or None
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    J = np.squeeze(_sitk.sitk_jacobian_determinant(deformation))
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
