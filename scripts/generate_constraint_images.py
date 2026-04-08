"""Generate before/after constraint-comparison images for the README.

For a representative test case, runs iterative SLSQP with each constraint
configuration and saves side-by-side initial-vs-corrected Jacobian heatmaps.
"""
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from test_cases import SYNTHETIC_CASES, make_deformation
from dvfopt import iterative_serial, jacobian_det2D
from dvfopt.jacobian import shoelace_det2D, _monotonicity_diffs_2d

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP = "RdBu_r"

# Use a case with moderate complexity and visible folds
CASE_KEY = "03d_20x20_crossing"


def _make_norm(arrays):
    vmin = min(a.min() for a in arrays)
    vmax = max(a.max() for a in arrays)
    vmin = min(vmin, -1)
    vmax = max(vmax, 1)
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


def _add_neg_contour(ax, data, color="limegreen", lw=1.5):
    mask = (data <= 0).astype(float)
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=[lw])


def save_constraint_comparison(deformation, phi_corrected, title, filename,
                                extra_metric_initial=None,
                                extra_metric_corrected=None,
                                extra_label=None):
    """Save a 2x2 figure: initial vs corrected Jacobian + optional extra metric."""
    phi_init_2hw = deformation[[1, 2], 0, :, :]
    J_init = np.squeeze(jacobian_det2D(phi_init_2hw))
    J_corr = np.squeeze(jacobian_det2D(phi_corrected))
    neg_init = int(np.sum(J_init <= 0))
    neg_corr = int(np.sum(J_corr <= 0))

    has_extra = extra_metric_initial is not None

    nrows = 2 if has_extra else 1
    fig, axs = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))
    if nrows == 1:
        axs = axs[np.newaxis, :]

    # Row 0: Jacobian determinant
    norm_j = _make_norm([J_init, J_corr])
    im0 = axs[0, 0].imshow(J_init, cmap=CMAP, norm=norm_j)
    _add_neg_contour(axs[0, 0], J_init)
    axs[0, 0].set_title(f"Initial Jdet ({neg_init} negative)", fontsize=11)

    im1 = axs[0, 1].imshow(J_corr, cmap=CMAP, norm=norm_j)
    _add_neg_contour(axs[0, 1], J_corr)
    axs[0, 1].set_title(f"Corrected Jdet ({neg_corr} negative)", fontsize=11)

    cbar = fig.colorbar(im1, ax=axs[0, :].tolist(), fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Jacobian determinant", fontsize=9)

    # Row 1: Extra metric (shoelace or monotonicity)
    if has_extra:
        E_init = extra_metric_initial
        E_corr = extra_metric_corrected
        neg_e_init = int(np.sum(E_init <= 0))
        neg_e_corr = int(np.sum(E_corr <= 0))
        norm_e = _make_norm([E_init, E_corr])

        im2 = axs[1, 0].imshow(E_init, cmap=CMAP, norm=norm_e)
        _add_neg_contour(axs[1, 0], E_init)
        axs[1, 0].set_title(f"Initial {extra_label} ({neg_e_init} negative)", fontsize=11)

        im3 = axs[1, 1].imshow(E_corr, cmap=CMAP, norm=norm_e)
        _add_neg_contour(axs[1, 1], E_corr)
        axs[1, 1].set_title(f"Corrected {extra_label} ({neg_e_corr} negative)", fontsize=11)

        cbar2 = fig.colorbar(im3, ax=axs[1, :].tolist(), fraction=0.046, pad=0.04, shrink=0.85)
        cbar2.set_label(extra_label, fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def compute_shoelace_map(phi_2hw):
    """Get per-cell shoelace areas as (H-1, W-1) array."""
    areas = shoelace_det2D(phi_2hw)  # (1, H-1, W-1)
    return areas[0]


def compute_monotonicity_map(phi_2hw):
    """Get per-pixel worst monotonicity diff as (H, W)."""
    dy, dx = phi_2hw[0], phi_2hw[1]
    h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
    H, W = dy.shape
    result = np.full((H, W), np.inf)
    result[:, :-1] = np.minimum(result[:, :-1], h_mono)
    result[:, 1:]  = np.minimum(result[:, 1:],  h_mono)
    result[:-1, :] = np.minimum(result[:-1, :], v_mono)
    result[1:, :]  = np.minimum(result[1:, :],  v_mono)
    return result


def main():
    print(f"=== Generating constraint comparison images ({CASE_KEY}) ===\n")

    deformation, ms, fs = make_deformation(CASE_KEY)
    phi_init_2hw = deformation[[1, 2], 0, :, :]

    # ---- 1. Jacobian-only (default) ----
    print("1/4  Jacobian-only constraint ...")
    t0 = time.time()
    phi_jac = iterative_serial(
        deformation, verbose=0, enforce_shoelace=False, enforce_injectivity=False,
    )
    t_jac = time.time() - t0
    print(f"     Done in {t_jac:.1f}s")
    save_constraint_comparison(
        deformation, phi_jac,
        f"Jacobian-only constraint  ({t_jac:.1f}s)",
        "constraint_jacobian_only.png",
    )

    # ---- 2. Jacobian + Shoelace ----
    print("2/4  Jacobian + Shoelace ...")
    t0 = time.time()
    phi_shoe = iterative_serial(
        deformation, verbose=0, enforce_shoelace=True, enforce_injectivity=False,
    )
    t_shoe = time.time() - t0
    print(f"     Done in {t_shoe:.1f}s")
    save_constraint_comparison(
        deformation, phi_shoe,
        f"Jacobian + Shoelace constraint  ({t_shoe:.1f}s)",
        "constraint_jacobian_shoelace.png",
        extra_metric_initial=compute_shoelace_map(phi_init_2hw),
        extra_metric_corrected=compute_shoelace_map(phi_shoe),
        extra_label="Shoelace cell area",
    )

    # ---- 3. Jacobian + Injectivity ----
    print("3/4  Jacobian + Injectivity ...")
    t0 = time.time()
    phi_inject = iterative_serial(
        deformation, verbose=0, enforce_shoelace=False, enforce_injectivity=True,
    )
    t_inject = time.time() - t0
    print(f"     Done in {t_inject:.1f}s")
    save_constraint_comparison(
        deformation, phi_inject,
        f"Jacobian + Injectivity constraint  ({t_inject:.1f}s)",
        "constraint_jacobian_injectivity.png",
        extra_metric_initial=compute_monotonicity_map(phi_init_2hw),
        extra_metric_corrected=compute_monotonicity_map(phi_inject),
        extra_label="Monotonicity (worst diff)",
    )

    # ---- 4. All three ----
    print("4/4  All constraints ...")
    t0 = time.time()
    phi_all = iterative_serial(
        deformation, verbose=0, enforce_shoelace=True, enforce_injectivity=True,
    )
    t_all = time.time() - t0
    print(f"     Done in {t_all:.1f}s")
    # Show both extra metrics for the full-constraint case
    save_constraint_comparison(
        deformation, phi_all,
        f"All constraints (Jdet + Shoelace + Injectivity)  ({t_all:.1f}s)",
        "constraint_all.png",
        extra_metric_initial=compute_shoelace_map(phi_init_2hw),
        extra_metric_corrected=compute_shoelace_map(phi_all),
        extra_label="Shoelace cell area",
    )

    # ---- L2 error comparison summary ----
    def l2(phi_corr):
        return float(np.sqrt(np.sum((phi_corr - phi_init_2hw) ** 2)))

    print("\n=== L2 error comparison ===")
    print(f"  Jacobian-only       : L2 = {l2(phi_jac):.4f}  ({t_jac:.1f}s)")
    print(f"  + Shoelace          : L2 = {l2(phi_shoe):.4f}  ({t_shoe:.1f}s)")
    print(f"  + Injectivity       : L2 = {l2(phi_inject):.4f}  ({t_inject:.1f}s)")
    print(f"  All three           : L2 = {l2(phi_all):.4f}  ({t_all:.1f}s)")

    print(f"\nDone — 4 images in {OUT_DIR}")


if __name__ == "__main__":
    main()
