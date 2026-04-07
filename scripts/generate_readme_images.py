"""Generate example images for each test case in the README.

Creates docs/images/ with one .png per synthetic/random DVF test case.
Each image shows Jacobian determinant heatmap + quiver plot.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from dvfopt.testcases import (
    SYNTHETIC_CASES, RANDOM_DVF_CASES,
    make_deformation, make_random_dvf,
)
from dvfopt import jacobian_det2D

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP = "RdBu_r"


def save_deformation_preview(deformation, title, filename,
                              msample=None, fsample=None):
    """Save a 1x2 Jdet-heatmap + quiver plot as PNG."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    phi_2hw = deformation[[1, 2], 0, :, :]  # (2, H, W) — [dy, dx]
    J = np.squeeze(jacobian_det2D(phi_2hw))
    neg = int(np.sum(J <= 0))
    norm = mcolors.TwoSlopeNorm(
        vmin=min(J.min(), -3), vcenter=0, vmax=max(J.max(), 3)
    )

    # Left: Jacobian heatmap with correspondence arrows
    im = axs[0].imshow(J, cmap=CMAP, norm=norm)
    if msample is not None and fsample is not None:
        for i in range(len(msample)):
            axs[0].annotate(
                "", xy=(fsample[i][2], fsample[i][1]),
                xytext=(msample[i][2], msample[i][1]),
                arrowprops=dict(facecolor="black", shrink=0.045,
                                headwidth=8, headlength=10, width=3),
            )
        axs[0].scatter(msample[:, 2], msample[:, 1], c="g",
                        zorder=5, label="Moving")
        axs[0].scatter(fsample[:, 2], fsample[:, 1], c="violet",
                        zorder=5, label="Fixed")
        axs[0].legend(fontsize=8)
    axs[0].set_title(f"Jacobian determinant ({neg} negative)")

    # Right: quiver on Jdet background
    H, W = deformation.shape[2], deformation.shape[3]
    x, y = np.meshgrid(range(W), range(H), indexing="xy")
    axs[1].imshow(J, cmap=CMAP, norm=norm)
    axs[1].quiver(x, y, deformation[2, 0], -deformation[1, 0])
    axs[1].set_title("Displacement vector field")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Jacobian determinant")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print("=== Synthetic test cases ===")
    for key, case in SYNTHETIC_CASES.items():
        print(f"  Generating {key} ...")
        deformation, ms, fs = make_deformation(key)
        save_deformation_preview(
            deformation, case["title"], f"synthetic_{key}.png",
            msample=ms, fsample=fs,
        )

    print("\n=== Random DVF test cases ===")
    for key, case in RANDOM_DVF_CASES.items():
        print(f"  Generating {key} ...")
        deformation = make_random_dvf(key)
        save_deformation_preview(
            deformation, case["title"], f"random_{key}.png",
        )

    print(f"\nDone — {len(SYNTHETIC_CASES) + len(RANDOM_DVF_CASES)} images in {OUT_DIR}")


if __name__ == "__main__":
    main()
