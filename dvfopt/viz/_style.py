"""Shared styling constants and internal helpers for visualisation."""

import matplotlib.colors as mcolors
import numpy as np

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
