"""Folding close-up visualisations: reference vs deformed grid neighbourhoods."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from dvfopt.jacobian import jacobian_det2D


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _quad_signed_areas(def_x, def_y):
    """Shoelace-formula signed area for each quad cell.

    Returns shape ``(rows-1, cols-1)``.
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
    jac_patch : ndarray, shape ``(rows-1, cols-1)`` or None
    ref_x, ref_y : ndarray or None
    label_vertices : bool
    color_by_direction : bool
    """
    rows, cols = def_x.shape

    ROW_COLOR = '#2166ac'
    COL_COLOR = '#b2182b'

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


# ---------------------------------------------------------------------------
# Three-column folding close-up
# ---------------------------------------------------------------------------
def plot_checkerboard_before_after(deformation_i, phi_corrected, figsize=None,
                                   title="", max_panels=4, half_win=1):
    """Three-column folding close-up: Reference | Initial | Corrected.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    phi_corrected : ndarray, shape ``(2, H, W)``
    half_win : int
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

        ref_yy, ref_xx = np.mgrid[y0:y1, x0:x1].astype(float)

        def _deformed(phi):
            dy = phi[0][y0:y1, x0:x1]
            dx = phi[1][y0:y1, x0:x1]
            return ref_xx + dx, ref_yy + dy

        ix, iy = _deformed(phi_init)
        cx_, cy_ = _deformed(phi_corrected)

        area_init = _quad_signed_areas(ix, iy)
        area_corr = _quad_signed_areas(cx_, cy_)

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
            f"Initial \u2014 {n_neg} folded  (min area={area_init.min():.2f})",
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
            f"Corrected \u2014 {n_neg_c} folded  (min area={area_corr.min():.2f})",
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
# Correction displacement arrows
# ---------------------------------------------------------------------------
def plot_neg_jdet_neighborhoods(deformation_i, phi_corrected, figsize=None,
                                title="", max_panels=4, half_win=2):
    """Per-region displacement arrows showing the correction applied.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    phi_corrected : ndarray, shape ``(2, H, W)``
    half_win : int
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

        area_init = _quad_signed_areas(ix, iy)
        area_corr = _quad_signed_areas(cx_, cy_)

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
