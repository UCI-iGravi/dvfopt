"""High-impact "money shot" overview plots for DVF folding artifacts.

The functions in this module are the recommended starting point for
showing a DVF + its folds:

* :func:`plot_fold_overview` — single 4-panel figure capturing everything
  you'd want to see about a folded DVF (Jdet heatmap, warped grid with
  fold-cell highlights, Jdet distribution, per-row/col fold counts).
* :func:`plot_before_after` — side-by-side comparison of original vs
  corrected, with shared color scale.
* :func:`plot_solver_comparison` — multi-method panel for benchmarking
  different solvers on the same input.

All three apply the package theme via :func:`apply_theme` and use the
:class:`Palette` for consistent colors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.jacobian.tetrahedron_sign import (
    six_tet_fold_classification,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
from dvfopt.jacobian.triangle_sign import triangle_sign_areas2D
from dvfopt.viz.theme import PALETTE, apply_theme, jdet_norm


def _coerce_to_2hw(phi):
    """Accept (2,H,W) / (3,H,W) / (3,1,H,W) / (2,1,H,W). Returns (2,H,W) [dy,dx]."""
    phi = np.asarray(phi)
    if phi.ndim == 4:
        if phi.shape[0] == 3 and phi.shape[1] == 1:
            return np.stack([phi[1, 0], phi[2, 0]])
        if phi.shape[0] == 2 and phi.shape[1] == 1:
            return phi[:, 0]
    if phi.ndim == 3:
        if phi.shape[0] == 3:
            return np.stack([phi[1], phi[2]])
        if phi.shape[0] == 2:
            return phi
    raise ValueError(f'cannot interpret shape {phi.shape} as a 2D DVF')


def _fold_stats(phi2):
    """Return (fold_mask, T1, T2, n_neg, min_T, jdet).

    Shapes: fold_mask, T1, T2 are ``(H-1, W-1)``; jdet is ``(H, W)``.
    """
    T = triangle_sign_areas2D(phi2)
    T1, T2 = T[0], T[1]
    fold_mask = T.min(axis=0) <= 0
    n_neg = int(fold_mask.sum())
    min_T = float(T.min())
    jdet = jacobian_det2D(phi2)[0]
    return fold_mask, T1, T2, n_neg, min_T, jdet


# ---------------------------------------------------------------------------
# Money-shot: 4-panel fold overview
# ---------------------------------------------------------------------------


def plot_fold_overview(
    phi: np.ndarray,
    *,
    threshold: float = 0.01,
    title: Optional[str] = None,
    quiver_subsample: int = 4,
    show_jdet_values: bool = False,
    figsize: tuple = (12, 8.5),
    save_path: Optional[str] = None,
) -> Figure:
    """Single 4-panel figure capturing the state of a (possibly folded) DVF.

    Panel layout::

        +-------------------------+-------------------------+
        |  Jdet heatmap (top-L)   |  Warped grid (top-R)    |
        |  with fold contour      |  with folded cells red  |
        +-------------------------+-------------------------+
        |  Jdet distribution      |  Per-axis fold count    |
        |  (bottom-L)             |  (bottom-R)             |
        +-------------------------+-------------------------+

    Parameters
    ----------
    phi : ndarray
        DVF in any supported 2D shape (``(2, H, W)``, ``(3, H, W)``,
        ``(3, 1, H, W)``, ``(2, 1, H, W)``). dz is ignored if present.
    threshold : float
        Feasibility threshold (matches solver default 0.01). Shown as a
        green vline on the histogram + as the contour level on the
        heatmap.
    title : str, optional
        Suptitle for the figure.
    quiver_subsample : int
        Show every Nth grid vertex when drawing the warped grid. ``4``
        works for ~30×30; bump higher for larger slices.
    show_jdet_values : bool
        Annotate per-cell Jdet values on small grids (≤ 25×25). Off by
        default; turn on for tiny debug cases.
    figsize : tuple
    save_path : str, optional
        If set, also save the figure to this path (PNG/PDF/SVG auto-detected
        from extension).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_theme()
    import matplotlib.pyplot as plt

    phi2 = _coerce_to_2hw(phi)
    fold_mask, T1, T2, n_neg, min_T, jdet = _fold_stats(phi2)
    H, W = phi2.shape[1], phi2.shape[2]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # -----------------------------------------------------------------
    # Top-left: Jdet heatmap with fold contour
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    norm = jdet_norm([jdet], threshold=threshold)
    im = ax.imshow(jdet, cmap=PALETTE.cmap_jdet, norm=norm, interpolation='nearest')
    if (jdet <= 0).any():
        ax.contour(
            (jdet <= 0).astype(float),
            levels=[0.5],
            colors=PALETTE.fold,
            linewidths=1.4,
        )
    ax.set_title(f'Jacobian determinant  ({n_neg} folded, min={min_T:+.4f})')
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.axhline(0, color='black', lw=0.6)
    cb.ax.axhline(threshold, color=PALETTE.feasible, lw=0.8, ls='--')

    if show_jdet_values and max(H, W) <= 25:
        for r in range(H):
            for c in range(W):
                v = jdet[r, c]
                ax.text(
                    c,
                    r,
                    f'{v:.2f}',
                    ha='center',
                    va='center',
                    fontsize=max(4, 100 / max(H, W)),
                    color='white' if v < -0.2 else '#222222',
                    fontweight='bold' if v <= 0 else 'normal',
                )

    # -----------------------------------------------------------------
    # Top-right: warped grid with folded cells highlighted
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    yy, xx = np.mgrid[:H, :W]
    gx = xx + phi2[1]
    gy = yy + phi2[0]
    step = max(1, quiver_subsample)
    # Reference grid (light gray) — original lattice.
    for i in range(0, H, step):
        ax.plot(xx[i], yy[i], color=PALETTE.grid_ref, lw=0.4, zorder=0)
    for j in range(0, W, step):
        ax.plot(xx[:, j], yy[:, j], color=PALETTE.grid_ref, lw=0.4, zorder=0)
    # Warped grid (blue).
    for i in range(0, H, step):
        ax.plot(gx[i], gy[i], color=PALETTE.grid_warp, lw=0.7, zorder=1)
    for j in range(0, W, step):
        ax.plot(gx[:, j], gy[:, j], color=PALETTE.grid_warp, lw=0.7, zorder=1)
    # Per-triangle fold visualization.
    #
    # Each cell (r, c) is split TR-BL into:
    #     T1 = {TR, BL, BR}   (lower-right triangle)
    #     T2 = {TL, BL, TR}   (upper-left  triangle)
    # The TR-BL diagonal is the shared edge.
    #
    # We classify each cell:
    #   - T1<=0 and T2<=0  → both flipped (dense fold). Fill both deep red.
    #   - T1<=0 only       → fill T1 orange (single-triangle flip).
    #   - T2<=0 only       → fill T2 orange.
    # Folded-cell quad outline + the TR-BL diagonal are drawn in red on top.
    n_t1_only = n_t2_only = n_both = 0
    if fold_mask.any():
        from matplotlib.patches import Polygon

        cy, cx = np.where(fold_mask)
        for r, c in zip(cy, cx):
            tl_x, tl_y = gx[r, c], gy[r, c]
            tr_x, tr_y = gx[r, c + 1], gy[r, c + 1]
            bl_x, bl_y = gx[r + 1, c], gy[r + 1, c]
            br_x, br_y = gx[r + 1, c + 1], gy[r + 1, c + 1]
            t1_bad = T1[r, c] <= 0
            t2_bad = T2[r, c] <= 0
            if t1_bad and t2_bad:
                n_both += 1
                fill = PALETTE.fold
                alpha = 0.45
                # Both triangles: fill T1 = (TR, BL, BR), T2 = (TL, BL, TR).
                ax.add_patch(
                    Polygon(
                        [(tr_x, tr_y), (bl_x, bl_y), (br_x, br_y)],
                        closed=True,
                        facecolor=fill,
                        alpha=alpha,
                        edgecolor='none',
                        zorder=2,
                    )
                )
                ax.add_patch(
                    Polygon(
                        [(tl_x, tl_y), (bl_x, bl_y), (tr_x, tr_y)],
                        closed=True,
                        facecolor=fill,
                        alpha=alpha,
                        edgecolor='none',
                        zorder=2,
                    )
                )
            elif t1_bad:
                n_t1_only += 1
                ax.add_patch(
                    Polygon(
                        [(tr_x, tr_y), (bl_x, bl_y), (br_x, br_y)],
                        closed=True,
                        facecolor=PALETTE.orange,
                        alpha=0.55,
                        edgecolor='none',
                        zorder=2,
                    )
                )
            elif t2_bad:
                n_t2_only += 1
                ax.add_patch(
                    Polygon(
                        [(tl_x, tl_y), (bl_x, bl_y), (tr_x, tr_y)],
                        closed=True,
                        facecolor=PALETTE.orange,
                        alpha=0.55,
                        edgecolor='none',
                        zorder=2,
                    )
                )
            # TR-BL diagonal in red on top.
            ax.plot([tr_x, bl_x], [tr_y, bl_y], color=PALETTE.fold, lw=0.9, zorder=3)
            # Quad outline in red.
            ax.plot(
                [tl_x, tr_x, br_x, bl_x, tl_x],
                [tl_y, tr_y, br_y, bl_y, tl_y],
                color=PALETTE.fold,
                lw=1.2,
                zorder=3,
            )

    ax.set_aspect('equal')
    ax.invert_yaxis()
    parts = []
    if n_both:
        parts.append(f'{n_both} both')
    if n_t1_only:
        parts.append(f'{n_t1_only} T1-only')
    if n_t2_only:
        parts.append(f'{n_t2_only} T2-only')
    triangle_summary = '  ·  '.join(parts) if parts else 'no triangle flips'
    ax.set_title(f'Warped grid (every {step} cells)\n{triangle_summary}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # -----------------------------------------------------------------
    # Bottom-left: Jdet distribution
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    flat = jdet.ravel()
    lo = min(float(flat.min()), -0.05)
    hi = max(float(flat.max()), threshold * 3)
    bins = np.linspace(lo, hi, 60)
    ax.hist(
        flat,
        bins=bins,
        color=PALETTE.blue,
        edgecolor='white',
        linewidth=0.4,
        alpha=0.85,
    )
    ax.axvline(0, color='black', lw=0.8, label='Jdet = 0 (fold boundary)')
    ax.axvline(
        threshold,
        color=PALETTE.feasible,
        lw=1.0,
        ls='--',
        label=f'threshold {threshold}',
    )
    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('Jacobian determinant')
    ax.set_ylabel('Pixels (symlog)')
    ax.set_title('Distribution of Jdet values')
    ax.legend(loc='upper right')

    # -----------------------------------------------------------------
    # Bottom-right: per-row + per-column fold counts
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    if fold_mask.any():
        per_row = fold_mask.sum(axis=1)
        per_col = fold_mask.sum(axis=0)
        ax.plot(
            np.arange(per_row.size),
            per_row,
            color=PALETTE.fold,
            lw=1.4,
            label='per row',
            marker='.',
            markersize=3,
        )
        ax.plot(
            np.arange(per_col.size),
            per_col,
            color=PALETTE.orange,
            lw=1.4,
            label='per column',
            marker='.',
            markersize=3,
            alpha=0.85,
        )
        ax.set_xlabel('row / column index')
        ax.set_ylabel('Folded cells')
        ax.set_title('Where the folds live')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            'No folded cells\n(feasible field)',
            ha='center',
            va='center',
            transform=ax.transAxes,
            color=PALETTE.feasible,
            fontsize=12,
            fontweight='bold',
        )
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Before/after comparison
# ---------------------------------------------------------------------------


def plot_before_after(
    phi_before: np.ndarray,
    phi_after: np.ndarray,
    *,
    threshold: float = 0.01,
    title: Optional[str] = None,
    figsize: tuple = (13, 5.5),
    save_path: Optional[str] = None,
) -> Figure:
    """Side-by-side Jdet comparison of an original vs corrected DVF.

    Both panels share a single :class:`TwoSlopeNorm` so colors are
    directly comparable. A third skinny panel shows the per-pixel
    correction magnitude.
    """
    apply_theme()
    import matplotlib.pyplot as plt

    phi_b = _coerce_to_2hw(phi_before)
    phi_a = _coerce_to_2hw(phi_after)

    _, _, _, n_b, m_b, jdet_b = _fold_stats(phi_b)
    _, _, _, n_a, m_a, jdet_a = _fold_stats(phi_a)
    norm = jdet_norm([jdet_b, jdet_a], threshold=threshold)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1]}, constrained_layout=True
    )

    for ax, jdet, label, n, m in [
        (axes[0], jdet_b, 'BEFORE', n_b, m_b),
        (axes[1], jdet_a, 'AFTER', n_a, m_a),
    ]:
        im = ax.imshow(jdet, cmap=PALETTE.cmap_jdet, norm=norm, interpolation='nearest')
        if (jdet <= 0).any():
            ax.contour((jdet <= 0).astype(float), levels=[0.5], colors=PALETTE.fold, linewidths=1.2)
        ax.set_title(f'{label}\nn_neg={n}  min={m:+.4f}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar to the right of the AFTER panel.
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.ax.axhline(0, color='black', lw=0.6)
    cb.ax.axhline(threshold, color=PALETTE.feasible, lw=0.8, ls='--')

    # Right panel: per-pixel correction magnitude.
    delta = np.sqrt(((phi_a - phi_b) ** 2).sum(axis=0))
    ax = axes[2]
    im2 = ax.imshow(delta, cmap=PALETTE.cmap_magnitude, interpolation='nearest')
    ax.set_title(f'|after − before|\nmax={float(delta.max()):.3f}')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Multi-solver comparison
# ---------------------------------------------------------------------------


def plot_solver_comparison(
    phi_in: np.ndarray,
    results: dict[str, np.ndarray],
    *,
    threshold: float = 0.01,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Compare N solvers' outputs on the same input, one Jdet panel each.

    Parameters
    ----------
    phi_in : ndarray
        Original (folded) DVF.
    results : dict[str, ndarray]
        Map of solver-label → corrected DVF. All entries must have the
        same shape as ``phi_in``.
    """
    apply_theme()
    import matplotlib.pyplot as plt

    phi_in = _coerce_to_2hw(phi_in)
    items = [('INPUT', phi_in)] + [(k, _coerce_to_2hw(v)) for k, v in results.items()]
    n = len(items)
    if figsize is None:
        figsize = (3.6 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False, constrained_layout=True)
    axes = axes[0]

    # Shared norm across all solvers.
    jdets = [jacobian_det2D(phi)[0] for _, phi in items]
    norm = jdet_norm(jdets, threshold=threshold)

    for ax, (label, _phi), jdet in zip(axes, items, jdets):
        n_neg = int((jdet <= 0).sum())
        im = ax.imshow(jdet, cmap=PALETTE.cmap_jdet, norm=norm, interpolation='nearest')
        if n_neg:
            ax.contour((jdet <= 0).astype(float), levels=[0.5], colors=PALETTE.fold, linewidths=1.0)
        ax.set_title(f'{label}\nn_neg={n_neg}  min={float(jdet.min()):+.4f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# 3D money shot
# ---------------------------------------------------------------------------


def _coerce_to_3dhw(phi):
    """Validate ``phi`` is a 3D DVF: ``(3, D, H, W)``."""
    phi = np.asarray(phi)
    if phi.ndim == 4 and phi.shape[0] == 3:
        return phi
    raise ValueError(f'cannot interpret shape {phi.shape} as a 3D DVF (need (3, D, H, W))')


def plot_fold_overview_3d(
    phi: np.ndarray,
    *,
    threshold: float = 0.01,
    title: Optional[str] = None,
    elev: float = 25.0,
    azim: float = -60.0,
    max_positive_points: int = 3000,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """4-panel money-shot for a (possibly folded) 3D DVF.

    Panel layout::

        +-------------------------+-------------------------+
        |  3D Jdet scatter        |  Worst z-slice Jdet     |
        |  (top-L, neg prominent) |  heatmap (top-R)        |
        +-------------------------+-------------------------+
        |  Jdet distribution      |  Per-axis fold counts   |
        |  + per-tet flip hist    |  (folds vs Z / Y / X)   |
        |  (bottom-L)             |  (bottom-R)             |
        +-------------------------+-------------------------+

    Direct 3D analogue of :func:`plot_fold_overview`. Adds a
    *per-tetrahedron* flip histogram in the bottom-left panel (using the
    simplex (3D) decomposition from :func:`six_tet_volumes_3d`), which
    distinguishes a few-tet-flipped voxel (light correction job) from a
    whole-voxel collapse (heavy job for SLSQP / barrier).

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    threshold : float
    title : str, optional
    elev, azim : float
        Viewing angle for the 3D scatter panel.
    max_positive_points : int
        Cap on positive-voxel points drawn as a faint cloud.
    figsize : tuple
    save_path : str, optional
    """
    apply_theme()
    import matplotlib.pyplot as plt

    phi3 = _coerce_to_3dhw(phi)
    _, D, H, W = phi3.shape

    jdet = jacobian_det3D(phi3)
    neg_mask = jdet <= 0
    n_neg = int(neg_mask.sum())

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax_scatter = fig.add_subplot(2, 2, 1, projection='3d')
    ax_slice = fig.add_subplot(2, 2, 2)
    ax_hist = fig.add_subplot(2, 2, 3)
    ax_axes = fig.add_subplot(2, 2, 4)

    # -----------------------------------------------------------------
    # Top-left: 3D Jdet scatter (negative voxels prominent)
    # -----------------------------------------------------------------
    norm = jdet_norm([jdet], threshold=threshold)
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    z_flat, y_flat, x_flat, j_flat = zz.ravel(), yy.ravel(), xx.ravel(), jdet.ravel()
    pos = ~neg_mask.ravel()
    neg = neg_mask.ravel()

    if pos.any() and max_positive_points > 0:
        pos_idx = np.flatnonzero(pos)
        if pos_idx.size > max_positive_points:
            rng = np.random.default_rng(0)
            pos_idx = rng.choice(pos_idx, size=max_positive_points, replace=False)
        ax_scatter.scatter(
            x_flat[pos_idx],
            y_flat[pos_idx],
            z_flat[pos_idx],
            c=j_flat[pos_idx],
            cmap=PALETTE.cmap_jdet,
            norm=norm,
            s=6,
            alpha=0.06,
            edgecolors='none',
            depthshade=True,
        )
    if neg.any():
        ax_scatter.scatter(
            x_flat[neg],
            y_flat[neg],
            z_flat[neg],
            c=j_flat[neg],
            cmap=PALETTE.cmap_jdet,
            norm=norm,
            s=120,
            alpha=0.95,
            edgecolors='black',
            linewidth=0.5,
            depthshade=False,
        )
    # Faint bounding cube for spatial reference.
    cube = [
        (0, 0, 0),
        (W, 0, 0),
        (W, H, 0),
        (0, H, 0),
        (0, 0, D),
        (W, 0, D),
        (W, H, D),
        (0, H, D),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        ax_scatter.plot(
            [cube[a][0], cube[b][0]],
            [cube[a][1], cube[b][1]],
            [cube[a][2], cube[b][2]],
            color=PALETTE.gray,
            linewidth=0.5,
            alpha=0.4,
        )
    ax_scatter.set_xlim(0, W)
    ax_scatter.set_ylim(0, H)
    ax_scatter.set_zlim(0, D)
    ax_scatter.set_xlabel('X')
    ax_scatter.set_ylabel('Y')
    ax_scatter.set_zlabel('Z')
    ax_scatter.view_init(elev=elev, azim=azim)
    ax_scatter.set_title(f'3D Jdet  ({n_neg} folded, min={float(jdet.min()):+.4f})')

    # -----------------------------------------------------------------
    # Top-right: worst z-slice (the slice with the most negative voxels)
    # -----------------------------------------------------------------
    neg_per_z = neg_mask.sum(axis=(1, 2))
    worst_z = int(neg_per_z.argmax()) if n_neg else D // 2
    im = ax_slice.imshow(
        jdet[worst_z],
        cmap=PALETTE.cmap_jdet,
        norm=norm,
        interpolation='nearest',
    )
    if (jdet[worst_z] <= 0).any():
        ax_slice.contour(
            (jdet[worst_z] <= 0).astype(float),
            levels=[0.5],
            colors=PALETTE.fold,
            linewidths=1.3,
        )
    ax_slice.set_title(f'Worst slice  (z={worst_z}, {int(neg_per_z[worst_z])} folded)')
    ax_slice.set_xticks([])
    ax_slice.set_yticks([])
    cb = fig.colorbar(im, ax=ax_slice, fraction=0.046, pad=0.02)
    cb.ax.axhline(0, color='black', lw=0.6)
    cb.ax.axhline(threshold, color=PALETTE.feasible, lw=0.8, ls='--')

    # -----------------------------------------------------------------
    # Bottom-left: Jdet distribution + per-tet flip histogram
    # -----------------------------------------------------------------
    # Use a twin x-axis so the two distributions read cleanly even with
    # different domains. Left axis = Jdet histogram (continuous).
    flat = j_flat
    lo = min(float(flat.min()), -0.05)
    hi = max(float(flat.max()), threshold * 3)
    bins = np.linspace(lo, hi, 60)
    ax_hist.hist(
        flat,
        bins=bins,
        color=PALETTE.blue,
        edgecolor='white',
        linewidth=0.4,
        alpha=0.85,
    )
    ax_hist.axvline(0, color='black', lw=0.8, label='Jdet=0')
    ax_hist.axvline(
        threshold,
        color=PALETTE.feasible,
        lw=1.0,
        ls='--',
        label=f'threshold {threshold}',
    )
    ax_hist.set_yscale('symlog', linthresh=10)
    ax_hist.set_xlabel('Jacobian determinant')
    ax_hist.set_ylabel('Voxels (symlog)', color=PALETTE.blue)
    ax_hist.tick_params(axis='y', labelcolor=PALETTE.blue)

    # Per-tet flip distribution: an inset bar chart in the same panel.
    # Counts cells by "how many of 6 tets are flipped" (0..6).
    n_flip = six_tet_fold_classification(phi3)
    counts = np.bincount(n_flip.ravel(), minlength=7)[:7]
    ax_tet = ax_hist.inset_axes([0.55, 0.55, 0.42, 0.4])
    bar_colors = [
        PALETTE.feasible,  # 0 flipped
        PALETTE.orange,  # 1
        PALETTE.orange,  # 2
        PALETTE.orange,  # 3
        PALETTE.fold,  # 4
        PALETTE.fold,  # 5
        PALETTE.fold,  # 6
    ]
    ax_tet.bar(
        np.arange(7),
        counts,
        color=bar_colors,
        edgecolor='white',
        linewidth=0.4,
    )
    if counts[1:].sum() > 0:
        ax_tet.set_yscale('symlog', linthresh=1)
    ax_tet.set_xticks(np.arange(7))
    ax_tet.set_xlabel('Tets flipped / voxel', fontsize=7)
    ax_tet.set_ylabel('Voxel cells', fontsize=7)
    ax_tet.tick_params(labelsize=7)
    ax_tet.set_title('Per-voxel tet flips (simplex (3D) split)', fontsize=8)

    ax_hist.set_title('Distribution of Jdet values')
    ax_hist.legend(loc='upper left', fontsize='x-small')

    # -----------------------------------------------------------------
    # Bottom-right: per-axis fold counts (Z / Y / X projections)
    # -----------------------------------------------------------------
    if n_neg:
        per_z = neg_mask.sum(axis=(1, 2))
        per_y = neg_mask.sum(axis=(0, 2))
        per_x = neg_mask.sum(axis=(0, 1))
        ax_axes.plot(
            np.arange(per_z.size),
            per_z,
            color=PALETTE.fold,
            lw=1.4,
            marker='.',
            markersize=3,
            label=f'along Z (D={D})',
        )
        ax_axes.plot(
            np.arange(per_y.size),
            per_y,
            color=PALETTE.orange,
            lw=1.4,
            marker='.',
            markersize=3,
            label=f'along Y (H={H})',
            alpha=0.85,
        )
        ax_axes.plot(
            np.arange(per_x.size),
            per_x,
            color=PALETTE.blue,
            lw=1.4,
            marker='.',
            markersize=3,
            label=f'along X (W={W})',
            alpha=0.85,
        )
        ax_axes.set_xlabel('axis index')
        ax_axes.set_ylabel('Folded voxels')
        ax_axes.set_title('Folds projected onto each axis')
        ax_axes.legend(loc='upper right', fontsize='x-small')
        ax_axes.grid(axis='y', alpha=0.3)
    else:
        ax_axes.text(
            0.5,
            0.5,
            'No folded voxels\n(feasible field)',
            ha='center',
            va='center',
            transform=ax_axes.transAxes,
            color=PALETTE.feasible,
            fontsize=12,
            fontweight='bold',
        )
        ax_axes.set_xticks([])
        ax_axes.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig


def plot_before_after_3d(
    phi_before: np.ndarray,
    phi_after: np.ndarray,
    *,
    threshold: float = 0.01,
    title: Optional[str] = None,
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """3D before/after — pair of 3D Jdet scatters with a shared norm.

    Light 3D analogue of :func:`plot_before_after`. Shows only the 3D
    scatter pair (no magnitude panel — meaningful 3D magnitude visuals
    need a separate volumetric view, out of scope here).
    """
    apply_theme()
    import matplotlib.pyplot as plt

    phi_b = _coerce_to_3dhw(phi_before)
    phi_a = _coerce_to_3dhw(phi_after)
    _, D, H, W = phi_b.shape

    jdet_b = jacobian_det3D(phi_b)
    jdet_a = jacobian_det3D(phi_a)
    n_b = int((jdet_b <= 0).sum())
    n_a = int((jdet_a <= 0).sum())
    norm = jdet_norm([jdet_b, jdet_a], threshold=threshold)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    for idx, (jd, label, n_neg) in enumerate([(jdet_b, 'BEFORE', n_b), (jdet_a, 'AFTER', n_a)]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        neg = jd <= 0
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        if neg.any():
            ax.scatter(
                xx[neg],
                yy[neg],
                zz[neg],
                c=jd[neg],
                cmap=PALETTE.cmap_jdet,
                norm=norm,
                s=120,
                alpha=0.95,
                edgecolors='black',
                linewidth=0.5,
                depthshade=False,
            )
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(0, D)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'{label} — {n_neg} folded, min={float(jd.min()):+.4f}')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig


__all__ = [
    'plot_before_after',
    'plot_before_after_3d',
    'plot_fold_overview',
    'plot_fold_overview_3d',
    'plot_solver_comparison',
]
