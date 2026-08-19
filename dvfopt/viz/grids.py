"""Deformation grid visualisations."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from scipy.ndimage import map_coordinates

import dvfopt.jacobian.sitk_jdet as _sitk
from dvfopt.jacobian import jacobian_det2D


# ---------------------------------------------------------------------------
# Shared cell-polygon rendering
# ---------------------------------------------------------------------------
def _draw_jdet_cell_polygons(ax, def_x, def_y, jac, *, cmap, norm, spacing=1, linewidth=0.5):
    """Add one matplotlib Polygon per cell, coloured by ``jac[i, j]``.

    Cells with non-positive Jdet are drawn opaque with a yellow border
    (high zorder so they stand out); positive-Jdet cells get a light tint.

    Shared between :func:`plot_grid` and :func:`plot_grid_before_after` —
    the iteration pattern and colouring used to be duplicated verbatim
    across both call sites.
    """
    H, W = def_x.shape
    for i in range(0, H - spacing, spacing):
        for j in range(0, W - spacing, spacing):
            ci = [i, i, i + spacing, i + spacing]
            cj = [j, j + spacing, j + spacing, j]
            corners = [
                (
                    def_x[np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)],
                    def_y[np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)],
                )
                for r, c in zip(ci, cj)
            ]
            jval = jac[i, j]
            fc = cmap(norm(jval))
            if jval <= 0:
                poly = Polygon(
                    corners,
                    closed=True,
                    facecolor=fc,
                    edgecolor="yellow",
                    linewidth=max(linewidth * 3, 1.5),
                    zorder=2,
                )
            else:
                poly = Polygon(
                    corners, closed=True, facecolor=(*fc[:3], 0.25), edgecolor="none", zorder=0
                )
            ax.add_patch(poly)


# ---------------------------------------------------------------------------
# Simple deformed-grid plots
# ---------------------------------------------------------------------------
def plot_2d_deformation_grid(
    deformation, spacing=1, xlim=None, ylim=None, title="2D Deformation Grid", highlight_point=None
):
    """Visualise a ``(3, 1, Y, X)`` deformation as a deformed grid.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, 1, Y, X)``
    spacing : int
    highlight_point : tuple ``(y, x)`` or None
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    y_coords, x_coords = np.meshgrid(
        np.arange(0, H, spacing), np.arange(0, W, spacing), indexing="ij"
    )
    new_y = y_coords + dy[y_coords, x_coords]
    new_x = x_coords + dx[y_coords, x_coords]

    _fig, ax = plt.subplots(figsize=(8, 8))
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
            ax.plot((*xs, xs[0]), (*ys, ys[0]), color="black", linewidth=1.5)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_deformed_quads(
    deformation, center_y, center_x, spacing=1, patch_size=20, title="Deformed Quadrilateral Mesh"
):
    """Plot a zoomed-in deformed quadrilateral mesh.

    Parameters
    ----------
    deformation : ndarray ``(3, 1, Y, X)``
    center_y, center_x : int
    spacing : int
    patch_size : int
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    y0 = max(center_y - patch_size // 2, 0)
    y1 = min(center_y + patch_size // 2, H - spacing - 1)
    x0 = max(center_x - patch_size // 2, 0)
    x1 = min(center_x + patch_size // 2, W - spacing - 1)

    _fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(y0, y1, spacing):
        for j in range(x0, x1, spacing):
            corners = [(j, i), (j + spacing, i), (j + spacing, i + spacing), (j, i + spacing)]
            deformed = [(x + dx[y, x], y + dy[y, x]) for x, y in corners]
            poly = Polygon(
                deformed, closed=True, edgecolor="red", facecolor="lightgray", linewidth=0.8
            )
            ax.add_patch(poly)

    ax.set_xlim(x0, x1 + spacing)
    ax.set_ylim(y1 + spacing, y0)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_deformed_quads_colored(
    deformation, center_y, center_x, spacing=1, patch_size=20, cmap="bwr"
):
    """Plot a zoomed quad mesh coloured by Jacobian determinant.

    Negative-Jdet quads are outlined in yellow.
    """
    _, _, H, W = deformation.shape
    dy = deformation[1, 0]
    dx = deformation[2, 0]

    J = np.squeeze(_sitk.sitk_jacobian_determinant(deformation))
    norm = mcolors.TwoSlopeNorm(vmin=min(J.min(), -3), vcenter=0, vmax=max(J.max(), 3))
    colormap = plt.get_cmap(cmap)

    y0 = max(center_y - patch_size // 2, 0)
    y1 = min(center_y + patch_size // 2, H - spacing - 1)
    x0 = max(center_x - patch_size // 2, 0)
    x1 = min(center_x + patch_size // 2, W - spacing - 1)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for i in range(y0, y1, spacing):
        for j in range(x0, x1, spacing):
            corners = [(j, i), (j + spacing, i), (j + spacing, i + spacing), (j, i + spacing)]
            deformed = [(x + dx[y, x], y + dy[y, x]) for x, y in corners]

            ec = "yellow" if J[i, j] < 0 else "black"
            lw = 2.0 if J[i, j] < 0 else 0.5
            poly = Polygon(
                deformed, closed=True, edgecolor=ec, facecolor=colormap(norm(J[i, j])), linewidth=lw
            )
            ax.add_patch(poly)

    ax.set_xlim(x0, x1 + spacing)
    ax.set_ylim(y1 + spacing, y0)
    ax.set_title("Deformed Mesh (coloured by Jacobian)")
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    # Use the fig+ax form (rather than plt.colorbar without ax) so the
    # call is compatible with the theme's constrained_layout=True.
    fig.colorbar(sm, ax=ax, label="Jacobian Determinant")


# ---------------------------------------------------------------------------
# Inverse displacement for push-forward view
# ---------------------------------------------------------------------------
def _invert_displacement(dy, dx, iterations=50):
    """Compute inverse displacement field via fixed-point iteration.

    Given a pull-back displacement field *d* where ``T(x) = x + d(x)``
    is the forward warp, compute ``d_inv`` where
    ``T^{-1}(y) = y + d_inv(y)`` is the push-forward (inverse) warp.
    """
    H, W = dy.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    inv_dy = -dy.copy()
    inv_dx = -dx.copy()

    for _ in range(iterations):
        sample_y = np.clip(yy + inv_dy, 0, H - 1)
        sample_x = np.clip(xx + inv_dx, 0, W - 1)

        sampled_dy = map_coordinates(dy, [sample_y, sample_x], order=1, mode='nearest')
        sampled_dx = map_coordinates(dx, [sample_y, sample_x], order=1, mode='nearest')

        inv_dy = -sampled_dy
        inv_dx = -sampled_dx

    return inv_dy, inv_dx


# ---------------------------------------------------------------------------
# Single deformation grid (Jdet-coloured)
# ---------------------------------------------------------------------------
def plot_grid(
    deformation,
    title="",
    figsize=(7, 6),
    spacing=1,
    linewidth=0.5,
    inverse=False,
    jdet_vmax=None,
    jac_override=None,
    ax=None,
):
    """Plot a single deformation grid coloured by Jacobian determinant.

    Single-panel counterpart to :func:`plot_grid_before_after` — same
    rendering (yellow outlines on negative-Jdet cells, wireframe overlay),
    no comparison.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, 1, H, W)``
    title : str
    figsize : tuple
    spacing : int
    linewidth : float
    inverse : bool
        If False (default), show the pull-back displacement field.
        If True, show the push-forward (inverse) field via fixed-point
        iteration.
    jdet_vmax : float or None
        If set, caps the colormap range to ``[-jdet_vmax, jdet_vmax]``.
    jac_override : ndarray, shape ``(H, W)``, optional
        Pre-computed Jacobian determinant map to use for colouring instead
        of recomputing from the deformation. Useful when visualising a
        z-slice of a 3D field (pass the relevant slice of the full 3D
        Jdet). Ignored when ``inverse=True``.
    ax : matplotlib Axes, optional
        If given, draw into this axes instead of creating a new figure.
    """
    _, _, H, W = deformation.shape
    phi_init = np.stack([deformation[1, 0], deformation[2, 0]])  # [dy, dx]

    if inverse:
        inv_dy, inv_dx = _invert_displacement(phi_init[0], phi_init[1])
        phi_view = np.stack([inv_dy, inv_dx])

        jac_fwd = np.squeeze(jacobian_det2D(phi_init))
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        preimg_y = np.clip(yy + inv_dy, 0, H - 1)
        preimg_x = np.clip(xx + inv_dx, 0, W - 1)
        jac_fwd_at_preimg = map_coordinates(jac_fwd, [preimg_y, preimg_x], order=1, mode='nearest')
        with np.errstate(divide='ignore', invalid='ignore'):
            jac = np.where(np.abs(jac_fwd_at_preimg) > 1e-10, 1.0 / jac_fwd_at_preimg, 0.0)
    else:
        phi_view = phi_init
        if jac_override is not None:
            jac = np.asarray(jac_override)
        else:
            jac = np.squeeze(jacobian_det2D(phi_init))

    vmin = min(float(jac.min()), -0.5)
    vmax = max(float(jac.max()), 1.5)
    if jdet_vmax is not None:
        vmin = max(vmin, -jdet_vmax)
        vmax = min(vmax, jdet_vmax)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("bwr")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    direction = "push-forward" if inverse else "pull-back"
    neg = int((jac <= 0).sum())
    label = f"{direction} field (neg Jdet = {neg})"

    dy = phi_view[0]
    dx = phi_view[1]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    def_x = xx + dx
    def_y = yy + dy

    _draw_jdet_cell_polygons(
        ax, def_x, def_y, jac, cmap=cmap, norm=norm, spacing=spacing, linewidth=linewidth
    )

    row_indices = list(range(0, H, spacing))
    if (H - 1) not in row_indices:
        row_indices.append(H - 1)
    col_indices = list(range(0, W, spacing))
    if (W - 1) not in col_indices:
        col_indices.append(W - 1)

    for i in row_indices:
        ax.plot(def_x[i, col_indices], def_y[i, col_indices], 'k-', linewidth=linewidth, zorder=1)
    for j in col_indices:
        ax.plot(def_x[row_indices, j], def_y[row_indices, j], 'k-', linewidth=linewidth, zorder=1)

    pad = max(W, H) * 0.03
    ax.set_xlim(def_x.min() - pad, def_x.max() + pad)
    ax.set_ylim(def_y.max() + pad, def_y.min() - pad)
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=11)

    if standalone:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Jacobian determinant", shrink=0.85)
        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        plt.show()


# ---------------------------------------------------------------------------
# Before / after deformation grid comparison
# ---------------------------------------------------------------------------
def plot_grid_before_after(
    deformation_i,
    phi_corrected,
    figsize=(14, 6),
    title="",
    spacing=1,
    linewidth=0.5,
    inverse=False,
    jdet_vmax=None,
    jac_init_override=None,
    jac_corr_override=None,
):
    """Side-by-side deformation grids coloured by Jacobian determinant.

    Parameters
    ----------
    deformation_i : ndarray, shape ``(3, 1, H, W)``
    phi_corrected : ndarray, shape ``(2, H, W)``
    figsize : tuple
    title : str
    spacing : int
    linewidth : float
    inverse : bool
        If False (default), show the pull-back displacement field.
        If True, show the push-forward (inverse) field via fixed-point
        iteration.
    jdet_vmax : float or None
        If set, caps the colormap range to ``[-jdet_vmax, jdet_vmax]``.
        Useful when extreme outliers wash out the colour scale.
    jac_init_override, jac_corr_override : ndarray, shape ``(H, W)``, optional
        Pre-computed Jacobian determinant maps to use for colouring instead
        of recomputing from ``phi``. Essential when visualising a z-slice
        of a 3D field: the 2D Jdet of the in-plane (dy, dx) stack does
        **not** match the 3D Jdet the solver enforces, and can be negative
        even when the 3D field is valid. Pass the relevant slice of the
        full 3D Jdet here to get a faithful overlay. Ignored when
        ``inverse=True``.
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
            jac_fwd_init, [preimg_y, preimg_x], order=1, mode='nearest'
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            jac_init = np.where(np.abs(jac_fwd_at_preimg) > 1e-10, 1.0 / jac_fwd_at_preimg, 0.0)

        preimg_y = np.clip(yy + inv_corr_dy, 0, H - 1)
        preimg_x = np.clip(xx + inv_corr_dx, 0, W - 1)
        jac_fwd_at_preimg = map_coordinates(
            jac_fwd_corr, [preimg_y, preimg_x], order=1, mode='nearest'
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            jac_corr = np.where(np.abs(jac_fwd_at_preimg) > 1e-10, 1.0 / jac_fwd_at_preimg, 0.0)
    else:
        phi_init_view = phi_init
        phi_corr_view = phi_corrected
        if jac_init_override is not None:
            jac_init = np.asarray(jac_init_override)
        else:
            jac_init = np.squeeze(jacobian_det2D(phi_init))
        if jac_corr_override is not None:
            jac_corr = np.asarray(jac_corr_override)
        else:
            jac_corr = np.squeeze(jacobian_det2D(phi_corrected))

    # Shared colour normalisation across both panels
    J_all = np.concatenate([jac_init.ravel(), jac_corr.ravel()])
    vmin = min(float(J_all.min()), -0.5)
    vmax = max(float(J_all.max()), 1.5)
    if jdet_vmax is not None:
        vmin = max(vmin, -jdet_vmax)
        vmax = min(vmax, jdet_vmax)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("bwr")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    init_neg = int((jac_init <= 0).sum())
    corr_neg = int((jac_corr <= 0).sum())

    direction = "push-forward" if inverse else "pull-back"

    for ax, phi, jac, label in [
        (axes[0], phi_init_view, jac_init, f"Initial — {direction} field (neg Jdet = {init_neg})"),
        (
            axes[1],
            phi_corr_view,
            jac_corr,
            f"Corrected — {direction} field (neg Jdet = {corr_neg})",
        ),
    ]:
        dy = phi[0]
        dx = phi[1]

        # Deformed vertex positions for the full grid
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        def_x = xx + dx
        def_y = yy + dy

        # Layer 1: light Jdet-coloured cell fills
        _draw_jdet_cell_polygons(
            ax, def_x, def_y, jac, cmap=cmap, norm=norm, spacing=spacing, linewidth=linewidth
        )

        # Layer 2: wireframe grid lines
        row_indices = list(range(0, H, spacing))
        if (H - 1) not in row_indices:
            row_indices.append(H - 1)
        col_indices = list(range(0, W, spacing))
        if (W - 1) not in col_indices:
            col_indices.append(W - 1)

        for i in row_indices:
            ax.plot(
                def_x[i, col_indices], def_y[i, col_indices], 'k-', linewidth=linewidth, zorder=1
            )
        for j in col_indices:
            ax.plot(
                def_x[row_indices, j], def_y[row_indices, j], 'k-', linewidth=linewidth, zorder=1
            )

        pad = max(W, H) * 0.03
        ax.set_xlim(def_x.min() - pad, def_x.max() + pad)
        ax.set_ylim(def_y.max() + pad, def_y.min() - pad)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Jacobian determinant", shrink=0.85)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.show()
