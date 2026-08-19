"""3D deformation field and Jacobian determinant visualizations."""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dvfopt.jacobian import jacobian_det3D
from dvfopt.viz.theme import PALETTE, apply_theme

# ---------------------------------------------------------------------------
# Slice-based views
# ---------------------------------------------------------------------------

# Curated palette for connected-component outlines.
# Mixes the theme's categorical colors with one bright accent for contrast
# against the RdBu_r colormap. Stays visually distinct on both blue & red.
_COMPONENT_COLORS = (
    PALETTE.orange,
    PALETTE.green,
    PALETTE.purple,
    PALETTE.brown,
    "#FFAB00",  # amber — extra accent
    "#18FFFF",  # cyan — extra accent
)


def _outline_negative_cells(ax, neg_mask, linewidth=1.8, halo_color="black", halo_width=3.6):
    """Draw cell-edge outlines around each connected negative-Jdet region.

    Each 4-connected component gets its own colour from a short palette
    so neighbouring components are easy to tell apart when their outlines
    are close to each other. A dark halo keeps the stroke legible on both
    the red and blue ends of the ``RdBu_r`` heatmap. Boundary edges are
    drawn per-cell (no interpolation) so the outline accurately marks
    which cells are negative, unlike ``contour(levels=[0])``.
    """
    if not neg_mask.any():
        return
    from scipy.ndimage import label

    labels, n_comp = label(neg_mask.astype(np.uint8))  # 4-connectivity

    for k in range(1, n_comp + 1):
        comp = labels == k
        H, W = comp.shape
        segments = []
        for y in range(H + 1):
            top = comp[y - 1] if y > 0 else np.zeros(W, dtype=bool)
            bot = comp[y] if y < H else np.zeros(W, dtype=bool)
            diff = top ^ bot
            for x in np.nonzero(diff)[0]:
                segments.append(((x - 0.5, y - 0.5), (x + 0.5, y - 0.5)))
        for x in range(W + 1):
            left = comp[:, x - 1] if x > 0 else np.zeros(H, dtype=bool)
            right = comp[:, x] if x < W else np.zeros(H, dtype=bool)
            diff = left ^ right
            for y in np.nonzero(diff)[0]:
                segments.append(((x - 0.5, y - 0.5), (x - 0.5, y + 0.5)))
        if not segments:
            continue
        color = _COMPONENT_COLORS[(k - 1) % len(_COMPONENT_COLORS)]
        lc = LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            capstyle="butt",
            joinstyle="miter",
            zorder=5 + k,
        )
        lc.set_path_effects(
            [
                path_effects.Stroke(linewidth=halo_width, foreground=halo_color),
                path_effects.Normal(),
            ]
        )
        ax.add_collection(lc)


def plot_jdet_slices(jdet_before, jdet_after, title=None, max_slices=None):
    """Before/after Jdet heatmaps for selected z-slices.

    Parameters
    ----------
    jdet_before, jdet_after : ndarray, shape (D, H, W)
    title : str, optional
    max_slices : int, optional
        If given and ``D > max_slices``, only the ``max_slices`` z-slices
        with the most negative-Jdet voxels in ``jdet_before`` are shown
        (sorted by z). Keeps figure legible on deep volumes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_theme()
    D = jdet_before.shape[0]
    if max_slices is not None and max_slices < D:
        neg_per_z = (jdet_before <= 0).sum(axis=(1, 2))
        z_indices = np.sort(np.argsort(neg_per_z)[-max_slices:])
    else:
        z_indices = np.arange(D)
    n = len(z_indices)

    fig, axes = plt.subplots(2, n, figsize=(3 * n + 1, 6), constrained_layout=True)
    if n == 1:
        axes = axes[:, np.newaxis]

    all_vals = np.concatenate([jdet_before.ravel(), jdet_after.ravel()])
    vmin = min(float(all_vals.min()), -0.01)
    vmax = max(float(all_vals.max()), 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    for col, z in enumerate(z_indices):
        for row, (jdet, label) in enumerate([(jdet_before, "Before"), (jdet_after, "After")]):
            ax = axes[row, col]
            im = ax.imshow(jdet[z], cmap=PALETTE.cmap_jdet, norm=norm, origin="upper")
            _outline_negative_cells(ax, jdet[z] <= 0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
            if row == 0:
                ax.set_title(f"z={z}", fontsize=10)

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        location="right",
        label="Jacobian determinant",
        shrink=0.7,
        pad=0.02,
    )
    if title:
        fig.suptitle(title, fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# 3-D Jacobian determinant scatter
# ---------------------------------------------------------------------------


def plot_jdet_3d(
    jdet, title=None, ax=None, elev=25, azim=-60, max_positive_points=5000, show_bbox=True
):
    """3D scatter coloured by Jacobian determinant.

    Negative-Jdet voxels are plotted as large opaque markers with black
    edges. Positive voxels form a faint cloud for spatial context and are
    randomly subsampled to ``max_positive_points`` so dense volumes don't
    wash out the negatives. The colormap is clamped symmetrically around
    zero so positives can't drive the blue channel into invisibility on
    volumes with large outliers.

    Parameters
    ----------
    jdet : ndarray, shape (D, H, W)
    title : str, optional
    ax : Axes3D, optional
        If provided, draws into this axes instead of creating a new figure.
    elev, azim : float
        Viewing angle.
    max_positive_points : int
        Cap on number of positive voxels drawn (random subsample above).
        Set to 0 to hide positives entirely.
    show_bbox : bool
        Whether to draw a faint bounding-box wireframe for spatial reference
        (helps orient sparse negative clusters on large volumes).

    Returns
    -------
    fig or ax
    """
    apply_theme()
    D, H, W = jdet.shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    z_flat = zz.ravel()
    y_flat = yy.ravel()
    x_flat = xx.ravel()
    j_flat = jdet.ravel()

    neg = j_flat <= 0
    pos = ~neg

    # Symmetric colormap around zero so a large positive max doesn't
    # compress the negative colour range to nothing.
    j_abs_max = max(float(np.abs(j_flat).max()), 0.01)
    neg_max = float(j_flat[neg].min()) if neg.any() else -j_abs_max
    sym = max(abs(neg_max), 0.5)
    norm = TwoSlopeNorm(vmin=-sym, vcenter=0, vmax=sym)

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

    if pos.any() and max_positive_points > 0:
        pos_idx = np.flatnonzero(pos)
        if pos_idx.size > max_positive_points:
            rng = np.random.default_rng(0)
            pos_idx = rng.choice(pos_idx, size=max_positive_points, replace=False)
        ax.scatter(
            x_flat[pos_idx],
            y_flat[pos_idx],
            z_flat[pos_idx],
            c=j_flat[pos_idx],
            cmap=PALETTE.cmap_jdet,
            norm=norm,
            s=8,
            alpha=0.08,
            edgecolors="none",
            depthshade=True,
        )
    if neg.any():
        ax.scatter(
            x_flat[neg],
            y_flat[neg],
            z_flat[neg],
            c=j_flat[neg],
            cmap=PALETTE.cmap_jdet,
            norm=norm,
            s=140,
            alpha=0.95,
            edgecolors="black",
            linewidth=0.6,
            depthshade=False,
        )

    if show_bbox:
        # Faint wireframe cube for spatial reference
        corners = [
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
            xs = [corners[a][0], corners[b][0]]
            ys = [corners[a][1], corners[b][1]]
            zs = [corners[a][2], corners[b][2]]
            ax.plot(xs, ys, zs, color="gray", linewidth=0.5, alpha=0.4)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, D)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11)

    if own_fig:
        return fig
    return ax


def plot_jdet_3d_before_after(jdet_before, jdet_after, title=None, elev=25, azim=-60):
    """Side-by-side 3D Jdet scatter — before vs after correction.

    Parameters
    ----------
    jdet_before, jdet_after : ndarray, shape (D, H, W)

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_theme()
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    n_neg_b = int((jdet_before <= 0).sum())
    n_neg_a = int((jdet_after <= 0).sum())

    ax1 = fig.add_subplot(121, projection="3d")
    plot_jdet_3d(jdet_before, title=f"Before — {n_neg_b} neg voxels", ax=ax1, elev=elev, azim=azim)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_jdet_3d(jdet_after, title=f"After — {n_neg_a} neg voxels", ax=ax2, elev=elev, azim=azim)

    if title:
        plt.suptitle(title, fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# 3-D negative-voxel highlighting
# ---------------------------------------------------------------------------


def plot_neg_voxels_before_after(jdet_before, jdet_after, title=None, elev=25, azim=-60):
    """Side-by-side 3D voxel plots of negative-Jdet regions.

    Solid blue blocks mark voxels with Jdet <= 0.  Intensity scales with
    the magnitude of the negative Jdet.  Blue matches the `RdBu_r`
    colormap used by the other 3D Jdet views.

    Parameters
    ----------
    jdet_before, jdet_after : ndarray, shape (D, H, W)

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_theme()
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)

    for idx, (jdet, label) in enumerate([(jdet_before, "Before"), (jdet_after, "After")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        D, H, W = jdet.shape
        neg_mask = jdet <= 0
        n_neg = int(neg_mask.sum())

        if neg_mask.any():
            # Build RGBA facecolors — transpose (D,H,W) → (W,H,D) for voxels()
            neg_t = neg_mask.transpose(2, 1, 0)  # (W, H, D)
            jdet_t = jdet.transpose(2, 1, 0)  # (W, H, D)
            colors = np.zeros((*neg_t.shape, 4))

            neg_vals = jdet_t[neg_t]
            worst = min(float(neg_vals.min()), -1e-10)
            alpha = np.clip(0.4 + 0.6 * np.abs(neg_vals) / abs(worst), 0.4, 1.0)
            colors[neg_t, 0] = 0.02  # R
            colors[neg_t, 1] = 0.19  # G
            colors[neg_t, 2] = 0.38  # B  (matches RdBu_r at vmin)
            colors[neg_t, 3] = alpha  # A

            ax.voxels(neg_t, facecolors=colors, edgecolor="black", linewidth=0.2)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(0, D)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{label} — {n_neg} neg voxels", fontsize=11)

    if title:
        plt.suptitle(title, fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# 3-D deformation grid
# ---------------------------------------------------------------------------


def plot_deformation_grid_3d(phi, jdet=None, title=None, ax=None, spacing=1, elev=25, azim=-60):
    """3D wireframe of a deformed grid coloured by Jacobian determinant.

    Grid edges near negative-Jdet voxels are drawn thicker and red.

    Parameters
    ----------
    phi : ndarray, shape (3, D, H, W) — [dz, dy, dx]
    jdet : ndarray, shape (D, H, W), optional
        Pre-computed Jacobian determinant; computed if not provided.
    spacing : int
        Draw every *spacing*-th grid line (reduces clutter for large grids).
    elev, azim : float
        Viewing angle.

    Returns
    -------
    fig or ax
    """
    apply_theme()
    D, H, W = phi.shape[1], phi.shape[2], phi.shape[3]
    dz_f = phi[0]
    dy_f = phi[1]
    dx_f = phi[2]

    if jdet is None:
        jdet = jacobian_det3D(phi)

    # Sparse vertex indices (always include last)
    def _sparse_idx(n, s):
        idx = list(range(0, n, s))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        return idx

    zi = _sparse_idx(D, spacing)
    yi = _sparse_idx(H, spacing)
    xi = _sparse_idx(W, spacing)

    # Build deformed vertex positions  (nz, ny, nx)
    zg, yg, xg = np.meshgrid(zi, yi, xi, indexing="ij")
    vz = zg.astype(float) + dz_f[zg, yg, xg]
    vy = yg.astype(float) + dy_f[zg, yg, xg]
    vx = xg.astype(float) + dx_f[zg, yg, xg]

    # Colour mapping
    j_flat = jdet.ravel()
    vmin = min(float(j_flat.min()), -0.01)
    vmax = max(float(j_flat.max()), 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap(PALETTE.cmap_jdet)

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

    # For each sampled lattice vertex (zi[iz], yi[iy], xi[ix]), compute the
    # min Jdet in the enclosing block up to the next sampled vertex. Without
    # this, a sparse lattice (spacing >> 1) only inspects isolated voxels and
    # misses every negative that doesn't happen to land on a sampled point,
    # leaving the wireframe looking empty on large volumes.
    nz, ny, nx = zg.shape
    block_min = np.empty((nz, ny, nx), dtype=jdet.dtype)
    for iz in range(nz):
        z1 = zi[iz + 1] if iz + 1 < nz else D
        for iy in range(ny):
            y1 = yi[iy + 1] if iy + 1 < ny else H
            for ix in range(nx):
                x1 = xi[ix + 1] if ix + 1 < nx else W
                block_min[iz, iy, ix] = jdet[
                    zi[iz] : max(z1, zi[iz] + 1),
                    yi[iy] : max(y1, yi[iy] + 1),
                    xi[ix] : max(x1, xi[ix] + 1),
                ].min()

    segments = []
    colors = []
    linewidths = []

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                p = (vx[iz, iy, ix], vy[iz, iy, ix], vz[iz, iy, ix])
                j_val = block_min[iz, iy, ix]
                c = cmap(norm(j_val))
                lw = 2.2 if j_val <= 0 else 0.4

                if ix + 1 < nx:
                    p2 = (vx[iz, iy, ix + 1], vy[iz, iy, ix + 1], vz[iz, iy, ix + 1])
                    segments.append([p, p2])
                    colors.append(c)
                    linewidths.append(lw)

                if iy + 1 < ny:
                    p2 = (vx[iz, iy + 1, ix], vy[iz, iy + 1, ix], vz[iz, iy + 1, ix])
                    segments.append([p, p2])
                    colors.append(c)
                    linewidths.append(lw)

                if iz + 1 < nz:
                    p2 = (vx[iz + 1, iy, ix], vy[iz + 1, iy, ix], vz[iz + 1, iy, ix])
                    segments.append([p, p2])
                    colors.append(c)
                    linewidths.append(lw)

    if segments:
        lc = Line3DCollection(segments, colors=colors, linewidths=linewidths)
        ax.add_collection3d(lc)

    # Overlay negative voxels at their deformed positions so they remain
    # visible even when the sparse wireframe sampling misses them.
    neg_mask = jdet <= 0
    if neg_mask.any():
        nz_idx, ny_idx, nx_idx = np.nonzero(neg_mask)
        def_x = nx_idx + dx_f[neg_mask]
        def_y = ny_idx + dy_f[neg_mask]
        def_z = nz_idx + dz_f[neg_mask]
        ax.scatter(
            def_x,
            def_y,
            def_z,
            c=jdet[neg_mask],
            cmap=cmap,
            norm=norm,
            s=80,
            alpha=0.95,
            edgecolors="black",
            linewidth=0.5,
            depthshade=False,
            zorder=10,
        )

    ax.set_xlim(float(vx.min()) - 0.5, float(vx.max()) + 0.5)
    ax.set_ylim(float(vy.min()) - 0.5, float(vy.max()) + 0.5)
    ax.set_zlim(float(vz.min()) - 0.5, float(vz.max()) + 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11)

    if own_fig:
        return fig
    return ax


def plot_grid_before_after_3d(
    phi_before,
    phi_after,
    jdet_before=None,
    jdet_after=None,
    title=None,
    spacing=1,
    elev=25,
    azim=-60,
):
    """Side-by-side 3D deformation grids — before vs after correction.

    Parameters
    ----------
    phi_before, phi_after : ndarray, shape (3, D, H, W)
    jdet_before, jdet_after : ndarray, shape (D, H, W), optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_theme()
    if jdet_before is None:
        jdet_before = jacobian_det3D(phi_before)
    if jdet_after is None:
        jdet_after = jacobian_det3D(phi_after)

    n_neg_b = int((jdet_before <= 0).sum())
    n_neg_a = int((jdet_after <= 0).sum())

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)

    ax1 = fig.add_subplot(121, projection="3d")
    plot_deformation_grid_3d(
        phi_before,
        jdet_before,
        title=f"Before — {n_neg_b} neg voxels",
        ax=ax1,
        spacing=spacing,
        elev=elev,
        azim=azim,
    )

    ax2 = fig.add_subplot(122, projection="3d")
    plot_deformation_grid_3d(
        phi_after,
        jdet_after,
        title=f"After — {n_neg_a} neg voxels",
        ax=ax2,
        spacing=spacing,
        elev=elev,
        azim=azim,
    )

    if title:
        plt.suptitle(title, fontsize=13)
    return fig
