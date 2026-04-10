"""3D deformation field and Jacobian determinant visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dvfopt.jacobian import jacobian_det3D


# ---------------------------------------------------------------------------
# Slice-based views
# ---------------------------------------------------------------------------

def plot_jdet_slices(jdet_before, jdet_after, title=None):
    """Before/after Jdet heatmaps for every z-slice.

    Parameters
    ----------
    jdet_before, jdet_after : ndarray, shape (D, H, W)
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    D = jdet_before.shape[0]
    fig, axes = plt.subplots(2, D, figsize=(3 * D, 6))
    if D == 1:
        axes = axes[:, np.newaxis]

    all_vals = np.concatenate([jdet_before.ravel(), jdet_after.ravel()])
    vmin = min(float(all_vals.min()), -0.01)
    vmax = max(float(all_vals.max()), 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    for z in range(D):
        for row, (jdet, label) in enumerate(
            [(jdet_before, "Before"), (jdet_after, "After")]
        ):
            ax = axes[row, z]
            im = ax.imshow(jdet[z], cmap="RdBu_r", norm=norm, origin="upper")
            if (jdet[z] <= 0).any():
                ax.contour(jdet[z], levels=[0], colors="black", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if z == 0:
                ax.set_ylabel(label, fontsize=11)
            if row == 0:
                ax.set_title(f"z={z}", fontsize=10)

    fig.colorbar(im, ax=axes, label="Jacobian determinant", shrink=0.7)
    if title:
        plt.suptitle(title, fontsize=13)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    return fig


# ---------------------------------------------------------------------------
# 3-D Jacobian determinant scatter
# ---------------------------------------------------------------------------

def plot_jdet_3d(jdet, title=None, ax=None, elev=25, azim=-60):
    """3D scatter coloured by Jacobian determinant.

    Negative-Jdet voxels are shown larger and opaque; positive ones are
    small and semi-transparent.

    Parameters
    ----------
    jdet : ndarray, shape (D, H, W)
    title : str, optional
    ax : Axes3D, optional
        If provided, draws into this axes instead of creating a new figure.
    elev, azim : float
        Viewing angle.

    Returns
    -------
    fig or ax
    """
    D, H, W = jdet.shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    z_flat = zz.ravel()
    y_flat = yy.ravel()
    x_flat = xx.ravel()
    j_flat = jdet.ravel()

    vmin = min(float(j_flat.min()), -0.01)
    vmax = max(float(j_flat.max()), 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    neg = j_flat <= 0
    pos = ~neg

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    if pos.any():
        ax.scatter(
            x_flat[pos], y_flat[pos], z_flat[pos],
            c=j_flat[pos], cmap="RdBu_r", norm=norm,
            s=15, alpha=0.15, edgecolors="none", depthshade=True,
        )
    if neg.any():
        ax.scatter(
            x_flat[neg], y_flat[neg], z_flat[neg],
            c=j_flat[neg], cmap="RdBu_r", norm=norm,
            s=120, alpha=0.9, edgecolors="black", linewidth=0.5,
            depthshade=False,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11)

    if own_fig:
        return fig
    return ax


def plot_jdet_3d_before_after(jdet_before, jdet_after, title=None,
                              elev=25, azim=-60):
    """Side-by-side 3D Jdet scatter — before vs after correction.

    Parameters
    ----------
    jdet_before, jdet_after : ndarray, shape (D, H, W)

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(14, 6))
    n_neg_b = int((jdet_before <= 0).sum())
    n_neg_a = int((jdet_after <= 0).sum())

    ax1 = fig.add_subplot(121, projection="3d")
    plot_jdet_3d(jdet_before, title=f"Before — {n_neg_b} neg voxels",
                 ax=ax1, elev=elev, azim=azim)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_jdet_3d(jdet_after, title=f"After — {n_neg_a} neg voxels",
                 ax=ax2, elev=elev, azim=azim)

    if title:
        plt.suptitle(title, fontsize=13)
    fig.subplots_adjust(wspace=0.05)
    return fig


# ---------------------------------------------------------------------------
# 3-D negative-voxel highlighting
# ---------------------------------------------------------------------------

def plot_neg_voxels_before_after(jdet_before, jdet_after, title=None,
                                 elev=25, azim=-60):
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
    fig = plt.figure(figsize=(14, 6))

    for idx, (jdet, label) in enumerate(
        [(jdet_before, "Before"), (jdet_after, "After")]
    ):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        D, H, W = jdet.shape
        neg_mask = jdet <= 0
        n_neg = int(neg_mask.sum())

        if neg_mask.any():
            # Build RGBA facecolors — transpose (D,H,W) → (W,H,D) for voxels()
            neg_t = neg_mask.transpose(2, 1, 0)       # (W, H, D)
            jdet_t = jdet.transpose(2, 1, 0)           # (W, H, D)
            colors = np.zeros((*neg_t.shape, 4))

            neg_vals = jdet_t[neg_t]
            worst = min(float(neg_vals.min()), -1e-10)
            alpha = np.clip(
                0.4 + 0.6 * np.abs(neg_vals) / abs(worst), 0.4, 1.0
            )
            colors[neg_t, 0] = 0.02   # R
            colors[neg_t, 1] = 0.19   # G
            colors[neg_t, 2] = 0.38   # B  (matches RdBu_r at vmin)
            colors[neg_t, 3] = alpha   # A

            ax.voxels(neg_t, facecolors=colors, edgecolor="black",
                      linewidth=0.2)

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
    fig.subplots_adjust(wspace=0.05)
    return fig


# ---------------------------------------------------------------------------
# 3-D deformation grid
# ---------------------------------------------------------------------------

def plot_deformation_grid_3d(phi, jdet=None, title=None, ax=None,
                             spacing=1, elev=25, azim=-60):
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
    cmap = plt.get_cmap("RdBu_r")

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    nz, ny, nx = zg.shape
    segments = []
    colors = []
    linewidths = []

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                p = (vx[iz, iy, ix], vy[iz, iy, ix], vz[iz, iy, ix])
                j_val = jdet[zi[iz], yi[iy], xi[ix]]
                c = cmap(norm(j_val))
                lw = 1.8 if j_val <= 0 else 0.4

                if ix + 1 < nx:
                    p2 = (vx[iz, iy, ix + 1], vy[iz, iy, ix + 1],
                           vz[iz, iy, ix + 1])
                    segments.append([p, p2])
                    colors.append(c)
                    linewidths.append(lw)

                if iy + 1 < ny:
                    p2 = (vx[iz, iy + 1, ix], vy[iz, iy + 1, ix],
                           vz[iz, iy + 1, ix])
                    segments.append([p, p2])
                    colors.append(c)
                    linewidths.append(lw)

                if iz + 1 < nz:
                    p2 = (vx[iz + 1, iy, ix], vy[iz + 1, iy, ix],
                           vz[iz + 1, iy, ix])
                    segments.append([p, p2])
                    colors.append(c)
                    linewidths.append(lw)

    if segments:
        lc = Line3DCollection(segments, colors=colors, linewidths=linewidths)
        ax.add_collection3d(lc)

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


def plot_grid_before_after_3d(phi_before, phi_after, jdet_before=None,
                              jdet_after=None, title=None, spacing=1,
                              elev=25, azim=-60):
    """Side-by-side 3D deformation grids — before vs after correction.

    Parameters
    ----------
    phi_before, phi_after : ndarray, shape (3, D, H, W)
    jdet_before, jdet_after : ndarray, shape (D, H, W), optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if jdet_before is None:
        jdet_before = jacobian_det3D(phi_before)
    if jdet_after is None:
        jdet_after = jacobian_det3D(phi_after)

    n_neg_b = int((jdet_before <= 0).sum())
    n_neg_a = int((jdet_after <= 0).sum())

    fig = plt.figure(figsize=(15, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_deformation_grid_3d(
        phi_before, jdet_before,
        title=f"Before — {n_neg_b} neg voxels",
        ax=ax1, spacing=spacing, elev=elev, azim=azim,
    )

    ax2 = fig.add_subplot(122, projection="3d")
    plot_deformation_grid_3d(
        phi_after, jdet_after,
        title=f"After — {n_neg_a} neg voxels",
        ax=ax2, spacing=spacing, elev=elev, azim=azim,
    )

    if title:
        plt.suptitle(title, fontsize=13)
    fig.subplots_adjust(wspace=0.05)
    return fig
