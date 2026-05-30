"""Debug visualisations for the per-pixel two-triangle sign check.

The main entry points are:

- :func:`plot_triangle_debug` — single-pixel deep-dive showing the warped
  neighborhood, both T1/T2 triangles filled by sign, vertex coordinates,
  the cross-product formula, and the numeric inputs and result.
- :func:`plot_problematic_triangles` — finds all cells with at least one
  flipped or collapsed triangle and renders a grid of per-pixel plots.
  Designed to be called as a debug toggle inside an iterative solver:

      if debug:
          plot_problematic_triangles(phi_xy, title=f"iter {k}")
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from dvfopt.jacobian.triangle_sign import triangle_sign_det2D

# Triangle vertex templates (reference-grid offsets), mirroring the
# whiteboard convention: T1 at pixel (x, y) uses (x, y), (x-1, y+1),
# (x, y+1); T2 uses (x, y), (x, y+1), (x+1, y).
_T1_OFFSETS = [(0, 0), (-1, 1), (0, 1)]
_T2_OFFSETS = [(0, 0), (0, 1), (1, 0)]
_T1_LABELS = ["(x, y)", "(x-1, y+1)", "(x, y+1)"]
_T2_LABELS = ["(x, y)", "(x, y+1)", "(x+1, y)"]


def _fill_color(area, threshold=0.0):
    """Colors mirror the workspace ``RdBu_r`` convention: red = positive
    (valid), blue = negative (flip), gray at the collapse boundary.
    """
    if area < -threshold:
        return "#64b5f6"  # soft blue — flip
    if area <= threshold:
        return "#bdbdbd"  # gray — collapse
    return "#ef5350"  # soft red — valid


def _triangle_vertices(x, y, offsets, dy, dx):
    """Return list of (warped_x, warped_y) for a triangle at pixel (x, y)."""
    H, W = dy.shape
    out = []
    for ox, oy in offsets:
        px, py = x + ox, y + oy
        if not (0 <= px < W and 0 <= py < H):
            return None  # triangle goes out of bounds
        out.append((px + dx[py, px], py + dy[py, px]))
    return out


def _triangle_raw_cross(verts):
    """Raw 2D cross product AB.x*AC.y - AB.y*AC.x for vertices A, B, C."""
    (ax, ay), (bx, by), (cx, cy) = verts
    ab_x, ab_y = bx - ax, by - ay
    ac_x, ac_y = cx - ax, cy - ay
    cross = ab_x * ac_y - ab_y * ac_x
    return (ab_x, ab_y), (ac_x, ac_y), cross


def _plot_neighborhood(ax, x, y, dy, dx, radius=2):
    """Draw reference grid (gray) and warped grid (blue) around pixel (x, y)."""
    H, W = dy.shape
    x0, x1 = max(0, x - radius), min(W - 1, x + radius)
    y0, y1 = max(0, y - radius), min(H - 1, y + radius)

    # Reference grid (lattice)
    for yi in range(y0, y1 + 1):
        for xi in range(x0, x1 + 1):
            ax.plot(xi, yi, ".", color="#cfcfcf", markersize=4, zorder=1)
    for yi in range(y0, y1 + 1):
        ax.plot([x0, x1], [yi, yi], color="#e8e8e8", lw=0.5, zorder=0)
    for xi in range(x0, x1 + 1):
        ax.plot([xi, xi], [y0, y1], color="#e8e8e8", lw=0.5, zorder=0)

    # Warped grid: connect warped neighbors with thin blue lines
    for yi in range(y0, y1 + 1):
        pts = [(xi + dx[yi, xi], yi + dy[yi, xi]) for xi in range(x0, x1 + 1)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#5b7fb5", lw=0.6, zorder=2)
    for xi in range(x0, x1 + 1):
        pts = [(xi + dx[yi, xi], yi + dy[yi, xi]) for yi in range(y0, y1 + 1)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#5b7fb5", lw=0.6, zorder=2)


def _format_inputs(x, y, offsets, labels, dy, dx):
    """Return a multi-line string listing each vertex's (ref, δ, ε, warped)."""
    lines = []
    H, W = dy.shape
    for (ox, oy), label in zip(offsets, labels):
        px, py = x + ox, y + oy
        if not (0 <= px < W and 0 <= py < H):
            lines.append(f"  {label:<12s} OUT OF BOUNDS")
            continue
        dxv = dx[py, px]
        dyv = dy[py, px]
        wx = px + dxv
        wy = py + dyv
        lines.append(
            f"  {label:<12s} ref=({px},{py})  δ={dxv:+.3f}  "
            f"ε={dyv:+.3f}  warped=({wx:+.3f}, {wy:+.3f})"
        )
    return "\n".join(lines)


def plot_triangle_debug(phi_xy, x, y, ax=None, show_formula=True):
    """Plot T1 and T2 at pixel (x, y) with vertex positions, formula, and area.

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)``
        Channels ``[dy, dx]``.
    x, y : int
        Pixel coordinates (1 <= x <= W-2 and 0 <= y <= H-2 for both triangles
        to be in-bounds).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes (e.g. from a subplot grid). If ``None`` a new
        figure with two panels (grid + text) is created.
    show_formula : bool, default True
        If True, render a right-side text panel with formula and inputs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dy = phi_xy[0]
    dx = phi_xy[1]
    _H, _W = dy.shape

    t1_verts = _triangle_vertices(x, y, _T1_OFFSETS, dy, dx)
    t2_verts = _triangle_vertices(x, y, _T2_OFFSETS, dy, dx)
    t1_area = t2_area = None
    if t1_verts is not None:
        _, _, raw = _triangle_raw_cross(t1_verts)
        t1_area = -0.5 * raw
    if t2_verts is not None:
        _, _, raw = _triangle_raw_cross(t2_verts)
        t2_area = -0.5 * raw

    if ax is None:
        fig, (ax_grid, ax_text) = plt.subplots(
            1,
            2,
            figsize=(11, 5.5),
            gridspec_kw={"width_ratios": [1.1, 1.4]},
        )
    else:
        fig = ax.figure
        ax_grid = ax
        ax_text = None
        show_formula = False

    # ---- grid panel ----
    _plot_neighborhood(ax_grid, x, y, dy, dx, radius=2)

    for verts, area, label_color in [
        (t1_verts, t1_area, "T1"),
        (t2_verts, t2_area, "T2"),
    ]:
        if verts is None:
            continue
        poly = Polygon(
            verts,
            closed=True,
            facecolor=_fill_color(area),
            edgecolor="black",
            lw=1.6,
            alpha=0.7,
            zorder=3,
        )
        ax_grid.add_patch(poly)
        cx = sum(v[0] for v in verts) / 3
        cy = sum(v[1] for v in verts) / 3
        ax_grid.text(
            cx,
            cy,
            f"{label_color}\n{area:+.3f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#222",
        )

    # Mark the anchor pixel
    wx = x + dx[y, x]
    wy = y + dy[y, x]
    ax_grid.plot(wx, wy, "o", color="#222", markersize=6, zorder=4)
    ax_grid.annotate(
        f"(x={x}, y={y})",
        (wx, wy),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
        color="#222",
    )

    ax_grid.invert_yaxis()  # image convention: +y down
    ax_grid.set_aspect("equal")
    ax_grid.set_title(
        f"pixel (x={x}, y={y}) — T1 sign={np.sign(t1_area or 0):+.0f}  "
        f"T2 sign={np.sign(t2_area or 0):+.0f}"
    )
    ax_grid.set_xlabel("x (→)")
    ax_grid.set_ylabel("y (↓)")

    # ---- text panel ----
    if show_formula and ax_text is not None:
        ax_text.axis("off")
        lines = [
            "Formula (+y-down convention):",
            "  T = -0.5 * (AB.x * AC.y - AB.y * AC.x)",
            "  AB = B - A,  AC = C - A",
            "  warped(p, q) = (p + dx[q, p], q + dy[q, p])",
            "",
        ]
        for label, verts, area, offsets, vlabels in [
            ("T1", t1_verts, t1_area, _T1_OFFSETS, _T1_LABELS),
            ("T2", t2_verts, t2_area, _T2_OFFSETS, _T2_LABELS),
        ]:
            lines.append(f"── {label} at pixel (x, y) ──")
            lines.append(_format_inputs(x, y, offsets, vlabels, dy, dx))
            if verts is None:
                lines.append("  (out of bounds — triangle skipped)")
                lines.append("")
                continue
            AB, AC, cross = _triangle_raw_cross(verts)
            lines.append(f"  AB = ({AB[0]:+.3f}, {AB[1]:+.3f})   AC = ({AC[0]:+.3f}, {AC[1]:+.3f})")
            lines.append(
                f"  cross = AB.x*AC.y - AB.y*AC.x "
                f"= {AB[0]:+.3f}*{AC[1]:+.3f} - {AB[1]:+.3f}*{AC[0]:+.3f} "
                f"= {cross:+.4f}"
            )
            verdict = "FLIP" if area < 0 else "COLLAPSE" if area == 0 else "VALID"
            lines.append(f"  area = -0.5 * cross = {area:+.4f}   [{verdict}]")
            lines.append("")
        ax_text.text(
            0.0,
            1.0,
            "\n".join(lines),
            family="monospace",
            fontsize=9,
            va="top",
            ha="left",
            transform=ax_text.transAxes,
        )

    return fig


def find_problematic_pixels(phi_xy):
    """Return list of (x, y) where T1 or T2 has sign <= 0.

    Uses the cell-indexed sign array from ``triangle_sign_det2D`` and maps
    back to the pixel(s) that define each flipped triangle.
    """
    signs = triangle_sign_det2D(phi_xy)  # (2, H-1, W-1), cell-indexed
    bad = set()
    # T1 at cell (cy, cx) = triangle at pixel (cx+1, cy)
    for cy, cx in zip(*np.where(signs[0] <= 0)):
        bad.add((int(cx) + 1, int(cy)))
    # T2 at cell (cy, cx) = triangle at pixel (cx, cy)
    for cy, cx in zip(*np.where(signs[1] <= 0)):
        bad.add((int(cx), int(cy)))
    return sorted(bad)


def plot_problematic_triangles(
    phi_xy, title=None, max_plots=12, figsize_per_plot=(9.5, 4.5), show_formula=True
):
    """Render a page of per-pixel debug plots, one per problematic pixel.

    Intended as a debug toggle inside iterative solvers::

        if debug:
            plot_problematic_triangles(phi_xy, title=f"iter {k}")

    Parameters
    ----------
    phi_xy : ndarray, shape ``(2, H, W)``
    title : str, optional
        Figure suptitle (e.g. ``f"iter {k}"``).
    max_plots : int, default 12
        Cap on how many problematic pixels to render (sorted by (x, y));
        prevents a single call from producing hundreds of figures.
    figsize_per_plot : tuple
        Size per-pixel figure (used when rendering as separate figures).
    show_formula : bool, default True
        Pass through to :func:`plot_triangle_debug`.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One figure per problematic pixel (up to ``max_plots``). Empty list
        if no problematic pixels were found.
    """
    bad = find_problematic_pixels(phi_xy)
    if not bad:
        print("no problematic triangles" + (f" at {title}" if title else ""))
        return []
    if len(bad) > max_plots:
        print(f"{len(bad)} problematic pixels; rendering first {max_plots}")
        bad = bad[:max_plots]

    figs = []
    for x, y in bad:
        fig = plot_triangle_debug(phi_xy, x, y, show_formula=show_formula)
        if title:
            fig.suptitle(f"{title}   pixel (x={x}, y={y})", fontsize=11)
        # Skip tight_layout; if the figure was created with
        # constrained_layout=True (under the package theme), tight_layout
        # would emit a "figure layout has changed to tight" warning.
        figs.append(fig)
    return figs
