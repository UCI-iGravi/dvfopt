"""Toggleable per-iteration debug tracer for dvfopt iterative solvers."""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dvfopt.jacobian import jacobian_det2D
from dvfopt.viz._style import (
    CMAP, INTERP,
    _make_jdet_norm, _annotate_neg_contour, _annotate_jdet_values,
)
from dvfopt.viz.snapshots import _draw_windows


class DebugTracer:
    """Per-iteration debug recorder and visualizer for dvfopt iterative solvers.

    Pass to ``iterative_serial`` as ``debug=``:

    >>> tracer = DebugTracer(show_every=1)
    >>> phi = iterative_serial(dvf, debug=tracer)
    >>> tracer.plot_history()

    Parameters
    ----------
    show_every : int
        Show a 3-panel snapshot (before Jdet / after Jdet / |Δφ|) every
        *show_every* outer iterations during the run.
        ``1`` = every iteration, ``0`` = record silently.
    show_sub_iters : bool
        If ``True``, also show a Jacobian heatmap after each sub-iteration
        within the inner pixel-fix loop.
    figsize : tuple
        Width × height in inches for per-iteration snapshot figures.
    """

    def __init__(self, show_every=1, show_sub_iters=False, figsize=(14, 4)):
        self.show_every = show_every
        self.show_sub_iters = show_sub_iters
        self.figsize = figsize

        # Per-outer-iteration history (index 0 = initial state before iter 1)
        self.iterations = []        # [0, 1, 2, ...]
        self.neg_counts = []        # neg Jdet pixel count
        self.min_jdets = []         # minimum Jdet value
        self.window_sizes = []      # (sy, sx) window used — populated from iter 1
        self.fix_centers = []       # (cy, cx) window centre — from iter 1
        self.sub_iter_counts = []   # number of inner sub-iterations — from iter 1
        self.elapsed = []           # wall-clock seconds — from iter 1

        # Internal per-iteration state
        self._jac_before = None     # Jdet snapshot before the current fix
        self._phi_before = None     # phi snapshot before the current fix
        self._iter_start_t = None
        self._sub_idx = 0           # sub-iteration counter within current fix

    # ------------------------------------------------------------------
    # Hooks (called by the solver at specific points)
    # ------------------------------------------------------------------

    def _on_init(self, jacobian_matrix, phi, H, W):
        """Record initial state (iteration 0)."""
        neg = int((jacobian_matrix <= 0).sum())
        mn = float(jacobian_matrix.min())
        self.iterations = [0]
        self.neg_counts = [neg]
        self.min_jdets = [mn]

    def _on_iter_start(self, iteration, neg_index, jacobian_matrix, phi):
        """Snapshot pre-fix state; call before ``_serial_fix_pixel``."""
        self._jac_before = jacobian_matrix.copy()
        self._phi_before = phi.copy()
        self._iter_start_t = time.perf_counter()
        self._sub_idx = 0

    def _on_iter_end(self, iteration, neg_index, window_center,
                     jacobian_matrix, phi, submatrix_size, per_index_iter):
        """Record post-fix state and optionally display the snapshot."""
        elapsed = (time.perf_counter() - self._iter_start_t
                   if self._iter_start_t is not None else 0.0)

        neg = int((jacobian_matrix <= 0).sum())
        mn = float(jacobian_matrix.min())

        self.iterations.append(iteration)
        self.neg_counts.append(neg)
        self.min_jdets.append(mn)
        self.window_sizes.append(
            submatrix_size if isinstance(submatrix_size, tuple)
            else (int(submatrix_size), int(submatrix_size))
        )
        self.fix_centers.append(window_center)
        self.sub_iter_counts.append(per_index_iter)
        self.elapsed.append(elapsed)

        if self.show_every and iteration % self.show_every == 0:
            self._plot_snapshot(iteration, neg_index, window_center,
                                jacobian_matrix, phi, submatrix_size, neg, mn)

    def plot_callback(self, deformation_i, phi):
        """Sub-iteration callback.

        Pass as ``plot_callback=tracer.plot_callback`` to ``iterative_serial``,
        or it is wired automatically when ``debug=tracer`` is used.
        No-ops when ``show_sub_iters=False``.
        """
        if not self.show_sub_iters:
            return
        self._sub_idx += 1
        jac = jacobian_det2D(phi)
        neg = int((jac <= 0).sum())
        mn = float(jac.min())

        fig, ax = plt.subplots(figsize=(5, 4))
        norm = _make_jdet_norm([jac[0]])
        ax.imshow(jac[0], cmap=CMAP, norm=norm, interpolation=INTERP)
        _annotate_neg_contour(ax, jac[0])
        _annotate_jdet_values(ax, jac[0])
        ax.set_title(
            f"Sub-iter {self._sub_idx}  |  neg={neg}  min={mn:+.4f}",
            fontsize=9)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Per-iteration 3-panel snapshot
    # ------------------------------------------------------------------

    def _plot_snapshot(self, iteration, neg_index, window_center,
                       jacobian_matrix, phi, submatrix_size, neg, mn):
        """3-panel: before Jdet / after Jdet / displacement change |Δφ|."""
        if isinstance(submatrix_size, (int, float)):
            sy = sx = int(submatrix_size)
        else:
            sy, sx = int(submatrix_size[0]), int(submatrix_size[1])
        cy, cx = window_center

        # Shared colour norm across both Jdet panels
        norm = _make_jdet_norm([self._jac_before[0], jacobian_matrix[0]])

        # Displacement change magnitude
        delta_phi = np.sqrt(
            (phi[0] - self._phi_before[0]) ** 2 +
            (phi[1] - self._phi_before[1]) ** 2
        )

        fig, axes = plt.subplots(1, 3, figsize=self.figsize,
                                 constrained_layout=True)
        ax_b, ax_a, ax_d = axes

        # ---- Before Jdet (with window overlay) ----
        ax_b.imshow(self._jac_before[0], cmap=CMAP, norm=norm,
                    interpolation=INTERP)
        _annotate_neg_contour(ax_b, self._jac_before[0])
        _annotate_jdet_values(ax_b, self._jac_before[0])
        _draw_windows(ax_b, [(cy, cx, submatrix_size, False)])
        neg_b = int((self._jac_before <= 0).sum())
        ax_b.set_title(
            f"Before — iter {iteration}\n"
            f"neg={neg_b}  fix=({neg_index[0]},{neg_index[1]})",
            fontsize=9)

        # ---- After Jdet ----
        im_a = ax_a.imshow(jacobian_matrix[0], cmap=CMAP, norm=norm,
                           interpolation=INTERP)
        _annotate_neg_contour(ax_a, jacobian_matrix[0])
        _annotate_jdet_values(ax_a, jacobian_matrix[0])
        ax_a.set_title(
            f"After — iter {iteration}\nneg={neg}  min={mn:+.4f}",
            fontsize=9)
        fig.colorbar(im_a, ax=[ax_b, ax_a], shrink=0.8, label="Jdet")

        # ---- |Δφ| displacement change ----
        im_d = ax_d.imshow(delta_phi, cmap="viridis", interpolation=INTERP)
        hy, hx = sy // 2, sx // 2
        hy_hi, hx_hi = sy - hy, sx - hx
        ax_d.add_patch(Rectangle(
            (cx - hx - 0.5, cy - hy - 0.5), sx, sy,
            linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--"))
        fig.colorbar(im_d, ax=ax_d, shrink=0.8, label="|Δφ|")
        ax_d.set_title(
            f"Displacement change |Δφ|\nwin {sy}×{sx}  centre ({cy},{cx})",
            fontsize=9)

        plt.show()

    # ------------------------------------------------------------------
    # Post-run convergence plots
    # ------------------------------------------------------------------

    def plot_history(self, figsize=(13, 4)):
        """Plot convergence curves after the run.

        Three panels: negative Jdet count, minimum Jdet, and window area
        (pixels²) vs outer iteration number.
        """
        if len(self.iterations) < 2:
            print("No iteration history to plot.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        # Neg count
        ax = axes[0]
        ax.plot(self.iterations, self.neg_counts, "o-",
                color="crimson", markersize=4, linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Neg Jdet pixel count")
        ax.set_title("Neg Jdet count")
        ax.grid(True, alpha=0.35)

        # Min Jdet
        ax = axes[1]
        ax.plot(self.iterations, self.min_jdets, "o-",
                color="steelblue", markersize=4, linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Min Jdet")
        ax.set_title("Minimum Jdet")
        ax.grid(True, alpha=0.35)

        # Window area evolution (iter 1+)
        ax = axes[2]
        if self.window_sizes:
            iters = self.iterations[1:]
            areas = [sy * sx for sy, sx in self.window_sizes]
            ax.step(iters, areas, where="post",
                    color="darkorange", linewidth=1.5)
            ax.scatter(iters, areas, s=18, color="darkorange", zorder=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Window area (pixels)")
        ax.set_title("Window size evolution")
        ax.grid(True, alpha=0.35)

        plt.show()
