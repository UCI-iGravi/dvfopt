"""Rendering, overview strip, metric caches, and the render loop for LiveSolverWindow.

Extracted verbatim from ``app.py``'s ``LiveSolverWindow`` — pure code
motion. The mixin holds no state of its own; every attribute it touches
is created in ``LiveSolverWindow.__init__``.
"""

from __future__ import annotations

import numpy as np

from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt_gui._shared import (
    CONSTRAINT_JDET3D,
    VIEW_2TRI,
    VIEW_DIFF,
    VIEW_GRID,
    VIEW_INJ,
    VIEW_JDET,
    _folded_cells_path,
    _grid_lines,
    _min_gap_2d,
    _min_tri_from_phi,
    _quiver_lines,
)
from dvfopt_gui.overview import OverviewWorker


class RenderMixin:
    # ----- per-slice fold overview strip --------------------------------------

    def _restart_overview(self) -> None:
        """(Re)compute the per-slice fold counts in the background. Called on
        load and whenever a finished run splices the volume."""
        if self._overview_worker is not None and self._overview_worker.isRunning():
            self._overview_worker.cancel()
            if not self._overview_worker.wait(2_000):
                # Per-slice cancel checks make a hang near-impossible, but
                # dropping the last reference to a still-running QThread
                # can crash Qt — force it down before reassigning.
                self._overview_worker.terminate()
                self._overview_worker.wait(1_000)
        D = self._volume.shape[1] if self._volume is not None else 1
        if self._volume is None or D <= 1:
            self._overview_strip.setVisible(False)
            self._overview_counts = None
            return
        self._overview_strip.setVisible(True)
        self._overview_counts = np.zeros(D, dtype=np.int64)
        self._overview_strip.set_counts(self._overview_counts)
        self._overview_strip.set_current(self._z)
        self._overview_worker = OverviewWorker(self._volume, parent=self)
        self._overview_worker.chunkReady.connect(self._on_overview_chunk)
        self._overview_worker.start()

    def _on_overview_chunk(self, start: int, counts) -> None:
        # Same stale-signal guard as ``_on_finished``: a rapid reload calls
        # ``_restart_overview`` again before the previous worker's already-
        # queued ``chunkReady`` emissions have been delivered. Without this
        # check, one of those late signals lands after ``_overview_counts``
        # has been reallocated for the NEW (possibly differently-sized)
        # volume — silently writing another volume's counts into it, or
        # raising a shape-mismatch ``ValueError`` if the sizes differ.
        # ``sender()`` is None only for a direct Python call (e.g. tests),
        # in which case there is no "other" worker to guard against.
        sender = self.sender()
        if sender is not None and sender is not self._overview_worker:
            return
        if self._overview_counts is None:
            return
        counts = np.asarray(counts)
        self._overview_counts[start : start + len(counts)] = counts
        self._overview_strip.set_counts(self._overview_counts)

    # ----- rendering ---------------------------------------------------------

    def _refresh_display_from_volume(self):
        """When the volume / z-slice / view-mode changes (but no live
        solver state is available), recompute the image from the
        volume itself and clear overlays."""
        if self._volume is None:
            return
        self._invalidate_metric_caches()
        if self._is_3d_run:
            self._set_view_3d(self._volume, fast=False)
            self._window_rect.setRect(0, 0, 0, 0)
            self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[], y=[])
            self._stats_label.setText(self._format_stats(None))
            self._inspector_label.setText(self._format_inspector(None))
            self._refresh_convergence()
            return
        phi_2hw = self._volume[1:, self._z]  # (2, H, W)
        jac = jacobian_det2D(phi_2hw)[0]
        self._set_view(phi_2hw, jac)
        self._window_rect.setRect(0, 0, 0, 0)
        self._opt_rect.setVisible(False)
        self._target_marker.setData(x=[], y=[])
        self._stats_label.setText(self._format_stats(None))
        self._inspector_label.setText(self._format_inspector(None))
        self._refresh_convergence()

    def _on_threshold_changed(self, _v) -> None:
        """Repaint the idle stats panel with the new threshold.

        The metric FIELD is threshold-independent (thr only affects the
        reductions computed over it), so no cache invalidation is needed.
        Gate on a *running* worker, not a merely-existing one: the worker
        reference survives a finished run, and a threshold tweak right
        after a run must still repaint. Mid-run the snapshot stream owns
        the display — skip.
        """
        w = self._worker
        if w is None or not getattr(w, 'isRunning', lambda: False)():
            self._refresh_display_from_volume()

    def _set_view(self, phi_2hw: np.ndarray, jacobian: np.ndarray, *, fast: bool = False) -> None:
        """Update the central plot to reflect the current view mode.

        ``fast=True`` skips the per-frame fold-overlay rebuild in
        VIEW_GRID — the live-render tick uses this on big fields to
        avoid the pure-Python QPainterPath construction that scales
        with folded-cell count. The wireframe (grid curves) is still
        updated; only the magenta overlay is dropped until the next
        scrub or live tick falls back to ``fast=False``.
        """
        mode = self._view_mode
        if mode == VIEW_JDET:
            self._img.setImage(jacobian, autoLevels=False)
            self._apply_levels(jacobian)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_2TRI:
            min_tri = _min_tri_from_phi(phi_2hw)
            self._img.setImage(min_tri, autoLevels=False)
            self._apply_levels(min_tri)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_INJ:
            gap = _min_gap_2d(phi_2hw)
            self._img.setImage(gap, autoLevels=False)
            self._apply_levels(gap)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_DIFF:
            # Current minus originally-loaded per-pixel Jdet. Positive
            # (red) = Jdet rose toward feasible; negative (blue) = fell.
            diff = jacobian - self._input_jacobian()
            self._img.setImage(diff, autoLevels=False)
            self._apply_levels(diff)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        elif mode == VIEW_GRID:
            self._cbar.setVisible(False)
            # Pure grid view: hide the Jdet heatmap entirely and draw
            # only the warped wireframe. Folded cells (min(T1,T2) <= 0)
            # are overlaid with a translucent magenta fill — unless
            # ``fast`` is set, in which case we drop the overlay for
            # this frame (the QPainterPath construction is the bottleneck
            # on big fields with many folds).
            self._img.setVisible(False)
            stride = max(1, min(phi_2hw.shape[1:]) // 40)
            xs, ys = _grid_lines(phi_2hw, stride=stride)
            self._grid_curve.setData(xs, ys)
            self._grid_curve.setVisible(True)
            if fast:
                self._fold_overlay.setVisible(False)
            else:
                self._fold_overlay.setPath(_folded_cells_path(phi_2hw))
                self._fold_overlay.setVisible(True)

        # Arrow overlay sits on top of whichever view is active.
        self._update_quiver(phi_2hw)

    def _heatmap_slice_3d(self, phi3d: np.ndarray) -> np.ndarray:
        """The per-slice 3D fold field for the current z (default 6-tet
        min volume; Jdet3D when that constraint is selected). Padded to
        (H, W) with NaN at the trailing row/col so it lines up with the
        grid (the tet field is (D-1, H-1, W-1)).

        Reads through ``_metric3d_field`` — the expensive whole-volume
        kernel runs at most once per displayed field; z only changes
        which slice of the cached field gets returned."""
        z = min(self._z, phi3d.shape[1] - 1)
        if self._view_mode == VIEW_INJ:
            field = self._metric3d_field(phi3d, 'inj3d')  # (D, H, W)
            return field[z]
        if self._constraint_combo.currentData() == CONSTRAINT_JDET3D:
            field = self._metric3d_field(phi3d, 'jdet3d')  # (D, H, W)
            return field[z]
        mv = self._metric3d_field(phi3d, 'tet3d')  # (D-1, H-1, W-1)
        H, W = phi3d.shape[2:]
        out = np.full((H, W), np.nan)
        zz = min(z, mv.shape[0] - 1)
        out[: H - 1, : W - 1] = mv[zz]
        return out

    def _set_view_3d(self, phi3d: np.ndarray, *, fast: bool = False) -> None:
        """3D heatmap: the fold-metric slice at the current z. The grid /
        2-tri / Jdet views fall back to the (dy,dx) of the current slice."""
        z = min(self._z, phi3d.shape[1] - 1)
        slice_2hw = phi3d[1:, z]  # (2, H, W) [dy, dx]
        mode = self._view_mode
        if mode == VIEW_GRID:
            self._img.setVisible(False)
            self._cbar.setVisible(False)
            stride = max(1, min(slice_2hw.shape[1:]) // 40)
            xs, ys = _grid_lines(slice_2hw, stride=stride)
            self._grid_curve.setData(xs, ys)
            self._grid_curve.setVisible(True)
            if not fast:
                self._fold_overlay.setPath(_folded_cells_path(slice_2hw))
                self._fold_overlay.setVisible(True)
            else:
                self._fold_overlay.setVisible(False)
        else:
            field = self._heatmap_slice_3d(phi3d)
            self._img.setImage(field, autoLevels=False)
            self._apply_levels(field)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        self._update_quiver(slice_2hw)

    def _update_quiver(self, phi_2hw: np.ndarray) -> None:
        """Refresh the displacement-arrow overlay for ``phi_2hw`` (or hide
        it when the Arrows toggle is off)."""
        if not self._arrows_check.isChecked():
            self._quiver_curve.setVisible(False)
            return
        stride = max(1, min(phi_2hw.shape[1:]) // 30)
        xs, ys = _quiver_lines(phi_2hw, stride=stride)
        self._quiver_curve.setData(xs, ys)
        self._quiver_curve.setVisible(True)

    def _on_arrows_toggled(self, _on: bool):
        """Re-render the current frame so the overlay appears/clears now."""
        if self._latest is not None and self._latest_jacobian is not None:
            if self._latest.phi.ndim == 4:
                self._set_view_3d(self._latest.phi)
            else:
                self._set_view(self._latest.phi, self._latest_jacobian)
        else:
            self._refresh_display_from_volume()

    def _refresh_convergence(self) -> None:
        """Sync the convergence chart with the current worker history.

        Rebuilds the curve only when the history grows (cheap during long
        SLSQP runs) but always re-positions the step cursor to the slider.
        Clears the chart when there's no run to show.
        """
        worker = self._worker
        if worker is None or worker.history_len() == 0:
            self._conv_plot.clear_data()
            self._conv_len = -1
            return
        n = worker.history_len()
        total = worker.history_total
        offset = total - n  # absolute step at buffer index 0
        if n != self._conv_len:
            steps = np.arange(offset, offset + n)
            n_neg = np.fromiter(
                (worker.history_get(i).n_neg for i in range(n)), dtype=float, count=n
            )
            min_T = np.fromiter(
                (worker.history_get(i).min_T for i in range(n)), dtype=float, count=n
            )
            self._conv_plot.set_data(steps, n_neg, min_T)
            # Phase-boundary markers: wallbreaker / SLP stage snapshots
            # carry their stage name; windowed per-step snapshots don't.
            marks = [
                (offset + i, worker.history_get(i).stage)
                for i in range(n)
                if getattr(worker.history_get(i), 'stage', None) not in (None, '', 'input')
            ]
            self._conv_plot.set_stage_markers([m[0] for m in marks], [m[1] for m in marks])
            self._conv_plot.set_threshold(self._display_threshold())
            self._conv_len = n
        self._conv_plot.set_cursor(offset + self._history_slider.value())

    def _input_jacobian(self) -> np.ndarray:
        """Per-pixel Jdet of the originally-loaded field for the active
        slice — the baseline for the ``Δ Jdet vs input`` view. Falls back
        to the current volume if no pristine copy exists."""
        base = self._original_volume if self._original_volume is not None else self._volume
        return jacobian_det2D(base[1:, self._z])[0]

    def _apply_levels(self, arr: np.ndarray) -> None:
        """Set the heatmap (and its colorbar) levels for ``arr``.

        Fixed ±1 when Auto-levels is off; otherwise a symmetric
        autoscale to the array's extent so the diverging colormap stays
        centred on zero (white) and large-magnitude fields don't clip.
        Driving the colorbar's levels also updates the linked image.
        """
        if self._autolevel_check.isChecked():
            finite = arr[np.isfinite(arr)]
            m = float(np.max(np.abs(finite))) if finite.size else 1.0
            if m <= 0:
                m = 1.0
            levels = (-m, m)
        else:
            levels = (-1.0, 1.0)
        self._cbar.setLevels(levels)
        self._img.setLevels(levels)

    def _on_autolevel_toggled(self, _on: bool):
        """Re-render the current frame so the new level policy takes
        effect immediately."""
        if self._latest is not None and self._latest_jacobian is not None:
            if self._latest.phi.ndim == 4:
                self._set_view_3d(self._latest.phi)
            else:
                self._set_view(self._latest.phi, self._latest_jacobian)
        else:
            self._refresh_display_from_volume()

    def _on_view_changed(self, idx: int):
        self._view_mode = self._view_combo.itemData(idx)
        if self._latest is not None and self._latest_jacobian is not None:
            if self._latest.phi.ndim == 4:
                self._set_view_3d(self._latest.phi)
            else:
                self._set_view(self._latest.phi, self._latest_jacobian)
        else:
            self._refresh_display_from_volume()

    # ----- render loop -------------------------------------------------------

    def _render_snapshot(self, snap, *, fast: bool = False) -> None:
        """Push ``snap`` to the plot + overlays + stats panels. Shared
        between the live-render path (auto-track latest) and the history
        slider (replay a past step). Recomputes the jacobian from
        ``snap.phi`` once per call (the snapshot no longer caches it)
        and stashes it on ``self._latest_jacobian`` so the inspector
        and any subsequent reads in this frame don't recompute.

        ``fast=True`` is set by the live-render tick on big fields —
        it skips the fold-overlay rebuild in VIEW_GRID (pure-Python
        QPainterPath construction that scales with folded-cell count
        and dominates render time on large slices). The slider scrub
        path always uses ``fast=False`` for full fidelity.
        """
        # 3D z-scrub (``_on_z_changed``) re-renders this exact SAME
        # snapshot object just to re-slice the display at a new z —
        # comparing identity against the previously-rendered snapshot
        # (captured *before* ``self._latest`` is overwritten below) lets
        # that path skip the whole-volume metric invalidation so
        # ``_heatmap_slice_3d`` hits ``_metric3d_field``'s cache instead
        # of re-running the 6-tet/Jdet3D kernel on every tick. A genuinely
        # new snapshot (different object) still invalidates as before.
        same_field = snap is self._latest
        self._latest = snap
        if snap.phi.ndim == 4:  # 3D volume snapshot
            if not same_field:
                self._invalidate_metric_caches()
            self._latest_jacobian = self._heatmap_slice_3d(snap.phi)
            self._set_view_3d(snap.phi, fast=fast)
            self._window_rect.setRect(0, 0, 0, 0)
            self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[], y=[])
            self._stats_label.setText(self._format_stats(snap))
            self._refresh_convergence()
            return
        self._latest_jacobian = jacobian_det2D(snap.phi)[0]
        self._invalidate_metric_caches()
        self._set_view(snap.phi, self._latest_jacobian, fast=fast)
        self._window_rect.setRect(
            snap.window_x0,
            snap.window_y0,
            snap.window_x1 - snap.window_x0,
            snap.window_y1 - snap.window_y0,
        )
        if snap.is_padded:
            self._opt_rect.setRect(
                snap.opt_x0,
                snap.opt_y0,
                snap.opt_x1 - snap.opt_x0,
                snap.opt_y1 - snap.opt_y0,
            )
            self._opt_rect.setVisible(True)
        else:
            self._opt_rect.setVisible(False)
        self._target_marker.setData(x=[snap.neg_x], y=[snap.neg_y])
        self._stats_label.setText(self._format_stats(snap))
        if self._picked_yx is not None:
            self._inspector_label.setText(self._format_inspector(self._picked_yx))
        self._refresh_convergence()

    def _on_render_tick(self):
        if self._worker is None:
            return
        snap = self._worker.take_latest()
        # Update the history slider's range to cover everything emitted
        # so far, and (if Live is on) advance to the latest frame.
        self._history.on_tick()
        # Render the latest snapshot only when Live is on. In freeze mode
        # the slider handler controls what's shown.
        if snap is not None and self._history.is_live():
            # Big-field protection: skip the fold overlay rebuild during
            # live ticks once H·W exceeds the threshold. Scrubbing the
            # slider (post-run, or when paused) still gets the full
            # overlay — HistoryController renders with the default
            # ``fast=False``.
            H, W = snap.phi.shape[-2:]
            fast = self._fast_render_pixel_threshold < (H * W)
            self._render_snapshot(snap, fast=fast)
        else:
            # Frozen view: still extend the convergence curve as the run
            # progresses (cursor stays where the user parked it).
            self._refresh_convergence()

        # Live progress bar / ETA for the active run.
        self._update_progress()

        # One-time "Auto → <label>" note once the worker resolves it.
        if (
            not getattr(self, '_auto_label_shown', True)
            and self._worker is not None
            and getattr(self._worker, 'resolved_strategy_label', None)
        ):
            self._auto_label_shown = True
            self.statusBar().showMessage(f'Auto → {self._worker.resolved_strategy_label}', 8_000)

        # cb-rate once per second — only while a solve is actually
        # running. A loaded run is backed by a non-running ReplayHistory
        # (callback_count == 0); updating here would clobber "idle" with a
        # misleading "0 callbacks · 0.0 cb/s" for a static viewed run.
        if (
            self._last_tick.elapsed() >= 1000
            and self._worker is not None
            and self._worker.isRunning()
        ):
            dt_s = self._last_tick.restart() / 1000.0
            cb_count = self._worker.callback_count
            delta = cb_count - self._last_count
            self._last_count = cb_count
            self._fps_label.setText(f'{cb_count} callbacks · {delta / dt_s:.1f} cb/s')

    # ----- mouse pick --------------------------------------------------------

    def _on_mouse_click(self, ev):
        if not self._plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        mouse_point = self._plot.plotItem.vb.mapSceneToView(ev.scenePos())
        x = round(mouse_point.x())
        y = round(mouse_point.y())
        self._picked_yx = (y, x)
        self._inspector_label.setText(self._format_inspector((y, x)))

    def _on_mouse_moved(self, evt):
        """Hover readout — track the cursor so the inspector updates
        without a click. ``evt`` is the ``(scenePos,)`` tuple delivered
        by the rate-limiting :class:`pg.SignalProxy`. Updating
        ``_picked_yx`` (not a separate field) means the live-render path
        keeps showing whatever the cursor last hovered."""
        if self._volume is None:
            return
        scene_pos = evt[0]
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = self._plot.plotItem.vb.mapSceneToView(scene_pos)
        x = round(mouse_point.x())
        y = round(mouse_point.y())
        if self._picked_yx == (y, x):
            return
        self._picked_yx = (y, x)
        self._inspector_label.setText(self._format_inspector((y, x)))

    def _triangle_areas_cached(self, phi: np.ndarray):
        """Return ``(T1, T2)`` for the currently-displayed ``phi``.

        The cache is invalidated explicitly whenever the displayed field
        changes (see ``_invalidate_metric_caches`` calls in the render
        / refresh paths), so repeated hovers over the same frame reuse
        one computation instead of an O(H·W) triangle-area recompute per
        mouse-move. (The volume-path ``phi`` is a fresh view object each
        call, so identity-keying wouldn't hit — hence explicit
        invalidation.)"""
        if self._inspector_tri is None:
            self._inspector_tri = _triangle_areas_2d(phi[0], phi[1])
        return self._inspector_tri

    def _metric3d_field(self, phi3d: np.ndarray, kind: str) -> np.ndarray:
        """Whole-volume 3D metric field, cached per kind until the displayed
        field changes (``_invalidate_metric_caches``). Counts are cheap numpy
        reductions over this array; only the kernel is expensive."""
        field = self._metric3d_cache.get(kind)
        if field is None:
            # Resolve through the app module at call time so app-level
            # monkeypatching of _metric_field_3d (test contract) works.
            from dvfopt_gui import app as _app

            field = _app._metric_field_3d(phi3d, kind)
            self._metric3d_cache[kind] = field
        return field

    def _invalidate_metric_caches(self) -> None:
        """Drop cached per-field metrics (2D T1/T2 and 3D volume metric) —
        call whenever the displayed field changes."""
        self._inspector_tri = None
        self._metric3d_cache = {}
