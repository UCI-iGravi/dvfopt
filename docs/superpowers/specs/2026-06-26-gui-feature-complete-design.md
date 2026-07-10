# dvfopt GUI: feature-complete, fast, robust DVF-fixing interface

**Date:** 2026-06-26
**Status:** Approved design, pending implementation plan
**Scope:** `dvfopt_gui/` + one backward-compatible library change
(`correct_dvf_25d` progress hook). Builds on the merged true-3D mode.

## 1. Problem

The GUI is missing the package's flagship correction methods and has known
performance/robustness gaps on realistic (volume-scale) data:

- **Missing methods/pipelines:** `SLPStrategy` (the 2-tri champion),
  `correct_dvf_25d` (the productized 2.5D marching pipeline),
  `correct_dvf_3d` (end-to-end 3D orchestrator), `BarrierTet3DTorchStrategy`
  (GPU 3D barrier), and an "Auto" strategy picker (`auto_strategy`).
- **Interop:** only `.npy/.npz` load; no NIfTI/MHA import despite
  `SimpleITK`/`nibabel` being available; no plain export of the corrected
  volume; no NaN/Inf validation on load.
- **Performance:** in 3D mode, every z-slider tick recomputes the 6-tet
  min-volume over the **whole volume** up to 3× (heatmap slice at
  [app.py:1411-1416], idle stats ×2 at [app.py:2055-2056]) and every
  inspector hover recomputes it again ([app.py:2154]). `np.load` of GB-scale
  files blocks the GUI thread ([app.py:1040]).
- **Robustness:** the undo stack keeps up to 30 full `volume.copy()`
  snapshots (count-capped, byte-unaware — ~55 GB worst case on B0039);
  solver threshold (0.01) is hard-wired and not user-editable.
- **Navigation:** no per-slice fold overview for volumes; no per-strategy
  parameter editing; ROI is 2D-only.

## 2. Decisions locked with the user

- **2.5D UX:** BOTH a dedicated "Run 2.5D marching" action AND a one-click
  "Full pipeline (2D + 2.5D)" that chains Run-all-z (selected 2D method)
  then 2.5D marching. Presented as a compact **"Pipeline ▾"** dropdown
  toolbar button (two actions), mirrored in the Run menu.
- **SLP default:** SLP becomes the top entry AND the fresh-install default
  of the 2-tri method menu (`accuracy='fast'`). Saved QSettings still
  restore the user's last selection.

## 3. Verified facts (contracts the implementation relies on)

- `SLPStrategy(accuracy='fast'|'max', cluster_pixel_threshold=…, …)` —
  dataclass; `'max'` requires torch; no `step_callback` support (one-shot).
- `correct_dvf_25d(phi, *, threshold=0.01, origin='auto', mop=True, …,
  verbose=0) -> (phi_out, Correct25DReport)`; requires `dz≡0`
  (`dz_tol=1e-12`), never writes `phi[0]`, does not mutate the input.
  `Correct25DReport`: `feasible, n_neg_in, n_neg_out, n_below_out,
  min_T_out, l1_from_input, wall_s, origin, stages`.
- `correct_dvf_3d(phi, *, threshold=0.01, recover_threshold=None,
  bulk='auto', …, verbose=0) -> (phi_out, Correct3DReport)`.
  `Correct3DReport` adds `best_diag_floor_in/out, residual_cubes`.
- `auto_strategy(constraint, init_n_neg, init_min, objective_label) -> str`
  (a registry label); `make_strategy(label)` builds the instance.
- `Solver(constraint, objective, strategy, threshold=None)` — threshold is
  a constructor param defaulting to `DEFAULT_PARAMS['threshold']` (0.01).
  The windowed path (`iterative_serial`) also accepts `threshold=`.
- SimpleITK + nibabel import successfully in this environment. The
  package's numpy↔sitk convention is documented in
  `dvfopt/jacobian/sitk_jdet.py`: numpy `(3,…)` `[dz,dy,dx]` ↔ sitk array
  `(…, 3)` with components reordered `[2,1,0]` (zyx→xyz). The importer
  mirrors that file (single source of truth — reuse its helpers if
  importable, else replicate with a cross-test against it).
- Existing GUI machinery reused throughout: `_volume_snapshot` (3D
  snapshots), `_run_via_solver_3d` (whole-volume runs, memory guard, crop
  guard), `_run_all_remaining` (Run-all-z chain), `_metric_counts_3d`,
  `HistoryController`, QSettings persistence.

## 4. Methods & pipelines

### 4.1 SLP (2-tri champion)
- `_METHOD_SPECS_2TRI` gains `('slp', 'SLP (champion: cluster trust-region
  SLP + HiGHS L1)')` as the FIRST entry; `DEFAULT_METHOD_BY_CONSTRAINT['2tri']
  = 'slp'`.
- Worker: `slp_2tri` → `SLPStrategy()` (accuracy default 'fast'); one-shot
  through `_run_via_solver` (init+final snapshots).
- `accuracy='max'` is reachable via the strategy-params panel (§7.2), which
  must render it as a choice widget and surface a clear error dialog if
  torch is missing (the strategy raises).

### 4.2 Auto strategy picker
- New menu entry `('auto', 'Auto (pick by fold stats)')` in BOTH 2D
  families (`2tri`, `jdet`). Not offered for 3D (no heuristic exists).
- Worker `_build_strategy` for `auto_2tri`/`auto_jdet`: compute the input's
  `(n_neg, min_T)` under the run metric, call
  `auto_strategy(constraint_instance, n_neg, min_T, objective_label)`,
  build via `make_strategy(label)`. Record the resolved label on the worker
  (e.g. `self.resolved_strategy_label`); the window shows
  `Auto → <label>` in the status bar when the run starts.
- The resolved label is what gets saved in the run NPZ `method` metadata
  (prefixed, e.g. `auto:<label>`), so reloads are self-describing.

### 4.3 2.5D marching (library change + GUI)
- **Library:** `correct_dvf_25d(..., progress_callback=None)`. When
  supplied, called at each sweep-slice completion and each mop round with
  `{'phase': 'sweep'|'mop', 'index': i, 'total': n, 'n_neg': int}` (n_neg =
  current residual 3D fold count if cheaply available at that point, else
  −1). Exceptions from the callback — specifically `KeyboardInterrupt` —
  propagate (no swallowing). Default `None` preserves behavior exactly.
- **Worker:** new method id `marching25d_tet3d` routed to a dedicated
  `_run_marching_25d()`: validates shape, emits init 3D snapshot
  (`tet3d` metric), runs `correct_dvf_25d(vol, threshold=<GUI thr>,
  progress_callback=…)`; the callback checks the stop flag (raise
  `KeyboardInterrupt`) and records occasional full-volume 3D snapshots
  (respecting the existing memory guard — snapshot at most every
  ceil(total/6) progress events so ≤ ~6 stage snapshots + init + final).
  Final snapshot from `phi_out`; `Correct25DReport` fields go to the
  status bar ("2.5D: 1,058,831 → 33 folds, feasible=False, 412 s").
- **GUI:** "Pipeline ▾" QToolButton (with menu) in the run row, + the same
  two actions under the Run menu:
  - **Run 2.5D marching** — enabled when a `D>1` volume is loaded and no
    run is active. On click: check `|dz|max ≤ 1e-9`; if violated, dialog:
    "dz must be zero (per-slice 2D correction first). Run the full
    pipeline instead?" with buttons [Run full pipeline] [Cancel].
  - **Full pipeline (2D + 2.5D)** — pushes ONE undo entry up front and
    suppresses the Run-all-z-internal undo push for the batch, then runs
    the existing Run-all-z chain with the currently selected 2D method,
    and on clean batch completion automatically starts the 2.5D run (flag
    `_pipeline_after_run_all`; cleared on stop/error). Status bar narrates
    the phase ("Pipeline: per-slice 2D (slice 12/528)…" → "Pipeline: 2.5D
    marching…").
  - **Input source (deliberate exception):** unlike every other run —
    which reads the pristine `_original_volume` — the 2.5D stage runs on
    the **current** `self._volume`, because its input IS the per-slice-2D-
    corrected state (that is the pipeline's precondition). The dedicated
    worker is handed `self._volume.copy()`.
  - Both actions put the viewer in the 3D (`tet3d`) display mode for the
    duration/result (constraint combo switched to `tet3d`, which the
    existing machinery already handles).
- Progress bar during 2.5D: `slice i/D (sweep)` then `mop r/n` from the
  callback, with elapsed time.

### 4.4 Full 3D pipeline (`correct_dvf_3d`)
- tet3d method menu gains `('pipeline3d', 'Full 3D pipeline (bulk auto +
  k-ring escape)')`. Worker routes `pipeline3d_tet3d` to a dedicated
  one-shot `_run_pipeline_3d()` (init + final 3D snapshots; busy progress
  bar + elapsed; Stop is best-effort — documented in the Stop tooltip).
  `Correct3DReport` summary to the status bar.

### 4.5 Torch 3D barrier
- tet3d menu gains `('barrier_torch', 'Barrier GPU (torch; CPU fallback)')`
  ONLY when `importlib.util.find_spec('torch')` is non-None (checked once
  at menu population). Worker: `barrier_torch_tet3d` →
  `BarrierTet3DTorchStrategy()` through `_run_via_solver_3d`.

## 5. Interop

### 5.1 Import (new module `dvfopt_gui/io_formats.py`)
- Load dialog filter gains `*.nii *.nii.gz *.mha *.mhd *.nrrd` (present
  only when SimpleITK imports; the module exposes
  `sitk_available() -> bool`).
- `load_dvf_sitk(path) -> np.ndarray (3,D,H,W) [dz,dy,dx]`:
  `sitk.ReadImage` → accept (a) vector images with 3 components
  (`GetArrayFromImage` → `(D,H,W,3)` xyz) and (b) 2-component 2D vector
  images (→ `(H,W,2)` xy, mapped to `(3,1,H,W)` with dz=0). Component
  reorder per the sitk_jdet convention. Reject anything else with a clear
  `ValueError` (shown in the existing load-error dialog).
- `_on_load` dispatches by extension: `.npy/.npz` → existing path; sitk
  extensions → `load_dvf_sitk` → `LoadedRun(volume=…)`.

### 5.2 Export
- File menu: **Export corrected DVF…** (enabled when a volume is loaded):
  save-dialog with `.npy` (writes `self._volume` as-is, float64,
  `[dz,dy,dx]`) and `.nii.gz` (via `io_formats.save_dvf_sitk`, reverse
  convention) filters. Status-bar confirmation with the path.

### 5.3 Load validation (all paths)
- After any successful parse (npy, npz, sitk): if `not np.isfinite(vol).all()`,
  reject with a dialog reporting the count of non-finite values and the
  first offending index. Applied in ONE place (`_apply_loaded_run` entry or
  a shared `_validate_finite` helper) so every format is covered.

## 6. Performance & robustness

### 6.1 3D metric cache (top priority)
- Window-level cache: `self._metric3d_cache: dict | None` holding, for the
  currently displayed 3D field: `{'kind': str, 'field': ndarray,
  'n_neg': int, 'min_T': float, 'infeasible': int}` — computed at most once
  per displayed-field change per kind.
- All three hot paths read through it: `_heatmap_slice_3d` (slice of
  `field`), `_format_stats` idle (counts), `_format_inspector` 3D (indexed
  lookup into `field` — note tet field is min-of-6, which is exactly what
  the inspector shows today).
- Invalidation piggybacks on the existing `_invalidate_inspector_cache`
  call sites (rename to `_invalidate_metric_caches`, clearing both the 2D
  tri cache and the 3D cache). The threshold spinbox (§6.3) also
  invalidates (infeasible count depends on it).
- **Test:** monkeypatch-count `six_tet_min_volume_3d` calls; assert a
  z-change and a hover on an unchanged field trigger ZERO new full-volume
  computations after the first render.

### 6.2 Undo byte budget + threaded load
- `_UNDO_MAX_BYTES = 2 * 1024**3` module constant; `_push_undo_state`
  evicts oldest entries while `sum(v.nbytes) > budget`, always retaining
  at least the newest entry. `_UNDO_MAX` (30) stays as a secondary cap.
- Loading moves off the GUI thread: a minimal `LoadWorker(QThread)`
  (in `worker.py`) does `np.load`/`parse_loaded`/`load_dvf_sitk` and emits
  `loaded(LoadedRun)` or `failed(str)`. `_on_load` starts it, disables the
  Load button, shows "Loading <name>…" in the status bar; results land on
  the GUI thread via the signal. Errors show the existing dialogs.

### 6.3 Threshold spinbox
- Method bar gains `thr:` `QDoubleSpinBox` (decimals=4, range 0.0–1.0,
  step 0.005, default `DEFAULT_PARAMS['threshold']`), persisted in
  QSettings. Threaded through: `Solver(threshold=…)` in `_run_via_solver`
  AND `_run_via_solver_3d`, `iterative_serial(threshold=…)`,
  `correct_dvf_25d(threshold=…)`, `correct_dvf_3d(threshold=…)`.
- The stats panel's infeasible(<thr) lines and the snapshot `min_T` flag
  use the spinbox value instead of the module constant (constant remains
  the default).

## 7. Navigation & control

### 7.1 Per-slice fold overview strip (new `dvfopt_gui/overview.py`)
- `SliceOverviewStrip(pg.PlotWidget)` — fixed-height (~44 px) bar chart
  under the z-slider row, visible only when `D > 1`: x = slice index,
  y = per-slice 2-tri fold count (`_metric_counts` per slice); a vertical
  marker tracks the current z; clicking a bar sets the z-slider.
- Counts computed in a background `QThread` (`OverviewWorker`) emitting
  progressive chunks (e.g. every 32 slices) so the strip fills in without
  blocking; recomputed on load and after any run splices the volume
  (invalidate + restart; cancel any in-flight compute first).
- API: `set_counts(np.ndarray)`, `set_current(z)`, signal `sliceClicked(int)`.

### 7.2 Strategy params panel (new `dvfopt_gui/strategy_params.py`)
- Params dialog gains a **Strategy** tab: widgets auto-generated from
  `dataclasses.fields(type(strategy))` of the currently selected method's
  strategy class — `int`→QSpinBox, `float`→QDoubleSpinBox,
  `bool`→QCheckBox, `str`→QLineEdit (with a QComboBox special-case for
  `accuracy` ∈ {fast, max}); tuple/None-typed fields are shown read-only
  (out of scope to edit). Fields whose names the toolbar already owns
  (`time_budget_s`) are omitted.
- Overrides stored per method id: `self._strategy_overrides: dict[str,
  dict]`, persisted in QSettings as JSON; a **Reset** button clears the
  current method's overrides. Worker applies them at construction:
  `strategy_cls(**{**defaults_from_toolbar, **overrides})`, with a
  try/except surfacing bad values as an error dialog instead of a crash.
- Methods that aren't dataclass-backed (`auto`, `marching25d`, `pipeline3d`,
  `nmvf`) show an informative "no editable parameters" label (pipeline
  knobs stay out of scope for this pass).

### 7.3 3D sub-volume ROI
- In 3D mode the Rect ROI becomes visible again for (y, x); two spinboxes
  `z0:`/`z1:` appear next to the z-slider (only in 3D mode; default full
  range, clamped/validated `z1 ≥ z0`).
- "Run section" is re-enabled in 3D: solves
  `vol[:, z0:z1+1, y0:y1+1, x0:x1+1].copy()` with the selected 3D method
  via `_run_via_solver_3d`, then splices back into `self._volume` (new 3D
  branch in `_on_finished` keyed off stored `_section_bounds_3d`).
  Minimum size 3×3×3; same seam-caveat status hint as 2D. Undo = one entry.

## 8. Out of scope

Image-underlay warp preview; autosave/crash recovery; editing tuple-typed
strategy fields; composite objectives UI; `notebooks/` formatting;
progress callback for `correct_dvf_3d` (busy-bar only this pass).

## 9. Files touched

| File | Change |
|---|---|
| `dvfopt/pipeline_25d.py` | `progress_callback=None` + per-slice/mop-round calls (default no-op) |
| `dvfopt_gui/app.py` | menus (SLP/Auto/pipeline3d/torch/2.5D), Pipeline ▾ button, threshold spinbox, 3D metric cache, undo byte budget, threaded-load wiring, export action, finite validation, 3D ROI, overview-strip wiring, status reporting |
| `dvfopt_gui/worker.py` | `slp/auto/pipeline3d/barrier_torch/marching25d` dispatch, `_run_marching_25d`, `_run_pipeline_3d`, auto-resolution, threshold param, `LoadWorker` |
| `dvfopt_gui/io_formats.py` | NEW — sitk import/export + availability probe |
| `dvfopt_gui/overview.py` | NEW — per-slice fold strip + background counter |
| `dvfopt_gui/strategy_params.py` | NEW — auto-generated strategy param tab |
| `dvfopt_gui/persistence.py` | `method` metadata accepts `auto:<label>`; no schema change |
| `tests/` | per-item TDD (see §10) |
| `CLAUDE.md` | GUI section: new methods/pipelines/import-export |

## 10. Testing

- **Library:** 25d `progress_callback` fires per slice/mop with the
  documented dict; `KeyboardInterrupt` propagates; `None` → behavior
  unchanged (existing pipeline tests still pass).
- **Worker:** dispatch/metric tests for the five new method ids; auto
  resolution picks a label and records it; 2.5D run on a small synthetic
  per-slice-feasible volume completes and reduces/keeps-zero 3D folds;
  threshold parameter reaches `Solver` (assert via a stub strategy
  receiving `threshold`).
- **Widget (offscreen):** menu contents incl. SLP-first/default + torch
  gating (monkeypatch `find_spec`); Pipeline ▾ actions gated by D>1;
  dz≠0 dialog path; full-pipeline chaining flag; threshold spinbox
  persists and reaches the stats panel; 3D ROI spinboxes appear only in
  3D and Run-section splices a sub-volume; overview strip populates,
  click jumps z; export writes a loadable `.npy`/`.nii.gz` round-trip;
  non-finite load rejected with dialog.
- **Performance:** metric-kernel call-count test (§6.1); LoadWorker
  delivers via signal (no GUI-thread `np.load` in `_on_load`).
- **Interop:** sitk round-trip — synthetic `(3,D,H,W)` volume →
  `save_dvf_sitk` → `load_dvf_sitk` → allclose + channel-order cross-check
  against `dvfopt/jacobian/sitk_jdet.py`'s convention.
- **Full suite + ruff check/format** (CI gate incl. `dvfopt_gui`) at the
  end.

## 11. Risks & mitigations

- **25d callback insertion** touches the productized pipeline → default-
  `None` no-op, calls only at loop boundaries, full pipeline tests rerun.
- **Auto label → strategy mismatch** (label not constructible) →
  `make_strategy` is the same registry the package uses; on failure fall
  back to the per-constraint default method with a status-bar note.
- **sitk axis/component order bugs** → single source of truth
  (sitk_jdet convention) + round-trip and cross-check tests.
- **Strategy-params panel type edge cases** (tuple/Optional fields) →
  rendered read-only; constructor errors surfaced as dialogs.
- **app.py growth** → all substantial new UI lives in the three new
  modules; app.py gets wiring only.
- **Full-pipeline state machine** (Run-all-z → 2.5D chain) → single flag
  checked in `_on_finished`'s existing batch-completion branch; cleared on
  stop/error; covered by a widget test with stubbed workers.
- **Worker dispatch ordering** — `marching25d_tet3d` and `pipeline3d_tet3d`
  end in `_tet3d`, which the generic route sends to `_run_via_solver_3d`.
  `run()` must special-case these two ids to their dedicated runners
  BEFORE the generic `tet3d`/`jdet3d` route (regression test asserts the
  dedicated runner is invoked, not `_build_strategy`).
