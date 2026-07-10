# GUI Feature-Complete Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dvfopt GUI a feature-complete, fast, robust DVF-fixing interface: SLP (default) + Auto + 2.5D marching + full-3D pipeline + torch barrier, NIfTI import/export, threshold control, 3D metric caching, threaded load, undo byte budget, per-slice overview strip, auto-generated strategy params, and a 3D sub-volume ROI.

**Architecture:** All new UI lives in three new focused modules (`io_formats.py`, `overview.py`, `strategy_params.py`); `app.py` gets wiring only. The worker gains two dedicated volume-level runners (`_run_marching_25d`, `_run_pipeline_3d`) special-cased in `run()` BEFORE the generic `_tet3d` route. One backward-compatible library change: `correct_dvf_25d(progress_callback=None)`.

**Tech Stack:** Python 3.13, PyQt5, pyqtgraph, NumPy, SimpleITK (optional at runtime), pytest offscreen.

**Spec:** `docs/superpowers/specs/2026-06-26-gui-feature-complete-design.md`

## Global Constraints

- **Phi conventions:** GUI volume `(3, D, H, W)` = `[dz, dy, dx]`; 2D snapshot phi `(2, H, W)` = `[dy, dx]`; 3D snapshot phi = full volume (ndim 4 discriminator).
- **sitk convention** (single source of truth: `dvfopt/jacobian/sitk_jdet.py`): numpy → sitk is `np.transpose(vol, (1, 2, 3, 0))` then component reorder `[..., [2, 1, 0]]` (`[dz,dy,dx]` → `[dx,dy,dz]`); import is the exact reverse.
- **Library changes default to no-op:** `progress_callback=None` ⇒ byte-identical behavior; `KeyboardInterrupt` from callbacks propagates (no bare `except`).
- **Dispatch:** `method_id = <algo>_<tag>` via `rpartition('_')`. New ids: `slp_2tri`, `auto_2tri`, `auto_jdet`, `pipeline3d_tet3d`, `barrier_torch_tet3d`, `marching25d_tet3d`. The two pipeline ids MUST be special-cased in `run()` before the generic `tet3d`/`jdet3d` route.
- **Threshold:** all solve paths take the GUI spinbox value via `params['threshold']`; display infeasible counts use the same value.
- **2.5D input exception:** the 2.5D stage runs on the CURRENT `self._volume` (per-slice-corrected state), not `_original_volume`.
- **Tests:** `QT_QPA_PLATFORM=offscreen` for widget tests. CI gate: `python -m ruff check dvfopt dvfopt_gui tests benchmarks` AND `python -m ruff format --check dvfopt dvfopt_gui tests benchmarks` must stay clean; run both before every commit.
- **Git:** local commit per task (scoped `git add`), no push until the final PR.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `dvfopt/pipeline_25d.py` | 2.5D pipeline | `progress_callback=None` + per-slice/mop calls |
| `dvfopt_gui/worker.py` | solver thread, dispatch | slp/auto/pipeline3d/torch dispatch, `_run_marching_25d`, `_run_pipeline_3d`, threshold, overrides, `LoadWorker` |
| `dvfopt_gui/app.py` | window wiring | menus, Pipeline ▾, thr spinbox, metric cache, undo budget, threaded load, export, validation, 3D ROI, overview wiring, reports |
| `dvfopt_gui/io_formats.py` | NEW: sitk import/export | `sitk_available`, `is_sitk_path`, `load_dvf_sitk`, `save_dvf_sitk` |
| `dvfopt_gui/overview.py` | NEW: per-slice strip | `SliceOverviewStrip`, `OverviewWorker` |
| `dvfopt_gui/strategy_params.py` | NEW: params tab | `strategy_class_for`, `editable_fields`, `StrategyParamsTab` |
| `tests/test_pipeline_25d_callback.py` | NEW | library callback tests |
| `tests/test_gui_logic.py`, `tests/test_gui_app.py`, `tests/test_gui_io_formats.py` (NEW) | GUI tests | per task |

---

### Task 1: `correct_dvf_25d` progress callback (library)

**Files:**
- Modify: `dvfopt/pipeline_25d.py` (signature ~line 93-140; sweep loops ~226/240; mop block ~258-271)
- Test: `tests/test_pipeline_25d_callback.py` (new)

**Interfaces:**
- Produces: `correct_dvf_25d(..., progress_callback=None)`. When supplied, called after each sweep-slice repair and after the mop with `{'phase': 'sweep'|'mop', 'index': int, 'total': int, 'n_neg': int, 'phi': <live (3,D,H,W) buffer — copy if kept>}`. For `'sweep'`, `index` = slices completed so far, `total` = `D`; for `'mop'`, `index=total=1`. Exceptions (incl. `KeyboardInterrupt`) propagate. Default `None` ⇒ unchanged behavior.

- [ ] **Step 1: Write the failing tests** (create `tests/test_pipeline_25d_callback.py`)

```python
"""progress_callback hook on the 2.5D marching pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt.pipeline_25d import correct_dvf_25d


def _interlayer_folded_volume(D=4, H=8, W=8):
    """dz==0, per-slice 2D-feasible, but adjacent slices' dx alternate sign
    strongly at one column -> inter-layer 6-tet folds."""
    vol = np.zeros((3, D, H, W), dtype=np.float64)
    for k in range(D):
        vol[2, k, :, 4] = 0.7 if k % 2 == 0 else -0.7
    # Self-check the construction: it must actually have 3D folds.
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    assert (six_tet_min_volume_3d(vol) <= 0).sum() > 0, 'fixture has no 3D folds'
    return vol


def test_progress_callback_fires_with_contract_keys():
    vol = _interlayer_folded_volume()
    events = []
    correct_dvf_25d(vol, verbose=0, progress_callback=lambda e: events.append(dict(e)))
    assert events, 'no progress events fired'
    sweep = [e for e in events if e['phase'] == 'sweep']
    assert sweep, 'no sweep events'
    for e in events:
        assert set(e) == {'phase', 'index', 'total', 'n_neg', 'phi'}
        assert e['phase'] in ('sweep', 'mop')
        assert e['phi'].shape == vol.shape
    # sweep indices increase, total is D
    assert [e['index'] for e in sweep] == sorted(e['index'] for e in sweep)
    assert all(e['total'] == vol.shape[1] for e in sweep)


def test_progress_callback_keyboardinterrupt_propagates():
    vol = _interlayer_folded_volume()

    def cb(e):
        raise KeyboardInterrupt('stop')

    with pytest.raises(KeyboardInterrupt):
        correct_dvf_25d(vol, verbose=0, progress_callback=cb)


def test_default_none_unchanged():
    vol = _interlayer_folded_volume()
    out, report = correct_dvf_25d(vol, verbose=0)
    assert out.shape == vol.shape
    assert report.n_neg_out <= report.n_neg_in
```

- [ ] **Step 2: Run to verify RED**

Run: `python -m pytest tests/test_pipeline_25d_callback.py -v`
Expected: FAIL with `TypeError: correct_dvf_25d() got an unexpected keyword argument 'progress_callback'`

- [ ] **Step 3: Implement.** In `dvfopt/pipeline_25d.py`:

Add to the `correct_dvf_25d` signature after `verbose: int = 0,`:

```python
    progress_callback=None,
```

Document it in the docstring's Parameters block:

```
    progress_callback : callable or None
        When supplied, called after each sweep-slice repair and after the
        mop with ``{'phase': 'sweep'|'mop', 'index', 'total', 'n_neg',
        'phi'}`` (``phi`` is the live output buffer — copy if you keep it).
        Exceptions — notably ``KeyboardInterrupt`` — propagate, so a GUI
        can use it to stop between slices. ``None`` (default) is a no-op.
```

Right before the up-sweep loop (`for z in range(origin_idx + 1, D):`), add:

```python
    _sweep_done = 0

    def _emit_progress(phase, index, total, n_neg):
        if progress_callback is not None:
            progress_callback(
                {'phase': phase, 'index': index, 'total': total,
                 'n_neg': int(n_neg), 'phi': out}
            )
```

(The volume being updated in the sweep is the array the slices are written
into — use whatever local name the loops mutate; at these lines it is the
array `march_slice` writes back into via `cur`/`out`. Pass THAT array.)

Immediately after EACH of the two `march_slice(...)` call lines (up loop
~line 229 and down loop ~line 243), add:

```python
        _sweep_done += 1
        _emit_progress('sweep', _sweep_done, D, n_after)
```

After the mop stage's `stages.append(...)` (~line 266-270), add:

```python
        _emit_progress('mop', 1, 1, n_neg_mop)
```

(Use the mop-result count variable in scope there — the grep shows
`n_neg_mop` in the verbose print at ~line 271.)

- [ ] **Step 4: Run GREEN + regression**

Run: `python -m pytest tests/test_pipeline_25d_callback.py tests/test_pipeline_25d.py -q`
Expected: all PASS (existing 2.5D tests must be untouched by the default-None path)

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt/pipeline_25d.py tests/test_pipeline_25d_callback.py
python -m ruff format --check dvfopt/pipeline_25d.py tests/test_pipeline_25d_callback.py
git add dvfopt/pipeline_25d.py tests/test_pipeline_25d_callback.py
git commit -m "feat(25d): optional progress_callback at sweep/mop boundaries"
```

---

### Task 2: SLP (default) + Auto strategy picker

**Files:**
- Modify: `dvfopt_gui/app.py` (`_METHOD_SPECS_2TRI`, `_METHOD_SPECS_JDET`, `DEFAULT_METHOD_BY_CONSTRAINT`, `_build_save_payload`, `_on_render_tick`)
- Modify: `dvfopt_gui/worker.py` (`_build_strategy`, `__init__`)
- Test: `tests/test_gui_logic.py`, `tests/test_gui_app.py`

**Interfaces:**
- Consumes: `dvfopt.SLPStrategy`, `dvfopt.make_strategy(label)`, `dvfopt.solver.auto_strategy(constraint, n_neg, min_T, objective_label) -> str` (labels: 2tri → `'m10'|'m14_schwarz'|'m14'|'barrier'|'slsqp'`; jdet → `'barrier'|'slsqp_windowed'`).
- Produces: method ids `slp_2tri`, `auto_2tri`, `auto_jdet`; `SolverWorker.resolved_strategy_label: str | None` (set by auto dispatch); saved-run `method` metadata becomes `'auto:<label>'` for auto runs.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_logic.py`:

```python
def test_slp_and_auto_dispatch():
    from dvfopt import SLPStrategy

    vol2d = np.zeros((3, 1, 8, 8))
    w = SolverWorker(deformation_i=vol2d, method_id='slp_2tri')
    assert isinstance(w._build_strategy(), SLPStrategy)
    assert w._trajectory_metric_kind() == '2tri'

    # Auto on a mildly folded field resolves to a registry label and
    # records it on the worker.
    phi = np.zeros((3, 1, 8, 8))
    phi[2, 0, 3, 3] = 1.2
    phi[2, 0, 3, 4] = -1.2
    wa = SolverWorker(
        deformation_i=phi, method_id='auto_2tri', params={'objective_id': 'l1'}
    )
    strat = wa._build_strategy()
    assert strat is not None
    assert wa.resolved_strategy_label in ('m10', 'm14_schwarz', 'm14', 'barrier', 'slsqp')
    wj = SolverWorker(
        deformation_i=phi, method_id='auto_jdet', params={'objective_id': 'l1'}
    )
    wj._build_strategy()
    assert wj.resolved_strategy_label in ('barrier', 'slsqp_windowed')
```

Append to `tests/test_gui_app.py`:

```python
def test_slp_is_first_and_default_2tri(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / 'fresh.ini')
    monkeypatch.setattr(
        LiveSolverWindow,
        '_settings',
        staticmethod(lambda: QtCore.QSettings(ini, QtCore.QSettings.IniFormat)),
    )
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))  # fresh settings -> defaults
    win._select_combo_data(win._constraint_combo, '2tri')
    assert win._method_combo.itemData(0) == 'slp'
    assert win._method_combo.currentData() == 'slp'
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'auto' in algos
    win._select_combo_data(win._constraint_combo, 'jdet')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'auto' in algos
```

- [ ] **Step 2: RED** — `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_slp_and_auto_dispatch tests/test_gui_app.py::test_slp_is_first_and_default_2tri -v` → FAIL (`unknown method_id='slp_2tri'`; itemData(0) != 'slp').

- [ ] **Step 3: Implement.**

`dvfopt_gui/app.py` — `_METHOD_SPECS_2TRI` becomes (SLP first, auto last):

```python
_METHOD_SPECS_2TRI = [
    ('slp', 'SLP (champion: cluster trust-region SLP + HiGHS L1)'),
    ('m14', 'M14 (Harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz', 'M14-Schwarz (cluster decomposition + global polish)'),
    ('m10', 'M10 (Harmonic + ALM + barrier polish)'),
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed (live progress)'),
    ('slsqp_fullgrid', 'SLSQP full-grid (2-tri; KKT, smallest L1 on mild folds)'),
    ('schwarz', 'Schwarz (2-tri; overlapping-tile decomposition)'),
    ('auto', 'Auto (pick by fold stats)'),
]
```

Append `('auto', 'Auto (pick by fold stats)')` to `_METHOD_SPECS_JDET`.
Set `DEFAULT_METHOD_BY_CONSTRAINT[CONSTRAINT_2TRI] = 'slp'`.

`dvfopt_gui/worker.py` — in `SolverWorker.__init__` (after `self._callback_count = 0`):

```python
        # Set by the 'auto' dispatch: the registry label auto_strategy
        # resolved to (e.g. 'm14_schwarz'); None for explicit methods.
        self.resolved_strategy_label: str | None = None
```

In `_build_strategy`, add BEFORE the final `raise`:

```python
        if mid in ('auto_2tri', 'auto_jdet'):
            from dvfopt import (
                JdetConstraint2D,
                TriConstraint2DFullCoverage,
                make_strategy,
            )
            from dvfopt.solver import auto_strategy

            phi_2hw = np.stack(
                [
                    self._deformation_i[1, 0].astype(np.float64),
                    self._deformation_i[2, 0].astype(np.float64),
                ]
            )
            H, W = phi_2hw.shape[1:]
            kind = '2tri' if mid.endswith('_2tri') else 'jdet'
            n_neg, min_T = _metric_counts(phi_2hw, kind)
            constraint = (
                TriConstraint2DFullCoverage(shape=(H, W))
                if kind == '2tri'
                else JdetConstraint2D(shape=(H, W))
            )
            label = auto_strategy(
                constraint, n_neg, min_T, str(self._params.get('objective_id', 'l1'))
            )
            try:
                strategy = make_strategy(label)
            except Exception:
                # Registry label unavailable — fall back to the family default.
                label = 'm14' if kind == '2tri' else 'barrier'
                strategy = make_strategy(label)
            self.resolved_strategy_label = label
            return strategy
        if mid == 'slp_2tri':
            from dvfopt import SLPStrategy

            return SLPStrategy()
```

`dvfopt_gui/app.py` — surface the resolved label once per run. In
`_start_worker`, after `self._active_method_id = method_id`, add
`self._auto_label_shown = False`. In `_on_render_tick`, after the
`_update_progress()` call, add:

```python
        # One-time "Auto → <label>" note once the worker resolves it.
        if (
            not getattr(self, '_auto_label_shown', True)
            and self._worker is not None
            and getattr(self._worker, 'resolved_strategy_label', None)
        ):
            self._auto_label_shown = True
            self.statusBar().showMessage(
                f'Auto → {self._worker.resolved_strategy_label}', 8_000
            )
```

(Initialize `self._auto_label_shown = True` in `__init__` next to
`self._active_method_id` so the getattr default never fires mid-run.)

`_build_save_payload` — where `method=` is passed, replace with:

```python
        method = self._method_combo.currentData() or ''
        worker = self._worker
        if method == 'auto' and getattr(worker, 'resolved_strategy_label', None):
            method = f'auto:{worker.resolved_strategy_label}'
```

and pass `method=method`. (Note the existing local `worker` variable — reuse it.)

- [ ] **Step 4: GREEN** — rerun Step-2 command + `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py tests/test_gui_logic.py -q` → all PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py
git add dvfopt_gui/app.py dvfopt_gui/worker.py tests/test_gui_logic.py tests/test_gui_app.py
git commit -m "feat(gui): SLP champion (default) + Auto strategy picker"
```

---

### Task 3: Threshold spinbox end-to-end

**Files:**
- Modify: `dvfopt_gui/app.py` (method bar ~line 695-709; `_start_worker`; `_format_stats`; `_restore_settings`/`_save_settings`)
- Modify: `dvfopt_gui/worker.py` (`_run_via_solver`, `_run_via_solver_3d`, `_run_windowed_slsqp`)
- Test: `tests/test_gui_logic.py`, `tests/test_gui_app.py`

**Interfaces:**
- Produces: `self._thr_spin` (QDoubleSpinBox, decimals 4, range 0.0–1.0, step 0.005, default `FEASIBILITY_THRESHOLD`); `params['threshold']` consumed by all worker solve paths; `LiveSolverWindow._display_threshold() -> float` used by the stats panel.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_logic.py`:

```python
def test_threshold_param_reaches_solver(monkeypatch):
    import dvfopt

    captured = {}

    class _FakeSolver:
        def __init__(self, *, constraint, objective, strategy, threshold=None):
            captured['threshold'] = threshold

        def fit(self, phi, **kw):
            class R:
                corrected = np.zeros((2, 6, 6))

            return R()

    monkeypatch.setattr(dvfopt, 'Solver', _FakeSolver)
    w = SolverWorker(
        deformation_i=np.zeros((3, 1, 6, 6)),
        method_id='m14_2tri',
        params={'threshold': 0.02, 'objective_id': 'l1'},
    )
    w._run_via_solver(w._build_strategy(), '2tri', metric_kind='2tri')
    assert captured['threshold'] == pytest.approx(0.02)
```

Append to `tests/test_gui_app.py`:

```python
def test_threshold_spinbox_feeds_params_and_stats(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    assert win._thr_spin.value() == pytest.approx(0.01)
    win._thr_spin.setValue(0.05)
    captured = {}
    monkeypatch.setattr(
        'dvfopt_gui.worker.SolverWorker.start', lambda self: captured.setdefault('p', self._params)
    )
    win._on_run(use_roi=False)
    assert captured['p']['threshold'] == pytest.approx(0.05)
    # Idle stats use the spinbox threshold, not the module constant.
    assert '0.05' in win._format_stats(None)
```

- [ ] **Step 2: RED** — run both new tests → FAIL (`Solver` got no `threshold` / no attribute `_thr_spin`).

- [ ] **Step 3: Implement.**

`worker.py`:
- `_run_via_solver`: change `solver = Solver(constraint=constraint, objective=objective, strategy=strategy)` to add `threshold=self._params.get('threshold')` (Solver treats `None` as its default).
- `_run_via_solver_3d`: same addition.
- `_run_windowed_slsqp`: after the existing `kwargs` construction add:

```python
        if self._params.get('threshold') is not None:
            kwargs['threshold'] = float(self._params['threshold'])
```

`app.py`:
- In the method bar, after the `max_iter` spin widget block and before the Params button, add:

```python
        method_bar.addWidget(QtWidgets.QLabel('thr:'))
        self._thr_spin = QtWidgets.QDoubleSpinBox()
        self._thr_spin.setDecimals(4)
        self._thr_spin.setRange(0.0, 1.0)
        self._thr_spin.setSingleStep(0.005)
        self._thr_spin.setValue(FEASIBILITY_THRESHOLD)
        self._thr_spin.setToolTip(
            'Solver feasibility threshold: every constraint is enforced as '
            'C(phi) >= thr. Also drives the stats panel\'s infeasible(<thr) '
            'counts. Default 0.01 (package default).'
        )
        method_bar.addWidget(self._thr_spin)
```

- Add helper near `_max_abs_disp`:

```python
    def _display_threshold(self) -> float:
        """The user-selected feasibility threshold (spinbox), used for both
        solving and the stats panel's infeasible counts."""
        return float(self._thr_spin.value())
```

- `_start_worker`: add `'threshold': self._display_threshold(),` to the `params` dict.
- `_format_stats`: in the 2D idle branch replace `thr = FEASIBILITY_THRESHOLD` with `thr = self._display_threshold()` and pass it: `infeas_jdet = _infeasible_count(phi_2hw, 'jdet', thr)`, `infeas_tri = _infeasible_count(phi_2hw, '2tri', thr)`. In the 3D idle branch replace `thr = FEASIBILITY_THRESHOLD` and `_infeasible_count_3d(self._volume, kind)` with `_infeasible_count_3d(self._volume, kind, thr)`. In the snapshot branch replace the `feas_flag` comparison constant with `self._display_threshold()` (both the comparison and the label text).
- Persistence: in `_save_settings` add `s.setValue('threshold', self._display_threshold())`; in `_restore_settings` add:

```python
        thr = s.value('threshold', 0.0, type=float)
        if thr:
            self._thr_spin.setValue(thr)
```

- [ ] **Step 4: GREEN** — new tests + `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py tests/test_gui_logic.py -q` all PASS. (If an existing stats test asserts the literal `0.01` line, it still passes — the default is unchanged.)

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py
git add dvfopt_gui/app.py dvfopt_gui/worker.py tests/test_gui_logic.py tests/test_gui_app.py
git commit -m "feat(gui): user-editable solver threshold, wired to solvers and stats"
```

---

### Task 4: 3D metric cache (kill per-tick whole-volume recomputes)

**Files:**
- Modify: `dvfopt_gui/app.py` (`_heatmap_slice_3d`, `_format_stats` 3D idle, `_format_inspector` 3D, `_invalidate_inspector_cache` → `_invalidate_metric_caches` + all call sites, `__init__`, thr-spin hook)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: `_metric_field_3d(phi3d, kind)` from `dvfopt_gui.worker` (add to app.py's worker-import block).
- Produces: `LiveSolverWindow._metric3d_field(phi3d, kind) -> ndarray` — cached per `kind`, invalidated by `_invalidate_metric_caches()`. **Invariant:** between two invalidations, all 3D consumers pass the SAME phi (the render/refresh paths already invalidate before switching fields).

- [ ] **Step 1: Failing test.** Append to `tests/test_gui_app.py`:

```python
def test_3d_metric_cached_across_zscrub_and_hover(qapp, monkeypatch):
    import dvfopt_gui.app as A
    from dvfopt_gui.worker import _metric_field_3d as real_field
    from dvfopt_gui.worker import _volume_snapshot

    calls = {'n': 0}

    def counting(phi3d, kind):
        calls['n'] += 1
        return real_field(phi3d, kind)

    monkeypatch.setattr(A, '_metric_field_3d', counting)
    vol = np.zeros((3, 5, 8, 8))
    vol[2, :, 3:5, 3:5] = 1.4
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    snap = _volume_snapshot(vol, n_neg=5, min_T=-0.2, outer_iter=1)
    win._render_snapshot(snap)
    baseline = calls['n']
    assert baseline >= 1
    # z-scrub + inspector hover on the SAME field: zero new kernel runs.
    win._z_slider.setValue(2)
    win._z_slider.setValue(3)
    win._format_inspector((2, 2))
    win._format_inspector((3, 3))
    assert calls['n'] == baseline
    # A new snapshot invalidates and recomputes.
    win._render_snapshot(_volume_snapshot(vol * 0.5, n_neg=0, min_T=0.1, outer_iter=2))
    assert calls['n'] > baseline
```

- [ ] **Step 2: RED** — the current code calls `six_tet_min_volume_3d` directly inside `_heatmap_slice_3d`/`_format_inspector`, so either the monkeypatched counter is never hit (assert `baseline >= 1` fails) or z-scrub recomputes (equality fails). Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py::test_3d_metric_cached_across_zscrub_and_hover -v` → FAIL.

- [ ] **Step 3: Implement.**

- Add `_metric_field_3d` to app.py's `from dvfopt_gui.worker import (...)` block.
- `__init__` (next to `self._inspector_tri = None`): `self._metric3d_cache: dict = {}` (maps `kind -> ndarray`).
- New method next to `_triangle_areas_cached`:

```python
    def _metric3d_field(self, phi3d: np.ndarray, kind: str) -> np.ndarray:
        """Whole-volume 3D metric field, cached per kind until the displayed
        field changes (``_invalidate_metric_caches``). Counts are cheap numpy
        reductions over this array; only the kernel is expensive."""
        field = self._metric3d_cache.get(kind)
        if field is None:
            field = _metric_field_3d(phi3d, kind)
            self._metric3d_cache[kind] = field
        return field
```

- Rename `_invalidate_inspector_cache` → `_invalidate_metric_caches` (update ALL call sites — grep for the old name) and extend the body:

```python
    def _invalidate_metric_caches(self) -> None:
        """Drop cached per-field metrics (2D T1/T2 and 3D volume metric) —
        call whenever the displayed field changes."""
        self._inspector_tri = None
        self._metric3d_cache = {}
```

- `_heatmap_slice_3d`: replace the direct `six_tet_min_volume_3d(phi3d)` / `jacobian_det3D(phi3d)` computations with `field = self._metric3d_field(phi3d, 'jdet3d' if <jdet3d selected> else 'tet3d')` (keep the existing kind selection, NaN padding, and z clamping — for `jdet3d` the field is `(D,H,W)`, return `field[z]` directly as today).
- `_format_stats` 3D idle branch: replace `_metric_counts_3d(self._volume, kind)` + `_infeasible_count_3d(...)` with reductions over the cached field:

```python
                field = self._metric3d_field(self._volume, kind)
                n_neg = int((field <= 0).sum())
                min_T = float(field.min())
                thr = self._display_threshold()
                infeas = int((field < thr).sum())
```

- `_format_inspector` 3D branch: replace `mv = six_tet_min_volume_3d(phi3d)` with `mv = self._metric3d_field(phi3d, 'tet3d')` (drop the local import).
- `_on_z_changed` 3D branch already re-renders the current snapshot — no change needed (the cache makes it cheap).
- Thr-spin hookup (counts depend on thr only via reductions — no invalidation needed since the FIELD is thr-independent; but the stats must repaint): connect in `__init__` after creating the spin: `self._thr_spin.valueChanged.connect(lambda _v: self._refresh_display_from_volume() if self._worker is None else None)`.

- [ ] **Step 4: GREEN** — Step-1 test + `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -q` all PASS (existing 3D render/stats tests exercise the refactored paths).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_app.py
git add dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "perf(gui): cache the whole-volume 3D metric across z-scrub/hover/stats"
```

---

### Task 5: Undo byte budget + non-finite load rejection

**Files:**
- Modify: `dvfopt_gui/app.py` (`_UNDO_MAX` area, `_push_undo_state`, `_apply_loaded_run`)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Produces: module constant `UNDO_MAX_BYTES = 2 * 1024 ** 3`; pure helper `validate_finite(vol) -> str | None` (module-level in app.py, returns an error message or None) applied at the top of `_apply_loaded_run`.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_app.py`:

```python
def test_undo_stack_byte_budget(qapp, monkeypatch):
    import dvfopt_gui.app as A

    win = LiveSolverWindow(np.zeros((3, 1, 64, 64)))
    entry_bytes = win._volume.nbytes
    # Budget that fits exactly two entries.
    monkeypatch.setattr(A, 'UNDO_MAX_BYTES', int(entry_bytes * 2.5))
    for _ in range(5):
        win._push_undo_state()
    assert len(win._undo_stack) == 2
    assert sum(v.nbytes for v in win._undo_stack) <= entry_bytes * 2.5


def test_undo_budget_keeps_at_least_one(qapp, monkeypatch):
    import dvfopt_gui.app as A

    win = LiveSolverWindow(np.zeros((3, 1, 64, 64)))
    monkeypatch.setattr(A, 'UNDO_MAX_BYTES', 1)  # smaller than one entry
    win._push_undo_state()
    assert len(win._undo_stack) == 1


def test_nonfinite_load_rejected(qapp, monkeypatch):
    from dvfopt_gui.app import validate_finite
    from dvfopt_gui.persistence import LoadedRun

    bad = np.zeros((3, 1, 5, 5))
    bad[2, 0, 2, 2] = np.nan
    msg = validate_finite(bad)
    assert msg is not None and 'non-finite' in msg
    assert validate_finite(np.zeros((3, 1, 4, 4))) is None

    win = LiveSolverWindow()
    seen = {}
    monkeypatch.setattr(
        QtWidgets.QMessageBox, 'critical', staticmethod(lambda *a, **k: seen.setdefault('called', True))
    )
    prev = win._volume
    win._apply_loaded_run(LoadedRun(volume=bad))
    assert seen.get('called')
    assert win._volume is prev  # rejected load leaves state untouched
```

- [ ] **Step 2: RED** — run the three tests → FAIL (`UNDO_MAX_BYTES`/`validate_finite` missing).

- [ ] **Step 3: Implement** in `dvfopt_gui/app.py`:

Module level (near the other constants):

```python
# Byte budget for the undo stack. Full-volume snapshots are cheap for 2D
# slices but ~1.8 GB each for a B0039-scale float64 volume — a count cap
# alone (30) would allow ~55 GB. Oldest entries are evicted past this
# budget; the most recent entry is always retained so Undo keeps working.
UNDO_MAX_BYTES = 2 * 1024 ** 3


def validate_finite(vol: np.ndarray) -> str | None:
    """Return an error message if ``vol`` contains NaN/Inf, else None."""
    bad = ~np.isfinite(vol)
    n = int(bad.sum())
    if n == 0:
        return None
    first = tuple(int(i) for i in np.argwhere(bad)[0])
    return (
        f'The loaded field contains {n} non-finite value(s) (NaN/Inf); '
        f'first at index {first}. Fix the field before loading — solvers '
        'and fold metrics are undefined on non-finite data.'
    )
```

`_push_undo_state` — after the existing count-cap block, add:

```python
        while (
            len(self._undo_stack) > 1
            and sum(v.nbytes for v in self._undo_stack) > UNDO_MAX_BYTES
        ):
            self._undo_stack.pop(0)
```

`_apply_loaded_run` — at the very top (before `self._volume = ...`):

```python
        msg = validate_finite(np.asarray(run.volume))
        if msg is not None:
            QtWidgets.QMessageBox.critical(self, 'Invalid DVF', msg)
            return
```

- [ ] **Step 4: GREEN** — the three tests + full `tests/test_gui_app.py` PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_app.py
git add dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "fix(gui): byte-budgeted undo stack + reject non-finite DVFs on load"
```

---

### Task 6: sitk import/export + threaded load

**Files:**
- Create: `dvfopt_gui/io_formats.py`
- Modify: `dvfopt_gui/worker.py` (add `LoadWorker`)
- Modify: `dvfopt_gui/app.py` (`_on_load` refactor, File-menu Export action, `_build_menus`)
- Test: `tests/test_gui_io_formats.py` (new), `tests/test_gui_app.py`

**Interfaces:**
- Produces (io_formats): `SITK_EXTENSIONS = ('.nii', '.nii.gz', '.mha', '.mhd', '.nrrd')`; `sitk_available() -> bool`; `is_sitk_path(path) -> bool`; `load_dvf_sitk(path) -> np.ndarray (3,D,H,W) [dz,dy,dx] float64` (raises `ValueError` on non-2/3-component images); `save_dvf_sitk(path, vol)`.
- Produces (worker): `LoadWorker(QtCore.QThread)` with `path` ctor arg, signals `loadedRun = pyqtSignal(object)` (a `LoadedRun`) and `failed = pyqtSignal(str)`; its `run()` dispatches npy/npz → `parse_loaded`, sitk paths → `load_dvf_sitk` → `LoadedRun(volume=...)`.

- [ ] **Step 1: Failing tests.** Create `tests/test_gui_io_formats.py`:

```python
"""SimpleITK DVF import/export round-trip + LoadWorker dispatch."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('PyQt5', reason='dvfopt_gui requires the [gui] extra')
sitk = pytest.importorskip('SimpleITK', reason='sitk interop tests need SimpleITK')

from dvfopt_gui import io_formats


def _vol(D=3, H=4, W=5):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.2, (3, D, H, W)).astype(np.float64)


def test_roundtrip_nii(tmp_path):
    vol = _vol()
    p = tmp_path / 'field.nii.gz'
    io_formats.save_dvf_sitk(p, vol)
    back = io_formats.load_dvf_sitk(p)
    assert back.shape == vol.shape
    np.testing.assert_allclose(back, vol, atol=1e-6)


def test_channel_convention_matches_sitk_jdet(tmp_path):
    # Component order must follow dvfopt/jacobian/sitk_jdet.py: sitk stores
    # [dx, dy, dz]; our numpy layout is [dz, dy, dx].
    vol = np.zeros((3, 2, 3, 3))
    vol[0] = 1.0  # dz
    vol[2] = 3.0  # dx
    p = tmp_path / 'conv.mha'
    io_formats.save_dvf_sitk(p, vol)
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)  # (D,H,W,3) components [dx,dy,dz]
    assert arr[..., 0].max() == pytest.approx(3.0)  # dx component
    assert arr[..., 2].max() == pytest.approx(1.0)  # dz component


def test_load_rejects_scalar_image(tmp_path):
    img = sitk.GetImageFromArray(np.zeros((3, 4, 5)))
    p = tmp_path / 'scalar.nii'
    sitk.WriteImage(img, str(p))
    with pytest.raises(ValueError):
        io_formats.load_dvf_sitk(p)


def test_2d_vector_image_maps_to_single_slice(tmp_path):
    arr = np.zeros((4, 5, 2))  # (H,W,2) components [dx,dy]
    arr[..., 0] = 2.0  # dx
    img = sitk.GetImageFromArray(arr, isVector=True)
    p = tmp_path / 'twod.mha'
    sitk.WriteImage(img, str(p))
    vol = io_formats.load_dvf_sitk(p)
    assert vol.shape == (3, 1, 4, 5)
    assert vol[2].max() == pytest.approx(2.0)  # dx channel
    assert vol[0].max() == 0.0  # dz zero


def test_is_sitk_path():
    assert io_formats.is_sitk_path('x.nii.gz') and io_formats.is_sitk_path('X.MHA')
    assert not io_formats.is_sitk_path('x.npy')


def test_loadworker_npy_and_sitk(tmp_path, qapp_placeholder=None):
    from dvfopt_gui.worker import LoadWorker

    npy = tmp_path / 'f.npy'
    np.save(npy, np.zeros((3, 2, 4, 4)))
    results = []
    w = LoadWorker(str(npy))
    w.loadedRun.connect(lambda r: results.append(r))
    w.run()  # synchronous: exercise the body without a thread
    assert results and results[0].volume.shape == (3, 2, 4, 4)

    sp = tmp_path / 'f.nii.gz'
    io_formats.save_dvf_sitk(sp, _vol())
    w2 = LoadWorker(str(sp))
    w2.loadedRun.connect(lambda r: results.append(r))
    w2.run()
    assert results[-1].volume.shape == (3, 3, 4, 5)
```

- [ ] **Step 2: RED** — `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_io_formats.py -v` → FAIL (`No module named 'dvfopt_gui.io_formats'`).

- [ ] **Step 3: Implement.**

Create `dvfopt_gui/io_formats.py`:

```python
"""Displacement-field import/export via SimpleITK (NIfTI/MetaImage/NRRD).

Channel/axis convention mirrors :mod:`dvfopt.jacobian.sitk_jdet` (the
package's single source of truth): the canonical numpy layout is
``(3, D, H, W)`` with channels ``[dz, dy, dx]``; sitk vector images store
``(D, H, W, 3)`` arrays with components ``[dx, dy, dz]``. Conversion is a
``(1, 2, 3, 0)`` transpose plus a ``[2, 1, 0]`` component reorder.

SimpleITK is optional at runtime: :func:`sitk_available` gates the GUI's
file-dialog filters; the load/save functions import lazily.
"""

from __future__ import annotations

import numpy as np

SITK_EXTENSIONS = ('.nii', '.nii.gz', '.mha', '.mhd', '.nrrd')


def sitk_available() -> bool:
    """True when SimpleITK can be imported."""
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        return False
    return True


def is_sitk_path(path) -> bool:
    """True when ``path`` has a SimpleITK-handled extension."""
    lower = str(path).lower()
    return any(lower.endswith(ext) for ext in SITK_EXTENSIONS)


def load_dvf_sitk(path) -> np.ndarray:
    """Load a displacement field into the canonical ``(3, D, H, W)``
    ``[dz, dy, dx]`` float64 layout.

    Accepts 3-component 3D vector images and 2-component 2D vector images
    (mapped to a single-slice volume with ``dz = 0``). Raises ``ValueError``
    for anything else (e.g. scalar images).
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    ncomp = img.GetNumberOfComponentsPerPixel()
    if ncomp == 3 and arr.ndim == 4:
        arr = arr[..., [2, 1, 0]]  # components [dx,dy,dz] -> [dz,dy,dx]
        return np.ascontiguousarray(np.transpose(arr, (3, 0, 1, 2))).astype(np.float64)
    if ncomp == 2 and arr.ndim == 3:
        H, W = arr.shape[:2]
        vol = np.zeros((3, 1, H, W), dtype=np.float64)
        vol[1, 0] = arr[..., 1]  # dy
        vol[2, 0] = arr[..., 0]  # dx
        return vol
    raise ValueError(
        f'not a 2/3-component displacement field: array shape {arr.shape}, '
        f'{ncomp} component(s) per pixel'
    )


def save_dvf_sitk(path, vol) -> None:
    """Write ``(3, D, H, W)`` ``[dz, dy, dx]`` as a 3-component vector image."""
    import SimpleITK as sitk

    vol = np.asarray(vol, dtype=np.float64)
    if vol.ndim != 4 or vol.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W); got {vol.shape}')
    arr = np.transpose(vol, (1, 2, 3, 0))[..., [2, 1, 0]]  # -> (D,H,W,3) [dx,dy,dz]
    sitk.WriteImage(sitk.GetImageFromArray(arr, isVector=True), str(path))
```

Add `LoadWorker` to `dvfopt_gui/worker.py` (after `ReplayHistory`):

```python
class LoadWorker(QtCore.QThread):
    """Load a DVF file off the GUI thread.

    Dispatches by extension: ``.npy``/``.npz`` through
    :func:`dvfopt_gui.persistence.parse_loaded` (full saved-run support),
    SimpleITK formats through :func:`dvfopt_gui.io_formats.load_dvf_sitk`.
    Emits ``loadedRun`` with a ``LoadedRun`` on success, else ``failed``
    with a message. GB-scale ``np.load`` + float64 conversion no longer
    freeze the window.
    """

    loadedRun = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = str(path)

    def run(self):
        try:
            from dvfopt_gui.io_formats import is_sitk_path, load_dvf_sitk
            from dvfopt_gui.persistence import LoadedRun, parse_loaded

            if is_sitk_path(self._path):
                run = LoadedRun(volume=load_dvf_sitk(self._path))
            else:
                loaded = np.load(self._path, allow_pickle=False)
                try:
                    run = parse_loaded(loaded)
                finally:
                    if isinstance(loaded, np.lib.npyio.NpzFile):
                        loaded.close()
            self.loadedRun.emit(run)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
```

Refactor `app.py::_on_load` to use it (replace the whole body after the
`self._last_dir = ...` line):

```python
        flt = 'DVF files (*.npy *.npz'
        from dvfopt_gui.io_formats import SITK_EXTENSIONS, sitk_available

        if sitk_available():
            flt += ' ' + ' '.join(f'*{e}' for e in SITK_EXTENSIONS)
        flt += ');;NumPy arrays (*.npy);;NumPy compressed (*.npz)'
        if sitk_available():
            flt += ';;Medical images (' + ' '.join(f'*{e}' for e in SITK_EXTENSIONS) + ')'
        flt += ';;All files (*)'
```

(the `getOpenFileName` call uses `flt`), then:

```python
        self._load_btn.setEnabled(False)
        self.statusBar().showMessage(f'Loading {Path(path).name}…', 0)
        self._load_worker = LoadWorker(path, parent=self)
        self._load_worker.loadedRun.connect(lambda run: self._on_load_finished(path, run))
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.start()
```

with the two slots:

```python
    def _on_load_finished(self, path: str, run) -> None:
        self._load_btn.setEnabled(True)
        self._apply_loaded_run(run)
        n_hist = len(run.snapshots)
        suffix = f'  ({n_hist} history step(s))' if n_hist else ''
        self.statusBar().showMessage(f'Loaded {path}{suffix}', 5_000)

    def _on_load_failed(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Load failed', msg)
```

The Load button is currently a local `load_btn` variable — rename it to
`self._load_btn` where created. Import `LoadWorker` in the worker-import
block. Note the ValueError-vs-generic split from the old `_on_load` is
subsumed by the single failure dialog (message carries the type name).

Export action — in `_build_menus`, File menu after 'Save…':

```python
        file_menu.addAction('Export corrected DVF…', self._on_export)
```

and the handler:

```python
    def _on_export(self):
        """Write just the corrected volume (no run history) as .npy or, when
        SimpleITK is available, .nii.gz — for interop with the rest of the
        registration pipeline."""
        if self._volume is None:
            QtWidgets.QMessageBox.information(
                self, 'Nothing to export', 'Load a DVF first via "Load DVF…".'
            )
            return
        from dvfopt_gui.io_formats import save_dvf_sitk, sitk_available

        filters = 'NumPy array (*.npy)'
        if sitk_available():
            filters += ';;NIfTI (*.nii.gz)'
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export corrected DVF', str(Path(self._last_dir) / 'corrected_dvf.npy'), filters
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        try:
            if 'NIfTI' in chosen or path.lower().endswith(('.nii', '.nii.gz')):
                if not path.lower().endswith(('.nii', '.nii.gz')):
                    path += '.nii.gz'
                save_dvf_sitk(path, self._volume)
            else:
                if not path.lower().endswith('.npy'):
                    path += '.npy'
                np.save(path, self._volume)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Export failed', f'{type(exc).__name__}: {exc}')
            return
        self.statusBar().showMessage(f'Exported {path}', 8_000)
```

Widget test — append to `tests/test_gui_app.py`:

```python
def test_export_writes_npy(qapp, tmp_path, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 2, 5, 5)))
    out = tmp_path / 'corr.npy'
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **k: (str(out), 'NumPy array (*.npy)')),
    )
    win._on_export()
    assert out.exists() and np.load(out).shape == (3, 2, 5, 5)


def test_load_worker_path_used_by_on_load(qapp, tmp_path, monkeypatch):
    # _on_load must go through LoadWorker (GUI thread does no np.load).
    npy = tmp_path / 'f.npy'
    np.save(npy, np.zeros((3, 1, 6, 6)))
    win = LiveSolverWindow()
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *a, **k: (str(npy), '')),
    )
    win._on_load()
    win._load_worker.wait(10_000)
    for _ in range(50):
        QtWidgets.QApplication.processEvents()
    assert win._volume is not None and win._volume.shape == (3, 1, 6, 6)
    assert win._load_btn.isEnabled()
```

- [ ] **Step 4: GREEN** — `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_io_formats.py tests/test_gui_app.py -q` → PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_io_formats.py tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_io_formats.py tests/test_gui_app.py
git add dvfopt_gui/io_formats.py dvfopt_gui/worker.py dvfopt_gui/app.py tests/test_gui_io_formats.py tests/test_gui_app.py
git commit -m "feat(gui): sitk DVF import/export + threaded, non-blocking load"
```

---

### Task 7: Worker runners — 2.5D marching, full-3D pipeline, torch barrier

**Files:**
- Modify: `dvfopt_gui/worker.py` (`run()` dispatch, `_build_strategy`, new `_run_marching_25d`, `_run_pipeline_3d`, `__init__` attrs)
- Modify: `dvfopt_gui/app.py` (`_METHOD_SPECS_TET3D` + torch gating in `_repopulate_method_combo`, `_update_progress` marching branch, `_on_finished` report surfacing)
- Test: `tests/test_gui_logic.py`, `tests/test_gui_app.py`

**Interfaces:**
- Consumes: Task 1's `progress_callback` contract; Task 3's `params['threshold']`; `dvfopt.correct_dvf_25d`, `dvfopt.correct_dvf_3d` (both return `(phi_out, report)`); `dvfopt.BarrierTet3DTorchStrategy`.
- Produces: method ids `marching25d_tet3d` (no menu entry — launched by Task 8's Pipeline UI), `pipeline3d_tet3d`, `barrier_torch_tet3d` (both in the tet3d menu; torch item disabled when torch missing). Worker attrs: `self.pipeline_report` (the 25d/3d report or None), `self.marching_progress` (last `(phase, index, total, n_neg)` tuple or None). Module helper `_torch_available() -> bool` in app.py.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_logic.py`:

```python
def test_marching_and_pipeline3d_dispatch(monkeypatch):
    vol = np.zeros((3, 4, 8, 8))
    called = {}
    monkeypatch.setattr(
        SolverWorker, '_run_marching_25d', lambda self: called.setdefault('m', True) or vol
    )
    w = SolverWorker(deformation_i=vol, method_id='marching25d_tet3d')
    w.run()
    assert called.get('m'), 'marching runner not dispatched'
    assert w._trajectory_metric_kind() == 'tet3d'

    monkeypatch.setattr(
        SolverWorker, '_run_pipeline_3d', lambda self: called.setdefault('p', True) or vol
    )
    w2 = SolverWorker(deformation_i=vol, method_id='pipeline3d_tet3d')
    w2.run()
    assert called.get('p'), 'pipeline3d runner not dispatched'


def test_marching_25d_end_to_end():
    # Per-slice-feasible, inter-layer-folded fixture (same as the library test).
    vol = np.zeros((3, 4, 8, 8))
    for k in range(4):
        vol[2, k, :, 4] = 0.7 if k % 2 == 0 else -0.7
    from dvfopt_gui.worker import _metric_counts_3d

    n0, _ = _metric_counts_3d(vol, 'tet3d')
    assert n0 > 0
    w = SolverWorker(
        deformation_i=vol, method_id='marching25d_tet3d', params={'threshold': 0.01}
    )
    out = w._run_marching_25d()
    n1, _ = _metric_counts_3d(out, 'tet3d')
    assert n1 < n0
    assert w.pipeline_report is not None and w.pipeline_report.n_neg_out == n1
    assert w.history_len() >= 2 and w.history_get(0).phi.ndim == 4


def test_pipeline3d_dispatch_uses_stub(monkeypatch):
    import dvfopt

    vol = np.zeros((3, 4, 8, 8))

    class _R:
        n_neg_in, n_neg_out, feasible, wall_s = 3, 0, True, 0.1

    monkeypatch.setattr(dvfopt, 'correct_dvf_3d', lambda v, **kw: (v.copy(), _R()))
    w = SolverWorker(deformation_i=vol, method_id='pipeline3d_tet3d', params={'threshold': 0.02})
    out = w._run_pipeline_3d()
    assert out.shape == vol.shape
    assert w.pipeline_report is not None


def test_barrier_torch_dispatch():
    pytest.importorskip('torch', reason='torch barrier needs torch')
    from dvfopt import BarrierTet3DTorchStrategy

    w = SolverWorker(deformation_i=np.zeros((3, 4, 8, 8)), method_id='barrier_torch_tet3d')
    assert isinstance(w._build_strategy(), BarrierTet3DTorchStrategy)
```

Append to `tests/test_gui_app.py`:

```python
def test_tet3d_menu_pipeline_and_torch_gating(qapp, monkeypatch):
    import dvfopt_gui.app as A

    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'pipeline3d' in algos
    assert 'barrier_torch' in algos
    # Torch missing -> the item is disabled (greyed), still listed.
    monkeypatch.setattr(A, '_torch_available', lambda: False)
    win._repopulate_method_combo('tet3d')
    idx = win._method_combo.findData('barrier_torch')
    assert idx >= 0
    assert not win._method_combo.model().item(idx).isEnabled()
```

- [ ] **Step 2: RED** — run the five tests → FAIL (unknown ids / missing attrs / menu entries absent).

- [ ] **Step 3: Implement.**

`worker.py` `__init__` (after `resolved_strategy_label`):

```python
        # Set by the pipeline runners (_run_marching_25d / _run_pipeline_3d):
        # the Correct25DReport / Correct3DReport for status display, and the
        # last 2.5D progress event for the progress bar.
        self.pipeline_report = None
        self.marching_progress: tuple | None = None
```

`_build_strategy` — before the final `raise`:

```python
        if mid == 'barrier_torch_tet3d':
            from dvfopt import BarrierTet3DTorchStrategy

            return BarrierTet3DTorchStrategy()
```

`run()` — the dispatch currently reads `if mid == 'slsqp_windowed_jdet': ... elif mid == 'slsqp_windowed_2tri': ... else:`. Insert two branches before the `else`:

```python
            elif mid == 'marching25d_tet3d':
                phi_out = self._run_marching_25d()
            elif mid == 'pipeline3d_tet3d':
                phi_out = self._run_pipeline_3d()
```

New runners (after `_run_via_solver_3d`):

```python
    def _run_marching_25d(self):
        """Whole-volume 2.5D marching (fold PREVENTION): sweep + mop via
        ``correct_dvf_25d``. Input is the CURRENT (per-slice-corrected)
        volume the window handed us — the pipeline's precondition is
        dz == 0, which per-slice 2D correction guarantees."""
        from dvfopt import correct_dvf_25d

        vol = np.asarray(self._deformation_i, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[0] != 3:
            raise ValueError(f'2.5D marching needs (3, D, H, W); got {vol.shape}')
        _, D, H, W = vol.shape
        thr = self._params.get('threshold')
        thr = float(thr) if thr is not None else 0.01

        n0, m0 = _metric_counts_3d(vol, 'tet3d')
        self._record(_volume_snapshot(vol, n_neg=n0, min_T=m0, outer_iter=0))

        est = DEFAULT_HISTORY_MAX_3D * 3 * D * H * W * 8
        keep_stages = est <= MAX_3D_HISTORY_BYTES
        stride = max(1, D // 6)
        outer = [0]

        def _cb(event):
            if self._stop_requested:
                raise KeyboardInterrupt('user requested stop')
            self.marching_progress = (
                event['phase'], event['index'], event['total'], event['n_neg']
            )
            if not keep_stages:
                return
            if event['phase'] == 'sweep' and event['index'] % stride != 0:
                return
            outer[0] += 1
            n, m = _metric_counts_3d(event['phi'], 'tet3d')
            self._record(
                _volume_snapshot(event['phi'], n_neg=n, min_T=m, outer_iter=outer[0])
            )

        if self._stop_requested:
            raise KeyboardInterrupt()
        phi_out, report = correct_dvf_25d(
            vol, threshold=thr, verbose=0, progress_callback=_cb
        )
        self.pipeline_report = report
        nf, mf = _metric_counts_3d(phi_out, 'tet3d')
        self._record(_volume_snapshot(phi_out, n_neg=nf, min_T=mf, outer_iter=outer[0] + 1))
        return np.asarray(phi_out, dtype=np.float64)

    def _run_pipeline_3d(self):
        """One-shot end-to-end 3D orchestrator (``correct_dvf_3d``): bulk
        recovery + k-ring escape. No progress hook exists — init + final
        snapshots only; Stop is best-effort (checked before launch)."""
        import dvfopt

        vol = np.asarray(self._deformation_i, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[0] != 3:
            raise ValueError(f'3D pipeline needs (3, D, H, W); got {vol.shape}')
        thr = self._params.get('threshold')
        thr = float(thr) if thr is not None else 0.01

        n0, m0 = _metric_counts_3d(vol, 'tet3d')
        self._record(_volume_snapshot(vol, n_neg=n0, min_T=m0, outer_iter=0))
        if self._stop_requested:
            raise KeyboardInterrupt()
        phi_out, report = dvfopt.correct_dvf_3d(vol, threshold=thr, verbose=0)
        self.pipeline_report = report
        phi_out = np.asarray(phi_out, dtype=np.float64)
        nf, mf = _metric_counts_3d(phi_out, 'tet3d')
        self._record(_volume_snapshot(phi_out, n_neg=nf, min_T=mf, outer_iter=1))
        return phi_out
```

(`_run_pipeline_3d` uses `dvfopt.correct_dvf_3d` via the module attribute so
tests can monkeypatch `dvfopt.correct_dvf_3d`.)

`app.py`:
- Module helper near the other module functions:

```python
def _torch_available() -> bool:
    """True when PyTorch is importable (gates the GPU-barrier menu item)."""
    import importlib.util

    return importlib.util.find_spec('torch') is not None
```

- `_METHOD_SPECS_TET3D` — append:

```python
    ('pipeline3d', 'Full 3D pipeline (bulk auto + k-ring escape)'),
    ('barrier_torch', 'Barrier GPU (torch; CPU fallback)'),
```

- `_repopulate_method_combo` — after the population loop, add:

```python
        # GPU barrier needs torch; keep it visible but disabled when absent.
        idx = self._method_combo.findData('barrier_torch')
        if idx >= 0 and not _torch_available():
            self._method_combo.model().item(idx).setEnabled(False)
```

- `_update_progress` — add a branch BEFORE the `startswith(('m10', 'm14'))` branch:

```python
        if mid == 'marching25d_tet3d':
            prog = getattr(worker, 'marching_progress', None)
            if prog is not None:
                phase, index, total, n_neg = prog
                self._progress.setRange(0, max(1, int(total)))
                self._progress.setValue(int(index))
                self._progress.setFormat(f'{phase} {index}/{total} · n_neg {n_neg}')
            else:
                self._progress.setRange(0, 0)
                self._progress.setFormat(f'{elapsed:.0f}s')
            return
        if mid == 'pipeline3d_tet3d':
            self._progress.setRange(0, 0)
            self._progress.setFormat(f'{elapsed:.0f}s')
            return
```

- `_on_finished` — after the splice/`_refresh_display_from_volume()` block, surface reports:

```python
        report = getattr(self.sender(), 'pipeline_report', None)
        if report is not None:
            self.statusBar().showMessage(
                f'Pipeline: {report.n_neg_in} → {report.n_neg_out} folds, '
                f'feasible={report.feasible}, {report.wall_s:.0f}s',
                15_000,
            )
```

- [ ] **Step 4: GREEN** — Step-1 tests + full GUI suites PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py
git add dvfopt_gui/worker.py dvfopt_gui/app.py tests/test_gui_logic.py tests/test_gui_app.py
git commit -m "feat(gui): 2.5D marching + full-3D pipeline runners, torch barrier entry"
```

---

### Task 8: Pipeline ▾ UI + full-pipeline chaining

**Files:**
- Modify: `dvfopt_gui/app.py` (Pipeline ▾ button between Run-all and Stop; Run-menu actions; `_on_run_25d`, `_on_run_pipeline_full`, `_start_marching_25d`, `_begin_run_all_batch` extraction; `_run_all_step` chain trigger; `_on_finished`/`_on_error`/`_finalize_run_ui` flag handling; `_start_worker` `method_id` override + pipeline-btn disable; `_apply_mode_gating` enable rule)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: Task 7's `marching25d_tet3d` runner.
- Produces: `self._pipeline_btn` (QToolButton, InstantPopup menu with 'Run 2.5D marching' + 'Full pipeline (2D + 2.5D)'); flags `self._pipeline_active: bool`, `self._pipeline_after_run_all: bool`; `_start_worker(deformation_i, method_id=None)` override kwarg; `_begin_run_all_batch()` (batch start without undo push).

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_app.py`:

```python
def test_pipeline_button_exists_and_gates(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))  # 2D: disabled
    assert hasattr(win, '_pipeline_btn')
    assert not win._pipeline_btn.isEnabled()
    win._load_array(np.zeros((3, 4, 6, 6)))  # volume: enabled
    assert win._pipeline_btn.isEnabled()


def test_run_25d_rejects_nonzero_dz(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._volume[0, 1, 2, 2] = 0.5  # nonzero dz
    asked = {}
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        'question',
        staticmethod(lambda *a, **k: asked.setdefault('q', True) and QtWidgets.QMessageBox.No),
    )
    started = {}
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: started.setdefault('s', True))
    win._on_run_25d()
    assert asked.get('q'), 'dz violation must prompt'
    assert not started.get('s'), 'declined prompt must not start a run'


def test_run_25d_starts_marching_on_current_volume(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._volume[2, 1, 2, 2] = 0.3  # differs from _original_volume
    captured = {}

    def fake_start(def_i, method_id=None):
        captured['shape'] = def_i.shape
        captured['mid'] = method_id
        captured['val'] = float(def_i[2, 1, 2, 2])

    monkeypatch.setattr(win, '_start_worker', fake_start)
    win._on_run_25d()
    assert captured['mid'] == 'marching25d_tet3d'
    assert captured['shape'] == (3, 4, 6, 6)
    assert captured['val'] == pytest.approx(0.3)  # CURRENT volume, not original


def test_full_pipeline_chains_25d_after_batch(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 3, 6, 6)))
    monkeypatch.setattr(win, '_start_worker', lambda *a, **k: None)
    win._on_run_pipeline_full()
    assert win._pipeline_active and win._pipeline_after_run_all
    assert len(win._undo_stack) == 1  # exactly one entry for the whole pipeline
    # Simulate the batch draining to completion.
    started = {}
    monkeypatch.setattr(
        win, '_start_marching_25d', lambda: started.setdefault('m', True)
    )
    win._run_all_remaining = []
    win._run_all_step()
    assert started.get('m'), '2.5D stage must start after the batch'
    assert not win._pipeline_after_run_all
```

- [ ] **Step 2: RED** — run the four tests → FAIL (no `_pipeline_btn` / `_on_run_25d` / flags).

- [ ] **Step 3: Implement** in `app.py`.

`__init__` state (next to `_run_all_remaining`):

```python
        # Full-pipeline (per-slice 2D -> 2.5D marching) state. `_pipeline_active`
        # suppresses the per-run undo push (ONE entry covers the whole
        # pipeline); `_pipeline_after_run_all` arms the 2.5D stage to start
        # when the Run-all batch drains.
        self._pipeline_active = False
        self._pipeline_after_run_all = False
```

Toolbar — after `bar.addWidget(self._run_all_btn)`:

```python
        self._pipeline_btn = QtWidgets.QToolButton()
        self._pipeline_btn.setText('Pipeline ▾')
        self._pipeline_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._pipeline_btn.setToolTip(
            'Volume workflows: 2.5D marching (fold prevention; needs dz == 0, '
            'i.e. per-slice-corrected input) or the full pipeline (per-slice '
            '2D with the selected method, then 2.5D marching).'
        )
        pipe_menu = QtWidgets.QMenu(self._pipeline_btn)
        self._act_run_25d = pipe_menu.addAction('Run 2.5D marching', self._on_run_25d)
        self._act_run_pipeline = pipe_menu.addAction(
            'Full pipeline (2D + 2.5D)', self._on_run_pipeline_full
        )
        self._pipeline_btn.setMenu(pipe_menu)
        self._pipeline_btn.setEnabled(False)
        bar.addWidget(self._pipeline_btn)
```

Run menu (in `_build_menus`, after 'Run all z'):

```python
        run_menu.addAction('Run 2.5D marching', self._on_run_25d)
        run_menu.addAction('Full pipeline (2D + 2.5D)', self._on_run_pipeline_full)
```

Gating — in `_apply_mode_gating`, add:

```python
        self._pipeline_btn.setEnabled(D > 1)
```

and in `_start_worker`'s button-disable block add `self._pipeline_btn.setEnabled(False)` (re-enabled by `_finalize_run_ui` → `_apply_mode_gating`).

`_start_worker` — add the override kwarg. Change the signature to
`def _start_worker(self, deformation_i: np.ndarray, method_id: str | None = None):`
and where `method_id = _compose_method_id(algo, constraint)` is computed, wrap:

```python
        if method_id is None:
            algo = self._method_combo.currentData()
            constraint = self._constraint_combo.currentData()
            method_id = _compose_method_id(algo, constraint)
        else:
            constraint = self._constraint_combo.currentData()
```

(the objective/budget/max-iter param assembly below is unchanged; the
baseline `_input_n_neg` 3D branch keys off `self._is_3d_run`, which the
2.5D starters set by switching the constraint combo first).

Undo suppression — in `_on_finished`, the existing
`if self._run_all_remaining is None: self._push_undo_state()` becomes:

```python
            if self._run_all_remaining is None and not self._pipeline_active:
                self._push_undo_state()
```

Batch-start extraction — in `_on_run_all`, replace the final two lines
(`self._run_all_remaining = list(range(D))` / `self._run_all_step()`) with
`self._begin_run_all_batch()` and add:

```python
    def _begin_run_all_batch(self) -> None:
        """Start the per-slice batch WITHOUT pushing an undo entry (callers
        own the undo semantics: Run-all pushes one; the full pipeline pushes
        one covering both stages)."""
        D = self._volume.shape[1]
        self._run_all_remaining = list(range(D))
        self._run_all_step()
```

New handlers:

```python
    def _on_run_25d(self):
        """Run 2.5D marching on the CURRENT volume (which must be per-slice
        corrected: dz == 0)."""
        if self._volume is None:
            QtWidgets.QMessageBox.information(self, 'No DVF', 'Load a DVF first via "Load DVF…".')
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(self, 'Already running', 'Stop the current run first.')
            return
        if self._volume.shape[1] <= 1:
            QtWidgets.QMessageBox.information(
                self, '2.5D needs a volume', '2.5D marching needs a (3, D>1, H, W) volume.'
            )
            return
        if float(np.abs(self._volume[0]).max()) > 1e-9:
            ans = QtWidgets.QMessageBox.question(
                self,
                'dz is not zero',
                '2.5D marching requires dz == 0 (per-slice 2D-corrected input).\n'
                'Run the full pipeline (per-slice 2D, then 2.5D) instead?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans == QtWidgets.QMessageBox.Yes:
                self._on_run_pipeline_full()
            return
        self._start_marching_25d()

    def _start_marching_25d(self) -> None:
        """Launch the 2.5D worker on the CURRENT volume (the deliberate
        exception to the runs-read-the-pristine-original rule — the 2.5D
        input IS the per-slice-corrected state)."""
        self._select_combo_data(self._constraint_combo, CONSTRAINT_TET3D)
        self._section_bounds = None
        self.statusBar().showMessage('2.5D marching…', 0)
        self._start_worker(self._volume.copy(), method_id='marching25d_tet3d')

    def _on_run_pipeline_full(self):
        """One-click production workflow: per-slice 2D (selected method) →
        2.5D marching, as a single undoable operation."""
        if self._volume is None:
            QtWidgets.QMessageBox.information(self, 'No DVF', 'Load a DVF first via "Load DVF…".')
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(self, 'Already running', 'Stop the current run first.')
            return
        if self._volume.shape[1] <= 1:
            QtWidgets.QMessageBox.information(
                self, 'Pipeline needs a volume', 'The full pipeline needs a (3, D>1, H, W) volume.'
            )
            return
        if self._is_3d_run:
            # Per-slice stage needs a 2D method; drop back to the 2-tri family.
            self._select_combo_data(self._constraint_combo, DEFAULT_CONSTRAINT)
        self._push_undo_state()
        self._pipeline_active = True
        self._pipeline_after_run_all = True
        self.statusBar().showMessage('Pipeline: per-slice 2D…', 0)
        self._begin_run_all_batch()
```

Chain trigger — in `_run_all_step`'s batch-finished branch (where it
currently does `self._run_all_remaining = None; self._finalize_run_ui();
status 'Run all z finished.'`), insert BEFORE those lines:

```python
        if not self._run_all_remaining and self._pipeline_after_run_all:
            self._run_all_remaining = None
            self._pipeline_after_run_all = False
            self.statusBar().showMessage('Pipeline: 2.5D marching…', 0)
            self._start_marching_25d()
            return
```

Flag cleanup — in `_on_error` and in the `_on_finished` stopped-batch branch
(`info is not None` inside the run-all block), add
`self._pipeline_active = False; self._pipeline_after_run_all = False`. In
`_finalize_run_ui`, when `self._run_all_remaining is None`, also add
`self._pipeline_active = False` (the pipeline is over when its last worker
finalizes; the flag only needs to live until the 2.5D stage's undo check,
which happens in `_on_finished` BEFORE `_finalize_run_ui` runs).

- [ ] **Step 4: GREEN** — Step-1 tests + full `tests/test_gui_app.py` PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_app.py
git add dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "feat(gui): Pipeline menu — 2.5D marching + one-click 2D+2.5D workflow"
```

---

### Task 9: Per-slice fold overview strip

**Files:**
- Create: `dvfopt_gui/overview.py`
- Modify: `dvfopt_gui/app.py` (instantiate + wire)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Produces: `SliceOverviewStrip(pg.PlotWidget)` with `sliceClicked = pyqtSignal(int)`, `set_counts(counts: np.ndarray | None)`, `set_current(z: int)`; `OverviewWorker(QtCore.QThread)` with ctor `(volume)` and `chunkReady = pyqtSignal(int, object)` (start index, counts array for a chunk), plus `cancel()`.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_app.py`:

```python
def test_overview_strip_counts_and_click(qapp):
    from dvfopt_gui.overview import OverviewWorker, SliceOverviewStrip
    from dvfopt_gui.worker import _metric_counts

    vol = np.zeros((3, 4, 8, 8))
    vol[2, 2, 3, 3] = 1.2  # slice 2 has 2-tri folds
    vol[2, 2, 3, 4] = -1.2

    # Worker computes per-slice 2-tri fold counts (run synchronously).
    got = {}
    w = OverviewWorker(vol)
    w.chunkReady.connect(lambda start, arr: got.setdefault(start, np.asarray(arr)))
    w.run()
    counts = np.concatenate([got[k] for k in sorted(got)])
    assert counts.shape == (4,)
    assert counts[2] == _metric_counts(vol[1:, 2], '2tri')[0] > 0
    assert counts[0] == 0

    strip = SliceOverviewStrip()
    clicks = []
    strip.sliceClicked.connect(clicks.append)
    strip.set_counts(counts)
    strip.set_current(1)
    strip._emit_click_at(2.4)  # test hook: x-coordinate -> slice index
    assert clicks == [2]


def test_overview_strip_wired_into_window(qapp):
    vol = np.zeros((3, 4, 8, 8))
    vol[2, 1, 3, 3] = 1.2
    vol[2, 1, 3, 4] = -1.2
    win = LiveSolverWindow(vol)
    assert win._overview_strip.isVisibleTo(win)
    win._overview_worker.wait(10_000)
    for _ in range(50):
        QtWidgets.QApplication.processEvents()
    assert win._overview_counts is not None and win._overview_counts[1] > 0
    win._overview_strip.sliceClicked.emit(3)
    assert win._z_slider.value() == 3
    # 2D single-slice field hides the strip.
    win._load_array(np.zeros((3, 1, 6, 6)))
    assert not win._overview_strip.isVisibleTo(win)
```

- [ ] **Step 2: RED** — `No module named 'dvfopt_gui.overview'`.

- [ ] **Step 3: Implement.**

Create `dvfopt_gui/overview.py`:

```python
"""Per-slice fold overview strip for (3, D>1, H, W) volumes.

A thin clickable bar chart under the plot: x = slice index, y = per-slice
2-tri fold count. Instantly answers "which of my 528 slices are bad" and
doubles as navigation (click → jump z). Counts are computed off the GUI
thread by :class:`OverviewWorker`, streamed in chunks so the strip fills
progressively on big volumes.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from dvfopt_gui.worker import _metric_counts

_CHUNK = 32


class OverviewWorker(QtCore.QThread):
    """Compute per-slice 2-tri fold counts; emit ``(start, counts)`` chunks."""

    chunkReady = QtCore.pyqtSignal(int, object)

    def __init__(self, volume: np.ndarray, parent=None):
        super().__init__(parent)
        # Copy: the window's volume gets spliced by finishing runs while we
        # read it from this thread.
        self._volume = np.asarray(volume, dtype=np.float64).copy()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self):
        D = self._volume.shape[1]
        for start in range(0, D, _CHUNK):
            if self._cancel:
                return
            end = min(D, start + _CHUNK)
            counts = np.empty(end - start, dtype=np.int64)
            for i, z in enumerate(range(start, end)):
                if self._cancel:
                    return
                counts[i] = _metric_counts(self._volume[1:, z], '2tri')[0]
            self.chunkReady.emit(start, counts)


class SliceOverviewStrip(pg.PlotWidget):
    """Fixed-height clickable bar chart of per-slice fold counts."""

    sliceClicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background='w')
        self.setFixedHeight(44)
        pi = self.getPlotItem()
        pi.hideAxis('left')
        pi.setMenuEnabled(False)
        pi.setMouseEnabled(x=False, y=False)
        pi.hideButtons()
        self._bars = pg.BarGraphItem(x=[], height=[], width=0.9, brush='#e67e22')
        pi.addItem(self._bars)
        self._marker = pg.InfiniteLine(angle=90, pen=pg.mkPen('#000', width=2))
        pi.addItem(self._marker)
        self._marker.hide()
        self._n = 0

    def set_counts(self, counts) -> None:
        if counts is None:
            self._bars.setOpts(x=[], height=[])
            self._n = 0
            return
        counts = np.asarray(counts)
        self._n = len(counts)
        self._bars.setOpts(x=np.arange(self._n), height=counts)
        self.getPlotItem().setXRange(-0.5, max(0.5, self._n - 0.5), padding=0)

    def set_current(self, z: int) -> None:
        self._marker.setValue(int(z))
        self._marker.show()

    def _emit_click_at(self, x: float) -> None:
        """Map a view x-coordinate to a slice index and emit (test hook)."""
        if self._n == 0:
            return
        z = int(round(x))
        if 0 <= z < self._n:
            self.sliceClicked.emit(z)

    def mousePressEvent(self, ev):
        vb = self.getPlotItem().vb
        if self._n and self.sceneBoundingRect().contains(ev.pos()):
            point = vb.mapSceneToView(ev.pos())
            self._emit_click_at(point.x())
        super().mousePressEvent(ev)
```

Wire into `app.py`:
- Import: `from dvfopt_gui.overview import OverviewWorker, SliceOverviewStrip`.
- `__init__` after the split layout is added to `outer` (before the history
  row): create + hide:

```python
        # Per-slice fold overview (volumes only): computed in the background,
        # click to jump z.
        self._overview_strip = SliceOverviewStrip()
        self._overview_strip.setVisible(False)
        self._overview_strip.sliceClicked.connect(self._z_slider.setValue)
        outer.addWidget(self._overview_strip)
        self._overview_worker: OverviewWorker | None = None
        self._overview_counts: np.ndarray | None = None
```

- New methods:

```python
    def _restart_overview(self) -> None:
        """(Re)compute the per-slice fold counts in the background. Called on
        load and whenever a finished run splices the volume."""
        if self._overview_worker is not None and self._overview_worker.isRunning():
            self._overview_worker.cancel()
            self._overview_worker.wait(2_000)
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
        if self._overview_counts is None:
            return
        counts = np.asarray(counts)
        self._overview_counts[start : start + len(counts)] = counts
        self._overview_strip.set_counts(self._overview_counts)
```

- Call `self._restart_overview()` at the end of `_apply_loaded_run` and in
  `_on_finished` right after `self._refresh_display_from_volume()`.
- In `_on_z_changed` (both 2D and 3D paths, before returning) add
  `self._overview_strip.set_current(self._z)`.

- [ ] **Step 4: GREEN** — Step-1 tests + full `tests/test_gui_app.py` PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_app.py
git add dvfopt_gui/overview.py dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "feat(gui): clickable per-slice fold overview strip (background-computed)"
```

---

### Task 10: Auto-generated strategy params panel

**Files:**
- Create: `dvfopt_gui/strategy_params.py`
- Modify: `dvfopt_gui/app.py` (`ParamsDialog` gains a Strategy tab; `_on_open_params`; `_start_worker` passes overrides; settings persistence)
- Modify: `dvfopt_gui/worker.py` (`_build_strategy` applies overrides)
- Test: `tests/test_gui_logic.py`, `tests/test_gui_app.py`

**Interfaces:**
- Produces (strategy_params): `strategy_class_for(algo: str)` → Strategy class or None (mapping for: slp, m14, m14_schwarz, m10, barrier, slsqp_windowed, slsqp_fullgrid, schwarz, nmvf→NMVFStrategy, barrier_torch, and the 3D algo names; auto/pipeline3d/marching25d → None); `editable_fields(cls) -> list[tuple[name, kind, default]]` with kind ∈ `('int','float','bool','choice','str','readonly')` (`accuracy` → choice `('fast','max')`; `time_budget_s` excluded; tuple/None-typed defaults → readonly); `StrategyParamsTab(QWidget)` with `.build(algo, overrides: dict)` and `.values() -> dict` (only values differing from defaults).
- Produces (worker): `params['strategy_overrides']` (dict) merged into strategy construction; invalid overrides raise `ValueError` (surfaces via the existing `errored` dialog).
- Produces (window): `self._strategy_overrides: dict[str, dict]` persisted to QSettings key `'strategy_overrides'` as JSON.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_logic.py`:

```python
def test_strategy_params_introspection():
    from dvfopt import SLPStrategy
    from dvfopt_gui.strategy_params import editable_fields, strategy_class_for

    assert strategy_class_for('slp') is SLPStrategy
    assert strategy_class_for('auto') is None
    fields = {name: (kind, default) for name, kind, default in editable_fields(SLPStrategy)}
    assert fields['accuracy'][0] == 'choice'
    assert 'time_budget_s' not in fields


def test_worker_applies_strategy_overrides():
    from dvfopt import SLPStrategy

    w = SolverWorker(
        deformation_i=np.zeros((3, 1, 6, 6)),
        method_id='slp_2tri',
        params={'strategy_overrides': {'cluster_pixel_threshold': 123}},
    )
    strat = w._build_strategy()
    assert isinstance(strat, SLPStrategy)
    assert strat.cluster_pixel_threshold == 123


def test_worker_bad_override_raises():
    w = SolverWorker(
        deformation_i=np.zeros((3, 1, 6, 6)),
        method_id='slp_2tri',
        params={'strategy_overrides': {'no_such_field': 1}},
    )
    with pytest.raises(ValueError):
        w._build_strategy()
```

Append to `tests/test_gui_app.py`:

```python
def test_params_dialog_strategy_tab_and_persistence(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / 's.ini')
    monkeypatch.setattr(
        LiveSolverWindow,
        '_settings',
        staticmethod(lambda: QtCore.QSettings(ini, QtCore.QSettings.IniFormat)),
    )
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._strategy_overrides['slp'] = {'cluster_pixel_threshold': 99}
    win._save_settings()
    win2 = LiveSolverWindow()
    assert win2._strategy_overrides.get('slp') == {'cluster_pixel_threshold': 99}
    # Overrides reach the worker params.
    captured = {}
    monkeypatch.setattr(
        'dvfopt_gui.worker.SolverWorker.start',
        lambda self: captured.setdefault('p', self._params),
    )
    win._select_combo_data(win._constraint_combo, '2tri')
    win._select_combo_data(win._method_combo, 'slp')
    win._on_run(use_roi=False)
    assert captured['p']['strategy_overrides'] == {'cluster_pixel_threshold': 99}
```

- [ ] **Step 2: RED** — `No module named 'dvfopt_gui.strategy_params'`.

- [ ] **Step 3: Implement.**

Create `dvfopt_gui/strategy_params.py`:

```python
"""Auto-generated per-strategy parameter editing.

Every dvfopt Strategy is a dataclass, so its knobs are introspectable:
``editable_fields`` maps dataclass fields to simple widget kinds and
``StrategyParamsTab`` renders them. Overrides are stored per-method by the
window and applied at worker construction — no bespoke UI per method.
"""

from __future__ import annotations

import dataclasses

from PyQt5 import QtWidgets

# Fields the toolbar already owns, or that make no sense to override here.
_EXCLUDED_FIELDS = {'time_budget_s'}
# Literal-choice fields (dataclasses can't express Literal defaults cleanly).
_CHOICE_FIELDS = {'accuracy': ('fast', 'max')}


def strategy_class_for(algo: str):
    """Strategy class for a GUI method algo tag, or None for non-dataclass
    methods (auto / pipelines / marching)."""
    import dvfopt

    mapping = {
        'slp': dvfopt.SLPStrategy,
        'm14': dvfopt.HarmonicALMRefineRepairStrategy,
        'm14_schwarz': dvfopt.SchwarzHarmonicALMRefineRepairStrategy,
        'm10': dvfopt.HarmonicALMBarrierStrategy,
        'barrier': dvfopt.BarrierStrategy,
        'slsqp_windowed': dvfopt.SLSQPWindowedStrategy,
        'slsqp_fullgrid': dvfopt.SLSQPFullGridStrategy,
        'schwarz': dvfopt.SchwarzStrategy,
        'nmvf': dvfopt.NMVFStrategy,
        'barrier_torch': dvfopt.BarrierTet3DTorchStrategy,
        # 3D family (method combo algo tags in tet3d mode)
        'm14_tet3d_alias': None,  # unused; 3D algos below
    }
    mapping.update(
        {
            'm14_3d': dvfopt.HarmonicALMRefineRepair3DStrategy,
        }
    )
    # 3D-mode algo tags reuse the 2D names; resolve by (algo, family) at the
    # call site instead when needed. For the GUI's current method ids the 2D
    # names above suffice; tet3d algos map here:
    tet3d = {
        'm14@tet3d': dvfopt.HarmonicALMRefineRepair3DStrategy,
        'm14_schwarz@tet3d': dvfopt.SchwarzHarmonicALMRefineRepair3DStrategy,
        'm10@tet3d': dvfopt.HarmonicALMBarrier3DStrategy,
        'slsqp_fullgrid@tet3d': dvfopt.SLSQPFullGrid3DStrategy,
        'active_band@tet3d': dvfopt.ActiveBandALM3DStrategy,
        'coupled_kring@tet3d': dvfopt.CoupledKRing3DStrategy,
    }
    mapping.update(tet3d)
    return mapping.get(algo)


def editable_fields(cls) -> list:
    """``(name, kind, default)`` for each editable dataclass field.

    kind: 'int' | 'float' | 'bool' | 'choice' | 'str' | 'readonly'.
    """
    out = []
    for f in dataclasses.fields(cls):
        if f.name in _EXCLUDED_FIELDS or f.name.startswith('_'):
            continue
        default = (
            f.default
            if f.default is not dataclasses.MISSING
            else (f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        )
        if f.name in _CHOICE_FIELDS:
            out.append((f.name, 'choice', default))
        elif isinstance(default, bool):
            out.append((f.name, 'bool', default))
        elif isinstance(default, int):
            out.append((f.name, 'int', default))
        elif isinstance(default, float):
            out.append((f.name, 'float', default))
        elif isinstance(default, str):
            out.append((f.name, 'str', default))
        else:  # tuples, None, other — visible but not editable
            out.append((f.name, 'readonly', default))
    return out


class StrategyParamsTab(QtWidgets.QWidget):
    """Form of widgets for one strategy class; returns only overrides."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QtWidgets.QFormLayout(self)
        self._widgets: dict = {}
        self._defaults: dict = {}

    def build(self, algo: str, overrides: dict) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._defaults.clear()
        cls = strategy_class_for(algo)
        if cls is None:
            self._form.addRow(QtWidgets.QLabel('<i>No editable parameters for this method.</i>'))
            return
        for name, kind, default in editable_fields(cls):
            value = overrides.get(name, default)
            if kind == 'int':
                w = QtWidgets.QSpinBox()
                w.setRange(-1_000_000_000, 1_000_000_000)
                w.setValue(int(value))
            elif kind == 'float':
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(6)
                w.setRange(-1e12, 1e12)
                w.setValue(float(value))
            elif kind == 'bool':
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(value))
            elif kind == 'choice':
                w = QtWidgets.QComboBox()
                for c in _CHOICE_FIELDS[name]:
                    w.addItem(c)
                w.setCurrentText(str(value))
            elif kind == 'str':
                w = QtWidgets.QLineEdit(str(value))
            else:  # readonly
                w = QtWidgets.QLabel(repr(default))
                self._form.addRow(f'{name}:', w)
                continue
            self._widgets[name] = (kind, w)
            self._defaults[name] = default
            self._form.addRow(f'{name}:', w)

    def values(self) -> dict:
        """Only the values that differ from the dataclass defaults."""
        out = {}
        for name, (kind, w) in self._widgets.items():
            if kind == 'int':
                v = int(w.value())
            elif kind == 'float':
                v = float(w.value())
            elif kind == 'bool':
                v = bool(w.isChecked())
            elif kind == 'choice':
                v = str(w.currentText())
            else:
                v = str(w.text())
            if v != self._defaults[name]:
                out[name] = v
        return out
```

(Note: the `strategy_class_for` mapping's `@tet3d` variants are used by the
dialog when `_is_3d_run` — the window passes `f'{algo}@tet3d'` in that case;
plain algo tags otherwise. Delete the placeholder `'m14_tet3d_alias'` and
`'m14_3d'` entries if implementing cleanly — the final mapping is: the 2D
dict + the `@tet3d` dict, nothing else.)

`worker.py` — apply overrides. In `_build_strategy`, add at the top (after
`mid = self._method_id`):

```python
        overrides = dict(self._params.get('strategy_overrides') or {})

        def _make(cls, **base):
            try:
                return cls(**{**base, **overrides})
            except TypeError as exc:
                raise ValueError(f'invalid strategy parameter(s) for {cls.__name__}: {exc}') from exc
```

and change every `return SomeStrategy(...)` in the method to
`return _make(SomeStrategy, ...)` (keeping existing kwargs like
`time_budget_s=time_budget` as the `base`). The `auto_*` branch does NOT
apply overrides (labels vary) — leave its `make_strategy(label)` as-is.

`app.py`:
- `__init__`: `self._strategy_overrides: dict[str, dict] = {}`.
- `_start_worker` params dict: add

```python
            'strategy_overrides': self._strategy_overrides.get(
                self._current_params_algo(), {}
            ),
```

with helper:

```python
    def _current_params_algo(self) -> str:
        """Key for strategy-override storage: the algo tag, family-qualified
        in 3D mode (the 3D classes have different knobs)."""
        algo = self._method_combo.currentData() or ''
        return f'{algo}@tet3d' if self._is_3d_run else algo
```

- `ParamsDialog`: add ctor kwargs `strategy_algo: str, strategy_overrides: dict`;
  add a tab:

```python
        from dvfopt_gui.strategy_params import StrategyParamsTab

        self._strategy_tab = StrategyParamsTab()
        self._strategy_tab.build(strategy_algo, dict(strategy_overrides))
        reset_row = QtWidgets.QWidget()
        reset_lay = QtWidgets.QVBoxLayout(reset_row)
        reset_lay.setContentsMargins(0, 0, 0, 0)
        reset_lay.addWidget(self._strategy_tab)
        reset_btn = QtWidgets.QPushButton('Reset to defaults')
        reset_btn.clicked.connect(lambda: self._strategy_tab.build(strategy_algo, {}))
        reset_lay.addWidget(reset_btn)
        tabs.addTab(reset_row, 'Strategy')
```

  and include `'strategy_overrides': self._strategy_tab.values()` in
  `result_values()`.
- `_on_open_params`: pass `strategy_algo=self._current_params_algo()`,
  `strategy_overrides=self._strategy_overrides.get(self._current_params_algo(), {})`;
  on accept store `self._strategy_overrides[self._current_params_algo()] = vals['strategy_overrides']`
  (delete the key when the dict is empty).
- Settings: `_save_settings` adds
  `s.setValue('strategy_overrides', json.dumps(self._strategy_overrides))`;
  `_restore_settings` adds

```python
        raw = s.value('strategy_overrides', '', type=str)
        if raw:
            try:
                self._strategy_overrides = {
                    k: dict(v) for k, v in json.loads(raw).items()
                }
            except (ValueError, TypeError, AttributeError):
                self._strategy_overrides = {}
```

  (add `import json` to app.py's imports).

- [ ] **Step 4: GREEN** — Step-1 tests + both GUI suites PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_logic.py tests/test_gui_app.py
git add dvfopt_gui/strategy_params.py dvfopt_gui/app.py dvfopt_gui/worker.py tests/test_gui_logic.py tests/test_gui_app.py
git commit -m "feat(gui): auto-generated per-strategy parameter panel with persistence"
```

---

### Task 11: 3D sub-volume ROI

**Files:**
- Modify: `dvfopt_gui/app.py` (`z0`/`z1` spinboxes; `_apply_mode_gating`; `_on_run` 3D-ROI branch; `_on_finished` 3D sub-box splice; `__init__` attr)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Produces: `self._z0_spin`, `self._z1_spin` (QSpinBox, visible only in 3D mode); `self._section_bounds_3d: tuple | None` = `(z0, z1ex, y0, y1, x0, x1)`; 3D "Run section" enabled.

- [ ] **Step 1: Failing tests.** Append to `tests/test_gui_app.py`:

```python
def test_3d_roi_spinboxes_and_run_section(qapp, monkeypatch):
    win = LiveSolverWindow(np.zeros((3, 6, 20, 20)))
    assert not win._z0_spin.isVisibleTo(win)  # hidden in 2D mode
    win._select_combo_data(win._constraint_combo, 'tet3d')
    assert win._z0_spin.isVisibleTo(win) and win._z1_spin.isVisibleTo(win)
    assert win._run_roi_btn.isEnabled()  # 3D ROI now supported
    assert (win._z0_spin.value(), win._z1_spin.value()) == (0, 5)

    win._section_roi.setPos(4, 4)
    win._section_roi.setSize([10, 10])
    win._z0_spin.setValue(1)
    win._z1_spin.setValue(4)
    captured = {}
    monkeypatch.setattr(
        win, '_start_worker', lambda def_i, method_id=None: captured.setdefault('shape', def_i.shape)
    )
    win._on_run(use_roi=True)
    assert captured['shape'] == (3, 4, 10, 10)
    assert win._section_bounds_3d == (1, 5, 4, 14, 4, 14)


def test_3d_roi_splice_back(qapp):
    win = LiveSolverWindow(np.zeros((3, 6, 20, 20)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    win._worker = None  # sender() is None -> guard passes
    win._section_bounds_3d = (1, 5, 4, 14, 4, 14)
    sub = np.full((3, 4, 10, 10), 2.0)
    win._on_finished(sub, None)
    assert win._volume[1, 2, 5, 5] == pytest.approx(2.0)
    assert win._volume[1, 0, 5, 5] == 0.0  # outside the box untouched
```

- [ ] **Step 2: RED** — no `_z0_spin`.

- [ ] **Step 3: Implement** in `app.py`.

`__init__` — after `self._section_bounds ... = None`:
`self._section_bounds_3d: tuple | None = None`.

Top bar — right after `bar.addWidget(self._z_label)`:

```python
        # 3D-mode sub-volume z-range (hidden in 2D). Pairs with the Rect ROI
        # (which supplies y/x) for "Run section" on a sub-volume.
        self._z0_label = QtWidgets.QLabel('z0:')
        self._z0_spin = QtWidgets.QSpinBox()
        self._z1_label = QtWidgets.QLabel('z1:')
        self._z1_spin = QtWidgets.QSpinBox()
        for wdg in (self._z0_label, self._z0_spin, self._z1_label, self._z1_spin):
            bar.addWidget(wdg)
            wdg.setVisible(False)
```

`_apply_mode_gating` — replace the ROI-disable rule so 3D supports the ROI:

```python
    def _apply_mode_gating(self) -> None:
        """Reflect 2D/3D mode in the run controls."""
        D = self._volume.shape[1] if self._volume is not None else 1
        self._run_roi_btn.setEnabled(self._volume is not None)
        self._run_all_btn.setEnabled((not self._is_3d_run) and D > 1)
        self._pipeline_btn.setEnabled(D > 1)
        self._section_roi.setVisible(self._volume is not None)
        show_z = self._is_3d_run and D > 1
        for wdg in (self._z0_label, self._z0_spin, self._z1_label, self._z1_spin):
            wdg.setVisible(show_z)
        if show_z:
            self._z0_spin.setRange(0, D - 1)
            self._z1_spin.setRange(0, D - 1)
            if self._z1_spin.value() == 0:
                self._z1_spin.setValue(D - 1)
```

(Keep the run-ROI tooltip meaningful: update `self._run_roi_btn`'s tooltip
string to mention the 3D z-range spinboxes.)

`_on_run` — the existing 3D branch (`if self._is_3d_run:`) becomes:

```python
        if self._is_3d_run:
            if use_roi:
                D, H, W = self._volume.shape[1:]
                x, y = self._section_roi.pos()
                w, h = self._section_roi.size()
                y0, x0 = max(0, round(y)), max(0, round(x))
                y1, x1 = min(H, round(y + h)), min(W, round(x + w))
                z0, z1 = int(self._z0_spin.value()), int(self._z1_spin.value())
                if z1 < z0:
                    z0, z1 = z1, z0
                z1ex = z1 + 1
                if (z1ex - z0) < 3 or (y1 - y0) < 3 or (x1 - x0) < 3:
                    QtWidgets.QMessageBox.warning(
                        self, 'Section too small', 'The 3D section must be at least 3×3×3.'
                    )
                    return
                self._section_bounds_3d = (z0, z1ex, y0, y1, x0, x1)
                self.statusBar().showMessage(
                    'Run section (3D): solving the sub-volume — check the box '
                    'boundary for seam folds after it completes.',
                    6_000,
                )
                sub = self._original_volume[:, z0:z1ex, y0:y1, x0:x1].copy()
                self._start_worker(sub)
                return
            self._section_bounds_3d = None
            self._section_bounds = None
            self._start_worker(self._original_volume.copy())
            return
```

Also set `self._section_bounds_3d = None` in `_start_marching_25d` and in
the 2D branches of `_on_run` (next to the existing `_section_bounds`
assignments).

`_on_finished` — the ndim==4 splice becomes:

```python
            if phi_out.ndim == 4:  # full-volume or 3D-ROI result [dz,dy,dx]
                sb3 = self._section_bounds_3d
                if sb3 is not None:
                    z0, z1ex, y0, y1, x0, x1 = sb3
                    self._volume[:, z0:z1ex, y0:y1, x0:x1] = phi_out
                else:
                    self._volume[...] = phi_out
```

- [ ] **Step 4: GREEN** — Step-1 tests + full `tests/test_gui_app.py` PASS. NOTE: the pre-existing test `test_selecting_3d_constraint_enters_3d_mode_and_gates_runs` asserts `not win._run_roi_btn.isEnabled()` in 3D — UPDATE that assertion to `win._run_roi_btn.isEnabled()` (3D ROI is now a feature; keep the Run-all assertion as-is). Same for `test_run_all_stays_disabled_in_3d_after_run_finishes` if it checks the ROI button — adjust only the ROI expectation.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check dvfopt_gui tests/test_gui_app.py && python -m ruff format --check dvfopt_gui tests/test_gui_app.py
git add dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "feat(gui): 3D sub-volume ROI (Rect ROI + z-range) with splice-back"
```

---

### Task 12: Docs + full verification

**Files:**
- Modify: `CLAUDE.md` (GUI paragraph)
- Test: full suite + CI gate

- [ ] **Step 1: Update CLAUDE.md.** In the `dvfopt_gui/` directory-layout entry, extend the description with:

> The method menu now includes **SLP (default 2-tri champion)** and an **Auto** picker (`auto_strategy`); the **Pipeline ▾** button runs `correct_dvf_25d` (2.5D marching, needs dz≡0) or the one-click **full pipeline** (per-slice 2D → 2.5D). The tet3d menu adds the **full 3D pipeline** (`correct_dvf_3d`) and a torch-gated GPU barrier. Loads accept NIfTI/MetaImage/NRRD displacement fields via SimpleITK (and export back to `.npy`/`.nii.gz`); loads are threaded and reject non-finite fields. The feasibility threshold is editable (`thr:` spinbox), 3D metrics are cached (fast z-scrub/hover), the undo stack is byte-budgeted, a clickable per-slice fold strip sits under the plot, every strategy's dataclass knobs are editable via Params → Strategy, and "Run section" works on 3D sub-volumes (Rect ROI + z-range).

- [ ] **Step 2: Full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q`
Expected: all PASS (≈1130+, incl. the ~25 new tests).

- [ ] **Step 3: CI gate**

Run: `python -m ruff check dvfopt dvfopt_gui tests benchmarks && python -m ruff format --check dvfopt dvfopt_gui tests benchmarks`
Expected: `All checks passed!` / `... files already formatted`

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the feature-complete GUI surface"
```

---

## Self-Review

**Spec coverage:** §4.1 SLP → T2; §4.2 Auto → T2; §4.3 2.5D (library hook, runner, Pipeline ▾, full-pipeline chain, current-volume exception, single undo) → T1 + T7 + T8; §4.4 pipeline3d → T7; §4.5 torch → T7; §5.1 import → T6; §5.2 export → T6; §5.3 finite validation → T5; §6.1 metric cache → T4; §6.2 undo budget + threaded load → T5 + T6; §6.3 threshold → T3; §7.1 overview → T9; §7.2 params panel → T10; §7.3 3D ROI → T11; docs/verification → T12. No gaps.

**Type consistency:** `params['threshold']` (T3) consumed by T7 runners; `_volume_snapshot`/`_metric_counts_3d`/`DEFAULT_HISTORY_MAX_3D`/`MAX_3D_HISTORY_BYTES` (existing) used in T7; `_start_worker(deformation_i, method_id=None)` (T8) matches T8/T11 call sites and T7's dispatch ids; `_metric_field_3d` import (T4) exists in worker (Task 2 of the previous plan); `pipeline_report`/`marching_progress` (T7) read by T7's app changes; `LoadedRun` reused by T6's LoadWorker; overview uses `_metric_counts` (existing). T10's `_make` wrapper wraps existing `_build_strategy` returns — including T2's `slp` and T7's `barrier_torch` branches (implementers of T10 must convert ALL constructor returns, including ones added by T2/T7).

**Placeholder scan:** every code step has complete code; the one deliberately-loose instruction (T1 "use the array the loops mutate") names the exact variable candidates and lines; T10 notes the mapping cleanup explicitly. No TBDs.

**Known sequencing:** strictly in order T1→T12 (T7 needs T1+T3; T8 needs T7; T11 adjusts a T7-era gating rule and two pre-existing tests; T10 must see T2+T7 constructor branches).
