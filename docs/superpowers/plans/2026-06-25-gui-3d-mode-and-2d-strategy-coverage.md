# dvfopt GUI: true-3D mode + full 2D strategy coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the two missing 2D strategies into the GUI and add a true-3D mode that solves whole volumes with the 6-tet / Jdet3D constraints and complete 3D pipelines, with live per-phase staging and a responsive Stop for the wallbreaker methods.

**Architecture:** The GUI already composes `dvfopt.Solver(constraint, objective, strategy)` per run. We extend the Constraint dropdown with two 3D entries (gated to `D>1` volumes); selecting one switches the window to 3D mode and routes runs through a new `SolverWorker._run_via_solver_3d` that hands the whole `(3,D,H,W)` volume to `solver.fit`. The 3D constraints accept the GUI's native `[dz,dy,dx]` order directly (no reorder). Live staging + Stop come from a new optional `step_callback` threaded through the 3D wallbreaker orchestrators, fired at phase boundaries. The viewer stays 2D-per-slice, defaulting to a 6-tet min-volume heatmap of the current z.

**Tech Stack:** Python 3.13, PyQt5, pyqtgraph, NumPy, SciPy, pytest (offscreen Qt via `QT_QPA_PLATFORM=offscreen`).

**Spec:** `docs/superpowers/specs/2026-06-25-gui-3d-mode-and-2d-strategy-coverage-design.md`

## Global Constraints

- **Phi conventions:** GUI volume is `(3, D, H, W)` = `[dz, dy, dx]`; 2D snapshot phi is `(2, H, W)` = `[dy, dx]`. 3D snapshot phi is the full `(3, D, H, W)` `[dz, dy, dx]`. Branch 2D-vs-3D on `phi.ndim == 4`.
- **3D constraints accept `[dz,dy,dx]` directly** — pass the volume to `solver.fit` as-is; `result.corrected` returns `[dz,dy,dx]`. No GUI-side channel reorder.
- **Fold convention:** folds counted `<= 0`; solver-infeasible counted `< FEASIBILITY_THRESHOLD` (0.01, from `dvfopt._defaults.DEFAULT_PARAMS['threshold']`, already exported as `dvfopt_gui.worker.FEASIBILITY_THRESHOLD`).
- **`method_id = <algo>_<tag>`**, recovered via `rpartition('_')`. 3D tags: `tet3d`, `jdet3d` (single tokens).
- **Library changes must default to no-op:** every new `step_callback` parameter defaults to `None`; phase-boundary calls only happen when a callback is supplied. Let `KeyboardInterrupt` propagate (no bare `except` swallowing it).
- **Git:** the repo owner performs all commits/pushes. Treat each "Commit" step as a checkpoint: stage the listed files and propose the commit message, but only run `git commit` after the user confirms. Never `git push`.
- **Run tests** with `QT_QPA_PLATFORM=offscreen` for any test that constructs a widget.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `dvfopt_gui/app.py` | Window, toolbar, constraint/method menus, rendering, stats | Constraint dropdown +2 (3D), mode state + gating, ndim-aware render/stats/inspector, full-volume run wiring |
| `dvfopt_gui/worker.py` | Solver thread, snapshots, metrics, dispatch | 3D metric helpers, `_build_strategy` 3D cases, `_run_via_solver_3d`, dispatch, memory guard |
| `dvfopt_gui/persistence.py` | NPZ save/load | `dim` flag + 3D history |
| `dvfopt/strategies/wallbreakers.py` | 3D wallbreaker strategies | explicit `step_callback=None` + phase calls / forwarding |
| `dvfopt/core/wallbreakers/_refine_repair_3d.py` | M14Tet core | optional `step_callback`, 4 phase calls |
| `dvfopt/core/wallbreakers/_m14_schwarz_3d.py` | M14-Schwarz3D wrapper | forward `step_callback` into per-cluster core |
| `tests/test_gui_logic.py` | Headless GUI logic | 3D metric, worker 3D, persistence 3D tests |
| `tests/test_gui_app.py` | Offscreen widget | mode gating, render, dispatch tests |
| `tests/test_wallbreakers_3d_callback.py` | New | library callback/stop tests |

---

## Task 1: Add SLSQP-fullgrid + Schwarz to the 2-tri method menu

**Files:**
- Modify: `dvfopt_gui/app.py` (the `_METHOD_SPECS_2TRI` list, ~line 248)
- Modify: `dvfopt_gui/worker.py` (`SolverWorker._build_strategy`, ~line 541)
- Test: `tests/test_gui_app.py`, `tests/test_gui_logic.py`

**Interfaces:**
- Consumes: existing `Solver`, `SLSQPFullGridStrategy`, `SchwarzStrategy` from `dvfopt`.
- Produces: method ids `slsqp_fullgrid_2tri`, `schwarz_2tri` dispatchable by `_build_strategy`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_logic.py`)

```python
def test_build_strategy_adds_2d_fullgrid_and_schwarz():
    from dvfopt import SLSQPFullGridStrategy, SchwarzStrategy

    w1 = SolverWorker(deformation_i=np.zeros((3, 1, 6, 6)), method_id='slsqp_fullgrid_2tri')
    assert isinstance(w1._build_strategy(), SLSQPFullGridStrategy)
    w2 = SolverWorker(deformation_i=np.zeros((3, 1, 6, 6)), method_id='schwarz_2tri')
    assert isinstance(w2._build_strategy(), SchwarzStrategy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_build_strategy_adds_2d_fullgrid_and_schwarz -v`
Expected: FAIL with `ValueError: unknown method_id='slsqp_fullgrid_2tri'`

- [ ] **Step 3: Add the dispatch cases** in `dvfopt_gui/worker.py` `_build_strategy`, after the `m14_schwarz_2tri` case and before `nmvf_jdet`:

```python
        if mid == 'slsqp_fullgrid_2tri':
            from dvfopt import SLSQPFullGridStrategy

            return SLSQPFullGridStrategy()
        if mid == 'schwarz_2tri':
            from dvfopt import SchwarzStrategy

            return SchwarzStrategy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_build_strategy_adds_2d_fullgrid_and_schwarz -v`
Expected: PASS

- [ ] **Step 5: Add the menu entries + widget test.** In `dvfopt_gui/app.py`, append to `_METHOD_SPECS_2TRI`:

```python
    ('slsqp_fullgrid', 'SLSQP full-grid (2-tri; KKT, smallest L1 on mild folds)'),
    ('schwarz', 'Schwarz (2-tri; overlapping-tile decomposition)'),
```

Append to `tests/test_gui_app.py`:

```python
def test_2tri_menu_has_fullgrid_and_schwarz(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))
    win._select_combo_data(win._constraint_combo, '2tri')
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'slsqp_fullgrid' in algos
    assert 'schwarz' in algos
```

- [ ] **Step 6: Run both test files**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py tests/test_gui_logic.py -q`
Expected: PASS (all)

- [ ] **Step 7: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/app.py dvfopt_gui/worker.py tests/test_gui_app.py tests/test_gui_logic.py
git commit -m "feat(gui): expose SLSQP-fullgrid and Schwarz 2-tri strategies"
```

---

## Task 2: 3D fold-metric helpers in the worker

**Files:**
- Modify: `dvfopt_gui/worker.py` (add helpers near `_metric_counts`, ~line 165)
- Test: `tests/test_gui_logic.py`

**Interfaces:**
- Consumes: `dvfopt.jacobian.tetrahedron_sign.six_tet_min_volume_3d`, `dvfopt.jacobian.numpy_jdet.jacobian_det3D`.
- Produces:
  - `_metric_counts_3d(phi3d, kind) -> tuple[int, float]` — `kind in {'tet3d','jdet3d'}`; folds `<= 0`, min over volume.
  - `_infeasible_count_3d(phi3d, kind, threshold=FEASIBILITY_THRESHOLD) -> int` — count `< threshold`.
  - `phi3d` is `(3, D, H, W)` `[dz, dy, dx]`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_logic.py`)

```python
def _folded_volume_3d(D=4, H=8, W=8):
    # A z-direction shear large enough to invert tets.
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    phi = np.zeros((3, D, H, W))
    phi[2] = -1.8 * xx  # dx ramp -> strong compression, inverts cells
    return phi


def test_metric_counts_3d_tet_and_jdet():
    from dvfopt_gui.worker import _metric_counts_3d, _infeasible_count_3d

    phi = _folded_volume_3d()
    n_tet, min_tet = _metric_counts_3d(phi, 'tet3d')
    n_jdet, min_jdet = _metric_counts_3d(phi, 'jdet3d')
    assert n_tet > 0 and min_tet < 0
    assert n_jdet > 0 and min_jdet < 0
    # Identity volume: no folds, nothing infeasible.
    ident = np.zeros((3, 4, 8, 8))
    assert _metric_counts_3d(ident, 'tet3d') == (0, pytest.approx(1 / 6))
    assert _infeasible_count_3d(ident, 'tet3d') == 0
    assert _infeasible_count_3d(ident, 'jdet3d') == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_metric_counts_3d_tet_and_jdet -v`
Expected: FAIL with `ImportError: cannot import name '_metric_counts_3d'`

- [ ] **Step 3: Implement the helpers** in `dvfopt_gui/worker.py` after `_infeasible_count`:

```python
def _metric_field_3d(phi3d, kind: str) -> np.ndarray:
    """Per-cell 3D metric field for ``phi3d`` ``(3, D, H, W)`` ``[dz,dy,dx]``.

    ``kind='tet3d'`` → per-cell min 6-tet signed volume
    ``(D-1, H-1, W-1)``; ``kind='jdet3d'`` → per-voxel 3D Jacobian
    determinant ``(D, H, W)``.
    """
    if kind == 'tet3d':
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        return six_tet_min_volume_3d(phi3d)
    if kind == 'jdet3d':
        from dvfopt.jacobian.numpy_jdet import jacobian_det3D

        return jacobian_det3D(phi3d)
    raise ValueError(f'unknown 3D metric kind={kind!r}')


def _metric_counts_3d(phi3d, kind: str) -> tuple[int, float]:
    """``(n_neg, min_T)`` over the whole volume under one 3D metric.
    Folds counted ``<= 0`` (matching the 2D convention)."""
    field = _metric_field_3d(phi3d, kind)
    return int((field <= 0).sum()), float(field.min())


def _infeasible_count_3d(phi3d, kind: str, threshold: float = FEASIBILITY_THRESHOLD) -> int:
    """Voxels/cells the solver still considers infeasible: metric ``< threshold``."""
    return int((_metric_field_3d(phi3d, kind) < threshold).sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_metric_counts_3d_tet_and_jdet -v`
Expected: PASS

- [ ] **Step 5: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/worker.py tests/test_gui_logic.py
git commit -m "feat(gui): 3D fold-metric helpers (6-tet / Jdet3D)"
```

---

## Task 3: `step_callback` in the M14Tet core

**Files:**
- Modify: `dvfopt/core/wallbreakers/_refine_repair_3d.py` (`iterative_3d_tet_refine_repair`)
- Test: `tests/test_wallbreakers_3d_callback.py` (new)

**Interfaces:**
- Produces: `iterative_3d_tet_refine_repair(..., step_callback=None)`. When supplied, called as `step_callback({'phi': <(3,D,H,W) [dz,dy,dx]>, 'stage': <str>})` after each of: `'seed'`, `'pull'`, `'repair'`, `'polish'`. Default `None` ⇒ no calls (unchanged behavior). A raised `KeyboardInterrupt` propagates out.

- [ ] **Step 1: Write the failing test** (create `tests/test_wallbreakers_3d_callback.py`)

```python
"""3D wallbreaker step_callback hook: live staging + stop."""
from __future__ import annotations

import numpy as np
import pytest

from dvfopt.core.wallbreakers._refine_repair_3d import iterative_3d_tet_refine_repair


def _folded_volume_3d(D=4, H=10, W=10):
    _, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    phi = np.zeros((3, D, H, W))
    phi[2, :, 4:6, 4:6] = 1.5  # local dx bump -> a few folded cells
    return phi


def test_m14tet_core_fires_step_callback_per_phase():
    phi = _folded_volume_3d()
    seen = []

    def cb(state):
        assert state['phi'].shape == phi.shape
        seen.append(state['stage'])

    iterative_3d_tet_refine_repair(phi, time_budget_s=30.0, verbose=0, step_callback=cb)
    # seed + pull always fire; repair fires only if residual; polish if strict.
    assert 'seed' in seen and 'pull' in seen
    assert seen == sorted(seen, key=['seed', 'pull', 'repair', 'polish'].index)


def test_m14tet_core_stop_via_callback_raises():
    phi = _folded_volume_3d()

    def cb(state):
        raise KeyboardInterrupt('stop')

    with pytest.raises(KeyboardInterrupt):
        iterative_3d_tet_refine_repair(phi, time_budget_s=30.0, verbose=0, step_callback=cb)


def test_m14tet_core_default_callback_none_unchanged():
    phi = _folded_volume_3d()
    out = iterative_3d_tet_refine_repair(phi, time_budget_s=30.0, verbose=0)
    assert out.shape == phi.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wallbreakers_3d_callback.py::test_m14tet_core_fires_step_callback_per_phase -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'step_callback'`

- [ ] **Step 3: Add the parameter + phase calls.** In `dvfopt/core/wallbreakers/_refine_repair_3d.py`:

Add to the signature (after `record_history: bool = False,`):

```python
    step_callback=None,
```

Add a tiny helper right after `phi_in` is validated (after the `_, D, H, W = phi_in.shape` line):

```python
    def _emit(phi_stage, stage):
        if step_callback is not None:
            step_callback({'phi': phi_stage, 'stage': stage})
```

Then insert calls after each stage's field is finalized:
- after `info['stage1_seed'] = ...` block: `_emit(seed, 'seed')`
- after `info['stage2_pull'] = ...` block: `_emit(pulled, 'pull')`
- after `info['stage3_repair'] = ...` block: `_emit(repaired, 'repair')`
- in the early-return branch (`if repaired_min <= threshold:`), before `return`: `_emit(repaired, 'polish')`
- after `phi_out = np.stack([dz, dy, dx])` near the end (after `final_min`/`final_L2` computed): `_emit(phi_out, 'polish')`

(The `polish` stage emits the final field in both the early-return and the normal path, so consumers always get a terminal snapshot.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wallbreakers_3d_callback.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the 3D wallbreaker regression tests** (ensure default path unchanged)

Run: `python -m pytest tests/ -k "3d or tet or refine_repair" -q`
Expected: PASS

- [ ] **Step 6: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt/core/wallbreakers/_refine_repair_3d.py tests/test_wallbreakers_3d_callback.py
git commit -m "feat(3d): optional step_callback at M14Tet phase boundaries"
```

---

## Task 4: `step_callback` in the 3D wallbreaker strategies + Schwarz wrapper

**Files:**
- Modify: `dvfopt/strategies/wallbreakers.py` (`HarmonicALMBarrier3DStrategy.solve` ~L516, `HarmonicALMRefineRepair3DStrategy.solve` ~L686, `SchwarzHarmonicALMRefineRepair3DStrategy.solve` ~L761)
- Modify: `dvfopt/core/wallbreakers/_m14_schwarz_3d.py` (`iterative_3d_tet_refine_repair_schwarz`)
- Test: `tests/test_wallbreakers_3d_callback.py`

**Interfaces:**
- Consumes: `iterative_3d_tet_refine_repair(..., step_callback=...)` (Task 3).
- Produces: all three 3D wallbreaker `solve()` methods accept `step_callback=None` (via explicit kwarg, replacing the silent `**_` for this name) and route it:
  - M10Tet (`HarmonicALMBarrier3DStrategy`): fires `{'phi': phi_h, 'stage': 'harmonic'}` and `{'phi': phi_alm, 'stage': 'alm'}`.
  - M14Tet (`HarmonicALMRefineRepair3DStrategy`): forwards to the core.
  - M14-Schwarz3D: forwards into `iterative_3d_tet_refine_repair_schwarz(..., step_callback=...)`, which threads it into the per-cluster `inner_solve` core call (so `phi` there is a **crop**, used by the consumer for the stop check only).

- [ ] **Step 1: Write the failing test** (append to `tests/test_wallbreakers_3d_callback.py`)

```python
def test_m10tet_strategy_fires_harmonic_and_alm():
    from dvfopt import Solver, Tet6Constraint3D, L2Objective, M10TetStrategy

    phi = _folded_volume_3d()
    seen = []
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L2Objective(),
        strategy=M10TetStrategy(),
    )
    solver.fit(phi, step_callback=lambda s: seen.append(s['stage']))
    assert 'harmonic' in seen and 'alm' in seen


def test_m14tet_strategy_forwards_callback():
    from dvfopt import Solver, Tet6Constraint3D, L2Objective, M14TetStrategy

    phi = _folded_volume_3d()
    seen = []
    solver = Solver(
        constraint=Tet6Constraint3D(shape=phi.shape[1:]),
        objective=L2Objective(),
        strategy=M14TetStrategy(time_budget_s=30.0),
    )
    solver.fit(phi, step_callback=lambda s: seen.append(s['stage']))
    assert 'seed' in seen


def test_m14schwarz3d_stop_via_callback_raises():
    from dvfopt import Solver, Tet6Constraint3D, L2Objective, M14Schwarz3DStrategy

    phi = _folded_volume_3d()
    with pytest.raises(KeyboardInterrupt):
        Solver(
            constraint=Tet6Constraint3D(shape=phi.shape[1:]),
            objective=L2Objective(),
            strategy=M14Schwarz3DStrategy(time_budget_s=30.0),
        ).fit(phi, step_callback=lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wallbreakers_3d_callback.py::test_m10tet_strategy_fires_harmonic_and_alm -v`
Expected: FAIL (no `'harmonic'` in `seen` — the `**_` swallows `step_callback`)

- [ ] **Step 3: M10Tet — accept + fire.** In `HarmonicALMBarrier3DStrategy.solve`, change the signature line `**_,` to:

```python
        step_callback=None,
        **_,
```

After the harmonic-seed `harmonic_phase = {...}` block, add:

```python
        if step_callback is not None:
            step_callback({'phi': phi_h, 'stage': 'harmonic'})
```

After the ALM `alm_phase = {...}` block, add:

```python
        if step_callback is not None:
            step_callback({'phi': phi_alm, 'stage': 'alm'})
```

- [ ] **Step 4: M14Tet — forward.** In `HarmonicALMRefineRepair3DStrategy.solve`, change `**_,` to `step_callback=None,\n        **_,` and add `step_callback=step_callback,` to the `iterative_3d_tet_refine_repair(...)` call.

- [ ] **Step 5: M14-Schwarz3D — forward.** In `SchwarzHarmonicALMRefineRepair3DStrategy.solve`, change `**_,` to `step_callback=None,\n        **_,` and add `step_callback=step_callback,` to the `iterative_3d_tet_refine_repair_schwarz(...)` call.

In `dvfopt/core/wallbreakers/_m14_schwarz_3d.py`, add `step_callback=None,` to the `iterative_3d_tet_refine_repair_schwarz` signature, and pass `step_callback=step_callback` into the `iterative_3d_tet_refine_repair(...)` call inside `inner_solve`.

- [ ] **Step 6: Run the new tests**

Run: `python -m pytest tests/test_wallbreakers_3d_callback.py -v`
Expected: PASS (all)

- [ ] **Step 7: Verify `Solver.fit` forwards `step_callback` to 3D solve.** If `test_m10tet_strategy_fires_harmonic_and_alm` still fails because `fit` does not pass `step_callback` through, confirm `Solver.fit(self, phi, *, verbose=0, record_history=False, **kwargs)` forwards `**kwargs` to `strategy.solve`. The 2D path already relies on this (existing `_run_via_solver` passes `step_callback`), so no change should be needed; if `fit`'s signature names `step_callback` is absent and kwargs are dropped, add `**kwargs` forwarding in `dvfopt/solver.py` `fit`.

- [ ] **Step 8: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt/strategies/wallbreakers.py dvfopt/core/wallbreakers/_m14_schwarz_3d.py tests/test_wallbreakers_3d_callback.py
git commit -m "feat(3d): thread step_callback through 3D wallbreaker strategies"
```

---

## Task 5: Worker 3D run path

**Files:**
- Modify: `dvfopt_gui/worker.py` (`_build_strategy`, `_trajectory_metric_kind`, `run`, new `_run_via_solver_3d`, new constants)
- Test: `tests/test_gui_logic.py`

**Interfaces:**
- Consumes: Task 2 helpers; Task 3/4 `step_callback`; `dvfopt.Solver`, `Tet6Constraint3D`, `JdetConstraint3D`, the 3D strategies.
- Produces:
  - Constants `DEFAULT_HISTORY_MAX_3D = 8`, `MAX_3D_HISTORY_BYTES = 2 * 1024**3`.
  - `_build_strategy` handles `m14_tet3d`, `m14_schwarz_tet3d`, `m10_tet3d`, `slsqp_fullgrid_tet3d`, `barrier_jdet3d`, `slsqp_windowed_jdet3d`.
  - `_trajectory_metric_kind` returns `'tet3d'` for `*_tet3d`, `'jdet3d'` for `*_jdet3d`.
  - `run()` dispatches 3D tags to `_run_via_solver_3d`.
  - In 3D mode the worker's `deformation_i` is the full `(3, D, H, W)` volume; `_run_via_solver_3d` returns the corrected `(3, D, H, W)`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_logic.py`)

```python
def test_worker_3d_trajectory_metric_and_strategy():
    from dvfopt import (
        HarmonicALMRefineRepair3DStrategy,
        SLSQPFullGrid3DStrategy,
        BarrierStrategy,
    )

    vol = np.zeros((3, 4, 8, 8))
    assert SolverWorker(deformation_i=vol, method_id='m14_tet3d')._trajectory_metric_kind() == 'tet3d'
    assert SolverWorker(deformation_i=vol, method_id='barrier_jdet3d')._trajectory_metric_kind() == 'jdet3d'
    assert isinstance(
        SolverWorker(deformation_i=vol, method_id='m14_tet3d')._build_strategy(),
        HarmonicALMRefineRepair3DStrategy,
    )
    assert isinstance(
        SolverWorker(deformation_i=vol, method_id='slsqp_fullgrid_tet3d')._build_strategy(),
        SLSQPFullGrid3DStrategy,
    )
    assert isinstance(
        SolverWorker(deformation_i=vol, method_id='barrier_jdet3d')._build_strategy(),
        BarrierStrategy,
    )


def test_worker_3d_solve_reaches_feasibility():
    # Small folded volume; M14Tet should clear folds end-to-end.
    _, yy, xx = np.meshgrid(np.arange(4), np.arange(10), np.arange(10), indexing='ij')
    vol = np.zeros((3, 4, 10, 10))
    vol[2, :, 4:6, 4:6] = 1.5
    from dvfopt_gui.worker import _metric_counts_3d

    n_before, _ = _metric_counts_3d(vol[:, :], 'tet3d')
    assert n_before > 0
    w = SolverWorker(deformation_i=vol, method_id='m14_tet3d', params={'time_budget_s': 60.0})
    phi_out = w._run_via_solver_3d(w._build_strategy(), 'tet3d', metric_kind='tet3d')
    assert phi_out.shape == (3, 4, 10, 10)
    n_after, _ = _metric_counts_3d(phi_out, 'tet3d')
    assert n_after <= n_before
    # history has an input snapshot (ndim 4) + at least the final.
    assert w.history_len() >= 2
    assert w.history_get(0).phi.ndim == 4


def test_worker_3d_memory_guard_keeps_init_and_final(monkeypatch):
    import dvfopt_gui.worker as W

    monkeypatch.setattr(W, 'MAX_3D_HISTORY_BYTES', 1)  # force the guard
    _, yy, xx = np.meshgrid(np.arange(4), np.arange(10), np.arange(10), indexing='ij')
    vol = np.zeros((3, 4, 10, 10))
    vol[2, :, 4:6, 4:6] = 1.5
    w = SolverWorker(deformation_i=vol, method_id='m14_tet3d', params={'time_budget_s': 60.0})
    w._run_via_solver_3d(w._build_strategy(), 'tet3d', metric_kind='tet3d')
    # Guard tripped: only the input + final snapshots, no mid stages.
    assert w.history_len() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_worker_3d_trajectory_metric_and_strategy -v`
Expected: FAIL (`unknown method_id='m14_tet3d'`)

- [ ] **Step 3: Add constants** near `DEFAULT_HISTORY_MAX` in `dvfopt_gui/worker.py`:

```python
# 3D runs emit few (phase-level) snapshots, but each is a full
# (3, D, H, W) volume. Cap the deque small and guard total bytes: past
# the budget we keep only the input + final snapshots.
DEFAULT_HISTORY_MAX_3D = 8
MAX_3D_HISTORY_BYTES = 2 * 1024 ** 3  # ~2 GB
```

- [ ] **Step 4: Extend `_build_strategy`** — add before the final `raise`:

```python
        if mid == 'm10_tet3d':
            from dvfopt import HarmonicALMBarrier3DStrategy

            return HarmonicALMBarrier3DStrategy(time_budget_s=time_budget)
        if mid == 'm14_tet3d':
            from dvfopt import HarmonicALMRefineRepair3DStrategy

            return HarmonicALMRefineRepair3DStrategy(time_budget_s=time_budget)
        if mid == 'm14_schwarz_tet3d':
            from dvfopt import SchwarzHarmonicALMRefineRepair3DStrategy

            return SchwarzHarmonicALMRefineRepair3DStrategy(time_budget_s=time_budget)
        if mid == 'slsqp_fullgrid_tet3d':
            from dvfopt import SLSQPFullGrid3DStrategy

            return SLSQPFullGrid3DStrategy()
        if mid in ('barrier_jdet3d',):
            return BarrierStrategy()
```

Note: `HarmonicALMBarrier3DStrategy` / `HarmonicALMRefineRepair3DStrategy` accept `time_budget_s`; `SchwarzHarmonicALMRefineRepair3DStrategy` accepts `time_budget_s`. `BarrierStrategy` and `SLSQPFullGrid3DStrategy` take no time budget. (For `slsqp_windowed_jdet3d`, see Step 6.)

- [ ] **Step 5: Extend `_trajectory_metric_kind`** — replace the final `return` line with:

```python
        if mid.endswith('_tet3d'):
            return 'tet3d'
        if mid.endswith('_jdet3d'):
            return 'jdet3d'
        return '2tri' if mid.endswith('_2tri') else 'jdet'
```

- [ ] **Step 6: Add `_run_via_solver_3d`** to `SolverWorker` (after `_run_via_solver`):

```python
    def _run_via_solver_3d(self, strategy, constraint_kind: str, *, metric_kind: str):
        """Whole-volume 3D path. ``self._deformation_i`` is the full
        ``(3, D, H, W)`` ``[dz, dy, dx]`` volume. The 3D constraints accept
        that layout directly (no reorder); ``result.corrected`` returns it.

        Phased wallbreakers (m10/m14/m14_schwarz) fire ``step_callback`` at
        phase boundaries — we record full-volume stages (and check the stop
        flag). Non-phased methods (slsqp_fullgrid, barrier, slsqp_windowed)
        run to completion and only the init + final snapshots land.
        """
        from dvfopt import JdetConstraint3D, Solver, Tet6Constraint3D

        vol = np.asarray(self._deformation_i, dtype=np.float64)
        if vol.ndim != 4 or vol.shape[0] != 3:
            raise ValueError(f'3D run needs (3, D, H, W); got {vol.shape}')
        _, D, H, W = vol.shape
        if constraint_kind == 'tet3d':
            constraint = Tet6Constraint3D(shape=(D, H, W))
        elif constraint_kind == 'jdet3d':
            constraint = JdetConstraint3D(shape=(D, H, W))
        else:
            raise ValueError(f'unknown 3D constraint_kind={constraint_kind!r}')
        objective = self._build_objective()
        solver = Solver(constraint=constraint, objective=objective, strategy=strategy)

        # Memory guard: keep mid stages only if the full deque fits the budget.
        est = DEFAULT_HISTORY_MAX_3D * 3 * D * H * W * 8
        keep_stages = est <= MAX_3D_HISTORY_BYTES

        # Initial snapshot (input volume), under the run metric.
        n0, m0 = _metric_counts_3d(vol, metric_kind)
        self._record(_volume_snapshot(vol, n_neg=n0, min_T=m0, outer_iter=0))

        outer = [0]

        def _stage_callback(state):
            if self._stop_requested:
                raise KeyboardInterrupt('user requested stop')
            phi = np.asarray(state['phi'])
            # Schwarz emits per-cluster crops — use them only for the stop
            # check above; snapshot only the full-volume phases.
            if phi.shape != vol.shape:
                return
            if not keep_stages:
                return
            n, m = _metric_counts_3d(phi, metric_kind)
            outer[0] += 1
            self._record(_volume_snapshot(phi, n_neg=n, min_T=m, outer_iter=outer[0]))

        if self._stop_requested:
            raise KeyboardInterrupt()
        result = solver.fit(vol, step_callback=_stage_callback)
        corrected = np.asarray(result.corrected, dtype=np.float64)
        nf, mf = _metric_counts_3d(corrected, metric_kind)
        self._record(_volume_snapshot(corrected, n_neg=nf, min_T=mf, outer_iter=outer[0] + 1))
        return corrected
```

Add a module-level snapshot factory near `_emit_synthetic_snapshot` usage (top-level function in `worker.py`):

```python
def _volume_snapshot(phi3d, *, n_neg: int, min_T: float, outer_iter: int) -> StateSnapshot:
    """Build a 3D StateSnapshot: phi is the full (3, D, H, W) volume;
    window/opt rects collapse to zero (no active-window overlay in 3D)."""
    return StateSnapshot(
        phi=np.asarray(phi3d, dtype=np.float64).copy(),
        window_y0=0, window_y1=0, window_x0=0, window_x1=0,
        opt_y0=0, opt_y1=0, opt_x0=0, opt_x1=0,
        is_padded=False, neg_y=0, neg_x=0,
        per_index_iter=0, outer_iter=int(outer_iter),
        n_neg=int(n_neg), min_T=float(min_T),
    )
```

- [ ] **Step 7: Dispatch in `run()`** — replace the `else` branch's body that computes `algo, _, kind = mid.rpartition('_')` so 3D tags route to the 3D path:

```python
            else:
                algo, _, kind = mid.rpartition('_')
                if kind in ('tet3d', 'jdet3d'):
                    phi_out = self._run_via_solver_3d(
                        self._build_strategy(), kind, metric_kind=metric_kind
                    )
                elif kind in ('2tri', 'jdet'):
                    phi_out = self._run_via_solver(
                        self._build_strategy(), kind, metric_kind=metric_kind
                    )
                else:
                    raise ValueError(f'unknown method_id={mid!r}')
```

Also guard the initial-snapshot emission in `run()`: the existing `self._emit_initial_snapshot(metric_kind)` assumes a 2D slice. Wrap it:

```python
            metric_kind = self._trajectory_metric_kind()
            if metric_kind not in ('tet3d', 'jdet3d'):
                self._emit_initial_snapshot(metric_kind)
```

(The 3D path emits its own input snapshot inside `_run_via_solver_3d`.)

For `slsqp_windowed_jdet3d`: add a `_build_strategy` case returning `SLSQPWindowedStrategy()` and let `_run_via_solver_3d` handle it (it has no usable callback; init+final only). Add:

```python
        if mid == 'slsqp_windowed_jdet3d':
            from dvfopt import SLSQPWindowedStrategy

            return SLSQPWindowedStrategy()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py -k "3d" -v`
Expected: PASS

- [ ] **Step 9: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/worker.py tests/test_gui_logic.py
git commit -m "feat(gui): worker 3D run path with staged history + memory guard"
```

---

## Task 6: Persistence — 3D save/load

**Files:**
- Modify: `dvfopt_gui/persistence.py` (`build_save_payload`, `parse_loaded`)
- Test: `tests/test_gui_logic.py`

**Interfaces:**
- Produces: payload key `dim` (0-d int, 2 or 3). When `dim == 3`, `history_phi` has shape `(N, 3, D, H, W)`; `parse_loaded` reconstructs 3D snapshots. Absence of `dim` ⇒ 2D (back-compat).

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_logic.py`)

```python
def test_persistence_3d_history_roundtrip(tmp_path):
    from dvfopt_gui.worker import _volume_snapshot

    D, H, W = 3, 5, 5
    vol = np.zeros((3, D, H, W))
    snaps = [
        _volume_snapshot(np.full((3, D, H, W), float(i)), n_neg=i, min_T=float(-i), outer_iter=i)
        for i in range(3)
    ]
    payload = persistence.build_save_payload(
        phi_active=vol[1:, 0],
        full_volume=vol,
        z=0,
        constraint='tet3d',
        method='m14_tet3d',
        objective='l2',
        time_budget_s=60.0,
        max_iterations=200,
        history_max_size=8,
        history_snaps=snaps,
        history_total=3,
        input_volume=vol,
        dim=3,
    )
    assert int(payload['dim']) == 3
    assert payload['history_phi'].shape == (3, 3, D, H, W)
    path = tmp_path / 'run3d.npz'
    np.savez_compressed(path, **payload)
    loaded = np.load(path, allow_pickle=False)
    run = persistence.parse_loaded(loaded)
    loaded.close()
    assert len(run.snapshots) == 3
    assert run.snapshots[2].phi.shape == (3, D, H, W)
    assert run.snapshots[1].n_neg == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_persistence_3d_history_roundtrip -v`
Expected: FAIL (`build_save_payload` has no `dim` kwarg / `history_phi` is 2D-shaped)

- [ ] **Step 3: Update `build_save_payload`.** Add `dim: int = 2,` to the signature. After the `payload` dict is created, add `payload['dim'] = np.int64(dim)`. Replace the history block's per-snapshot phi packing so it adapts to `dim`:

```python
    n = len(history_snaps)
    if n > 0:
        if dim == 3:
            _, Dv, Hv, Wv = history_snaps[0].phi.shape
            phi_hist = np.empty((n, 3, Dv, Hv, Wv), dtype=np.float64)
        else:
            H, W = phi_active.shape[1:]
            phi_hist = np.empty((n, 2, H, W), dtype=np.float64)
        n_neg_arr = np.empty(n, dtype=np.int64)
        min_T_arr = np.empty(n, dtype=np.float64)
        outer_arr = np.empty(n, dtype=np.int64)
        sub_arr = np.empty(n, dtype=np.int64)
        for i, snap in enumerate(history_snaps):
            phi_hist[i] = snap.phi
            n_neg_arr[i] = snap.n_neg
            min_T_arr[i] = snap.min_T
            outer_arr[i] = snap.outer_iter
            sub_arr[i] = snap.per_index_iter
        payload['n_history_steps'] = np.int64(n)
        payload['history_phi'] = phi_hist
        payload['history_n_neg'] = n_neg_arr
        payload['history_min_T'] = min_T_arr
        payload['history_outer_iter'] = outer_arr
        payload['history_per_index_iter'] = sub_arr
        payload['history_total'] = np.int64(history_total)
    else:
        payload['n_history_steps'] = np.int64(0)
```

- [ ] **Step 4: Update `parse_loaded`.** After `phi_hist = np.asarray(mapping['history_phi'], ...)`, the per-snapshot reconstruction already copies `phi_hist[i]` into `StateSnapshot(phi=phi_hist[i].copy(), ...)`. Since `phi_hist[i]` is `(3,D,H,W)` for 3D, the existing loop already produces correctly-shaped 3D snapshots — **no change needed** beyond confirming the window/opt fields stay zero (they do). Add a `dim` read for completeness on `LoadedRun` only if needed; not required for the test.

- [ ] **Step 5: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_logic.py::test_persistence_3d_history_roundtrip -v`
Expected: PASS

- [ ] **Step 6: Wire the GUI save to pass `dim`.** In `dvfopt_gui/app.py` `_build_save_payload`, add `dim=3 if self._is_3d_run else 2,` to the `build_save_payload(...)` call. (`_is_3d_run` is introduced in Task 7; until then this references an attribute — order Task 7 before committing the app.py change, or temporarily use `getattr(self, '_is_3d_run', False)`.) Use:

```python
            dim=3 if getattr(self, '_is_3d_run', False) else 2,
```

- [ ] **Step 7: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/persistence.py dvfopt_gui/app.py tests/test_gui_logic.py
git commit -m "feat(gui): persist 3D runs (dim flag + 3D history)"
```

---

## Task 7: GUI constraint dropdown, mode switching, run gating

**Files:**
- Modify: `dvfopt_gui/app.py` (constraint specs, method specs, `_is_3d_run`, gating, `_on_constraint_changed`, `_apply_loaded_run`, `_on_run`/`_start_worker`, `_on_finished`)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: worker 3D dispatch (Task 5).
- Produces:
  - `_CONSTRAINT_SPECS` gains `('tet3d', '6-tet (3D)')`, `('jdet3d', 'Jdet (3D)')`.
  - `_METHOD_SPECS_BY_CONSTRAINT['tet3d'] = [...]`, `['jdet3d'] = [...]`; defaults in `DEFAULT_METHOD_BY_CONSTRAINT`.
  - `self._is_3d_run: bool` — true when the selected constraint tag is `tet3d`/`jdet3d`.
  - In 3D mode: Run-section + Run-all disabled; "Run full" passes the full `(3, D, H, W)` volume to the worker; `_on_finished` splices the whole corrected volume.
  - 3D constraint entries disabled unless the loaded volume has `D > 1`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_app.py`)

```python
def test_3d_constraints_gated_by_volume_depth(qapp):
    win = LiveSolverWindow(np.zeros((3, 1, 6, 6)))  # 2D section
    tet_idx = win._constraint_combo.findData('tet3d')
    assert tet_idx >= 0
    model = win._constraint_combo.model()
    assert not model.item(tet_idx).isEnabled()  # disabled for D == 1
    win._load_array(np.zeros((3, 4, 6, 6)))  # 3D volume
    assert model.item(win._constraint_combo.findData('tet3d')).isEnabled()


def test_selecting_3d_constraint_enters_3d_mode_and_gates_runs(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._select_combo_data(win._constraint_combo, 'tet3d')
    assert win._is_3d_run
    assert not win._run_roi_btn.isEnabled()
    assert not win._run_all_btn.isEnabled()
    algos = [win._method_combo.itemData(i) for i in range(win._method_combo.count())]
    assert 'm14' in algos and 'm14_schwarz' in algos and 'm10' in algos and 'slsqp_fullgrid' in algos
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py::test_3d_constraints_gated_by_volume_depth -v`
Expected: FAIL (`findData('tet3d')` returns -1)

- [ ] **Step 3: Add constraint specs + method specs** in `dvfopt_gui/app.py`:

```python
CONSTRAINT_TET3D = 'tet3d'
CONSTRAINT_JDET3D = 'jdet3d'
```

Extend `_CONSTRAINT_SPECS`:

```python
    (CONSTRAINT_TET3D, '6-tet (3D; whole-volume true 3D)'),
    (CONSTRAINT_JDET3D, 'Jdet (3D; whole-volume central-diff)'),
```

Add method specs + defaults:

```python
_METHOD_SPECS_TET3D = [
    ('m14', 'M14Tet (harmonic + ALM + L2 refine + repair + polish)'),
    ('m14_schwarz', 'M14-Schwarz3D (cluster decomposition + global polish)'),
    ('m10', 'M10Tet (harmonic + ALM + barrier polish)'),
    ('slsqp_fullgrid', 'SLSQP full-grid 3D (KKT)'),
]
_METHOD_SPECS_JDET3D = [
    ('barrier', 'Barrier (penalty → log-barrier L-BFGS-B)'),
    ('slsqp_windowed', 'SLSQP windowed 3D'),
]
```

Extend `_METHOD_SPECS_BY_CONSTRAINT` and `DEFAULT_METHOD_BY_CONSTRAINT`:

```python
    CONSTRAINT_TET3D: _METHOD_SPECS_TET3D,
    CONSTRAINT_JDET3D: _METHOD_SPECS_JDET3D,
```
```python
    CONSTRAINT_TET3D: 'm14',
    CONSTRAINT_JDET3D: 'barrier',
```

**Important:** the method-algo tags (`m14`, `m14_schwarz`, `m10`, `slsqp_fullgrid`, `barrier`, `slsqp_windowed`) combine with the constraint tag `tet3d`/`jdet3d` to form `m14_tet3d`, `m14_schwarz_tet3d`, etc. — matching the worker's `_build_strategy` keys (Task 5). Confirm `_compose_method_id('m14_schwarz', 'tet3d') == 'm14_schwarz_tet3d'`.

- [ ] **Step 4: Add `_is_3d_run` + gating helper.** In `__init__`, after `self._section_bounds = None`:

```python
        self._is_3d_run = False
```

Add a method:

```python
    def _constraint_is_3d(self, tag: str) -> bool:
        return tag in (CONSTRAINT_TET3D, CONSTRAINT_JDET3D)

    def _update_3d_constraint_enabled(self) -> None:
        """Enable the 3D constraint entries only for D>1 volumes."""
        D = self._volume.shape[1] if self._volume is not None else 1
        model = self._constraint_combo.model()
        for tag in (CONSTRAINT_TET3D, CONSTRAINT_JDET3D):
            idx = self._constraint_combo.findData(tag)
            if idx >= 0:
                model.item(idx).setEnabled(D > 1)

    def _apply_mode_gating(self) -> None:
        """Reflect 2D/3D mode in the run controls."""
        D = self._volume.shape[1] if self._volume is not None else 1
        self._run_roi_btn.setEnabled(not self._is_3d_run)
        self._run_all_btn.setEnabled((not self._is_3d_run) and D > 1)
        self._section_roi.setVisible((not self._is_3d_run) and self._volume is not None)
```

- [ ] **Step 5: Hook constraint changes.** In `_on_constraint_changed`:

```python
    def _on_constraint_changed(self, idx: int):
        constraint = self._constraint_combo.itemData(idx)
        self._is_3d_run = self._constraint_is_3d(constraint)
        self._repopulate_method_combo(constraint)
        self._apply_mode_gating()
```

In `_apply_loaded_run`, after setting `self._run_all_btn.setEnabled(D > 1)`, add:

```python
        self._update_3d_constraint_enabled()
        # A freshly-loaded D==1 field can't stay in a 3D constraint.
        if self._is_3d_run and D <= 1:
            self._select_combo_data(self._constraint_combo, DEFAULT_CONSTRAINT)
        self._apply_mode_gating()
```

- [ ] **Step 6: Pass the full volume in 3D mode.** In `_on_run`, branch at the top of the method (after the running-guard):

```python
        if self._is_3d_run:
            self._section_bounds = None
            self._start_worker(self._original_volume.copy())
            return
```

In `_on_finished`, the splice must handle a full-volume 3D result. Replace the splice block:

```python
        if phi_out is not None and self._volume is not None:
            if self._run_all_remaining is None:
                self._push_undo_state()
            phi_out = np.asarray(phi_out)
            if phi_out.ndim == 4:  # full-volume 3D result [dz,dy,dx]
                self._volume[...] = phi_out
            else:
                sb = self._section_bounds
                if sb is not None:
                    y0, y1, x0, x1 = sb
                    self._volume[1, self._z, y0:y1, x0:x1] = phi_out[0]
                    self._volume[2, self._z, y0:y1, x0:x1] = phi_out[1]
                else:
                    self._volume[1, self._z] = phi_out[0]
                    self._volume[2, self._z] = phi_out[1]
            self._refresh_display_from_volume()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -k "3d" -v`
Expected: PASS

- [ ] **Step 8: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "feat(gui): 3D mode via constraint dropdown + run-control gating"
```

---

## Task 8: ndim-aware rendering, stats, inspector (6-tet default view)

**Files:**
- Modify: `dvfopt_gui/app.py` (`_render_snapshot`, `_set_view`, `_refresh_display_from_volume`, `_format_stats`, `_format_inspector`, `_on_z_changed`)
- Test: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: `six_tet_min_volume_3d`, `jacobian_det3D`, `_metric_counts_3d`, `_infeasible_count_3d`.
- Produces: when the active snapshot/volume is 3D (`phi.ndim == 4`), the heatmap shows the 6-tet min-volume slice at `z` (Jdet3D runs: 3D Jdet slice), stats show whole-volume 3D counts, the inspector shows the voxel min-tet volume, and the z-slider re-slices without resetting the worker.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_app.py`)

```python
def test_3d_render_and_stats(qapp):
    from dvfopt_gui.worker import _volume_snapshot

    vol = np.zeros((3, 4, 8, 8))
    vol[2, :, 3:5, 3:5] = 1.4
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    snap = _volume_snapshot(vol, n_neg=5, min_T=-0.2, outer_iter=1)
    win._render_snapshot(snap)  # must not raise on a 4-D phi
    assert win._img.isVisible() or win._grid_curve.isVisible()
    s = win._format_stats(snap)
    assert 'min_T' in s
    # Idle 3D stats mention the volume shape.
    idle = win._format_stats(None)
    assert '4×8×8' in idle


def test_3d_zslider_reslices_without_dropping_worker(qapp):
    from dvfopt_gui.worker import ReplayHistory, _volume_snapshot

    vol = np.zeros((3, 4, 8, 8))
    win = LiveSolverWindow(vol)
    win._select_combo_data(win._constraint_combo, 'tet3d')
    snap = _volume_snapshot(vol, n_neg=0, min_T=0.16, outer_iter=1)
    win._worker = ReplayHistory([snap], 1)
    win._latest = snap
    win._z_slider.setValue(2)  # re-slice
    assert win._worker is not None  # not reset in 3D mode
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py::test_3d_render_and_stats -v`
Expected: FAIL (render path computes `jacobian_det2D` on a 4-D array / `_on_z_changed` resets the worker)

- [ ] **Step 3: Add a 3D metric-slice helper** to `LiveSolverWindow`:

```python
    def _heatmap_slice_3d(self, phi3d: np.ndarray) -> np.ndarray:
        """The per-slice 3D fold field for the current z (default 6-tet
        min volume; Jdet3D when that constraint is selected). Padded to
        (H, W) with NaN at the trailing row/col so it lines up with the
        grid (the tet field is (D-1, H-1, W-1))."""
        from dvfopt.jacobian.numpy_jdet import jacobian_det3D
        from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

        z = min(self._z, phi3d.shape[1] - 1)
        if self._constraint_combo.currentData() == CONSTRAINT_JDET3D:
            return jacobian_det3D(phi3d)[z]
        mv = six_tet_min_volume_3d(phi3d)  # (D-1, H-1, W-1)
        H, W = phi3d.shape[2:]
        out = np.full((H, W), np.nan)
        zz = min(z, mv.shape[0] - 1)
        out[: H - 1, : W - 1] = mv[zz]
        return out
```

- [ ] **Step 4: Branch `_render_snapshot`/`_set_view` on ndim.** At the top of `_render_snapshot`, before computing `self._latest_jacobian`:

```python
        self._latest = snap
        if snap.phi.ndim == 4:  # 3D volume snapshot
            self._latest_jacobian = self._heatmap_slice_3d(snap.phi)
            self._invalidate_inspector_cache()
            self._set_view_3d(snap.phi, fast=fast)
            self._window_rect.setRect(0, 0, 0, 0)
            self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[], y=[])
            self._stats_label.setText(self._format_stats(snap))
            self._refresh_convergence()
            return
        # ... existing 2D body unchanged ...
```

Add `_set_view_3d`:

```python
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
            field = self._heatmap_slice_3d(phi3d)
            self._img.setImage(field, autoLevels=False)
            self._apply_levels(field)
            self._img.setVisible(True)
            self._img.setOpacity(1.0)
            self._cbar.setVisible(True)
            self._grid_curve.setVisible(False)
            self._fold_overlay.setVisible(False)
        self._update_quiver(slice_2hw)
```

- [ ] **Step 5: 3D idle render + stats.** In `_refresh_display_from_volume`, branch when `self._is_3d_run`:

```python
        if self._volume is None:
            return
        self._invalidate_inspector_cache()
        if self._is_3d_run:
            self._set_view_3d(self._volume, fast=False)
            self._window_rect.setRect(0, 0, 0, 0)
            self._opt_rect.setVisible(False)
            self._target_marker.setData(x=[], y=[])
            self._stats_label.setText(self._format_stats(None))
            self._inspector_label.setText(self._format_inspector(None))
            self._refresh_convergence()
            return
        # ... existing 2D body ...
```

In `_format_stats`, in the `snap is None` (idle) branch, add a 3D path at the top:

```python
            if self._is_3d_run:
                D, H, W = self._volume.shape[1:]
                kind = 'tet3d' if self._constraint_combo.currentData() == CONSTRAINT_TET3D else 'jdet3d'
                n_neg, min_T = _metric_counts_3d(self._volume, kind)
                infeas = _infeasible_count_3d(self._volume, kind)
                thr = FEASIBILITY_THRESHOLD
                return (
                    '<b>Stats (3D)</b><br>'
                    f'volume . . . . {D}×{H}×{W}<br>'
                    f'metric . . . . {kind}<br>'
                    f'3D folds . . . {n_neg}<br>'
                    f'min signed . . {min_T:+.5f}<br>'
                    f'infeasible(&lt;{thr:g}) {infeas}<br>'
                    '(idle — press <i>Run full</i> to start)'
                )
```

Import `_metric_counts_3d`, `_infeasible_count_3d` at the top of `app.py` (extend the `from dvfopt_gui.worker import (...)` block).

- [ ] **Step 6: z-slider re-slice in 3D.** In `_on_z_changed`, branch before the worker-reset logic:

```python
    def _on_z_changed(self, value: int):
        self._z = int(value)
        D = self._volume.shape[1] if self._volume is not None else 1
        self._z_label.setText(f'{self._z} / {D - 1}')
        if self._is_3d_run:
            # In 3D the run spans the whole volume; changing z only
            # re-slices the view — keep the worker/history.
            if self._latest is not None and self._latest.phi.ndim == 4:
                self._render_snapshot(self._latest)
            else:
                self._refresh_display_from_volume()
            return
        # ... existing 2D body (drops worker, resets history) ...
```

- [ ] **Step 7: Inspector voxel readout.** In `_format_inspector`, when `self._latest is not None and self._latest.phi.ndim == 4`, show the voxel's 6-tet min volume:

```python
        if self._latest is not None and self._latest.phi.ndim == 4:
            from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

            phi3d = self._latest.phi
            z = min(self._z, phi3d.shape[1] - 1)
            mv = six_tet_min_volume_3d(phi3d)
            Dm, Hm, Wm = mv.shape
            if not (0 <= y < Hm and 0 <= x < Wm):
                return '<b>Pixel inspector</b><br>(out of bounds)'
            zz = min(z, Dm - 1)
            return (
                '<b>Pixel inspector (3D)</b><br>'
                f'(z={zz}, y={y}, x={x})<br>'
                f'min 6-tet V . {mv[zz, y, x]:+.5f}'
            )
```

(Place this branch at the very start of `_format_inspector`, after the `yx is None` guard.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -k "3d" -v`
Expected: PASS

- [ ] **Step 9: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/app.py tests/test_gui_app.py
git commit -m "feat(gui): ndim-aware 3D rendering, stats, inspector (6-tet default)"
```

---

## Task 9: Progress bar, Stop tooltip, docs, full-suite verification

**Files:**
- Modify: `dvfopt_gui/app.py` (`_update_progress`, Stop tooltip)
- Modify: `CLAUDE.md` (GUI 3D note)
- Test: `tests/test_gui_app.py`; full suite

**Interfaces:**
- Consumes: `_active_method_id` values `m10_tet3d`, `m14_tet3d`, `m14_schwarz_tet3d`, `slsqp_fullgrid_tet3d`, `barrier_jdet3d`, `slsqp_windowed_jdet3d`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_app.py`)

```python
def test_progress_3d_wallbreaker_is_time_budget(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'm14_tet3d'
    win._budget_spin.setValue(60.0)
    win._update_progress()
    assert '/ 60s' in win._progress.format()


def test_progress_3d_fullgrid_is_busy(qapp):
    win = LiveSolverWindow(np.zeros((3, 4, 6, 6)))
    win._worker = _RunningStub()
    win._run_elapsed.restart()
    win._active_method_id = 'slsqp_fullgrid_tet3d'
    win._update_progress()
    assert win._progress.maximum() == 0  # busy indicator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py::test_progress_3d_fullgrid_is_busy -v`
Expected: FAIL (`slsqp_fullgrid_tet3d` starts with `slsqp` → hits the iter-fraction branch, not busy)

- [ ] **Step 3: Update `_update_progress`.** Replace the dispatch so wallbreakers (2D + 3D) use time budget and only `slsqp_windowed*` uses the iter fraction:

```python
        if mid.startswith(('m10', 'm14')):
            budget = float(self._budget_spin.value())
            frac = min(1.0, elapsed / budget) if budget > 0 else 0.0
            self._progress.setRange(0, 100)
            self._progress.setValue(int(frac * 100))
            self._progress.setFormat(f'{elapsed:.0f}s / {budget:.0f}s')
        elif mid.startswith('slsqp_windowed'):
            mx = int(self._max_iter_spin.value())
            cur = self._latest.outer_iter if self._latest is not None else 0
            frac = min(1.0, cur / mx) if mx > 0 else 0.0
            self._progress.setRange(0, 100)
            self._progress.setValue(int(frac * 100))
            self._progress.setFormat(f'iter {cur} / {mx}  ·  {elapsed:.0f}s')
        else:
            # barrier / nmvf / slsqp_fullgrid: busy indicator + elapsed.
            self._progress.setRange(0, 0)
            self._progress.setFormat(f'{elapsed:.0f}s')
```

- [ ] **Step 4: Stop tooltip note.** Where `self._stop_btn` is created, extend the tooltip:

```python
        self._stop_btn.setToolTip(
            'Request the running solve to stop (Esc). In 3D the wallbreaker '
            'methods (M10Tet/M14Tet/M14-Schwarz3D) stop at the next phase '
            'boundary; SLSQP-fullgrid-3D / Barrier run to completion '
            '(bound them with time_budget_s / max_iter).'
        )
```

- [ ] **Step 5: Run the progress tests**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_app.py -k progress -v`
Expected: PASS

- [ ] **Step 6: Document in CLAUDE.md.** Under the GUI section (the `dvfopt_gui/` description), add a sentence:

> The GUI also supports a **true-3D mode**: load a `(3, D, H, W)` volume and pick the `6-tet (3D)` or `Jdet (3D)` constraint to solve the whole volume with the 3D pipelines (M14Tet/M14-Schwarz3D/M10Tet/SLSQP-fullgrid-3D, or Barrier/SLSQP-windowed for Jdet3D). 3D wallbreaker runs stream per-phase snapshots and honor Stop at phase boundaries; the viewer renders the 6-tet min-volume slice of the current z.

- [ ] **Step 7: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q`
Expected: PASS (all; new 3D + GUI tests included)

- [ ] **Step 8: Lint**

Run: `python -m ruff check dvfopt_gui/ dvfopt/strategies/wallbreakers.py dvfopt/core/wallbreakers/_refine_repair_3d.py dvfopt/core/wallbreakers/_m14_schwarz_3d.py tests/`
Expected: `All checks passed!`

- [ ] **Step 9: Commit** (checkpoint — confirm with user first)

```bash
git add dvfopt_gui/app.py CLAUDE.md tests/test_gui_app.py
git commit -m "feat(gui): 3D progress bar + Stop semantics; document 3D mode"
```

---

## Self-Review

**Spec coverage check:**
- §1 scope (2D additions / 3D methods) → Tasks 1, 5, 7. ✓
- §4 mode selection & gating → Task 7. ✓
- §5 library `step_callback` → Tasks 3, 4. ✓
- §6 worker 3D path → Task 5. ✓
- §7 snapshots + memory guard + rendering → Tasks 5 (guard/snapshot), 8 (render). ✓
- §8 persistence → Task 6. ✓
- §9 progress bar → Task 9. ✓
- §11 testing → tests in every task + full suite in Task 9. ✓
- §12 risks (crop guard, channel order, memory) → Task 5 crop guard + guard; Task 8 channel handling. ✓

**Type consistency:** `_metric_counts_3d`/`_infeasible_count_3d` (Task 2) used identically in Tasks 5, 8. `_volume_snapshot` (Task 5) used in Tasks 6, 8 tests. `_is_3d_run` defined Task 7, referenced Task 6 via `getattr` (until Task 7 lands) and Task 8. Method-id tags consistent: `_compose_method_id('m14_schwarz','tet3d')='m14_schwarz_tet3d'` matches `_build_strategy` key (Task 5) and method specs (Task 7). `step_callback` contract `{'phi','stage'}` identical in Tasks 3, 4, 5.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test step has real assertions.

**Ordering note:** Task 6 Step 6 references `_is_3d_run` via `getattr(..., False)` so it is safe to land before Task 7; once Task 7 lands, the attribute always exists. Recommend executing in numeric order (1→9).
