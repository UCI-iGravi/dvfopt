# dvfopt GUI: true-3D mode + full 2D strategy coverage

**Date:** 2026-06-25
**Status:** Approved design, pending implementation plan
**Scope:** `dvfopt_gui/` (GUI) + a small backward-compatible change to the
3D wallbreaker cores in `dvfopt/`.

## 1. Problem

The live-viz GUI (`dvfopt_gui`) exposes only a curated subset of dvfopt's
current optimization surface, and it is fundamentally 2D:

- **Missing 2D strategies:** `SLSQPFullGridStrategy` and `SchwarzStrategy`
  (2-tri) are implemented in the package but absent from the method menu.
- **No true 3D:** every native 3D strategy (`HarmonicALMBarrier3DStrategy`
  /M10Tet, `HarmonicALMRefineRepair3DStrategy`/M14Tet,
  `SchwarzHarmonicALMRefineRepair3DStrategy`/M14-Schwarz3D,
  `SLSQPFullGrid3DStrategy`, plus the Jdet3D path through `BarrierStrategy`
  / `SLSQPWindowedStrategy`) is unreachable, along with the 3D constraints
  (`Tet6Constraint3D`, `JdetConstraint3D`). The GUI loads `(3, D, H, W)`
  volumes but only ever solves per-slice in 2D ("Run all z"), with **no
  inter-slice coupling** — it never measures or corrects z-direction folds.

This spec adds the missing 2D strategies and a genuine 3D mode that solves
the whole volume with the 3D constraints + strategies.

## 2. Goals / non-goals

**Goals**
- Add `SLSQP-fullgrid` and `Schwarz` to the 2-tri method dropdown.
- Add a 3D mode selectable via the Constraint dropdown (`6-tet (3D)`,
  `Jdet (3D)`), enabled only for `D > 1` volumes.
- Expose the complete 3D pipelines: 6-tet → M14Tet, M14-Schwarz3D, M10Tet,
  SLSQP-fullgrid-3D; Jdet3D → Barrier, SLSQP-windowed.
- True-3D fold metric (6-tet signed volume / 3D Jdet) for stats, heatmap,
  convergence, and inspector.
- Live **per-phase** staging for the 3D wallbreaker methods, with a
  responsive Stop, via a new optional `step_callback` hook in the 3D
  wallbreaker cores.
- Memory-guarded staged history + persistence round-trip for 3D runs.

**Non-goals (explicitly out of scope)**
- SchwarzWrapper(inner=…) and composite objectives (`ScaledObjective`,
  `SumObjective`).
- Experimental / building-block 3D strategies (`CoupledKRing3DStrategy`,
  `ActiveBandALM3DStrategy`, `Harmonic3DStrategy`, `ALM3DStrategy`,
  `BarrierTet3DTorchStrategy` / torch).
- A 3D ROI ("Run section") or 3D volumetric rendering — the viewer stays
  2D-per-slice.
- Making Stop interrupt the *non-phased* 3D methods (SLSQP-fullgrid-3D,
  Barrier-Jdet3D) mid-solve — those stay best-effort, like the 2D Barrier
  today.

## 3. Background facts (verified against the codebase)

- `six_tet_min_volume_3d(phi)` expects `phi` as `(3, D, H, W)` =
  `[dz, dy, dx]` — **the GUI's own volume convention** — and returns the
  per-voxel min 6-tet signed volume. Fold count `= (min_V <= 0).sum()`;
  feasibility margin `< threshold` (0.01, per `DEFAULT_PARAMS`).
- `Tet6Constraint3D` / `JdetConstraint3D` **accept the canonical
  `(3, D, H, W)` `[dz, dy, dx]` array directly** (their `coerce`/`flatten`
  repack to the `[dx, dy, dz]` flat vector internally), and `unflatten`
  returns `[dz, dy, dx]`. So `solver.fit` is handed the GUI volume as-is
  and `result.corrected` comes back in the same `[dz, dy, dx]` convention —
  **no GUI-side channel reorder**. The cores likewise emit `[dz, dy, dx]`.
- `Solver.fit(phi, record_history=…, verbose=…, **kwargs)` forwards extra
  kwargs to `strategy.solve`. 2D wallbreaker `solve()` already accepts
  `step_callback`; **3D wallbreaker `solve()` currently has `**_`** that
  silently swallows it, and the 3D cores have **no callback / stop hook**
  at all today.
- 3D wallbreaker cores honor `time_budget_s` internally (self-limiting).

## 4. Mode selection & UI gating

The **Constraint** dropdown gains two entries:

| label | tag | family |
|---|---|---|
| `2-tri (full-coverage…)` | `2tri` | 2D (existing) |
| `Jdet (central-diff…)` | `jdet` | 2D (existing) |
| `6-tet (3D)` | `tet3d` | 3D (new) |
| `Jdet (3D)` | `jdet3d` | 3D (new) |

The two new tags (`tet3d`, `jdet3d`) are single tokens with no internal
underscore, so `method_id = <algo>_<tag>` round-trips cleanly through
`rpartition('_')` even for multi-word algos.

- The 3D entries are **enabled only when a `D > 1` volume is loaded**;
  otherwise greyed (with an explanatory tooltip).
- Selecting a 3D constraint repopulates the Method dropdown with the 3D
  methods (reuses `_repopulate_method_combo`) and switches the window to
  **3D mode** (`self._is_3d_run = True` derived from the selected
  constraint tag).
- Loading a `D == 1` field while a 3D constraint is selected auto-reverts
  the constraint to the previous 2D selection.

Per-constraint method specs:

```
tet3d  -> m14_tet3d, m14_schwarz_tet3d, m10_tet3d, slsqp_fullgrid_tet3d  (default m14_tet3d)
jdet3d -> barrier_jdet3d, slsqp_windowed_jdet3d                          (default barrier_jdet3d)
2tri   -> m14, m14_schwarz, m10, barrier, slsqp_windowed,
          slsqp_fullgrid, schwarz                                        (default m14)   # + 2 new
jdet   -> barrier, slsqp_windowed, nmvf                                  (unchanged)
```

`method_id = <algo>_<tag>` is preserved; `run()` recovers the algo with
`rpartition('_')`, e.g. `m14_schwarz_tet3d` → `('m14_schwarz', '_',
'tet3d')` and `slsqp_fullgrid_tet3d` → `('slsqp_fullgrid', '_', 'tet3d')`.

**Run controls in 3D mode:**
- **Run full** → solve the entire volume in 3D.
- **Run section** (ROI) and **Run all z** → disabled (whole volume is one
  run). Tooltip explains why.
- **z-slider** stays active — it re-slices the displayed snapshot for
  viewing, never re-solves.
- **Stop** → enabled; effective at the next phase boundary for the
  wallbreaker methods, best-effort for SLSQP-fullgrid-3D / Barrier (tooltip
  states this).

## 5. Library change: `step_callback` in the 3D wallbreaker cores

Add an optional `step_callback=None` parameter to:

- the 3D wallbreaker **strategy** `solve()` methods
  (`HarmonicALMBarrier3DStrategy`, `HarmonicALMRefineRepair3DStrategy`,
  `SchwarzHarmonicALMRefineRepair3DStrategy`) — replacing the silent `**_`
  swallow with an explicit forward; and
- the phase boundaries where each pipeline sequences stages. These live at
  the **orchestration level** (after each stage function returns a valid
  `(3, D, H, W)` field), so `_alm_3d.py` / `_harmonic_3d.py` internals are
  **not** touched:
  - **M14Tet:** in the core `iterative_3d_tet_refine_repair` — after seed,
    pull, repair, and final polish (4 calls).
  - **M10Tet:** in `HarmonicALMBarrier3DStrategy.solve` — after the
    harmonic seed and after ALM (the barrier-polish stage delegates to
    `BarrierStrategy`, which is one-shot; no extra call).
  - **M14-Schwarz3D:** thread `step_callback` through
    `_m14_schwarz_3d.iterative_3d_tet_refine_repair_schwarz` into its
    `inner_solve` (the per-cluster `iterative_3d_tet_refine_repair` call),
    so it fires at each cluster's phase boundaries. Those `phi` are
    **crops**, not the full volume — the worker uses them only for the stop
    check and skips snapshotting them (see Section 6 step 4).

**Contract** (mirrors the 2D wallbreaker `_stage_callback`):
`step_callback({'phi': <current (3,D,H,W) [dz,dy,dx] field>, 'stage': <str>})`
is invoked **at each phase boundary** (seed / pull / refine / repair /
polish, and per-cluster + global-polish for Schwarz). Only called when a
callback is supplied — default `None` preserves current behavior exactly,
so existing tests are unaffected.

The callback may raise `KeyboardInterrupt` to abort; cores must let it
propagate (no bare `except:` swallowing it). The orchestrator should fire
the callback *after* completing a phase so the emitted `phi` is a valid
intermediate field.

Non-phased methods (`SLSQPFullGrid3DStrategy`, `BarrierStrategy`) are **not**
modified — they receive no callback and run to completion / internal budget.

## 6. Worker: the 3D run path

`SolverWorker` gains a 3D branch. `run()` dispatches on the constraint tag
(`tet3d`/`jdet3d` → 3D path; `2tri`/`jdet` → existing 2D paths).

`_run_via_solver_3d(strategy, constraint_kind, *, metric_kind='tet3d'|'jdet3d')`:

1. Read the **full** volume `(3, D, H, W)` `[dz, dy, dx]` from the worker's
   `deformation_i` (in 3D mode the worker is handed the whole volume, not a
   single slice).
2. Build `Tet6Constraint3D(shape=(D,H,W))` or `JdetConstraint3D(shape=
   (D,H,W))`, the objective, and the strategy. The volume is passed to
   `fit` as-is — the constraint repacks internally (no reorder).
3. Emit an **initial** 3D snapshot (`stage='input'`) under `metric_kind`.
4. `solver.fit(phi3d, step_callback=_stage_callback_3d, record_history=False)`
   for the phased wallbreakers; for the non-phased methods, call without a
   usable callback (they ignore it) and rely on the final synthetic
   snapshot. `_stage_callback_3d` receives `phi` already in `[dz,dy,dx]`;
   it checks the stop flag (raise `KeyboardInterrupt`), and — **only when
   `phi.shape == (3, D, H, W)`** (full volume; Schwarz emits per-cluster
   crops that are skipped for snapshotting) — computes `(n_neg, min_T)` via
   the 3D metric and records a snapshot.
5. Emit a **final** synthetic snapshot from `result.corrected` (already
   `[dz,dy,dx]`), recounted under `metric_kind`.
6. Return the corrected volume `(3, D, H, W)` `[dz,dy,dx]`.

**3D metric helpers** (new, alongside `_metric_counts`):
- `_metric_counts_3d(phi3d, kind)` → `(n_neg, min_T)`:
  `kind='tet3d'` → `six_tet_min_volume_3d`; `kind='jdet3d'` →
  `jacobian_det3D`. Counts `<= 0`, min over the volume.
- `_infeasible_count_3d(phi3d, kind, threshold)` → `< threshold`.
These live in `dvfopt_gui/worker.py` next to their 2D counterparts (or a
small shared `_metrics.py` if `worker.py` grows unwieldy — decided during
implementation).

## 7. Snapshots, memory guard, rendering

**Snapshot generalization.** `StateSnapshot.phi` may be `(2, H, W)` (2D) or
`(3, D, H, W)` (3D). Consumers branch on `phi.ndim`. The scalar bookkeeping
(`n_neg`, `min_T`, `outer_iter`, `stage`/window fields) is unchanged; 3D
snapshots set window/opt rects to zero (no active-window overlay in 3D).

**Memory guard.** 3D history uses a small cap (default 8 — phases are few).
Before a 3D run, estimate `cap * 3*D*H*W*8` bytes; if it exceeds a budget
(~2 GB, a module constant), the worker keeps only **init + final** 3D
snapshots and posts a status-bar note. The same guard bounds Save (Section 8).

**Rendering in 3D mode** (`_render_snapshot` / `_set_view` branch on
`phi.ndim`; current `z` from the z-slider selects the slice):
- **Default 3D heatmap view:** `six_tet_min_volume_3d(phi3d)[z]` (Jdet3D
  runs: `jacobian_det3D(phi3d)[0][z]`) — diverging colormap, red=feasible.
- **2-tri / Jdet 2D views:** remain available as alternates, computed on the
  `(dy,dx)` of slice `z`.
- **Deformation-grid view:** warp of `(dy, dx)` at slice `z`; fold overlay
  from the 3D tet metric at `z`.
- **Convergence chart:** stage trajectory of whole-volume `n_neg` / `min_T`.
- **Inspector:** clicked voxel `(z, y, x)` shows its min 6-tet volume (3D)
  in addition to the existing 2D readouts.
- **Stats panel:** whole-volume 3D fold count + infeasible(<thr) count +
  min signed volume, plus the `D×H×W` shape.

The z-slider `valueChanged` in 3D mode re-renders the **current snapshot**
at the new slice (no worker reset — unlike the 2D per-slice path where
changing z invalidates the run).

## 8. Persistence

Extend the NPZ schema (back-compatible):
- Add `dim` (0-d int, 2 or 3) so load can branch; absence ⇒ 2D.
- 3D history stored as `history_phi` shape `(N, 3, D, H, W)` (bounded by the
  Section 7 guard). `phi_full_volume` / `phi_input_volume` are already
  `(3, D, H, W)`.
- `final_min_jdet` / `final_n_neg_jdet` generalize to the run's metric
  (record `final_metric_kind`).
- `parse_loaded` reconstructs 3D snapshots when `dim == 3`; existing 2D
  archives load unchanged.

`normalise_to_volume` already accepts `(3, D, H, W)`. Loading a saved 3D run
restores the volume, the input baseline, the staged history, and the
constraint/method/objective selections (which re-enter 3D mode via the
constraint).

## 9. Progress bar

`_update_progress` keys off `_active_method_id`:
- `m10_tet3d` / `m14_tet3d` / `m14_schwarz_tet3d` → elapsed /
  `time_budget_s` (all start with `m10`/`m14`, so the existing prefix
  check already routes them to the time-budget branch).
- `slsqp_fullgrid_tet3d` → busy indicator + elapsed (no clean fraction;
  needs a guard so it isn't mistaken for the windowed iter-fraction path).
- `barrier_jdet3d` → busy indicator + elapsed.
- `slsqp_windowed_jdet3d` → if the 3D windowed path reports outer-iter,
  iter/max; else busy + elapsed.

## 10. Components touched

| File | Change |
|---|---|
| `dvfopt_gui/app.py` | Constraint dropdown (+2 entries, D>1 gating), method specs (+2D, +3D), `_is_3d_run` state, run-control gating, ndim-aware `_set_view`/`_render_snapshot`/`_format_stats`/inspector, z-slider re-slice in 3D, `_build_strategy`/dispatch wiring through worker |
| `dvfopt_gui/worker.py` | `_run_via_solver_3d`, `_metric_counts_3d`, `_infeasible_count_3d`, 3D `_build_strategy` cases, `_trajectory_metric_kind` for `tet3d`/`jdet3d`, ndim-aware snapshot emit, memory guard |
| `dvfopt_gui/persistence.py` | `dim` flag, 3D `history_phi`, `parse_loaded` 3D branch |
| `dvfopt/strategies/wallbreakers.py` | explicit `step_callback=None` on the three 3D wallbreaker `solve()` methods; M10Tet fires after harmonic + ALM; M14Tet/Schwarz3D forward to their cores |
| `dvfopt/core/wallbreakers/_refine_repair_3d.py` | optional `step_callback`, fired after seed/pull/repair/polish; let `KeyboardInterrupt` propagate |
| `dvfopt/core/wallbreakers/_m14_schwarz_3d.py` | optional `step_callback`, forwarded into the per-cluster `inner_solve` core call |

## 11. Testing

**Headless (`tests/test_gui_logic.py`)**
- `_metric_counts_3d` / `_infeasible_count_3d` on a synthetic folded
  volume; folds at `<=0`, infeasible at `<thr`.
- A `(3,D,H,W)` `[dz,dy,dx]` volume passed through `solver.fit` with a 3D
  constraint round-trips channel order (corrected result still `[dz,dy,dx]`).
- `_run_via_solver_3d` end-to-end on a small (e.g. 4×8×8) folded volume:
  reaches `n_neg == 0` (or improves) with M14Tet; returns `(3,D,H,W)`.
- Memory guard: a volume whose estimate exceeds the budget yields only
  init+final snapshots.
- Persistence 3D round-trip (`dim=3`, `history_phi` shape, snapshots
  restored, input vs corrected distinct).
- `step_callback` fires per phase; raising `KeyboardInterrupt` aborts the
  3D wallbreaker core cleanly (returns/raises without corrupting state).
- Back-compat: existing 3D-core tests pass with `step_callback=None`.

**Widget (`tests/test_gui_app.py`, offscreen)**
- 3D constraint entries disabled for `D==1`, enabled for `D>1`; selecting
  one repopulates 3D methods and disables Run-section / Run-all.
- 2D additions present in the 2-tri menu; `_build_strategy` returns
  `SLSQPFullGridStrategy` / `SchwarzStrategy`.
- z-slider in 3D mode re-slices without dropping the worker/history.
- 3D idle/snapshot stats show the volume shape + 3D fold + infeasible line.

**Full suite** (`pytest tests/`) — run because library internals change.

## 12. Risks & mitigations

- **Library regression** from the `step_callback` plumbing → param defaults
  to `None`; phase-boundary calls guarded; full suite run; let
  `KeyboardInterrupt` propagate (audit for bare `except`).
- **Memory blow-up** on large volumes → the Section-7 guard caps in-memory
  and on-disk history; default cap 8.
- **`app.py` growth** (already ~2000 lines) → 3D metric/snapshot helpers go
  in `worker.py` (or a small `_metrics.py`); `app.py` gets only the
  mode-switch + ndim branches. No unrelated refactor.
- **Channel-order bugs** → low risk now that the 3D constraints accept the
  GUI's native `[dz,dy,dx]` directly (no GUI-side reorder). A `fit`
  round-trip test guards it.
- **Schwarz crop snapshots** (per-cluster `phi` are crops, not full
  volumes) → the worker's stage callback snapshots only when
  `phi.shape == (3, D, H, W)`; crops drive the stop check only.
