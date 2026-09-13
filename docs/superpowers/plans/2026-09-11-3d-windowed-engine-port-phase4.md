# 3D Windowed Engine Port — Phase 4 (Scale) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the windowed I-SLSQP engine on full-resolution 3D volumes — a banded full-volume driver whose workers receive only their z-slab, resumable checkpoints at round / tiler-sweep / band granularity, the engine's whole-field-per-sweep scans and whole-field-per-tile pickles removed — measured on a raw B0039 slab, the ds2 B0039 cohort volume, and then the full-res `(3, 528, 320, 456)` B0039 Laplacian-exterior field, reporting 0 fixed-6-tet folds at threshold and at 0, the best-diagonal floor, damage 0, L2/L1 move and wall.

**Architecture:** Three engine-side changes, all in `dvfopt/core/windowed/`: (1) the giant tiler counts folds on the giant box only (`_folds_in_box`, byte-identical to slicing the whole-field map) and its RAS workers receive the tile's ring+1-padded patch instead of the whole field; (2) `windowed_correct(checkpoint_dir=)` mirrors the field through the shared `RunCheckpoint` after the coarse stage, every round, every tiler sweep, the mop, the re-seed and the re-anchor, with `touched.npy` beside it so a resumed run keeps the no-damage accounting against the ORIGINAL input; (3) a new `windowed_correct_banded` in `_banded.py` splits z into overlapping bands solved from one snapshot on the shared spawn pool (restricted additive Schwarz across bands: each worker returns only its core), commits the cores, then runs one serial `windowed_correct` on the composed field as the seam pass — the parent holds ONE full copy, workers hold slabs. A benchmark driver builds the ds2 volumes with the existing `_downsample_2x` and records the certificate block. No routing, default or strategy-knob change; the 2D path stays byte-identical.

**Tech Stack:** numpy / scipy.ndimage, the numba certificate kernels in `dvfopt/jacobian/tetrahedron_sign.py`, `dvfopt/checkpoint.py` `RunCheckpoint`, `dvfopt/core/_pool.py` (shared spawn pool, `pool_map`, `pin_worker_threads`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-3d-windowed-engine-port-design.md` — phase-4 line (§ "Phases", item 4, ~line 103: "chunked full-volume driver (Part XX's memory reality: one field copy per process, workers get patches only), `checkpoint_dir` per round/tile, ds2 cohort volumes, then the full-res B0039. Certification = 0 folds at fixed-diagonal 6-tet, the best-diagonal floor reported alongside, damage 0, L2 move") and its "Review correction (memory)" (~line 143: today "~5 GB per process, not 'one copy'"). Survey facts this plan relies on (2026-09-11, main d1d9820): full-res units — 77.0 M voxels, float64 field 1.85 GB, cube-grid float64 616 MB, bool 77 MB; whole-run live set `phi` + `j0` + `orig_fold` + `touched` ≈ 2.6 GB; largest transient the final `best_diagonal_min_volume` block 2.45 GB; the tiler calls `min_field(constraint, phi)` on the WHOLE field once per sweep (`_common.py:1737`, `:1763`) and its RAS path pickles `snap = phi.copy()` per tile (`:1712-1731`); `RunCheckpoint` mirrors whole-field units when `slab=lambda _u: Ellipsis` (`pipeline_3d.py:234`); the 2.5D segment sweep already hands each worker only its slab (`pipeline_25d.py:377`); `research/strict_feasibility_3d/REPORT.md` Part XX: the full volume has 728 533 6-tet folds spread uniformly (~1 382 per cube layer), the proven research path was 22 overlapping z-bands (24 + 4 overlap) with a per-band checkpoint, ~58 h; the box has 63.7 GB RAM.

## Global Constraints

- **2D byte-identity:** every 2D path is unchanged — `benchmarks/windowed_2d_identity.py --out <set>` on the branch head vs the `route` set (`benchmarks/output/identity_2d/route`, main at ff07b2d/d1d9820) must print `IDENTITY PASS` at every task boundary that touches `dvfopt/core/windowed/`.
- **3D reproduction to the iteration:** after Task 1, `benchmarks/windowed_3d_vs_auto.py --case {twist,sliver,sub20,subvol16} --method isqp_windowed` must reproduce the h2h rows (`benchmarks/output/windowed_3d/h2h_<case>_isqp_windowed_l2.json`) in `folds_out`, `floor_out`, `l2_move` (to 1e-9) and the engine's SQP iterations (`sqp_iters` in the record is the driver's de-duplicated sum — compare the same key).
- **Certificate semantics unchanged:** fixed-diagonal 6-tet count at threshold and at 0, plus the best-diagonal floor (`n_neg_best_diagonal`), damage 0 against the ORIGINAL input.
- **No default, routing, strategy-field or GUI change** in this PR (`auto_strategy`, `DEFAULTS_BY_DIM`, `ISQPWindowedStrategy`'s dataclass fields untouched); the new entry points are engine kwargs and a new function.
- Python `>=3.10`; ruff 0.16.3 (`ruff check dvfopt dvfopt_gui tests benchmarks asv_bench`, `ruff format --check`); `mypy` clean. Tests under `tests/` use synthetic fields only (`tests/conftest.py` `planted_fold_3d`, `dvfopt.testdata`).
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; PR to `UCI-iGravi/dvfopt` (never heemmanshuu), squash-merge on all-green.
- Long runs are the controller's: background, from the main checkout root with `PYTHONPATH=<detached snapshot worktree>`; kill by PID only; check the box's load before every measurement; subagents never start them.

---

### Task 1: Box-scoped tiler fold counts + patch-only RAS tasks (byte-identical)

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — add `_folds_in_box` after `_pad_box` (~line 422); replace the two whole-field counts at ~`:1737` and ~`:1763`; the RAS task list at ~`:1712-1731` and `_ras_tile_task` at ~`:1815-1841`.
- Test: `tests/test_windowed_phase4.py` (new)

**Interfaces:**
- Consumes: `min_field(constraint, phi)` (`dvfopt/core/windowed/_locality.py:291`), `_pad_box(box, shape, pad)`, `_box_slices(box)`, `_solve_window(phi, constraint, box, threshold, objective, maxiter, ring, rep, *, margin_delta, allow_grow, inner, opts)`, `WindowRec` (`_common.py:454`; box-valued fields `patch_box` (per-axis `(lo, hi)` flat tuple), `fy0`, `fx0`).
- Produces: `_folds_in_box(constraint, phi, box, threshold) -> int`; `_ras_tile_task(args)` with `args = (patch, offset, constraint, tb_local, core_local, threshold, objective, maxiter, ring, margin_delta, inner, opts)` returning `(core_global, values, rep)` with every `WindowRec.patch_box` / `fy0` / `fx0` in GLOBAL coordinates.

Why the padded patch is equivalent: `build_subproblem` (`_common.py:255`) pads the free box by `ring` and clips to the array's shape, and the "image border" flags it derives are true exactly where that clipping happened. Handing the worker the tile padded by `ring + 1` (clipped to the true shape) makes the worker's own `ring` padding clip only where the true image border clipped it — so the flags, the patch content and every enforced row are identical to the whole-field call. One extra grid plane per side is the whole cost.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_windowed_phase4.py
import numpy as np
import pytest

from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
    SimplexConstraint3D,
)
from dvfopt.core.windowed import _common as cm
from dvfopt.core.windowed import min_field


def _rand_field(rng, shape, amp=0.6):
    return rng.normal(0.0, amp, size=(len(shape), *shape))


@pytest.mark.parametrize(
    'cls, shape',
    [
        (JdetConstraint2D, (23, 29)),
        (FiniteJdetConstraint2D, (23, 29)),
        (SimplexConstraint2D, (23, 29)),
        (SimplexConstraint2DBilinear, (23, 29)),
        (SimplexConstraint3D, (11, 13, 12)),
    ],
)
def test_folds_in_box_matches_whole_field_slice(cls, shape):
    rng = np.random.default_rng(7)
    phi = _rand_field(rng, shape)
    c = cls(shape=shape)
    whole = min_field(c, phi) < 0.01
    boxes = [
        tuple(v for n in shape for v in (0, n)),  # the whole grid
        tuple(v for n in shape for v in (0, max(3, n // 2))),  # touching the low border
        tuple(v for n in shape for v in (n // 3, n)),  # touching the high border
        tuple(v for n in shape for v in (2, n - 2)),  # strictly interior
    ]
    for box in boxes:
        want = int(whole[cm._box_slices(box)].sum())
        assert cm._folds_in_box(c, phi, box, 0.01) == want


def test_ras_tile_task_on_patch_equals_full_snapshot_solve():
    """The patch-only worker reproduces, byte for byte, a tile solved on a private copy
    of the WHOLE snapshot (the pre-phase-4 semantics), records translated to global."""
    pytest.importorskip('osqp')
    from dvfopt.objectives import L2Objective
    from tests.conftest import planted_fold

    phi = planted_fold(40, 44).astype(np.float64)  # (2, 40, 44) [dy, dx], one fold cluster
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    ring = cm._locality_of(c).ring
    opts = cm._resolve_opts_for_test(dim=2)  # see Step 3 — a tiny helper exposing _InnerOpts
    tb = (12, 30, 10, 32)  # a tile box (y0, y1, x0, x1) covering the fold
    core = (14, 26, 12, 28)
    # reference: old semantics
    ref = np.array(phi, dtype=np.float64, copy=True)
    rep_ref = cm.SliceReport()
    cm._solve_window(
        ref, c, tb, 0.01, L2Objective(), 400, ring, rep_ref,
        margin_delta=1e-3, allow_grow=False, inner='isqp', opts=opts,
    )
    want = ref[(slice(None), *cm._box_slices(core))].copy()
    # new: patch-only
    shape = phi.shape[1:]
    pb = cm._pad_box(tb, shape, ring + 1)
    off = tuple(pb[2 * a] for a in range(len(shape)))
    patch = phi[(slice(None), *cm._box_slices(pb))].copy()
    tb_local = tuple(tb[i] - off[i // 2] for i in range(len(tb)))
    core_local = tuple(core[i] - off[i // 2] for i in range(len(core)))
    got_core, got, rep = cm._ras_tile_task(
        (patch, off, c, tb_local, core_local, 0.01, L2Objective(), 400, ring, 1e-3, 'isqp', opts)
    )
    assert got_core == core
    np.testing.assert_array_equal(got, want)
    assert [w.patch_box for w in rep.windows] == [w.patch_box for w in rep_ref.windows]
    assert [(w.fy0, w.fx0) for w in rep.windows] == [(w.fy0, w.fx0) for w in rep_ref.windows]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_windowed_phase4.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module ... has no attribute '_folds_in_box'` (and `_resolve_opts_for_test`).

- [ ] **Step 3: Implement `_folds_in_box`, the test helper, and the patch-only RAS task**

In `_common.py`, after `_pad_box` (~line 422):

```python
def _folds_in_box(constraint, phi, box, threshold):
    """``int((min_field(constraint, phi)[_box_slices(box)] < threshold).sum())`` computed on
    the box padded by ONE grid point per side (clipped to the field), never on the whole
    field. Every cell / cube / central-difference stencil whose value lands inside ``box``
    reads at most one grid point beyond it, so the padded sub-field reproduces those values
    exactly; the sub-field's own last plane / row / column is ``+inf`` exactly where the
    whole field's is (the true border) and is outside ``box`` otherwise."""
    shape = phi.shape[1:]
    pb = _pad_box(box, shape, 1)
    sub = min_field(constraint, phi[(slice(None), *_box_slices(pb))])
    inner = tuple(
        slice(box[2 * a] - pb[2 * a], box[2 * a + 1] - pb[2 * a]) for a in range(len(shape))
    )
    return int((sub[inner] < threshold).sum())


def _resolve_opts_for_test(dim):
    """Test hook: the engine's resolved ``_InnerOpts`` at the defaults for ``dim`` (the
    tuple ``windowed_correct`` builds at its entry), so a test can call ``_solve_window``
    / ``_ras_tile_task`` directly with the shipped knobs."""
    kw = resolve_dim_defaults(dim)
    return _InnerOpts(**{k: kw[k] for k in _InnerOpts._fields if k in kw})
```

> The implementer reads how `windowed_correct` constructs `opts` (~lines 960-1001: an `_InnerOpts(...)` positional call from the resolved knobs) and makes `_resolve_opts_for_test` build the SAME object — if `_InnerOpts` is not a namedtuple with `_fields`, mirror the constructor call exactly instead of the dict comprehension. The helper is for tests only; keep it next to `_folds_in_box`.

Replace the two counts (~`:1737` and ~`:1763`):

```python
            nf = _folds_in_box(constraint, phi, giant_box, threshold)
```

(both sites; `gsl` stays defined for the other uses in the function, or delete it if it becomes unused — `ruff` will say.)

Replace the RAS task list (~`:1712-1731`) — the snapshot is still taken once per sweep, but every task carries only its padded patch:

```python
            # each task pickles only the tile's ring+1-padded patch (~tile³ floats, not the
            # field): `build_subproblem` pads the local box by `ring` and clips to the patch,
            # which clips exactly where the true image border clipped the +1, so the enforced
            # rows and border flags are those of the whole-field solve (see Task 1 of the
            # phase-4 plan; asserted byte-for-byte in tests/test_windowed_phase4.py).
            snap = phi.copy()
            shape = phi.shape[1:]
            args = []
            for tb, core in zip(tiles, cores):
                if not (_nonempty(tb) and _nonempty(core)):
                    continue
                pb = _pad_box(tb, shape, ring + 1)
                off = tuple(pb[2 * a] for a in range(len(shape)))
                args.append(
                    (
                        snap[(slice(None), *_box_slices(pb))].copy(),
                        off,
                        constraint,
                        tuple(tb[i] - off[i // 2] for i in range(len(tb))),
                        tuple(core[i] - off[i // 2] for i in range(len(core))),
                        threshold,
                        objective,
                        maxiter,
                        ring,
                        margin_delta,
                        inner,
                        opts,
                    )
                )
            del snap
```

and `_ras_tile_task`:

```python
def _ras_tile_task(args):
    """Pool worker: solve ONE giant tile on its ring+1-padded patch (never the whole
    field) and return the tile's core values, boxes translated back to global."""
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()
    patch, off, constraint, tb, core, threshold, objective, maxiter, ring, margin_delta, inner, opts = args
    phi = np.array(patch, dtype=np.float64, copy=True)
    rep = SliceReport()
    _solve_window(
        phi, constraint, tb, threshold, objective, maxiter, ring, rep,
        margin_delta=margin_delta, allow_grow=False, inner=inner, opts=opts,
    )
    nd = len(off)
    for w in rep.windows:
        if w.patch_box:
            w.patch_box = tuple(w.patch_box[i] + off[i // 2] for i in range(len(w.patch_box)))
        w.fy0 += off[nd - 2]
        w.fx0 += off[nd - 1]
    core_global = tuple(core[i] + off[i // 2] for i in range(len(core)))
    return core_global, phi[(slice(None), *_box_slices(core))].copy(), rep
```

> Check `WindowRec.fy0/fx0` semantics on 3D before translating (`_common.py:454-459`: "2D projections beside `patch_box`"); if on 3D they hold the last two axes' patch origin, the `off[nd - 2] / off[nd - 1]` translation above is right; if they are unused on 3D (0), translating by the offset would make them nonzero — guard with `if w.ph or w.pw:`. Read `_solve_window`'s record construction (~`:1900-2010`) to decide, and say which in the commit message.

- [ ] **Step 4: Run the tests, then the RAS and 3D-stage suites**

Run: `pytest tests/test_windowed_phase4.py tests/test_windowed_ras.py tests/test_windowed_3d_stages.py -q -p no:cacheprovider`
Expected: all PASS (`test_windowed_ras.py` holds the serial-vs-RAS and `giant_workers=0`-byte-identical pins; they must not change).

- [ ] **Step 5: Lint, format, mypy, commit**

```bash
ruff check dvfopt tests && ruff format --check dvfopt tests && mypy
git add dvfopt/core/windowed/_common.py tests/test_windowed_phase4.py
git commit -m "windowed: the giant tiler counts folds on the giant box only (_folds_in_box, byte-identical) and its RAS workers receive the tile's ring+1 patch, not the field"
```

**Controller gate after Task 1 (not the subagent's):** 2D identity set + compare vs `route` → `IDENTITY PASS`; 3D reproduction of twist / sliver / sub20 / subvol16 to the iteration (Global Constraints).

---

### Task 2: `checkpoint_dir` in `windowed_correct` — resumable at coarse / round / tiler-sweep / mop / re-seed / re-anchor granularity

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — signature (~`:662`, after `time_budget_s`): `checkpoint_dir=None, touched_out=None`; the entry block (~`:1003-1012`); marks after the coarse stage (~`:1087`), at the end of each round (after the `for box in find_windows(...)` loop, before the next iteration), after the mop (~`:1214`), after `_run_reseed()` (~`:1217`), after the re-anchor (~`:1253`), and `finish` before the final accounting (~`:1255`); `_solve_giant_schwarz` gains `on_sweep=None` (called with the sweep index after each sweep's count).
- Test: `tests/test_windowed_checkpoint.py` (new)

**Interfaces:**
- Consumes: `dvfopt.checkpoint.RunCheckpoint(checkpoint_dir, phi_in, meta, *, slab)` with `.open()`, `.finished`, `.done`, `.rows`, `.field`, `.mark(unit, slab=None, row=None)`, `.finish(out=None)`, `.dir`.
- Produces: `windowed_correct(..., checkpoint_dir=None, touched_out=None)`; `SliceReport.resumed_from: str = ''` (the last done unit at resume, `'finished'` when the mirror was final); `<dir>/touched.npy` beside `field.npy` / `state.json`; units `'coarse'`, `'round:<k>'`, `'giant:<g>:<s>'`, `'mop'`, `'reseed'`, `'reanchor'`; every row = `{'rounds': int, 'n_windows': int, 'giant_regions': int, 'mop_windows': int, 'mop_cleared': int, 'reseed_rounds_run': int, 'folds': int}`. `touched_out`: an optional bool array of the spatial shape the engine ORs its final `touched` into (Task 3 composes the no-damage accounting across bands with it).

Resume semantics (document in the docstring, verbatim): a resumed run reloads the mirrored field and `touched.npy`, restores the counters of the last row into the report, skips every stage whose unit is done, and re-enters the round loop with a fresh fold mask — so its `rep.windows` / `rep.history` cover only the post-resume work and the round loop's no-progress check restarts; the certificate and the damage accounting (against the ORIGINAL input, through the restored `touched`) are the same invariants as an uninterrupted run. `time_budget_s` counts from the resume.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_windowed_checkpoint.py
import json

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint2DBilinear, SimplexConstraint3D
from dvfopt.core.windowed import min_field, windowed_correct
from dvfopt.objectives import L2Objective
from tests.conftest import planted_fold, planted_fold_3d

pytest.importorskip('osqp')


def _run(phi, c, **kw):
    return windowed_correct(
        phi, 'isqp', constraint=c, objective=L2Objective(), threshold=0.01, verbose=0, **kw
    )


def test_checkpoint_writes_units_and_finished_run_reloads(tmp_path):
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out1, rep1 = _run(phi, c, checkpoint_dir=tmp_path)
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['stage'] == 'done'
    assert any(str(u).startswith('round:') for u in state['done'])
    assert (tmp_path / 'touched.npy').exists()
    out2, rep2 = _run(phi, c, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(out1, out2)
    assert rep2.resumed_from == 'finished'
    assert rep2.folds_after == rep1.folds_after == 0 and rep2.damage == 0


def test_checkpoint_mismatch_is_refused(tmp_path):
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    _run(phi, c, checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match='does not match'):
        _run(phi, c, checkpoint_dir=tmp_path, max_rounds=3)


def test_resume_after_a_round_keeps_the_invariants(tmp_path):
    """Simulate an interruption after round 1: rewrite the state to 'run' with only
    'round:1' done; the mirror already holds the corrected field, so the resumed run
    re-enters the loop, finds no fold, and finishes with damage 0 against the ORIGINAL."""
    phi = planted_fold(40, 44).astype(np.float64)
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    out1, rep1 = _run(phi, c, checkpoint_dir=tmp_path)
    sp = tmp_path / 'state.json'
    state = json.loads(sp.read_text())
    state['stage'] = 'run'
    state['done'] = [u for u in state['done'] if u == 'round:1']
    sp.write_text(json.dumps(state))
    touched_out = np.zeros(phi.shape[1:], bool)
    out2, rep2 = _run(phi, c, checkpoint_dir=tmp_path, touched_out=touched_out)
    assert rep2.resumed_from == 'round:1'
    assert rep2.rounds == 1  # restored from the row; no new round ran on a fold-free field
    assert rep2.folds_after == 0 and rep2.damage == 0
    np.testing.assert_array_equal(out2, out1)
    assert touched_out.any()  # the restored touched mask reached the caller


def test_touched_out_covers_every_moved_voxel_3d():
    phi = planted_fold_3d(8, 12, 12, depth=1.4)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    touched = np.zeros(phi.shape[1:], bool)
    out, rep = _run(phi, c, touched_out=touched)
    moved = np.any(out != phi, axis=0)
    assert rep.folds_after == 0 and rep.damage == 0
    assert not (moved & ~touched).any()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_windowed_checkpoint.py -q -p no:cacheprovider`
Expected: FAIL — `TypeError: windowed_correct() got an unexpected keyword argument 'checkpoint_dir'`.

- [ ] **Step 3: Implement**

`SliceReport` gains `resumed_from: str = ''` (after `patience_fallbacks`). Signature: add `checkpoint_dir=None, touched_out=None,` after `time_budget_s=None,`. At the entry, right after `touched = np.zeros(shape, bool)` (~`:1010`):

```python
    ck = None
    done_units = set()
    if checkpoint_dir is not None:
        from dvfopt.checkpoint import RunCheckpoint

        meta = dict(
            engine='windowed',
            inner=str(inner),
            constraint=type(constraint).__name__,
            objective=type(objective).__name__,
            threshold=float(threshold),
            margin=int(margin),
            maxiter=int(maxiter),
            max_rounds=int(max_rounds),
            max_window_area=int(max_window_area),
            mop_margin=int(mop_margin),
            orientation_delta=None if orientation_delta is None else float(orientation_delta),
            orientation_rows=str(orientation_rows),
            reseed_rounds=int(reseed_rounds),
            reanchor=str(reanchor),
        )
        ck = RunCheckpoint(checkpoint_dir, phi_in, meta, slab=lambda _u: Ellipsis).open()
        tp = ck.dir / 'touched.npy'
        if ck.finished:
            phi[...] = ck.field
            touched[...] = np.load(tp)
            rep.resumed_from = 'finished'
            log_info(f'[windowed resume] finished run reloaded from {ck.dir}')
        elif ck.done:
            phi[...] = ck.field
            touched[...] = np.load(tp)
            last = ck.done[-1]
            row = ck.rows.get(str(last), {})
            rep.rounds = int(row.get('rounds', 0))
            rep.giant_regions = int(row.get('giant_regions', 0))
            rep.mop_windows = int(row.get('mop_windows', 0))
            rep.mop_cleared = int(row.get('mop_cleared', 0))
            rep.reseed_rounds_run = int(row.get('reseed_rounds_run', 0))
            rep.resumed_from = str(last)
            done_units = set(map(str, ck.done))
            log_info(f'[windowed resume] {len(ck.done)} units from {ck.dir}, last {last!r}')

    def _mark(unit):
        if ck is None:
            return
        tmp = ck.dir / 'touched.npy.tmp'
        np.save(tmp, touched)
        os.replace(tmp, ck.dir / 'touched.npy')
        ck.mark(
            unit,
            phi,
            row=dict(
                rounds=rep.rounds,
                n_windows=len(rep.windows),
                giant_regions=rep.giant_regions,
                mop_windows=rep.mop_windows,
                mop_cleared=rep.mop_cleared,
                reseed_rounds_run=rep.reseed_rounds_run,
                folds=int(pixel_fold_mask(constraint, phi, threshold).sum()),
            ),
        )

    def _skip(unit):
        return unit in done_units or rep.resumed_from == 'finished'
```

(`import os` at the module top if absent.) Then:

- coarse stage: guard the `if (coarse_to_fine and ...)` with `and not _skip('coarse')`; `_mark('coarse')` at its end (after `_fire("coarse", phi)`).
- round loop: `for _rnd in range(rep.rounds, max_rounds):` (so a resumed run does not exceed `max_rounds`); when `rep.resumed_from == 'finished'` set `budget_hit = False` and skip the loop entirely (`if _skip('rounds'): ...` is wrong — use `if rep.resumed_from != 'finished':` around the loop); after the `for box in find_windows(...)` body completes for a round (still inside the `for _rnd` loop): `_mark(f'round:{rep.rounds}')`.
- tiler: `_solve_giant_schwarz(..., on_sweep=None)`; after each sweep's `nf = _folds_in_box(...)` (both the RAS and the serial branch) call `if on_sweep is not None: on_sweep(_sweep)` BEFORE the `return`/`continue`; the round-loop call site passes `on_sweep=lambda s, g=rep.giant_regions: _mark(f'giant:{g}:{s}')`.
- mop: guard `if mop_margin > 0 and not budget_hit and not _skip('mop'):`, `_mark('mop')` after `_fire("mop", phi)`.
- reseed: `if reseed_rounds > 0 and not budget_hit and not _skip('reseed'): _run_reseed(); _mark('reseed')`.
- reanchor: guard with `and not _skip('reanchor')`, `_mark('reanchor')` after `_fire("reanchor", phi)`.
- before `jf = min_field(constraint, phi)` (~`:1255`): `if ck is not None and not ck.finished: ck.finish(phi)`; and after the damage accounting: `if touched_out is not None: touched_out |= touched`.

Docstring: add a "Checkpoint / resume" paragraph with the semantics above verbatim.

- [ ] **Step 4: Run the tests and the windowed suites**

Run: `pytest tests/test_windowed_checkpoint.py tests/test_windowed_phase4.py tests/test_windowed_ras.py tests/test_windowed_3d_stages.py tests/test_windowed_3d.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Lint, format, mypy, commit**

```bash
ruff check dvfopt tests && ruff format --check dvfopt tests && mypy
git add dvfopt/core/windowed/_common.py tests/test_windowed_checkpoint.py
git commit -m "windowed: checkpoint_dir on windowed_correct (RunCheckpoint mirror after coarse / each round / each tiler sweep / mop / reseed / reanchor, touched.npy beside; resume from the mirror) + touched_out"
```

**Controller gate after Task 2:** 2D identity (the default `checkpoint_dir=None` path must be untouched) → `IDENTITY PASS`.

---

### Task 3: `windowed_correct_banded` — z-banded full-volume driver, slab-only workers, per-band checkpoint, serial seam pass

**Files:**
- Create: `dvfopt/core/windowed/_banded.py`
- Modify: `dvfopt/core/windowed/__init__.py` (export `windowed_correct_banded`, `BandedReport`)
- Test: `tests/test_windowed_banded.py` (new)

**Interfaces:**
- Consumes: `windowed_correct(phi, inner, *, constraint, objective, threshold, checkpoint_dir, touched_out, verbose, **engine_kw)` (Task 2), `min_field`, `pixel_fold_mask`, `dvfopt.core._pool.pool_map(worker, args, n_workers)`, `pin_worker_threads`, `RunCheckpoint`, `dvfopt.jacobian.tetrahedron_sign.best_diagonal_min_volume`.
- Produces:

```python
@dataclass
class BandedReport:
    bands: int = 0
    band: int = 0
    overlap: int = 0
    band_walls: list = field(default_factory=list)   # seconds per band, band order
    band_folds_after: list = field(default_factory=list)  # each band's own folds_after (slab-local)
    folds_before: int = 0
    folds_after: int = 0
    folds_after_zero: int = 0
    best_diag_floor_after: int = -1
    best_diag_floor_after_zero: int = -1
    min_before: float = 0.0
    min_after: float = 0.0
    damage: int = 0            # against the ORIGINAL input, touched = union over bands + seam pass
    n_windows: int = 0         # sum over bands + the seam pass
    seam_windows: int = 0      # the seam pass's windows (the cost of banding)
    seam_folds_before: int = 0 # folds on the composed field before the seam pass
    time_s: float = 0.0
    resumed_from: str = ''
    seam: SliceReport | None = None

def windowed_correct_banded(
    phi_in, inner='isqp', *, constraint, threshold, objective=None,
    band=24, overlap=8, n_workers=1, checkpoint_dir=None, verbose=1, **engine_kw,
) -> tuple[np.ndarray, BandedReport]
```

Design (state it in the module docstring): bands are the z cores `[k*band, min(D, (k+1)*band))`; each worker receives the slab `[core_lo - overlap, core_hi + overlap)` of ONE snapshot taken before the sweep, runs `windowed_correct` on it (`giant_workers=0`, `checkpoint_dir=None`, `verbose=0`, `touched_out=` a slab mask) and returns only its core planes and the core's touched planes; the parent commits the cores (restricted additive Schwarz across bands — a cluster crossing a core boundary is solved twice, truncated, and the seam it leaves is the seam pass's work); after all bands, ONE serial `windowed_correct` on the composed full field (`checkpoint_dir=<dir>/seam` when checkpointing, `touched_out=touched`) repairs the seams — it opens windows only where folds remain, so its cost is the banding's cost, reported as `seam_windows`. Memory: the parent holds `phi` + `j0` + `orig_fold` + `touched`; a worker holds its slab plus that slab's engine temporaries. `overlap` must be ≥ `margin + ring` of the family (assert), the default 8 = the phase-2 ring/margin sum plus slack. Requires a 3D constraint (`constraint.dim == 3`; raise `ValueError` otherwise — 2D slices are `DVFoptConfig`'s per-slice sweep).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_windowed_banded.py
import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.windowed import min_field, windowed_correct_banded
from dvfopt.objectives import L2Objective
from tests.conftest import planted_fold_3d

pytest.importorskip('osqp')


def _two_clusters(D=30, H=12, W=12):
    """Two planted fold clusters: one inside band 0, one straddling the z=15 core boundary."""
    a = planted_fold_3d(D, H, W, depth=1.4)  # planted near the middle by conftest
    # shift a copy of the planted patch so one cluster sits at z≈6 and one at z≈14..16
    phi = np.zeros_like(a)
    phi[:, 2:10] = a[:, D // 2 - 4 : D // 2 + 4]
    phi[:, 11:19] = a[:, D // 2 - 4 : D // 2 + 4]
    return phi


def _folds(phi, thr=0.01):
    return int((min_field(SimplexConstraint3D(shape=phi.shape[1:]), phi) < thr).sum())


@pytest.mark.parametrize('n_workers', [1, 2])
def test_banded_certifies_and_moves_locally(n_workers):
    phi = _two_clusters()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert _folds(phi) > 0
    out, rep = windowed_correct_banded(
        phi, 'isqp', constraint=c, threshold=0.01, objective=L2Objective(),
        band=15, overlap=4, n_workers=n_workers, verbose=0,
    )
    assert rep.bands == 2 and len(rep.band_walls) == 2
    assert rep.folds_after == 0 and rep.folds_after_zero == 0
    assert rep.best_diag_floor_after == 0
    assert rep.damage == 0
    moved = np.any(out != phi, axis=0)
    assert 0 < moved.mean() < 0.5  # local: the two clusters' neighbourhoods, not the volume
    assert rep.n_windows >= rep.seam_windows >= 0


def test_banded_rejects_2d_and_thin_overlap():
    from dvfopt.constraints import SimplexConstraint2DBilinear
    from tests.conftest import planted_fold

    phi2 = planted_fold(20, 20).astype(np.float64)
    with pytest.raises(ValueError, match='3D'):
        windowed_correct_banded(
            phi2, 'isqp', constraint=SimplexConstraint2DBilinear(shape=(20, 20)), threshold=0.01
        )
    phi = _two_clusters()
    with pytest.raises(ValueError, match='overlap'):
        windowed_correct_banded(
            phi, 'isqp', constraint=SimplexConstraint3D(shape=phi.shape[1:]),
            threshold=0.01, band=15, overlap=1,
        )


def test_banded_checkpoint_reloads_a_finished_run(tmp_path):
    phi = _two_clusters()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out1, rep1 = windowed_correct_banded(
        phi, 'isqp', constraint=c, threshold=0.01, band=15, overlap=4,
        checkpoint_dir=tmp_path, verbose=0,
    )
    assert (tmp_path / 'state.json').exists() and (tmp_path / 'seam' / 'state.json').exists()
    out2, rep2 = windowed_correct_banded(
        phi, 'isqp', constraint=c, threshold=0.01, band=15, overlap=4,
        checkpoint_dir=tmp_path, verbose=0,
    )
    np.testing.assert_array_equal(out1, out2)
    assert rep2.resumed_from == 'finished'
    assert rep2.damage == 0 and rep2.folds_after == 0
```

> If `planted_fold_3d`'s patch is not centred as assumed, the implementer adapts `_two_clusters` so that one cluster lies fully inside `[0, 15)` and one crosses z=15 — assert it in the test with `min_field` (fold voxels present in both `z < 15` and `z >= 15` for the second) rather than trusting the construction.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_windowed_banded.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'windowed_correct_banded'`.

- [ ] **Step 3: Implement `_banded.py`**

```python
"""Banded full-volume driver for the windowed engine (phase 4 of the 3D port).

<the design paragraph from the Interfaces block, verbatim>
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from dvfopt._logging import log_info
from dvfopt.core.windowed._common import SliceReport, windowed_correct
from dvfopt.core.windowed._locality import _locality_of, min_field, pixel_fold_mask


@dataclass
class BandedReport:
    ...  # exactly the Interfaces block


def _band_task(args):
    """Pool worker: the engine on ONE z-slab; returns the core planes + the core's touched."""
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()
    slab, lo_in_slab, hi_in_slab, inner, ctype, threshold, objective, kw = args
    c = ctype(shape=slab.shape[1:])
    touched = np.zeros(slab.shape[1:], bool)
    t = time.perf_counter()
    out, rep = windowed_correct(
        slab, inner, constraint=c, objective=objective, threshold=threshold,
        touched_out=touched, checkpoint_dir=None, giant_workers=0, verbose=0, **kw,
    )
    return (
        out[:, lo_in_slab:hi_in_slab].copy(),
        touched[lo_in_slab:hi_in_slab].copy(),
        int(rep.n_windows),
        int(rep.folds_after),
        time.perf_counter() - t,
    )


def windowed_correct_banded(
    phi_in, inner='isqp', *, constraint, threshold, objective=None,
    band=24, overlap=8, n_workers=1, checkpoint_dir=None, verbose=1, **engine_kw,
):
    if getattr(constraint, 'dim', 2) != 3:
        raise ValueError('windowed_correct_banded needs a 3D constraint (SimplexConstraint3D)')
    loc = _locality_of(constraint)
    need = int(engine_kw.get('margin', 3)) + loc.ring
    if overlap < need:
        raise ValueError(f'overlap {overlap} < margin + ring = {need}')
    from dvfopt.objectives import L2Objective

    objective = L2Objective() if objective is None else objective
    t0 = time.perf_counter()
    phi = np.array(phi_in, dtype=np.float64, copy=True)
    D = phi.shape[1]
    j0 = min_field(constraint, phi)
    orig_fold = j0 < threshold
    rep = BandedReport(band=int(band), overlap=int(overlap), folds_before=int(orig_fold.sum()),
                       min_before=float(j0.min()))
    del j0
    touched = np.zeros(phi.shape[1:], bool)
    cores = [(z, min(D, z + band)) for z in range(0, D, band)]
    rep.bands = len(cores)

    ck = None
    if checkpoint_dir is not None:
        from dvfopt.checkpoint import RunCheckpoint

        meta = dict(engine='windowed_banded', band=int(band), overlap=int(overlap),
                    threshold=float(threshold), inner=str(inner),
                    constraint=type(constraint).__name__, objective=type(objective).__name__)
        ck = RunCheckpoint(
            checkpoint_dir, phi_in, meta,
            slab=lambda u: (slice(None), slice(*cores[int(str(u).rpartition(':')[2])])),
        ).open()
        tp = ck.dir / 'touched.npy'
        if ck.finished:
            rep.resumed_from = 'finished'
            phi[...] = ck.field
            touched[...] = np.load(tp)
        elif ck.done:
            ck.restore_into(phi)
            touched[...] = np.load(tp)
            rep.resumed_from = str(ck.done[-1])
            for u in ck.done:
                r = ck.rows.get(str(u), {})
                rep.band_walls.append(float(r.get('wall_s', 0.0)))
                rep.band_folds_after.append(int(r.get('folds_after', -1)))
                rep.n_windows += int(r.get('n_windows', 0))
        if verbose and rep.resumed_from:
            log_info(f'[banded resume] {rep.resumed_from} from {ck.dir}')

    def _save_touched():
        if ck is None:
            return
        tmp = ck.dir / 'touched.npy.tmp'
        np.save(tmp, touched)
        os.replace(tmp, ck.dir / 'touched.npy')

    if rep.resumed_from != 'finished':
        kw = dict(engine_kw)
        todo = [k for k in range(len(cores)) if ck is None or not ck.is_done(f'band:{k}')]
        args = []
        for k in todo:
            lo, hi = cores[k]
            s0, s1 = max(0, lo - overlap), min(D, hi + overlap)
            args.append((phi[:, s0:s1].copy(), lo - s0, hi - s0, inner, type(constraint),
                         threshold, objective, kw))
        if n_workers > 1 and len(args) > 1:
            from dvfopt.core._pool import pool_map

            results = pool_map(_band_task, args, n_workers)
        else:
            results = [_band_task(a) for a in args]
        for k, (vals, tch, nw, fa, wall) in zip(todo, results):
            lo, hi = cores[k]
            phi[:, lo:hi] = vals
            touched[lo:hi] |= tch
            rep.band_walls.append(wall)
            rep.band_folds_after.append(fa)
            rep.n_windows += nw
            if ck is not None:
                _save_touched()
                ck.mark(f'band:{k}', vals, row=dict(wall_s=wall, folds_after=fa, n_windows=nw))
            if verbose:
                log_info(f'[banded] band {k + 1}/{len(cores)} z[{lo},{hi}) windows {nw} '
                         f'folds_after {fa} {wall:.0f}s')
        rep.seam_folds_before = int(pixel_fold_mask(constraint, phi, threshold).sum())
        seam_dir = None if ck is None else ck.dir / 'seam'
        out, srep = windowed_correct(
            phi, inner, constraint=constraint, objective=objective, threshold=threshold,
            checkpoint_dir=seam_dir, touched_out=touched, verbose=verbose, **engine_kw,
        )
        phi[...] = out
        rep.seam = srep
        rep.seam_windows = int(srep.n_windows)
        rep.n_windows += rep.seam_windows
        if ck is not None:
            _save_touched()
            ck.finish(phi)

    jf = min_field(constraint, phi)
    after = jf < threshold
    rep.folds_after = int(after.sum())
    rep.folds_after_zero = int((jf <= 0).sum())
    rep.min_after = float(jf.min())
    rep.damage = int((after & ~orig_fold & ~touched).sum())
    from dvfopt.jacobian.tetrahedron_sign import best_diagonal_min_volume

    best_min, _ = best_diagonal_min_volume(phi)
    rep.best_diag_floor_after = int((best_min <= threshold).sum())
    rep.best_diag_floor_after_zero = int((best_min <= 0.0).sum())
    rep.time_s = time.perf_counter() - t0
    return phi, rep


__all__ = ['BandedReport', 'windowed_correct_banded']
```

> Note for the implementer: `engine_kw` must not contain `giant_workers` / `checkpoint_dir` / `touched_out` / `verbose` (the band task sets them); pop and ignore `giant_workers` with a DEBUG log if present, raise on the others. The seam pass reuses `engine_kw` unchanged (it may use `giant_workers` from the caller if you pass it separately — keep it simple: the seam pass is serial in this PR).

Export in `dvfopt/core/windowed/__init__.py`: `from ._banded import BandedReport, windowed_correct_banded` and both names in `__all__`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_windowed_banded.py -q -p no:cacheprovider` (the `n_workers=2` case spawns the shared pool: ~30-60 s on Windows).
Expected: PASS.

- [ ] **Step 5: Lint, format, mypy, commit**

```bash
ruff check dvfopt tests && ruff format --check dvfopt tests && mypy
git add dvfopt/core/windowed/_banded.py dvfopt/core/windowed/__init__.py tests/test_windowed_banded.py
git commit -m "windowed: windowed_correct_banded — z-banded full-volume driver (slab-only workers, restricted additive Schwarz across bands, per-band checkpoint, serial seam pass)"
```

---

### Task 4: The full-volume benchmark driver + the ds2 cohort volumes

**Files:**
- Create: `benchmarks/windowed_3d_full.py`
- Modify: `data/dvfs/README.md` (a `cohort_ds2/` row), `.gitignore` only if `data/dvfs/cohort_ds2/*.npy` is not already covered by the data payload rules (check: `git check-ignore -v data/dvfs/cohort_ds2/x.npy`).

**Interfaces:**
- Consumes: `windowed_correct_banded`, `windowed_correct`, `dvfopt.core.wallbreakers._multiscale_3d._downsample_2x(phi)` (exact 2x block mean, displacements × 0.5), `benchmarks.benchmark_utils.load_cohort_field(brain, variant)`, `six_tet_min_volume_3d`, `n_neg_best_diagonal`, `dvfopt.io.fields.save_dvf`.
- Produces: CLI `python benchmarks/windowed_3d_full.py --build-ds2 B0039 [--variant laplacian_exterior]` → `data/dvfs/cohort_ds2/<brain>_<variant>_ds2.npy` `(3, 264, 160, 228)` float64 + prints its fold count; `--cut-slab z0 z1 [--src data/dvfs/b0039/b0039_laplacian_deformation_field.npy]` → `data/dvfs/crops_3d/slab_<z0>_<z1>.npy`; `--run PATH --tag TAG [--band 24 --overlap 8 --n-workers 4 --checkpoint DIR --serial]` → `benchmarks/output/windowed_3d/full_<tag>.json` with keys `case, shape, n_voxels, mode ('banded'|'serial'), band, overlap, n_workers, folds_in, folds_in_zero, floor_in, min_in, folds_out, folds_out_zero, floor_out, min_out, new_folds, damage, moved_frac, l1_move, l2_move, n_windows, seam_windows, seam_folds_before, bands, band_walls, wall_s, resumed_from, commit` and the corrected field at `data/dvfs/results/windowed_3d_full/<tag>.npy` (`save_dvf`); `--table` → `full.md`.

- [ ] **Step 1: Write the driver** (no unit test — a benchmark script; its smoke is the first Task-5 row). Structure: copy `benchmarks/windowed_3d_sweep.py`'s `argparse` / record / `--table` skeleton (lines 22-56 and its `run` / `table` functions) — same `OUT` directory, same `THR` from `windowed_3d_gate`; load inputs with `np.load(path, mmap_mode='r')` and `np.asarray(..., dtype=np.float64)` only for the run; `moved_frac` / `l1_move` / `l2_move` against the input; `new_folds = int(((six_tet_min_volume_3d(out) < THR) & ~(six_tet_min_volume_3d(phi) < THR)).sum())`; `commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])`. `--build-ds2` loads the cohort field with `load_cohort_field`, applies `_downsample_2x`, saves float64, prints `folds`, `folds_zero`, `floor` for the ds2 volume. `--serial` runs `windowed_correct` (with `checkpoint_dir`) instead of the banded driver — the A/B on the slab.

- [ ] **Step 2: Smoke the driver on the synthetic path** — `python -c "import numpy as np; from tests.conftest import planted_fold_3d; np.save('benchmarks/output/windowed_3d/tiny_full.npy', planted_fold_3d(30, 12, 12, depth=1.4))"` then `python benchmarks/windowed_3d_full.py --run benchmarks/output/windowed_3d/tiny_full.npy --tag smoke --band 15 --overlap 4 --n-workers 1` → a record with `folds_out 0`, `damage 0`; delete the two files after.

- [ ] **Step 3: README row + lint + commit**

```bash
ruff check benchmarks && ruff format --check benchmarks
git add benchmarks/windowed_3d_full.py data/dvfs/README.md
git commit -m "benchmarks: windowed_3d_full — banded / serial full-volume runs with the certificate block, --build-ds2 (exact 2x, displacements halved), --cut-slab"
```

---

### Task 5 (controller): the measurement ladder, pre-registered

**Pre-register in the ledger before the first run.** This PR changes no default: the ruling is only whether the volumes certify and at what cost. Rows (each a `full_<tag>.json`; chains from a detached snapshot worktree; idle box):

1. **Slab A/B** — `--cut-slab 240 288` of the raw B0039 field (`(3, 48, 320, 456)`, 7.0 M voxels, the z=240-288 region the 2D campaign knows): `--serial` (one engine call, `--checkpoint`) vs `--band 24 --overlap 8 --n-workers 1` vs `--n-workers 2`. Reported: certificate (0 / 0 / floor 0), damage 0, `seam_windows` and `seam_folds_before` (the banding cost), wall serial vs banded-1 (the seam overhead) vs banded-2 (the speedup), L2/L1 move. Ruling template: banding is viable if the banded runs certify with damage 0 and the seam pass is < 25 % of the windows; the 2-worker wall is the phase-4 throughput number.
2. **ds2 B0039 exterior** — `--build-ds2 B0039`, then `--band 24 --overlap 8 --n-workers 4 --checkpoint` (9.6 M voxels): the first full-volume certificate. Record the ds2 fold count first (Part XX's multiscale numbers are for a different stacking: ÷2 gave 19 500 folds there).
3. **Full-res B0039 exterior** — the capstone: `--run data/dvfs/cohort/B0039/laplacian_exterior/laplacian_deformation_field.npz` (the driver accepts `.npz` key `arr`), `--band 24 --overlap 8 --n-workers 4 --checkpoint benchmarks/output/windowed_3d/ck_full_b0039`, in the background with a monitor on its log, resumable across sessions. Expected days (Part XX: 58 h for the research path); the record's `band_walls` give the projection after the first bands — write the projection into the ledger and decide whether the run continues in this PR or is handed to the next session (the PR does not wait for it: it lands the driver, the checkpoint and the slab + ds2 rows; the full-res row is appended to the CHANGELOG when it lands).

**Ledger each row with:** the record path, the certificate block, damage, walls, and any deviation from the pre-registration as a labelled amendment.

---

### Task 6: Docs, CHANGELOG, PR

- [ ] Docs subagent: `CHANGELOG.md` entry (`### Added — 3D windowed engine, phase 4: banded full-volume driver + resumable checkpoints + the tiler's box-scoped counts / patch-only RAS tasks`; the measured rows as a table; the memory numbers from the survey; what is NOT in this PR: strategy/CLI plumbing for `checkpoint_dir` on the windowed path, a `giant_workers` seam pass, the full-res row if pending); `CLAUDE.md`: the "Resumable runs" paragraph gains the windowed engine (`checkpoint_dir=` on `windowed_correct` / `windowed_correct_banded`, units), the 3D bullet gains phase 4's two sentences (banded driver; the memory footprint numbers), `giant_workers` bullet: workers receive the tile patch; `ARCHITECTURE.md` only if it lists the engine's entry points; `docs/superpowers/notes/zero-folds-campaign-findings.md` a §10.x line pointing at the full-volume rows.
- [ ] Controller: full suite (`-n 4`, `QT_QPA_PLATFORM=offscreen`; check the summary line for `failed`); ruff / format / mypy; the 2D identity gate on the final head; push; PR (body: what lands, the measured table, the memory numbers, the scope statement); squash-merge on all-green; sync main; remove the worktrees; ledger to the main checkout's `.superpowers/sdd/`; memory (`todo-3d-windowed-engine-port.md` phase-4 paragraph + the MEMORY.md line).

---

## Self-review

- **Spec coverage:** "chunked full-volume driver … workers get patches only" → Task 1 (RAS tiles) + Task 3 (bands); "`checkpoint_dir` per round/tile" → Task 2 (round + tiler sweep) + Task 3 (band); "ds2 cohort volumes, then the full-res B0039" → Tasks 4-5; "Certification = 0 folds at fixed-diagonal 6-tet, the best-diagonal floor reported alongside, damage 0, L2 move" → the record keys in Task 4 and `BandedReport`. The spec's "one field copy per process" is met for the parent (`phi`; `j0` is freed after `orig_fold` in the banded driver — the seam pass's own `j0` is the engine's) and for workers (slab only).
- **Placeholders:** Task 1 Step 3 carries one deliberate instruction to the implementer (how `_InnerOpts` is built and the `fy0/fx0` semantics on 3D) because the plan cannot quote lines it did not read; both are verification steps with the decision rule stated, not gaps. Task 4 has no unit test by design (benchmark script; its smoke is Step 2).
- **Type consistency:** `_ras_tile_task` args order is identical in Task 1's test and implementation; `windowed_correct`'s new kwargs `checkpoint_dir`, `touched_out` are the names Task 3 passes; `SliceReport.resumed_from` / `BandedReport.resumed_from` are both `str`; `BandedReport.seam` holds a `SliceReport`.
