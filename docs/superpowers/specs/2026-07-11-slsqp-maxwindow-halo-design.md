# 3D SLSQP max-window halo constraints + Minors sweep — Design

**Date:** 2026-07-11
**Status:** Approved (approach chosen by user: halo no-damage constraints)
**Outcome:** Lift the `scipy<1.16` pin; the 3D windowed SLSQP solver makes progress under any SLSQP implementation. Plus a sweep of logged cosmetic Minors from the feature-complete pass.

---

## Workstream A — patch-based max-window sub-problem

### Problem

`iterative_3d`'s windowed SLSQP livelocks under scipy ≥ 1.16 (Fortran→C SLSQP port).
Evidence chain (investigation 2026-07-10, `.superpowers/sdd/progress.md`):

- At `window_reached_max=True`, `_build_constraints_3d` drops the frozen-rim equality
  constraints and covers all window voxels with the Jdet constraint
  ([dvfopt/core/slsqp/constraints3d.py](../../../dvfopt/core/slsqp/constraints3d.py)).
- The accept/rollback guard in `_serial_fix_voxel`
  ([dvfopt/core/solver3d.py:353-403](../../../dvfopt/core/solver3d.py)) measures the
  **window + 1-voxel border** (the check region) on the true full-field Jacobian.
- The sub-problem never constrains the border. Rim voxels are free at max window; their
  motion changes border Jdet through the central-difference stencil. scipy 1.15's
  Fortran optimum happened to be border-gentle; 1.16's C optimum is not → every solve
  is feasible in-window yet rejected by paste-back → solve/reject livelock
  (instrumented: 600/600 identical rejects under 1.18 vs 1 accepted move under 1.15.3).
- Second, pre-existing misalignment: in-window constraint rows evaluate `np.gradient`
  on the window **as its own little volume** (one-sided stencils at window edges), while
  the accept check uses full-field central differences. Feasible-in-window ≠
  acceptable-at-paste-back even for window rows.

Root design flaw: **the max-window sub-problem's feasible set does not match the accept
criterion.** Fix the sub-problem, not the accept check.

### Approach (approved)

When — and only when — `window_reached_max=True`, evaluate constraints on a **context
patch** (window grown by 2 voxels per side, clamped to the volume) with the decision
variables still window-only, and constrain Jdet over **window ∪ halo** with per-row
lower bounds:

| Row set | Voxels | Lower bound |
|---|---|---|
| window rows | the window | `threshold` (as today) |
| halo rows | check-region border = (window ± 1 ∩ volume) \ window | `min(threshold, current Jdet at that voxel)` |

Healthy border voxels must stay ≥ threshold (no new negatives can appear); already-bad
border voxels must not get worse (region min cannot degrade). `x0` satisfies every halo
row by construction (equality at worst), so SLSQP always starts halo-feasible.

**Geometric exactness.** Halo voxels sit ≥ 1 voxel inside the patch, so their
`np.gradient` stencils on the patch are the same central differences the accept check
uses on the full field; window-rim voxels sit ≥ 2 inside, same argument. Where the patch
is clamped by the volume edge, the patch edge coincides with the volume edge, so
one-sided stencils also match the full field exactly. Therefore **every constraint row
equals the exact quantity the accept check measures** — the sub-problem's feasible set
now equals the accept criterion, and this also fixes the pre-existing window-edge
stencil mismatch at max window.

**Why fold-fixing solves are now always accepted.** A successful solve has all window
rows ≥ threshold (KKT slop ~1e-6 ≪ `err_tol` = 1e-5, so no window voxel counts as
negative) and all halo rows no worse than current → new region negative count
`n_new` = (pre-existing border negatives) < `n_old` whenever the window contained folds
(it always does — that is why we are solving). The lexicographic accept
(`n_new < n_old` → accept) fires on every successful solve. Only zero-progress solves
can still be rejected, and rejecting those loses nothing. The rollback guard stays
untouched as a pure safety net.

**Failure path unchanged.** If no window fix exists without border damage, SLSQP
reports failure → rollback → existing stall/growth handling. Same as today's behavior,
minus the wasted accept-then-reject churn on solvable cases.

**Non-max path untouched.** With frozen rims, halo Jdet cannot change (its stencil
reads only rim + outside values, all frozen), so halo rows would be redundant. Zero
change to that code path.

### Components

**1. New builder in [dvfopt/core/slsqp/constraints3d.py](../../../dvfopt/core/slsqp/constraints3d.py):**

```python
def _build_constraints_3d_maxwindow(patch_flat, patch_size, win_start, win_size, threshold):
    """Constraints for a max-window solve: Jdet over window ∪ halo on the
    context patch, per-row lower bounds, variables = window voxels only.
    Returns [NonlinearConstraint]."""
```

- `patch_flat`: packed `[dx, dy, dz]` over the patch (same packing as `phi_sub_flat`).
- `win_start = (oz, oy, ox)`: window origin in patch coordinates; `win_size = (sz, sy, sx)`.
- Internals (all precomputed once in the builder):
  - `Np = pz*py*px`; window linear indices in patch order `patch_lin`
    (via `np.ravel_multi_index`); variable column indices
    `cols = concat([patch_lin, patch_lin + Np, patch_lin + 2*Np])` — matches the
    `[dx, dy, dz]` packing of the window decision vector.
  - `embed(x)`: copy of `patch_flat` with `vec[cols] = x`.
  - `region_mask`: boolean over patch = window dilated by 1, clamped to patch bounds
    (patch clamping makes this identical to volume clamping); `rows = flatnonzero(region_mask.ravel())`.
  - `jdet0 = _numpy_jdet_3d(patch at x0)`; `lb` vector aligned to `rows`:
    `threshold` on window rows, `np.minimum(threshold, jdet0)` on halo rows.
  - `fun(x) = _numpy_jdet_3d(embed(x)).ravel()[rows]`.
  - `jac(x) = jdet_constraint_jacobian_3d(embed(x), patch_size)[rows][:, cols]` —
    reuses the existing analytic Jacobian on the patch, sliced to constrained rows and
    window-variable columns. No new derivative math.
- Returns `[NonlinearConstraint(fun, lb, np.inf, jac=jac)]` (vector lower bound; no
  equality constraints at max window, unchanged).

**2. `_optimize_single_window_3d`** ([dvfopt/core/solver3d.py:138](../../../dvfopt/core/solver3d.py))
gains `patch_ctx=None` (tuple `(patch_flat, patch_size, win_start)`). When
`window_reached_max` and `patch_ctx` is provided → new builder; otherwise existing
builder (with an assertion that max-window calls supply the patch). Objective, x0,
return contract unchanged (window vars only; `_apply_result_3d` untouched).

**3. Caller wiring in `_serial_fix_voxel`** ([dvfopt/core/solver3d.py:335](../../../dvfopt/core/solver3d.py)):
when `window_reached_max`, extract the patch right before the solve —
`p{z,y,x}0 = max(win_lo - 2, 0)`, `p{z,y,x}1 = min(win_hi + 2, dim)` — flatten with the
standard `[phi[2], phi[1], phi[0]]` channel order, pass `patch_ctx`. Cost: one small
copy per max-window solve only.

**4. Unpin scipy** in [pyproject.toml](../../../pyproject.toml): `"scipy<1.16"` → `"scipy"`,
delete the TEMP comment block. Update the tripwire test's comment (it references the pin).

**5. 2D follow-up marker:** one comment in the 2D `_build_constraints` max-window branch
noting the same latent structure and pointing at `_build_constraints_3d_maxwindow` as
the template. No 2D behavior change (no observed failure; suite green under 1.18 in 2D).

### Performance

Patch adds +4 per dimension: a 3³ window solves against a 7³ patch (343 voxels, 125
constraint rows vs 27 today, 81 variables unchanged). The analytic-Jacobian intermediate
is (343 × 1029) sparse before slicing — negligible. Applies only to max-window solves.

### Testing (Workstream A)

New `tests/test_slsqp_maxwindow_halo.py`:

1. **Builder units:** lb vector correct (healthy halo → threshold, bad halo → its
   current Jdet); `fun(x0) ≥ lb` elementwise (x0 feasibility); analytic `jac` vs
   finite differences on a random small patch; volume-corner clamping (halo partially
   absent, shapes/rows correct).
2. **Border no-damage property:** a max-window solve on a field with a healthy border
   produces no new negatives in the check region after paste-back.
3. **Tripwire** (`tests/test_slsqp_review_fixes.py::TestFrozenEdgeReleaseAtMaxWindow3D::test_fold_larger_than_max_window_makes_progress`)
   stays as-is and must pass.

Validation matrix (local, via `pip install scipy==X` swap; the venv ends on latest
scipy, matching the unpinned dependency):

| scipy | run |
|---|---|
| 1.15.3 (current pin floor) | full suite |
| ≥ 1.18 (latest) | full suite, tripwire explicitly |

CI then enforces both forever: test.yml py3.10 resolves scipy 1.15.x (1.16 dropped 3.10),
py3.11/3.12 + ci.yml resolve latest.

### Non-goals

- 2D max-window patch constraints (documented follow-up only).
- Any change to non-max windows, the accept/rollback logic, `_apply_result_3d`,
  objective, or the parallel/tet-based 3D paths (single call site confirmed:
  `_optimize_single_window_3d` is used only by `_serial_fix_voxel`).
- No scipy version floor bump.

---

## Workstream B — logged cosmetic Minors sweep

Source: final-review Minor notes in `.superpowers/sdd/progress.md` (Tasks 2-10) plus
carried papercuts. Two batches:

### B1 — GUI worker-lifecycle & guards (`dvfopt_gui/app.py`, `worker.py`)

1. `closeEvent` doesn't wait on `_load_worker` → cancel + wait with terminate-on-timeout
   fallback (mirror the overview-worker teardown pattern).
2. File→Load not disabled while a load is in flight → disable at dispatch, re-enable in
   `_on_load_finished` (both success and failure paths).
3. `_latest` not reset in `_on_finished` + thr-spin repaint is a no-op right after a run
   (worker ref not cleared) → clear both so the threshold spinbox repaints immediately
   post-run.
4. Redo stack not byte-budgeted → apply the same byte budget used for undo
   (`UNDO_MAX_BYTES` accounting on push to redo).
5. `_restart_overview` ignores `wait()`'s return → check + terminate fallback (mirror
   closeEvent pattern).
6. Bare `except Exception` around the auto-strategy `make_strategy` fallback → narrow to
   the constructor/registry errors actually expected, log the fallback.
7. 3D-idle inspector falls back to the 2D readout → show a proper 3D idle readout
   (slice metrics from the cached 3D metrics at the current z).

### B2 — polish, docs, config

8. Misleading thr-spin comment (Task 4 note) → correct it.
9. `persistence.py` docstring: document `(N, 3, D, H, W)` `history_phi` and the absence
   of `dim` on `LoadedRun`.
10. Saved `threshold=0.0` silently restores to default (`if thr:` guard) → `is not None`.
11. `_CHOICE_FIELDS` keyed by bare field name → keep bare keys but add an import-time
    uniqueness assertion (cheapest correct fix; no algo currently collides).
12. Params restore doesn't validate inner override values → reject non-finite/wrong-type
    values on restore.
13. `argwhere(...)[0]` first-index allocation in the load path → `argmax`-style
    first-hit (one-liner).
14. Automated regression test for the overview stale-chunk race (sender-guard behavior,
    currently only hand-verified).
15. Pin ruff in the `dev` extra (`ruff==0.15.21`) so local and CI can't skew (the
    0.15.15→0.15.21 formatting drift bit once already).

Dropped (deliberate): ci.yml `timeout-minutes` bump (runtime is normal now that the
livelock is fixed; board is green), "auto stats recomputed 3×/run" (inherent to design,
noted in code), commit-01685a0 reword (history rewrite not warranted).

### Testing (Workstream B)

GUI suite additions per item where behavior changes (load-gating, redo budget, thr-spin
post-run repaint, choice-field assertion, restore validation, stale-chunk race test);
full GUI suite + lint gates green.

---

## Acceptance criteria

1. Tripwire test passes under scipy 1.15.3 **and** ≥ 1.18 locally; full suite green under both.
2. `pyproject.toml` no longer pins `scipy<1.16`.
3. New halo-builder unit tests + border no-damage property test pass.
4. All B1/B2 items implemented with tests where behavior changed; GUI suite + ruff gates green.
5. CI fully green on both workflows after merge.
