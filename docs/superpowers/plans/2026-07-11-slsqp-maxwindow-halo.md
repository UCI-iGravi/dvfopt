# 3D Max-Window Halo Constraints + Minors Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 3D windowed SLSQP solver's max-window sub-problem robust to any SLSQP implementation (lift the `scipy<1.16` pin), and sweep the logged cosmetic Minors from the feature-complete pass.

**Architecture:** At `window_reached_max`, constraints are evaluated on a context patch (window + 2 voxels/side, clamped to the volume) with decision variables still window-only. One `NonlinearConstraint` with per-row lower bounds covers window ∪ halo, making the sub-problem's feasible set equal the paste-back accept criterion — any successful solve is accepted by construction. Non-max path untouched. Spec: `docs/superpowers/specs/2026-07-11-slsqp-maxwindow-halo-design.md`.

**Tech Stack:** numpy, scipy (`minimize(method='SLSQP')`, `NonlinearConstraint` with vector lb, sparse Jacobians), pytest; PyQt5/pyqtgraph for the GUI tasks.

## Global Constraints

- Data conventions: 3D fields `(3, D, H, W)` with channels `[dz, dy, dx]`; flat window/patch vectors packed `[dx_flat, dy_flat, dz_flat]` (C-order per channel). Jdet threshold default `0.01`, `err_tol` `1e-5`.
- After the scipy unpin (Task 3), `pyproject.toml` must list plain `"scipy"` — no version bound in either direction.
- Lint gates: `python -m ruff check dvfopt dvfopt_gui tests benchmarks` and `python -m ruff format --check dvfopt dvfopt_gui tests benchmarks` must pass after every task.
- Run commands from the repo root `c:\Users\Andy\Documents\GitHub\UCI-iGravi\deformation-field-processing` (Windows, PowerShell). Use `python -m pytest -q ...`.
- Stage explicit paths in every commit (`git add <paths>` — never `git add -A`).
- GUI tests run headless: existing `tests/test_gui_*.py` modules already handle `QT_QPA_PLATFORM=offscreen` via their own boilerplate — copy the boilerplate of the module you extend.

---

### Task 1: Max-window patch constraint builder (`_build_constraints_3d_maxwindow`)

**Files:**
- Modify: `dvfopt/core/slsqp/constraints3d.py` (append new function after `_build_constraints_3d`)
- Test: `tests/test_slsqp_maxwindow_halo.py` (create)

**Interfaces:**
- Consumes: `dvfopt.jacobian.numpy_jdet._numpy_jdet_3d(dz, dy, dx)` (existing), `dvfopt.core.slsqp.gradients3d.jdet_constraint_jacobian_3d(phi_flat, subvolume_size)` (existing; returns sparse `(N, 3N)` CSR), `dvfopt._defaults._unpack_size_3d` (existing).
- Produces: `_build_constraints_3d_maxwindow(patch_flat, patch_size, win_start, win_size, threshold) -> list[NonlinearConstraint]` — exactly one `NonlinearConstraint` whose `fun` maps the window decision vector (length `3*sz*sy*sx`, packed `[dx, dy, dz]`) to Jdet values over the region (window dilated by 1, clamped to patch), with vector lower bound `lb` (threshold on window rows, `min(threshold, jdet_at_x0)` on halo rows) and sparse analytic `jac`. Task 2 calls this from `_optimize_single_window_3d`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slsqp_maxwindow_halo.py`:

```python
"""Tests for the max-window patch constraint builder (halo no-damage rows).

The builder makes the max-window sub-problem's feasible set equal the
solver's paste-back accept criterion: Jdet is constrained over the window
AND the 1-voxel halo ring around it, evaluated on a context patch with
the same stencils the full-field accept check uses.
"""

import numpy as np
import pytest

from dvfopt.core.slsqp.constraints3d import _build_constraints_3d_maxwindow
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d, jacobian_det3D

RNG = np.random.default_rng(42)
THR = 0.01


def _pack(phi):
    """(3, D, H, W) [dz, dy, dx] -> flat [dx, dy, dz] (the solver packing)."""
    return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])


def _region_masks(patch_size, win_start, win_size):
    """Reference (window, region) boolean masks over the patch, C-order."""
    pz, py, px = patch_size
    oz, oy, ox = win_start
    sz, sy, sx = win_size
    window = np.zeros((pz, py, px), dtype=bool)
    window[oz : oz + sz, oy : oy + sy, ox : ox + sx] = True
    region = np.zeros((pz, py, px), dtype=bool)
    region[
        max(oz - 1, 0) : min(oz + sz + 1, pz),
        max(oy - 1, 0) : min(oy + sy + 1, py),
        max(ox - 1, 0) : min(ox + sx + 1, px),
    ] = True
    return window, region


def _smooth_random_phi(shape, scale=0.3):
    """Small random smooth-ish displacement field (3, D, H, W)."""
    phi = scale * (RNG.random((3, *shape)) - 0.5)
    return phi


class TestMaxWindowBuilder:
    def test_single_constraint_row_count_and_lb_healthy(self):
        # Zero field: Jdet == 1 everywhere -> every halo row is healthy,
        # so lb == THR on ALL rows (window and halo alike).
        patch_size, win_start, win_size = (7, 7, 7), (2, 2, 2), (3, 3, 3)
        phi = np.zeros((3, *patch_size))
        cons = _build_constraints_3d_maxwindow(_pack(phi), patch_size, win_start, win_size, THR)
        assert len(cons) == 1
        nlc = cons[0]
        _, region = _region_masks(patch_size, win_start, win_size)
        assert nlc.fun(np.zeros(3 * 27)).size == int(region.sum())  # 5^3 = 125
        np.testing.assert_allclose(np.asarray(nlc.lb), THR)

    def test_lb_bad_halo_keeps_current_value(self):
        # Fold sheet through the halo (outside the window): dx jump at x=1.
        patch_size, win_start, win_size = (7, 7, 7), (2, 2, 2), (3, 3, 3)
        phi = np.zeros((3, *patch_size))
        phi[2, :, :, 1] = 3.0  # halo plane x=1 folds; window spans x in [2, 5)
        x0 = _pack(phi)
        cons = _build_constraints_3d_maxwindow(x0, patch_size, win_start, win_size, THR)
        nlc = cons[0]
        window, region = _region_masks(patch_size, win_start, win_size)
        rows = np.flatnonzero(region.ravel())
        window_rows = window.ravel()[rows]
        jdet0 = _numpy_jdet_3d(phi[0], phi[1], phi[2]).ravel()[rows]
        lb = np.asarray(nlc.lb)
        # Window rows demand full threshold even where currently folded.
        np.testing.assert_allclose(lb[window_rows], THR)
        # Halo rows: healthy keep THR, folded keep their current value.
        halo = ~window_rows
        np.testing.assert_allclose(lb[halo], np.minimum(THR, jdet0[halo]))
        assert (jdet0[halo] < THR).any(), 'fixture must plant a bad halo row'

    def test_x0_feasible_on_halo_rows(self):
        patch_size, win_start, win_size = (7, 6, 8), (2, 2, 2), (3, 2, 4)
        phi = _smooth_random_phi(patch_size)
        x0_win = _pack(phi[:, 2:5, 2:4, 2:6])
        cons = _build_constraints_3d_maxwindow(
            _pack(phi), patch_size, win_start, win_size, THR
        )
        nlc = cons[0]
        window, region = _region_masks(patch_size, win_start, win_size)
        window_rows = window.ravel()[np.flatnonzero(region.ravel())]
        vals = nlc.fun(x0_win)
        lb = np.asarray(nlc.lb)
        halo = ~window_rows
        assert np.all(vals[halo] >= lb[halo] - 1e-12)

    def test_fun_equals_full_field_jdet(self):
        # KEY exactness property: constraint rows == what the accept check
        # measures on the full field, for interior windows AND clamped ones.
        vol_shape = (12, 11, 13)
        phi = _smooth_random_phi(vol_shape)
        jdet_full = jacobian_det3D(phi)
        for win_lo in [(4, 4, 4), (0, 0, 0)]:  # interior; volume-corner clamp
            sz = sy = sx = 3
            lo_z, lo_y, lo_x = win_lo
            pz0, py0, px0 = max(lo_z - 2, 0), max(lo_y - 2, 0), max(lo_x - 2, 0)
            pz1 = min(lo_z + sz + 2, vol_shape[0])
            py1 = min(lo_y + sy + 2, vol_shape[1])
            px1 = min(lo_x + sx + 2, vol_shape[2])
            patch = phi[:, pz0:pz1, py0:py1, px0:px1]
            patch_size = (pz1 - pz0, py1 - py0, px1 - px0)
            win_start = (lo_z - pz0, lo_y - py0, lo_x - px0)
            x0_win = _pack(phi[:, lo_z : lo_z + sz, lo_y : lo_y + sy, lo_x : lo_x + sx])
            cons = _build_constraints_3d_maxwindow(
                _pack(patch), patch_size, win_start, (sz, sy, sx), THR
            )
            _, region = _region_masks(patch_size, win_start, (sz, sy, sx))
            rows = np.flatnonzero(region.ravel())
            vals = cons[0].fun(x0_win)
            # Full-field oracle: region voxels in volume coordinates.
            reg_idx = np.argwhere(region) + np.array([pz0, py0, px0])
            oracle = jdet_full[reg_idx[:, 0], reg_idx[:, 1], reg_idx[:, 2]]
            np.testing.assert_allclose(
                vals, oracle, atol=1e-12,
                err_msg='patch rows must equal full-field Jdet (same stencils)',
            )

    def test_jac_matches_finite_difference(self):
        patch_size, win_start, win_size = (5, 6, 7), (1, 2, 2), (3, 3, 3)
        phi = _smooth_random_phi(patch_size, scale=0.2)
        cons = _build_constraints_3d_maxwindow(
            _pack(phi), patch_size, win_start, win_size, THR
        )
        nlc = cons[0]
        n_win = 3 * 27
        x = _pack(phi[:, 1:4, 2:5, 2:5]) + 0.01 * (RNG.random(n_win) - 0.5)
        J = nlc.jac(x)
        J = J.toarray() if hasattr(J, 'toarray') else np.asarray(J)
        eps = 1e-6
        fd = np.empty_like(J)
        for j in range(n_win):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fd[:, j] = (nlc.fun(xp) - nlc.fun(xm)) / (2 * eps)
        np.testing.assert_allclose(J, fd, atol=1e-6)

    def test_fun_does_not_mutate_patch_baseline(self):
        # embed() must not corrupt the captured patch: two calls with
        # different x from the same builder give independent results.
        patch_size, win_start, win_size = (7, 7, 7), (2, 2, 2), (3, 3, 3)
        phi = _smooth_random_phi(patch_size)
        cons = _build_constraints_3d_maxwindow(
            _pack(phi), patch_size, win_start, win_size, THR
        )
        nlc = cons[0]
        x_a = np.zeros(3 * 27)
        first = nlc.fun(x_a).copy()
        nlc.fun(RNG.random(3 * 27))  # perturbed evaluation in between
        np.testing.assert_array_equal(nlc.fun(x_a), first)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slsqp_maxwindow_halo.py -q`
Expected: collection error / `ImportError: cannot import name '_build_constraints_3d_maxwindow'`.

- [ ] **Step 3: Implement the builder**

Append to `dvfopt/core/slsqp/constraints3d.py` (after `_build_constraints_3d`; `numpy`, `NonlinearConstraint`, `_numpy_jdet_3d`, `jdet_constraint_jacobian_3d`, `_unpack_size_3d` are already imported at module top):

```python
def _build_constraints_3d_maxwindow(patch_flat, patch_size, win_start, win_size, threshold):
    """Constraints for a max-window solve: Jdet over window ∪ halo on a
    context patch, with per-row lower bounds.

    The decision vector stays window-only (``[dx, dy, dz]`` packing over
    ``win_size``); constraint evaluation embeds it into the frozen
    *patch_flat* context (window + 2 voxels per side, clamped to the
    volume by the caller). Rows cover the window dilated by 1 — exactly
    the region the outer accept/rollback check measures — and, because
    every constrained voxel sits ≥ 1 voxel inside the patch (or on a
    patch edge that coincides with a volume edge), the ``np.gradient``
    stencils here equal the full-field ones: feasible ⇒ paste-back
    acceptable, for any SLSQP implementation's choice of optimum.

    Lower bounds: ``threshold`` on window rows; ``min(threshold, current
    Jdet)`` on halo rows (healthy border voxels must stay healthy,
    already-bad ones must not get worse — x0 is halo-feasible by
    construction).
    """
    pz, py, px = (int(s) for s in patch_size)
    n_patch = pz * py * px
    oz, oy, ox = (int(s) for s in win_start)
    sz, sy, sx = _unpack_size_3d(win_size)

    # Window-voxel linear indices in patch C-order; variable columns in
    # the [dx, dy, dz] channel-block layout.
    win_lin = (
        np.arange(oz, oz + sz)[:, None, None] * (py * px)
        + np.arange(oy, oy + sy)[None, :, None] * px
        + np.arange(ox, ox + sx)[None, None, :]
    ).ravel()
    cols = np.concatenate([win_lin, win_lin + n_patch, win_lin + 2 * n_patch])

    # Constrained rows: window dilated by 1, clamped to the patch. The
    # patch is clamped to the volume by the caller, so clamping to the
    # patch equals clamping to the volume (= the accept-check region).
    window = np.zeros((pz, py, px), dtype=bool)
    window[oz : oz + sz, oy : oy + sy, ox : ox + sx] = True
    region = np.zeros((pz, py, px), dtype=bool)
    region[
        max(oz - 1, 0) : min(oz + sz + 1, pz),
        max(oy - 1, 0) : min(oy + sy + 1, py),
        max(ox - 1, 0) : min(ox + sx + 1, px),
    ] = True
    rows = np.flatnonzero(region.ravel())
    window_rows = window.ravel()[rows]

    patch_base = np.asarray(patch_flat, dtype=np.float64).copy()

    def _patch_jdet(vec):
        dx = vec[:n_patch].reshape(pz, py, px)
        dy = vec[n_patch : 2 * n_patch].reshape(pz, py, px)
        dz = vec[2 * n_patch :].reshape(pz, py, px)
        return _numpy_jdet_3d(dz, dy, dx).ravel()

    def _embed(x):
        vec = patch_base.copy()
        vec[cols] = x
        return vec

    jdet0 = _patch_jdet(patch_base)[rows]
    lb = np.where(window_rows, threshold, np.minimum(threshold, jdet0))

    nlc = NonlinearConstraint(
        lambda x: _patch_jdet(_embed(x))[rows],
        lb,
        np.inf,
        jac=lambda x: jdet_constraint_jacobian_3d(_embed(x), (pz, py, px))[rows][:, cols].tocsr(),
    )
    return [nlc]
```

Also update the closing line of `_build_constraints_3d`'s docstring — append one sentence at the end of the docstring body:

```
    The serial solver no longer uses the ``window_reached_max=True`` branch
    of this function: max-window solves go through
    :func:`_build_constraints_3d_maxwindow` (patch-based halo no-damage
    constraints). The flag remains for API compatibility.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slsqp_maxwindow_halo.py tests/test_slsqp_review_fixes.py -q`
Expected: all PASS (the existing `_build_constraints_3d` unit tests must stay green — its behavior is unchanged).

- [ ] **Step 5: Lint gates**

Run: `python -m ruff check dvfopt tests` and `python -m ruff format --check dvfopt tests`
Expected: clean (run `python -m ruff format dvfopt tests` first if needed).

- [ ] **Step 6: Commit**

```powershell
git add dvfopt/core/slsqp/constraints3d.py tests/test_slsqp_maxwindow_halo.py
git commit -m "feat(slsqp3d): patch-based max-window constraint builder with halo no-damage rows"
```

---

### Task 2: Wire the patch builder into the serial solver

**Files:**
- Modify: `dvfopt/core/solver3d.py:11` (import), `dvfopt/core/solver3d.py:138-175` (`_optimize_single_window_3d`), `dvfopt/core/solver3d.py:329-344` (`_serial_fix_voxel` call site)
- Modify: `dvfopt/core/slsqp/constraints.py:147` (2D follow-up comment only)
- Test: `tests/test_slsqp_maxwindow_halo.py` (extend)

**Interfaces:**
- Consumes: `_build_constraints_3d_maxwindow(patch_flat, patch_size, win_start, win_size, threshold)` from Task 1.
- Produces: `_optimize_single_window_3d(..., window_reached_max=False, patch_ctx=None)` where `patch_ctx` is `(patch_flat, patch_size, win_start)`; required (asserted) when `window_reached_max=True`. Return contract unchanged: `(result_x, elapsed, success)` with `result_x` window-only.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_slsqp_maxwindow_halo.py`:

```python
class TestMaxWindowSolveIntegration:
    def _fold_sheet_volume(self):
        """The tripwire fixture: fold component far larger than a 3^3 window."""
        D = H = W = 8
        d = np.zeros((3, D, H, W), dtype=np.float64)
        d[2, :, :, 3] = 3.0
        return d

    def test_no_new_negatives_anywhere(self):
        # Border no-damage, globally: an accepted max-window solve fixes
        # window voxels and cannot create fresh negatives in the halo;
        # a rejected one is rolled back. So the set of negative voxels
        # never grows.
        from dvfopt.core.slsqp.iterative3d import iterative_3d

        d = self._fold_sheet_volume()
        neg_before = jacobian_det3D(d) <= THR - 1e-5
        phi = iterative_3d(d, verbose=0, max_window=(3, 3, 3), max_iterations=5)
        neg_after = jacobian_det3D(phi) <= THR - 1e-5
        assert not (neg_after & ~neg_before).any(), 'solver created new negative voxels'
        assert neg_after.sum() < neg_before.sum(), 'solver made no progress'

    def test_maxwindow_requires_patch_ctx(self):
        from dvfopt.core.solver3d import _optimize_single_window_3d

        x0 = np.zeros(3 * 27)
        with pytest.raises(AssertionError):
            _optimize_single_window_3d(
                x0, x0, (3, 3, 3), np.zeros((3, 3, 3), bool), THR, 50, 'SLSQP',
                window_reached_max=True,
            )
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `python -m pytest tests/test_slsqp_maxwindow_halo.py -q`
Expected: `test_maxwindow_requires_patch_ctx` FAILS (`DID NOT RAISE` / TypeError for the unknown kwarg). `test_no_new_negatives_anywhere` may already pass under scipy 1.15 — that is fine; it exists to hold under 1.18 after this task.

- [ ] **Step 3: Implement the wiring**

3a. In `dvfopt/core/solver3d.py`, extend the existing import (line 10-12):

```python
from dvfopt.core.slsqp.constraints3d import (
    _build_constraints_3d,
    _build_constraints_3d_maxwindow,
)
```

3b. Replace the constraint-building block of `_optimize_single_window_3d` (lines 138-161) with:

```python
def _optimize_single_window_3d(
    phi_sub_flat,
    phi_init_sub_flat,
    subvolume_size,
    freeze_mask,
    threshold,
    max_minimize_iter,
    method_name,
    window_reached_max=False,
    patch_ctx=None,
):
    """Run SLSQP on one 3D sub-volume.  Returns ``(result_x, elapsed, success)``.

    When *window_reached_max* is ``True``, the caller must supply
    *patch_ctx* = ``(patch_flat, patch_size, win_start)`` — the window's
    frozen surroundings (window + 2 voxels per side, clamped to the
    volume). Constraints are then built patch-based with halo no-damage
    rows (:func:`_build_constraints_3d_maxwindow`), so the sub-problem's
    feasible set equals the outer accept criterion and any successful
    solve survives paste-back regardless of which local optimum the
    SLSQP implementation picks (scipy ≥ 1.16 ports SLSQP to C and finds
    different valid optima than the Fortran one).
    """
    if window_reached_max:
        assert patch_ctx is not None, 'max-window solve requires patch_ctx'
        patch_flat, patch_size, win_start = patch_ctx
        constraints = _build_constraints_3d_maxwindow(
            patch_flat, patch_size, win_start, subvolume_size, threshold
        )
    else:
        constraints = _build_constraints_3d(
            phi_sub_flat,
            subvolume_size,
            freeze_mask,
            threshold,
            window_reached_max=False,
        )
```

(The `t0 = time.time()` / `minimize(...)` tail of the function stays exactly as is.)

3c. In `_serial_fix_voxel`, immediately before the `result_x, elapsed, opt_success = _optimize_single_window_3d(` call (current line 335), insert the patch extraction and pass it through:

```python
        patch_ctx = None
        if window_reached_max:
            _Dv, _Hv, _Wv = volume_shape
            pz0, pz1 = max(cz - hz - 2, 0), min(cz + hz_hi + 2, _Dv)
            py0, py1 = max(cy - hy - 2, 0), min(cy + hy_hi + 2, _Hv)
            px0, px1 = max(cx - hx - 2, 0), min(cx + hx_hi + 2, _Wv)
            _pslc = (slice(pz0, pz1), slice(py0, py1), slice(px0, px1))
            patch_flat = np.concatenate(
                [phi[2][_pslc].ravel(), phi[1][_pslc].ravel(), phi[0][_pslc].ravel()]
            )
            patch_ctx = (
                patch_flat,
                (pz1 - pz0, py1 - py0, px1 - px0),
                (cz - hz - pz0, cy - hy - py0, cx - hx - px0),
            )

        result_x, elapsed, opt_success = _optimize_single_window_3d(
            phi_sub_flat,
            phi_init_sub_flat,
            subvolume_size,
            freeze_mask,
            threshold,
            _eff_max_iter,
            method_name,
            window_reached_max=window_reached_max,
            patch_ctx=patch_ctx,
        )
```

3d. In `dvfopt/core/slsqp/constraints.py`, directly above line 147 (`exclude_bounds = not is_at_edge and not window_reached_max`), add the 2D follow-up comment:

```python
    # FOLLOW-UP (2D parity): at max window (exclude_bounds=False) this 2D
    # sub-problem has the same latent structure the 3D path fixed with
    # patch-based halo no-damage rows (_build_constraints_3d_maxwindow in
    # constraints3d.py): the border the outer accept check measures is
    # unconstrained here. No observed failure in 2D under scipy >= 1.16;
    # use the 3D builder as the template if one ever appears.
```

- [ ] **Step 4: Run the solver test battery**

Run: `python -m pytest tests/test_slsqp_maxwindow_halo.py tests/test_slsqp_review_fixes.py tests/test_iterative.py tests/test_integration_3d.py tests/test_solver3d_internals.py tests/test_constraints3d.py tests/test_constraints_and_params.py -q`
Expected: all PASS, including the tripwire `test_fold_larger_than_max_window_makes_progress`.

Known follow-on: `tests/test_integration_3d.py`, `tests/test_solver3d_internals.py`, and
`tests/test_slsqp_review_fixes.py` monkeypatch `_optimize_single_window_3d` with spy/fake
functions. If any fail with `TypeError: ... unexpected keyword argument 'patch_ctx'`,
update those spy signatures to accept and forward `**kwargs` (test-fixture maintenance
required by the new parameter — do not change production code to avoid it).

- [ ] **Step 5: Lint gates**

Run: `python -m ruff check dvfopt tests` and `python -m ruff format --check dvfopt tests`
Expected: clean.

- [ ] **Step 6: Commit**

```powershell
git add dvfopt/core/solver3d.py dvfopt/core/slsqp/constraints.py tests/test_slsqp_maxwindow_halo.py
git commit -m "feat(slsqp3d): route max-window solves through patch builder (halo no-damage)"
```

---

### Task 3: Lift the scipy pin + dual-scipy validation

**Files:**
- Modify: `pyproject.toml:6-15` (dependencies block)
- No new tests (this task validates the whole suite under both scipy generations).

**Interfaces:**
- Consumes: Tasks 1-2 merged into the working tree.
- Produces: unpinned `"scipy"` dependency; evidence that the full suite passes under scipy 1.15.3 AND latest (≥ 1.18).

- [ ] **Step 1: Baseline under scipy 1.15.3 (current env)**

Run: `python -c "import scipy; print(scipy.__version__)"` — expected `1.15.3`.
Run: `python -m pytest -q`
Expected: full suite PASSES (~1380+ tests, several minutes). Record the count.

- [ ] **Step 2: Unpin scipy in pyproject.toml**

Replace lines 6-15 of `pyproject.toml`:

```toml
dependencies = [
    "numpy",
    # TEMP pin: scipy 1.16 ported SLSQP from Fortran to C; the new
    # implementation converges to different (valid) local optima on the
    # windowed sub-problems, which the outer accept/rollback logic rejects
    # -> solve/reject livelock (no progress + huge runtimes). Tripwire:
    # tests/test_slsqp_review_fixes.py::TestFrozenEdgeReleaseAtMaxWindow3D
    # ::test_fold_larger_than_max_window_makes_progress fails under >=1.16.
    # Lift once the max-window accept logic is robust to alternative optima.
    "scipy<1.16",
```

with:

```toml
dependencies = [
    "numpy",
    "scipy",
```

- [ ] **Step 3: Validate under latest scipy**

```powershell
python -m pip install -q -U scipy
python -c "import scipy; print(scipy.__version__)"
```

Expected: ≥ 1.18.

Run: `python -m pytest -q`
Expected: full suite PASSES — in particular the tripwire
`tests/test_slsqp_review_fixes.py::TestFrozenEdgeReleaseAtMaxWindow3D::test_fold_larger_than_max_window_makes_progress`
and `tests/test_slsqp_maxwindow_halo.py`, in normal time (the livelock burned ~50× the
runtime before; if this run grinds for tens of minutes on the tripwire, that IS the
regression — stop and report rather than waiting it out).

- [ ] **Step 4: Cross-check 1.15.3 one more time, then land on latest**

```powershell
python -m pip install -q scipy==1.15.3
python -m pytest tests/test_slsqp_maxwindow_halo.py tests/test_slsqp_review_fixes.py tests/test_iterative.py -q
python -m pip install -q -U scipy
```

Expected: targeted battery PASSES under 1.15.3 (both scipy generations satisfy the
patch-based sub-problem); env ends on latest scipy, matching the unpinned dependency.

- [ ] **Step 5: Lint gates**

Run: `python -m ruff check dvfopt dvfopt_gui tests benchmarks` and `python -m ruff format --check dvfopt dvfopt_gui tests benchmarks`
Expected: clean.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml
git commit -m "build: lift scipy<1.16 pin — max-window sub-problem now robust to the C SLSQP port"
```

---

### Task 4: GUI Minors B1 — worker lifecycle & guards

All locations verified against the current tree (line numbers are current as of this plan).

**Files:**
- Modify: `dvfopt_gui/app.py` (closeEvent ~2802-2837; `_on_load` ~1170-1196; `_on_load_finished` ~1198-1205; `_on_load_failed` ~1207-1210; menu action ~1099; thr-spin hookup ~828-833; `_on_finished` ~2232-2326; `_push_undo_state` ~1310-1324; `_on_undo`/`_on_redo` ~1326-1340; `_restart_overview` ~1518-1535; `_format_inspector` ~2683-2731)
- Modify: `dvfopt_gui/worker.py:927-932` (auto fallback except)
- Test: `tests/test_gui_app.py` (extend)

**Interfaces:**
- Consumes: existing `LiveSolverWindow` internals as listed; `UNDO_MAX_BYTES` module constant (app.py:353).
- Produces: `LiveSolverWindow._cap_stack(stack: list) -> None`, `LiveSolverWindow._on_threshold_changed(_v) -> None`, `self._load_action` (QAction). Task 5's tests assume these exist.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_app.py`. Reuse the module's existing `qapp` fixture and its
`_snap(...)` / `_loaded_run(...)` builders; construct `LiveSolverWindow` exactly the way
the module's neighboring tests do (e.g. `test_overview_strip_wired_into_window`) — if the
constructor call below doesn't match the module's pattern, adapt to the module, not the
plan.

```python
class TestMinorsSweepLifecycle:
    def _win(self, qapp, D=1):
        vol = np.zeros((3, D, 8, 8), dtype=np.float64)
        win = LiveSolverWindow(np.zeros((3, 1, 8, 8)))
        win._apply_loaded_run(LoadedRun(volume=vol))
        return win

    def test_load_reentry_guarded_and_controls_reenabled(self, qapp, monkeypatch):
        win = self._win(qapp)
        monkeypatch.setattr(
            QtWidgets.QMessageBox, 'critical', staticmethod(lambda *a, **k: None)
        )

        class FakeWorker:
            def isRunning(self):
                return True

        win._load_worker = FakeWorker()
        # A second Ctrl+O while a load is in flight must return before
        # even opening the file dialog.
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            'getOpenFileName',
            staticmethod(lambda *a, **k: pytest.fail('dialog opened during in-flight load')),
        )
        win._on_load()
        # Finish/fail paths re-enable BOTH the toolbar button and menu action.
        win._load_btn.setEnabled(False)
        win._load_action.setEnabled(False)
        win._on_load_failed('boom')
        assert win._load_btn.isEnabled()
        assert win._load_action.isEnabled()

    def test_redo_stack_byte_budgeted(self, qapp, monkeypatch):
        import dvfopt_gui.app as app_mod

        win = self._win(qapp)
        vol_bytes = win._volume.nbytes
        monkeypatch.setattr(app_mod, 'UNDO_MAX_BYTES', 3 * vol_bytes)
        stack = [win._volume.copy() for _ in range(6)]
        win._cap_stack(stack)
        assert len(stack) <= 3
        assert sum(v.nbytes for v in stack) <= 3 * vol_bytes

    def test_undo_pushes_capped_redo(self, qapp, monkeypatch):
        import dvfopt_gui.app as app_mod

        win = self._win(qapp)
        monkeypatch.setattr(app_mod, 'UNDO_MAX_BYTES', 2 * win._volume.nbytes)
        win._undo_stack = [win._volume.copy() for _ in range(4)]
        win._redo_stack = [win._volume.copy(), win._volume.copy()]
        win._on_undo()
        assert sum(v.nbytes for v in win._redo_stack) <= 2 * win._volume.nbytes

    def test_thr_spin_repaints_after_run_finished(self, qapp, monkeypatch):
        win = self._win(qapp)

        class DoneWorker:
            def isRunning(self):
                return False

        win._worker = DoneWorker()  # run finished, ref not cleared
        calls = []
        monkeypatch.setattr(win, '_refresh_display_from_volume', lambda: calls.append(1))
        win._on_threshold_changed(0.02)
        assert calls, 'threshold change after a finished run must repaint'

    def test_thr_spin_noop_while_running(self, qapp, monkeypatch):
        win = self._win(qapp)

        class LiveWorker:
            def isRunning(self):
                return True

        win._worker = LiveWorker()
        calls = []
        monkeypatch.setattr(win, '_refresh_display_from_volume', lambda: calls.append(1))
        win._on_threshold_changed(0.02)
        assert not calls, 'must not repaint mid-run (stream owns the display)'

    def test_inspector_3d_idle_readout(self, qapp):
        win = self._win(qapp, D=4)
        assert win._latest is None
        html = win._format_inspector((2, 2))
        assert '3D' in html and 'min 6-tet' in html, (
            f'idle 3D volume must get the 3D readout, got: {html}'
        )

    def test_latest_cleared_on_finish(self, qapp):
        win = self._win(qapp)
        win._latest = _snap(np.zeros((2, 8, 8)))
        win._on_finished(np.zeros((2, 8, 8)), None)
        assert win._latest is None
```

Note on `test_load_reentry_guarded_and_controls_reenabled`: `_on_load_failed` opens a modal `QMessageBox.critical` — monkeypatch it too:
`monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', staticmethod(lambda *a, **k: None))` before calling. Include that line in the final test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_gui_app.py -k TestMinorsSweepLifecycle -q`
Expected: FAIL — `_load_action` / `_cap_stack` / `_on_threshold_changed` don't exist; the 3D idle inspector returns the 2D readout; `_latest` survives `_on_finished`.

- [ ] **Step 3: Implement the seven fixes in `dvfopt_gui/app.py` + `dvfopt_gui/worker.py`**

3a. **Menu action stored + re-entry guard + both controls gated (items 2).** At ~line 1099 change

```python
        file_menu.addAction('Load DVF…\tCtrl+O', self._on_load)
```

to

```python
        self._load_action = file_menu.addAction('Load DVF…\tCtrl+O', self._on_load)
```

In `_on_load` (top of method, before the file dialog):

```python
    def _on_load(self):
        # A load is already decoding on the worker thread — ignore re-entry
        # (the controls are disabled, but the Ctrl+O shortcut still fires).
        lw = self._load_worker
        if lw is not None and getattr(lw, 'isRunning', lambda: False)():
            return
```

(`self._load_worker` must exist before the first load — add `self._load_worker = None` in `__init__` next to the other worker attributes, near line 509.)

In the dispatch block (current lines 1190-1191), gate both controls:

```python
        self._load_btn.setEnabled(False)
        self._load_action.setEnabled(False)
```

In `_on_load_finished` and `_on_load_failed` (first lines), re-enable both:

```python
        self._load_btn.setEnabled(True)
        self._load_action.setEnabled(True)
```

3b. **closeEvent waits on the load worker (item 1).** After the overview-worker teardown block (current lines 2828-2836), before `super().closeEvent(ev)`:

```python
        # The load worker has no cancel flag (it is a one-shot decode);
        # give it a bounded wait so it cannot outlive the window, then
        # force it down — the process is exiting anyway.
        lw = self._load_worker
        if lw is not None and getattr(lw, 'isRunning', lambda: False)():
            lw.wait(5_000)
            if lw.isRunning():
                lw.terminate()
                lw.wait(1_000)
```

3c. **thr-spin: repaint after finished runs, accurate comment (items 3+8).** Replace lines 828-833 with:

```python
        self._thr_spin.valueChanged.connect(self._on_threshold_changed)
```

and add the method (near `_refresh_display_from_volume`):

```python
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
```

3d. **`_latest` cleared on finish (item 3).** In `_on_finished`, inside the `if phi_out is not None and self._volume is not None:` block, immediately after `self._refresh_display_from_volume()` (current line ~2277), add:

```python
            # The run is over and its result is spliced into the volume —
            # drop the last streamed snapshot so idle-path readers
            # (inspector, view toggles, thr-spin repaints) all see the
            # volume, exactly like after load/undo (which also set None).
            self._latest = None
```

3e. **3D idle inspector (item 7).** In `_format_inspector`, replace the opening of the 3D branch (current lines 2687-2688):

```python
        if self._latest is not None and self._latest.phi.ndim == 4:
            phi3d = self._latest.phi
```

with:

```python
        phi3d = None
        if self._latest is not None and self._latest.phi.ndim == 4:
            phi3d = self._latest.phi
        elif self._latest is None and self._volume is not None and self._volume.shape[1] > 1:
            # Idle with a true-3D volume loaded: read the volume directly
            # instead of falling through to the 2D single-slice readout.
            phi3d = self._volume
        if phi3d is not None:
```

(keep the existing body of the branch — `z = min(self._z, phi3d.shape[1] - 1)` etc. — indented under the new `if phi3d is not None:`).

3f. **Shared stack cap + redo budget (item 4).** Add the helper next to `_push_undo_state`:

```python
    def _cap_stack(self, stack: list) -> None:
        """Enforce the shared count + byte budget on an undo/redo stack
        (evicts oldest first; always keeps at least one entry)."""
        if len(stack) > self._UNDO_MAX:
            stack.pop(0)
        while len(stack) > 1 and sum(v.nbytes for v in stack) > UNDO_MAX_BYTES:
            stack.pop(0)
```

In `_push_undo_state`, replace the inline cap (the `if len(self._undo_stack) > self._UNDO_MAX:` + `while` lines) with `self._cap_stack(self._undo_stack)`. In `_on_undo`, after `self._redo_stack.append(self._volume.copy())` add `self._cap_stack(self._redo_stack)`. In `_on_redo`, after `self._undo_stack.append(self._volume.copy())` add `self._cap_stack(self._undo_stack)`.

3g. **`_restart_overview` checks `wait()` (item 5).** Replace lines 1521-1523:

```python
        if self._overview_worker is not None and self._overview_worker.isRunning():
            self._overview_worker.cancel()
            if not self._overview_worker.wait(2_000):
                # Per-slice cancel checks make a hang near-impossible, but
                # dropping the last reference to a still-running QThread
                # can crash Qt — force it down before reassigning.
                self._overview_worker.terminate()
                self._overview_worker.wait(1_000)
```

3h. **Narrow the auto-fallback except (item 6).** In `dvfopt_gui/worker.py` line 929, first verify the actual exception type:
`python -c "from dvfopt import make_strategy; make_strategy('definitely-not-a-label')"` — then narrow `except Exception:` to the observed type(s), expected:

```python
            try:
                strategy = make_strategy(label)
            except (KeyError, ValueError):
                # Registry label unavailable — fall back to the family default.
                label = 'm14' if kind == '2tri' else 'barrier'
                strategy = make_strategy(label)
```

(If the probe shows a different exception type, use that type — do not keep bare `Exception`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_gui_app.py -q`
Expected: all PASS (new class + the full existing GUI module — the thr-spin rewire, `_latest` clearing, and inspector change must not regress the ~138 existing GUI tests).

- [ ] **Step 5: Lint gates**

Run: `python -m ruff check dvfopt_gui tests` and `python -m ruff format --check dvfopt_gui tests`
Expected: clean.

- [ ] **Step 6: Commit**

```powershell
git add dvfopt_gui/app.py dvfopt_gui/worker.py tests/test_gui_app.py
git commit -m "fix(gui): worker-lifecycle minors — load gating, redo budget, thr-spin post-run, 3D idle inspector"
```

---

### Task 5: GUI Minors B2 — polish, docs, config

**Files:**
- Modify: `dvfopt_gui/app.py` (`validate_finite` ~356-367; `_restore_settings` ~2771-2773)
- Modify: `dvfopt_gui/persistence.py` (module docstring lines 1-38)
- Modify: `dvfopt_gui/strategy_params.py` (`_CHOICE_FIELDS` guard in `editable_fields` ~91-92; override validation in `build` ~124-141)
- Modify: `pyproject.toml:34-40` (dev extra: pin ruff) — also grep `requirements-dev.txt` for a ruff line and pin it identically if present
- Test: `tests/test_gui_app.py` (extend)

**Interfaces:**
- Consumes: Task 4's window fixtures; existing `editable_fields(cls)` / `StrategyParamsTab.build(algo, overrides)`; `OverviewWorker` (overview.py) and `_on_overview_chunk` (app.py ~1537-1554).
- Produces: `_valid_override(kind: str, name: str, value) -> bool` (module function in `strategy_params.py`). Nothing downstream depends on this task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_app.py`:

```python
class TestMinorsSweepPolish:
    def test_validate_finite_reports_first_index(self):
        from dvfopt_gui.app import validate_finite

        vol = np.zeros((3, 2, 4, 4))
        vol[1, 1, 2, 3] = np.nan
        msg = validate_finite(vol)
        assert msg is not None and '(1, 1, 2, 3)' in msg
        assert validate_finite(np.zeros((3, 1, 2, 2))) is None

    def test_choice_field_default_mismatch_asserts(self):
        import dataclasses

        from dvfopt_gui.strategy_params import editable_fields

        @dataclasses.dataclass
        class Bogus:
            accuracy: str = 'warp-speed'  # not in _CHOICE_FIELDS['accuracy']

        with pytest.raises(AssertionError, match='bare field name'):
            editable_fields(Bogus)

    def test_build_sanitizes_bad_overrides(self, qapp):
        from dvfopt_gui.strategy_params import StrategyParamsTab, editable_fields, strategy_class_for

        tab = StrategyParamsTab()
        algo = 'slp'
        cls = strategy_class_for(algo)
        fields = editable_fields(cls)
        float_fields = [n for n, k, _ in fields if k == 'float']
        int_fields = [n for n, k, _ in fields if k == 'int']
        assert float_fields and int_fields, 'test needs one field of each kind on slp'
        bad = {float_fields[0]: float('nan'), int_fields[0]: 'abc'}
        tab.build(algo, bad)  # must not crash
        vals = tab.values()
        # Sanitized fields fall back to defaults -> values() reports no override.
        assert float_fields[0] not in vals
        assert int_fields[0] not in vals

    def test_overview_stale_chunk_rejected(self, qapp):
        from dvfopt_gui.overview import OverviewWorker

        vol = np.zeros((3, 4, 8, 8), dtype=np.float64)
        win = LiveSolverWindow(np.zeros((3, 1, 8, 8)))
        win._apply_loaded_run(LoadedRun(volume=vol))
        win._overview_counts = np.zeros(4, dtype=np.int64)
        w_old = OverviewWorker(vol, parent=win)
        w_new = OverviewWorker(vol, parent=win)
        win._overview_worker = w_new
        # Neither thread is started: emitting from the test thread delivers
        # synchronously and sender() is the emitting worker inside the slot.
        w_old.chunkReady.connect(win._on_overview_chunk)
        w_new.chunkReady.connect(win._on_overview_chunk)
        w_old.chunkReady.emit(0, np.array([7, 7], dtype=np.int64))
        assert not win._overview_counts.any(), 'stale worker chunk must be rejected'
        w_new.chunkReady.emit(0, np.array([5, 5], dtype=np.int64))
        assert list(win._overview_counts[:2]) == [5, 5], 'current worker chunk must land'
```

Note: `test_overview_stale_chunk_rejected` is the automated race-regression coverage
(ledger item "no automated overview stale-chunk race test") — it pins the sender-guard
behavior that was previously only hand-verified. It should PASS already at Step 2;
that is expected (it is a regression lock, not a bug fix). The other three must fail.

- [ ] **Step 2: Run the tests to verify the three fix-tests fail**

Run: `python -m pytest tests/test_gui_app.py -k TestMinorsSweepPolish -q`
Expected: `test_validate_finite_reports_first_index` PASSES already (message format unchanged — the fix is allocation-only; keep it as a behavior lock), `test_choice_field_default_mismatch_asserts` FAILS (no assertion yet), `test_build_sanitizes_bad_overrides` FAILS (NaN applied verbatim), `test_overview_stale_chunk_rejected` PASSES (regression lock).

- [ ] **Step 3: Implement**

3a. **`validate_finite` first-hit without full materialization (item 13)** — replace line 362:

```python
    first = tuple(int(i) for i in np.argwhere(bad)[0])
```

with:

```python
    # argmax short-circuits to the first True in C-order — same index as
    # argwhere(bad)[0] without materialising every bad coordinate.
    first = tuple(int(i) for i in np.unravel_index(int(bad.argmax()), vol.shape))
```

3b. **Threshold 0.0 restore (item 10)** — replace lines 2771-2773:

```python
        thr = s.value('threshold', 0.0, type=float)
        if thr:
            self._thr_spin.setValue(thr)
```

with:

```python
        # ``if thr:`` would silently skip a legitimately-saved 0.0 —
        # presence, not truthiness, decides whether to restore.
        if s.contains('threshold'):
            self._thr_spin.setValue(s.value('threshold', 0.0, type=float))
```

3c. **`_CHOICE_FIELDS` collision guard (item 11)** — in `strategy_params.py` `editable_fields`, replace:

```python
        if f.name in _CHOICE_FIELDS:
            out.append((f.name, 'choice', default))
```

with:

```python
        if f.name in _CHOICE_FIELDS:
            choices = _CHOICE_FIELDS[f.name]
            assert str(default) in choices, (
                f'{cls.__name__}.{f.name}: default {default!r} not in {choices}. '
                f'_CHOICE_FIELDS is keyed by bare field name across ALL strategy '
                f'dataclasses — a colliding field with different choices needs '
                f'per-strategy keying.'
            )
            out.append((f.name, 'choice', default))
```

3d. **Override validation on restore (item 12)** — add to `strategy_params.py` (module level, near `_CHOICE_FIELDS`):

```python
def _valid_override(kind: str, name: str, value) -> bool:
    """Reject persisted override values that would corrupt the widgets:
    wrong-typed ints, non-finite floats, unknown choice strings. The
    overrides come from a JSON round-trip in QSettings, so anything a
    past version (or a hand-edited settings file) wrote can show up."""
    if kind == 'int':
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == 'float':
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    if kind == 'bool':
        return isinstance(value, bool)
    if kind == 'choice':
        return isinstance(value, str) and value in _CHOICE_FIELDS.get(name, ())
    return True
```

and in `build`, replace `value = overrides.get(name, default)` with:

```python
            value = overrides.get(name, default)
            if name in overrides and not _valid_override(kind, name, value):
                value = default
```

(Check the actual `kind` strings emitted by `editable_fields` — if it emits kinds beyond
`int`/`float`/`bool`/`choice` (e.g. `str`), extend `_valid_override` with an explicit
branch per kind rather than relying on the permissive tail return.)

3e. **persistence.py docstring (item 9)** — replace the two `history_phi` schema lines in the module docstring:

```python
* ``history_phi`` — ``(N, 2, H, W)`` float64, every snapshot's phi.
```

with:

```python
* ``history_phi`` — every snapshot's phi: ``(N, 2, H, W)`` float64 for
  2D runs, ``(N, 3, D, H, W)`` for true-3D runs (``dim`` == 3).
```

and add to the always-present key list (after the `history_max_size` bullet):

```python
* ``dim`` — 0-d int, 2 or 3: whether the run's snapshots are per-slice
  2D fields or full 3D volumes. Written by ``build_save_payload``;
  ``parse_loaded`` does not surface it on :class:`LoadedRun` — loaders
  derive dimensionality from ``volume.shape`` / snapshot ``ndim``.
```

3f. **Pin ruff (item 15)** — in `pyproject.toml` dev extra replace `"ruff>=0.4",` with `"ruff==0.15.21",`. Then `Select-String -Path requirements-dev.txt -Pattern ruff` — if it lists ruff, pin it to the same version.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_gui_app.py tests/test_gui_logic.py tests/test_gui_io_formats.py -q`
Expected: all PASS.

- [ ] **Step 5: Full suite + lint gates**

Run: `python -m pytest -q`
Expected: full suite PASSES.
Run: `python -m ruff --version` (must print 0.15.21 — `python -m pip install -q ruff==0.15.21` if not), then `python -m ruff check dvfopt dvfopt_gui tests benchmarks` and `python -m ruff format --check dvfopt dvfopt_gui tests benchmarks`
Expected: clean.

- [ ] **Step 6: Commit**

```powershell
git add dvfopt_gui/app.py dvfopt_gui/persistence.py dvfopt_gui/strategy_params.py pyproject.toml tests/test_gui_app.py
git commit -m "fix(gui): polish minors — finite-check allocation, thr=0 restore, override validation, choice-field guard, docs, ruff pin"
```

(add `requirements-dev.txt` to the `git add` if it was touched in 3f.)

---
