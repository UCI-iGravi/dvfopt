# Strict-feasibility 2D — LP/SLP for the 2-triangle constraint

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `research/strict_feasibility_2d/` containing two new LP-based algorithms (`lp_oneshot_2tri`, `slp_iter_2tri`) that achieve strict 2-triangle feasibility (`min(T1, T2) ≥ 0.01`) with minimised L1 deviation, plus a bake-off harness comparing them against five existing methods (m02 / m10 / m14 / m14-schwarz / cluster-pipeline) on a curated worst-case suite (synthetic + B0039 z=12).

**Architecture:** Exploit that the 2-tri constraint is affine in `phi` after fixing per-cell orientation to the canonical positive sign. `lp_oneshot` linearises around a harmonic-extension seed and solves one L1 LP via HiGHS (scipy). `slp_iter` repeats with adaptive trust region until exact-T feasibility holds. Comparison harness writes a single `comparison.csv` consumed by two analysis notebooks. Sparse Jacobians reuse the math already in `dvfopt.core.tri_primitives.tri_grad_T_v` — adapted from adjoint-only to row-wise sparse-matrix construction.

**Tech Stack:** Python 3.10+, numpy, scipy ≥ 1.6 (HiGHS via `scipy.optimize.linprog`), `dvfopt` (existing wallbreakers + tri primitives), pytest, jupyter.

---

## Spec reference

This plan implements [`docs/superpowers/specs/2026-06-14-strict-feasibility-2d-design.md`](../specs/2026-06-14-strict-feasibility-2d-design.md). Read that first.

## File layout (decided up-front)

**Created by this plan:**

- `research/strict_feasibility_2d/README.md` — status board + run instructions
- `research/strict_feasibility_2d/DESIGN.md` — pointer to the spec
- `research/strict_feasibility_2d/__init__.py`
- `research/strict_feasibility_2d/algorithms/__init__.py`
- `research/strict_feasibility_2d/algorithms/orientation_fix.py` — `canonical_signs(H, W)` helper
- `research/strict_feasibility_2d/algorithms/tri_linearize.py` — `linearize_T_2tri(phi_flat, H, W) -> (T_vals, J_sparse)` building the sparse `(K, 2HW)` Jacobian
- `research/strict_feasibility_2d/algorithms/highs_solver.py` — `solve_l1_lp_step(phi_in_flat, phi_lin_flat, T_lin, J_sparse, threshold, trust_radius=None)` wrapping `scipy.optimize.linprog(method='highs')`
- `research/strict_feasibility_2d/algorithms/lp_direct_2tri.py` — `lp_oneshot(...)` + `slp_iter(...)` public API
- `research/strict_feasibility_2d/worst_cases/catalog.md` — selected cases + rationale
- `research/strict_feasibility_2d/worst_cases/_load.py` — case-loader functions
- `research/strict_feasibility_2d/worst_cases/_build_adversarial.py` — generator for new synthetic cases
- `research/strict_feasibility_2d/runners/__init__.py`
- `research/strict_feasibility_2d/runners/_compare.py` — `run_method(name, phi_in) -> dict` dispatch
- `research/strict_feasibility_2d/runners/_run_lp_synthetic.py` — batch over synthetic cases
- `research/strict_feasibility_2d/runners/_run_lp_b0039.py` — batch over B0039 slices
- `research/strict_feasibility_2d/runners/output/.gitkeep`
- `research/strict_feasibility_2d/analysis/01_baseline_l1_gap.ipynb` — headline-table notebook
- `research/strict_feasibility_2d/analysis/02_lp_certifies_optimum.ipynb` — per-case L1 gap analysis
- `tests/research/__init__.py`
- `tests/research/strict_feasibility_2d/__init__.py`
- `tests/research/strict_feasibility_2d/test_orientation_fix.py`
- `tests/research/strict_feasibility_2d/test_tri_linearize.py`
- `tests/research/strict_feasibility_2d/test_highs_solver.py`
- `tests/research/strict_feasibility_2d/test_lp_direct.py`
- `tests/research/strict_feasibility_2d/test_worst_cases.py`
- `tests/research/strict_feasibility_2d/test_comparison.py`

**Modified by this plan:** none. The research subfolder is self-contained.

**Reused (read-only) from existing codebase:**

- `dvfopt.core.tri_primitives.tri_areas_flat` — exact T values
- `dvfopt.core.tri_primitives.tri_grad_T_v` — math reference for sparse Jacobian construction (we reimplement as explicit sparse, not adjoint)
- `dvfopt.jacobian.triangle_sign._triangle_areas_2d` — adversarial validation (exact T eval)
- `dvfopt.core.wallbreakers.harmonic_extension_2d` — feasible seed
- `dvfopt.iterative_2d_tri_harmonic_polished` (m10), `iterative_2d_tri_refine_repair` (m14), `iterative_2d_tri_refine_repair_schwarz` (m14-schwarz) — baselines
- `data/dvfs/canonical_2tri_2d/*.npz` — synthetic worst cases
- `data/dvfs/b0039/b0039_laplacian_deformation_field.npy` — B0039 source

---

## Constants used across tasks

These are referenced in multiple task code blocks. Defined here once for DRY.

```python
THRESHOLD = 0.01            # target floor for min(T1, T2)
SAFETY_TOL = 1e-5           # numerical slack for "feasible at exact eval"
TRUST_RADIUS_0 = 0.5        # initial trust region in cell units (slp_iter)
SLP_MAX_ITER = 20
SLP_FTOL = 1e-6             # ‖phi^(it+1) - phi^(it)‖∞ convergence threshold
```

---

## Task 1: Bootstrap folder structure + README + DESIGN pointer

**Files:**
- Create: `research/strict_feasibility_2d/__init__.py` (empty)
- Create: `research/strict_feasibility_2d/algorithms/__init__.py` (empty)
- Create: `research/strict_feasibility_2d/runners/__init__.py` (empty)
- Create: `research/strict_feasibility_2d/runners/output/.gitkeep` (empty)
- Create: `research/strict_feasibility_2d/README.md`
- Create: `research/strict_feasibility_2d/DESIGN.md`

This is a no-test scaffolding task. Pure documentation + skeleton.

- [ ] **Step 1: Create the empty `__init__.py` files + `.gitkeep`**

```bash
mkdir -p research/strict_feasibility_2d/{algorithms,runners/output,worst_cases/synthetic,worst_cases/b0039,analysis}
touch research/strict_feasibility_2d/__init__.py
touch research/strict_feasibility_2d/algorithms/__init__.py
touch research/strict_feasibility_2d/runners/__init__.py
touch research/strict_feasibility_2d/runners/output/.gitkeep
```

- [ ] **Step 2: Write `research/strict_feasibility_2d/README.md`**

```markdown
# research/strict_feasibility_2d/

Active research thread targeting **strict 2-triangle feasibility with
minimised L1 deviation** on worst-case 2D deformation fields.

See [`DESIGN.md`](DESIGN.md) for the design spec.

## Status

| Milestone | Status |
|---|---|
| Folder scaffolded | ☐ |
| Algorithms implemented (`lp_oneshot`, `slp_iter`) | ☐ |
| Worst-case catalog built | ☐ |
| Synthetic bake-off run | ☐ |
| B0039 z=12 bake-off run | ☐ |
| Analysis notebooks finalised | ☐ |

## How to run

```bash
# Run the synthetic bake-off (canonical + adversarial cases):
python research/strict_feasibility_2d/runners/_run_lp_synthetic.py

# Run the B0039 bake-off (z=12 + selected slices):
python research/strict_feasibility_2d/runners/_run_lp_b0039.py

# Then open the analysis notebooks:
#   research/strict_feasibility_2d/analysis/01_baseline_l1_gap.ipynb
#   research/strict_feasibility_2d/analysis/02_lp_certifies_optimum.ipynb
```

Outputs land in `runners/output/`:
- `comparison.csv` — one row per (case, method)
- `corrected/<case>_<method>.npz` — corrected `phi` arrays
```

- [ ] **Step 3: Write `research/strict_feasibility_2d/DESIGN.md`**

```markdown
# Design spec

This research thread is governed by
[`docs/superpowers/specs/2026-06-14-strict-feasibility-2d-design.md`](../../docs/superpowers/specs/2026-06-14-strict-feasibility-2d-design.md).

Read the spec for goal, approach, comparison plan, success criteria,
and fallback plan.
```

- [ ] **Step 4: Commit**

```bash
git add research/strict_feasibility_2d/
git commit -m "Scaffold research/strict_feasibility_2d/ folder"
```

---

## Task 2: Worst-case loader + catalog

**Files:**
- Create: `research/strict_feasibility_2d/worst_cases/_load.py`
- Create: `research/strict_feasibility_2d/worst_cases/catalog.md`
- Create: `tests/research/__init__.py`
- Create: `tests/research/strict_feasibility_2d/__init__.py`
- Test:   `tests/research/strict_feasibility_2d/test_worst_cases.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/strict_feasibility_2d/test_worst_cases.py
"""Smoke-test the worst-case loaders."""
import numpy as np
import pytest

from research.strict_feasibility_2d.worst_cases import _load


def test_load_synthetic_canonical():
    cases = _load.load_synthetic_canonical()
    keys = {name for name, phi, meta in cases}
    assert 'bowtie_7x7_shoelace' in keys
    assert '03c_20x20_opposite' in keys
    assert '03d_20x20_crossing' in keys


def test_synthetic_shapes():
    for name, phi, meta in _load.load_synthetic_canonical():
        assert phi.ndim == 3 and phi.shape[0] == 2, name
        assert phi.dtype == np.float64, name


def test_load_b0039_z12():
    name, phi, meta = _load.load_b0039_slice(12)
    assert name == 'b0039_z012'
    assert phi.shape == (2, 320, 456)
    assert meta['init_n_neg'] > 0


def test_b0039_load_invalid_z_raises():
    with pytest.raises(IndexError):
        _load.load_b0039_slice(99999)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/research/strict_feasibility_2d/test_worst_cases.py -v
```

Expected: ImportError / ModuleNotFoundError on `research.strict_feasibility_2d.worst_cases._load`.

- [ ] **Step 3: Create test package `__init__.py` files**

```bash
touch tests/research/__init__.py tests/research/strict_feasibility_2d/__init__.py
```

- [ ] **Step 4: Implement `_load.py`**

```python
# research/strict_feasibility_2d/worst_cases/_load.py
"""Loaders for the curated worst-case suite.

Synthetic cases come from ``data/dvfs/canonical_2tri_2d/*.npz`` (already
generated by ``scripts/_save_canonical_2tri_2d.py``). B0039 slices come
from the real registration DVF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

_REPO = Path(__file__).resolve().parents[3]
_CANONICAL_DIR = _REPO / 'data' / 'dvfs' / 'canonical_2tri_2d'
_B0039 = _REPO / 'data' / 'dvfs' / 'b0039' / 'b0039_laplacian_deformation_field.npy'


def _stats(phi_2hw: np.ndarray) -> dict:
    """Compute initial 2-tri fold count + min_T."""
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    return {
        'init_n_neg': int((np.minimum(T1, T2) <= 0).sum()),
        'init_min_T': float(np.minimum(T1, T2).min()),
        'shape': tuple(phi_2hw.shape[1:]),
    }


def load_synthetic_canonical():
    """Yield ``(name, phi_2hw, meta)`` for every NPZ in
    ``data/dvfs/canonical_2tri_2d/``."""
    out = []
    for path in sorted(_CANONICAL_DIR.glob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            phi = d['phi'].astype(np.float64)
        out.append((path.stem, phi, _stats(phi)))
    return out


def load_b0039_slice(z: int):
    """Load B0039 z-slice as ``(name, phi_2hw, meta)``."""
    vol = np.load(_B0039)
    D = vol.shape[1]
    if not (0 <= z < D):
        raise IndexError(f'z={z} out of [0, {D})')
    phi = vol[1:, z].astype(np.float64)
    name = f'b0039_z{z:03d}'
    return name, phi, _stats(phi)
```

- [ ] **Step 5: Write `worst_cases/catalog.md`**

```markdown
# Worst-case catalog

## Synthetic
| Case | Source | Shape | init n_neg | init min_T |
|---|---|---|---|---|
| `bowtie_7x7_shoelace` | data/dvfs/canonical_2tri_2d/ | 7×7 | 2 | -0.70 |
| `01a_10x10_crossing` | data/dvfs/canonical_2tri_2d/ | 10×10 | 24 | -0.74 |
| `01b_10x10_opposite` | data/dvfs/canonical_2tri_2d/ | 10×10 | 10 | -0.59 |
| `03a_10x10_opposite` | data/dvfs/canonical_2tri_2d/ | 10×10 | 23 | -0.81 |
| `03b_10x10_crossing` | data/dvfs/canonical_2tri_2d/ | 10×10 | 28 | -0.70 |
| `03c_20x20_opposite` | data/dvfs/canonical_2tri_2d/ | 20×20 | 58 | -0.81 |
| `03d_20x20_crossing` | data/dvfs/canonical_2tri_2d/ | 20×20 | 72 | -0.74 |

Adversarial cases (built by ``_build_adversarial.py``) land in
``worst_cases/synthetic/`` after Task 8.

## B0039
| Case | Source | Shape | Status |
|---|---|---|---|
| `b0039_z012` | data/dvfs/b0039/b0039_laplacian_deformation_field.npy z=12 | 320×456 | Manuscript-canonical hardest slice |

Empirical worst slices (from cluster-pipeline residuals) discovered
after the synthetic suite is settled — added inline.
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/research/strict_feasibility_2d/test_worst_cases.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add research/strict_feasibility_2d/worst_cases/ tests/research/
git commit -m "Add worst-case loader + catalog"
```

---

## Task 3: orientation_fix.py — canonical sign helper

**Files:**
- Create: `research/strict_feasibility_2d/algorithms/orientation_fix.py`
- Test:   `tests/research/strict_feasibility_2d/test_orientation_fix.py`

The "canonical sign" of each triangle on an undeformed grid is **+1** by
construction of `_triangle_areas_2d` (the formula returns half a positive
shoelace area for the canonical winding). All `T_k(phi=0) = 0.5 > 0`, so
the sign vector is `+ones(K)`. This task gives it a name and unit test.

- [ ] **Step 1: Write the failing test**

```python
# tests/research/strict_feasibility_2d/test_orientation_fix.py
import numpy as np

from research.strict_feasibility_2d.algorithms.orientation_fix import (
    canonical_signs,
    n_triangles,
)


def test_n_triangles_count():
    # 2 triangles per (H-1)*(W-1) cell
    assert n_triangles(7, 7) == 2 * 6 * 6
    assert n_triangles(10, 10) == 2 * 9 * 9
    assert n_triangles(20, 20) == 2 * 19 * 19


def test_canonical_signs_all_positive():
    s = canonical_signs(10, 10)
    assert s.shape == (n_triangles(10, 10),)
    assert np.all(s == 1.0)


def test_canonical_signs_dtype():
    s = canonical_signs(7, 7)
    assert s.dtype == np.float64
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/research/strict_feasibility_2d/test_orientation_fix.py -v
```

Expected: ImportError on `canonical_signs`.

- [ ] **Step 3: Implement `orientation_fix.py`**

```python
# research/strict_feasibility_2d/algorithms/orientation_fix.py
"""Per-triangle orientation lock for the LP reformulation.

``_triangle_areas_2d`` returns half the positive shoelace determinant
under canonical winding — every triangle is positively oriented when
``phi = 0``. Locking ``T_k >= +τ`` (i.e. sign = +1 for every triangle)
makes the constraint affine in ``phi`` after one linearisation step.
"""
from __future__ import annotations

import numpy as np


def n_triangles(H: int, W: int) -> int:
    """Total number of 2-tri triangles on an H x W grid (T1 + T2 per cell)."""
    return 2 * (H - 1) * (W - 1)


def canonical_signs(H: int, W: int) -> np.ndarray:
    """All +1 — the canonical positive orientation of an undeformed grid."""
    return np.ones(n_triangles(H, W), dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/research/strict_feasibility_2d/test_orientation_fix.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add research/strict_feasibility_2d/algorithms/orientation_fix.py tests/research/strict_feasibility_2d/test_orientation_fix.py
git commit -m "Add orientation_fix.canonical_signs helper"
```

---

## Task 4: tri_linearize.py — sparse Jacobian builder

**Files:**
- Create: `research/strict_feasibility_2d/algorithms/tri_linearize.py`
- Test:   `tests/research/strict_feasibility_2d/test_tri_linearize.py`

This is the math heart. The existing `dvfopt.core.tri_primitives.tri_grad_T_v(phi, H, W, v)`
returns `J^T @ v` for arbitrary `v` via vectorised scatter-adds. We need
the explicit sparse `J` of shape `(K, 2HW)` for the LP. We construct it
directly using the same per-triangle gradient formulas as `tri_grad_T_v`,
but emit `(row, col, value)` triples for a `scipy.sparse.coo_matrix`.

Each `T_k` depends on exactly 6 entries of phi (3 corners × 2 channels)
so `J` has 6 nonzeros per row → `6K = 12*(H-1)*(W-1)` nonzeros total.

- [ ] **Step 1: Write the failing test**

```python
# tests/research/strict_feasibility_2d/test_tri_linearize.py
import numpy as np
import scipy.sparse as sp

from dvfopt.core.tri_primitives import tri_areas_flat, tri_grad_T_v
from research.strict_feasibility_2d.algorithms.tri_linearize import (
    build_sparse_jacobian_T,
    linearize_T_2tri,
)
from research.strict_feasibility_2d.algorithms.orientation_fix import n_triangles


def _flat_zeros(H, W):
    return np.zeros(2 * H * W, dtype=np.float64)


def test_T_values_match_tri_areas_flat_at_zero():
    H, W = 7, 7
    phi_flat = _flat_zeros(H, W)
    T_vals, _ = linearize_T_2tri(phi_flat, H, W)
    assert T_vals.shape == (n_triangles(H, W),)
    np.testing.assert_allclose(T_vals, tri_areas_flat(phi_flat, H, W))


def test_jacobian_shape_and_sparsity():
    H, W = 7, 7
    K = n_triangles(H, W)
    J = build_sparse_jacobian_T(_flat_zeros(H, W), H, W)
    assert isinstance(J, sp.spmatrix)
    assert J.shape == (K, 2 * H * W)
    # 6 nonzeros per row.
    assert J.nnz == 6 * K


def test_jacobian_matches_adjoint_via_random_probes():
    """J @ e_col matches the adjoint-implied column, for a random sample."""
    rng = np.random.default_rng(0)
    H, W = 10, 10
    phi_flat = rng.uniform(-0.3, 0.3, size=2 * H * W)
    J = build_sparse_jacobian_T(phi_flat, H, W).tocsr()
    K = n_triangles(H, W)
    # For row k: J[k, :] == tri_grad_T_v(phi, H, W, e_k)
    for k in rng.integers(0, K, size=20):
        e_k = np.zeros(K)
        e_k[k] = 1.0
        row_expected = tri_grad_T_v(phi_flat, H, W, e_k)
        row_actual = J[k].toarray().ravel()
        np.testing.assert_allclose(row_actual, row_expected, atol=1e-12)


def test_linearization_first_order_accuracy():
    """T(phi + dphi) - T(phi) ≈ J @ dphi to O(‖dphi‖²)."""
    rng = np.random.default_rng(1)
    H, W = 10, 10
    phi_flat = rng.uniform(-0.2, 0.2, size=2 * H * W)
    T0, J = linearize_T_2tri(phi_flat, H, W)
    # Many small perturbations.
    errs_lin = []
    errs_const = []
    for _ in range(10):
        dphi = rng.normal(scale=1e-4, size=2 * H * W)
        T1 = tri_areas_flat(phi_flat + dphi, H, W)
        lin = T0 + J @ dphi
        errs_lin.append(np.max(np.abs(T1 - lin)))
        errs_const.append(np.max(np.abs(T1 - T0)))
    # Linearisation error must be at least 100x smaller than constant
    # prediction for these displacements (quadratic vs linear in dphi).
    assert np.mean(errs_lin) < np.mean(errs_const) / 100.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/research/strict_feasibility_2d/test_tri_linearize.py -v
```

Expected: ImportError on `linearize_T_2tri` and `build_sparse_jacobian_T`.

- [ ] **Step 3: Implement `tri_linearize.py`**

```python
# research/strict_feasibility_2d/algorithms/tri_linearize.py
"""Sparse Jacobian construction for the 2-triangle constraint.

The existing ``dvfopt.core.tri_primitives.tri_grad_T_v`` returns
``J^T @ v`` for arbitrary ``v`` via vectorised scatter-add — efficient
for L-BFGS adjoint products but not the explicit sparse ``J`` the LP
needs. This module emits ``J`` as a ``scipy.sparse.coo_matrix`` directly,
using the same per-triangle gradient pattern.

Each row of ``J`` corresponds to one triangle ``T_k`` and has exactly
six nonzero entries — the 3 corners × 2 displacement channels that
``T_k`` depends on. ``J.shape == (2*(H-1)*(W-1), 2*H*W)``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from dvfopt.core.tri_primitives import tri_areas_flat


def _ref_grid(H: int, W: int):
    """Reference (undeformed) corner coordinates — replicates the
    helper used in ``tri_primitives``."""
    ref_y, ref_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return ref_y.astype(np.float64), ref_x.astype(np.float64)


def build_sparse_jacobian_T(phi_flat: np.ndarray, H: int, W: int) -> sp.coo_matrix:
    """Build the sparse Jacobian ``J`` of the 2-tri constraint at ``phi_flat``.

    Decision-vector layout: ``phi_flat[:HW] = dy``, ``phi_flat[HW:] = dx``.
    Constraint vector layout: ``[T1.ravel(), T2.ravel()]``.

    Returns
    -------
    J : scipy.sparse.coo_matrix, shape (2*(H-1)*(W-1), 2*H*W).
    """
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_y = ref_y + dy
    def_x = ref_x + dx

    n_cells = (H - 1) * (W - 1)
    # Index helpers (flat index into phi for each corner block).
    # Corner labels: TL = (i, j), TR = (i, j+1), BL = (i+1, j), BR = (i+1, j+1)
    ii, jj = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing='ij')
    idx_TL = (ii * W + jj).ravel()
    idx_TR = (ii * W + (jj + 1)).ravel()
    idx_BL = ((ii + 1) * W + jj).ravel()
    idx_BR = ((ii + 1) * W + (jj + 1)).ravel()

    # Corner deformed coords, flattened to (n_cells,).
    y_tl = def_y[:-1, :-1].ravel()
    x_tl = def_x[:-1, :-1].ravel()
    y_tr = def_y[:-1, 1:].ravel()
    x_tr = def_x[:-1, 1:].ravel()
    y_bl = def_y[1:, :-1].ravel()
    x_bl = def_x[1:, :-1].ravel()
    y_br = def_y[1:, 1:].ravel()
    x_br = def_x[1:, 1:].ravel()

    rows = []
    cols = []
    vals = []

    # ----- T1 rows (rows 0 .. n_cells-1). T1 corners: A=TR, B=BL, C=BR.
    t1_row = np.arange(n_cells)  # row index in J for each cell's T1
    # gradient of T1 w.r.t. each corner+channel — match the formulas in
    # tri_grad_T_v exactly (T1 chunk).
    # dT1/dx[TR] = 0.5 * (y_br - y_bl)
    rows.append(t1_row); cols.append(idx_TR + HW); vals.append(0.5 * (y_br - y_bl))
    # dT1/dy[TR] = 0.5 * (x_bl - x_br)
    rows.append(t1_row); cols.append(idx_TR);      vals.append(0.5 * (x_bl - x_br))
    # dT1/dx[BL] = -0.5 * (y_br - y_tr)
    rows.append(t1_row); cols.append(idx_BL + HW); vals.append(-0.5 * (y_br - y_tr))
    # dT1/dy[BL] = 0.5 * (x_br - x_tr)
    rows.append(t1_row); cols.append(idx_BL);      vals.append(0.5 * (x_br - x_tr))
    # dT1/dx[BR] = 0.5 * (y_bl - y_tr)
    rows.append(t1_row); cols.append(idx_BR + HW); vals.append(0.5 * (y_bl - y_tr))
    # dT1/dy[BR] = -0.5 * (x_bl - x_tr)
    rows.append(t1_row); cols.append(idx_BR);      vals.append(-0.5 * (x_bl - x_tr))

    # ----- T2 rows (rows n_cells .. 2n_cells-1). T2 corners: A=TL, B=BL, C=TR.
    t2_row = np.arange(n_cells) + n_cells
    # dT2/dx[TL] = 0.5 * (y_tr - y_bl)
    rows.append(t2_row); cols.append(idx_TL + HW); vals.append(0.5 * (y_tr - y_bl))
    # dT2/dy[TL] = 0.5 * (x_bl - x_tr)
    rows.append(t2_row); cols.append(idx_TL);      vals.append(0.5 * (x_bl - x_tr))
    # dT2/dx[BL] = -0.5 * (y_tr - y_tl)
    rows.append(t2_row); cols.append(idx_BL + HW); vals.append(-0.5 * (y_tr - y_tl))
    # dT2/dy[BL] = 0.5 * (x_tr - x_tl)
    rows.append(t2_row); cols.append(idx_BL);      vals.append(0.5 * (x_tr - x_tl))
    # dT2/dx[TR] = 0.5 * (y_bl - y_tl)
    rows.append(t2_row); cols.append(idx_TR + HW); vals.append(0.5 * (y_bl - y_tl))
    # dT2/dy[TR] = -0.5 * (x_bl - x_tl)
    rows.append(t2_row); cols.append(idx_TR);      vals.append(-0.5 * (x_bl - x_tl))

    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    vals_arr = np.concatenate(vals)
    n_rows = 2 * n_cells
    return sp.coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_rows, 2 * HW))


def linearize_T_2tri(phi_flat: np.ndarray, H: int, W: int):
    """Return ``(T_vals, J)`` at ``phi_flat`` for the 2-tri constraint."""
    T_vals = tri_areas_flat(phi_flat, H, W)
    J = build_sparse_jacobian_T(phi_flat, H, W)
    return T_vals, J
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/research/strict_feasibility_2d/test_tri_linearize.py -v
```

Expected: 4 passed. The `test_jacobian_matches_adjoint_via_random_probes` test is the load-bearing one — it certifies that the explicit sparse Jacobian matches what the existing adjoint says it should be.

- [ ] **Step 5: Commit**

```bash
git add research/strict_feasibility_2d/algorithms/tri_linearize.py tests/research/strict_feasibility_2d/test_tri_linearize.py
git commit -m "Add tri_linearize: explicit sparse Jacobian for 2-tri LP"
```

---

## Task 5: highs_solver.py — L1 LP wrapper

**Files:**
- Create: `research/strict_feasibility_2d/algorithms/highs_solver.py`
- Test:   `tests/research/strict_feasibility_2d/test_highs_solver.py`

Builds the LP in epigraph form for `min ‖phi − phi_in‖₁` subject to
linearised triangle constraints. Decision vector is `[phi, t]` where `t`
is the L1 epigraph slack. Constraints are assembled as sparse blocks.

- [ ] **Step 1: Write the failing test**

```python
# tests/research/strict_feasibility_2d/test_highs_solver.py
import numpy as np
import scipy.sparse as sp

from research.strict_feasibility_2d.algorithms.highs_solver import solve_l1_lp_step


def test_no_constraint_returns_input():
    """With no triangle constraints, solution = phi_in (L1 = 0)."""
    H, W = 5, 5
    phi_in = np.zeros(2 * H * W)
    # Empty constraint set.
    T_lin = np.array([])
    J = sp.csr_matrix((0, 2 * H * W))
    phi_out, status = solve_l1_lp_step(
        phi_in_flat=phi_in,
        phi_lin_flat=phi_in,
        T_lin=T_lin,
        J_sparse=J,
        threshold=0.01,
    )
    assert status['success']
    np.testing.assert_allclose(phi_out, phi_in, atol=1e-9)


def test_lp_satisfies_linearized_constraint():
    """LP solution satisfies T_lin + J @ (phi - phi_lin) >= threshold."""
    rng = np.random.default_rng(0)
    H, W = 7, 7
    phi_in = rng.normal(scale=0.1, size=2 * H * W)
    phi_lin = phi_in.copy()

    from research.strict_feasibility_2d.algorithms.tri_linearize import linearize_T_2tri
    T_lin, J = linearize_T_2tri(phi_lin, H, W)
    phi_out, status = solve_l1_lp_step(
        phi_in_flat=phi_in,
        phi_lin_flat=phi_lin,
        T_lin=T_lin,
        J_sparse=J,
        threshold=0.01,
    )
    assert status['success']
    T_pred = T_lin + J @ (phi_out - phi_lin)
    assert np.all(T_pred >= 0.01 - 1e-6), f'worst slack = {(T_pred - 0.01).min():.4e}'


def test_trust_region_bounds_l_inf_step():
    """With trust_radius, ‖phi_out - phi_lin‖∞ <= trust_radius."""
    H, W = 5, 5
    phi_in = np.zeros(2 * H * W)
    phi_lin = phi_in.copy()
    from research.strict_feasibility_2d.algorithms.tri_linearize import linearize_T_2tri
    T_lin, J = linearize_T_2tri(phi_lin, H, W)
    phi_out, status = solve_l1_lp_step(
        phi_in_flat=phi_in,
        phi_lin_flat=phi_lin,
        T_lin=T_lin,
        J_sparse=J,
        threshold=0.01,
        trust_radius=0.1,
    )
    assert status['success']
    assert np.max(np.abs(phi_out - phi_lin)) <= 0.1 + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/research/strict_feasibility_2d/test_highs_solver.py -v
```

Expected: ImportError on `solve_l1_lp_step`.

- [ ] **Step 3: Implement `highs_solver.py`**

```python
# research/strict_feasibility_2d/algorithms/highs_solver.py
"""LP step solver: ``min ‖phi − phi_in‖_1`` subject to linearised
2-tri constraint + optional trust region. Backend is HiGHS via
``scipy.optimize.linprog(method='highs')``.

L1 epigraph reformulation
-------------------------
Decision vector: ``x = [phi (2HW), t (2HW)]``.

Objective:        ``min c^T x``,  ``c = [zeros(2HW), ones(2HW)]``.
L1 epigraph:      ``phi - t <= phi_in``  and  ``-phi - t <= -phi_in``.
Triangle (lin):   ``-J @ phi <= -threshold + T_lin - J @ phi_lin``.
Trust region:     ``phi - phi_lin in [-Δ, Δ]``  (only if ``trust_radius``).
``t`` bounds:     ``t >= 0`` (no upper bound).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def solve_l1_lp_step(
    *,
    phi_in_flat: np.ndarray,
    phi_lin_flat: np.ndarray,
    T_lin: np.ndarray,
    J_sparse: sp.spmatrix,
    threshold: float,
    trust_radius: float | None = None,
):
    """One LP iteration.

    Parameters
    ----------
    phi_in_flat : (2HW,) float64
        Anchor field for the L1 objective.
    phi_lin_flat : (2HW,) float64
        Linearisation point. Equal to ``phi_in_flat`` for ``lp_oneshot``;
        equal to the current SLP iterate for ``slp_iter``.
    T_lin : (K,) float64
        ``T(phi_lin_flat)`` from ``linearize_T_2tri``.
    J_sparse : (K, 2HW) sparse
        ``∂T/∂phi`` at ``phi_lin_flat``.
    threshold : float
        Lower bound on each triangle area.
    trust_radius : float or None
        L∞ box around ``phi_lin`` for SLP. ``None`` means unbounded
        (lp_oneshot path).

    Returns
    -------
    phi_out_flat : (2HW,)
    status : dict with keys ``success``, ``message``, ``fun``, ``status_code``, ``nit``.
    """
    n = phi_in_flat.size  # 2HW
    K = T_lin.size
    J_csr = J_sparse.tocsr() if not sp.isspmatrix_csr(J_sparse) else J_sparse

    # Objective: min sum(t) — zeros for phi, ones for t.
    c = np.concatenate([np.zeros(n), np.ones(n)])

    # --- A_ub @ x <= b_ub blocks ---
    blocks = []
    b_ub_blocks = []

    # 1) L1 epigraph upper:  phi - t <= phi_in
    A1 = sp.hstack([sp.eye(n), -sp.eye(n)])
    blocks.append(A1); b_ub_blocks.append(phi_in_flat)
    # 2) L1 epigraph lower: -phi - t <= -phi_in
    A2 = sp.hstack([-sp.eye(n), -sp.eye(n)])
    blocks.append(A2); b_ub_blocks.append(-phi_in_flat)

    # 3) Linearised triangle: -J @ phi <= -threshold + T_lin - J @ phi_lin
    if K > 0:
        rhs_tri = -threshold + T_lin - J_csr @ phi_lin_flat
        A3 = sp.hstack([-J_csr, sp.csr_matrix((K, n))])
        blocks.append(A3); b_ub_blocks.append(rhs_tri)

    # 4) Optional trust region: -Δ <= phi - phi_lin <= +Δ
    if trust_radius is not None:
        # phi - phi_lin <= +Δ
        A4 = sp.hstack([sp.eye(n), sp.csr_matrix((n, n))])
        blocks.append(A4); b_ub_blocks.append(phi_lin_flat + trust_radius)
        # -(phi - phi_lin) <= +Δ
        A5 = sp.hstack([-sp.eye(n), sp.csr_matrix((n, n))])
        blocks.append(A5); b_ub_blocks.append(-phi_lin_flat + trust_radius)

    A_ub = sp.vstack(blocks).tocsr()
    b_ub = np.concatenate(b_ub_blocks)

    # Bounds: phi unbounded, t >= 0.
    bounds = [(None, None)] * n + [(0.0, None)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    status = {
        'success': bool(result.success),
        'message': str(result.message),
        'fun': float(result.fun) if result.fun is not None else None,
        'status_code': int(result.status),
        'nit': int(getattr(result, 'nit', -1)),
    }
    if result.success:
        phi_out = result.x[:n]
    else:
        # On failure, return the linearisation point so callers can detect
        # via status['success'] without a NaN propagation.
        phi_out = phi_lin_flat.copy()
    return phi_out, status
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/research/strict_feasibility_2d/test_highs_solver.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add research/strict_feasibility_2d/algorithms/highs_solver.py tests/research/strict_feasibility_2d/test_highs_solver.py
git commit -m "Add highs_solver: L1 LP step via scipy/HiGHS"
```

---

## Task 6: lp_oneshot_2tri — single-LP from harmonic seed

**Files:**
- Create: `research/strict_feasibility_2d/algorithms/lp_direct_2tri.py`
- Test:   `tests/research/strict_feasibility_2d/test_lp_direct.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/strict_feasibility_2d/test_lp_direct.py
import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from research.strict_feasibility_2d.algorithms.lp_direct_2tri import lp_oneshot


def _bowtie_phi():
    phi = np.zeros((2, 7, 7), dtype=np.float64)
    phi[1, 3, 3] = +1.2
    phi[1, 3, 4] = -1.2
    return phi


def test_lp_oneshot_returns_phi_and_info():
    phi_in = _bowtie_phi()
    phi_out, info = lp_oneshot(phi_in, threshold=0.01)
    assert phi_out.shape == phi_in.shape
    assert phi_out.dtype == np.float64
    for k in ('seed', 'lp_status', 'L1_dev', 'final_min_T_exact', 'wall_s'):
        assert k in info, f'missing info key: {k}'


def test_lp_oneshot_strictly_improves_min_T_vs_input():
    phi_in = _bowtie_phi()
    T1_in, T2_in = _triangle_areas_2d(phi_in[0], phi_in[1])
    min_T_in = float(np.minimum(T1_in, T2_in).min())
    phi_out, info = lp_oneshot(phi_in, threshold=0.01)
    assert info['final_min_T_exact'] > min_T_in


def test_lp_oneshot_L1_is_smaller_than_harmonic_only():
    """LP should pull the seed back toward phi_in -- L1 must drop."""
    from dvfopt.core.wallbreakers import harmonic_extension_2d
    phi_in = _bowtie_phi()
    seed = harmonic_extension_2d(phi_in, threshold=0.01)
    seed_L1 = float(np.abs(seed - phi_in).sum())
    phi_out, info = lp_oneshot(phi_in, threshold=0.01)
    out_L1 = float(np.abs(phi_out - phi_in).sum())
    assert out_L1 <= seed_L1 + 1e-9, f'LP L1 {out_L1} > seed L1 {seed_L1}'
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/research/strict_feasibility_2d/test_lp_direct.py::test_lp_oneshot_returns_phi_and_info -v
```

Expected: ImportError on `lp_oneshot`.

- [ ] **Step 3: Implement `lp_oneshot` in `lp_direct_2tri.py`**

```python
# research/strict_feasibility_2d/algorithms/lp_direct_2tri.py
"""LP-direct strategies for strict 2-tri feasibility.

Two variants:

* :func:`lp_oneshot` — single LP linearised around a feasible harmonic seed.
  May leave a small residual fold at exact eval due to linearisation error.
* :func:`slp_iter`  — sequential LP loop with adaptive trust region.
  Iterates until exact-T feasibility holds. Guaranteed feasible at
  termination (or returns the best iterate with a non-converged flag).

Both minimise ``‖phi - phi_in‖_1`` and return ``(phi_out, info)``.
"""
from __future__ import annotations

import time

import numpy as np

from research.strict_feasibility_2d.algorithms.highs_solver import solve_l1_lp_step
from research.strict_feasibility_2d.algorithms.tri_linearize import linearize_T_2tri


def _exact_min_T(phi_2hw: np.ndarray) -> float:
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    return float(np.minimum(T1, T2).min())


def _flatten(phi_2hw: np.ndarray) -> np.ndarray:
    return np.concatenate([phi_2hw[0].ravel(), phi_2hw[1].ravel()])


def _unflatten(phi_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    HW = H * W
    return np.stack([phi_flat[:HW].reshape(H, W), phi_flat[HW:].reshape(H, W)])


def _harmonic_seed(phi_in_2hw: np.ndarray, threshold: float) -> np.ndarray:
    """Feasible-by-construction seed via Laplacian extension of fold cores."""
    from dvfopt.core.wallbreakers import harmonic_extension_2d
    return harmonic_extension_2d(phi_in_2hw, threshold=threshold)


def lp_oneshot(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    seed: str = 'harmonic',
):
    """Single-LP linearised around ``seed``.

    Parameters
    ----------
    phi_in_2hw : (2, H, W) float64
    threshold : float
    seed : {'harmonic', 'zero'}
        ``'harmonic'`` uses the Laplacian-extension feasible seed (default).
        ``'zero'`` linearises around ``phi = 0`` — used in ablation runs.

    Returns
    -------
    phi_out_2hw : (2, H, W) float64
    info : dict
    """
    t0 = time.time()
    H, W = phi_in_2hw.shape[1:]
    if seed == 'harmonic':
        seed_phi = _harmonic_seed(phi_in_2hw, threshold)
    elif seed == 'zero':
        seed_phi = np.zeros_like(phi_in_2hw)
    else:
        raise ValueError(f'unknown seed: {seed!r}')

    phi_in_flat = _flatten(phi_in_2hw)
    phi_lin_flat = _flatten(seed_phi)
    T_lin, J = linearize_T_2tri(phi_lin_flat, H, W)
    phi_out_flat, status = solve_l1_lp_step(
        phi_in_flat=phi_in_flat,
        phi_lin_flat=phi_lin_flat,
        T_lin=T_lin,
        J_sparse=J,
        threshold=threshold,
        trust_radius=None,
    )
    phi_out = _unflatten(phi_out_flat, H, W)
    info = {
        'seed': seed,
        'seed_min_T_exact': _exact_min_T(seed_phi),
        'final_min_T_exact': _exact_min_T(phi_out),
        'L1_dev': float(np.abs(phi_out - phi_in_2hw).sum()),
        'lp_status': status,
        'wall_s': time.time() - t0,
    }
    return phi_out, info
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/research/strict_feasibility_2d/test_lp_direct.py -v -k oneshot
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add research/strict_feasibility_2d/algorithms/lp_direct_2tri.py tests/research/strict_feasibility_2d/test_lp_direct.py
git commit -m "Add lp_oneshot_2tri: single-LP from harmonic seed"
```

---

## Task 7: slp_iter_2tri — sequential LP loop

**Files:**
- Modify: `research/strict_feasibility_2d/algorithms/lp_direct_2tri.py` (add `slp_iter`)
- Modify: `tests/research/strict_feasibility_2d/test_lp_direct.py` (add slp_iter tests)

- [ ] **Step 1: Write failing tests for `slp_iter`**

Append to `tests/research/strict_feasibility_2d/test_lp_direct.py`:

```python
from research.strict_feasibility_2d.algorithms.lp_direct_2tri import slp_iter


def test_slp_iter_returns_phi_and_info():
    phi_in = _bowtie_phi()
    phi_out, info = slp_iter(phi_in, threshold=0.01)
    assert phi_out.shape == phi_in.shape
    for k in ('iters', 'L1_dev', 'final_min_T_exact', 'converged', 'wall_s', 'trust_radius_final'):
        assert k in info, f'missing info key: {k}'


def test_slp_iter_strictly_feasible_on_bowtie():
    """The whole point: at termination, min(T1, T2) >= threshold - safety_tol."""
    phi_in = _bowtie_phi()
    phi_out, info = slp_iter(phi_in, threshold=0.01)
    # safety_tol = 1e-5
    assert info['final_min_T_exact'] >= 0.01 - 1e-5, info['final_min_T_exact']


def test_slp_iter_L1_le_lp_oneshot_L1():
    """Iteration should not increase L1 vs the one-shot baseline on feasible cases."""
    phi_in = _bowtie_phi()
    _, info_one = lp_oneshot(phi_in, threshold=0.01)
    _, info_slp = slp_iter(phi_in, threshold=0.01)
    # SLP is allowed to be slightly worse if lp_oneshot is infeasible at exact eval;
    # but on bowtie both should converge to the same neighbourhood.
    # Tolerate a 5% gap for numerical reasons.
    assert info_slp['L1_dev'] <= info_one['L1_dev'] * 1.05 + 1e-6


def test_slp_iter_terminates_within_max_iter():
    phi_in = _bowtie_phi()
    _, info = slp_iter(phi_in, threshold=0.01, max_iter=20)
    assert info['iters'] <= 20
    assert info['converged'] or info['iters'] == 20
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/research/strict_feasibility_2d/test_lp_direct.py -v -k slp_iter
```

Expected: ImportError on `slp_iter`.

- [ ] **Step 3: Implement `slp_iter` in the same file**

Append to `research/strict_feasibility_2d/algorithms/lp_direct_2tri.py`:

```python
def slp_iter(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    safety_tol: float = 1e-5,
    trust_radius_0: float = 0.5,
    max_iter: int = 20,
    ftol: float = 1e-6,
    trust_grow: float = 1.5,
    trust_shrink: float = 0.5,
):
    """Sequential LP loop with adaptive trust region.

    Termination: returns when either
      a) max-norm step < ``ftol`` AND exact ``min_T >= threshold - safety_tol``  (converged), or
      b) ``max_iter`` reached  (returns the best iterate that satisfies the
         exact feasibility check; ``converged=False`` if none does).

    On exact-T infeasibility after an LP step, the trust radius is
    halved and the iterate is rejected. On feasibility + step at the
    trust boundary, the radius is grown by ``trust_grow``.
    """
    t0 = time.time()
    H, W = phi_in_2hw.shape[1:]
    HW = H * W
    phi_in_flat = _flatten(phi_in_2hw)

    # Seed.
    seed_phi = _harmonic_seed(phi_in_2hw, threshold)
    phi_cur_flat = _flatten(seed_phi)
    trust_radius = float(trust_radius_0)

    best_phi_flat = phi_cur_flat.copy()
    best_L1 = float(np.abs(seed_phi - phi_in_2hw).sum())
    best_feasible = _exact_min_T(seed_phi) >= threshold - safety_tol

    iters = 0
    converged = False
    statuses = []

    for it in range(max_iter):
        iters = it + 1
        T_lin, J = linearize_T_2tri(phi_cur_flat, H, W)
        phi_new_flat, status = solve_l1_lp_step(
            phi_in_flat=phi_in_flat,
            phi_lin_flat=phi_cur_flat,
            T_lin=T_lin,
            J_sparse=J,
            threshold=threshold,
            trust_radius=trust_radius,
        )
        statuses.append(status)
        if not status['success']:
            # LP infeasible at this trust radius: shrink and retry.
            trust_radius *= trust_shrink
            if trust_radius < 1e-8:
                break
            continue

        phi_new_2hw = _unflatten(phi_new_flat, H, W)
        exact_min = _exact_min_T(phi_new_2hw)
        new_L1 = float(np.abs(phi_new_2hw - phi_in_2hw).sum())

        if exact_min < threshold - safety_tol:
            # Linearisation error: shrink trust region, reject step.
            trust_radius *= trust_shrink
            if trust_radius < 1e-8:
                break
            continue

        # Accept step.
        step_inf = float(np.max(np.abs(phi_new_flat - phi_cur_flat)))
        at_boundary = step_inf >= 0.99 * trust_radius
        phi_cur_flat = phi_new_flat
        if new_L1 <= best_L1 + 1e-12:
            best_phi_flat = phi_cur_flat.copy()
            best_L1 = new_L1
            best_feasible = True
        if step_inf < ftol:
            converged = True
            break
        if at_boundary:
            trust_radius *= trust_grow

    phi_out = _unflatten(best_phi_flat, H, W)
    info = {
        'iters': iters,
        'converged': converged,
        'L1_dev': best_L1,
        'final_min_T_exact': _exact_min_T(phi_out),
        'feasible_at_exact_eval': best_feasible,
        'lp_statuses': statuses,
        'trust_radius_final': trust_radius,
        'wall_s': time.time() - t0,
    }
    return phi_out, info
```

- [ ] **Step 4: Run all `test_lp_direct.py` tests**

```bash
pytest tests/research/strict_feasibility_2d/test_lp_direct.py -v
```

Expected: 7 passed (3 oneshot + 4 slp_iter).

- [ ] **Step 5: Commit**

```bash
git add research/strict_feasibility_2d/algorithms/lp_direct_2tri.py tests/research/strict_feasibility_2d/test_lp_direct.py
git commit -m "Add slp_iter_2tri: sequential LP with adaptive trust region"
```

---

## Task 8: Adversarial synthetic cases

**Files:**
- Create: `research/strict_feasibility_2d/worst_cases/_build_adversarial.py`
- Modify: `research/strict_feasibility_2d/worst_cases/catalog.md` (add the two new entries)
- Modify: `research/strict_feasibility_2d/worst_cases/_load.py` (include adversarial dir in synthetic load)

- [ ] **Step 1: Write `_build_adversarial.py`**

```python
# research/strict_feasibility_2d/worst_cases/_build_adversarial.py
"""Generate hand-designed adversarial synthetic cases.

Run from the repo root:

    python research/strict_feasibility_2d/worst_cases/_build_adversarial.py

Outputs:

* ``synthetic/dense_bowtie_cluster_15x15.npz`` — 15×15 grid with a
  3×3 cluster of bowtie pairs in the centre. ~18 folded cells in
  a single connected component, harder than the canonical bowtie.
* ``synthetic/tiny_margin_10x10.npz`` — 10×10 grid where every cell
  is folded but only by a small margin (min_T ~ -0.01). Stresses
  the linearisation since LP steps must be small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

OUTDIR = Path(__file__).parent / 'synthetic'


def _save_npz(name: str, phi_2hw: np.ndarray, title: str):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    n_neg = int((np.minimum(T1, T2) <= 0).sum())
    min_T = float(np.minimum(T1, T2).min())
    path = OUTDIR / f'{name}.npz'
    np.savez(
        path,
        phi=phi_2hw.astype(np.float64),
        msample=np.empty((0, 3), dtype=np.float64),
        fsample=np.empty((0, 3), dtype=np.float64),
        init_n_neg=np.int64(n_neg),
        init_min_T=np.float64(min_T),
        shape=np.asarray(phi_2hw.shape[1:], dtype=np.int64),
        title=np.asarray(title),
        key=np.asarray(name),
    )
    print(f'  {name:<32s} {phi_2hw.shape}  n_neg={n_neg:3d}  min_T={min_T:+.4f}')


def build_dense_bowtie_cluster_15x15() -> np.ndarray:
    H, W = 15, 15
    phi = np.zeros((2, H, W), dtype=np.float64)
    # A 3-cell-wide horizontal band of alternating crossings.
    for r in (6, 7, 8):
        for c in (5, 7, 9):
            phi[1, r, c]     = +1.2  # dx
            phi[1, r, c + 1] = -1.2
    return phi


def build_tiny_margin_10x10() -> np.ndarray:
    H, W = 10, 10
    phi = np.zeros((2, H, W), dtype=np.float64)
    # Small symmetric vertical shears that put every cell just barely below 0.
    phi[0, :, ::2] = +0.55  # dy on even columns
    phi[0, :, 1::2] = -0.55  # dy on odd columns
    return phi


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f'Writing adversarial cases to {OUTDIR}')
    _save_npz(
        'dense_bowtie_cluster_15x15',
        build_dense_bowtie_cluster_15x15(),
        'Dense bowtie cluster (15x15)',
    )
    _save_npz(
        'tiny_margin_10x10',
        build_tiny_margin_10x10(),
        'Tiny-margin alternating shear (10x10)',
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the builder**

```bash
python research/strict_feasibility_2d/worst_cases/_build_adversarial.py
```

Expected stdout:
```
Writing adversarial cases to .../strict_feasibility_2d/worst_cases/synthetic
  dense_bowtie_cluster_15x15      (2, 15, 15)  n_neg= 18  min_T=-0.7000
  tiny_margin_10x10                (2, 10, 10)  n_neg=...   min_T=...
```

(Adjust `tiny_margin_10x10` parameters in the builder if the resulting min_T isn't in the intended band — verification only; the exact numbers don't matter for downstream tests.)

- [ ] **Step 3: Update `_load.py` to include the adversarial dir**

In `research/strict_feasibility_2d/worst_cases/_load.py`, change `load_synthetic_canonical` to also scan the local `synthetic/` directory:

```python
_ADVERSARIAL_DIR = Path(__file__).parent / 'synthetic'


def load_synthetic_canonical():
    """Yield ``(name, phi_2hw, meta)`` for every NPZ in
    ``data/dvfs/canonical_2tri_2d/`` and the local ``synthetic/`` dir."""
    out = []
    for d in (_CANONICAL_DIR, _ADVERSARIAL_DIR):
        if not d.exists():
            continue
        for path in sorted(d.glob('*.npz')):
            with np.load(path, allow_pickle=False) as data:
                phi = data['phi'].astype(np.float64)
            out.append((path.stem, phi, _stats(phi)))
    return out
```

- [ ] **Step 4: Append adversarial rows to `catalog.md`**

```markdown
## Synthetic — adversarial (built by `_build_adversarial.py`)
| Case | Shape | init n_neg | Purpose |
|---|---|---|---|
| `dense_bowtie_cluster_15x15` | 15×15 | 18 | Dense single-cluster bowtie field |
| `tiny_margin_10x10` | 10×10 | many | Stress linearisation: every cell barely infeasible |
```

- [ ] **Step 5: Update the worst_cases test to include adversarial cases**

In `tests/research/strict_feasibility_2d/test_worst_cases.py`, add:

```python
def test_load_synthetic_includes_adversarial():
    cases = _load.load_synthetic_canonical()
    keys = {name for name, phi, meta in cases}
    assert 'dense_bowtie_cluster_15x15' in keys
    assert 'tiny_margin_10x10' in keys
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/research/strict_feasibility_2d/test_worst_cases.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add research/strict_feasibility_2d/worst_cases/ tests/research/strict_feasibility_2d/test_worst_cases.py
git commit -m "Add adversarial synthetic worst cases"
```

---

## Task 9: Comparison harness — `run_method` dispatch

**Files:**
- Create: `research/strict_feasibility_2d/runners/_compare.py`
- Test:   `tests/research/strict_feasibility_2d/test_comparison.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/strict_feasibility_2d/test_comparison.py
import numpy as np

from research.strict_feasibility_2d.runners._compare import (
    METHOD_NAMES,
    run_method,
)


def _bowtie_phi():
    phi = np.zeros((2, 7, 7), dtype=np.float64)
    phi[1, 3, 3] = +1.2
    phi[1, 3, 4] = -1.2
    return phi


def test_method_names_include_all_seven():
    expected = {
        'harmonic_only', 'm10', 'm14', 'm14_schwarz',
        'cluster_pipeline', 'lp_oneshot', 'slp_iter',
    }
    assert set(METHOD_NAMES) >= expected


def test_run_method_lp_oneshot_returns_expected_keys():
    phi_in = _bowtie_phi()
    rec = run_method('lp_oneshot', phi_in)
    for k in (
        'method', 'phi_out', 'init_n_neg_2tri', 'init_min_T',
        'final_n_neg_2tri', 'final_min_T', 'feasible',
        'L1_dev', 'L2_dev', 'Linf_dev', 'wall_s',
    ):
        assert k in rec, f'missing key {k!r}'
    assert rec['method'] == 'lp_oneshot'
    assert rec['phi_out'].shape == phi_in.shape


def test_run_method_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        run_method('not_a_real_method', _bowtie_phi())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/research/strict_feasibility_2d/test_comparison.py -v
```

Expected: ImportError on `run_method`.

- [ ] **Step 3: Implement `_compare.py`**

```python
# research/strict_feasibility_2d/runners/_compare.py
"""Per-method dispatch + uniform metric record.

``run_method(name, phi_2hw) -> dict`` runs ``name`` on ``phi_2hw`` and
returns a dict with all metrics specified in the design spec.
"""
from __future__ import annotations

import time

import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from research.strict_feasibility_2d.algorithms.lp_direct_2tri import (
    lp_oneshot,
    slp_iter,
)

THRESHOLD = 0.01
SAFETY_TOL = 1e-5

METHOD_NAMES = (
    'harmonic_only',
    'm10',
    'm14',
    'm14_schwarz',
    'cluster_pipeline',
    'lp_oneshot',
    'slp_iter',
)


def _stats(phi_2hw: np.ndarray):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    T_min = np.minimum(T1, T2)
    return {
        'n_neg_2tri': int((T_min <= 0).sum()),
        'min_T': float(T_min.min()),
    }


def _dispatch(name: str, phi_2hw: np.ndarray):
    """Return ``(phi_out, extra_info_dict)``."""
    if name == 'harmonic_only':
        from dvfopt.core.wallbreakers import harmonic_extension_2d
        phi_out = harmonic_extension_2d(phi_2hw, threshold=THRESHOLD)
        return phi_out, {}
    if name == 'm10':
        from dvfopt import iterative_2d_tri_harmonic_polished
        phi_out = iterative_2d_tri_harmonic_polished(phi_2hw, threshold=THRESHOLD, verbose=0)
        return phi_out, {}
    if name == 'm14':
        from dvfopt import iterative_2d_tri_refine_repair
        phi_out = iterative_2d_tri_refine_repair(phi_2hw, threshold=THRESHOLD, verbose=0)
        return phi_out, {}
    if name == 'm14_schwarz':
        from dvfopt import iterative_2d_tri_refine_repair_schwarz
        phi_out = iterative_2d_tri_refine_repair_schwarz(phi_2hw, threshold=THRESHOLD, verbose=0)
        return phi_out, {}
    if name == 'cluster_pipeline':
        # Imported lazily; lives in notebooks/manuscript. Wrap to (2, H, W).
        from notebooks.manuscript import _run_2d_clusters as cluster_mod
        phi_out, _info = cluster_mod.correct_slice(phi_2hw, threshold=THRESHOLD)
        return phi_out, {}
    if name == 'lp_oneshot':
        phi_out, info = lp_oneshot(phi_2hw, threshold=THRESHOLD)
        return phi_out, info
    if name == 'slp_iter':
        phi_out, info = slp_iter(phi_2hw, threshold=THRESHOLD)
        return phi_out, info
    raise ValueError(f'unknown method: {name!r} (known: {METHOD_NAMES})')


def run_method(name: str, phi_in_2hw: np.ndarray) -> dict:
    """Run ``name`` on ``phi_in_2hw`` and return a metrics record."""
    init = _stats(phi_in_2hw)
    t0 = time.time()
    try:
        phi_out, extra = _dispatch(name, phi_in_2hw)
        error = None
    except Exception as exc:
        phi_out = phi_in_2hw.copy()
        extra = {}
        error = f'{type(exc).__name__}: {exc}'
    wall = time.time() - t0
    final = _stats(phi_out)
    diff = phi_out.astype(np.float64) - phi_in_2hw.astype(np.float64)
    return {
        'method': name,
        'phi_out': phi_out,
        'init_n_neg_2tri': init['n_neg_2tri'],
        'init_min_T': init['min_T'],
        'final_n_neg_2tri': final['n_neg_2tri'],
        'final_min_T': final['min_T'],
        'feasible': final['n_neg_2tri'] == 0 and final['min_T'] >= THRESHOLD - SAFETY_TOL,
        'L1_dev': float(np.abs(diff).sum()),
        'L2_dev': float(np.linalg.norm(diff)),
        'Linf_dev': float(np.max(np.abs(diff))),
        'wall_s': wall,
        'error': error,
        'extra': extra,
    }
```

> **Note on `cluster_pipeline`:** If `notebooks/manuscript/_run_2d_clusters.py` doesn't expose a `correct_slice(phi, *, threshold)` function with that exact signature, this dispatch will raise on first call. Read the file before relying on this path — if the signature differs, either adapt the call here or skip `cluster_pipeline` from runs by removing it from `METHOD_NAMES` for the first iteration of the comparison; revisit in Task 14.

- [ ] **Step 4: Run tests**

```bash
pytest tests/research/strict_feasibility_2d/test_comparison.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add research/strict_feasibility_2d/runners/_compare.py tests/research/strict_feasibility_2d/test_comparison.py
git commit -m "Add comparison harness with run_method dispatch"
```

---

## Task 10: Synthetic batch runner

**Files:**
- Create: `research/strict_feasibility_2d/runners/_run_lp_synthetic.py`

This is a script, not a library. Verified by running it and inspecting the output CSV.

- [ ] **Step 1: Write `_run_lp_synthetic.py`**

```python
# research/strict_feasibility_2d/runners/_run_lp_synthetic.py
"""Batch comparison on every synthetic worst case.

Writes:
    runners/output/comparison_synthetic.csv
    runners/output/corrected/<case>_<method>.npz
"""
from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from research.strict_feasibility_2d.runners._compare import METHOD_NAMES, run_method
from research.strict_feasibility_2d.worst_cases._load import load_synthetic_canonical

OUTDIR = _HERE / 'output'
CORR_DIR = OUTDIR / 'corrected'

CSV_FIELDS = [
    'case_id', 'method', 'shape',
    'init_n_neg_2tri', 'init_min_T',
    'final_n_neg_2tri', 'final_min_T',
    'feasible', 'L1_dev', 'L2_dev', 'Linf_dev',
    'wall_s', 'error',
]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / 'comparison_synthetic.csv'

    cases = load_synthetic_canonical()
    print(f'Found {len(cases)} synthetic cases, {len(METHOD_NAMES)} methods.', flush=True)

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for case_id, phi_in, meta in cases:
            print(f'\n=== {case_id}  shape={meta["shape"]}  init_n_neg={meta["init_n_neg"]} ===', flush=True)
            for method in METHOD_NAMES:
                try:
                    rec = run_method(method, phi_in)
                except Exception:
                    print(f'  {method:<18s} UNCAUGHT — {traceback.format_exc(limit=2)}', flush=True)
                    continue
                row = {k: rec[k] for k in CSV_FIELDS if k in rec}
                row['case_id'] = case_id
                row['shape'] = f'{meta["shape"][0]}x{meta["shape"][1]}'
                writer.writerow(row)
                fh.flush()
                flag = 'OK ' if rec['feasible'] else 'INF'
                err = f'   err={rec["error"]}' if rec['error'] else ''
                print(
                    f'  {method:<18s} {flag}  n_neg={rec["final_n_neg_2tri"]:3d}  '
                    f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:.3f}  '
                    f'({rec["wall_s"]:.2f}s){err}',
                    flush=True,
                )
                # Save corrected phi as NPZ for later inspection.
                np.savez(
                    CORR_DIR / f'{case_id}_{method}.npz',
                    phi_out=rec['phi_out'].astype(np.float64),
                )

    print(f'\nWrote {out_csv}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the script**

```bash
python research/strict_feasibility_2d/runners/_run_lp_synthetic.py
```

Expected behaviour:
- Iterates over ~9 synthetic cases (7 canonical + 2 adversarial).
- Runs each of the 7 methods per case (`cluster_pipeline` may error and skip — that's acceptable for this task).
- Writes `runners/output/comparison_synthetic.csv` with one row per (case, method) that completed.
- Per case, `lp_oneshot` and `slp_iter` rows should report `feasible=True` on most cases (linearisation error may cause `lp_oneshot=False` on adversarial cases — that's a documented expected outcome).

- [ ] **Step 3: Sanity-check the CSV**

```bash
head -5 research/strict_feasibility_2d/runners/output/comparison_synthetic.csv
# Confirm header + first few rows
wc -l research/strict_feasibility_2d/runners/output/comparison_synthetic.csv
# Expect roughly 7 * 9 + 1 = 64 lines (give or take cluster_pipeline failures)
```

- [ ] **Step 4: Commit script + CSV**

The CSV is a real research artifact — commit it.

```bash
git add research/strict_feasibility_2d/runners/_run_lp_synthetic.py
git add research/strict_feasibility_2d/runners/output/comparison_synthetic.csv
git commit -m "Run synthetic bake-off + commit comparison_synthetic.csv"
```

(Don't commit `runners/output/corrected/` — large NPZ files. Add them to `.gitignore` if not already.)

- [ ] **Step 5: Add corrected/ to .gitignore**

Append to `.gitignore`:

```
# Strict-feasibility research artifacts (large per-case corrected fields).
research/strict_feasibility_2d/runners/output/corrected/
```

```bash
git add .gitignore
git commit -m "Gitignore strict-feasibility corrected NPZ outputs"
```

---

## Task 11: B0039 batch runner (z=12 first, expand later)

**Files:**
- Create: `research/strict_feasibility_2d/runners/_run_lp_b0039.py`

- [ ] **Step 1: Write `_run_lp_b0039.py`**

```python
# research/strict_feasibility_2d/runners/_run_lp_b0039.py
"""Batch comparison on selected B0039 slices.

Default slice set: z=12 (manuscript-canonical hard case) + a handful of
others to probe scale. Empirical-worst slice discovery comes after these
results land.

Writes:
    runners/output/comparison_b0039.csv
    runners/output/corrected/<case>_<method>.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from research.strict_feasibility_2d.runners._compare import METHOD_NAMES, run_method
from research.strict_feasibility_2d.worst_cases._load import load_b0039_slice
from research.strict_feasibility_2d.runners._run_lp_synthetic import CSV_FIELDS

OUTDIR = _HERE / 'output'
CORR_DIR = OUTDIR / 'corrected'

DEFAULT_SLICES = (12, 100, 200, 300, 400)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--slices', type=int, nargs='+', default=list(DEFAULT_SLICES),
        help='Z-slice indices to run.',
    )
    p.add_argument(
        '--methods', type=str, nargs='+', default=list(METHOD_NAMES),
        help='Subset of methods to run (default: all).',
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / 'comparison_b0039.csv'

    print(f'Slices: {args.slices}', flush=True)
    print(f'Methods: {args.methods}', flush=True)

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for z in args.slices:
            try:
                case_id, phi_in, meta = load_b0039_slice(z)
            except (IndexError, FileNotFoundError) as exc:
                print(f'\n[skip] z={z}: {exc}', flush=True)
                continue
            print(f'\n=== {case_id}  shape={meta["shape"]}  init_n_neg={meta["init_n_neg"]} ===', flush=True)
            for method in args.methods:
                try:
                    rec = run_method(method, phi_in)
                except Exception:
                    print(f'  {method:<18s} UNCAUGHT — {traceback.format_exc(limit=2)}', flush=True)
                    continue
                row = {k: rec[k] for k in CSV_FIELDS if k in rec}
                row['case_id'] = case_id
                row['shape'] = f'{meta["shape"][0]}x{meta["shape"][1]}'
                writer.writerow(row)
                fh.flush()
                flag = 'OK ' if rec['feasible'] else 'INF'
                err = f'   err={rec["error"]}' if rec['error'] else ''
                print(
                    f'  {method:<18s} {flag}  n_neg={rec["final_n_neg_2tri"]:4d}  '
                    f'min_T={rec["final_min_T"]:+.4f}  L1={rec["L1_dev"]:.1f}  '
                    f'({rec["wall_s"]:.1f}s){err}',
                    flush=True,
                )
                np.savez(
                    CORR_DIR / f'{case_id}_{method}.npz',
                    phi_out=rec['phi_out'].astype(np.float64),
                )

    print(f'\nWrote {out_csv}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-run on z=12 alone (fast subset to verify correctness)**

```bash
python research/strict_feasibility_2d/runners/_run_lp_b0039.py \
    --slices 12 \
    --methods harmonic_only lp_oneshot slp_iter
```

Expected: completes within a few minutes; `slp_iter` row in CSV reports `feasible=True` (or close — `final_min_T` should be at or near 0.01).

- [ ] **Step 3: Full B0039 run (default slice set)**

```bash
python research/strict_feasibility_2d/runners/_run_lp_b0039.py
```

This is slower (~tens of minutes depending on hardware). May fail on `cluster_pipeline` row if the signature mismatch from Task 9 hasn't been resolved — the row will be skipped with an error logged; other rows still complete.

- [ ] **Step 4: Commit the script + CSV**

```bash
git add research/strict_feasibility_2d/runners/_run_lp_b0039.py
git add research/strict_feasibility_2d/runners/output/comparison_b0039.csv
git commit -m "Run B0039 bake-off (z=12 + default set) + commit CSV"
```

---

## Task 12: Analysis notebook 01 — baseline L1 gap

**Files:**
- Create: `research/strict_feasibility_2d/analysis/01_baseline_l1_gap.ipynb`

Notebook content described as a cell-by-cell outline. Build it with `jupyter notebook` or by writing a `_build_01.py` cell-builder if you prefer (the existing repo pattern for builders is in `notebooks/experiments/`).

- [ ] **Step 1: Create the notebook with these cells in order**

Cell 1 (markdown):

```markdown
# 01 — Baseline L1 gap

Loads `comparison_synthetic.csv` + `comparison_b0039.csv`, builds the
headline table per the spec, and visualises per-case L1 deviation by method.

**Strict-feasibility rule:** rows with `feasible=False` are EXCLUDED from
the L1 ranking. No averaging over infeasible solutions.
```

Cell 2 (code):

```python
from pathlib import Path
import pandas as pd

_HERE = Path.cwd()  # assumes notebook is opened from its dir
OUTDIR = _HERE.parent / 'runners' / 'output'

df_synth = pd.read_csv(OUTDIR / 'comparison_synthetic.csv')
df_b0039 = pd.read_csv(OUTDIR / 'comparison_b0039.csv')
df = pd.concat([df_synth, df_b0039], ignore_index=True)
print(f'Loaded {len(df)} rows across {df.case_id.nunique()} cases and {df.method.nunique()} methods.')
df.head()
```

Cell 3 (code) — feasibility summary:

```python
# Per-method feasibility counts.
summary = df.groupby('method').agg(
    n_runs=('feasible', 'size'),
    n_feasible=('feasible', 'sum'),
).assign(
    feasible_frac=lambda d: d['n_feasible'] / d['n_runs'],
)
summary.sort_values('feasible_frac', ascending=False)
```

Cell 4 (code) — headline table (L1, feasible-only):

```python
df_feas = df[df['feasible']].copy()
pivot_L1 = df_feas.pivot_table(
    index='case_id', columns='method', values='L1_dev', aggfunc='first',
).round(4)

# Add an `L1_lp_oneshot` baseline column and the per-method gap.
if 'lp_oneshot' in pivot_L1.columns:
    pivot_L1['_gap_m14'] = pivot_L1.get('m14', float('nan')) - pivot_L1['lp_oneshot']
pivot_L1
```

Cell 5 (code) — wall-time table:

```python
pivot_wall = df.pivot_table(
    index='case_id', columns='method', values='wall_s', aggfunc='first',
).round(2)
pivot_wall
```

Cell 6 (markdown):

```markdown
## Reading the headline table

* Columns are sorted alphabetically. `lp_oneshot` and `slp_iter` are
  the new methods.
* A blank cell means that (case, method) was infeasible at exact eval
  and excluded — see the feasibility summary cell above for which method
  failed where.
* `_gap_m14` = `L1_dev(m14) - L1_dev(lp_oneshot)` — positive numbers
  mean LP wins, negative means M14 wins.
```

Cell 7 (code) — per-case bar chart:

```python
import matplotlib.pyplot as plt

cases = pivot_L1.index.tolist()
methods = [m for m in pivot_L1.columns if not m.startswith('_')]

fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(cases)), 5))
x = range(len(cases))
n = len(methods)
w = 0.8 / n
for i, m in enumerate(methods):
    vals = pivot_L1[m].values
    ax.bar([xi + (i - n/2) * w for xi in x], vals, width=w, label=m)
ax.set_xticks(list(x))
ax.set_xticklabels(cases, rotation=30, ha='right')
ax.set_ylabel('L1 deviation from input')
ax.set_title('L1 deviation by method (feasible runs only)')
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig(_HERE / 'l1_per_case.png', dpi=150)
plt.show()
```

- [ ] **Step 2: Run the notebook end-to-end** (`Kernel → Restart & Run All`).

Confirm:
* `df_synth` loads with the row count from Task 10 (~9 cases × 7 methods minus any error rows).
* `df_b0039` loads with the row count from Task 11 (5 slices × 7 methods).
* `pivot_L1` shows finite numbers in `lp_oneshot` / `slp_iter` columns for most cases.
* `l1_per_case.png` saves to the analysis dir.

- [ ] **Step 3: Commit**

```bash
git add research/strict_feasibility_2d/analysis/01_baseline_l1_gap.ipynb
git add research/strict_feasibility_2d/analysis/l1_per_case.png
git commit -m "Analysis 01: baseline L1 gap notebook + summary chart"
```

---

## Task 13: Analysis notebook 02 — LP optimum certification

**Files:**
- Create: `research/strict_feasibility_2d/analysis/02_lp_certifies_optimum.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Cell 1 (markdown):

```markdown
# 02 — LP optimum vs every baseline

For each case, the LP/SLP `L1_dev` is (by construction of the LP)
the L1 minimum within the orientation-fixed feasibility set. This
notebook quantifies how much L1 each existing method leaves on the
table, per case, and answers:

1. Is the LP/SLP route Pareto-best (strict feasibility AND smallest L1)?
2. Where do existing methods fall short, and by how much?
3. Does iteration matter? (`slp_iter` vs `lp_oneshot` L1 gap.)
```

Cell 2 (code) — reuse the loader from notebook 01:

```python
from pathlib import Path
import pandas as pd

_HERE = Path.cwd()
OUTDIR = _HERE.parent / 'runners' / 'output'
df = pd.concat([
    pd.read_csv(OUTDIR / 'comparison_synthetic.csv'),
    pd.read_csv(OUTDIR / 'comparison_b0039.csv'),
], ignore_index=True)
df.head(2)
```

Cell 3 (code) — gap-table:

```python
# Strict feasibility row filter for the L1 ranking.
df_feas = df[df['feasible']].copy()

ref = df_feas[df_feas['method'] == 'slp_iter'][['case_id', 'L1_dev']].rename(
    columns={'L1_dev': 'L1_slp'}
)
joined = df_feas.merge(ref, on='case_id', how='left')
joined['L1_gap_vs_slp'] = joined['L1_dev'] - joined['L1_slp']
joined['L1_gap_pct'] = 100 * joined['L1_gap_vs_slp'] / joined['L1_slp']

# Wide pivot.
gap = joined.pivot_table(
    index='case_id', columns='method',
    values='L1_gap_pct', aggfunc='first',
).round(1)
gap
```

Cell 4 (markdown):

```markdown
**Reading:** entries are `100 * (L1_method - L1_slp) / L1_slp` —
percentage L1 excess vs SLP per case. Positive = method leaves L1
on the table. 0 = matches SLP. Negative = method beats SLP (would
indicate a bug or that SLP didn't converge).
```

Cell 5 (code) — per-case dot plot:

```python
import matplotlib.pyplot as plt
import numpy as np

cases = gap.index.tolist()
methods = [m for m in gap.columns if m != 'slp_iter']
fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(cases)), 5))
for i, m in enumerate(methods):
    y = gap[m].values
    x = np.arange(len(cases)) + 0.1 * (i - len(methods)/2)
    ax.scatter(x, y, s=60, label=m)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xticks(np.arange(len(cases)))
ax.set_xticklabels(cases, rotation=30, ha='right')
ax.set_ylabel('L1 excess vs SLP (%)')
ax.set_title('How much L1 does each method leave on the table?')
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig(_HERE / 'l1_gap_vs_slp.png', dpi=150)
plt.show()
```

Cell 6 (code) — convergence-check: `slp_iter.L1_dev <= lp_oneshot.L1_dev` per case:

```python
ones = df_feas[df_feas['method'] == 'lp_oneshot'][['case_id', 'L1_dev']].rename(
    columns={'L1_dev': 'L1_oneshot'}
)
slps = df_feas[df_feas['method'] == 'slp_iter'][['case_id', 'L1_dev']].rename(
    columns={'L1_dev': 'L1_slp'}
)
osvs = ones.merge(slps, on='case_id')
osvs['slp_minus_oneshot'] = osvs['L1_slp'] - osvs['L1_oneshot']
osvs.sort_values('slp_minus_oneshot')
```

Cell 7 (markdown):

```markdown
## Conclusions

* If `slp_iter` has `feasible=True` on every case and the gap-table
  shows non-negative entries everywhere: the no-compromise goal is met.
* If `slp_iter` fails feasibility on any case: trigger fallback 4
  (SQP-2tri) per the design spec.
* If `slp_iter.L1_dev > cluster_pipeline.L1_dev` on >50% of B0039 cases:
  trigger fallback 5 (cluster_lp).
```

- [ ] **Step 2: Run the notebook end-to-end.** Confirm both saved PNGs exist.

- [ ] **Step 3: Commit**

```bash
git add research/strict_feasibility_2d/analysis/02_lp_certifies_optimum.ipynb
git add research/strict_feasibility_2d/analysis/l1_gap_vs_slp.png
git commit -m "Analysis 02: LP optimum certification notebook"
```

---

## Task 14: README status update + final wrap-up

**Files:**
- Modify: `research/strict_feasibility_2d/README.md`

- [ ] **Step 1: Check all checkboxes in the status board + add results summary**

Edit the README's status table to:

```markdown
## Status

| Milestone | Status |
|---|---|
| Folder scaffolded | ✓ |
| Algorithms implemented (`lp_oneshot`, `slp_iter`) | ✓ |
| Worst-case catalog built | ✓ |
| Synthetic bake-off run | ✓ |
| B0039 z=12 bake-off run | ✓ |
| Analysis notebooks finalised | ✓ |
```

And add a Results section below:

```markdown
## Results (as of <YYYY-MM-DD>)

Headline numbers from `analysis/01_baseline_l1_gap.ipynb` and
`analysis/02_lp_certifies_optimum.ipynb`:

* `slp_iter` reaches `feasible=True` on <N>/<TOTAL> cases.
* On feasible cases, `slp_iter` L1 vs `m14` L1: median gap <X>% better.
* On feasible cases, `slp_iter` L1 vs `cluster_pipeline` L1: <result>.
* Wall-time at B0039 z=12: `slp_iter` = <Ts>, `cluster_pipeline` = <Ts>.

See [the two analysis notebooks](analysis/) for full breakdown.
Failure modes (if any) feeding back into the [spec fallback plan](../../docs/superpowers/specs/2026-06-14-strict-feasibility-2d-design.md#fallback-plan):

* <list specific cases that failed and which fallback they trigger>
```

Fill in the actual numbers and `<YYYY-MM-DD>` from your runs before committing.

- [ ] **Step 2: Run the full test suite once to confirm nothing regressed**

```bash
pytest tests/research/strict_feasibility_2d/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Final commit**

```bash
git add research/strict_feasibility_2d/README.md
git commit -m "README status board: research thread milestones complete"
```

---

## Self-review notes

After executing the plan, verify against the spec:

- [ ] Folder structure matches Section 1 of the spec (`worst_cases/`, `analysis/`, `algorithms/`, `runners/`).
- [ ] Both `lp_oneshot_2tri` AND `slp_iter_2tri` exist and are tested.
- [ ] All seven methods from the bake-off table run via `run_method` (`cluster_pipeline` may be skipped if its signature mismatch wasn't resolved — flag if so).
- [ ] `comparison_synthetic.csv` and `comparison_b0039.csv` both exist with all expected columns.
- [ ] Both analysis notebooks run end-to-end and save their PNGs.
- [ ] No fallback algorithms have been built (per spec: only on observed need).
- [ ] No spec requirement is left unimplemented.

If `slp_iter` fails strict feasibility on any worst-case row, that's the trigger for the **fallback plan** in the spec — do **not** silently relax tolerances. Stop and open a follow-up plan keyed to the specific failure mode (per the spec's fallback row table).
