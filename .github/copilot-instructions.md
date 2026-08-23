# Copilot Instructions — Deformation Field Correction

## Project Overview

Research codebase for correcting **negative Jacobian determinants** in 2D and 3D deformation (displacement) fields. Everything installable lives in the single `dvfopt/` package; `notebooks/` demonstrates the methods and `benchmarks/` compares them. Real data and results are gitignored under `data/`.

Two companion docs are the source of truth and worth reading before a non-trivial change:

- [CLAUDE.md](../CLAUDE.md) — the map of what exists (API surface, strategy→implementation table, directory layout).
- [ARCHITECTURE.md](../ARCHITECTURE.md) — the rules: import-direction constraints, phi-pack conventions, and the checklists for adding a method / constraint / objective.

## Core Data Conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays — channels are `[dz, dy, dx]`. For 2D slice work the z-slice dimension is 1. 3D is `(3, D, H, W)`. Convention is **pull-back** (backward mapping): `fixed_pos + displacement = moving_pos`.
- **Points/coordinates:** Always `[z, y, x]` ordering. Correspondences are `(N, 3)` arrays.
- **SimpleITK interop:** Displacement arrays are transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz) before calling SimpleITK. See `dvfopt/jacobian/sitk_jdet.py:sitk_jacobian_determinant()`.
- **Jacobian computation:** `dvfopt/jacobian/numpy_jdet.py` uses a pure-numpy Jacobian determinant via `np.gradient` central differences. This matches SimpleITK for interior pixels and avoids the ~3 ms/call SimpleITK overhead that made SLSQP numerical gradients infeasible.
- **Jacobian threshold:** `0.01` (strictly positive, not ≥0), from `dvfopt/_defaults.py`. Error tolerance `1e-5`.
- **Plotting:** `indexing='xy'` for meshgrid; y-axis inverted (`invert_yaxis()`) to match image convention.

## The three axes

A solve is `Solver(constraint=..., objective=..., strategy=...).fit(phi)`:

- **Constraints** (`dvfopt/constraints.py`) — `TriConstraint2D`, `TriConstraint2DFullCoverage`, `JdetConstraint2D`, `JdetConstraint3D`, `Tet6Constraint3D`.
- **Objectives** (`dvfopt/objectives.py`) — `L1Objective(eps)`, `L2Objective()`, `NoneObjective()`, over the shared `anchor_term`.
- **Strategies** (`dvfopt/strategies/`) — `SLPStrategy` (the 2-tri champion), `BarrierStrategy`, `SLSQPWindowedStrategy`, `SLSQPFullGridStrategy`, `SchwarzStrategy`, the wallbreaker families (`M10`/`M14`, 2D and 3D), `NMVFStrategy`. Each declares `accepts_constraints` / `accepts_objectives` / `supports_3d`; `Solver.__init__` rejects a bad triple at construction.

`dvfopt/core/` is method-first — one sub-package per algorithm family (`primitives`, `nmvf`, `barrier`, `slsqp_windowed`, `slsqp_fullgrid`, `schwarz`, `wallbreakers`, `slp`, `marching`). Strategies delegate into these; the sub-packages never import each other (only `core/primitives/` and the two shared engines, `barrier/_core.py` and `schwarz/_common.py`).

### Windowed SLSQP specifics

1. Find the interior pixel with the lowest Jdet (excluding edges).
2. Take the connected component of negative-Jdet pixels around it (8-connectivity), bounding-box it, add a +1 pixel positive border (minimum 3×3).
3. Extract that sub-window, freeze its edge pixels to their initial values.
4. Run the traced SLSQP driver (`dvfopt/core/primitives/slsqp.py`) on the sub-window; grow by +2 and retry if it does not converge; fall back to full-grid as a last resort.
5. Repeat for the next-worst pixel.

## Test Cases & Data

- **Synthetic grids:** `SYNTHETIC_CASES` in `dvfopt/testdata/_cases.py` — `(msample, fsample, grid_size)` tuples. Common sizes 10×10, 20×20; types `crossing`, `opposites`, `checkerboard`.
- **Random DVFs:** `RANDOM_DVF_CASES` in the same module, generated via `generate_random_dvf` from `dvfopt.dvf`.
- **Real data:** `.npy` files in `data/` (gitignored), configured in `REAL_DATA_SLICES`.

## Key Dependencies

`numpy`, `scipy>=1.15,<1.19` (SLSQP driver + sparse LGMRES; the pin is load-bearing — `dvfopt/core/primitives/slsqp.py` vendors scipy's `_slsqplib` internals, which exist only on scipy >=1.16/Python >=3.11; on scipy 1.15.x, e.g. Python 3.10, it transparently falls back to scipy's own `minimize(method='SLSQP')`), `SimpleITK`, `nibabel`, `matplotlib`, `scikit-image`. Python >= 3.10.

## Working With This Codebase

- Notebooks in `archive/` are historical; notebooks in `notebooks/` are canonical.
- **Never cross the two phi-pack conventions.** `Constraint.pack` declares which one applies: `PhiPack.DY_FIRST` (`[dy, dx]`) for the 2D 2-triangle constraints, `PhiPack.DX_FIRST` (`[dx, dy(, dz)]`) for Jdet 2D/3D **and 6-tet**. See ARCHITECTURE.md.
- Anchor/objective code goes through an `Objective` (`__call__(diff) -> (value, grad)`); there are no string-anchor parameters left.
- Laplacian matrix construction in `dvfopt/laplacian/utils.py` uses `z*ny*nx + y*nx + x` flattening — be careful with axis ordering when modifying.
