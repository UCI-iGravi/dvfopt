# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for correcting **negative Jacobian determinants** in 2D/3D deformation (displacement) fields. The installable `dvfopt/` package implements three correction methods (heuristic NMVF, full-grid SLSQP, iterative SLSQP). Notebooks in `notebooks/` demonstrate each method; `benchmarks/` compares performance across registration algorithms.

## Setup & Commands

```bash
# Install core package (editable)
pip install -e .

# Install with benchmark dependencies (itk-elastix, opencv, timm, torch, voxelmorph)
pip install -e ".[benchmarks]"

# Or install all dev dependencies (includes voxelmorph from GitHub, pandas, ipykernel)
pip install -r requirements-dev.txt
```

Tests live in `tests/` and are run with `pytest`. There is no linter or CI pipeline. Additional validation is done through Jupyter notebooks.

```bash
# Run all tests
pytest

# Run a specific test module
pytest tests/test_iterative.py
```

## Architecture

### Data conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays with channels `[dz, dy, dx]`. For 2D work the z-slice dim is 1. Convention is pull-back (backward mapping).
- **3D fields:** `(3, D, H, W)` with `[dz, dy, dx]`.
- **Coordinates/correspondences:** always `[z, y, x]` ordering, shape `(N, 3)`.
- **Jacobian threshold:** `0.01` (from `dvfopt/_defaults.py`). Error tolerance `1e-5`.
- **SimpleITK interop:** arrays transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz). See `dvfopt/jacobian/sitk_jdet.py`.

### Optimization internals

- **phi flattening — TWO conventions exist, do not cross-mix them:**
  - **SLSQP modules** (`dvfopt/core/slsqp/constraints*.py`, `gradients*.py`, `iterative*.py`, the windowed solver, `dvfopt/core/iterative2d_barrier.py`, `dvfopt/core/iterative3d_barrier*.py`, `dvfopt/core/barrier_objective.py`):
    `phi[:N]` = `dx`, `phi[N:2N]` = `dy` (3D: also `phi[2N:]` = `dz`). I.e. **x-channel first**.
  - **Tri-barrier module** (`dvfopt/core/iterative2d_tri_barrier.py` — the 2-triangle penalty/barrier solver):
    `phi[:N]` = `dy`, `phi[N:]` = `dx`. I.e. **y-channel first**.
  - A flat phi vector from one module CANNOT be passed to a helper from the other without channel-swapping. If you write a new helper that may consume both, add an assertion on the layout.
- **Laplacian matrix:** uses `z*ny*nx + y*nx + x` flattening in `laplacian/utils.py`.
- **Windowed approach:** iterative SLSQP finds worst-Jdet pixel, computes bounding box of connected negative region + 1px positive border (min 3×3), runs `scipy.optimize.minimize(method='SLSQP')` on that sub-window with frozen edges. Grows window by 2 if needed.
- **Parallel variant:** `iterative_parallel()` batches non-overlapping windows into `ProcessPoolExecutor`. Falls back to serial for single windows (avoids Windows spawn overhead).

### Constraint modes

The 2D solver accepts `enforce_shoelace=True` (geometric quad-cell area) and `enforce_injectivity=True` (coordinate monotonicity) flags in addition to the default Jacobian determinant constraint. The 3D solver (`iterative_3d`) does not yet support these extra constraint modes — only the Jacobian determinant constraint is available in 3D.

### Key entry points

**SLSQP windowed solvers (Jdet constraint):**

| Function | Module | Purpose |
|----------|--------|---------|
| `iterative_serial()` | `dvfopt.core.iterative` | Serial 2D iterative SLSQP (primary) |
| `iterative_parallel()` | `dvfopt.core.parallel` | Parallel 2D variant |
| `iterative_3d()` | `dvfopt.core.iterative3d` | 3D iterative SLSQP |

**2D 2-triangle solvers** (the strict PL-bijectivity constraint; see `dvfopt/__init__.py` docstring for guidance on which to pick):

| Function | Strategy | Use case |
|----------|----------|----------|
| `iterative_2d_tri_slsqp()` | Full-grid SLSQP + L1/L2 + warm-restart (notebook 14) | Mild folds, full-grid fits in memory |
| `iterative_2d_tri_barrier()` | Penalty → log-barrier L-BFGS-B | Mild to moderate folds, all sizes |
| `iterative_2d_tri_schwarz()` | Hybrid overlapping-tile Schwarz + per-cluster SLSQP | Many small clusters across a big slice |
| `iterative_2d_tri_harmonic_polished()` (m10) | Harmonic seed + ALM + barrier polish | Dense folds, **100% feasibility guaranteed** |
| `iterative_2d_tri_refine_repair()` (m14, anchor='l1' = m14_l1) | m10 seed + soft-penalty pull + repair + polish | Dense folds, **smallest L2/L1 cost** |

**Building blocks:**

| Function | Module | Purpose |
|----------|--------|---------|
| `solve_cluster_2tri_2d()` | `dvfopt.core._cluster_2tri` | Per-cluster SLSQP with frozen-edge interior mask |
| `harmonic_extension_2d()` (m02) | `dvfopt.core.wallbreakers._harmonic` | Laplacian extension over fold cores |
| `augmented_lagrangian_2d()` (m03) | `dvfopt.core.wallbreakers._alm` | PHR-ALM with L-BFGS-B |
| `l2_refine_2d()` (m12) | `dvfopt.core.wallbreakers._l2_refine` | Soft-penalty refinement of a feasible seed |
| `tri_areas_flat()` / `tri_grad_T_v()` | `dvfopt.core.tri_primitives` | Canonical 2-triangle constraint + adjoint |

**Other primitives:**

| Function | Module | Purpose |
|----------|--------|---------|
| `jacobian_det2D()` / `jacobian_det3D()` | `dvfopt.jacobian.numpy_jdet` | Fast numpy Jacobian determinant |
| `solveLaplacianFromCorrespondences()` | `laplacian.solver` | Build DVF from correspondences |
| `sliceToSlice3DLaplacian()` | `laplacian.correspondence` | Full slice-to-slice Laplacian registration pipeline |
| `make_deformation()` / `make_random_dvf()` | `test_cases` | Generate test deformation fields |

### Directory layout

- `dvfopt/` — installable package (core solvers, jacobian, dvf utils, viz, io)
- `laplacian/` — standalone Laplacian interpolation package (matrix construction, CG/LGMRES solvers, contour correspondence matching)
- `test_cases/` — standalone test case definitions and builders (synthetic, random DVF, real-data slices)
- `notebooks/` — canonical experiment notebooks
- `benchmarks/` — performance comparison notebooks, grouped into subfolders:
  - `solvers/slsqp/` — SLSQP windowed solver comparisons (serial vs parallel, constraint modes, windowed vs fullgrid, 3D correction)
  - `solvers/barrier/` — penalty/barrier L-BFGS solver (3D barrier, CPU vs GPU)
  - `scaling/` — performance vs grid size, folding severity, L2-Jdet correlation
  - `registration/` — external registration methods (Elastix, VoxelMorph, TransMorph, ANTs, OpenCV) + post-hoc correction
  - `pipelines/` — end-to-end 3D slice-wise correction pipelines
  - `benchmark_utils.py` — shared helpers; notebooks add `..` to sys.path to import it
- `scripts/` — image generation scripts for docs
- `data/` — real data NIfTI files and `.npy` test case arrays
- `archive/` — historical notebooks (not canonical)
