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

# Or install from requirements.txt (includes voxelmorph from GitHub)
pip install -r requirements.txt
```

There is no test suite, linter, or CI pipeline. Validation is done through Jupyter notebooks.

## Architecture

### Data conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays with channels `[dz, dy, dx]`. For 2D work the z-slice dim is 1. Convention is pull-back (backward mapping).
- **3D fields:** `(3, D, H, W)` with `[dz, dy, dx]`.
- **Coordinates/correspondences:** always `[z, y, x]` ordering, shape `(N, 3)`.
- **Jacobian threshold:** `0.01` (from `dvfopt/_defaults.py`). Error tolerance `1e-5`.
- **SimpleITK interop:** arrays transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz). See `dvfopt/jacobian/sitk_jdet.py`.

### Optimization internals

- **phi flattening:** `phi[:len(phi)//2]` = dy, `phi[len(phi)//2:]` = dx. Preserve this when modifying objective/constraint functions.
- **Laplacian matrix:** uses `z*ny*nx + y*nx + x` flattening in `dvfopt/laplacian/matrix.py`.
- **Windowed approach:** iterative SLSQP finds worst-Jdet pixel, computes bounding box of connected negative region + 1px positive border (min 3×3), runs `scipy.optimize.minimize(method='SLSQP')` on that sub-window with frozen edges. Grows window by 2 if needed.
- **Parallel variant:** `iterative_parallel()` batches non-overlapping windows into `ProcessPoolExecutor`. Falls back to serial for single windows (avoids Windows spawn overhead).

### Constraint modes

The solver accepts `enforce_shoelace=True` (geometric quad-cell area) and `enforce_injectivity=True` (coordinate monotonicity) flags in addition to the default Jacobian determinant constraint.

### Key entry points

| Function | Module | Purpose |
|----------|--------|---------|
| `iterative_serial()` | `dvfopt.core.iterative` | Serial 2D iterative SLSQP (primary) |
| `iterative_parallel()` | `dvfopt.core.parallel` | Parallel 2D variant |
| `iterative_3d()` | `dvfopt.core.iterative3d` | 3D iterative SLSQP |
| `jacobian_det2D()` / `jacobian_det3D()` | `dvfopt.jacobian.numpy_jdet` | Fast numpy Jacobian determinant |
| `sliceToSlice3DLaplacian()` | `dvfopt.laplacian.solver` | Build DVF from correspondences |
| `make_deformation()` / `make_random_dvf()` | `dvfopt.testcases` | Generate test deformation fields |

### Directory layout

- `dvfopt/` — installable package (core solvers, jacobian, dvf utils, laplacian, viz, io, testcases)
- `notebooks/` — canonical experiment notebooks
- `benchmarks/` — performance comparison notebooks (serial vs parallel, constraint modes, scalability, registration methods)
- `scripts/` — image generation scripts for docs
- `data/` — real data NIfTI files and `.npy` test case arrays
- `archive/` — historical notebooks (not canonical)
