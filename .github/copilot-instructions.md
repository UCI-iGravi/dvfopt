# Copilot Instructions — Deformation Field Correction

## Project Overview

Research codebase for correcting **negative Jacobian determinants** in 2D deformation (displacement) fields. Three correction methods are implemented in Jupyter notebooks, with shared utilities in `modules/`. Results are organized under `paper_outputs/` and `test/`.

## Core Data Conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays — channels are `[dz, dy, dx]`. For 2D slice work, the z-slice dimension is 1.
- **Points/coordinates:** Always `[z, y, x]` ordering. Correspondences are `(N, 3)` arrays.
- **SimpleITK interop:** Displacement arrays are transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz) before calling SimpleITK. See `modules/jacobian.py:sitk_jacobian_determinant()`.
- **Jacobian threshold:** `0.01` (strictly positive, not ≥0). Error tolerance `1e-5`.
- **Plotting:** Uses `indexing='xy'` for meshgrid; y-axis is inverted (`invert_yaxis()`) to match image convention.

## Three Correction Methods

| Method | Notebook | Key Function | Tradeoff |
|--------|----------|-------------|----------|
| **Heuristic (NMVF)** | `heuristic-neg-jacobian.ipynb` | `heuristic_negative_jacobian_correction()` | Fastest, highest L2 error |
| **Full SLSQP** | `slsqp-full-modified.ipynb` | `full_slsqp()` | Lowest L2 error, slowest (full grid optimization) |
| **Iterative SLSQP** | `slsqp-iterative.ipynb` | `iterative_with_jacobians2()` | Near-optimal L2, faster (windowed sub-optimizations) |

All methods take a `(3, 1, H, W)` deformation, fix negative-Jdet regions, and return a corrected field.

### Iterative SLSQP specifics (primary method)
1. Finds pixel with lowest Jdet (excluding edges via `argmin_excluding_edges()`).
2. Extracts a submatrix window (starting 9×9, grows by 2 if needed up to grid size).
3. Runs `scipy.optimize.minimize(method='SLSQP')` on the submatrix with frozen edge constraints.
4. Repeats for next-worst pixel. Tracks `window_counts` per size.

## Shared Modules (`modules/`)

- **`dvfopt.py`** — Core iterative SLSQP optimisation. Key entry point: `iterative_with_jacobians2(deformation, method, ...)`. Also exports `jacobian_det2D()`, objective/constraint helpers, and windowed sub-optimisation utilities. No matplotlib or pandas dependency.
- **`dvfviz.py`** — All visualisation and convenience orchestration. `plot_deformations()`: 2×2 initial-vs-corrected panel. `plot_jacobians_iteratively()`: grid of Jacobian snapshots. `run_lapl_and_correction()`: end-to-end Laplacian → correction → plot pipeline. `plot_step_snapshot()`: single-panel per-iteration heatmap (called lazily from `dvfopt` when `plot_every` is set).
- **`jacobian.py`** — `sitk_jacobian_determinant(deformation)`: wraps SimpleITK Jacobian computation. `surrounding_points()`: debug utility.
- **`laplacian.py`** — `laplacianA3D()`: builds sparse Laplacian matrix with Dirichlet BCs. `compute3DLaplacianFromShape()`: solves Laplacian system via LGMRES. `sliceToSlice3DLaplacian()`: end-to-end pipeline from NIfTI.
- **`correspondences.py`** — `remove_duplicates()`, `do_lines_intersect()`, `swap_correspondences()`, `downsample_points()`: handle point correspondences and detect/resolve crossing displacement vectors.

## Test Cases & Data

- **Synthetic grids:** Defined inline as `msample`/`fsample` point arrays. Common sizes: 10×10, 20×20. Types: `crossing` (intersecting vectors), `opposites` (opposing vectors), `checkerboard`.
- **Real data:** `.npy` files in `experiments/` (e.g., `02b_320x456_slice200.npy`). Downscaled versions at 64×91.
- **Random DVFs:** Generated via `generate_random_dvf(shape=(3,1,H,W), max_magnitude=5.0)`.

## Output Structure

Results save to a directory with:
- `results.txt` — Settings, runtime, L2 error, neg-Jdet counts, min Jdet
- `phi.npy` — Corrected deformation field
- `error_list_l2.npy`, `num_neg_jac.npy`, `iter_times.npy`, `min_jdet_list.npy` — Per-iteration metrics
- `window_counts.csv` — (Iterative SLSQP only) Window size usage histogram

Organized as `paper_outputs/experiments/{method}/{grid_size}/{test_case}/` and `test/{method}/`.

## Key Dependencies

`numpy`, `scipy` (SLSQP optimizer + sparse LGMRES), `SimpleITK` (Jacobian determinant), `nibabel` (NIfTI I/O), `matplotlib` (visualization).

## Working With This Codebase

- The `slsqp-iterative copy.ipynb` is a working copy of `slsqp-iterative.ipynb` — check for divergence before editing.
- Notebooks in `archive/` are historical iterations; the root-level notebooks are canonical.
- When modifying optimization functions (`objectiveEuc`, constraint functions), preserve the `phi` flattening convention: `phi[:len(phi)//2]` = dy, `phi[len(phi)//2:]` = dx.
- Laplacian matrix construction in `modules/laplacian.py` uses `z*ny*nz + y*nz + x` flattening — be careful with axis ordering when modifying.
