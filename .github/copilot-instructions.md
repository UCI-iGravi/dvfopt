# Copilot Instructions — Deformation Field Correction

## Project Overview

Research codebase for correcting **negative Jacobian determinants** in 2D deformation (displacement) fields. Three correction methods are implemented in Jupyter notebooks, with shared utilities in `modules/`. Results are organized under `paper_outputs/` and `test/`.

## Core Data Conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays — channels are `[dz, dy, dx]`. For 2D slice work, the z-slice dimension is 1. Convention is **pull-back** (backward mapping): for each point in the fixed image, the displacement vector points to where in the moving image that pixel's value comes from. I.e. `fixed_pos + displacement = moving_pos`.
- **Points/coordinates:** Always `[z, y, x]` ordering. Correspondences are `(N, 3)` arrays.
- **SimpleITK interop:** Displacement arrays are transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz) before calling SimpleITK. See `modules/jacobian.py:sitk_jacobian_determinant()`.
- **Jacobian computation (optimiser):** `dvfopt.py` uses a pure-numpy 2D Jacobian determinant (`_numpy_jdet_2d`) via `np.gradient` central differences. This matches SimpleITK for interior pixels and avoids the ~3 ms/call SimpleITK overhead that made SLSQP numerical gradients infeasible. `modules/jacobian.py` still provides the SimpleITK wrapper for other uses.
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
2. Computes the bounding box of the connected negative-Jdet region around it via `neg_jdet_bounding_window()`, adds a +1 pixel positive-Jdet border (minimum 3×3). Both even and odd window sizes are supported.
3. Extracts a submatrix window of that size (grows by 2 if needed up to grid size).
4. Runs `scipy.optimize.minimize(method='SLSQP')` on the submatrix with frozen edge constraints.
5. Repeats for next-worst pixel. Tracks `window_counts` per size.

## Shared Modules (`modules/`)

- **`dvfopt.py`** — Core optimisation module containing both serial and hybrid-parallel iterative SLSQP algorithms. Key entry points: `iterative_with_jacobians2(deformation, method, ...)` (serial) and `iterative_parallel(deformation, method, ..., max_workers=None)` (hybrid parallel). Also exports `jacobian_det2D()`, objective/constraint helpers, windowed sub-optimisation utilities, `generate_random_dvf()`, `scale_dvf()`, and `neg_jdet_bounding_window()`. No matplotlib or pandas dependency.
- **`dvfviz.py`** — All visualisation and convenience orchestration. `plot_deformations()`: 2×2 initial-vs-corrected panel. `plot_jacobians_iteratively()`: grid of Jacobian snapshots. `run_lapl_and_correction()`: end-to-end Laplacian → correction → plot pipeline (also calls `plot_grid_before_after`). `plot_step_snapshot()`: single-panel per-iteration heatmap (called lazily from `dvfopt` when `plot_every` is set). `plot_deformation_field()`: single-field Jacobian + quiver preview. `plot_2d_deformation_grid()`: deformed grid lines. `plot_deformed_quads()` / `plot_deformed_quads_colored()`: quad mesh visualisation colored by Jacobian determinant. `plot_grid_before_after()`: side-by-side initial-vs-corrected deformation grids coloured by Jacobian determinant, with yellow outlines on negative-Jdet cells.
- **`testcases.py`** — Test case registry and data-loading utilities. `SYNTHETIC_CASES`: dict of 8 correspondence-based test cases. `RANDOM_DVF_CASES`: dict of 4 random DVF configs. `REAL_DATA_SLICES`: dict of 8 real-data slice configs. `make_deformation(case_key)`: builds a `(3,1,H,W)` deformation from correspondences via Laplacian. `make_random_dvf(case_key)`: generates a random DVF. `load_slice(slice_idx, ...)`: loads a real `.npy` slice with optional downscaling. `save_and_summarize(deformation, save_path)`: saves deformation + prints neg-Jdet summary.
- **`checkerboard.py`** — Checkerboard image creation. `create_checkerboard()`: generates a binary checkerboard array.
- **`jacobian.py`** — `sitk_jacobian_determinant(deformation)`: wraps SimpleITK Jacobian computation. `surrounding_points()`: debug utility.
- **`laplacian.py`** — `laplacianA3D()`: builds sparse Laplacian matrix with Dirichlet BCs. `compute3DLaplacianFromShape()`: solves Laplacian system via LGMRES. `sliceToSlice3DLaplacian()`: end-to-end pipeline from NIfTI.
- **`correspondences.py`** — `remove_duplicates()`, `do_lines_intersect()`, `swap_correspondences()`, `downsample_points()`: handle point correspondences and detect/resolve crossing displacement vectors.

## Test Cases & Data

- **Synthetic grids:** Defined in `modules/testcases.py` as `SYNTHETIC_CASES` dict mapping case keys to `(msample, fsample, grid_size)` tuples. Common sizes: 10×10, 20×20. Types: `crossing` (intersecting vectors), `opposites` (opposing vectors), `checkerboard`.
- **Random DVFs:** Defined in `modules/testcases.py` as `RANDOM_DVF_CASES` dict. Generated via `generate_random_dvf(shape=(3,1,H,W), max_magnitude=5.0)` from `dvfopt.py`.
- **Real data:** `.npy` files in `experiments/` (e.g., `02b_320x456_slice200.npy`). Configured in `REAL_DATA_SLICES` dict. Downscaled versions at 64×91 via `scale_dvf()`.

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
