# Copilot Instructions — Deformation Field Correction

## Project Overview

Research codebase for correcting **negative Jacobian determinants** in 2D and 3D deformation (displacement) fields. Three correction methods are implemented in Jupyter notebooks under `notebooks/`, with shared code in the installable `dvfopt/` package. Results are organized under `output/` and `data/`.

## Core Data Conventions

- **Deformation fields:** `(3, 1, H, W)` numpy arrays — channels are `[dz, dy, dx]`. For 2D slice work, the z-slice dimension is 1. Convention is **pull-back** (backward mapping): for each point in the fixed image, the displacement vector points to where in the moving image that pixel's value comes from. I.e. `fixed_pos + displacement = moving_pos`.
- **Points/coordinates:** Always `[z, y, x]` ordering. Correspondences are `(N, 3)` arrays.
- **SimpleITK interop:** Displacement arrays are transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx→xyz) before calling SimpleITK. See `dvfopt/jacobian/sitk_jdet.py:sitk_jacobian_determinant()`.
- **Jacobian computation (optimiser):** `dvfopt/jacobian/numpy_jdet.py` uses a pure-numpy 2D Jacobian determinant (`_numpy_jdet_2d`) via `np.gradient` central differences. This matches SimpleITK for interior pixels and avoids the ~3 ms/call SimpleITK overhead that made SLSQP numerical gradients infeasible.
- **Jacobian threshold:** `0.01` (strictly positive, not ≥0). Error tolerance `1e-5`.
- **Plotting:** Uses `indexing='xy'` for meshgrid; y-axis is inverted (`invert_yaxis()`) to match image convention.

## Three Correction Methods

| Method | Notebook | Key Function | Tradeoff |
|--------|----------|-------------|----------|
| **Heuristic (NMVF)** | `notebooks/heuristic-neg-jacobian.ipynb` | `heuristic_negative_jacobian_correction()` | Fastest, highest L2 error |
| **Full SLSQP** | `legacy_code/slsqp-full-modified.ipynb` | `full_slsqp()` | Lowest L2 error, slowest (full grid optimization) |
| **Iterative SLSQP** | `notebooks/slsqp-iterative-refactored.ipynb` | `iterative_serial()` | Near-optimal L2, faster (windowed sub-optimizations) |

All methods take a `(3, 1, H, W)` deformation, fix negative-Jdet regions, and return a corrected field.

### Iterative SLSQP specifics (primary method)
1. Finds pixel with lowest Jdet (excluding edges via `argmin_excluding_edges()`).
2. Computes the bounding box of the connected negative-Jdet region around it via `neg_jdet_bounding_window()`, adds a +1 pixel positive-Jdet border (minimum 3×3). Both even and odd window sizes are supported.
3. Extracts a submatrix window of that size (grows by 2 if needed up to grid size).
4. Runs `scipy.optimize.minimize(method='SLSQP')` on the submatrix with frozen edge constraints.
5. Repeats for next-worst pixel. Tracks `window_counts` per size.

## Package Structure (`dvfopt/`)

- **`dvfopt.core`** — Optimization algorithms. `core/objective.py`: L2 objective. `core/constraints.py`: Jacobian/shoelace/injectivity constraints. `core/spatial.py`: window selection, bounding boxes, edge logic. `core/solver.py`: single-window SLSQP. `core/iterative.py`: `iterative_serial()` (serial 2D). `core/parallel.py`: `iterative_parallel()` (hybrid parallel 2D). `core/solver3d.py` + `core/iterative3d.py`: 3D extension.
- **`dvfopt.jacobian`** — Jacobian computation. `numpy_jdet.py`: pure-numpy 2D/3D via `np.gradient`. `sitk_jdet.py`: SimpleITK wrapper. `shoelace.py`: geometric quad-cell area constraint. `monotonicity.py`: injectivity/monotonicity constraint.
- **`dvfopt.dvf`** — DVF utilities. `generation.py`: `generate_random_dvf()` (2D/3D). `scaling.py`: `scale_dvf()` bicubic rescaling (2D/3D).
- **`dvfopt.laplacian`** — Laplacian interpolation. `matrix.py`: sparse Laplacian matrix with Dirichlet BCs. `solver.py`: LGMRES solver, `sliceToSlice3DLaplacian()` end-to-end pipeline.
- **`dvfopt.viz`** — All visualization. `snapshots.py`: per-iteration heatmaps. `fields.py`: deformation field plots. `grids.py`: deformed quad-grid visualization colored by Jdet. `closeups.py`: checkerboard and neighborhood views. `pipeline.py`: `run_lapl_and_correction()` end-to-end pipeline.
- **`dvfopt.io`** — I/O. `nifti.py`: NIfTI loading via nibabel.
- **`dvfopt.utils`** — Helpers. `checkerboard.py`, `correspondences.py`, `transform.py`.
- **`dvfopt.testcases`** — Test case registry. `SYNTHETIC_CASES`, `RANDOM_DVF_CASES`, `REAL_DATA_SLICES`, `make_deformation()`, `make_random_dvf()`, `load_slice()`, `save_and_summarize()`.
- **`correspondences.py`** — `remove_duplicates()`, `do_lines_intersect()`, `swap_correspondences()`, `downsample_points()`: handle point correspondences and detect/resolve crossing displacement vectors.
- **`correspondences.py`** — `remove_duplicates()`, `do_lines_intersect()`, `swap_correspondences()`, `downsample_points()`: handle point correspondences and detect/resolve crossing displacement vectors.

## Test Cases & Data

- **Synthetic grids:** Defined in `dvfopt/testcases.py` as `SYNTHETIC_CASES` dict mapping case keys to `(msample, fsample, grid_size)` tuples. Common sizes: 10×10, 20×20. Types: `crossing` (intersecting vectors), `opposites` (opposing vectors), `checkerboard`.
- **Random DVFs:** Defined in `dvfopt/testcases.py` as `RANDOM_DVF_CASES` dict. Generated via `generate_random_dvf(shape=(3,1,H,W), max_magnitude=5.0)` from `dvfopt.dvf`.
- **Real data:** `.npy` files in `data/` (e.g., `02b_320x456_slice200.npy`). Configured in `REAL_DATA_SLICES` dict. Downscaled versions at 64×91 via `scale_dvf()`.

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

- Notebooks in `archive/` are historical iterations; notebooks in `notebooks/` are canonical.
- When modifying optimization functions (`objectiveEuc`, constraint functions), preserve the `phi` flattening convention: `phi[:len(phi)//2]` = dy, `phi[len(phi)//2:]` = dx.
- Laplacian matrix construction in `dvfopt/laplacian/matrix.py` uses `z*ny*nz + y*nz + x` flattening — be careful with axis ordering when modifying.
