# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

Tests live in `tests/` and are run with `pytest`. CI runs the full pytest suite on Ubuntu (Python 3.11/3.12) via `.github/workflows/ci.yml`, and `.github/workflows/test.yml` additionally gates on `ruff check` + `ruff format --check`. Additional validation is done through Jupyter notebooks.

```bash
# Run all tests
pytest

# Run a specific test module
pytest tests/test_slp_strategy.py

# Lint + format (ruff is pinned to 0.15.21 in the dev extras; config in pyproject.toml)
ruff check dvfopt dvfopt_gui tests benchmarks
ruff format --check dvfopt dvfopt_gui tests benchmarks
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
  - **Windowed Jdet SLSQP family** (`dvfopt/core/slsqp/constraints*.py`, `gradients*.py`, `iterative*.py`, the windowed serial/parallel solvers, `dvfopt/core/iterative2d_barrier.py`, `dvfopt/core/iterative3d_barrier*.py`, `dvfopt/core/barrier_objective.py`):
    `phi[:N]` = `dx`, `phi[N:2N]` = `dy` (3D: also `phi[2N:]` = `dz`). I.e. **x-channel first**.
  - **All 2-triangle solvers** (the 2tri barrier, full-grid SLSQP, Schwarz, wall-breakers m02/m03/m10/m12/m14, `_cluster_2tri`, the shared `tri_primitives` module):
    `phi[:N]` = `dy`, `phi[N:]` = `dx`. I.e. **y-channel first**.
  - A flat phi vector from a Jdet-SLSQP helper CANNOT be passed to a 2-triangle helper without channel-swapping. If you write a new helper that may consume both, add an assertion on the layout. Helpers in `tri_primitives.py` and `_barrier_core.py` (anchor term) are the single sources of truth for the 2-triangle world — reuse them rather than re-deriving partials.
- **Laplacian matrix:** uses `z*ny*nx + y*nx + x` flattening in `laplacian/utils.py`.
- **Windowed approach:** iterative SLSQP finds worst-Jdet pixel, computes bounding box of connected negative region + 1px positive border (min 3×3), runs `scipy.optimize.minimize(method='SLSQP')` on that sub-window with frozen edges. Grows window by 2 if needed.
- **Parallel variant:** `iterative_parallel()` batches non-overlapping windows into `ProcessPoolExecutor`. Falls back to serial for single windows (avoids Windows spawn overhead).

### Constraint modes

The 2D solver accepts `enforce_shoelace=True` (geometric quad-cell area) and `enforce_injectivity=True` (coordinate monotonicity) flags in addition to the default Jacobian determinant constraint. The 3D solver (`iterative_3d`) supports `enforce_injectivity=True` (axial monotonicity of deformed coordinates — linear rows, see `dvfopt/jacobian/monotonicity.py`); the 3D analogue of `enforce_shoelace` (geometric cell volume) is served by the dedicated 6-tet constraint family, and `SLSQPWindowedStrategy(enforce_shoelace=True)` on a 3D constraint raises accordingly. Both flags are exposed on `SLSQPWindowedStrategy`.

### Parameterized API (v0.2 — current)

The public surface is organized around three orthogonal axes composed via `Solver`:

```python
from dvfopt import Solver, TriConstraint2D, L1Objective, BarrierStrategy
result = Solver(
    constraint=TriConstraint2D(shape=(H, W)),
    objective=L1Objective(eps=1e-4),
    strategy=BarrierStrategy(),
).fit(phi)
```

**Constraints** ([dvfopt/constraints.py](dvfopt/constraints.py)) — `TriConstraint2D`, `TriConstraint2DFullCoverage`, `JdetConstraint2D`, `JdetConstraint3D`. Each provides `values()`, `adjoint(v)`, optional `jacobian()`, plus `flatten/unflatten` between `(C, *shape)` arrays and the flat decision vector. The pack convention is encoded in `Constraint.pack` (`PhiPack.DY_FIRST` for 2-tri, `PhiPack.DX_FIRST` for Jdet).

**Objectives** ([dvfopt/objectives.py](dvfopt/objectives.py)) — `L1Objective(eps)`, `L2Objective()`, `NoneObjective()`. Wrap the shared `anchor_term` from `_barrier_core.py`. Composition (`+`, `*`) supported for research.

**Strategies** ([dvfopt/strategies.py](dvfopt/strategies.py)) — `NMVFStrategy` (heuristic neighborhood-mean smoother, original method), `SLPStrategy` (sequential-LP / `auto_slp` — the L1-minimising strict-feasibility champion: per-cluster trust-region SLP + m14 seed + HiGHS L1 step, continuous parallel cluster scheduler; promoted from `research/strict_feasibility_2d` into `dvfopt/core/slp/`; also accepts the 3D 6-tet constraint via the promoted `lp_direct_6tet`/`cluster_lp_6tet` solvers with the research-validated `seed_3d='m10'` default), `BarrierStrategy`, `SLSQPFullGridStrategy`, `SLSQPWindowedStrategy`, `SchwarzStrategy`, `SchwarzWrapperStrategy(inner=…)` (generic Schwarz wrapper around any 2-tri or 6-tet inner — auto-detects 2D vs 3D), `HarmonicALMBarrierStrategy` (alias `M10Strategy`), `HarmonicALMRefineRepairStrategy` (alias `M14Strategy`), `SchwarzHarmonicALMRefineRepairStrategy` (alias `M14SchwarzStrategy`). 3D analogues for the wallbreakers: `HarmonicALMBarrier3DStrategy` (alias `M10TetStrategy`), `HarmonicALMRefineRepair3DStrategy` (alias `M14TetStrategy`), `SchwarzHarmonicALMRefineRepair3DStrategy` (alias `M14Schwarz3DStrategy`). The class names are phase-stack-explicit: each algorithm in the pipeline (harmonic Laplacian extension, PHR-ALM, log-barrier polish, soft-penalty L2 refine, harmonic repair, Schwarz domain decomposition) appears in the name. The dedicated `Schwarz*` classes are equivalent to `SchwarzWrapperStrategy(inner=...)` with the inner pinned — both run through the shared `dvfopt.core.wallbreakers._schwarz_common` core (one implementation of the Schwarz decomposition, not two). Each Strategy is a dataclass with strategy-specific knobs. `accepts_constraints` and `supports_3d` class attrs declare compatibility; `Solver.__init__` checks at construction.

**Solver** ([dvfopt/solver.py](dvfopt/solver.py)) — composes the three; provides `from_spec(constraint='2tri', ...)` string-based construction and one-shot `correct_dvf(phi, ...)`. `auto_strategy(constraint, init_n_neg, init_min, objective_label)` encodes the strategy-selection heuristic: the 2-tri constraint family with the L1 objective auto-routes to `'slp'` (the champion) at every fold tier; other regimes keep the density-tiered heuristic. 6-tet 3D tiers like 2D: extremes (n_neg > 5000 or min < -10) route to the 3D wallbreakers (`m10_3d` for L2, `m14_schwarz_3d` on >200K-voxel volumes, else `m14_3d`); everything else keeps `barrier`.

**DVFopt facade** ([dvfopt/unified.py](dvfopt/unified.py)) — per-slice orchestration over `Solver`: 2D/3D auto-detection, tabular reports, plots. Use when you want `DVFoptConfig` string-based config and per-slice analysis across a 3D volume.

### Implementation modules (internal — strategies delegate to these)

The legacy `iterative_*` functions are no longer part of the public API but remain as internal implementations the strategies call into:

| Strategy | Delegates to |
|---|---|
| `NMVFStrategy` (Jdet 2D) | `dvfopt.core._nmvf.nmvf_correct_2d` |
| `SLPStrategy` (2-tri + 6-tet) | 2D: `dvfopt.core.slp.cluster_slp_iter` (large) / `slp_iter` (small); 3D: `cluster_slp_iter_3d` / `slp_iter_3d`; auto-routes by pixel/voxel count |
| `BarrierStrategy` (any constraint) | `_barrier_core.run_penalty_barrier_lbfgs` |
| `SLSQPFullGridStrategy` (2-tri) | `dvfopt.core.iterative2d_tri_slsqp.iterative_2d_tri_slsqp` |
| `SLSQPWindowedStrategy` (Jdet) | `dvfopt.core.slsqp.iterative.iterative_serial` / `iterative3d` |
| `SchwarzStrategy` (2-tri) | `dvfopt.core.iterative2d_tri_schwarz.iterative_2d_tri_schwarz` |
| `HarmonicALMBarrierStrategy` (alias `M10Strategy`) | `dvfopt.core.wallbreakers.iterative_2d_tri_harmonic_polished` |
| `HarmonicALMRefineRepairStrategy` (alias `M14Strategy`) | `dvfopt.core.wallbreakers.iterative_2d_tri_refine_repair` |
| `SchwarzHarmonicALMRefineRepairStrategy` (alias `M14SchwarzStrategy`) | `dvfopt.core.wallbreakers.iterative_2d_tri_refine_repair_schwarz` (thin closure shim around `_schwarz_common.cluster_schwarz_2d_tri`) |
| `SchwarzWrapperStrategy(inner=...)` | `dvfopt.core.wallbreakers._schwarz_common.cluster_schwarz_2d_tri` / `cluster_schwarz_3d_tet` directly, calling `inner.solve` per cluster |
| `HarmonicALMBarrier3DStrategy` (alias `M10TetStrategy`) | `dvfopt.core.wallbreakers._alm_3d` (harmonic + ALM-3D + polish) |
| `HarmonicALMRefineRepair3DStrategy` (alias `M14TetStrategy`) | `dvfopt.core.wallbreakers._refine_repair_3d` |
| `SchwarzHarmonicALMRefineRepair3DStrategy` (alias `M14Schwarz3DStrategy`) | `dvfopt.core.wallbreakers._refine_repair_3d_schwarz` |

### Building blocks (still public, still useful for custom pipelines)

| Function | Module | Purpose |
|----------|--------|---------|
| `harmonic_extension_2d()` (m02) | `dvfopt.core.wallbreakers._harmonic` | Laplacian extension over fold cores |
| `augmented_lagrangian_2d()` (m03) | `dvfopt.core.wallbreakers._alm` | PHR-ALM with L-BFGS-B |
| `l2_refine_2d()` (m12) | `dvfopt.core.wallbreakers._l2_refine` | Soft-penalty refinement of a feasible seed |
| `solve_cluster_2tri_2d()` | `dvfopt.core._cluster_2tri` | Per-cluster SLSQP with frozen-edge interior mask |
| `tri_areas_flat()` / `tri_grad_T_v()` | `dvfopt.core.tri_primitives` | Canonical 2-tri constraint + adjoint |
| `anchor_term()` / `run_penalty_barrier_lbfgs()` | `dvfopt.core._barrier_core` | Shared anchor + penalty→barrier homotopy |

### 2.5D marching (3D fold *prevention*)

`correct_dvf_25d()` ([dvfopt/pipeline_25d.py](dvfopt/pipeline_25d.py)) prevents
inter-layer 6-tet folds instead of repairing them. **Precondition: `dz ≡ 0`**
(i.e. the input is per-slice 2D-corrected) — the inter-layer 6-tet volume then
depends only on adjacent slices' `dy/dx`. The pipeline validates this and raises
if `dz ≠ 0`; it never writes `phi[0]`.

It auto-picks the mildest inter-layer as a frozen seed (no layer is cold-started
against raw data), sweeps outward in both directions repairing each slice against
its already-repaired neighbour (`march_slice`, elastic LP over the free plane's
interior with a frozen ring), then runs a frozen-rim 3D-interior mop
(`mop_interior_3d`) for folds that need *both* slices of a pair to move — which
the single-frozen-plane sweep structurally cannot fix.

On the full 528-slice B0039 volume this took the 3D fold count from **1,058,831 →
33** (99.997%). The residual ~33 are the **geometric floor** of the fixed-diagonal
6-tet decomposition (no feasible tet split exists), not a solver limitation — an
exact-feasibility solver with escalating freedom cannot move them.

| Function | Module | Purpose |
|----------|--------|---------|
| `correct_dvf_25d()` / `Correct25DReport` | `dvfopt.pipeline_25d` | End-to-end 2.5D marching orchestrator |
| `march_slice()` / `layer_min_v()` | `dvfopt.core.marching` | Per-slice sweep repair + inter-layer min-volume |
| `mop_interior_3d()` | `dvfopt.core.marching` | Frozen-rim 3D-interior elastic-SLP residual mop |

**Other primitives:**

| Function | Module | Purpose |
|----------|--------|---------|
| `correct_dvf_3d()` / `Correct3DReport` | `dvfopt.pipeline_3d` | End-to-end true-3D fold-*repair* orchestrator (6-tet feasibility); complements the 2.5D *prevention* pipeline above |
| `jacobian_det2D()` / `jacobian_det3D()` | `dvfopt.jacobian.numpy_jdet` | Fast numpy Jacobian determinant |
| `solveLaplacianFromCorrespondences()` | `laplacian.solver` | Build DVF from correspondences |
| `sliceToSlice3DLaplacian()` | `laplacian.correspondence` | Full slice-to-slice Laplacian registration pipeline |
| `make_deformation()` / `make_random_dvf()` | `test_cases` | Generate test deformation fields |

### Directory layout

- `dvfopt/` — installable package (core solvers, jacobian, dvf utils, viz, io)
- `dvfopt_gui/` — PyQtGraph live-solver GUI (`app.py` + the `LiveSolverWindow` mixins `_win_fileio.py`/`_win_render.py`/`_win_run.py` and shared helpers `_shared.py`, plus `worker.py`, `convergence.py`, `history.py`, `persistence.py`, `demo.py`, `overview.py`, `strategy_params.py`, `io_formats.py`). The GUI also supports a **true-3D mode**: load a `(3, D, H, W)` volume and pick the `6-tet (3D)` or `Jdet (3D)` constraint to solve the whole volume with the 3D pipelines (M14Tet/M14-Schwarz3D/M10Tet/SLSQP-fullgrid-3D, or Barrier/SLSQP-windowed for Jdet3D). 3D wallbreaker runs stream per-phase snapshots and honor Stop at phase boundaries; the viewer renders the 6-tet min-volume slice of the current z. The method menu now includes **SLP (default 2-tri champion; also in the tet3d menu as SLP-3D)** and an **Auto** picker (`auto_strategy`, available for the 2D families AND the 3D constraints); the **Pipeline ▾** button runs `correct_dvf_25d` (2.5D marching, needs dz≡0 — a violation prompts an explicit, undoable consent dialog to zero the dz channel before running) or the one-click **full pipeline** (per-slice 2D → 2.5D). The tet3d menu adds the **full 3D pipeline** (`correct_dvf_3d`) and a torch-gated GPU barrier. Loads accept NIfTI/MetaImage/NRRD displacement fields via SimpleITK (and export back to `.npy`/`.nii.gz`); loads are threaded and reject non-finite fields. The feasibility threshold is editable (`thr:` spinbox), 3D metrics are cached (fast z-scrub/hover), the undo stack is byte-budgeted, a clickable per-slice fold strip sits under the plot, every strategy's dataclass knobs — spanning the 2D, tet3d, and jdet3d families — are editable via Params → Strategy, and "Run section" works on 3D sub-volumes (Rect ROI + z-range). Solver-path runs record their SolveInfo: the convergence chart marks pipeline-stage boundaries (stage names ride on the history snapshots and survive save/load), View → "Save convergence report…" renders `plot_solve_info`, and a View → "Solver log" dock streams the dvfopt logger live (its level drives the worker's `verbose`). An **Injectivity gap (min axial)** view mode renders the monotonicity-gap map in 2D and 3D, and the Params dialog renders `float | None` knobs (e.g. `injectivity_threshold`) as checkbox-enabled overrides — the 2D windowed method exposes exactly its constraint-mode toggles.
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
