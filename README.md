# Deformation Field Correction

Correct **negative Jacobian determinants** (folding) in 2D and 3D displacement fields produced by image registration. Three correction methods are provided — a fast heuristic, a full-grid constrained optimizer, and an iterative windowed optimizer — all ensuring the corrected field stays as close as possible to the original while eliminating folding.

## Table of Contents

- [Background](#background)
- [Problem Formulation](#problem-formulation)
- [Jacobian Determinant](#jacobian-determinant)
- [Correction Methods](#correction-methods)
  - [Heuristic (NMVF)](#1-heuristic-neighborhood-mean-vector-filter)
  - [Full-Grid SLSQP](#2-full-grid-slsqp)
  - [Iterative SLSQP](#3-iterative-slsqp)
  - [Hybrid Parallel](#hybrid-parallel-variant)
- [Constraints](#constraints)
- [3D Extension](#3d-extension)
- [Data Conventions](#data-conventions)
- [Project Structure](#project-structure)
- [Test Cases](#test-cases)
- [Installation](#installation)
- [Usage](#usage)

---

## Background

Deformable image registration computes a displacement field $\phi$ that maps each pixel in a fixed image to its corresponding location in a moving image. When the Jacobian determinant of $\phi$ becomes negative at a pixel, the mapping **folds** — meaning the deformed grid crosses over itself both locally and globally, creating a physically implausible transformation.

This project corrects such folding by minimally adjusting the displacement field so that $J_{\det}(\phi) > 0$ everywhere, while keeping the corrected field as close to the original as possible. This is posed as a constrained optimization problem, which can be applied as a post-processing step to any existing registration pipeline.

## Problem Formulation

Given an input displacement field $\phi_{\text{init}}$ with regions of negative Jacobian determinant, find a corrected field $\phi^*$ that solves:

$$\phi^* = \arg\min_\phi \|\phi - \phi_{\text{init}}\|_2$$

subject to:

$$J_{\det}(\phi)(x, y) \geq \tau \quad \forall \; (x, y) \in \Omega$$

where $\tau = 0.01$ is the Jacobian determinant threshold (strictly positive) and $\Omega$ is the spatial domain.

### Breaking it down

**The objective** — $\|\phi - \phi_{\text{init}}\|_2$:

- $\phi$ is the displacement field we are searching for (the corrected output).
- $\phi_{\text{init}}$ is the original displacement field that came out of registration — the one with folding.
- $\|\cdot\|_2$ is the L2 (Euclidean) norm: the straight-line distance between two vectors. Concretely, $\|\phi - \phi_{\text{init}}\|_2 = \sqrt{\sum_i (\phi_i - \phi_{\text{init},i})^2}$, summing over every displacement component at every pixel.
- **Why L2?** We want the corrected field to stay as close as possible to the original. The L2 norm penalizes large deviations more than many small ones, so it favors spreading tiny adjustments across the field rather than making one big change somewhere.

**The constraint** — $J_{\det}(\phi)(x, y) \geq \tau$:

- $J_{\det}(\phi)(x, y)$ is the Jacobian determinant of the displacement field at pixel $(x, y)$. It measures how much the local area around that pixel stretches or shrinks (explained in the next section).
- $\geq \tau$ means the Jacobian determinant must be at least $\tau = 0.01$ — strictly positive.
- $\forall \; (x, y) \in \Omega$ means this must hold at **every** pixel in the domain.
- **Why not $\geq 0$?** A threshold of exactly zero would leave the optimizer balanced on a knife-edge — any numerical rounding could push a pixel back to negative. The small positive margin ($0.01$) gives a safety buffer.

**$\arg\min$:**

- $\arg\min_\phi$ means "find the $\phi$ that makes the expression as small as possible." So the full statement reads: _find the displacement field $\phi$ that is closest to the original, subject to no pixel having a Jacobian determinant below 0.01._

## Jacobian Determinant

### 2D Computation

For a displacement field with components $(u_x, u_y)$, the deformation gradient is:

$$F = I + \nabla u = \begin{pmatrix} 1 + \frac{\partial u_x}{\partial x} & \frac{\partial u_x}{\partial y} \[4pt] \frac{\partial u_y}{\partial x} & 1 + \frac{\partial u_y}{\partial y} \end{pmatrix}$$

The Jacobian determinant is the determinant of $F$:

$$J_{\det} = \left(1 + \frac{\partial u_x}{\partial x}\right)\left(1 + \frac{\partial u_y}{\partial y}\right) - \frac{\partial u_x}{\partial y} \cdot \frac{\partial u_y}{\partial x}$$

Spatial derivatives are computed via `np.gradient` (central differences at interior pixels, one-sided at boundaries). This matches SimpleITK for interior pixels while avoiding the ~3 ms/call overhead that made SLSQP numerical gradients infeasible.

#### What each piece means

**$u_x$ and $u_y$** are the horizontal and vertical components of the displacement at each pixel. They say "move this pixel _this far_ in x and _this far_ in y."

**$\frac{\partial u_x}{\partial x}$** is the rate of change of the x-displacement as you step one pixel to the right. If your neighbor's x-displacement is much larger than yours, this derivative is large — meaning the spacing between deformed pixels is stretching in the x-direction.

**The $1 +$ terms.** With zero displacement, adjacent pixels are exactly 1 unit apart (pixel spacing). The deformation gradient captures the _total_ spacing after deformation: original spacing ($1$) plus the change from displacement ($\frac{\partial u}{\partial \cdot}$). So $1 + \frac{\partial u_x}{\partial x}$ is the deformed x-spacing in the x-direction.

**Why a $2 \times 2$ matrix?** Each pixel's displacement has 2 components ($u_x, u_y$), and each can vary along 2 directions ($x, y$). The $2 \times 2$ deformation gradient $F$ captures all four combinations: how x-displacement changes in x, how x-displacement changes in y, how y-displacement changes in x, and how y-displacement changes in y.

**The determinant formula** — $(1 + \frac{\partial u_x}{\partial x})(1 + \frac{\partial u_y}{\partial y}) - \frac{\partial u_x}{\partial y} \cdot \frac{\partial u_y}{\partial x}$:

- The first product is the "diagonal" contribution — how much the grid stretches independently along x and y. If both are positive, the grid is locally expanding or compressing but not rotating.
- The second product is the "off-diagonal" or shear contribution — how much each displacement component leans into the other axis. This captures rotation and skew.
- Subtracting the shear from the diagonal gives the **signed area** of the deformed unit cell. Positive means the cell kept its orientation (no fold). Negative means it flipped — the grid crossed over itself.

**Intuition:** Think of a small square of paper. Stretch it, compress it, or rotate it — $J_{\det}$ stays positive. But fold it over (flip it inside-out) and $J_{\det}$ goes negative. That fold is exactly what we need to fix.

#### Interpreting the value

- $J_{\det} = 1$: no deformation (identity)
- $J_{\det} > 1$: local expansion
- $0 < J_{\det} < 1$: local compression
- $J_{\det} \leq 0$: **folding** (invalid, needs correction)

### 3D Computation

The 3D deformation gradient $F \in \mathbb{R}^{3 \times 3}$. Its determinant is computed via cofactor expansion along the first row:

$$\det(F) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

where $a_{ij} = \delta_{ij} + \frac{\partial u_j}{\partial x_i}$, computed from 9 partial derivatives via `np.gradient`.

## Correction Methods

### 1. Heuristic (Neighborhood Mean Vector Filter)

**Notebook:** `heuristic-neg-jacobian.ipynb`

The fastest but least accurate method. Replaces displacement vectors in the neighborhood of negative-Jdet pixels with the local mean:

1. Compute Jacobian determinant over the entire field
2. Find all pixels where $J_{\det} \leq 0$
3. For each negative pixel at $(y, x)$:
   - Extract the 3×3 neighborhood centered on $(y, x)$
   - Compute the mean displacement vector (excluding the center)
   - Replace displacement vectors in the 3×3 neighborhood with this mean
4. Repeat until no negative Jacobian determinants remain (or max iterations)

**Tradeoff:** Fastest convergence, but highest L2 error (largest deviation from original field). No optimality guarantee.

### 2. Full-Grid SLSQP

**Entry point:** `_full_grid_step()` in `modules/dvfopt.py`

Optimizes the entire displacement field simultaneously via Sequential Least Squares Programming (SLSQP):

1. Flatten the full $(2, H, W)$ displacement field into a single vector $\phi$
2. Run `scipy.optimize.minimize(method='SLSQP')` with:
   - **Objective:** $\min \|\phi - \phi_{\text{init}}\|_2$
   - **Constraint:** $J_{\det}(\phi)(x, y) \geq 0.01$ for all interior pixels
3. Reshape the result back to $(2, H, W)$

**Tradeoff:** Lowest L2 error (globally optimal), but slowest — runtime scales quadratically with grid size since all pixels are optimized together.

### 3. Iterative SLSQP

**Entry point:** `iterative_with_jacobians2()` in `modules/dvfopt.py`

The primary method. Instead of optimizing the full grid, it repeatedly identifies the worst folding region and corrects it with a small local optimization:

```
1. Compute Jacobian determinant over full grid
2. While any interior pixel has J_det <= threshold - ε:
   a. Find the interior pixel with the lowest J_det
   b. Identify the connected component of negative-J_det pixels
      around it (8-connectivity)
   c. Compute the bounding box of that component,
      expand by +1 pixel on all sides (minimum 3×3)
   d. Extract the sub-window of displacement vectors
   e. Freeze edge pixels of the window to their initial values
   f. Run SLSQP on the sub-window:
      - Objective: minimize L2 distance from initial
      - Constraint: J_det >= 0.01 at all interior window pixels
      - Linear constraint: edge pixels remain fixed
   g. Write corrected sub-window back into the full field
   h. If window didn't converge, grow by +2 pixels and retry
   i. If window reached full grid size, run full-grid fallback
3. Return corrected field
```

**Key design choices:**

- **Adaptive window sizing:** The bounding-box approach ensures each window is only as large as needed, keeping the per-step optimization fast.
- **Frozen edges:** Edge pixels of each sub-window are constrained to their initial values. This prevents "pushing" negativity outside the window. The +1 positive-border expansion ensures frozen edges are feasible.
- **Edge feasibility check:** Before running the optimizer, the algorithm verifies that frozen edge pixels have positive Jacobian determinants. If not, it grows the window (by +2 pixels each dimension) and retries.
- **Full-grid fallback:** For non-square grids where the window has grown to grid size, falls back to full-grid SLSQP as a last resort.

**Tradeoff:** Near-optimal L2 error with substantially faster runtime than full-grid SLSQP, especially for fields with sparse folding regions.

### Hybrid Parallel Variant

**Entry point:** `iterative_parallel()` in `modules/dvfopt.py`

Extends the iterative method to process multiple non-overlapping windows simultaneously:

1. Identify **all** negative-Jdet pixels, sorted worst-first
2. Assign each an adaptive window size (bounding box + border)
3. **Greedy batch selection:** Pick the worst pixel's window, then greedily add windows that don't overlap any selected window
4. **Execute batch:**
   - Single window → run serially (avoids process-spawn overhead)
   - Multiple windows → dispatch to `ProcessPoolExecutor` in parallel
5. Apply all results, recompute Jacobian, repeat
6. **Escalation:** If a batch produces no improvement, force all windows to grow by +2 globally, increasing the chance of convergence at the cost of serialization

## Constraints

### Primary: Jacobian Determinant

The core constraint ensuring no folding:

$$J_{\det}(\phi)(x, y) \geq 0.01 \quad \forall \; \text{interior pixels}$$

Implemented as a `NonlinearConstraint` in SciPy's SLSQP. The threshold is strictly positive (not $\geq 0$) to provide a margin against numerical drift.

**Why only interior pixels?** Edge pixels of the grid have one-sided finite differences (only one neighbor instead of two), making their Jacobian determinant less reliable. Constraining them would force the optimizer to satisfy a less accurate measurement. Interior pixels use symmetric central differences, giving a trustworthy value to constrain.

**Why nonlinear?** The Jacobian determinant involves _products_ of displacement components (the $\frac{\partial u_x}{\partial x} \cdot \frac{\partial u_y}{\partial y}$ term), so it is a nonlinear function of the optimization variables. SLSQP handles this by locally linearizing the constraint at each iteration step.

### Frozen Edges (Linear)

Edge pixels of each optimization window are fixed:

$$\phi_k = \phi_{k}^{\text{init}} \quad \forall \; k \in \text{edge indices}$$

Implemented as a `LinearConstraint` with a selection matrix $A$ that picks edge pixel entries from the flattened $\phi$ vector.

**What this means in practice:** When we extract a small sub-window from the full field, the outermost ring of pixels in that window is "frozen" — the optimizer is not allowed to change them. Only the interior of the window is free to move.

**Why freeze edges?** Without this, the optimizer could "solve" a local folding problem by pushing the bad displacement outward — making the window look clean while creating a new fold just outside it. Freezing the border forces the correction to be self-contained: the fix must happen _within_ the interior of the window.

**Why linear?** Each frozen-edge constraint is just "this variable equals this constant" — that is a linear equation. Linear constraints are cheap for the optimizer to enforce (no iteration needed), and they never cause convergence issues.

### Optional: Shoelace Area

Cell-based fold detection via the shoelace formula for signed area of each deformed quadrilateral cell:

$$A_{\text{cell}} = \frac{1}{2}\left|(x_0 y_1 - x_1 y_0) + (x_1 y_2 - x_2 y_1) + (x_2 y_3 - x_3 y_2) + (x_3 y_0 - x_0 y_3)\right|$$

where $(x_0, y_0), \ldots, (x_3, y_3)$ are the deformed quad vertices (TL, TR, BR, BL). Positive signed area means no geometric fold. Enabled via `enforce_shoelace=True`.

**How this differs from the Jacobian constraint:** The Jacobian determinant is computed _per pixel_ using finite-difference derivatives — it is a continuous, differential measure. The shoelace area is computed _per cell_ (the quadrilateral formed by four neighboring pixels) using exact vertex positions — it is a discrete, geometric measure. A cell can have positive Jacobian at all four corners but still be geometrically folded if the quad self-intersects (a "bowtie" shape). The shoelace constraint catches these cases.

**The shoelace formula itself:** Imagine walking around the four corners of the deformed cell in order (TL → TR → BR → BL). At each step, compute the cross product of consecutive vertex positions. Sum them up and divide by 2 — you get the signed area of the polygon. If the polygon is traced counterclockwise, the area is positive. If the quad folds over and you end up tracing clockwise, the area goes negative.

### Optional: Injectivity (Monotonicity)

Forward-difference check ensuring the deformed grid remains monotonically ordered:

$$1 + u_x[i, j+1] - u_x[i, j] > 0 \quad \text{(horizontal)}$$
$$1 + u_y[i+1, j] - u_y[i, j] > 0 \quad \text{(vertical)}$$

This is a sufficient (but not necessary) condition for global injectivity on a structured grid with unit spacing. Enabled via `enforce_injectivity=True`.

**Reading the formula:** On the original grid, adjacent pixels are 1 unit apart. After deformation, pixel $(i, j)$ moves to position $j + u_x[i, j]$ and its right neighbor moves to $j + 1 + u_x[i, j+1]$. The gap between them is:

$$(j + 1 + u_x[i, j+1]) - (j + u_x[i, j]) = 1 + u_x[i, j+1] - u_x[i, j]$$

If this gap is positive, the right neighbor is still to the right after deformation — no crossing. If it goes to zero or negative, the two pixels have swapped order — a fold.

**Why "sufficient but not necessary"?** Monotonicity along grid rows and columns guarantees injectivity, but there exist injective (non-folding) deformations where a pixel briefly moves past a neighbor along one axis while being pulled back by displacement along the other axis. Such fields would fail the monotonicity check even though they don't actually fold. In practice this is rare, and the constraint is useful as a stronger-than-needed guarantee when desired.

## 3D Extension

The 3D extension in `modules/dvfopt3d.py` generalizes all 2D operations:

| Aspect | 2D | 3D |
|--------|----|----|
| Deformation shape | $(3, 1, H, W)$ | $(3, D, H, W)$ |
| Jacobian matrix | $2 \times 2$ | $3 \times 3$ determinant via cofactor expansion |
| Connected component | 8-connectivity | 26-connectivity (3×3×3 structuring element) |
| Frozen boundary | 4 edges | 6 faces of sub-volume |
| Phi packing | $[\text{dx}, \text{dy}]$ | $[\text{dx}, \text{dy}, \text{dz}]$ |
| Constraint indices | 2 per pixel | 3 per voxel |

The iterative algorithm is structurally identical — find worst voxel, compute 3D bounding box, extract sub-volume, freeze 6 faces, run SLSQP, write back.

## Data Conventions

- **Deformation fields:** `(3, 1, H, W)` NumPy arrays with channels `[dz, dy, dx]`. For 2D work the z-slice dimension is 1.
- **Pull-back convention:** Each displacement vector points from a fixed-image pixel to its source in the moving image: $\text{fixed} + \text{displacement} = \text{moving}$.
- **Coordinate ordering:** `[z, y, x]` everywhere. Correspondences are `(N, 3)` arrays.
- **SimpleITK interop:** Arrays are transposed `(3,1,H,W)` → `(1,H,W,3)` and axis-reordered `[2,1,0]` (zyx → xyz) before calling SimpleITK.
- **Phi flattening:** During optimization, the displacement field is flattened as `phi[:len(phi)//2]` = dy, `phi[len(phi)//2:]` = dx (2D) or `[dx, dy, dz]` concatenated (3D).
- **Plotting:** `meshgrid` with `indexing='xy'`; y-axis inverted to match image convention.

## Project Structure

```
├── modules/                        # Shared Python modules
│   ├── dvfopt.py                   # Core 2D optimization (serial + parallel)
│   ├── dvfopt3d.py                 # 3D extension
│   ├── dvfviz.py                   # Visualization utilities
│   ├── jacobian.py                 # SimpleITK Jacobian wrapper
│   ├── laplacian.py                # Laplacian interpolation from correspondences
│   ├── correspondences.py          # Point correspondence utilities
│   └── testcases.py                # Test case registry and data loaders
│
├── slsqp-iterative-refactored.ipynb    # Iterative SLSQP — primary notebook
├── heuristic-neg-jacobian.ipynb        # Heuristic (NMVF) correction
├── slsqp-3d.ipynb                      # 3D correction tests
├── run-parallel-corrections.ipynb      # Batch parallel corrections
├── generate_test_cases.ipynb           # Generate and save test data
├── test-shoelace-constraint.ipynb      # Shoelace constraint tests
├── test-injectivity-constraint.ipynb   # Injectivity constraint tests
│
├── benchmarks/                     # Benchmark notebooks
│   ├── benchmark-serial-vs-parallel.ipynb
│   ├── voxelmorph-registration.ipynb
│   └── correct-ants-warps.ipynb
│
├── data/                           # Test case data (.npy files)
├── output/                         # Correction output results
├── archive/                        # Historical notebook iterations
└── legacy_code/                    # Previous notebook versions
```

### Module Reference

| Module | Purpose |
|--------|---------|
| `dvfopt.py` | Objective/constraint functions, `_numpy_jdet_2d`, `iterative_with_jacobians2` (serial), `iterative_parallel` (hybrid), `generate_random_dvf`, `scale_dvf`, `neg_jdet_bounding_window` |
| `dvfopt3d.py` | 3D Jacobian (`_numpy_jdet_3d`), 26-connectivity windowing, 6-face boundary freezing, `iterative_3d` |
| `dvfviz.py` | `plot_deformations` (2×2 panel), `plot_grid_before_after` (colored quad grids), `plot_step_snapshot` (per-iteration heatmap), `run_lapl_and_correction` (end-to-end pipeline) |
| `laplacian.py` | Sparse Laplacian matrix with Dirichlet BCs, LGMRES solver for displacement interpolation from correspondences |
| `testcases.py` | `SYNTHETIC_CASES` (8 correspondence-based), `RANDOM_DVF_CASES` (4 random), `REAL_DATA_SLICES` (8 real-data configs) |
| `correspondences.py` | Duplicate removal, line intersection detection, correspondence swapping, downsampling |

## Test Cases

### Synthetic (Correspondence-Based)

Defined in `testcases.py` as `SYNTHETIC_CASES`. Deformation fields constructed by solving a Laplacian system with Dirichlet boundary conditions at correspondence points:

$$\nabla^2 u = 0 \quad \text{(interior)}, \qquad u(\mathbf{p}_i) = \mathbf{m}_i - \mathbf{f}_i \quad \text{(correspondences)}$$

Types include crossing vectors, opposing vectors, and checkerboard patterns on 10×10 and 20×20 grids.

### Random DVFs

Generated via `generate_random_dvf(shape, max_magnitude)` — uniform random displacements optionally rescaled via bicubic interpolation (`scale_dvf`).

### Real Data

Axial slices from ANTs registration warps (`.npy` files), available at full resolution (320×456) and downscaled (64×91).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Dependencies

`numpy`, `scipy` (SLSQP optimizer, sparse LGMRES), `SimpleITK` (Jacobian computation), `nibabel` (NIfTI I/O), `matplotlib` (visualization).

## Usage

### Iterative SLSQP (Recommended)

```python
from modules.dvfopt import iterative_parallel, jacobian_det2D

# deformation: (3, 1, H, W) numpy array with channels [dz, dy, dx]
phi_corrected = iterative_parallel(
    deformation,
    method="SLSQP",
    verbose=1,
    max_workers=4,               # parallel workers (None = auto)
    enforce_shoelace=False,      # optional shoelace constraint
    enforce_injectivity=False,   # optional injectivity constraint
)

# Verify
jdet = jacobian_det2D(phi_corrected)
assert jdet.min() > 0
```

### Heuristic (Fast)

```python
# See heuristic-neg-jacobian.ipynb
phi_corrected = heuristic_negative_jacobian_correction(deformation, max_iter=1000)
```

### 3D Volumes

```python
from modules.dvfopt3d import iterative_3d, jacobian_det3D

# deformation: (3, D, H, W) numpy array
phi_corrected = iterative_3d(deformation, method="SLSQP", verbose=1)
```

### Output Format

Corrections save to a directory containing:

| File | Contents |
|------|----------|
| `results.txt` | Settings, runtime, L2 error, neg-Jdet summary |
| `phi.npy` | Corrected displacement field |
| `error_list_l2.npy` | Per-iteration L2 error |
| `num_neg_jac.npy` | Per-iteration negative-Jdet count |
| `min_jdet_list.npy` | Per-iteration minimum Jdet |
| `iter_times.npy` | Per-iteration wall time |
| `window_counts.csv` | Window size histogram (iterative only) |

