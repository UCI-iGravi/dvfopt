# Performance Optimization Design: dvfopt Package

## Context

The dvfopt package corrects negative Jacobian determinants in 2D/3D deformation fields using iterative SLSQP optimization. Profiling reveals that the dominant cost is **redundant constraint evaluation** — SLSQP estimates gradients via finite differencing, calling the Jacobian-determinant constraint function ~N extra times per step (where N = number of optimization variables). A 7x7 window has 98 variables, meaning ~98 extra constraint calls per SLSQP step, each recomputing `_numpy_jdet_2d`.

Secondary costs: full-grid Jacobian recomputation after each window optimization, dense constraint matrix allocation, redundant `_quality_map` Jacobian calls, O(n^2) overlap checking in parallel batch selection, and inefficient Laplacian matrix construction.

## Approach

Provide SLSQP with analytical gradient functions to eliminate finite-difference overhead. Combined with caching, sparse matrices, and parallel improvements to reduce per-call cost of remaining evaluations.

## Changes

### P1: Analytical Objective Gradient

**File:** `dvfopt/core/objective.py`

Current `objectiveEuc(phi, phi_init)` returns `np.linalg.norm(phi - phi_init)` with no gradient. SLSQP estimates the gradient via N+1 function evaluations.

Add `objectiveEuc_with_grad(phi, phi_init)` returning `(value, gradient)` where `gradient = (phi - phi_init) / norm`. Update `_optimize_single_window` and `_optimize_single_window_3d` to pass `jac=True` to `minimize()`.

### P2: Analytical Jdet Constraint Gradient (2D)

**New file:** `dvfopt/core/gradients.py`

The 2D Jacobian determinant at interior pixel (i,j) using central differences is:
```
J(i,j) = (1 + (dx[i,j+1] - dx[i,j-1])/2) * (1 + (dy[i+1,j] - dy[i-1,j])/2)
        - ((dx[i+1,j] - dx[i-1,j])/2) * ((dy[i,j+1] - dy[i,j-1])/2)
```

The gradient of J(i,j) w.r.t. each phi variable is nonzero only for the 3x3 stencil neighborhood. Implement `jdet_constraint_jacobian_2d(phi_flat, submatrix_size, ...)` returning a sparse CSR matrix of shape `(n_constraints, n_variables)`.

Update `_build_constraints` in `constraints.py` to attach this as the `jac` parameter on `NonlinearConstraint`.

### P3: Analytical Jdet Constraint Gradient (3D)

**New file:** `dvfopt/core/gradients3d.py`

3D Jacobian determinant is `det(I + nabla u)` — a 3x3 determinant with 9 partial derivatives. Gradient uses the cofactor matrix. Each J(i,j,k) depends on a 3x3x3 stencil across 3 components.

Implement `jdet_constraint_jacobian_3d(phi_flat, subvolume_size, ...)` returning sparse CSR matrix. Update `_build_constraints_3d` in `constraints3d.py`.

### P4: Patch-Based Jacobian Update

**Files:** `dvfopt/core/solver.py`, `dvfopt/core/solver3d.py`

After `_apply_result` writes optimized values back into phi, `_update_metrics` currently recomputes `jacobian_det2D(phi)` over the entire grid. For a 7x7 window on a 100x100 grid, this recomputes 10,000 pixels when only ~81 changed (window + 1px gradient border).

Add `_patch_jacobian(jacobian_matrix, phi, cy, cx, sub_size)` that recomputes only the affected sub-region + 1px border and writes it back into the existing Jacobian array. The full-grid recompute becomes the initial computation only.

Shared helper (works for both 2D and 3D): takes the cached Jacobian array, phi, window center/size, and grid shape. Computes the expanded region bounds, slices phi, calls the appropriate `_numpy_jdet` on the sub-region, and patches the result back.

### P5: Pass Cached Jacobian into `_quality_map`

**Files:** `dvfopt/core/constraints.py`, `dvfopt/core/solver.py`

`_quality_map()` internally calls `jacobian_det2D(phi)` but the caller already has the Jacobian. Add optional `jacobian_matrix` parameter; when provided, skip recomputation.

### P6: Shoelace and Injectivity Analytical Gradients

**File:** `dvfopt/core/gradients.py`

Shoelace area is bilinear in 4 corner displacements — gradient is linear, 8 nonzero entries per cell. Implement as sparse matrix.

Injectivity diffs are `1 + diff(dx)` and `1 + diff(dy)` — gradient is +1/-1 at the two involved pixels. Trivially sparse.

Attach to respective `NonlinearConstraint` objects in `_build_constraints`.

### P7: Sparse Frozen-Edge Constraint Matrices

**Files:** `dvfopt/core/constraints.py`, `dvfopt/core/constraints3d.py`

Replace:
```python
A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
for row, col in enumerate(fixed_indices):
    A_eq[row, col] = 1
```

With:
```python
A_eq = scipy.sparse.csr_matrix(
    (np.ones(len(fixed_indices)), (np.arange(len(fixed_indices)), fixed_indices)),
    shape=(len(fixed_indices), phi_sub_flat.size)
)
```

`LinearConstraint` accepts sparse matrices natively.

### P8: Cache Shoelace Reference Grid

**File:** `dvfopt/jacobian/shoelace.py`

`_shoelace_areas_2d()` creates `np.mgrid[:H, :W]` on every call. Add module-level LRU cache keyed on `(H, W)` so the reference grid is allocated once per window size.

### P9: Occupancy-Grid Batch Selection

**File:** `dvfopt/core/spatial.py`

Replace O(n^2) pairwise overlap checking in `_select_non_overlapping()` with a boolean occupancy grid. When selecting a window, mark its bounding box in the grid. To check a candidate, test if any cells in its bounding box are already marked. O(window_area) per candidate instead of O(n_selected).

### P10: Reduce Parallel Serialization

**File:** `dvfopt/core/parallel.py`

Pre-extract all `phi_sub_flat` and `phi_init_sub_flat` arrays in the main process. Submit worker functions that accept only array data and scalar parameters — no closures capturing mutable state.

### P11: Boolean Mask Indexing in Laplacian Builder

**File:** `dvfopt/laplacian/matrix.py`

Replace:
```python
invalid = np.concatenate([np.where(X == 0)[0], boundaryIndices])
rids = np.delete(ids_0, invalid)
cids = np.delete(cids, invalid)
```

With:
```python
valid = (X > 0)
valid[boundaryIndices] = False
rids = ids_0[valid]
cids = cids_x1[valid]
```

Eliminates 12 `np.delete()` calls (each O(n) with copy). Also replace meshgrid+flatten with direct arithmetic from `np.arange`.

### P12: In-Place Objective Computation

**File:** `dvfopt/core/objective.py`

Replace `np.linalg.norm(phi - phi_init)` with `np.sqrt(np.dot(d, d))` where `d = phi - phi_init`, or use `np.sum((phi - phi_init)**2)` to avoid intermediate norm computation. Minor savings, but called thousands of times.

## Validation Strategy

Each optimization must produce numerically identical results within floating-point tolerance:

1. **Correctness:** Run existing benchmark notebooks (`benchmark-scalability.ipynb`, `benchmark-serial-vs-parallel.ipynb`, `benchmark-constraint-modes.ipynb`). Compare final phi arrays (within ~1e-10), final neg-Jdet counts (identical), min Jdet values.
2. **Performance:** Compare wall-clock times before/after on the scalability benchmark (10x10 through 200x200).
3. **Gradient validation:** For P1-P3/P6, validate analytical gradients against finite-difference approximations using `scipy.optimize.check_grad` or manual epsilon-perturbation on small test cases.

## Implementation Order

```
P1 (objective gradient)     → simple, validates the jac=True plumbing
P2 (2D Jdet gradient)       → highest impact, validate on small grids first
P4 (patch-based update)     → independent of P2, large standalone impact
P5 (cached quality_map)     → small change, piggybacks on P4
P7 (sparse constraints)     → independent, drop-in replacement
P6 (shoelace/inject grads)  → extends P2 machinery
P3 (3D Jdet gradient)       → port P2 approach to 3D
P8-P12                      → independent small wins, any order
```

## What We're NOT Doing

| Idea | Why Not |
|------|---------|
| Numba JIT for Jacobian kernel | Doesn't reduce call count (the real problem); adds dependency |
| Generic ndim gradient implementation | 2D formula (4-term product) and 3D formula (3x3 cofactor expansion) are structurally different; generic version would be slower |
| GPU acceleration | Overhead of CPU-GPU transfer dominates for typical grid sizes (10x10 to 200x200) |
| Replace SLSQP with different optimizer | SLSQP handles the mixed equality/inequality constraints well; alternatives (IPOPT, COBYLA) would require significant rearchitecting |
| Cython compilation | Maintenance burden; NumPy vectorization with analytical gradients should be sufficient |
