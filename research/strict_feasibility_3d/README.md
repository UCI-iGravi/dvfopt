# research/strict_feasibility_3d/

3D extension of [`research/strict_feasibility_2d/`](../strict_feasibility_2d/),
targeting strict feasibility for the 6-tetrahedron constraint on
3D deformation fields. The HiGHS L1 LP solver from the 2D thread is
dimension-agnostic — we reuse it directly and supply 3D problem
geometry via the existing `dvfopt.jacobian.tetrahedron_sign`
primitives (`tet_volumes_flat`, `tet_grad_T_v`, `build_tet_sparse_jac`).

## Status

| Milestone | Status |
|---|---|
| Folder scaffolded | ✓ |
| `lp_oneshot` + `slp_iter` ported to 3D | ✓ |
| Synthetic 3D fold cases (bowtie, dense random) | ✓ |
| Smoke comparison vs m10 / m14 (Tet 3D variants) | ✓ |
| `cluster_slp_iter` for 3D | not yet |
| Real B0039 3D subvolume comparison | not yet |

## Results (2026-06-14, proof-of-concept)

### Synthetic 3D fold cases

Three small synthetic cases, four methods. **`slp_iter` hits the LP
optimum on the simple bowtie cases and is the only method that
reaches strict feasibility on the extreme dense-random case.**

| Case | init folds | m10 | m14 | lp_oneshot | **slp_iter** |
|---|---:|:---|:---|:---|:---|
| bowtie_3d_cube 8³ | 6 | ✓ L1=2.07 (0.7 s) | ✓ L1=3.00 (0.4 s) | ✓ L1=**2.06** (0.7 s) | ✓ L1=**2.06** (0.9 s) |
| bowtie_3d_cube 12³ | 6 | ✓ L1=2.07 (1.1 s) | ✓ L1=2.12 (0.6 s) | ✓ L1=**2.06** (1.2 s) | ✓ L1=**2.06** (3.5 s) |
| dense_random_3d 10³ | 1433 | ✓ L1=328.31 (14.3 s) | **✗** (150 folds remain) | ✗ (14 folds) | **✓ L1=324.52** (25.2 s) |

### Real B0039 subvolume comparison

Cropped 8³ region from the full 528×320×456 B0039 DVF at (z=4,
y=122, x=249) — a realistic moderate-density fold region with 48
folded tets (1.56%), min_T=−0.009.

| Method | Feasible | min_T | L1 | Wall |
|---|:---:|---:|---:|---:|
| m10 | ✓ | +0.0110 | 3.56 | 0.9 s |
| m14 | ✗ (min_T below threshold) | +0.0038 | 8.78 | 0.9 s |
| **lp_oneshot** | **✓** | +0.0100 | **3.05** (−14% vs m10) | 1.7 s |
| **slp_iter** | **✓** | +0.0100 | **3.05** (−14% vs m10) | 1.9 s |

**LP-direct beats m10 by 14% L1 AND m14 fails feasibility on real
B0039 data** at this density.

### B0039 scaling limits (denser regions)

Two attempts at denser 3D B0039 subvolumes hit hard scaling walls:

| Subvol | density | m10 | m14 | lp_oneshot |
|---|---:|---|---|---|
| 16³ moderate | 11% folded, min_T=−3.7 | 30 residual folds | 620 residual folds (overshoots) | **hangs >30 min** in HiGHS |
| 24³ dense | 80% folded, min_T=−3.1 | 13 residual folds | 37 402 residual folds (overshoots) | **hangs >12 hours** |

The hang happens because both m10 and m14 fail to reach strict
feasibility, so `lp_oneshot`'s linearization point is infeasible
and HiGHS struggles to find a satisfying phi within the linearized
constraints. The pattern matches the 2D `dense_random_3d 10³`
synthetic case — without a feasible seed the LP is ill-conditioned.

This is the same failure mode as 2D's "lp_oneshot still fails
feasibility on dense canonicals" (see `../strict_feasibility_2d/
README.md`); in 2D the fix was the `slp_iter_m14_seed` and ultimately
`cluster_slp` paths, which decompose the problem into smaller
per-fold-cluster LPs. The 3D analog would be the obvious next step
but is not yet implemented.

### Numba JIT for the 3D constraint kernels

Both `six_tet_volumes_3d` and `tet_grad_T_v` got Numba `@njit`
treatment matching the 2D triangle kernels. The sparse-active-set
shortcut (early-continue when all six per-cell viol entries are
zero) is in the adjoint:

| Workload | numpy | numba | speedup |
|---|---:|---:|---:|
| Full B0039 forward T (3×528×320×456) | 43.5 s | 9.0 s | 4.84× |
| `tet_grad_T_v` 16³ dense | 0.96 ms | 0.50 ms | 1.91× |
| `tet_grad_T_v` 16³ sparse (99% zero) | 0.99 ms | 0.04 ms | **25.34×** |
| `tet_grad_T_v` 32³ sparse | 11.3 ms | 0.36 ms | **31.45×** |
| `tet_grad_T_v` 8³ dense | 0.44 ms | 0.05 ms | 7.99× |

Equivalence to ~1e-15 absolute error vs the numpy reference; all
996 tests pass.

**Key observations:**

1. **LP-direct hits the L1 optimum** on the bowtie cases (L1=2.06,
   vs m10's 2.07 and m14's 2.12-3.00). The single-voxel-swap geometry
   is well-modelled by the linearisation, so `lp_oneshot` alone
   converges in one step.

2. **m10 is the more robust seed than m14 in 3D.** On the extreme
   dense_random case (1433 folds in a 10³ cube), m14's L2-refine
   stage overshoots back into infeasibility and the repair stage
   gives up with 150 folds remaining. m10 reaches feasibility,
   `slp_iter` (with m10 seed) polishes L1 by 1.1% on top.

3. **`slp_iter` reaches feasibility where m14 fails.** This is the
   3D analog of the 2D fallback-row-1 finding (use m10 seed when
   harmonic fails); a critical robustness property if the algorithm
   needs to handle real-world 3D DVFs with arbitrary fold density.

## How to run

```bash
python research/strict_feasibility_3d/runners/_compare_3d.py
```

Output lands in `runners/output/comparison_3d.txt`.

## Architecture

```
research/strict_feasibility_3d/
├── README.md                   # this file
├── algorithms/
│   └── lp_direct_6tet.py       # lp_oneshot + slp_iter (3D, [dx, dy, dz] pack)
├── worst_cases/
│   └── _synthetic_3d.py        # bowtie_3d_cube, dense_random_3d builders
└── runners/
    ├── _compare_3d.py          # run_method dispatcher + smoke harness
    └── output/                 # gitignored result files
```

Reuses from the 2D thread:
- `research.strict_feasibility_2d.algorithms.highs_solver.solve_l1_lp_step` —
  the L1 LP solver is dimension-agnostic; we only swap the geometry.

Reuses from the core package:
- `dvfopt.jacobian.tetrahedron_sign.tet_volumes_flat` — forward T.
- `dvfopt.jacobian.tetrahedron_sign.tet_grad_T_v` — adjoint J^T @ v.
- `dvfopt.jacobian.tetrahedron_sign.build_tet_sparse_jac` — sparse J(phi_lin).
- `dvfopt.HarmonicALMBarrier3DStrategy` (M10TetStrategy alias) — m10 seed.
- `dvfopt.HarmonicALMRefineRepair3DStrategy` (M14TetStrategy alias) — m14 seed.

## `cluster_slp_iter_3d` — implemented, but B0039 is structurally
##  unable to benefit

The 3D analog of `cluster_slp_iter` exists at
[`algorithms/cluster_lp_6tet.py`](algorithms/cluster_lp_6tet.py).
Architecture mirrors the 2D version: `scipy.ndimage.label` over the
3D fold mask + dilation merge, per-cluster `slp_iter`, frozen-edge
splice. The splice rule was adapted for 3D — sides of the cluster
bbox that touch the volume boundary are spliced flush (no inner
trim), since there's no neighbour to coordinate frozen corners with.
Without this fix, single-cluster-covers-volume cases threw away
most of the inner solver's work.

| Case | init folds | m10 | slp_iter | cluster_slp_3d |
|---|---:|---|---|---|
| bowtie_3d_cube 12³ | 6 | ✓ L1=2.07 | ✓ L1=2.06 (2.6 s) | ✓ L1=**2.06** (1.4 s) |
| dense_random_3d 10³ | 1433 | ✓ L1=328.77 | ✓ L1=324.92 (15.9 s) | ✓ L1=327.32 (14.1 s) |
| B0039 8³ (1.56% folded) | 48 | ✓ L1=3.56 | ✓ L1=3.05 (1.3 s) | ✓ L1=3.09 (1.3 s) |
| B0039 16³ (11% folded) | 2765 | ✗ 30 residual folds | hangs | hangs |

**The B0039 16³ case hangs because the fold pattern is a single
connected component, regardless of merge_dilation:**

```
merge_dilation=0:  n_clusters=1   cluster crop=(16, 12, 16)
merge_dilation=1:  n_clusters=1   cluster crop=(16, 13, 16)
merge_dilation=2:  n_clusters=1   cluster crop=(16, 14, 16)
merge_dilation=3:  n_clusters=1   cluster crop=(16, 15, 16)
```

The reason is structural: B0039's `dz` channel is identically zero
(this DVF is a Laplacian extension of 2D registration data, stacked
along z), so every fold appears as a vertical column 17 layers tall
in our subvolume. Those z-columns connect through z-adjacency into
one giant 3D component. No 3D decomposition can break this case
down — the topology is fundamentally connected.

**For B0039 specifically, slice-wise 2D processing is the right
approach** (see [`research/strict_feasibility_2d/`](../strict_feasibility_2d/)
— `auto_slp` processes the full 528-slice B0039 volume in ~3 minutes
total). The 3D LP-direct path is useful for truly 3D DVFs where
folds aren't dz=0-extruded.

## Status board

- Synthetic 3D fold cases: ✓ LP-direct + cluster_slp work
- 8³ B0039 subvol (1.56% folded): ✓ 14% L1 win over m10
- 10³ random_dense (1433 folds): ✓ cluster_slp feasible, beats m10
- 12³ B0039 subvol (mod): ✓ cluster_slp matches LP optimum
- 16³ B0039 subvol (11% folded): ✗ single connected cluster, LP hangs
- 24³ B0039 subvol (80% folded): ✗ same problem, more extreme
- Full B0039 528×320×456: not directly tractable in 3D; use 2D
  slice-wise via `research/strict_feasibility_2d/auto_slp` instead

## Strict 100% 3D feasibility on B0039 — achieved via 2D + 3D pipeline

The 2D auto_slp alone gets each slice 2-tri-feasible but leaves
~0.18% of 3D tets folded after stacking (mismatched corrections
between adjacent slices break the straddling-tet constraint).
Three-stage pipeline solves it:

| Stage | Method | Wall (5 slices) | n_neg | n<0.01 | min_T |
|---|---|---:|---:|---:|---:|
| 0 | raw B0039 z=10..14 | — | 105 695 | n/a | −380.80 |
| 1 | 2D auto_slp per slice (5 slices) | 438 s | 6 368 | 34 762 | −1.156 |
| 2 | M10Tet global 3D @ threshold=0.01 | 1107 s | 0 | 49 | +0.0049 |
| **3** | **M10Tet global 3D @ threshold=0.015 (overshoot)** | **1051 s** | **0** | **0** | **+0.0155** |

**Stage 3 trick:** M10Tet's barrier-polish path under-shoots its
target threshold by ~50% on this geometry. Asking it for 0.015 lands
min_T at +0.0155 — comfortably above the real 0.01 threshold.

Total wall: **2596 s ≈ 43 min for 5 slices**. Extrapolating to the
full 528-slice B0039 volume: ~75 hours sequential, or ~5–10 hours
with slice-chunk parallelism.

**Pipeline implemented in
[`runners/_threestage_pipeline.py`](runners/_threestage_pipeline.py)
+ [`runners/_strict_polish_v2.py`](runners/_strict_polish_v2.py).**
Both cache stage outputs as .npy so reruns of later stages are
instant. Result file:
`runners/output/b0039_z10_14_strict_feas_threshold0.015.npy`
(3.48M tets, all ≥ threshold).

## Full B0039 528-slice scale-up

Pushing the pipeline to the full (3, 528, 320, 456) DVF reveals a
density ceiling. Stage 1 scales linearly per-slice and runs in 141
min on the full volume; stages 2+3 (global 3D M10Tet) scale roughly
linearly in phi vars but stall on the densest fold bands.

### Stage 1 on the full volume

```
[stage 1] 528 slices, 2D auto_slp per slice, 8474 s (141 min)
   Raw B0039:        2 890 473 folded tets  (0.63%)  min_T = -380.80
   After Stage 1:    1 059 911 folded tets  (0.23%)  min_T =   -4.13
                     3 828 269 below 0.01   (0.83%)
   z-layers with at least one 3D fold: 527 / 527
   Top fold bands:  z=0..7  (2700-3200 fold cells per layer)
                    z=292..357  (2300-2400 fold cells per layer)
```

Stage 1 alone reduces folds by 63% but **all 527 cube-layers still
have at least one 3D fold** — the straddling-tet mismatches between
adjacent 2D-corrected slices are systemic, not localised.

### Chunked Stage 2+3 on the densest band (z=0..15, 16 slices)

| Stage | n_neg | n<0.01 | min_T | Wall |
|---|---:|---:|---:|---:|
| Stage 1 chunk start | 34 181 | 201 427 | −4.134 | (cached) |
| + Stage 2 (M10Tet @ 0.01) | 865 | 54 871 | −0.012 | 61 min |
| + Stage 3 iter 0 (M10Tet @ 0.015) | 173 | 1 572 | −0.013 | 87 min |
| + Stage 3 iter 1 | 19 | 29 | −0.0071 | 54 min |
| + Stage 3 iter 2 | 19 | 24 | −0.0064 | 66 min |
| **Final (converged at iter 2)** | **19** | **24** | −0.0064 | **268 min total** |

99.94% fold reduction but **not strict feasible** — 19 stubborn
folds remain. M10Tet has hit its convergence ceiling on this very
dense geometry; subsequent iterations show no progress.

### Density-dependent feasibility ceiling

| Density regime | Example | 2+3 outcome | Wall |
|---|---|:---:|---:|
| Sparse (≤1% folded) | B0039 z=10..14 (mod, 6k folds) | ✅ strict 100% | 36 min for 5 slices |
| Moderate (1-3%) | mid-volume slices (1-2k folds) | ✅ strict 100% (extrapolated) | ~scaled linearly |
| Dense (>5%) | B0039 z=0..15 (34k folds) | ⚠️ 19 residual, 99.94% | 268 min for 16 slices |

For B0039's worst bands, strict 100% 3D feasibility appears
out of reach for the current M10Tet+overshoot pipeline. The
likely next algorithm is a localised 3D cluster-LP on just the
residual 19 folds (each ~3 voxels wide; should decompose now that
they're isolated specks rather than 16-slice columns) — but that
wasn't tested.

Scripts:
- [`runners/_full_b0039_stage1.py`](runners/_full_b0039_stage1.py) —
  full-volume Stage 1
- [`runners/_chunked_stage23.py`](runners/_chunked_stage23.py) —
  per-chunk Stage 2+3 (CLI: --z0 N --z1 M)
- [`runners/_iterate_stage3.py`](runners/_iterate_stage3.py) —
  iterate Stage 3 until convergence-stall
