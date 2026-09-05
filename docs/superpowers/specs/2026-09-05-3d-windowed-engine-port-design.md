# 3D port of the windowed I-SLSQP engine — design

2026-09-05. Extends *the* method — the no-damage, cluster-windowed, elastic-QP
SQP engine (`dvfopt/core/windowed/`, `ISQPWindowedStrategy`) with its
prevention rows, giant-region Schwarz tiler, coarse-to-fine warm start, mop and
harmonic re-seed — from `(2, H, W)` fields on the 2D simplex families to
`(3, D, H, W)` fields on `SimplexConstraint3D` (6-tet simplicial Jacobian).

**Why this and not the alternatives.** The 2.5D marching chain (shipped,
findings §10) is a different algorithm and requires `dz ≡ 0`: a slice-stack
fast path, not "3D". `correct_dvf_3d` is true-3D *repair* but from the
ALM/wallbreaker family. The paper's 3D claim is the windowed engine itself
in 3D, with the same certificate (0 folds at every gauge) and the same
no-damage invariant.

Evidence base carried over: 528/528 full-res + 203/203 cohort 2D certifications
on stock defaults (0.6.0); the 2.5D full-volume verdict (sweep 66 → 50 folds,
true best-diagonal floor 23 → 12 with the rows); the 3D research thread
(REPORT Parts II–XXIII: the fixed-6-tet floor, per-cell splits nonconforming
by the GF(2) argument, the 16³ moderate-density B0039 sub-volume, Part XX
full-volume scale reality: 1.85 GB per field copy, spawn workers re-load it).

## Goals / non-goals

**Goals.** (1) `windowed_correct` / `ISQPWindowedStrategy` accept
`SimplexConstraint3D` and run end-to-end on 3D fields: windows, prevention
rows, tiler, coarse-to-fine, mop, re-seed, no-damage accounting, checkpoints.
(2) Same certificate semantics: report the fixed-diagonal 6-tet count *and*
the best-of-4-diagonals floor (`n_neg_best_diagonal`) at threshold and at 0.
(3) Measured, gated promotion in the 2D campaign's style: crop pack → chunks →
ds2 volume → full-res, each with 0 folds / damage 0 / wall / L2 move.
(4) `checkpoint_dir` works for the 3D engine like the others (per round /
tile, whole-field units).

**Non-goals.** Jdet3D in the windowed engine (the simplex family is the
certificate; Jdet3D keeps `SLSQPWindowedStrategy`); GPU; changing 2D
behaviour (every 2D path stays byte-identical — the suite's identity gates
enforce it); adaptive/mixed tet decompositions (nonconforming, Part XXIII).

## Architecture — the seams, and what each becomes

The engine is already written against a *per-family adapter*, not against 2D
arrays; 3D is mostly geometry plus cost.

| Seam (today) | 2D assumption | 3D form |
|---|---|---|
| `_locality.LOCALITY[type] = WindowLocality(ring, min_field, influenced)` | `(2,H,W)` field → `(H,W)` fold map; `influenced(c, free_mask, ph, pw, borders)` | register `SimplexConstraint3D`: `ring=1` (tet volumes are exact — a free voxel's influenced cells are its ≤ 8 corner cells, all in-patch once 1 in), `min_field` = `six_tet_min_volume_3d` per cube (`(D-1,H-1,W-1)`), `influenced` = the `_influenced_2tri` idiom over 8 corners with `k = 6` rows/cell; CPR colouring via `adjoint` (`tet_grad_T_v`) or the native `build_tet_sparse_jac` (measure which is cheaper per patch shape) |
| `build_subproblem(constraint, phi, free_box, …)` | `free_box = (fy0,fy1,fx0,fx1)`, `H, W = phi.shape[1:]`, `free_mask (ph,pw)`, `np.stack([m, m])` | boxes become 6-tuples, masks `(pd,ph,pw)`, `np.stack([m]*3)`; everything else (enforced rows, `cons`/`cons_jac`, objective triplet, `WindowSub`) is pack-agnostic already — `WindowSub.patch_box` and `free_mask` just carry one more axis |
| `find_windows(mask, margin, ring)` | 2D dilation/label/`find_objects` | identical code on 3D arrays (`ndimage` is n-D); the border-guard applies per face |
| `_orientation_rows` (edge monotonicity, DY_FIRST 2D) | 2D edge projections + anti-diagonal convexity | reuse `slsqp_windowed.constraints3d._injectivity_linear_constraint_3d` (axial gaps ≥ δ, DX_FIRST `[dx,dy,dz]` blocks, frozen-pair filtering) as the `'edges'` kind; no 3D convexity rows (the 2D `'full'` kind was a measured fidelity cost anyway). Lift the DY_FIRST gate in `windowed_correct` to "family provides rows" |
| `isqp_solve` step rule | `'exact_ls'` needs rows bilinear along a line (`c(a)=c+a Jd+a²q`) | 6-tet rows are trilinear → cubic along the line. Phase 1 runs `'tr'` (byte-tested path; the 2D gate at `windowed_correct` already forbids `exact_ls` on 3D). Phase 3 (only if measured load-bearing): cubic exact line-min — fit `c(a)` from `cons` at `a = 0, ½, 1` plus `J d` (one extra evaluation), same breakpoint sweep on a cubic merit |
| giant tiler `_solve_giant_schwarz` / `_fit_tile` | 2D tiles stepping by `tile − (2 ring + 2)`, RAS cores | 3D tiles (target `giant_tile` per axis, fitted per axis), same sweep/RAS logic; the region-area cap becomes a voxel cap |
| coarse-to-fine `_restrict`/`_prolongate` | `reshape(2, hc, f, wc, f).mean` and 2D `ndimage.zoom` | 3-axis block mean and 3D zoom (order 1); displacement rescale unchanged |
| mop `_mop_pass`, re-seed `_reseed_stage` (`_harmonic_fill`) | 2D clusters, 2D harmonic fill | n-D `ndimage` clusters; a 3D harmonic fill (7-point Laplacian, same solver) |
| damage / `touched` / `min_field` reporting | 2D masks | n-D masks; `SliceReport` gains `n_neg_best_diag` at threshold and at 0 |
| `RunCheckpoint` | units: z-slices or whole field | 3D engine: whole-field unit per round (`round:<k>`) and per giant tile (`tile:<k>:<i>`), `finish()` after the mop/re-seed |

Everything in `isqp.py` (OSQP/Clarabel/QPALM backends, hybrid policy,
trust region, a*-collapse bail, `feas_tol`/`ftol`) is pack- and
dimension-agnostic and is **not touched**.

## Components and sequencing (one PR each, gated)

1. **Family plumbing** — `LOCALITY[SimplexConstraint3D]`, n-D `build_subproblem`
   / `find_windows` / masks, `WindowedWrapperStrategy.supports_3d = True` for the
   simplex family, the 3D rows kind, certificate reporting. Gate: unit tests
   (rows/Jacobian vs finite differences; influenced-cell sets; paste-back
   no-damage) + the 5×10×10 `testcases_3d` and the 16³ moderate sub-volume
   reach 0 fixed-diagonal folds, damage 0, on `'tr'`.
2. **Tiler, coarse-to-fine, mop, re-seed in 3D** — the round loop runs the
   whole ladder. Gate: a 3D crop pack cut from B0039 the way `make_hard_crops.py`
   cut the 2D one (a twist, a dense cluster, a sliver, each ≤ 48³), all 0
   folds / damage 0; wall and L2 move recorded as the reference table.
3. **Inner cost** — measure per-QP cost vs window volume on the crop pack
   (expect 10–20× a 64² tile's variables at 32³; ADMM convergence dominates as
   in 2D). Then, only as measured: the cubic exact line-min; `qp_max_iter` /
   hybrid thresholds re-tuned for 3D; `giant_tile` per-axis defaults.
4. **Scale** — chunked full-volume driver (Part XX's memory reality: one
   field copy per process, workers get patches only), `checkpoint_dir` per
   round/tile, ds2 cohort volumes, then the full-res B0039. Certification =
   0 folds at fixed-diagonal 6-tet, the best-diagonal floor reported alongside,
   damage 0, L2 move.

Phase 1 is where the two unknowns that decide the plan get measured: the 3D
per-QP cost, and whether the ratio test alone converges on real 3D windows.

## Data flow / conventions

`(3, D, H, W)` `[dz, dy, dx]`; `SimplexConstraint3D.pack = DX_FIRST`
(`[dx, dy, dz]` flat). The engine already asserts pack lengths at the one
place it mixes families (`core/marching`); the 3D windowed path is single-pack
(DX_FIRST) end-to-end — the DY_FIRST-only gate on the rows is the one place a
pack check must change. Windows are boxes `(fz0,fz1,fy0,fy1,fx0,fx1)`;
patch = box grown by the ring, clipped to the volume; free voxels = box (∩
`free_extra`); enforced rows = the 6 tets of every cell with a free corner.

## Error handling

`windowed_correct` keeps its entry gates: `exact_ls` on a 3D field raises
until phase 3 ships it; an unregistered family raises the existing
`IncompatibleConstraintError` path; a 3D field on a 2D-only knob
(`orientation_rows='full'`) raises with the reason. Memory: the driver never
holds more than one full-field copy plus per-window patches (Part XX).

## Testing

- Byte-identity gates: every existing 2D test and the invariant suite stay
  green unchanged (the 3D path is additive).
- 3D unit tests: rows/Jacobian vs finite differences on random patches;
  `find_windows`/`build_subproblem` geometry (ring, borders, free sets); the
  no-damage invariant (untouched voxels bit-identical; damage 0); the
  certificate (fixed vs best-diagonal counts) on planted folds.
- Gate scripts (like the 2D crop gate): the 3D crop pack with 0-fold /
  damage-0 assertions and a reference table (SQP iterations, wall, L2).
- asv: a `WindowedEngine3D` bench pinned to iterations and move.

## Risks

- **Cost.** A 32³ window is ~98k variables / ~180k rows; minutes per QP is
  plausible, and the box is memory-bandwidth bound (~4 useful workers).
  Mitigations already in the design: coarse-to-fine, giant tiler with RAS,
  `qp_max_iter`, hybrid IP, and the futility mechanisms. Phase 3 exists to
  measure before tuning.
- **Convergence without `exact_ls`.** In 2D the exact line search was
  −18 %/−28 % (wall/iterations) and removed a crop regression only with its
  bail; 3D starts on `'tr'`. If 3D windows plateau, the cubic line-min is the
  planned lever, not a new mechanism.
- **The certificate.** The fixed 6-tet split has artifacts (Part XXIII: of 33
  residuals, 19 were re-split artifacts, 14 true floor). Report both counts;
  never claim the fixed-diagonal count as the floor.
- **Rotated-branch analogue.** Expect a 3D trap set; the edge rows are the
  prevention, the harmonic re-seed the repair — both port directly.

## Open questions (decide in phase 1 by measurement)

1. CPR colouring via `adjoint` vs the native sparse tet Jacobian per patch.
2. `giant_tile` per axis for 3D (2D: 64, fitted); the voxel cap that replaces
   `max_window_area`.
3. Whether 3D windows need the patience rung and backend rung at all.

## Artefacts to build the gates from

`data/dvfs/testcases_3d/*.npy` (5×10×10 planted cases),
`research/strict_feasibility_3d/runners/output/b0039_subvol_16_moderate.npy`
(16³, ~11 % folded tets), `benchmarks/make_hard_crops.py` (to cut a 3D crop
pack), the full-res B0039 (`data/dvfs/b0039/`), ds2 cohort volumes.
