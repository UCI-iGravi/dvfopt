# 3D port of the windowed I-SLSQP engine — design

2026-09-05. Extends *the* method — the no-damage, cluster-windowed, elastic-QP
SQP engine (`dvfopt/core/windowed/`, `ISQPWindowedStrategy`) with its
prevention rows, giant-region Schwarz tiler, coarse-to-fine warm start, mop and
harmonic re-seed — from `(2, H, W)` fields on the 2D simplex families to
`(3, D, H, W)` fields on `SimplexConstraint3D` (6-tet simplicial Jacobian).

Revised the same day after the seam-by-seam code read that preceded the
phase-1 plan; the paragraphs marked **Review correction** replace what the
first draft said (the rows predicate, the `step_rule` default, the phase-1
cap, the artefact floors, the Jacobian cache, the memory claim, the QP size).

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
enforce it); adaptive/mixed tet decompositions (nonconforming, Part XXIII);
GUI wiring (no `_MID_TO_LABEL` row — the menu parity test is untouched, and
`dvfopt_gui/strategy_params.py` already excludes `supports_3d` from the
editable knobs).

## Architecture — the seams, and what each becomes

The engine is already written against a *per-family adapter*, not against 2D
arrays; 3D is mostly geometry plus cost.

| Seam (today) | 2D assumption | 3D form |
|---|---|---|
| `_locality.LOCALITY[type] = WindowLocality(ring, min_field, influenced)` | `(2,H,W)` field → `(H,W)` fold map; `influenced(c, free_mask, ph, pw, borders)` | register `SimplexConstraint3D`: `ring=1` (tet volumes are exact — a free voxel's influenced cells are its ≤ 8 corner cells, all in-patch once 1 in), `min_field` = `six_tet_min_volume_3d` per cube placed at its corner voxel (`(D-1,H-1,W-1)`, +inf on the last plane/row/column), `influenced` = the `_influenced_2tri` idiom over 8 corners with `k = 6` rows/cell. The adapter call generalises as `loc.influenced(c, free_mask, *free_mask.shape, borders)`: the 2D adapters keep their `(ph, pw)` signature, the 3D one takes `(pd, ph, pw)`. **Review correction (Jacobian):** phase 1 uses the native `build_tet_sparse_jac` (one vectorised numpy pass per iteration, 72 nnz per cell); the adjoint (CPR) colouring would be 6 tets × 8 corner-parity classes = 48 kernel passes per Jacobian. `Constraint._cached_jac_builder` memoises on the INSTANCE and `build_subproblem` makes a fresh constraint per window, so the builder is cached per patch SHAPE in the locality module (bounded LRU, as `core/marching/_mono_rows.axial_mono_rows` does). Phase 1 measures the Jacobian's share of the per-iteration cost; the colouring is revisited only if that share is large |
| `build_subproblem(constraint, phi, free_box, …)` | `free_box = (fy0,fy1,fx0,fx1)`, `H, W = phi.shape[1:]`, `free_mask (ph,pw)`, `np.stack([m, m])` | boxes become `2·ndim`-tuples `(z0,z1,y0,y1,x0,x1)` (per-axis `(lo, hi)` pairs, the 2D order today), masks `(pd,ph,pw)`, `np.stack([m] * phi.shape[0])`; the box arithmetic (patch slicing, the `touched` footprint, grow-on-failure, paste-back) goes through three n-D helpers — `_box_slices` / `_box_size` / `_pad_box` — that are byte-identical in 2D; `borders` stays the per-axis `(lo == 0, hi == n)` pairs; everything else (enforced rows, `cons`/`cons_jac`, objective triplet, `WindowSub`) is pack-agnostic already — `WindowSub.patch_box` and `free_mask` just carry one more axis |
| `find_windows(mask, margin, ring)` | 2D dilation/label/`find_objects` | the same dilate / label / `find_objects` (all n-D in `ndimage`); the `sy, sx` unpacking becomes a per-axis loop with the same border rule |
| `_orientation_rows` (edge monotonicity, DY_FIRST 2D) | 2D edge projections + anti-diagonal convexity | **Review correction.** `slsqp_windowed.constraints3d._injectivity_linear_constraint_3d(size, δ, freeze_mask=)` keeps a row only when BOTH endpoints are free; the 2D `_orientation_rows` keeps every edge with AT LEAST ONE free endpoint (`fm[i,j] or fm[i,j+1]`) — the free-to-frozen-ring edges are the rows that stop a free voxel rotating against its pinned neighbour, i.e. the prevention itself. So the 3D `'edges'` kind builds the full axial row matrix with `freeze_mask=None` (all three axes, DX_FIRST `[dx|dy|dz]` blocks, `b = 1 − δ`) and keeps the rows that touch a free column — the predicate `core/marching/_mono_rows.mono_block` already uses — dropping frozen-frozen rows (a violated constant row would sit in the elastic slack for the whole solve and distort the merit). No 3D convexity rows (the 2D `'full'` kind was a measured fidelity cost anyway); `orientation_rows='full'` on a 3D field raises. The DY_FIRST-only gate in `windowed_correct` becomes "family provides rows" (DY_FIRST 2D, or the registered 3D simplex family) |
| `isqp_solve` step rule | `'exact_ls'` needs rows bilinear along a line (`c(a)=c+a Jd+a²q`) | 6-tet rows are trilinear → cubic along the line. **Review correction.** The engine's and the strategies' default is `'exact_ls'`, and the entry *raises* on a 3D field, so `ISQPWindowedStrategy()` on a 3D constraint would raise out of the box. The entry degrades `'exact_ls'` to `'tr'` on a 3D field (debug log), the way it already drops `orientation_delta` on non-DY_FIRST packs and `'hybrid'` degrades to OSQP without clarabel. Consequence: the patience rung is `exact_ls`-only by construction, so it is OFF in 3D until phase 3 — not a knob to decide. Phase 3 (only if measured load-bearing): cubic exact line-min — fit `c(a)` from `cons` at `a = 0, ½, 1` plus `J d` (one extra evaluation), same breakpoint sweep on a cubic merit |
| giant tiler `_solve_giant_schwarz` / `_fit_tile` | 2D tiles stepping by `tile − (2 ring + 2)`, RAS cores | 3D tiles (target `giant_tile` per axis, fitted per axis), same sweep/RAS logic; the region-area cap becomes a voxel cap. **Phase 1:** the cap is ADVISORY on a 3D field — an over-cap region is solved whole, warned once per call, counted in `giant_regions` (the 17³ sub-volume is one such region and is exactly the per-QP-cost measurement point); phase 2 restores the cap with the 3D tiler |
| coarse-to-fine `_restrict`/`_prolongate` | `reshape(2, hc, f, wc, f).mean` and 2D `ndimage.zoom` | 3-axis block mean and 3D zoom (order 1); displacement rescale unchanged. The skip rule `min(shape) >= 4 · giant_tile` would fire only on ≥256-voxel-edge volumes (full-res B0039 qualifies, chunks do not) — phase 2 decides the 3D rule; skipped on 3D in phase 1 |
| mop `_mop_pass`, re-seed `_reseed_stage` (`_harmonic_fill`) | 2D clusters, 2D harmonic fill | n-D `ndimage` clusters; a 3D harmonic fill (7-point stencil, same solver). Skipped on 3D in phase 1 (report fields keep their did-not-run values); `reanchor` / `polish` requested on a 3D field raise until phase 2 |
| damage / `touched` / `min_field` reporting | 2D masks | n-D masks (the accounting is n-D already); `SliceReport` gains `folds_after_zero`, `best_diag_floor_after` (at threshold) and `best_diag_floor_after_zero` (−1 on 2D fields). `n_neg_best_diagonal` counts `<= threshold` where the engine's fold test is `< threshold`; the rows are driven to `threshold + margin_delta`, so no cell lands exactly on the threshold in practice |
| `RunCheckpoint` | units: z-slices or whole field | 3D engine: whole-field unit per round (`round:<k>`) and per giant tile (`tile:<k>:<i>`), `finish()` after the mop/re-seed |

Everything in `isqp.py` (OSQP/Clarabel/QPALM backends, hybrid policy,
trust region, a*-collapse bail, `feas_tol`/`ftol`) is pack- and
dimension-agnostic and is **not touched** — verified: the only 2D-specific
code there is `_exact_line_min` and its gate. **Review correction (QP size):**
the elastic QP carries one slack per enforced row, so a 32³ window is ~89k
free variables + ~197k slacks ≈ 286k QP variables per SQP iteration (the
first draft's "98k variables / 180k rows" counted the field, not the QP).

## Components and sequencing (one PR each, gated)

1. **Family plumbing** — `LOCALITY[SimplexConstraint3D]`; the n-D box
   helpers; n-D `find_windows` / `build_subproblem` / round loop /
   `_solve_window` / damage accounting; the 3D `'edges'` rows; the `'tr'`
   degrade; the giant / coarse / mop / re-seed stages skipped and `reanchor` /
   `polish` refused on 3D fields (see the table); the certificate fields;
   `WindowedWrapperStrategy.supports_3d = True` (the existing 3D-rejection
   test inverts; Jdet3D stays rejected because it is not in `LOCALITY`).
   Gate: (a) every existing 2D test AND a direct main-vs-branch byte-identity
   A/B of `windowed_correct` outputs and reports — seeded random fields across
   the four 2D families and both objectives, the three hard crops, a
   giant-tiler case and a coarse-to-fine case; (b) 3D unit tests (rows and
   Jacobian vs finite differences; the eight-corner influenced-cell rule;
   patch-vs-global row identity; paste-back no-damage; untouched voxels
   bit-identical; certificate counts on planted folds); (c) the three
   `testcases_3d` fields reach 0 fixed-6-tet folds at damage 0 on `'tr'`;
   (d) the 16³ sub-volume is the *measured* result with its floor beside it —
   a residual there is phase 2's target (no mop, re-seed or tiler yet), not a
   phase-1 failure (see the artefact table).
2. **Tiler, coarse-to-fine, mop, re-seed in 3D** — the round loop runs the
   whole ladder and the voxel cap binds again. Gate: a 3D crop pack cut from
   B0039 the way `make_hard_crops.py` cut the 2D one (a twist, a dense
   cluster, a sliver, each ≤ 48³), all 0 folds / damage 0; wall and L2 move
   recorded as the reference table.
3. **Inner cost** — measure per-QP cost vs window volume on the crop pack
   (phase 1's curve is the prior). Then, only as measured: the cubic exact
   line-min; `qp_max_iter` / hybrid thresholds re-tuned for 3D; `giant_tile`
   per-axis defaults; the Jacobian colouring if its share warrants it.
4. **Scale** — chunked full-volume driver (Part XX's memory reality: one
   field copy per process, workers get patches only), `checkpoint_dir` per
   round/tile, ds2 cohort volumes, then the full-res B0039. Certification =
   0 folds at fixed-diagonal 6-tet, the best-diagonal floor reported alongside,
   damage 0, L2 move.

Phase 1 is where the two unknowns that decide the plan get measured:

- **U1 — per-SQP-iteration cost vs window volume.** Single frozen-ring
  windows over the interior of 9³ / 17³ / 25³ / 33³ cubes (the three research
  sub-volumes plus a 33³ cut of the raw B0039 field whose fold fraction is
  closest to 10 %), `maxiter` capped, recording every QP solve's wall, ADMM
  iteration count and backend (a timing proxy around `isqp._make_qp` in the
  gate script — the per-iteration trace carries no QP timing), the Jacobian
  build and constraint evaluation separately, and the QP size (free + slack).
- **U2 — whether the ratio test alone converges on real 3D windows.**
  Per-window `trace['exit']` histograms, feasibility, SQP iterations and the
  fold / floor counts on the testcases and the 16³ under the four configs
  {L2, none} × {edge rows on, off}.

## Data flow / conventions

`(3, D, H, W)` `[dz, dy, dx]`; `SimplexConstraint3D.pack = DX_FIRST`
(`[dx, dy, dz]` flat). The engine already asserts pack lengths at the one
place it mixes families (`core/marching`); the 3D windowed path is single-pack
(DX_FIRST) end-to-end — the DY_FIRST-only gate on the rows is the one place a
pack check must change. Windows are boxes `(fz0,fz1,fy0,fy1,fx0,fx1)`;
patch = box grown by the ring, clipped to the volume; free voxels = box (∩
`free_extra`); enforced rows = the 6 tets of every cell with a free corner.
Row layout is `values()`'s: row `k · n_cells + cell` for tet `k` of the
C-ordered cube grid, the layout `tet_volumes_flat` and `build_tet_sparse_jac`
share.

## Error handling

`windowed_correct` keeps its entry gates: an unregistered family raises the
existing `IncompatibleConstraintError` path; `orientation_rows='full'`,
`reanchor != 'none'` and `polish` on a 3D field raise with the reason;
`'exact_ls'` on a 3D field degrades to `'tr'` (debug log) until phase 3.

**Review correction (memory).** Today the engine holds the caller's field,
its working copy, the initial min-field map (`j0`, alive for the whole run)
and a per-round min-field temporary — on the 1.85 GB full-res field that is
~5 GB per process, not "one copy". Phase 4's chunked driver is where "one
field copy plus per-window patches" becomes true; until then the claim is a
target, not a property.

## Testing

- Byte-identity gates: every existing 2D test and the invariant suite stay
  green unchanged (the 3D path is additive), plus the main-vs-branch A/B
  script (`benchmarks/windowed_2d_identity.py`: `--out DIR` on each tree,
  `--compare A B`).
- 3D unit tests (`tests/test_windowed_3d.py`): rows/Jacobian vs finite
  differences on random patches; `find_windows` / `build_subproblem` geometry
  (ring, borders, free sets, the eight-corner rule); the no-damage invariant
  (untouched voxels bit-identical; damage 0); the certificate (fixed vs
  best-diagonal counts) on planted folds; the `'tr'` degrade and the refused
  stages.
- Gate script (`benchmarks/windowed_3d_gate.py`): `--gate` (the artefacts ×
  four configs, 0-fold / damage-0 assertions on the testcases, the reference
  table) and `--cost` (U1).
- asv: a `WindowedEngine3D` bench pinned to wall, SQP iterations, L2 move and
  folds-after.

## Risks

- **Cost.** ~286k QP variables at 32³ (see above); minutes per QP is
  plausible, and the box is memory-bandwidth bound (~4 useful workers).
  Mitigations already in the design: coarse-to-fine, giant tiler with RAS,
  `qp_max_iter`, hybrid IP, and the futility mechanisms. U1 exists to measure
  before tuning.
- **Convergence without `exact_ls`.** In 2D the exact line search was
  −18 %/−28 % (wall/iterations) and removed a crop regression only with its
  bail; 3D starts on `'tr'`. If 3D windows plateau, the cubic line-min is the
  planned lever, not a new mechanism.
- **The certificate.** The fixed 6-tet split has artifacts (Part XXIII: of 33
  residuals, 19 were re-split artifacts, 14 true floor). Report both counts;
  never claim the fixed-diagonal count as the floor. The floor is a property
  of the field STATE (corners move), so an input floor bounds nothing.
- **Rotated-branch analogue.** Expect a 3D trap set; the edge rows are the
  prevention, the harmonic re-seed the repair — both port directly.

## Open questions (decide in phase 1 by measurement)

1. The Jacobian's share of the per-iteration cost (native builder vs the
   48-pass adjoint colouring).
2. `giant_tile` per axis for 3D (2D: 64, fitted) and the voxel cap that
   replaces `max_window_area` — from U1's curve.
3. Whether 3D windows need the backend rung at all (the patience rung is
   moot until phase 3).

## Artefacts to build the gates from

Measured 2026-09-05 (`six_tet_min_volume_3d` / `best_diagonal_min_volume`;
"fixed" = the shipped Kuhn diagonal, "floor" = negative under every main
diagonal; counts at `<= 0` / `< 0.01`):

| artefact | shape | dz ≡ 0 | fixed ≤ 0 / < 0.01 | floor ≤ 0 / < 0.01 | min tet volume |
|---|---|---|---|---|---|
| `data/dvfs/testcases_3d/slice090_5x10x10.npy` | (3, 5, 10, 10) | yes | 56 / 61 | 45 / 53 | −3.03 |
| `…/slice200_5x10x10.npy` | (3, 5, 10, 10) | yes | 29 / 30 | 18 / 18 | −2.57 |
| `…/slice350_5x10x10.npy` | (3, 5, 10, 10) | yes | 67 / 70 | 53 / 57 | −2.71 |
| `research/strict_feasibility_3d/runners/output/b0039_subvol_16_moderate.npy` | (3, 17, 17, 17) | yes | 614 / 721 | 590 / 702 | −3.71 |
| `…/b0039_subvol_8_easy.npy` | (3, 9, 9, 9) | yes | 15 / 47 | 15 / 47 | −0.009 |
| `…/b0039_subvol_24x24x24.npy` | (3, 25, 25, 25) | yes | 11659 / 12166 | 11646 / 12144 | −3.07 |

`correct_dvf_3d` left 1 residual at −7e-5 on the 16³ (REPORT Part XVIII).
Every artefact has dz ≡ 0 (slice stacks): the 3D engine is the first thing
that moves dz. The full-res raw field is
`data/dvfs/b0039/b0039_laplacian_deformation_field.npy`, `(3, 528, 320, 456)`
float32; `benchmarks/make_hard_crops.py` is the template for the phase-2 crop
pack; ds2 cohort volumes for phase 4.
