# Changelog

Tracks user-visible changes to `dvfopt`. Format inspired by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed — `correct_dvf` / `dvfopt correct` rejected the canonical `(3, 1, H, W)` layout

- `correct_dvf` with a string 2D constraint inferred its shape as `phi.shape[1:]`,
  which is `(1, H, W)` for the canonical `[dz, dy, dx]` single-slice layout every
  loader produces, so the constraint rejected its own input with
  `SolverConfigError: deformation spatial shape (H, W) does not match ... (configured
  for (1, H, W))`; `dvfopt correct` with the default `--pipeline solver` funnels into
  the same call and exited 2 on such files (`--pipeline slices` passes `shape=`
  explicitly and was fine, as were `(2, H, W)` files and 3D labels on volumes). No
  test called `correct_dvf` with a string 2D label on a 4-D input. The rule now lives
  once, as `dvfopt.constraints.infer_shape(name, phi)` — the trailing 3 dims for a 3D
  family, the trailing 2 for a 2D one, array-likes accepted — used by both
  `correct_dvf` and `constraint_fold_stats` (which had its own inline copy). 3D
  labels on a `(3, 1, H, W)` file still need `D >= 3` and exit 2 as before.
  `Solver.from_spec` alone still requires `shape=` for string labels.

### Changed — `dvf_origins`: one directory per mechanism, `m<k>_<tool>_<data>_<variant>` names, manifest

- `generate` writes `data/origins/<mechanism dir>/<case>.npy + .json` (`m1_interpolation`,
  `m2_dense_optimization`, `m3_learned`, `m4_diffeomorphic` — mechanism and directory are
  one table) and `generate` / `sweep` rebuild `data/origins/manifest.json` from the tree
  (case → file, mechanism, tool, source, shape, build time; atomic write, never merged,
  so concurrent torch-venv / main-venv runs and interrupts cannot leave it inconsistent).
  `sweep` reads the tree, reports any field at a path no case maps to, and — for a full
  sweep of the default root, or `--latest` — writes the stable
  `output/origins/results_latest.csv` for the paper build to point at (a scratch sweep
  cannot silently replace it; a locked file is reported, not a traceback).
- Every case is renamed to `m<k>_<tool>_<data>_<variant>` so a file says what made it
  and from what even when copied out of its directory (`<data>` vocabulary and layout in
  `dvf_origins/README.md`); the self-check enforces the shape. E.g. `m2_demons_brain` →
  `m2_demons_brainpair_weak`, `m3_voxelmorph_direct_cohort` → `m3_voxelmorph_cohort_direct`,
  `m4_ants_B0039_z264` → `m4_ants_cohort_B0039_z264`. Existing fields were moved in
  place (no retraining; the JSON sidecars carry `renamed_from`). The registry's
  `case` / `mechanism` now win over anything a builder puts in its meta.

### Changed — the windowed engine's formulation: edge-monotonicity rows + in-solve L2 (behaviour change)

- **Defaults:** `orientation_delta=0.01`, `orientation_rows='edges'` on `windowed_correct`
  and the windowed strategies — every window now carries the LINEAR edge-monotonicity
  rows (each deformed grid edge keeps a projection ≥ 0.01 on its own direction)
  next to the bilinear area rows. `build_subproblem`'s own defaults stay off. The
  robust recipe's objective becomes **`'l2'`** (the in-solve distance to the input).
- **Why — one self-contained formulation.** The residual every fallback plateaued on
  is the rotated orientation branch of the bilinear rows; the edge rows exclude that
  branch, so the feasible set is single-basin and the in-solve L2 objective no
  longer traps folds. Measured (same-process A/B on raw B0039 z16): plain 621 s /
  L2 268.0 vs edge rows + L2 659 s / L2 264.1 — same cost, slightly closer to the
  input; on the trapped slices: full-resolution z=2 **2039 s / L2 1979** vs
  10,168 s / 2732 (plain), z11 **2595 s / L2 783** vs the re-seed path's 14,372 s /
  1017. The anti-diagonal convexity rows (`orientation_rows='full'`) were the rows'
  fidelity cost (z11 1566 → 980 when dropped) and stay opt-in.
- The re-seed stage, the mop rule and the ladder are unchanged and act as nets; on
  the measured slices the re-seed did not fire.

### Changed — windowed engine: the mop's big windows get a single attempt (no retries, no grow)

- **Why.** Trace of the full-resolution B0039 exterior z=1 (15 657 s of inner-solver
  time): 665 of 694 inner calls fail; the 373 calls on windows with more than
  3000 free pixels take 15 173 s (97 %) at ~3.7 s per SQP iteration (0.04 s on
  ordinary windows), and the terminal mop alone takes **12 367 s (79 %)** — its
  whole-cluster windows (up to 4 × `max_window_area` free pixels) run the entire
  escalation ladder (10 calls per box) on the 50-cell rotated-branch residual that
  no rung can solve and that the re-seed then clears in 7 s. The eight volume-edge
  slices of that volume are 42 % of its 42.4 serial hours for this reason.
- **Change.** A mop window above `max_window_area` now gets ONE attempt (no no-TR
  retry, no backend retry, no patience rung, no grow — `_InnerOpts.ladder=False`);
  the re-seed stage after the mop handles what it leaves. Small mop windows keep
  the full ladder: the sliver-type residual (`z0_sliver`) needs it, is cheap, and
  stays byte-identical.
- **Measured and rejected:** running the re-seed BEFORE the mop (`reseed_before_mop`,
  kept as an opt-in knob, default off). It removes the same cost, but the harmonic
  fill is far too blunt for sliver residual: `z0_sliver` L2 137.8 vs 21.5.

### Added — `dvf_origins.learned`: the learned rows on REAL data (`data=cohort_data`)

- `learned.cohort_data` builds a real training set from the RegTools cohort
  outputs (external, `DVF_ORIGINS_REGTOOLS`): each brain's axis-aligned volume is
  resampled onto the template grid through its ANTs affine
  (`fwd_transforms/ants_affine_1.mat` — verified on B0039 z=264: slice correlation
  with the template 0.18 identity → 0.87 affine → 0.94 SyN result, so what the
  network has to learn is the nonlinear residual SyN solved), coronal planes
  `z = 60..468 step 12` of the six training brains are paired with the template's
  plane, and B0039 at z=264 — the plane the real m1 / m4 rows use — is held out as
  the test pair. Planes are block-mean downsampled ×3 and centre-cropped to 96×128
  (the VoxelMorph UNet has five levels → multiples of 32; ~85 % of the field of view,
  so these rows sit on a different grid than the native 320×456 m1/m4 rows of the
  same plane — compare fold fractions); 210 training pairs, cached under
  `data/origins/cache/` keyed by a hash of every input and verified on load. Both
  generators take `data=None` (the notebooks' synthetic images, unrelated random
  pairs) or a callable such as `learned.cohort_data` (paired real slices); four new
  `CASES` rows `m3_{voxelmorph,transmorph}_{direct,diffeo}_cohort`.
- The convention check now warps with the NETWORK's own off-image padding
  (VoxelMorph zeros, the Swin sampler border), so the RMSE is exact over the whole
  image. On brain slices whose crop edges are not black the old nearest-padding
  comparison read a 5e-3 "mismatch" for VoxelMorph that was padding semantics, not
  channel order; it is 2e-7 vs 4.5e-2 swapped now. A non-finite field raises.
- **Measured (seed 0, CPU, B0039 z=264 on the 96×128 grid, 12 065 simplex cells):**
  VoxelMorph direct **218 folded cells (1.8 %) in 21 clusters** (median 5, max 43),
  min −0.24, loss 0.020, 2 % off-image, 547 s; TransMorph-style direct **236 cells
  (2.0 %) in 26 clusters** (median 6, max 37), min −0.27, loss 0.025, 619 s. So on
  real paired slices the direct regressors fold in MANY SMALL clusters — the
  scattered signature the proxy assumes, unlike the synthetic-toy rows' few giant
  clusters — at a fold fraction 2–3× the native-resolution Laplacian row of the same
  plane (1112 of 145 145 cells, 0.77 %). Both diffeo variants train on real data
  (the synthetic collapse was the ill-posed unrelated-pairs task, not the
  architecture): VoxelMorph diffeo 1 cell + 13 bilinear-only (min −0.0016, loss
  0.019); TransMorph-style diffeo 6 cells + 10 bilinear-only in 2 clusters, one deep
  sliver (min −2.5), 653 s. `warp_rmse` 2e-7 vs ≥ 0.1 swapped on every row.

### Added — `dvf_origins.learned`: mechanism 3 with real networks

- `learned.voxelmorph` / `learned.transmorph` train the `benchmarks/registration/`
  notebooks' networks (VoxelMorph `VxmPairwise`; the TransMorph-style Swin-Tiny +
  ConvNet `SwinRegNet`) on their synthetic ellipse images — same 200 × 50 steps,
  MSE + 0.05·smoothness — seeded, and return the inference field on a held-out
  pair. `integration_steps=0` is a direct displacement regressor (the paper's
  mechanism 3), `7` a learned diffeomorphism. Four `CASES` rows
  (`m3_{voxelmorph,transmorph}_{direct,diffeo}`); the phantom "drop a saved
  notebook output here" rows are gone (one generic `m3_external_saved` remains).
- They need torch, which the main venv deliberately does not carry: a separate
  CPU venv (`uv venv .venv-torch` + `--torch-backend=cpu`, recipe in
  `dvf_origins/README.md` and `learned.py`) builds them in ~8 min (VoxelMorph)
  / ~29 min (TransMorph) per row; without torch `generate` skips them.
- Each row records `warp_rmse` — pull-back-warping the source by the returned
  field must reproduce the network's own warped output — next to the same number
  with the channels swapped, so the `[dy, dx]` / `moving(x + u(x))` convention is
  measured rather than assumed. It caught a ±0.5 px identity stretch in the
  TransMorph notebook's sampler (`linspace(-1, 1, n)` grid with
  `align_corners=False`): 2.4e-2 RMSE against its own warp, 1e-7 once the harness
  copy samples pixel centers exactly (`align_corners=True`, `2d/(n-1)`; the
  notebook cell got the same fix).
- One deliberate deviation from the TransMorph notebook (`feature_stage=0`): its
  decoder reads the Swin encoder's 2×2 bottleneck, which can only emit near-global
  fields and, trained the notebook's way, settles on a constant −58 px translation
  that shifts the whole source off-image — border padding returns black, the MSE
  equals mean(target²) ≈ 0.08, and the "field" is a fold-free translation. The
  harness reads the stage-0 (16×16) feature map instead: at 1000 steps loss 0.074
  and 3 % off-image with a genuinely local field, vs 0.103 / 60 % for the
  bottleneck, 5.7× faster. `meta['off_image_frac']` is the collapse detector
  (`feature_stage=None` restores the notebook's design) and rides into the sweep CSV.
- **Measured (seed 0, CPU, 64², simplex cells of 3969):** VoxelMorph direct **850
  folded cells in 5 clusters** (median 75, max 591), min −7.5, loss 0.040, 147 s;
  VoxelMorph diffeo (7 squarings) **415 cells / 22 clusters, 179 bilinear-only**, min
  −1.6, 181 s — a learned diffeomorphism still folding at the discrete level, i.e.
  mechanism 4's signature from a mechanism-3 tool; TransMorph-style direct **680
  cells / 5 clusters**, min −11.9, loss 0.052, 3 % off-image, 275 s. TransMorph-style
  diffeo does not train on this setup: seeds 0 and 1 collapse off-image
  (`off_image_frac` 1.0, loss = mean(target²) 0.079), seed 2 stays at the identity
  (loss 0.114, zero displacement) — the row is kept, flagged by the CSV column, and is
  not a usable field. All three real rows rebuild byte-identically on this machine.

### Fixed — `load_dvf_sitk` honours image geometry (direction matrix + spacing)

- `dvfopt.io.fields.load_dvf_sitk` (behind `load_dvf`, the CLI's and the GUI's
  NIfTI / MetaImage / NRRD loader) read the stored PHYSICAL displacement vectors as
  if they were index-space voxels — no spacing division, no direction matrix. That
  is exact for files this module writes (identity direction, unit spacing) and wrong
  for real registration warps: the cohort ANTs SyN warps carry a signed-permutation
  direction `D = [[0,0,-1],[1,0,0],[0,-1,0]]` at 0.025 mm spacing, and loading them
  the old way reads **4667 voxels with 3D Jdet ≤ 0** on a warp that is diffeomorphic
  by construction. New `dvf_from_sitk_image(img)` does the conversion
  (`D⁻¹·v/spacing`, identity fast-path so existing files load byte-identically);
  `load_dvf_sitk` is now `ReadImage` + that. Measured on the same warp: **0** folds.
  Hand-derived 2D and 3D tests pin the mapping.

### Added — `dvf_origins/`: sample DVFs per fold-origin mechanism (standalone harness)

- New top-level folder, deliberately NOT part of the `dvfopt` package (imports it
  only for the Laplacian solve and the fold metrics; not installed, run from the
  repo root). Its self-check `pytest dvf_origins` is in pytest's `testpaths` and the
  CI / nox test and ruff commands so it cannot rot (data-gated tests skip on CI).
  `python -m dvf_origins
  {list, generate, sweep}` builds one field per (mechanism, tool, severity) into
  `data/origins/` and writes the fold-morphology table (`output/origins/<ts>/
  results.csv`): central-difference Jdet, simplex and bilinear certificates,
  bilinear-only cells, fold fraction, cluster count / median / max area.
- Mechanisms: (1) Laplacian of corrupted correspondences — outliers, adjacent
  many-to-one collapses, jitter — plus a real cohort slice; (2) skimage TV-L1 /
  ILK on a textured pair plus real SimpleITK demons / B-spline FFD / TV-L1 runs on
  the `data/mouse_brain` slice pair; (3) a labeled learned-field PROXY (smooth
  warp + grid-scale noise) plus a generic saved-field loader (the real learned
  rows arrived in the `dvf_origins.learned` entry above); (4) SVF scaling-and-squaring with decimation / coarse steps /
  sub-pixel-only folds plus the real ANTs slice — converted to index space by the
  library's `dvf_from_sitk_image` (see *Fixed* above) and re-laid-out onto the
  Laplacian field's `(i, j, k)` grid so `z` names the same plane in both real rows.
  Measured signatures match the mechanism: the
  proxy gives ~1100 scattered clusters of median 4 cells; the sub-pixel SVF case
  has central-difference Jdet ≥ 0.035 everywhere yet 421 simplex-folded cells,
  297 of them bilinear-only.

### Added — windowed engine: linear orientation rows (`orientation_delta`, off by default)

- **`orientation_delta=None`** on `windowed_correct` and the windowed strategies. A
  float (e.g. `0.01`) appends the LINEAR orientation rows to every window
  sub-problem: each deformed grid edge keeps a positive projection of at least
  `orientation_delta` on its own direction, plus the anti-diagonal convexity rows
  (the injectivity conditions of `dvfopt.jacobian.monotonicity`), for every edge /
  cell with a free pixel. A cell on the rotated orientation branch (the residual
  the re-seed stage repairs) violates them, so with the rows the QP never heads
  there — *prevention*, where the re-seed is *repair* — and they are linear, hence
  exact in the QP (no thin-cell linearisation error). 2D simplex-family
  (`DY_FIRST`) only; other packs raise.
- **Measured from raw with the re-seed off:** every plateau slice of the cohort
  clears — B0304 ext z128 8956 → 0 (200 windows, 3643 s), B0032 ext z1 4556 → 0
  (1 round / 31 windows / 1066 s where the plain engine left 70 after 8902 s / 374
  windows), B0039 lap_all z11 4633 → 0 (1 round / 37 windows), B0039 ext z1
  3957 → 0 (2 rounds / 47 windows); damage 0 throughout.
- **Why it is not the default:** fidelity. The rows are stricter than "no fold"
  (they exclude legitimately fold-free cells rotated by more than 90°), and the
  cost is case-dependent: +0.7 % / +0.8 % / +1.7 % L2 on z128 / B0032 z1 / B0039
  z1, but **+69 %** on z11 (1566 vs 929) where the re-seed path reaches 0 folds at
  1017 (+9.5 %). On the small crops the over-constraint shows as +50–70 % L2. So
  the re-seed stage stays the default repair and the rows are an opt-in speed
  lever (one round instead of many on the hardest slices).

### Added — windowed engine: terminal harmonic re-seed stage (default on) — 0 folds on every cohort residual

- **`reseed_rounds=3` / `reseed_radius=2`** on `windowed_correct` and the windowed
  strategies. After the round loop and the mop, while folds remain (bounded
  rounds, deadline-aware), each residual fold cluster's neighbourhood (its cells'
  corner pixels dilated by `reseed_radius`) is replaced by the discrete-harmonic
  interpolation of its ring and the engine polishes the re-seeded field
  (recursively, stage off). `report.reseed_rounds_run` / `reseed_px` /
  `reseed_folds_before` / `reseed_folds_after` record it.
- **Why (the deep dive).** Every residual cluster of the 7-brain cohort sweep is a
  set of cells whose corner images the solver drove onto the ROTATED orientation
  branch: both edge factors of the bilinear area negative, product positive —
  locally fold-free, but not joinable to the surrounding un-rotated field, and the
  seam between the branches is a merit *maximum*. Measured on the plateaued
  B0304 z181 cluster: every axis-aligned move (splitting the glued pins, shifting
  the pin column into its neighbours' interval) raises the merit *linearly*, and
  the per-iteration traces show every rung, step rule and trust radius failing
  identically (`a*` → 0, `step-tol` / `a-collapse` / `tr-collapse`). A local
  method cannot cross the seam; the harmonic interpolation of the ring is on the
  right branch by construction, so re-seeding + polishing crosses it.
- **Measured on the five plateaued cohort slices** (B0304 z181 / z128, B0039 z11 /
  ext z1, B0032 ext z1; 7–79 residual cells each, every rung exhausted):
  **0 simplex and 0 bilinear folds on all five, damage 0, 10–40 s** for the
  re-seed + polish. From-raw reruns and the crop-pack / raw-z16 byte-identity
  (the stage never fires on a field the mop cleared) are reported on the PR.
- The re-seeded pixels are fold neighbourhoods and the polish's window footprints
  join `touched`, so the no-damage invariant is unchanged.

### Added — windowed engine: patience rung (bail-free exact-LS continuation) in the window ladder

- **`patience_retry=True`** on `windowed_correct` / `WindowedWrapperStrategy` /
  `ISQPWindowedStrategy` — the last rung of the per-window escalation ladder
  before grow-on-failure. A window still *genuinely* folded after the solve, the
  no-TR retry and the backend retry continues its `exact_ls` iteration from the
  best iterate with the a\*-collapse bail off (`exact_ls_fallback_steps=0`).
- **Why.** The 7-brain cohort sweep's residual clusters all sit on prescribed
  Laplacian correspondences whose displacement disagrees with their neighbours by
  tens of pixels (8/8 clusters on B0039 z11 within 4 px of a pin vs a 5% base
  rate; B0304 z181's cluster on five pins 174 px off the slice median). There the
  bail (default 3) stops the window after a few tiny but *productive* steps and
  hands it to rungs that cannot continue it (the no-TR line search stalls, the
  backend retry bails again, grow repeats the pattern). Measured on a crop of the
  B0304 z181 residual: full ladder → folded at −0.044 after 101 s; the same
  window with the bail off → +0.011 in **1 s** (L2 move 46 — it walked the pin
  out). `report.patience_fallbacks` / `WindowRec.patience_fallback` count the rung.

### Fixed — windowed engine: no-damage accounting and the deadline on budget-cut runs

- **Coarse-to-fine warm start + `time_budget_s` could report `damage > 0`.** The
  warm start's prolongated correction is masked to the window boxes
  `find_windows` opens on the input fold mask, but those boxes were not marked
  *touched* — only windows the fine loop actually solved were. On a run the
  budget cut before the fine loop reached a warm-started box, a fold the warm
  start created there was booked as damage to untouched area. It is a residual
  inside a fold neighbourhood (the warm start IS a move over the engine's own
  boxes), and is now accounted as `residual_in_window`. Found by the 7-brain
  cohort sweep (B0304 z181 / z128, both budget-cut: "damage" 3440 / 38);
  reproduced on raw B0039 z16 under a 40 s budget (damage 3 with the warm
  start, 0 without). Completed runs are unaffected — every box with a fold left
  is opened and solved, so the invariant already held there — and the output
  field is byte-identical either way; only the damage / residual split of a
  budget-cut report changes.
- **The giant-region tiler now checks the deadline between tiles.** A giant
  region is many window solves, not one; the 40 s budget above ran 189 s before
  the check existed. The tiler returns early (`-1` if no sweep completed) and
  the engine finishes on the best-so-far field as documented.

### Changed — `auto_strategy` routes the measured 2D zero-folds recipe (behaviour change)

- **`'bilinear'` -> `'isqp_windowed'` at every fold tier**, for every objective
  the engine accepts (l1/l2/none). It previously tiered like Jdet, i.e. dense
  fields went to `barrier`.
- **`'simplex_standard'` + `objective='none'` -> `'isqp_windowed'`** at every
  fold tier (previously `slsqp` / `barrier` / the wallbreakers by density).
- **`'simplex*'` + `'l1'` still routes to `'slp'`** — the L1-optimal champion.
  An anchor is a different fidelity request and is never silently swapped out;
  instead a one-line hint naming the recipe is logged, once per process, on the
  `dvfopt` logger.
- Both new routes need `osqp`; without it they fall through to the existing
  fold-density heuristic (verified to still compose). `'simplex'`
  (full coverage) has no windowed-engine locality entry, so only
  `'simplex_standard'` takes the `none` route. **3D routing is untouched.**
- Why: `constraint='bilinear'` + `strategy='isqp_windowed'` + `objective='none'`
  reaches 0 simplex folds from the RAW field on every B0039 slice tested (z16:
  3890 folds -> 0 in ~200 s, damage 0) where the 2-triangle-row methods stall on
  twisted cells — the bilinear rows are what give non-degenerate constraint
  gradients at bow-tie cells.
- **New: [`docs/recipe-2d-zero-folds.md`](docs/recipe-2d-zero-folds.md)** — the
  one-call recipe, what every engine default does and what bought it, the fast
  crop pack, the measured dead ends (float32 OSQP, GPU batched ADMM, Newton-SQP,
  dual warm starts, row pruning, OSQP settings), and the one known cost
  (`z0_sliver` under `exact_ls`). Linked from README and CLAUDE.md; the routing
  table is in CLAUDE.md, ARCHITECTURE.md and `dvfopt correct --strategy` help.
- **`benchmarks/make_hard_crops.py`** no longer monkeypatches `isqp_solve` to
  force `trust_region=False`, and no longer runs a retry loop: the per-window
  no-TR fallback and the backend fallback are engine defaults now, so validation
  calls `windowed_correct` once per gauge. The discriminator/recipe semantics are
  unchanged and now uniform across all three crops (the discriminator leaves
  bilinear folds — unclearable on `z16_twist`/`z0_cluster`, unseen on the
  simplex-clean `z0_sliver`; the recipe clears both gauges to zero).
### Added — windowed engine: optional post-feasibility re-anchor stage

- **`reanchor='none'` (DEFAULT — opt-in, no behaviour change)** on
  `windowed_correct` and `WindowedWrapperStrategy` / `ISQPWindowedStrategy`, with
  `'l2'` / `'l1'` selecting a new stage that runs after the mop. Knobs:
  `reanchor_maxiter=60`, `reanchor_sweeps=3`, `reanchor_tile=48`.
- **Why.** The robust recipe solves with `objective='none'` — pure feasibility,
  which is what keeps the windowed isqp out of the objective-basin traps a
  distance anchor pins it in (the measured trap). The price is that the
  correction is close to the input only *by construction*: nothing in the solve
  is minimising the departure. Once the field is fold-free there is no fold left
  to trap the inner, so the fidelity can be bought back separately.
- **What it does.** Tile the MOVED region (`reanchor_tile` px, overlapping by 8),
  build each tile with the engine's own `build_subproblem` on the *current*
  field, swap only the objective triplet for one anchored at the INPUT patch
  (`'l2'` = `||x - ref||^2`, `'l1'` = the eps-smoothed L1 — both from the
  engine's existing `_objective_fns`), solve it with the configured inner for
  `reanchor_maxiter` iterations, and **accept the tile only if every enforced row
  is still at or above `threshold`** and the tile actually reduced the distance —
  otherwise revert it. Up to `reanchor_sweeps` sweeps, stopping once a sweep buys
  < 1% of the L2 move.
- **The no-damage invariant is unaffected.** Each tile's free set is intersected
  with the moved mask (new optional `build_subproblem(..., free_extra=)`), so the
  stage only ever moves pixels the main solve already moved — the moved set can
  shrink, never grow — and those pixels, plus the rows they influence, are inside
  `touched` by construction. Damage accounting reads exactly the same. A global
  re-check after the stage reverts the whole thing and logs a warning if a fold
  appeared anyway (unreachable given the per-tile verification, but guarded).
- **Gated on feasibility.** The stage is skipped when the main solve left a fold
  in the engine's own constraint (`z0_cluster` below is that case: 1 residual
  bilinear row, so no re-anchor runs) — fidelity is never traded for the
  certificate.
- **`report.reanchor_sweeps_run` / `reanchor_tiles` / `reanchor_accepted` /
  `reanchor_l2_before` / `reanchor_l2_after`** record the stage; `0` sweeps means
  it did not run.

Measured on this branch (bilinear rows, objective `none`, threshold 0.01,
maxiter 600, engine defaults, OMP/BLAS/RAYON pinned to 1). **Every row is 0
simplex folds and damage 0** — the axes that move are the move norms and cost.
Wall times were taken on a loaded box (another job running) and are indicative
only; the fidelity columns are the point.

| case | reanchor | wall | L2 move | L1 move | tiles (accepted) | sweeps |
|---|---|---|---|---|---|---|
| z16_twist (crop) | none | 143 s | 103.4 | 3559 | — | 0 |
| z16_twist (crop) | l2 | 306 s | 102.4 | 3476 | 4 (3) | 1 |
| z16_twist (crop) | **l1** | 677 s | **69.9** | **1565** | 12 (11) | 3 |
| z0_cluster (crop) | none | 118 s | 535.1 | 17550 | — | 0 |
| z0_cluster (crop) | l2 | 109 s | 535.1 | 17550 | — | 0 (skipped) |
| z0_cluster (crop) | l1 | 110 s | 535.1 | 17550 | — | 0 (skipped) |
| z0_sliver (crop) | none | 865 s | 39.7 | 1227 | — | 0 |
| z0_sliver (crop) | l2 | 927 s | 38.6 | 1152 | 8 (5) | 2 |
| z0_sliver (crop) | **l1** | 800 s | **31.8** | **807** | 8 (5) | 2 |
| B0039 z16 (raw slice) | none | 893 s | 268.4 | 24812 | — | 0 |
| B0039 z16 (raw slice) | l2 | 2494 s | 255.7 | 22347 | 38 (16) | 2 |
| B0039 z16 (raw slice) | **l1** | 2804 s | **208.1** | **12876** | 57 (50) | 3 |

`z0_cluster` is the feasibility gate firing: that solve leaves 1 residual
bilinear row, so the stage does not run and the three rows are identical.

- **Headline:** on the raw B0039 z16 slice `reanchor='l1'` takes the L2 move
  268.4 -> 208.1 (**-22%**) and the L1 move 24812 -> 12876 (**-48%**) at 0 simplex
  folds and damage 0, accepting 50 of 57 tiles over 3 sweeps.
- **`'l1'` is the stronger re-anchor, even measured in L2.** The smoothed-L1
  Gauss-Newton diagonal floors at 0.1 against a unit gradient, so each SQP
  iteration proposes a steady sign-directed step the trust region caps at
  `tr_delta`; the L2 diagonal (2.0) against a `2 d` gradient proposes the whole
  jump to the reference, which the constraints clip hard. On `z16_twist` the L2
  leg's first sweep bought only 0.95% and the < 1% rule stopped it, while `'l1'`
  ran the full 3 sweeps.
- **It is not free.** The stage roughly doubles-to-triples the wall time of a
  slice, which is why it is opt-in and off by default: the zero-fold certificate
  is one concern, fidelity another.


### Added — windowed engine: exact merit line search + its a\*-collapse bail (now the default step rule)

- **`step_rule='exact_ls'` (DEFAULT — behaviour change)** on `windowed_correct`,
  `WindowedWrapperStrategy` / `ISQPWindowedStrategy` and
  `dvfopt.core.primitives.isqp.isqp_solve` (whose own default stays `'tr'`, so
  direct callers of the primitive are unchanged). Instead of accepting or
  rejecting the whole QP step by the trust-region ratio test, the inner now takes
  the **exact minimiser of the merit along that step**.
- **Why it is exact, and why it is free.** Every 2D row family the engine serves
  (2tri, bilinear, jdet, finite) is a BILINEAR form in `(dy, dx)`, so along the
  line `x + a d` a row is exactly quadratic:
  `c(a) = c + a (J d) + a^2 q` with `q = cons(x + d) - c - J d` — and that
  `cons(x + d)` is the evaluation the ratio test **already makes**, so the model
  costs no extra constraint evaluation and needs no per-family Hessian table. The
  merit `m(a) = f(a) + rho . max(0, -c(a))` is then piecewise quadratic with
  breakpoints exactly at the rows' roots, and its global minimiser on `[0, 1]`
  (the trust region already bounds the QP) is a vectorised O(m log m) breakpoint
  sweep. The trust region is still built, still bounds the QP and is still
  adapted — now from the achieved `a*`.
- **Guarded.** Only the objective along the line is *fitted* (from `obj` at
  `a = 0, 1/2, 1`: exact for `NoneObjective` / `L2Objective`, an approximation
  for the eps-smoothed L1), so the TRUE merit at `a*` is evaluated before the
  step is taken; if it did not decrease, that iteration falls back to the `'tr'`
  acceptance. `'exact_ls'` therefore cannot regress a window.
- **The futility bail is kept.** An exact minimiser always finds SOME decrease,
  so it never fires the ratio test's fast bail-out — and a window that cannot be
  solved at its current size then grinds instead of handing off to the engine's
  escalation ladder. The ratio test's own threshold (achieved <= 1e-3 x
  predicted, the existing constant — no new knob) is retained purely as the
  `tr-collapse` termination signal.
- **`'tr'` is the stock path byte for byte** (a test sabotages the line minimiser
  and pins an unchanged `'tr'` run). New trace fields under `'exact_ls'`:
  `alpha` (the accepted `a*`) and `rule` per iteration.
- **2D only** — a 6-tet volume row is trilinear, hence *cubic* along a line, so
  neither the model nor the identity transfers. `windowed_correct` rejects
  `step_rule='exact_ls'` on a non-2D field at its entry.
- **`exact_ls_fallback_steps=3` — the a\*-collapse bail** (engine / strategy
  default; `isqp_solve`'s own default is `0` = off, so direct callers of the
  primitive are unchanged). The futility test above is *relative* — achieved
  <= 1e-3 x the QP's PREDICTED decrease — so a window whose `a*` has collapsed
  keeps clearing it by a hair while going nowhere, and on the engine's
  no-trust-region rung there is no futility test at all, so `'exact_ls'` burns
  the whole `fallback_maxiter` there. After this many CONSECUTIVE accepted exact
  steps with `a* < 0.25` (the same threshold the trust region shrinks at) the
  window now STOPS — `trace['exit'] == 'a-collapse'`, the step still taken — and
  hands itself to the escalation ladder, which is precisely what makes
  `'tr-collapse'` cheap. **Stopping is the whole mechanism**: handing the
  remaining iterations to the `'tr'` *acceptance* instead was measured and is
  WORSE (z0_sliver 2350 SQP iterations vs 1684 with no bail at all), because
  mid-run the ratio test lands in the regime where it ACCEPTS tiny steps rather
  than rejecting them, so it grinds too.
- **Why 3.** The discriminator is the consecutive-collapse run length on real
  windows: over the crops' first-round windows the longest run of `a* < 0.25` is
  **2** on `z16_twist` — the window `'exact_ls'` turns from a 108-iteration
  failure into a 46-iteration solve, so 3 never fires there — against **4** on
  `z0_sliver` and **6** on `z0_cluster`. 5 was measured too and is too loose (it
  never fires on `z0_sliver`).
- **REFUTED, not shipped**: the maximal fold-free step cap (the nonlinear
  analogue of `monotone=True`). On real windows `a_max` is ~1e-3-1e-1, which
  strangles the elastic mechanism — measured end violations of 40-84 against the
  baseline's 0.027. Do not add one.
- **RE-MEASURED and still refuted**: scoping `'exact_ls'` out of the engine's
  no-trust-region rung (the prototype's `ls_exact_tr`). On the shipped
  implementation that costs `z0_sliver` 1918 SQP iterations against 1684 — the
  rung is not the problem, the missing stop signal was.

Measured on this branch (bilinear rows, objective `none`, threshold 0.01,
maxiter 600, engine defaults, OMP/BLAS/RAYON pinned to 1). **Every row has 0
simplex folds and damage 0** — the only axes that move are cost and L2 move:

| case | rule | wall | SQP iterations | L2 move |
|---|---|---|---|---|
| z16_twist (crop) | **exact_ls** | **25.2 s** | **47** | **103.4** |
| z16_twist (crop) | tr | 39.6 s | 128 | 125.7 |
| z0_cluster (crop) | **exact_ls** | **56.9 s** | **328** | **535.1** |
| z0_cluster (crop) | tr | 57.6 s | 387 | 542.7 |
| z0_sliver (crop) | exact_ls | 350.5 s | 1684 | 39.7 |
| z0_sliver (crop) | **tr** | **76.7 s** | **540** | **25.3** |
| **raw B0039 z16** | **exact_ls** | **200.4 s** | **552** (+11 coarse) | **268.4** |
| **raw B0039 z16** | tr | 244.1 s | 762 (+18 coarse) | 280.3 |

The `tr` rows reproduce the recorded reference exactly (raw z16: 762 fine
iterations, L2 280.3), so these are directly comparable to the existing engine
numbers.

`z0_sliver` was the one regression in that table, and the a\*-collapse bail
removes it. The mechanism the bail attacks is exactly the one that table
exposes: that crop starts with 0 simplex folds (min +0.0110 against a 0.01
threshold) and only ~1e-4-scale bilinear violations, its one hard window starts
at violation 0.0114 and NEITHER rule can clear it — `'tr'` discovers that in 11
iterations and hands the window straight to the escalation ladder, while an exact
minimiser always finds *some* decrease and grinds to `step-tol` first.

**With the bail at its default** (same settings). An unrelated job shared the
machine throughout this measurement, so the three variants were re-run
back-to-back in ONE process for a like-for-like wall comparison: within a row the
walls are comparable, but they run ~1.4-2.4x the uncontended figures in the table
above. SQP iteration counts are deterministic and reproduce that table exactly.
**Every entry: 0 simplex folds, damage 0.**

| case | `'tr'` | `exact_ls` | `exact_ls` + bail (default) |
|---|---|---|---|
| z16_twist (crop) | 72.1 s / 128 / L2 125.7 | 70.7 s / **47** / 103.4 | 75.1 s / **47** / 103.4 |
| z0_cluster (crop) | 94.7 s / 387 / 542.7 | 88.0 s / 328 / 535.1 | 88.7 s / **287** / 535.1 |
| z0_sliver (crop) | 181.6 s / 540 / 25.3 | 768.2 s / 1684 / 39.7 | **170.0 s / 212 / 19.4** |
| **raw B0039 z16** | 552.0 s / 780 / 280.3 | 436.6 s / 563 / 268.4 | **344.5 s / 396 / 268.0** |

(wall / SQP iterations including the coarse warm start / L2 move.)

`z0_sliver` goes from 3.1x `'tr'`'s iterations to **0.39x** — the bail does not
merely erase the regression, it beats the rule the regression was measured
against, on the case built to expose it, at a *smaller* L2 move than either
(19.4 vs 25.3 / 39.7). `z16_twist` — the case carrying `'exact_ls'`'s biggest win
— is **bit-identical**: 47 iterations, L2 103.4, the bail never fires there.
`z0_cluster` improves (-12% iterations, identical L2), and the real slice
improves again on top of `'exact_ls'`'s own win: **-30% iterations / -21% wall**,
i.e. -49% iterations against `'tr'`, with the move unchanged (268.0 vs 268.4).

Extended to five real B0039 slices (SQP iterations including the coarse warm
start; `exact_ls_fallback_steps=0` reproduces the `'exact_ls'` column above
exactly, which is how the baselines below were cross-checked). **Every row: 0
simplex folds, damage 0.**

| slice | `exact_ls` iters | + bail iters | d iters | L2 `exact_ls` -> + bail |
|---|---|---|---|---|
| z16 | 563 | **396** | -30% | 268.4 -> **268.0** |
| z112 | 139 | 139 | 0% | 27.1 -> 27.1 |
| z256 | 695 | **654** | -6% | 75.1 -> **74.5** |
| z304 | 1095 | **793** | -28% | 91.7 -> **89.9** |
| z496 | 1023 | **887** | -13% | 79.8 -> **77.8** |
| **total** | **3515** | **2869** | **-18%** | |

No slice regresses on either axis, and the L2 move is smaller or identical on
every one — the bail only ever removes iterations a window was spending without
getting anywhere. z112 is the shape to expect where `'exact_ls'` is already
healthy: bit-identical, because no window there collapses three times running.

**The real-data sweep is what settles the default.** Nine real B0039 slices
(the reference z16 plus every 48th from z=64, fold counts 835-3890), same engine
defaults, same thread pinning. SQP iterations include the coarse warm-start
solve. **Every row: 0 simplex folds, damage 0, for both rules.**

| slice | folds in | tr wall / iters | exact_ls wall / iters | d wall | d iters | L2 tr -> exact_ls |
|---|---|---|---|---|---|---|
| z16 | 3890 | 244.1 s / 780 | **200.4 s / 563** | -18% | -28% | 280.3 -> **268.4** |
| z64 | 1392 | 166.8 s / 726 | **146.9 s / 587** | -12% | -19% | 72.0 -> **67.7** |
| z112 | 835 | 63.0 s / 181 | **52.7 s / 139** | -16% | -23% | 27.6 -> **27.1** |
| z160 | 1193 | 82.7 s / 374 | **55.6 s / 248** | -33% | -34% | 42.8 -> **41.2** |
| z208 | 2560 | 289.5 s / 858 | **222.8 s / 706** | -23% | -18% | 77.1 -> **73.3** |
| z256 | 2413 | 339.2 s / 1165 | **254.8 s / 695** | -25% | -40% | 76.9 -> **75.1** |
| z304 | 3167 | 500.4 s / 1488 | **393.6 s / 1095** | -21% | -26% | 95.3 -> **91.7** |
| z400 | 2238 | 166.9 s / 591 | **133.3 s / 452** | -20% | -24% | 79.1 -> **76.2** |
| z496 | 2052 | 421.1 s / 1378 | **370.5 s / 1023** | -12% | -26% | 85.7 -> **79.8** |
| **total** | | **2273.7 s / 7541** | **1830.6 s / 5508** | **-19%** | **-27%** | |

**9 of 9 wall wins, 9 of 9 iteration wins, and a SMALLER L2 move on every single
slice** — the speed is not bought with fidelity. The mechanism is visible in the
iteration counts: the ratio test throws away 92 QP directions on raw z16 and pays
a whole extra QP solve to re-derive each one, while the exact minimiser never
rejects — every solved QP produces a step.


### Added — windowed engine: coarse-to-fine warm start (on by default)

- **`coarse_to_fine=True` / `coarse_factor=4`** (default factor raised 2 -> 4: 182 s / L2 280 vs 189 s / L2 321 on raw z16) on `windowed_correct` and
  `WindowedWrapperStrategy` / `ISQPWindowedStrategy`. Before the round loop the
  engine now solves the SAME problem on a `coarse_factor`x box-averaged field
  (displacements rescaled into coarse pixel units), bilinearly prolongates the
  resulting CORRECTION back (rescaled up), and starts the fine solve from
  `phi + delta` instead of cold. The fine windows then converge in far fewer
  SQP iterations. The coarse call always passes `coarse_to_fine=False` — never
  recursive.
- **No-damage is preserved by construction.** The prolongated delta is masked to
  the free boxes `find_windows` opens on the fine fold mask, so the warm start
  can only move pixels the engine was going to free anyway; healthy area outside
  every fold neighbourhood stays byte-identical, and the final damage accounting
  still runs against the ORIGINAL input (not the warmed field).
- **Skipped** — leaving the path byte-identical to `coarse_to_fine=False` — on a
  fold-free field or when `min(H, W) < 4 * giant_tile`: below that the coarse
  problem is too small to be a useful preview and its own solve is not
  amortised. Every small crop and the whole test suite take the skip path.
- **New report counters**: `coarse_solve_s`, `coarse_folds_before`,
  `coarse_folds_after`, `coarse_iters`, `warm_folds` on `SliceReport`
  (`-1` = the stage did not run).

Measured on the full raw B0039 z16 slice (bilinear rows, objective `none`,
threshold 0.01, maxiter 600, BLAS/OMP pinned to 1):

| | wall | SQP iterations | simplex folds | damage | L2 move |
|---|---|---|---|---|---|
| `coarse_to_fine=True` (default) | **205 s** (incl. 16 s coarse) | **909** (841 fine + 68 coarse) | 3890 -> 0 | 0 | 320.6 |
| `coarse_to_fine=False` | 283 s | 1320 | 3890 -> 0 | 0 | 325.1 |

-28% wall and -31% SQP iterations, and the speed is not bought with fidelity —
the move is slightly *smaller*. The coarse solve cleared 1054 -> 0 folds on its
own grid in 16 s / 68 iterations; the warmed fine field still had 2840 folds, so
the win is a better basin for the fine windows, not folds removed up front.

### Added — the isqp trust region is now a knob, not a constant

- **`tr_delta=2.0` / `tr_max=16.0`** on `dvfopt.core.primitives.isqp.isqp_solve`
  (initial radius / cap, grid units), threaded through `solve_window_inner`,
  `windowed_correct` and the windowed strategy dataclasses. They were hard-coded
  locals; **the defaults are unchanged, so every prior measurement stands and
  the default path is byte-identical.**
- The measured trade, raw B0039 z16: `tr_delta=1.0` runs **267 s / 1022 SQP
  iterations at L2 move 344** vs 300 s / 1320 / L2 325 at 2.0 — -11% wall and
  -23% iterations, but a visibly larger departure from the input. 2.0 stays the
  default; coarse-to-fine is the speedup that costs no fidelity. `tr_max` never
  binds on the measured B0039 windows.

### Fixed — every process pool pins its workers to one compute thread

- **One shared helper, `dvfopt.core._pool.pin_worker_threads()` /
  `pinned_thread_env()`**, forcing `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `NUMBA_NUM_THREADS` and
  `RAYON_NUM_THREADS` to `1` (plus `numba.set_num_threads(1)` when numba is
  importable). `pinned_thread_env()` wraps pool *submits* in the parent —
  children inherit the environment at interpreter start, the only point early
  enough for OpenBLAS/MKL, which read it once at import; `pin_worker_threads()`
  runs at the top of each worker for late imports and nested pools.
- **Wired into every pool in the package**: the CLI's `--n-workers` per-slice
  pool (which pinned only the three BLAS vars), `DVFoptConfig(n_workers=...)`
  (which pinned **nothing** — the actual gap), the persistent 3D pool's warmup
  initializer and `pool_map`, `iterative_parallel`'s window pool, the Laplacian
  correspondence pool, and the cohort benchmark's section pool.
- **Serial paths are byte-identical**: `pin_worker_threads()` no-ops outside a
  child process, so an in-process solve never has its environment rewritten.
- **Thread census** (24-logical-core i7-13700, 8 P-cores + 8 E-cores). An
  unpinned worker carries **53 OS threads** before it does any work: 4 baseline,
  **+23 from `import numpy`**, **+26 more from scipy** — numpy and scipy each
  start a full-width OpenBLAS/OpenMP pool. Clarabel contributes **zero** (its
  qdldl path never starts a rayon pool, so `RAYON_NUM_THREADS` changes nothing —
  it is pinned defensively). Pinned, the same worker carries 1-4 threads.

### Changed — measured `n_workers` guidance: keep it SMALL (2-4), not the core count

Pinning is resource hygiene, **not** a scaling fix — measuring it says so.
24 identical `windowed_correct` solves of the 50x50 `z16_twist` crop
(`inner='isqp'`, bilinear constraint, no objective), wall seconds for all 24 and
mean per-solve; serial reference **34 s/solve** (pinned 34.0 s, unpinned 33.5 s —
the solve is single-threaded work, so pinning costs it nothing):

| workers | wall before | wall after | per-solve before | per-solve after |
|---|---|---|---|---|
| 6  | 399 s | 448 s | 99 s  | 111 s |
| 12 | 624 s | 555 s | 310 s | 275 s |
| 16 | —     | 560 s | —     | 321 s |
| 24 | 646 s | 589 s | 633 s | 581 s |
| 12 (`qp_backend='osqp'`) | 826 s | 617 s | 411 s | 307 s |

So pinning buys 9-25% at >= 12 workers and is inside the noise at 6. The
`'osqp'` A/B also **exonerates Clarabel**: with it out of the loop the unpinned
collapse is *worse*, not better.

The real ceiling is **memory bandwidth**. Pinned throughput on 8 jobs:

| workers | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| wall | 277 s | 150 s | **106 s** | 123 s |
| per-solve | 34 s | 37 s | 52 s | 120 s |
| speedup | 1.0x | 1.8x | **2.6x** | 2.2x |

Throughput peaks at ~2.6x around **4 workers** and declines past it. Two
controls separate the causes: N single-threaded processes running a pure-integer
loop inflate only 1.4x each at N=6 / 2.6x at N=24 (near-linear scaling, 4.4x /
9.2x aggregate), while N processes *streaming a past-L3 array* inflate 3.2x /
9.3x — matching the solve's 3.3x / 17x. SMT siblings and E-cores are not the
issue and adding them does not help: 12 -> 24 workers makes wall time worse in
both the pinned and unpinned runs.

**Recommendation: `n_workers` / `--n-workers` of 2-4** on a machine like this,
and measure rather than assume on a different one. Setting it to the physical
(16) or logical (24) core count is 4-5x slower per solve and *lower* throughput
than 4 workers.

### Changed — hybrid QP backend for the windowed isqp inner (behavior change, default ON)

- **`qp_backend` / `ip_cold` / `ip_after_admm_iters`** on `isqp_solve`,
  `windowed_correct` and `WindowedWrapperStrategy` / `ISQPWindowedStrategy`.
  `'hybrid'` solves a window's **cold** first QP, and any QP that follows an
  ADMM run of `>= ip_after_admm_iters` (default 800) iterations, with
  interior-point **Clarabel**; every other QP stays on warm-started OSQP. The
  IP solution seeds OSQP's warm start, and any IP failure (bad status,
  non-finite, exception) falls through to ADMM — the backend can be faster,
  never less feasible.
- **The engine default changes to `'hybrid'`** (`windowed_correct`,
  `ISQPWindowedStrategy`). The *primitive* default is unchanged:
  `isqp_solve(qp_backend='osqp')` is still the pre-hybrid path, byte for byte,
  and passing `qp_backend='osqp'` anywhere restores it exactly.
- Why hybrid rather than interior-point everywhere: on real giant-tile QPs
  (16k vars, 27k rows) Clarabel takes ~0.25 s / 15-25 iterations at ~1e-9
  feasibility against OSQP's 0.4-2.2 s / 700-4000 ADMM iterations at ~1e-3 —
  but an in-engine *warm-started* OSQP solve averages 0.175 s, so
  Clarabel-always is **slower** (raw B0039 z16: 381 s vs 300 s, and 34% more
  SQP iterations). Hybrid on raw B0039 z16: **262 s vs 300 s (-13%)**, 0
  simplex folds, damage 0, and better fidelity (L2 move 325 vs 346). Policy
  sweep: cold-only 296 s, threshold 400 -> 289 s, **800 -> 262 s (best)**,
  1500 -> 269 s, no-cold/800 -> 281 s.
- **New escalation rung, `report.backend_fallbacks` / `WindowRec.backend_fallback`.**
  The IP legs change the SQP trajectory and on some windows steer it into a
  basin with no escape. A real window (never a giant tile) left GENUINELY
  folded — `cons < -margin_delta`, not merely short of the margin-shifted
  target — is now re-attempted whole on plain OSQP from its ORIGINAL start
  state, ahead of grow-on-failure. Without it the z0_cluster crop finishes one
  triangle inverted at -1.2e-4; with it all three hard crops reach 0 simplex
  folds and raw z16 is bit-identical to the un-rung run (264.5 s, L2 325.1, 0
  fallbacks).
- - **New core dependency `clarabel>=0.9`** (pure-Rust wheels on every supported
  interpreter/platform). Without it, `'hybrid'` silently behaves as `'osqp'`
  (logged once at DEBUG on the `dvfopt` logger).

### Changed — windowed isqp: faster and more robust (behavior change, defaults ON)

- **Per-window no-trust-region fallback** (`no_tr_fallback=True`,
  `fallback_maxiter=200`). A window that fails to reach its target is retried
  ONCE on the same box with the trust region off (legacy backtracking line
  search), warm-started from the failed iterate, *before* grow-on-failure. The
  TR ratio test freezes on sliver-scale violations (~1e-4, inside OSQP's own
  noise) that the line search still clears. The retry keeps whichever iterate
  has the higher constraint minimum, so it is never worse.
- **Two-tier OSQP iteration caps** — `qp_max_iter=2000` (normal window solves)
  and `qp_max_iter_fallback=500` (the fallback solves), threaded into the new
  `isqp_solve(osqp_max_iter=...)` argument (`None` keeps OSQP's 8000 default).
  ~2x faster at unchanged feasibility.
- All four knobs are exposed on `windowed_correct` and `ISQPWindowedStrategy`
  (and hence editable in the GUI's Params → Strategy tab); `WindowRec.fallback`
  records which windows used the retry.
- **Giant-region tiler knobs** — `giant_tile` / `giant_max_sweeps` on
  `windowed_correct` and `ISQPWindowedStrategy`, previously hard-wired inside
  `_solve_giant_schwarz`. **The default tile changes 32 -> 64** (behavior
  change: giant regions are now decomposed into fewer, larger overlapping
  tiles). On a full raw B0039 z16 slice (bilinear rows, objective `none`)
  tile 64 ran 362 s / 22 windows / 1 round / no mop vs tile 32's 685 s /
  264 windows / 3 rounds / 4 mop — 1.9x faster, zero simplex folds and zero
  damage on both, and a smaller move (L2 316 vs 404). Pass `giant_tile=32`
  to restore the promoted-benchmark tiling.
- **Geometry-fit giant tiles** — `giant_tile_fit=True` (default) on
  `windowed_correct` / `ISQPWindowedStrategy` turns `giant_tile` into a
  *target*: `_fit_tile` shrinks it to the largest tile covering the region's
  longest side with an integer number of near-equal tiles (clamped to
  `[0.75, 1.5] x giant_tile`). Tile size acts on cost through grid
  **alignment** — how many Schwarz sweep rounds the tiling needs — not through
  the size itself; a tile that leaves a thin remainder strip along the long
  side costs an extra round. On the raw B0039 z16 giant (a 125x152 box) tile
  64 happens to align (1 round, 374 s) while 56 and 80 do not (2 rounds,
  ~600 s); the fitted 51 aligns by construction (1 round, 345 s).
  `giant_tile_fit=False` is byte-identical to the previous literal-tile
  behavior. Overlap semantics are unchanged.
- Validated on the three hard B0039 crops with
  `correct_dvf(phi, constraint='bilinear', strategy='isqp_windowed',
  objective='none')`: simplex folds 645/598/0 → 0/0/0, damage 0, in
  32s / 22s / 106s.

### Added

- **`dvfopt correct --n-workers N`** — the `--pipeline slices` sweep solves N
  z-slices at once in a `ProcessPoolExecutor` (module-level worker, picklable
  args, spawn-safe; BLAS/OpenMP threads pinned to 1 per worker). Output order,
  `summary.json` and the exit code are identical to the serial path, which is
  unchanged for `N` in (unset, 0, 1) or a single-slice volume. Relatedly,
  `dvfopt.core._pool.get_pool` now caps its request to 1 inside a worker
  process, so no solver nests process pools.
- **`DVFoptConfig(n_workers=N)`** — the DVFopt facade solves the z-slices of a
  volume in a `ProcessPoolExecutor` when `N > 1` and there is more than one
  slice (module-level worker, picklable args, spawn-safe). Serial otherwise;
  results and slice order are identical to the serial path. A *script* calling
  `fit()` with `n_workers > 1` on Windows/macOS must guard the call under
  `if __name__ == '__main__':`.

### Changed — simplex terminology (pure rename, zero behavioral change)

The "2-tri" / "6-tet" fold metrics are renamed to the **simplex** metric:
they are the Jacobian determinant of the piecewise-linear simplicial
interpolant (2 triangles/cell in 2D, 6 tetrahedra/cell in 3D). Full
backwards compatibility via aliases:

- **Classes** (old names stay importable, same class objects):
  `TriConstraint2D` → `SimplexConstraint2D`,
  `TriConstraint2DFullCoverage` → `SimplexConstraint2DFullCoverage`,
  `TriConstraint2DBilinear` → `SimplexConstraint2DBilinear`,
  `Tet6Constraint3D` → `SimplexConstraint3D`.
- **Registry labels** (legacy labels remain registered): `'simplex'` (was
  `'2tri'`), `'simplex_standard'` (was `'2tri_standard'`), `'simplex_3d'`
  (was `'6tet'` / `'6tet_3d'`); `'bilinear'`, `'jdet*'`, `'finite'`
  unchanged.
- **Defaults / displays** now spell the new labels: `correct_dvf` /
  `Solver.from_spec` / `DVFoptConfig` / CLI `correct --constraint` default
  to `'simplex'`; `constraint_fold_stats(..., 'auto')` resolves to
  `'simplex'` / `'simplex_3d'` (the resolved-name return value — and hence
  the CLI `summary.json` `constraint` field — changes spelling); GUI menus
  and reports say "Simplex (2D)" / "Simplex (3D)".

Windowed-engine promotion (the PR #61–64 benchmark fold-corrector moves into
the library) plus an over-engineering cleanup.

### Added

- **`TriConstraint2DBilinear`** (label `'bilinear'`) — the bilinear cell-min
  Jacobian (`cell_min_jdet_2d`) as a constraint: four smooth triangle rows per
  cell (both diagonal splits; the TL-BR pair reuses the TR-BL kernels on the
  x-mirrored field, `core/primitives/tri.py`). `min` of the rows equals
  `½·cell_min_jdet_2d`, so feasibility certifies the bilinear interpolant
  injective on every cell. Accepted by the barrier, the windowed engine
  (`ISQPWindowedStrategy`, locality entry + structural sparsity pattern) and
  `SLSQPWindowedStrategy` (its triangle mode already enforced all four
  triangles); the other 2-tri-specialised strategies reject it at construction.
  `auto_strategy` tiers it like Jdet.
- **`SLSQPWindowedStrategy.accepts_constraints`** — declared (Jdet 2D/3D, the
  2-tri family, bilinear), so an unsupported constraint is rejected at `Solver`
  construction instead of a `TypeError` mid-solve; `auto_strategy`'s no-osqp
  fallback reads that declaration (`'finite'` keeps `barrier`).
- **Windowed engine** — the triangle families' CPR sparsity pattern is built by
  index arithmetic instead of dense probing (`np.eye(m)`, O(m²) memory — a
  cap-sized mop window under bilinear would have needed ~19 GB).
- **`DVFopt` facade** — constraint labels resolve through the constraint
  registry (`make_constraint`) instead of a parallel if-chain, so every 2D
  label (`'finite'` included) is accepted; `core/primitives/constraint_values.py`
  is gone (both callers use the registry). `plot_feasibility` now handles the
  default `'2tri'` snapshots (corner-patch rows) and the per-pixel Jdet map.
- **Windowed engine** — `dvfopt.core.windowed` (`windowed_correct`), the
  third shared engine: one small frozen-ring window per fold cluster,
  no-damage by construction, grow-on-failure, overlapping-tile decomposition
  for giant regions, and a terminal large-margin mop pass. The engine code is
  byte-identical to the PR #61–64 benchmark implementation (verified by the
  promotion's identity gates — 57 gate assertions total).
- **`WindowedWrapperStrategy(inner=<label>)` / `ISQPWindowedStrategy`** —
  wrapper strategies over the engine (registry labels `'windowed_wrapper'` /
  `'isqp_windowed'`). The inner is a window-solver *label*
  (`'isqp'`/`'slsqp'`/`'slsqp+trust-constr'`), not a Strategy — each window
  is a frozen-ring reduced problem a crop-level `Strategy.fit` cannot
  express. `ISQPWindowedStrategy` pins the tuned elastic-QP inner
  (zero-arg constructible; 528/528 B0039 slices cleared, damage = 0 on all
  2178 benchmark tasks). Also exposed in the GUI's 2-tri and Jdet menus
  (visible-but-disabled without `osqp`).
- **`FiniteJdetConstraint2D`** (label `'finite'`) — forward-difference cell
  determinant as a real registered constraint (analytic sparse Jacobian;
  math in `core/primitives/finite_jdet.py`), plus the promoted
  `core/primitives/isqp.py` (elastic-QP SQP, `HAS_OSQP` gate) and
  `core/primitives/coloring.py` (CPR-coloring Jacobians).
- **`solvers` extra** — `pip install dvfopt[solvers]` pulls `osqp` (the isqp
  windowed inner); `osqp` also joins the `dev` extra so CI's `[dev,gui]`
  legs run the no-damage suite instead of skipping it.

### Changed

- **`auto_strategy`** — the Jdet mild tier (`n_neg <= 500` and
  `init_min >= -1`) now prefers `'isqp_windowed'` when `osqp` is installed
  and the constraint is 2D; otherwise it keeps `'slsqp_windowed'`.
- **Dependencies** — dropped `joblib` (replaced its one call site,
  `dvfopt.laplacian.correspondence`'s slice-to-slice correspondence
  search, with stdlib `concurrent.futures.ProcessPoolExecutor`) and
  moved `tqdm` out of the core install (its one call site, the same
  function's progress bar, now logs periodically through the module's
  existing `log_fn` convention) into the `[benchmarks]` extra, where
  `benchmarks/registration/transmorph-registration.ipynb` still needs it.
- **Schwarz strategies** — `SchwarzHarmonicALMRefineRepairStrategy`
  (`M14SchwarzStrategy`) and `SchwarzHarmonicALMRefineRepair3DStrategy`
  (`M14Schwarz3DStrategy`) now build their pinned inner strategy
  (`HarmonicALMRefineRepairStrategy`/`3DStrategy`) from their own knobs
  and delegate directly to the shared `dvfopt.core.schwarz._common`
  core — the same core `SchwarzWrapperStrategy` uses. Public API
  (class names, dataclass knobs, registry labels, aliases) is
  unchanged; the internal standalone shim modules
  `dvfopt/core/wallbreakers/_m14_schwarz.py` and `_m14_schwarz_3d.py`
  (and their module-level functions) are deleted as consumer-free.

### Removed

- **`benchmarks/windowed_isqp.py` + `benchmarks/finite_jdet.py`** — promoted
  into the library (`dvfopt.core.windowed`, `FiniteJdetConstraint2D`, and
  `core/primitives/{isqp,coloring,finite_jdet}.py`) and deleted, following
  the 0.5.0 `slsqp_traced` promote-then-delete precedent. The retained
  harnesses (`fullslice_bench`, `windowed_bench`, `comprehensive_bench`,
  `windowed_escape`, `escape_bench`, `b0039_isqp_bench`, `slsqp_variants`,
  `trace_parity_check`) were repointed at the promoted code — family-string
  translation lives in the new `benchmarks/_windowed_compat.py`, and
  `slsqp_variants._isqp_solve_osqp` remains only as a thin back-compat shim
  over `dvfopt.core.primitives.isqp.isqp_solve` (CLI/printed behaviour of
  every harness unchanged).
- `scripts/check_ci.py` (dead — wired into nothing in CI/nox/pyproject).
- `tools/rewrite_imports.py` (spent one-shot migration tool from the
  0.5.0 reorg).
- Two vacuous regression tests in `tests/test_slsqp_review_fixes.py`
  that asserted the absence of warning code which no longer exists, and
  `tests/test_tri_slsqp.py::test_invalid_anchor_raises`, which only
  re-tested `make_objective`'s own error path (now covered directly by
  `tests/test_objective.py::test_make_objective_invalid_label_raises`).

## [0.5.0] — 2026-08-22

Library reorganization. Behaviour is unchanged — no solver produces a
different number — but **import paths moved**. See the old → new map below,
and [ARCHITECTURE.md](ARCHITECTURE.md) for the rules the new layout enforces.

### Changed

- **BREAKING — `dvfopt.core` is method-first.** One sub-package per algorithm
  family instead of ~20 flat `iterative*_*.py` modules: `primitives/` (shared
  constraint math + the traced SLSQP driver, zero method logic), `nmvf/`,
  `barrier/`, `slsqp_windowed/`, `slsqp_fullgrid/`, `schwarz/`,
  `wallbreakers/`, `slp/`, `marching/`. Sibling method packages never import
  each other; anything two of them need lives in `core/primitives/` or in one
  of the two shared engines, `barrier/_core.py` (penalty→barrier homotopy) and
  `schwarz/_common.py` (domain decomposition).
- **BREAKING — one package.** The top-level `laplacian/` and `test_cases/`
  packages were absorbed into the distribution as `dvfopt.laplacian` and
  `dvfopt.testdata`. `pip install dvfopt` no longer installs (or collides on)
  two generically-named top-level packages.
- **BREAKING — `requires-python >= 3.10`** (was `>= 3.9`) and
  **`scipy>=1.15,<1.19`**. scipy 1.15 already dropped 3.9, and the upper bound
  is load-bearing: the traced SLSQP driver vendors scipy's `_slsqplib`
  private internals (the SLSQP C core), which exist only on scipy >=1.16 —
  itself requiring Python >=3.11 — through 1.18. On scipy 1.15.x (what
  Python 3.10 resolves to) the driver transparently falls back to scipy's
  own `minimize(method='SLSQP')`: same numerics, no per-iteration trace (see
  the Fixed entry below). `uv.lock` and `requirements-dev.txt` are aligned
  with the pin.
- **Objective is a real axis.** Every solver in the package now takes
  `objective=<Objective>` end-to-end; the parallel `anchor='l2' / eps_l1=...`
  string parameters are gone, and `objective_euc` was deleted. `anchor_term`
  moved from the barrier core to [dvfopt/objectives.py](dvfopt/objectives.py)
  (pure numpy — the engine imports *from* it, never the reverse). Kernels that
  cannot call back into Python (numba wallbreakers, torch autograd) take the
  `(kind, eps_l1)` pair from `objectives._kind_eps(objective)`.
- **CLAUDE.md correction:** the phi-pack split is *not* "2-tri/6-tet vs Jdet".
  `Tet6Constraint3D` declares `PhiPack.DX_FIRST` (`[dx, dy, dz]`) so it can
  share the 3D barrier plumbing with `JdetConstraint3D`; only the 2D 2-triangle
  constraints are `DY_FIRST`. `Constraint.pack` is the only thing to trust.
- **`research/` and `archive/` are frozen provenance.** They were
  deliberately not migrated — scripts there still reference pre-0.5.0 module
  paths and are not runnable against this version (use the git history at
  0.4.x).

### Added

- **`ARCHITECTURE.md`** — dependency rules, the phi-pack table, and the
  checklists for adding a method, a constraint, or an objective.
- **Traced C-SLSQP driver** — `minimize_slsqp_traced` / `ineq_dict` at
  [dvfopt/core/primitives/slsqp.py](dvfopt/core/primitives/slsqp.py), now the
  single driver behind all ten SLSQP call sites (full-grid 2-tri and 6-tet,
  Schwarz per-cluster, windowed 2D/3D, coupled k-ring). Byte-identical results
  to `scipy.optimize.minimize(method='SLSQP')` — it *is* scipy's own C core —
  and adds per-major-iteration tracing on top. The windowed path routes through
  a `_window_minimize` shim that falls back to plain scipy when a caller pins a
  non-SLSQP `method_name`.
- **`SolveInfo.extras['slsqp_trace']`** — with `record_history=True` the SLSQP
  strategies lift each run's per-major-iteration trace to this stable path, so
  the GUI and reports never reach into per-phase `PhaseInfo.extras`.
- **`accepts_objectives` + `IncompatibleObjectiveError`** — the objective-side
  analogue of `accepts_constraints`. `Solver.__init__` now rejects a bad
  strategy × objective pair at construction instead of mid-solve;
  `SLPStrategy` declares `(L1Objective, NoneObjective)`.
  `BarrierStrategy(objective_override=...)` lets a composed pipeline pin the
  barrier leg's objective independently of the Solver's.
- **Interactive report — solver-trajectory animation.** The cohort's
  interactive report viewer gains a play/scrub timeline that animates how a
  field's Jacobian-determinant map deforms across the solver's iterations
  (not just before → after). Frames are captured from `correct_dvf_25d`'s
  `progress_callback` (`make_25d_corrector` now takes an opt-in `frames`
  sink), sampled to K ≤ 8 Jdet slices, and embedded (self-contained). Solvers
  that don't stream intermediate fields (e.g. `slp`/`slsqp`) show before/after
  as before.
- **Developer tooling.** `[tool.pytest.ini_options]` scopes collection to
  `tests/` (bare `pytest` no longer over-collects the notebook scratch
  scripts); `pytest-randomly` (order-shuffling), `pytest-xdist`
  (`pytest -n auto`), and `pytest-cov` added; `mypy` gate scoped to the
  cleanly-typed modules (`[tool.mypy]`); a `nox` task runner (`noxfile.py`);
  `asv` solver-perf benchmarks (`asv_bench/`, `asv.conf.json`); Dependabot
  (`.github/dependabot.yml`); and the `ruff-pre-commit` pin bumped to
  `v0.16.3` to match pyproject/CI. `test.yml` now also runs mypy, an
  installed-CLI smoke, and a coverage job. (`pytest -n auto` speeds local
  runs on many-core boxes; CI stays serial — few-core runners don't benefit.)

### Fixed

- **Visualization theme no longer leaks global matplotlib state.**
  `apply_theme` used to set `figure.constrained_layout.use = True` in the
  process-global `rcParams`, so **any** later figure — including
  non-dvfopt code — inherited constrained layout and its
  `fig.tight_layout()` (with a colorbar) raised
  `RuntimeError: Colorbar layout of new layout engine not compatible…`.
  This broke `benchmarks/cohort_benchmark.py` /
  `benchmarks/interactive_report.py` and caused a Qt canvas abort in the
  GUI suite when a dvfopt plot ran earlier in the same process.
  `apply_theme` now leaves the global default alone; each dvfopt viz
  helper passes `constrained_layout=True` at figure creation instead
  (regression test:
  `tests/test_viz_theme.py::TestApplyTheme::test_apply_theme_does_not_leak_layout`).
  Removed the `test_cli.py` workaround fixture that restored rcParams.
- **`iterative_3d_tet_barrier_torch`** evaluated its objective before the
  torch-missing `ImportError` guard; the guard now runs first.
- **Importing `dvfopt` on Python 3.10 no longer crashes.**
  `scipy.optimize._slsqplib` (the SLSQP C core) requires scipy >=1.16, which
  itself requires Python >=3.11; on 3.10, pip/uv resolve to scipy 1.15.x,
  which lacks it, and `dvfopt/core/primitives/slsqp.py` raised `ImportError`
  at *module* import time, taking the whole `dvfopt.core` import graph down
  with it. The module now sets `HAS_TRACED_SLSQP = False` instead of raising,
  and `minimize_slsqp_traced` transparently delegates to
  `scipy.optimize.minimize(method='SLSQP')` when tracing is unavailable —
  identical numerics, no per-iteration trace.

### Import map (old → new)

Dotted module paths, longest-old-first. Nothing was renamed *within* a module —
only the module it lives in changed.

| # | Old | New |
|---|---|---|
| 1 | `dvfopt.core.tri_primitives` | `dvfopt.core.primitives.tri` |
| 2 | `dvfopt.core.barrier_objective` | `dvfopt.core.primitives.jdet3d` |
| 3 | `dvfopt.core._internal.constraint_values` | `dvfopt.core.primitives.constraint_values` |
| 4 | `dvfopt.core._barrier_core` | `dvfopt.core.barrier._core` |
| 5 | `dvfopt.core.iterative2d_barrier` | `dvfopt.core.barrier.jdet2d` |
| 6 | `dvfopt.core.iterative3d_barrier_torch` | `dvfopt.core.barrier.jdet3d_torch` |
| 7 | `dvfopt.core.iterative3d_barrier` | `dvfopt.core.barrier.jdet3d` |
| 8 | `dvfopt.core.iterative2d_tri_barrier` | `dvfopt.core.barrier.tri2d` |
| 9 | `dvfopt.core.iterative3d_tet_barrier_torch` | `dvfopt.core.barrier.tet3d_torch` |
| 10 | `dvfopt.core._internal.io` | `dvfopt.core.slsqp_windowed._io` |
| 11 | `dvfopt.core._internal.metrics` | `dvfopt.core.slsqp_windowed._metrics` |
| 12 | `dvfopt.core._internal.window` | `dvfopt.core.slsqp_windowed._window` |
| 13 | `dvfopt.core.solver3d` | `dvfopt.core.slsqp_windowed.coordinator3d` |
| 14 | `dvfopt.core.solver` | `dvfopt.core.slsqp_windowed.coordinator` |
| 15 | `dvfopt.core.objective` | *(deleted — `objective_euc` is gone; use an `Objective`)* |
| 16 | `dvfopt.core.slsqp` | `dvfopt.core.slsqp_windowed` |
| 17 | `dvfopt.core.iterative2d_tri_slsqp` | `dvfopt.core.slsqp_fullgrid.tri2d` |
| 18 | `dvfopt.core.iterative3d_tet_slsqp` | `dvfopt.core.slsqp_fullgrid.tet3d` |
| 19 | `dvfopt.core.iterative2d_tri_schwarz` | `dvfopt.core.schwarz.tri2d` |
| 20 | `dvfopt.core.wallbreakers._schwarz_common` | `dvfopt.core.schwarz._common` |
| 21 | `dvfopt.core._cluster_2tri` | `dvfopt.core.schwarz._cluster` |
| 22 | `dvfopt.core._nmvf` | `dvfopt.core.nmvf` |
| 23 | `laplacian` | `dvfopt.laplacian` |
| 24 | `test_cases` | `dvfopt.testdata` |
| 25 | `slsqp_traced` *(benchmarks-local module)* | `dvfopt.core.primitives.slsqp` |

Also moved: `anchor_term` (`dvfopt.core._barrier_core` → `dvfopt.objectives`).

## [0.4.0] — 2026-08-19

### Added

- **Command-line interface** — `dvfopt {info, correct, gui}` console
  script + `python -m dvfopt` ([dvfopt/cli.py](dvfopt/cli.py)). `info`
  reports fold metrics (with a `--check` exit code); `correct` runs the
  solver, per-slice sweep, 2.5D marching, or full-3D repair and writes
  `summary.json` + `convergence.png` reports; `gui` launches the live
  solver. `-v`/`-vv`/`--log-file` route the `dvfopt` logger. Exit codes
  0 feasible / 1 folds remain / 2 usage errors.
- **`dvfopt.metrics`** — canonical `FoldStats` / `fold_stats` /
  `constraint_fold_stats` ([dvfopt/metrics.py](dvfopt/metrics.py)); the
  single definition of n_neg / n_below / min / fold-severity. The 2.5D
  and 3D pipeline `_stats` helpers now delegate to it.
- **`dvfopt.io.fields`** — field I/O (`.npy`/`.npz` + NIfTI/MetaImage/
  NRRD) moved out of `dvfopt_gui.io_formats` into the library, with new
  extension-dispatching `load_dvf` / `save_dvf`. Usable without the
  `[gui]` extra.
- **Benchmarks** — cohort 2D-section runner parallelized across
  processes (`n_workers`, PR #40); interactive multi-constraint HTML
  report with ROI selection
  ([benchmarks/interactive_report.py](benchmarks/interactive_report.py),
  PR #41).
- **`SLSQPFullGrid3DStrategy`** — full-grid SLSQP for the 6-tet
  constraint ([dvfopt/strategies/slsqp.py](dvfopt/strategies/slsqp.py),
  [dvfopt/core/iterative3d_tet_slsqp.py](dvfopt/core/iterative3d_tet_slsqp.py)).
  Registered as `'slsqp_3d_tet'`. Uses `Tet6Constraint3D.jacobian()`
  (the sparse Jacobian shipped in PR #12) + the smoothed-L1/L2 anchor
  helper. Comes with the scaling caveat documented in the docstring
  (3D SLSQP doesn't scale to realistic registration problem sizes —
  active-set QP step dominates wall-clock past ~32³ voxels). Tests
  cover direct composition + registry resolution + 2-tri rejection.

- **GPU tet barrier** — penalty → log-barrier homotopy for the 6-tet
  constraint on `torch` tensors via autograd
  ([dvfopt/core/iterative3d_tet_barrier_torch.py](dvfopt/core/iterative3d_tet_barrier_torch.py)).
  Uses the torch forward from PR #11 (`six_tet_volumes_3d_torch`);
  two phases (LBFGS-on-quadratic-penalty then LBFGS-on-log-barrier)
  match the numpy/scipy barrier path. Full-grid only —
  windowed/active-set machinery from `iterative3d_barrier_torch.py`
  (857 LOC of dilation + max-pool + per-component patches) is
  deferred. Optional torch import; raises a clear `ImportError` if
  called without it.

- **`Harmonic3DStrategy` — 3D harmonic wallbreaker** for the 6-tet
  constraint
  ([dvfopt/core/wallbreakers/_harmonic_3d.py](dvfopt/core/wallbreakers/_harmonic_3d.py),
  [dvfopt/strategies/wallbreakers.py](dvfopt/strategies/wallbreakers.py)).
  Registered as `'harmonic_3d'`. Finds 3D fold cores via
  `six_tet_fold_classification`, dilates a ring of feasible boundary,
  and solves a 7-point Laplacian on each displacement channel
  (Dirichlet boundary). The 3D analog of the 2D m02 harmonic step —
  foundation that 3D m10 / m14 / m14-Schwarz would build on (the full
  pipeline is deferred). `polish=True` (default) runs `BarrierStrategy`
  from the harmonic seed to tighten L2/L1 from the input.

### Changed

- **GUI menu strategies construct through the dvfopt registry.**
  `SolverWorker._build_strategy`'s hand-maintained class ladder collapses
  onto `make_strategy` via a method-id → registry-label table
  (`_MID_TO_LABEL`); menu ↔ registry parity is test-enforced
  ([tests/test_gui_strategy_parity.py](tests/test_gui_strategy_parity.py)).
  The toolbar time budget applies uniformly to any strategy exposing the
  knob.
- **Notebook archive sweep.** Moved 6 superseded legacy notebooks to
  `archive/notebooks/` — each was already covered by either
  `notebooks/two-triangle-check/` or a benchmark notebook:
  `run-parallel-corrections.ipynb`, `shoelace-artifact-example.ipynb`,
  `test-shoelace-constraint.ipynb`, `test-injectivity-constraint.ipynb`,
  `test-global-folding.ipynb`, `triangle-jdet-criterion.ipynb`. Three
  others (`slsqp-iterative-refactored.ipynb`, `slsqp-3d.ipynb`,
  `debug-iterative.ipynb`) were flagged for porting to the new API
  but left untouched in this round — they're real demos with legacy
  `iterative_*` imports.

### Added (PR #12 follow-up — already shipped)
- **Sparse forward Jacobian for `Tet6Constraint3D`**, completing API
  symmetry with `TriConstraint2D`. New public helper
  `build_tet_sparse_jac(D, H, W)` in
  [dvfopt/jacobian/tetrahedron_sign.py](dvfopt/jacobian/tetrahedron_sign.py)
  returns a callable `jac(phi_flat) -> csr_matrix` of shape
  `(6*(D-1)(H-1)(W-1), 3*D*H*W)`. The `Tet6Constraint3D.jacobian()`
  method delegates to it. Verified against the analytical adjoint to
  4e-16 and against a dense finite-difference Jacobian to 7e-11.
  End-to-end SLSQP on a planted 3D fold (8 folded tets → 0, threshold
  reached in 8 iterations) covered by a new test. Note: no
  `SLSQPFullGrid3DStrategy` is wired yet — 3D SLSQP at realistic
  problem sizes doesn't scale (active-set QP step dominates). Users
  who want SLSQP-on-tet today call scipy's `NonlinearConstraint(...,
  jac=Tet6Constraint3D.jacobian)` directly; see
  [tests/test_tetrahedron_sign.py:TestSLSQPOnTet](tests/test_tetrahedron_sign.py)
  for the pattern.

- **`dvfopt.jacobian.tetrahedron_sign_torch`** — torch forward for the
  6-tet signed-volume check
  ([dvfopt/jacobian/tetrahedron_sign_torch.py](dvfopt/jacobian/tetrahedron_sign_torch.py)).
  Bit-exact parity with the numpy forward; autograd through it matches
  the analytical adjoint to 4e-16. Building block for a future
  GPU-accelerated barrier-on-tet path; the full windowed barrier
  integration (mirroring
  [iterative3d_barrier_torch.py](dvfopt/core/iterative3d_barrier_torch.py))
  is deferred — torch is in the `[benchmarks]` extra, not core.

### Changed
- **`_compute_constraint_2d` consolidated.** Two near-duplicate copies
  (in [`dvfopt/_plots.py`](dvfopt/_plots.py) and
  [`dvfopt/unified.py`](dvfopt/unified.py)) now share a single helper in
  [`dvfopt/core/_internal/constraint_values.py`](dvfopt/core/_internal/constraint_values.py).
  An `include_patches` flag selects the right behavior per call site
  (plotting code wants `False` to keep the reshape happy; stats code
  wants `True` to match what the solver sees).
- **Type hints added** on the new tet primitives
  (`six_tet_volumes_3d`, `six_tet_fold_classification`,
  `tet_volumes_flat`, `tet_grad_T_v`) and the viz overview functions
  (`plot_fold_overview`, `plot_fold_overview_3d`, `plot_before_after`,
  `plot_before_after_3d`, `plot_solver_comparison`, `jdet_norm`).
- **Lint clean.** Full `ruff check dvfopt/ tests/` pass with 0 errors.
- **Tet primitives now re-exported from `dvfopt.jacobian`**:
  `six_tet_volumes_3d`, `six_tet_fold_classification`,
  `tet_volumes_flat`, `tet_grad_T_v`.

### Fixed
- **`SolveResult.info` type annotation** corrected to `SolveInfo` (was
  `dict`); it has always carried a `SolveInfo` at runtime.
- **GUI M10Tet raised `ValueError` on selection.** The `m10_tet3d` menu
  entry passed `time_budget_s` to `HarmonicALMBarrier3DStrategy`, which
  has no such field. The registry-driven construction applies the budget
  only when the field exists.
- **CI test failure on Ubuntu (torch missing).** `dvfopt/core/iterative2d_barrier.py`
  had an unconditional top-level `import torch`. CI installs only
  `[dev]` (torch is in `[benchmarks]`), so the import failed at module
  load and cascaded into ~50 unrelated test failures (every test that
  used `JdetConstraint2D`, which imports this module for its CPU
  helpers). The bug was masked on PR #10 by the prior lint failure that
  stopped CI before tests ran. Fix:

  - `import torch` is now wrapped in `try/except ImportError` with a
    `torch = None` fallback. The numpy CPU path is unaffected; the
    `iterative_2d_barrier_torch` public entry raises a clear
    `ImportError` if called without torch installed.
  - `dtype=torch.float32` default in `iterative_2d_barrier_torch` was
    evaluated at module-import time; changed to `dtype=None`, resolved
    to `torch.float32` inside the function.
  - `TestBarrier2DTorch` ([tests/test_integration_2d_barrier.py](tests/test_integration_2d_barrier.py))
    and `TestBarrier3DTorch` ([tests/test_integration_3d_barrier.py](tests/test_integration_3d_barrier.py))
    now `pytest.importorskip('torch')` in `setup_method`, so they
    skip cleanly on torch-less installs instead of crashing.
  - [scripts/check_ci.py](scripts/check_ci.py) gained a new
    "no-torch import smoke" job that uses an import-blocker to confirm
    `dvfopt` + `JdetConstraint2D` + `iterative2d_barrier` all import
    successfully without torch. Would have caught this class of bug.

- **CI lint failure in [PR #10](https://github.com/UCI-iGravi/dvfopt/pull/10).**
  `ruff check` failed on Ubuntu (Python 3.10/3.11/3.12) with
  `RUF100: Unused noqa directive (non-enabled: F401)` at
  [tests/test_tetrahedron_sign.py:110](tests/test_tetrahedron_sign.py#L110).
  Replaced the `try/except ImportError + # noqa: F401` pattern with
  `pytest.importorskip('torch')` — same semantics, no noqa needed.

  Root cause: local runs were `ruff check dvfopt/ tests/`; CI runs
  `ruff check dvfopt tests benchmarks` (note the third directory).
  The unused-noqa rule's interaction with the active rule set differed
  enough that the directive was flagged on CI but not locally.

  **New helper** to prevent recurrence:
  [scripts/check_ci.py](scripts/check_ci.py) replays the CI workflow
  steps locally (ruff check + ruff format check + benchmark py_compile
  smoke + pytest). README has a "Development" section pointing at it.

- **`plot_step_snapshot` + `plot_deformed_quads_colored` theme conflict**
  closed — both functions now use `fig.colorbar(im, ax=ax)` + drop the
  manual `tight_layout`/`plt.show` calls. The 2 xfail markers in
  [tests/test_viz_smoke.py](tests/test_viz_smoke.py) are removed.
- **`plot_problematic_triangles` theme warning** silenced — dropped
  the `fig.tight_layout()` call that conflicted with the theme's
  `constrained_layout=True` default.
- **`auto_strategy` for `Tet6Constraint3D`** — added explicit tet-family
  branch that always returns `'barrier'` (the only strategy that
  currently supports tet). Previously fell through to
  `'slsqp_windowed'` which doesn't accept tet constraints and crashed.

## [0.3.0] — 2026-05-29

### Changed (breaking)
- **`'2tri'` now resolves to `TriConstraint2DFullCoverage`**, not
  `TriConstraint2D`. The standard TR-BL split leaves the two
  diagonally-opposite grid corners `(0, 0)` and `(H-1, W-1)` in only
  one triangle each — an asymmetric coverage gap. The full-coverage
  variant adds two opposite-diagonal corner patches at those cells so
  every grid vertex is in ≥ 2 triangles. The patches are 2 scalar
  constraints + 6 gradient terms total — measurable in microseconds.

  - **What changed**: the registry alias `'2tri'` resolves to the
    full-coverage class; the previous standard behavior is preserved
    as `'2tri_standard'`.
  - **`'2tri_full'` removed.** It was a transitional alias for the
    full-coverage class; with `'2tri'` now being full-coverage, it's
    redundant. Existing callers using `'2tri_full'` should switch to
    `'2tri'`. The internal back-compat guards in `_plots.py` and
    `unified.py` were also removed.
  - **Migration**: most code keeps working — `correct_dvf(..., constraint='2tri', ...)`
    and `DVFopt(constraint='2tri', ...)` continue to compile and run,
    just with 2 extra constraints. To exactly reproduce numbers from
    benchmarks recorded before this change, switch to
    `constraint='2tri_standard'`.
  - **What might shift**: L-BFGS-B / SLSQP iteration paths can differ
    slightly because the multiplier vector grew by 2 entries.
    Convergence quality and feasibility verdict are unchanged in
    practice (the corner patches are virtually never folded in real
    data).
  - The class names `TriConstraint2D` and `TriConstraint2DFullCoverage`
    are unchanged — only the registry mapping flipped.

### Added
- **`dvfopt.viz.theme`** — central matplotlib + seaborn theme
  ([dvfopt/viz/theme.py](dvfopt/viz/theme.py)). Single source of
  truth for fonts/spines/dpi/colormaps; previously each plot
  function set its own ad-hoc style. Public:

  - `apply_theme(context='paper')` — idempotent, applies seaborn
    `ticks` style + paper context + custom rcParams (dpi=130,
    no top/right spines, `RdBu_r` default cmap, `savefig.dpi=200`).
  - `reset_theme()` — restore matplotlib defaults.
  - `PALETTE` / `Palette` — curated palette with semantic colors
    (`fold`, `feasible`, `anchor`, `grid_warp`, `grid_ref`) and
    canonical colormaps (`cmap_jdet='RdBu_r'`,
    `cmap_severity='YlOrRd'`, `cmap_magnitude='magma'`).
  - `jdet_norm(jdet_arrays, threshold=0.01)` — TwoSlopeNorm
    builder centered on 0 for diverging Jdet panels.

- **`dvfopt.viz.overview`** — high-impact "money shot" plots
  ([dvfopt/viz/overview.py](dvfopt/viz/overview.py)):

  - `plot_fold_overview(phi)` — 4-panel figure for a folded 2D DVF:
    Jdet heatmap with fold contour, warped grid with **per-triangle
    fold classification** (TR-BL diagonal drawn, T1/T2 shaded
    orange for single-flip / deep red for both-flipped), Jdet
    distribution with threshold marker, per-row/column fold
    counts.
  - `plot_before_after(phi_before, phi_after)` — side-by-side
    Jdet panels with a shared norm + a correction-magnitude
    panel.
  - `plot_solver_comparison(phi_in, results={'slsqp': ..., ...})`
    — N+1 panel comparison of solvers on the same input,
    shared norm.
  - `plot_fold_overview_3d(phi)` — 3D analogue: 3D Jdet scatter,
    worst-z slice heatmap, Jdet distribution with embedded
    **per-voxel tet-flip histogram** (0-6 tets flipped per voxel
    cell, using the new 6-tet decomposition), and per-axis fold
    projections (folds vs Z / Y / X).
  - `plot_before_after_3d(phi_before, phi_after)` — pair of
    3D Jdet scatters with a shared norm.

  All accept the same input layouts as the validator and apply
  the theme automatically; all accept `save_path=...`.

- **`dvfopt.jacobian.tetrahedron_sign`** — 6-tetrahedron signed
  volumes per voxel ([dvfopt/jacobian/tetrahedron_sign.py](dvfopt/jacobian/tetrahedron_sign.py)).
  Decomposes each voxel cell into 6 tetrahedra sharing the main
  diagonal `C0`→`C7`; identity field yields exactly `+1/6` per
  tet, `+1.0` total volume. Public:

  - `six_tet_volumes_3d(phi)` → `(6, D-1, H-1, W-1)` signed
    volumes per tet.
  - `six_tet_fold_classification(phi)` → `(D-1, H-1, W-1)`
    int8 count of flipped tets per voxel cell.
  - `tet_volumes_flat(phi_flat, D, H, W)` — flat-pack form for
    the constraint system.
  - `tet_grad_T_v(phi_flat, D, H, W, v)` — analytical
    `J^T @ v` adjoint via the cross-product form
    `V = sgn * (1/6) * (B-A) · ((C-A) × (D-A))`. Verified to
    1e-10 against central-difference gradient.

- **`Tet6Constraint3D`** — 3D analogue of `TriConstraint2D`
  ([dvfopt/constraints.py](dvfopt/constraints.py)). Enforces every
  per-tet signed volume ≥ threshold; smoother than the per-voxel
  Jdet constraint at fold boundaries. Phi pack `[dx, dy, dz]` (DX_FIRST,
  shared with `JdetConstraint3D`). Registered as `'6tet'` /
  `'6tet_3d'`. Works end-to-end with `BarrierStrategy` — small
  folded fields are feasibilised through the standard penalty →
  log-barrier homotopy.

- **2-triangle primitive home moved.** The flat-pack
  `tri_areas_flat` / `tri_grad_T_v` / *_full_coverage variants
  now live in [dvfopt/core/tri_primitives.py](dvfopt/core/tri_primitives.py)
  (matching its docstring). Old underscore-prefixed names in
  `iterative2d_tri_barrier.py` are aliases for back-compat —
  the 16 callers that imported them keep working without
  changes.

- **Existing 3D viz now uses the theme.** All six functions in
  [dvfopt/viz/fields3d.py](dvfopt/viz/fields3d.py)
  (`plot_jdet_slices`, `plot_jdet_3d`, `plot_jdet_3d_before_after`,
  `plot_neg_voxels_before_after`, `plot_deformation_grid_3d`,
  `plot_grid_before_after_3d`) call `apply_theme()` and use the
  theme's `RdBu_r` cmap + `PALETTE` colors. The
  `constrained_layout` warnings from `subplots_adjust` calls are
  gone.

- **`seaborn`** added as a runtime dependency in
  [pyproject.toml](pyproject.toml).

- **`dvfopt.validation`** — single-source-of-truth input validation
  ([dvfopt/validation.py](dvfopt/validation.py)). Every entry point
  (`Solver.fit`, `DVFopt.fit`, `correct_dvf`, `Constraint.coerce`)
  routes user input through `validate_dvf()`. Public helpers:
  `validate_dvf`, `validate_finite`, `validate_spatial_min_size`,
  `coerce_to_ndarray`.

### Fixed
- **Input handling is now graceful at the boundary**, not 5 frames
  deep:
  - Lists / tuples / array-likes are accepted (auto-`asarray`).
  - `(2, 1, H, W)` and `(3, 1, H, W)` singleton-D layouts accepted.
  - NaN/Inf rejected with a count, before the solver starts.
  - Sub-minimum spatial sizes (H/W < 3, zero-size axes) rejected
    with an actionable message naming the bad axis.
  - Wrong channel counts (`(4, H, W)`, `(H, W)`) rejected with a
    list of accepted layouts.
  - All shape/finite errors raise `SolverConfigError` / `ValueError`
    (never raw numpy errors).
- `int16` / `float32` inputs silently up-promote to `float64` (was
  already true; now documented).
- `DVFopt.fit` defensively copies — the input array is guaranteed
  not to be mutated.

### Changed
- **`dvfopt/strategies.py` split into a `dvfopt/strategies/` subpackage**
  with one file per strategy:

  - [`base.py`](dvfopt/strategies/base.py) — `Strategy` ABC, registry,
    `_build_solve_info` helper
  - [`barrier.py`](dvfopt/strategies/barrier.py) — `BarrierStrategy`
  - [`slsqp.py`](dvfopt/strategies/slsqp.py) — `SLSQPFullGridStrategy`,
    `SLSQPWindowedStrategy`
  - [`schwarz.py`](dvfopt/strategies/schwarz.py) — `SchwarzStrategy`
  - [`wallbreakers.py`](dvfopt/strategies/wallbreakers.py) — `M10Strategy`,
    `M14Strategy`, `M14SchwarzStrategy`

  Public imports are unchanged: `from dvfopt import BarrierStrategy`
  and `from dvfopt.strategies import BarrierStrategy` both work
  via re-export. Strategies register themselves via
  `@register_strategy('label')` at module import time.
- **Private utilities moved under `dvfopt/core/_internal/`**:
  `_io.py`, `_metrics.py`, `_window.py` (the windowed-SLSQP loop's
  internal helpers) now live at
  `dvfopt/core/_internal/{io,metrics,window}.py`. Signals "do not
  import from user code" more strongly than a single underscore.
  `dvfopt.core.solver` still re-exports them for back-compat.

### Deferred (honest)
- **Pulling `iterative_*.py` algorithm bodies into Strategy classes**
  is genuinely multi-day. Each legacy implementation file has 11+
  test / notebook / benchmark importers; cleaning that migration is
  not "polish." The current 2-layer split (Strategy → function) costs
  ~30–50 lines of indirection per strategy and hasn't actively caused
  problems.
- **Full `dvfopt/core/` reorganization** into
  `_math/_loop/_solvers/` subpackages is similarly deferred. Current
  layout (private modules underscore-prefixed, public modules at the
  top level) is readable and stable. The aesthetic gain from deeper
  subpackaging doesn't justify the import-path churn across the test
  / notebook / benchmark fleet.

- **Strategies build `SolveInfo` directly** via the new
  ``_build_solve_info`` helper in :mod:`dvfopt.strategies`. The
  back-compat normalization in :meth:`Solver.fit` still exists but is
  rarely hit — external strategies that haven't migrated continue to
  work transparently.
- Type hints on `Objective` composition classes (`SumObjective`,
  `ScaledObjective`). The remainder of the public surface
  (`Constraint`, `Strategy`, `Solver`, `SolveResult`, `SolveInfo`,
  `PhaseInfo`) is now fully annotated.
- `__all__` added to `dvfopt/unified.py`. CI YAML quotes the `"on":`
  key to avoid YAML-1.1 → bool coercion (cosmetic; GitHub Actions
  handled it either way).

### Added
- **Exception hierarchy** ([dvfopt/exceptions.py](dvfopt/exceptions.py)):
  `DVFoptError` (base), `SolverConfigError` (sub-`ValueError`),
  `IncompatibleConstraintError` (sub-`TypeError`), `FeasibilityError`,
  `BudgetExhaustedError` (sub-`FeasibilityError`). Existing `except
  ValueError` / `except TypeError` handlers continue to work; user
  code can now catch DVFopt-specific failures uniformly.
- **Package logger** at `dvfopt.logger` plus
  `dvfopt.enable_default_handler()`. All solver progress can be
  routed through Python's standard `logging` module; callers control
  verbosity via the normal logging API.
- `SolveInfo.from_legacy_history()` adapter — every strategy's
  free-form `info` dict is normalized into a populated `SolveInfo`
  with `phases: list[PhaseInfo]`. The contract is now used in earnest,
  not just declared.
- Top-level `dvfopt._plots` module (visualization helpers extracted
  from `unified.py`). Matplotlib stays out of `unified.py`'s import
  graph until a plot is actually called.
- `benchmarks/_run_canonical_2tri_suite.py` — demonstrates the
  declarative `BenchmarkSuite` workflow as the migration path from
  hand-written `_run_*.py` scripts.
- New test files: `test_solve_info_and_exceptions.py` (10 tests) and
  `test_logging_setup.py` (5 tests).

### Changed
- `Solver.fit()` now normalizes every strategy's `info` return value
  into a `SolveInfo` instance via a new `_normalize_info` helper.
  Strategies that produce list-of-dicts, dict-with-history, or
  stage-keyed dicts all converge on the same `SolveInfo.phases`
  output. The unified `Result.history_df()` and `plot_convergence`
  consume this uniform shape.
- `Strategy._check_constraint` now raises
  `IncompatibleConstraintError` (instead of plain `TypeError`).
- `DVFopt._validate` now raises `SolverConfigError` (instead of plain
  `ValueError`).
- `unified.py` shrunk from 686 → 538 lines by extracting plot methods
  to `dvfopt/_plots.py`. The `Result.plot_*` methods now delegate to
  the extracted functions.

### Removed
- Committed CSVs and PNGs under `benchmarks/results/` — these are
  regenerated artifacts and now gitignored. Use `git add -f` to
  commit one explicitly when needed.

### Added
- `BenchmarkSuite` — declarative harness in
  [benchmarks/benchmark_suite.py](benchmarks/benchmark_suite.py).
  Replaces hand-written `_run_*.py` scripts with a `(cases, solvers)`
  dict + `.run()` returning a pandas DataFrame. Streams CSV row-by-row.
- `register_constraint` and `register_strategy` decorators for plugin
  extensibility. External packages can register new constraint families
  / strategies and make them available via `make_constraint('foo', shape)` /
  `make_strategy('bar')`.
- `PhaseInfo` and `SolveInfo` dataclasses (in `dvfopt.solver`) —
  standardized history container for cross-strategy comparability.
  Strategies opt in by populating `SolveInfo`; legacy free-form `info`
  dicts still supported.
- Property tests for constraint adjoints via Hypothesis
  ([tests/test_constraint_properties.py](tests/test_constraint_properties.py)).
  Randomizes shape / seed / amplitude over 60+ examples per test,
  catching boundary cases fixed-seed tests miss.
- CI workflow ([.github/workflows/test.yml](.github/workflows/test.yml))
  — runs ruff + pytest on every push to main / nightly across
  Python 3.10 / 3.11 / 3.12.
- Pre-commit hooks ([.pre-commit-config.yaml](.pre-commit-config.yaml))
  — ruff, ruff-format, trailing whitespace, YAML/TOML/merge-conflict
  checks, large-file guard, nbstripout.
- Ruff config in [pyproject.toml](pyproject.toml) with project-tuned
  ignores (E501/E701/E702 for math-heavy code, RUF001-3 for unicode
  in docstrings).

### Changed
- `Constraint.coerce()` — new method on the base class that accepts
  loose input shapes (`(2, H, W)`, `(3, H, W)`, `(3, 1, H, W)`) and
  returns the canonical `(C, *shape)` float64 ndarray. Strategies no
  longer need to do their own coercion. Subclasses override for
  family-specific accommodations (e.g. `TriConstraint2D` accepts the
  legacy 3-channel layout).
- `Strategy.accepts_constraints` (tuple of accepted `Constraint`
  subclasses) replaces the previous `requires_2tri` bool. Documents
  precisely what each strategy accepts and works correctly for future
  constraint types.
- `SliceResult` now extends `SolveResult` rather than duplicating its
  fields. Legacy field names (`init_min`, `final_min`) are kept as
  properties for backward compatibility with the dataframe + plot
  code.
- `DVFoptConfig` slimmed: strategy-specific knobs (`lam_schedule`,
  `mu_schedule`, `barrier_max_iter`, etc.) removed from the dataclass.
  Pass a pre-built `Strategy` instance to `solver=...` for non-default
  knobs, or set them via `strategy_kwargs={...}`.

### Fixed
- m14-Schwarz `fallback_size_ratio` check: previously compared
  cell-space span against corner counts (off by one + units mismatch),
  delaying or skipping the global-m14 fallback for near-full clusters.
  Now uses inclusive cell-space comparison.

### Removed
- Legacy `iterative_*` exports at the top-level `dvfopt` namespace.
  Solver implementations remain accessible from `dvfopt.core.*` for
  internal use but are no longer the public API. Migrate to `Solver` /
  `correct_dvf` / `DVFopt`.

---

## [0.2.0] — Parameterized solver refactor

### Added
- Parameterized public API around three orthogonal axes:
  - `Constraint`: `TriConstraint2D`, `TriConstraint2DFullCoverage`,
    `JdetConstraint2D`, `JdetConstraint3D` — flattening, sparse
    Jacobian, adjoint, all with FD validation.
  - `Objective`: `L1Objective`, `L2Objective`, `NoneObjective` (+
    composition via `+` / `*`).
  - `Strategy`: `BarrierStrategy`, `SLSQPFullGridStrategy`,
    `SLSQPWindowedStrategy`, `SchwarzStrategy`, `M10Strategy`,
    `M14Strategy`, `M14SchwarzStrategy` — uniform interface, wraps the
    existing solver implementations.
  - `Solver` composes the three and returns a `SolveResult`.
- `correct_dvf(phi, constraint=..., objective=..., strategy=...)`
  one-shot convenience.
- `auto_strategy(constraint, init_n_neg, init_min, objective_label)`
  heuristic — picks barrier/m10/m14/m14_schwarz/slsqp based on fold
  density.
- `iterative_2d_tri_refine_repair_schwarz` — cluster-localized m14.
  ~5× faster than global m14 on the full B0039 z=12 slice with ~11%
  lower L1. Includes a final global barrier polish to recover the
  safety margin if Schwarz overlap nicks it.
- `max_grow_iters` parameter on m10 and m14 — speed-vs-L1 tuning knob
  for the harmonic-extension stage. Was previously hardcoded to 8.
- Canonical 2D 2-triangle benchmark suite
  (`test_cases.canonical_2tri_2d`) — the six synthetic correspondence
  cases promoted from notebook 14. Used as the standardized smoke test
  for any new solver.
- `quick_tour.ipynb` demonstrating the parameterized API end-to-end.

### Changed
- `DVFopt` rewired to dispatch through `Solver` instead of the legacy
  `_run_*` methods.
- `_resolve_solver` heuristic moved from `DVFopt._resolve_solver` to
  `dvfopt.solver.auto_strategy`. Now considers slice size to pick
  `m14_schwarz` over `m14` for large slices.

### Removed
- `iterative_2d_tri_smoothmin` — wall-test showed it's strictly inferior
  to barrier in every regime (worse than baseline SLSQP at 400+ folds,
  fails outright at 30×30/379).
- `continuation_steps` parameter on `iterative_2d_tri_slsqp` — marginal
  benefit (~3-5% L1) at API complexity cost.
- `DVFopt._run_trust_constr` — experimental and unmaintained; trust-constr
  can be plugged back in later as a Strategy subclass.

---

## [0.1.0] — Initial release

Initial SLSQP-based correction of negative Jacobian determinants in
2D and 3D deformation fields. Includes:

- Windowed iterative SLSQP (`iterative_serial`, `iterative_parallel`,
  `iterative_3d`).
- Penalty / log-barrier L-BFGS-B (`iterative_2d_barrier`,
  `iterative_2d_tri_barrier`, `iterative_3d_barrier`).
- 2-triangle constraint family (full-grid SLSQP, Schwarz hybrid,
  wallbreakers m02/m03/m10/m12/m14).
- `DVFopt` high-level facade with per-slice tabular reports and plots.
- Canonical synthetic test cases + B0039 real-data slice fixtures.
