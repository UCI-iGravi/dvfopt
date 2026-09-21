# Changelog

Tracks user-visible changes to `dvfopt`. Format inspired by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Docs — canonical 2D origins re-measured to 27/27 under the re-seed fix (pass 5)

`m2_ffd_brainpair_coarse` × `isqp_none`, the one uncertified row left in the canonical 2D origins
taxonomy (`origins_all_v4`, 26/27), was re-measured under `main` `2edd8ff` after the terminal
re-seed fix (PR #127, `a5a3a51`): certified, bilinear cells below gauge 6 -> 0, worst value
−0.00052 -> +0.01099, damage 0, 5 rounds / 147 windows / 2,572 SQP iterations, 982.9 s wall. Origins
`isqp_none` is now 27/27 (`isqp_l2` and `auto` stay 27/27, `isqp_l1` 24/27). The tracked run
`docs/paper/results/2d_canonical/origins/` is now `origins_all_v5` (214 rows reused, 2 rerun — the
re-measured pair plus an unplanned third `WorkerCrash` rerun of `m2_ffd_brainpair_fine` ×
`slsqp_windowed`, unchanged under ruling R18). Findings note §12.11 and
`docs/paper/results/2d_canonical/README.md` carry the full numbers and root cause.

### Docs — findings section 13: the 3D windowed engine at band scale

Two spikes and a full-volume route attempt (2026-09-18 to 09-21) asked whether the 3D windowed
engine, which certifies crops and sub-volumes up to 24 voxels per side (15/15 finished runs at 0
folds, 0 best-diagonal floor, damage 0), can reach band or full-volume scale. It cannot yet: the
QP backend, factorization reuse, tile size, `qp_max_iter=2000`, the 3D coarse warm start and
crop-and-paste composition were all measured and refuted as levers; the one lever found is the
`giant_workers` RAS tile pool (2.77x the serial clearing rate at damage 0, saturating at 4
workers) plus pinning Clarabel to one thread (7.49 → 1.08 cores, −33% wall, no change to 2D
numerics). The banded full-volume driver was cut at 35 h 48 m with band 0 — B0039's z[0, 24),
holding 489 of the 2.5D residual's 517 folds — still unfinished. See
`docs/superpowers/notes/zero-folds-campaign-findings.md`, section 13.

### Changed — 3D windowed-engine defaults from spike 2: the coarse warm start is off on 3D, and Clarabel is pinned to one thread

Values and settings only; no engine logic changed. Evidence: `benchmarks/output/spike_3d_2/REPORT.md` (gitignored).

- **`coarse_to_fine` joins `DEFAULTS_BY_DIM` with the 3D column `False`** (2D stays `True`, unchanged). The stage needs 64 voxels per axis (`min(shape) >= 4 * giant_tile`), so it had never run on the 24³ crops the phase-3 3D column was tuned on. Measured on a 64³ cut of the B0039 Laplacian field (24,238 of 250,047 cubes below 0.01, 3 h budget, `giant_tile=16`): `coarse_factor=4` is **neutral** (1,015 vs 1,028 SQP iterations to essentially the same fold count, −1.3 %) but the axial edge-projection census shows it **introduces 12 negative projections / 4 rotated cells** where the stage off introduces **0**; `coarse_factor=2` introduces **568 / 95**, leaves 21,694 folds against 20,457 at 98 % of the wall, spends half the budget (5,565 s) on the coarse solve, and is the only arm of that spike to make the worst cell worse (−8.4258 → −9.5413). On the 2.5D residual's densest box (65×128×152, 493 folds) the default warm start took it to **23,277 folds** with the minimum **−0.035 → −13.1** and L2 move 625, against 286 folds / min −0.096 / L2 39.9 with the stage off — `damage` **0 in both**, because every new fold sits inside the set the solve was entitled to move. The sharp edge is the table's usual one: `coarse_to_fine=True` on a 3D field reads as "the 2D default" and resolves to `False`; `dim_defaults=False` takes it literally.
- **Clarabel's `max_threads = 1`** in `_HybridQP._solve_ip` (`dvfopt/core/primitives/isqp.py`). Clarabel spawns its own Rayon pool that `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `NUMBA_NUM_THREADS` do not reach. Measured on one 17³ frozen-ring window (10,125 free variables, 35,376 rows): **7.49 → 1.08 cores**, an identical SQP iteration count and constraint minima agreeing to **1e-12**, and **−33 % wall** on an idle box — its internal parallelism was a net loss at this window size. `dvfopt.core._pool` already exports `RAYON_NUM_THREADS=1` to spawned pool workers; this covers serial runs.

  **The pin changes 3D numerics, and that is accepted.** Clarabel's multithreaded factorization returns a different — but equally deterministic — answer than its single-threaded one, which perturbs the SQP trajectory on windows large enough for its internal parallelism to engage (3D only; the 2D windows never do). The consequence is a *convergence*: `dvfopt.core._pool` already pinned spawned pool workers via `RAYON_NUM_THREADS`, so before this change a serial 3D run and a pooled one disagreed on the same input. **They now match bit for bit.** Measured main vs branch on the six crops (`windowed_correct`, threshold 0.01; 3D = `SimplexConstraint3D` + `L2Objective`, 2D = bilinear + `NoneObjective`; engine defaults; the two sides ran concurrently on a 24-core box, so walls are lightly loaded):

  | pair | `array_equal` | max abs diff | wall branch | wall main | L2 branch | L2 main |
  |---|---|---|---|---|---|---|
  | 2D `z16_twist` | **True** | 0 | 3.3 s | 3.5 s | 123.1413 | 123.1413 |
  | 2D `z0_cluster` | **True** | 0 | 1.8 s | 1.8 s | 787.4865 | 787.4865 |
  | 2D `z0_sliver` | **True** | 0 | 43.7 s | 44.7 s | 805.7709 | 805.7709 |
  | 3D `twist` | False | 6.78e-1 | 603.3 s | 810.7 s | 27.0550 | 27.0306 |
  | 3D `sliver` | False | 4.68e-1 | 1030.5 s | 1159.0 s | 23.3012 | 23.2803 |
  | 3D `moderate` | False | 1.70e+0 | 821.3 s | 980.5 s | 58.7958 | 58.7325 |

  **Every row reaches 0 folds at damage 0 on both sides** (403 / 1094 / 1217 → 0 on the 3D crops, 663 / 605 / 18 → 0 on the 2D ones); the L2 move rises 0.09 / 0.09 / 0.11 % and the pinned side is 26 / 11 / 16 % faster. The divergence is attributed, not assumed: repeating the 3D `twist` solve on each side gives **main vs main byte-identical** and **branch vs branch byte-identical**, so both paths are individually deterministic and the difference is the thread count, not scheduling noise. The 3D coarse-warm-start change above is not involved — `warm_folds == -1` on all six crops, the stage needing 64 voxels per axis.

  **The pre-registered byte-identity gate for 3D was therefore FAILED as written** ("the thread pin must not change numerics"). By the user's ruling (`.superpowers/sdd/2026-09-19-3d-defaults/progress.md`, 2026-09-19) it is re-stated as: 2D byte-identical (met), 3D **equal verdicts** — 0 folds, damage 0 — with the numeric difference **attributed** (met). The pin is kept, the new 3D outputs are the baseline, and no knob is added.
- **Documented, not changed: `giant_workers=4` is the 3D band-scale throughput setting.** On a 48×64×64 cut with 17,722 folds under a 3 h budget, 4 RAS workers cleared **14,883** folds where serial cleared **3,781** in the same wall (**2.77×** the clearing rate) at damage 0, and it was the only setting in the spike to improve the worst cell (−8.4258 → −4.5949). **8 workers are byte-identical to 4 in every recorded field, at the same wall** — a sweep is bound by its slowest tile, so the extra workers are waste. Two engine behaviours callers must know, both recorded as facts rather than changed: (i) **`time_budget_s` is not checked inside a RAS sweep**, so a `giant_workers > 1` run finishes the sweep regardless — both pool arms overran their 3 h budget by 46 %; (ii) **the engine's certificate does not compose by crop-and-paste** — `windowed_correct` freezes each WINDOW's ring, not the rim of the sub-volume it is handed, so a window reaching the box border moves border voxels whose remaining corners were never in that box's constraint set. On the full-volume 2.5D residual, 10 of 11 boxes solved to 0 folds locally at damage 0 (four of them having moved their entire rim), yet pasting them back took the volume **33 → 43 folds** below zero (best-diagonal floor 14 → 29), **16 of the 31 new cells on box seams**. Freeze the handed sub-volume's rim, or hand the engine the whole field.

### Fixed — the terminal harmonic re-seed stage now sees orientation-row violations, not only folds

- **The defect.** `_reseed_stage` (`dvfopt/core/windowed/_common.py`) built its re-seed mask from `pixel_fold_mask` alone. A 180°-rotated cell (both axial edge projections negative) has a POSITIVE determinant, so a rotated strip is area-FEASIBLE and enters no fold mask — only the sliver folds at its seam do. Dilating those by `reseed_radius=2` does not reach the strip, so the harmonic fill's ring is itself rotated, the fill inherits the rotation, and the stage stops on no progress. Measured on `m2_ffd_brainpair_coarse` × (bilinear, isqp, `objective='none'`) — the one uncertified `isqp_none` row of the canonical 2D origins sweep: the giant-tile Schwarz sweep leaves a rotated column strip at col 193 (rows ~8-29) and the engine ends at 6 bilinear folds (worst −5.2e-4) with 22 rotated cells; re-running the engine on that stalled state reproduced it (162 s, min −6.6e-4).

- **The fix.** `_orientation_violation_mask(phi, delta)` (next to `_orientation_rows` / `_orientation_rows_3d`, pack-agnostic: it reads the `(C, *shape)` field, channel `a` being the displacement along axis `a`, any rank) marks both endpoints of every axial edge with `1 + diff(phi[a], axis=a) < delta`. When the engine's orientation rows are on (`orientation_delta`, default `0.01`) `_reseed_stage` ORs that mask into the fold cells' corner pixels before the `reseed_radius` dilation. Everything else is unchanged: the stage still returns immediately on a fold-free field, so **every field that certifies today is byte-identical**, the no-progress test stays on fold counts, and `reseed_px` still counts the mask. With `orientation_delta=None` the path is unchanged.

- **Measured.** `m2_ffd_brainpair_coarse` from raw (bilinear / isqp / `objective='none'` / threshold 0.01 / engine defaults, input 16,519 of 41,817 cells folded, min −3.78): **0 bilinear folds, 0 simplex folds, 0 rotated cells, damage 0**, min 0.0110, L2 move 4751, 5 rounds / 147 windows, re-seed 1 round / 1139 px / 3 → 0 folds, 1117 s (box shared with a test run). On the engine's own stalled output the re-run now certifies (763 s) where it previously ended at −6.6e-4. Byte-identity re-checked on the three hard B0039 crops (`z16_twist` / `z0_cluster` / `z0_sliver`, same config): `np.array_equal` vs `main` on all three, walls 3.7 / 2.0 / 45.4 s (main 3.6 / 2.0 / 46.2 s). Tests: `tests/test_windowed_reseed.py` gains a unit test of the mask on a synthetic rotated strip and a (gitignored-artefact-gated) integration test on the stalled field.

### Added — canonical 2D benchmark: `benchmarks/canonical_2d.py`, its tests, and the tracked results in `docs/paper/results/2d_canonical/`

- **The driver.** `benchmarks/canonical_2d.py` runs one pre-registered protocol over every 2D source in the repo: `origins` (27 `dvf_origins` fields), `cohort` (7 brains, `z in range(0, 528, 48)` plus the 8 named hard slices, 85 slices), `ants` (the same slices of the ANTs warps, the "already injective" control), `crops` (the 3 hard B0039 crops, labelled the TUNING SET) and `synthetic` (13 cases). Seven configs (`CONFIGS`): `isqp_none`, `isqp_l2`, `auto` (all bilinear), `slp`, `barrier`, `m14` (simplex_standard) and `slsqp_windowed` (jdet). Each pair calls `correct_dvf(phi, threshold=0.01, record_history=True, **config)` with no per-case knobs. `dvfopt/` is untouched on the branch.
- **The metric block** (`metrics()`), one row per `(input, output, result)`: the `cohort_benchmark` row schema; four certificate families (`simplex`, `bilinear`, `finite`, `jdet`) before and after, each counted at 0 (`n_neg`) and at the 0.01 gauge (`n_below`), plus minima; `certified` = 0 bilinear values below `0.01 - 1e-5` (the bilinear column is `cell_min_jdet_2d / 2`); `frac_nonpos_jdet` and whole-slice `sdlogj`; engine accounting on windowed rows only (`damage`, `n_windows`, `giant_regions`, signed `mop_cleared`, `rounds`, `sqp_iters`); moves (`moved_frac`, L1 / L2 / max, `mean_move_moved`); correspondence residuals on cohort rows; IFT-radius diagnostics. `-1` is a sentinel on every other row type, and aggregates skip it by row type. `sqp_iters` excludes the re-seed and re-anchor polish iterations, which the engine records with `n_iter = 0`.
- **The runner.** Spawn pools with workers pinned to one BLAS thread. Rows are written incrementally, and nothing is dropped: a load failure, a dead worker or a watchdog cut is a full `-1` row in every denominator, and a solve that raises is never certified. Pool-break recovery rebuilds the pool and reruns the break's suspects alone, so a pair that kills its solo pool becomes a measured `WorkerCrash` row. `--resume RUN_DIR` reuses measured rows and their sha256-verified fields, reruns only `BrokenProcessPool` / `WorkerCrash` losses, and refuses a mismatched `threshold`, `cap_s`, `timing_mode` or `watchdog_timeout_s` and a resume into the same directory. `--isolate-config CFG...` runs those configs one pair at a time in their own single-worker pool. A pool closes without blocking on worker exit, which fixes D1, a shutdown hang when workers hold nested pools. The no-progress watchdog (`CANONICAL_2D_NO_PROGRESS_S`, default 6 h) records the stalled running pairs as kept `WatchdogTimeout` rows with `hit_cap=True`. `--cap-s` (default 7200 s) is recorded, not enforced. Provenance in `summary.json`: `git_commit`, `git_dirty`, `dvfopt_path`, `driver_path`, `dvfopt.__version__`, the configs, `n_workers`, python and OS, box load at start, `cap_s`, `watchdog_timeout_s`. Other flags: `--source`, `--config`, `--n-workers`, `--serial-timing`, `--run-dir`, `--figures`, `--table`, `--hist-case`, `--sample`.
- **Tests.** `tests/test_canonical_2d.py`, synthetic inputs only: the metric block, engine stats, correspondence stats, ANTs loading, load-failure and raised-solve rows, pool-break recovery, `--isolate-config`, `--resume` and its refusals, deduplicated sources and configs, the nested-pool shutdown and a deterministic watchdog test. 38 passed at `e1d3a88`.
- **Tracked results** (`docs/paper/results/2d_canonical/`, README with protocol, metric and gauge definitions, walls, caveats and regeneration commands). Damage 0 on every windowed row.

  | source | config | certified | wall median, s |
  |---|---|---|---|
  | crops (TUNING SET, throughput, load 76 %) | `isqp_none` / `isqp_l2` / `auto` | 3/3 each | 6.28 / 41.43 / 32.17 |
  | crops | `barrier` / `m14` / `slp` / `slsqp_windowed` | 0/3 each (`slsqp_windowed` 3/3 own gauge) | 2.195 / 6.617 / 8.286 / 915.6 |
  | synthetic (throughput, load 24 %) | `isqp_none` / `isqp_l2` / `auto` | 13/13 each | 0.1276 / 0.3028 / 0.2246 |
  | synthetic | `barrier` / `m14` / `slp` / `slsqp_windowed` | 1/13, 1/13, 4/13, 0/13 (`slsqp_windowed` 13/13 own gauge) | 0.4013 / 0.5038 / 0.7087 / 0.06125 |
  | origins (serial, load 2 %) | `isqp_none` / `isqp_l2` | 26/27 / 27/27 | 4.348 / 10.25 |
  | ANTs controls (load 2 %) | `isqp_none` | 85/85, 0 pixels moved | 0.4254 |
  | cohort (throughput, load 39 %) | `isqp_none` / `isqp_l2` | 85/85 / 85/85 | 86.45 / 209.6 |

  **Final origins taxonomy** (`origins_all_v4`, all 8 configs including the `isqp_l1` amendment, throughput/pooled): `isqp_l2` 27/27, `auto` 27/27, `isqp_none` 26/27, `isqp_l1` 24/27, `slp` 3/27, `m14` 4/27, `barrier` 3/27, `slsqp_windowed` 3/27 certified. The `slsqp_windowed` baseline (central-difference Jdet, dense SLSQP) ran each pair isolated under a 6 h watchdog cap: 5 of its 27 pairs hit that cap (`WatchdogTimeout`), 1 raised a 429 GiB `MemoryError`, 1 crashed twice under isolation (`WorkerCrash`, ruling R18 — a measured failure of the config, not an infrastructure loss) and the remaining 17 finished but were not bilinear-certified. **`isqp_l1` amendment (ruling R19):** the windowed engine's own L1 anchor, added on the small sources (origins/crops/synthetic) after the pre-registered protocol closed — 24/27 origins, 3/3 crops, 13/13 synthetic certified; its three origins misses are the three heaviest brain-pair fields, and its measured signature is a sparser move (moved fraction 0.108 vs 0.181 for `isqp_l2`) rather than a smaller or faster one, so on this evidence it is less robust than `none`/`l2` on dense fold fields and not a faster route.

  The one uncertified origins serial row is `m2_ffd_brainpair_coarse` × `isqp_none` (39.5 % folded input, 6 bilinear cells short, worst −5.09e-4); `isqp_l2` certifies it. All 8 named cohort hard slices certify under both configs, at a landmark-residual cost on the pin-collapsed ones (B0039 z1 median residual 1.56 → 132.7 px under `isqp_l2`). Serial vs throughput over the 54 origins pairs: 0 verdict flips, 0 discrete-field differences. `slsqp_windowed` hits a 429 GiB dense SLSQP workspace on a 37,920-free-pixel window. Details in findings note §12.

### Added — 3D windowed engine, phase 4 (Scale): banded full-volume driver, resumable checkpoints in the engine, the tiler's box-scoped counts and patch-only RAS tasks

- **The engine.** `_folds_in_box(constraint, phi, box, threshold)` (`dvfopt/core/windowed/_common.py`) replaces the two whole-field `min_field` fold counts inside `_solve_giant_schwarz`'s sweep loop (serial and RAS) with a box-scoped count: the box is padded by one grid point per side (clipped to the field) and `min_field` runs on that sub-field only — every cell/cube/central-difference stencil landing inside the box reads at most one grid point beyond it, so the padded sub-field reproduces those values bit-identically, border flags included. The RAS (`giant_workers > 1`) branch now cuts each tile's task from the tile's ring+1-padded patch instead of the whole field — `build_subproblem` pads the local box by `ring` and clips to the patch exactly where the true image border would clip it, so the enforced rows and border flags reproduce the whole-field solve exactly (asserted byte-for-byte in `tests/test_windowed_phase4.py`); a worker now holds one tile's patch, not the field. `windowed_correct` gains `checkpoint_dir=` (`None` = off, unchanged) and `touched_out=`: a unit is marked after the coarse stage, after each round, after each giant-tiler sweep, after the mop, after the re-seed stage and after the re-anchor stage, mirroring the field to `<checkpoint_dir>/field.npy`/`state.json` and `touched` to `<checkpoint_dir>/touched.npy`. A resumed run reloads both, restores the last row's counters into the report, skips every unit already done, and re-enters the round loop with a fresh fold mask — `rep.windows`/`rep.history` cover only the post-resume work, `time_budget_s` counts from the resume, and the certificate and damage accounting (against the ORIGINAL input, through the restored `touched`) are the same invariants an uninterrupted run has. `report.resumed_from` names the last done unit (`''` uninterrupted, `'finished'` a completed reload); `ck.finish` — the mark that flips a checkpoint's `stage` to `'done'` — fires only when the run actually completed (not on a `time_budget_s` exit), so a budget-cut run resumes from its last real unit rather than being mistaken for finished. A checkpoint whose stored knobs/shape/input hash disagree with the call raises `ValueError`. `touched_out` (optional bool array) is OR-ed with the engine's final `touched` mask in place, letting a caller compose no-damage accounting across multiple `windowed_correct` calls — the hook the banded driver's seam pass uses.

- **`dvfopt.checkpoint`.** `atomic_replace(tmp, dst, *, retries=8)` — `os.replace` with the Windows `PermissionError` backoff (a scanner/indexer can hold the target for a moment) — is factored out of `RunCheckpoint._save` so the engine's own `touched.npy` write, and the banded driver's, share it instead of duplicating the retry loop. `RunCheckpoint.note(key, row)` records `row` under `state['rows'][key]` WITHOUT appending to `done`, for run-level counters a resumed run needs back that are not a resumable unit (the banded driver's seam-pass counters — a `'seam'` entry in `done` would make the slab lambda's `int(str(u))` raise inside `restore_into`).

- **The banded driver.** `windowed_correct_banded(phi_in, inner='isqp', *, constraint, threshold, objective=None, band=24, overlap=8, n_workers=1, checkpoint_dir=None, verbose=1, **engine_kw)` (`dvfopt/core/windowed/_banded.py`, exported from `dvfopt.core.windowed`) drives a large `(3, D, H, W)` volume without ever handing `windowed_correct` the whole field at once. Design: z splits into cores `[k*band, min(D,(k+1)*band))`; every band's slab (`core ± overlap`) is cut from ONE pre-sweep snapshot — every worker sees the same field, so this is restricted additive, Jacobi-style, Schwarz across bands, not a sequential chain — solved by the full engine on that slab alone (`giant_workers=0`, `checkpoint_dir=None`, `verbose=0`), returning only the core planes plus the core's own `touched`. The parent pastes, checkpoints (`band:<k>` unit) and logs each core as its result arrives — a streamed `ProcessPoolExecutor.map`, not a batch collected after the whole sweep — so a kill mid-sweep loses only the bands still running. After every band is committed, ONE unconditional serial call to `windowed_correct` runs on the composed field (`checkpoint_dir=<dir>/seam`, `touched_out=touched`) and repairs whatever the banding left; it is unconditional, never gated on folds being present, because each band discards its overlap on return, leaving one mixed voxel plane at every core-to-core boundary that no band's own `touched` covers — this pass is what folds that plane into `touched` at all. Its cost (`report.seam_windows`) is the cost of banding; a boundary-free volume never triggers real work there. `overlap` must be at least `margin + ring` (the frozen context a window's free pixels need to evaluate correctly) and `band` must be `>= 1`, both asserted at entry. Damage is measured against the ORIGINAL input. `BandedReport` fields: `bands`, `band`, `overlap`, `band_walls` (seconds/band), `band_folds_after`, `folds_before/after/after_zero`, `best_diag_floor_after[_zero]`, `min_before/after`, `damage`, `n_windows` (bands + seam), `seam_windows`, `seam_folds_before`, `time_s`, `resumed_from`, `seam` (the seam pass's own `SliceReport`, `None` on a finished reload — its counters are restored from a `note('seam', ...)` row instead).

- **Memory** (final-review measurement, full-res `(3, 528, 320, 456)` banded run). Before this branch's fix wave the parent peaked at ≈17 GB, at the seam tail: the working field, the band slabs, the results list kept alive through the seam pass, the seam engine's own copy of the field, and the final certificate block, all live at once. After the wave, ≈8-10 GB: the parent holds the working field, `orig_fold`, `touched`, at most one in-flight band result at a time, and then — once the sweep is done and its slabs/results are freed — the seam engine's own copy, its min-field map, and the 2.45 GB `best_diagonal_min_volume` block. Each worker holds one slab (well under 1 GB). The spec's "one field copy per process" is true of the workers; it was never true of the parent, and the module docstring said otherwise before this fix.

- **Gates.** 2D byte-identity (`IDENTITY PASS`, 21 keys vs the #122 `route` reference set) held on eadf6a9, e258223 and the final code head b454ead; the one commit after it, 1b39f93 (the R15 checkpoint-write-failure follow-up), touches `_banded.py` only, which has no 2D entry point. 3D reproduction on eadf6a9 of twist / sliver / sub20 / subvol16 matched the pre-change records exactly in `folds_out`, `folds_out_zero`, `floor_out`, `new_folds`, `damage`, `moved_frac`, `l2_move` and `l1_move`; the only record differences are the #123 driver's de-duplicated `sqp_iters` and a `rounds` key the old records predate.

- **Measured (2026-09-12 → 14).** The pre-registered rows below were not achieved: both full-volume chains were stopped — 38 h and 33 h of wall time spent, no band finished either. In their place, a 30-min instrumented probe supplies the phase-4 cadence number, the projection it implies, and the resulting scope call for this engine.

  | run | workers | wall | CPU per worker | bands done |
  |---|---|---|---|---|
  | slab A/B, banded, 2 workers (`slab_banded2`) | 2 | 38 h | 37 CPU-h | 0 |
  | ds2 B0039 exterior, banded, 4 workers (`ck_ds2_b0039`) | 4 | 33 h | 33 CPU-h | 0 |

  Both ran with the box shared with an unrelated job throughout (load 59-72 %). The band workers run at `verbose=0`, so neither chain gave any per-window visibility — hence a 30-min budgeted probe (`probe_band.py` / `probe_band.log`) on ds2 band 0's slab alone, from the `-gate` snapshot:

  | slab shape | folds | giant region | tiles/sweep | windows solved | s/window (median / max) | SQP iters (median / max) | feasible | folds before → after |
  |---|---|---|---|---|---|---|---|---|
  | (3, 40, 160, 228), 1.46 M voxels | 12,832 (27 % of the volume's 48,274, in one band) | bbox (0..40, 41..134, 68..160), 342k voxels | ≈256 sixteen-cube tiles | 4 in 1,881 s | 493 / 854 | 29 / 61 | 1/4 | 12,832 → 12,759 |

  Patch 5,508 voxels; ≈17 s per SQP iteration on the contended box (phase 1 measured 11 s at 17³ idle); 3 no-TR fallbacks, 0 backend/patience; damage 0. Projection: ≈256 tiles × ~600 s ≈ 43 h per sweep per band, up to 8 sweeps, then rounds / mop / re-seed on top of that — days per ds2 band, weeks for the ds2 volume, months for the full-res volume; the slab bands (3.5 M voxels, ~110k folds each) are larger still. Not stuck: this is the intrinsic per-QP cost of a 16³ tile times the tile count at real fold densities (the phase-1 U1 curve: 0.29 / 11 / 69 s per SQP iteration at 9³ / 17³ / 25³).

  Conclusion: band-scale and full-volume 3D certification with this engine needs a ~100x cheaper window QP (the phase-3 QP-backend candidate) or far smaller tiles (9³ = 0.29 s/iteration, 5.5x cheaper per voxel-iteration, at the price of more seams). Until then the 3D windowed engine is a crop / sub-volume certifier (the 24³ pack, the 17³ / 20³ artefacts), the composite `m10_windowed_3d` is the cheap route where `m10_3d` certifies alone, and full volumes remain the 2.5D pipeline's territory (1,058,831 → 33 in 16.5 h on the 528-slice B0039). The banded driver and its checkpoints are correct and tested (the unit tests and the fix-wave gates above) but unmeasured at scale beyond this probe.

  | artefact | shape | voxels | folds @ 0.01 | folds @ 0 | best-diag floor | min |
  |---|---|---|---|---|---|---|
  | `slab_240_288` | (3, 48, 320, 456) | 7.0 M | 223,287 | 138,583 | 188,111 | −12.10 |
  | `B0039_laplacian_exterior_ds2` | (3, 264, 160, 228) | 9.6 M | 48,274 | 40,870 | 34,382 | −14.89 |

- **Not in this PR.** No `auto` / default / strategy-field / GUI change. `checkpoint_dir` is an engine kwarg (and the banded driver's own), not yet plumbed through `ISQPWindowedStrategy` or the CLI's windowed path. The seam pass is serial — no `giant_workers` inside it. The mop's own tiler is not checkpointed per sweep. `_banded.py` sits outside mypy's `files=`.

### Added — 3D composite strategy `m10_windowed_3d` (`m10_3d` then the windowed engine on its residual): measured against the pre-registered criterion, ships as an opt-in label; `auto` unchanged

- **The strategy.** `HarmonicALMBarrierWindowed3DStrategy` (alias `M10WindowedTetStrategy`; registry labels `'harmonic_alm_barrier_windowed_3d'` and `'m10_windowed_3d'`; `dvfopt/strategies/composite3d.py`, exported from `dvfopt`) runs the `m10_3d` wallbreaker (`HarmonicALMBarrier3DStrategy`: harmonic seed + PHR-ALM + barrier polish) and then `ISQPWindowedStrategy` on its output — the composite the entry below recommended. Its two dataclass fields `bulk` and `windowed` are full strategy instances with their own knobs; the composite adds only `time_budget_s`, forwarded to the **windowed stage** (`HarmonicALMBarrier3DStrategy` has no budget knob), so the GUI toolbar budget reaches the stage that can honour it. `accepts_constraints = (SimplexConstraint3D,)`, every objective, needs `osqp`. Both stages always run with `record_history=True`; the merged `SolveInfo` (`strategy_name='m10_windowed_3d'`) carries the bulk stage's phases prefixed `bulk:` followed by the windowed stage's prefixed `windowed:`, `total_iter` the sum, `feasible_after_phase` the FIRST index at 0 folds (the bulk's if it already cleared), and `extras` = the windowed stage's extras (`damage`, `n_windows`, `l2_move`, …) plus `bulk_n_neg_after`, `bulk_n_below_after`, `bulk_min_after`, `bulk_wall_s`, `bulk_extras`. **The windowed stage's `damage` is measured against the bulk output, not the input**; the driver's `new_folds` and `moved_frac` are against the input. GUI: the simplex-3D menu gains `M10Tet → I-SQP windowed repair (fast certificate; needs osqp)` as its second row (`m10_windowed_3d`, knobs under Params → Strategy as `'m10_windowed_3d@tet3d'`); the windowed engine stays the pinned default; the osqp gate is now the tuple `_OSQP_GATED_ALGOS = ('isqp_windowed', 'm10_windowed_3d')`, so both rows disable without `osqp`.

- **Measured** — the twelve artefacts of the entry below, objective `l2`, threshold 0.01, one run each (records `h2h_<case>_m10_windowed_3d_l2.json`, table `h2h.md`). **Pre-registered before the runs:** the composite becomes the ≤ 5000-fold `auto` default only if ALL hold — (1) certifies all 12 (0 fixed-6-tet folds at threshold, 0 best-diagonal floor) with `new_folds == 0` against the input and the windowed stage's `damage == 0`; (2) wall below the windowed engine's on at least 10 of 12; (3) L2 move not more than 10 % above the windowed engine's on ANY artefact. *windowed* = `isqp_windowed` alone (the current `auto` pick), *m10_3d* = the wallbreaker alone, both from the records below; *iters* is the driver's `sqp_iters` (every phase's `n_iter`, `giant` entries excluded — the bulk stage's own phases included, which is the 45 on a row whose windowed stage opened no window).

  | artefact | folds in | windowed wall s | windowed L2 | m10_3d folds out | m10_3d wall s | m10_3d L2 | composite bulk folds after | composite iters | composite wall s | composite L2 | composite moved |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | B0032_moderate | 1217 | 1042 | 64.22 | 0 | 194.5 | 42.37 | 0 | 45 | 185.9 | 42.37 | 20.0 % |
  | B0049_moderate | 1217 | 934.7 | 43.97 | 0 | 170.5 | 27.54 | 0 | 45 | 165.1 | 27.54 | 17.8 % |
  | B0053_moderate | 1217 | 931 | 50.34 | 0 | 177.5 | 45.37 | 0 | 45 | 166 | 45.37 | 20.7 % |
  | B0200_moderate | 1216 | 1901 | 55.85 | 0 | 169.1 | 42.32 | 0 | 45 | 158.5 | 42.32 | 22.2 % |
  | B0213_moderate | 1217 | 1791 | 40.38 | 3 | 167 | 35.78 | 2 | 909 | 5830 | 38.28 | 75.2 % |
  | B0304_moderate | 1217 | 704 | 60.85 | 0 | 219.1 | 29.7 | 0 | 45 | 198.2 | 29.7 | 17.9 % |
  | cluster | 3038 | 6449 | 90.44 | 1 | 170.2 | 77.76 | 1 | 819 | 5670 | 84.05 | 84.3 % |
  | moderate | 1217 | 1020 | 58.73 | 0 | 186.9 | 59.46 | 0 | 45 | 176.3 | 59.46 | 26.1 % |
  | sliver | 1094 | 703.5 | 23.28 | 0 | 188.6 | 14.83 | 0 | 45 | 196.2 | 14.83 | 13.5 % |
  | sub20 | 686 | 611.1 | 19.12 | 0 | 126 | 20.88 | 0 | 45 | 127 | 20.88 | 18.2 % |
  | subvol16 | 721 | 660.7 | 55.81 | 3 | 68.89 | 44.67 | 2 | 2158 | 8283 | 114.1 | 100 % |
  | twist | 403 | 719 | 27.03 | 0 | 170 | 25.97 | 0 | 45 | 196.8 | 25.97 | 9.1 % |

  **The tally.** Clause 1 PASS: 12/12 certify (0 folds at threshold and at 0, 0 floor), `new_folds` 0 and `damage` 0 on every row. Clause 2 PASS at exactly 10/12: the losses are subvol16 (8283 vs 660.7 s) and B0213 (5830 vs 1791 s). Clause 3 **FAIL**: subvol16 L2 114.1 vs 55.81, **+104 %**; the next largest are sub20 +9.2 % and moderate +1.2 %, the other nine are below the windowed engine's (closer to the input on 9/12). The criterion is conjunctive, so **the ruling: `m10_windowed_3d` ships as an opt-in label; `auto` keeps `'isqp_windowed'` at ≤ 5000 folds; the GUI simplex-3D default is unchanged.** The mechanism, on the record: where `m10_3d` certifies alone (9 of 12) the composite is `m10_3d` plus a certificate check — the windowed stage opens no window (`rounds` 0) — at 3.6-12x less wall than the windowed engine (B0304 3.6, sliver 3.6, twist 3.7, sub20 4.8, B0032 5.6, B0053 5.6, B0049 5.7, moderate 5.8, B0200 12.0) and a smaller move on 7 of those 9 (moderate and sub20 are `m10_3d`'s own +1.2 % / +9.2 %); where `m10_3d` leaves a residual (subvol16 2 folds, cluster 1, B0213 2) that residual is a plateau the windowed ladder escalates all the way up on — 819-2158 SQP iterations, 75-100 % of the volume moved — so the composite's cost is bimodal: cluster −12 % wall / −7 % L2, B0213 3.3x the wall at −5 % L2, subvol16 12.5x the wall at +104 % L2. **Protocol amendment, labelled in the ledger:** the registered protocol set no per-step cap; after row 1 (subvol16, 8283 s, the ruling already decided by its L2) the remaining eleven rows ran under a per-row `timeout` of 4x the windowed-alone wall on that artefact, a row hitting it to be reported as `hit_cap` (no record) rather than dropped — no row hit it. Walls are single-run on a shared box (subvol16 with an unrelated ~1-core job at ~37 % load); the iteration counts are the load-independent column.

- **Changed alongside.** `dvfopt correct --pipeline` help: `solver` on a `(3, D, H, W)` volume with `--constraint simplex_3d` is the whole-volume Solver route (there `--strategy auto` picks the 3D default) and `3d` is the packaged `correct_dvf_3d` full-volume fold-repair pipeline; the solver-path `summary.json` gains `strategy_resolved` (`res.info.strategy_name`) beside the requested `strategy`, so an `auto` run records what it resolved to (pinned by `tests/test_cli.py`). `benchmarks/windowed_3d_vs_auto.py`: `METHODS` gains `m10_windowed_3d`; `sqp_iters` is now the sum of `n_iter` over the phases not named `giant…` — a `giant` history entry is nested inside its `round` entry, so `SolveInfo.total_iter` counted the tiles twice, the defect the entry below deferred (sliver 1398 = 2 × 699); the `isqp_windowed` rows of `h2h.md` were not re-run, so their `sqp_iters` remain the double-counted values and the table above omits them; new columns `bulk_folds_after` / `bulk_wall_s` (−1 for the other methods). `benchmarks/windowed_3d_sweep.py --set`: `orientation_delta`, `orientation_rows` and `threshold` are popped from the `--set` dict and passed explicitly (a `--set` on any of them used to raise a duplicate-keyword `TypeError`), so `--set orientation_delta=None` works.

### Changed — 3D windowed engine: auto routes `SimplexConstraint3D` to the windowed engine at ≤ 5000 folds (measured head-to-head); the GUI simplex-3D default

- **The rule.** `auto_strategy` on a `SimplexConstraint3D` with `init_n_neg <= 5000` and `osqp` importable now returns `'isqp_windowed'`, for every objective the engine accepts — so `strategy='auto'` (the `Solver` default path, `correct_dvf`, `dvfopt correct --strategy auto`, the GUI's Auto picker) gets the no-damage certificate instead of a method that leaves folds. Above 5000 folds, **or without `osqp`**, the pre-rule tiers are unchanged: extremes (`n_neg > 5000` or `init_min < -10`) → `m10_3d` for L2, `m14_schwarz_3d` on > 200K-voxel volumes, else `m14_3d`; everything else → `barrier`. The gate is keyed on fold COUNT, not depth — the engine's cost is per fold region, and `twist` (403 folds, minimum −13.4) clears in 468 SQP iterations. **The measurement regime, stated once:** the twelve artefacts top out at 24³ / 3038 folds; 5000 is the ruled extrapolation, and the gate carries no volume-size guard, so a full-resolution volume with ≤ 5000 folds now routes to an engine measured only at these sizes (the cost is per fold region, so it should scale; memory at full resolution is what the phase-4 chunked driver addresses).

- **The head-to-head** (`benchmarks/output/windowed_3d/h2h.md`, records `h2h_<case>_<method>_l2.json`): twelve 3D artefacts × five methods, objective `l2`, threshold 0.01, one run each. *Certifies* = 0 fixed-6-tet folds AND 0 best-diagonal floor. The windowed engine certifies **12/12**, every one at `damage == 0` and `new_folds == 0`; `m10_3d` 9/12 (fails the two dense regions — `subvol16` at 17.6 % of cubes below threshold, `cluster` at 25 % — AND one 10 % cohort crop, `B0213`, at 1-3 residual folds); `pipeline3d` 6/12 (three of them — `B0213`, `B0304`, `cluster` — only by moving 100 % of the voxels); `barrier` **0/12**, and `barrier` is what `auto` picked on 11 of the 12 (every artefact but `twist`, whose −13.4 minimum put it in the depth-extreme tier where `auto` picked `m10_3d`, which certifies it); `m14_3d` 0/12. `barrier` and `m14_3d` also CREATE folds on every artefact (`new_folds` 7-225 and 3-100).

  | case | method | auto's pick | folds in → out | floor out | new folds | damage | moved frac | wall s | L2 move | certifies |
  |---|---|---|---|---|---|---|---|---|---|---|
  | B0032_moderate | isqp_windowed | barrier | 1217 → 0 | 0 | 0 | 0 | 94.9 % | 1042 | 64.22 | ✓ |
  | B0032_moderate | barrier | barrier | 1217 → 285 | 240 | 63 | — | 18.4 % | 28.07 | 28.21 | ✗ |
  | B0032_moderate | m14_3d | barrier | 1217 → 150 | 123 | 28 | — | 60.6 % | 91.58 | 123.5 | ✗ |
  | B0032_moderate | m10_3d | barrier | 1217 → 0 | 0 | 0 | — | 20.0 % | 194.5 | 42.37 | ✓ |
  | B0032_moderate | pipeline3d | barrier | 1217 → 1 | 0 | 0 | — | 16.8 % | 290 | 42.49 | ✗ |
  | B0049_moderate | isqp_windowed | barrier | 1217 → 0 | 0 | 0 | 0 | 89.2 % | 934.7 | 43.97 | ✓ |
  | B0049_moderate | barrier | barrier | 1217 → 33 | 23 | 7 | — | 18.6 % | 28.29 | 28.64 | ✗ |
  | B0049_moderate | m14_3d | barrier | 1217 → 153 | 106 | 43 | — | 54.7 % | 93.05 | 77.5 | ✗ |
  | B0049_moderate | m10_3d | barrier | 1217 → 0 | 0 | 0 | — | 17.8 % | 170.5 | 27.54 | ✓ |
  | B0049_moderate | pipeline3d | barrier | 1217 → 0 | 0 | 0 | — | 17.5 % | 226.7 | 29.44 | ✓ |
  | B0053_moderate | isqp_windowed | barrier | 1217 → 0 | 0 | 0 | 0 | 82.9 % | 931 | 50.34 | ✓ |
  | B0053_moderate | barrier | barrier | 1217 → 226 | 201 | 31 | — | 19.4 % | 28.62 | 29.39 | ✗ |
  | B0053_moderate | m14_3d | barrier | 1217 → 78 | 51 | 11 | — | 52.2 % | 90.65 | 91.89 | ✗ |
  | B0053_moderate | m10_3d | barrier | 1217 → 0 | 0 | 0 | — | 20.7 % | 177.5 | 45.37 | ✓ |
  | B0053_moderate | pipeline3d | barrier | 1217 → 0 | 0 | 0 | — | 15.8 % | 458.7 | 48.11 | ✓ |
  | B0200_moderate | isqp_windowed | barrier | 1216 → 0 | 0 | 0 | 0 | 89.3 % | 1901 | 55.85 | ✓ |
  | B0200_moderate | barrier | barrier | 1216 → 198 | 167 | 26 | — | 20.9 % | 30.64 | 27.68 | ✗ |
  | B0200_moderate | m14_3d | barrier | 1216 → 148 | 99 | 39 | — | 54.3 % | 88 | 172.6 | ✗ |
  | B0200_moderate | m10_3d | barrier | 1216 → 0 | 0 | 0 | — | 22.2 % | 169.1 | 42.32 | ✓ |
  | B0200_moderate | pipeline3d | barrier | 1216 → 0 | 0 | 0 | — | 60.4 % | 238.4 | 41.48 | ✓ |
  | B0213_moderate | isqp_windowed | barrier | 1217 → 0 | 0 | 0 | 0 | 99.1 % | 1791 | 40.38 | ✓ |
  | B0213_moderate | barrier | barrier | 1217 → 86 | 73 | 14 | — | 18.2 % | 27.63 | 26.52 | ✗ |
  | B0213_moderate | m14_3d | barrier | 1217 → 249 | 154 | 47 | — | 53.1 % | 86.88 | 101.6 | ✗ |
  | B0213_moderate | m10_3d | barrier | 1217 → 3 | 3 | 0 | — | 19.0 % | 167 | 35.78 | ✗ |
  | B0213_moderate | pipeline3d | barrier | 1217 → 0 | 0 | 0 | — | 100 % | 633.9 | 60.55 | ✓ |
  | B0304_moderate | isqp_windowed | barrier | 1217 → 0 | 0 | 0 | 0 | 65.9 % | 704 | 60.85 | ✓ |
  | B0304_moderate | barrier | barrier | 1217 → 86 | 72 | 7 | — | 17.2 % | 31.25 | 21.31 | ✗ |
  | B0304_moderate | m14_3d | barrier | 1217 → 204 | 154 | 48 | — | 38.2 % | 104.1 | 153.9 | ✗ |
  | B0304_moderate | m10_3d | barrier | 1217 → 0 | 0 | 0 | — | 17.9 % | 219.1 | 29.7 | ✓ |
  | B0304_moderate | pipeline3d | barrier | 1217 → 0 | 0 | 0 | — | 100 % | 200.7 | 104.8 | ✓ |
  | cluster | isqp_windowed | barrier | 3038 → 0 | 0 | 0 | 0 | 1.00 | 6449 | 90.4 | ✓ |
  | cluster | barrier | barrier | 3038 → 1316 | 1076 | 225 | — | 40.7 % | 31.75 | 36.73 | ✗ |
  | cluster | m14_3d | barrier | 3038 → 436 | 304 | 100 | — | 81.9 % | 96.99 | 176.8 | ✗ |
  | cluster | m10_3d | barrier | 3038 → 1 | 1 | 0 | — | 46.2 % | 170.2 | 77.76 | ✗ |
  | cluster | pipeline3d | barrier | 3038 → 0 | 0 | 0 | — | 100 % | 1286 | 120.2 | ✓ |
  | moderate | isqp_windowed | barrier | 1217 → 0 | 0 | 0 | 0 | 96.5 % | 1020 | 58.73 | ✓ |
  | moderate | barrier | barrier | 1217 → 440 | 372 | 107 | — | 21.9 % | 31.5 | 28.46 | ✗ |
  | moderate | m14_3d | barrier | 1217 → 181 | 144 | 37 | — | 65.4 % | 92.72 | 129.5 | ✗ |
  | moderate | m10_3d | barrier | 1217 → 0 | 0 | 0 | — | 26.1 % | 186.9 | 59.46 | ✓ |
  | moderate | pipeline3d | barrier | 1217 → 2 | 2 | 0 | — | 22.2 % | 652.5 | 59.13 | ✗ |
  | sliver | isqp_windowed | barrier | 1094 → 0 | 0 | 0 | 0 | 91.3 % | 703.5 | 23.28 | ✓ |
  | sliver | barrier | barrier | 1094 → 29 | 17 | 9 | — | 13.9 % | 27.02 | 15.13 | ✗ |
  | sliver | m14_3d | barrier | 1094 → 24 | 18 | 3 | — | 22.5 % | 99.33 | 34.9 | ✗ |
  | sliver | m10_3d | barrier | 1094 → 0 | 0 | 0 | — | 13.5 % | 188.6 | 14.83 | ✓ |
  | sliver | pipeline3d | barrier | 1094 → 12 | 12 | 0 | — | 10.4 % | 282.6 | 16.27 | ✗ |
  | sub20 | isqp_windowed | barrier | 686 → 0 | 0 | 0 | 0 | 45.4 % | 611.1 | 19.12 | ✓ |
  | sub20 | barrier | barrier | 686 → 20 | 14 | 7 | — | 17.8 % | 18.57 | 19.42 | ✗ |
  | sub20 | m14_3d | barrier | 686 → 88 | 80 | 22 | — | 43.9 % | 62.55 | 62.75 | ✗ |
  | sub20 | m10_3d | barrier | 686 → 0 | 0 | 0 | — | 18.2 % | 126 | 20.88 | ✓ |
  | sub20 | pipeline3d | barrier | 686 → 5 | 4 | 0 | — | 14.7 % | 187.5 | 20.33 | ✗ |
  | subvol16 | isqp_windowed | barrier | 721 → 0 | 0 | 0 | 0 | 76.5 % | 660.7 | 55.81 | ✓ |
  | subvol16 | barrier | barrier | 721 → 210 | 171 | 37 | — | 25.8 % | 13.05 | 23.7 | ✗ |
  | subvol16 | m14_3d | barrier | 721 → 257 | 201 | 61 | — | 51.1 % | 35.4 | 179.5 | ✗ |
  | subvol16 | m10_3d | barrier | 721 → 3 | 3 | 0 | — | 26.4 % | 68.89 | 44.67 | ✗ |
  | subvol16 | pipeline3d | barrier | 721 → 2 | 1 | 1 | — | 49.6 % | 897.7 | 49.9 | ✗ |
  | twist | isqp_windowed | m10_3d | 403 → 0 | 0 | 0 | 0 | 80.1 % | 719 | 27.03 | ✓ |
  | twist | barrier | m10_3d | 403 → 124 | 101 | 27 | — | 7.8 % | 31.51 | 15.09 | ✗ |
  | twist | m14_3d | m10_3d | 403 → 122 | 86 | 20 | — | 35.2 % | 87.38 | 54.04 | ✗ |
  | twist | m10_3d | m10_3d | 403 → 0 | 0 | 0 | — | 9.1 % | 170 | 25.97 | ✓ |
  | twist | pipeline3d | m10_3d | 403 → 28 | 18 | 16 | — | 13.5 % | 626.6 | 30.17 | ✗ |

  `damage` (folds created OUTSIDE the engine's touched set, 0 by construction) is only defined for the windowed rows; `new_folds` (fold-free in, folded out) is the cross-method analogue and is carried for all. The `cluster` / `isqp_windowed` row is the driver's record like every other (2 rounds, moved fraction 1.00, 6449 s, L2 90.4; the same solve in the phase-3 crop pack: 94 windows, 2818 SQP iterations). **Wall:** the windowed engine costs 3.2-11x `m10_3d` — per case twist 4.2, sliver 3.7, moderate 5.5, sub20 4.8, subvol16 9.6, B0032 5.4, B0049 5.5, B0053 5.2, B0200 11.2, B0213 10.7, B0304 3.2, and 35-48x on `cluster` — and 23-65x `barrier` (187x on `cluster`), which never certifies. Walls are single-run on a shared box (the phase-3 contention caveat applies); the driver's `sqp_iters` column is omitted because it double-counts (below). **Move:** where `m10_3d` certifies, it is usually CLOSER to the input than the windowed engine — L2 ratios windowed / `m10_3d`: twist 1.04, sub20 0.91, moderate 0.99, sliver 1.57, B0032 1.51, B0049 1.60, B0053 1.11, B0200 1.32, B0304 2.05 — because the artefacts are small enough that the windowed engine's tiles cover 66-99 % of the voxels on the 24³ cuts (`moved_frac`; 45 % on the 20³ `sub20`, 76 % on the 17³ `subvol16`) where `m10_3d` moves 9-26 %. The pre-registered win criterion compared the windowed engine to `auto`'s CURRENT pick — `barrier`, which never certifies; on the single artefact where the current pick certifies (`twist`, `m10_3d`) the windowed L2 is +4 %, inside the 10 % bound — so the rule passes as pre-registered. The honest summary: **the windowed engine is the only certificate; `m10_3d` is the fast, usually-closer alternative that fails one artefact in four.**

- **The artefacts and the driver.** The B0039 pack (`twist`, `sliver`, `moderate`, `cluster`, all 24³, plus the 20³ `sub20`), the 17³ `subvol16`, and one 24³ cut of each of six cohort brains' exterior Laplacian field: `python benchmarks/windowed_3d_vs_auto.py --cut <brain>` scans stride 12 for the 24³ box whose share of cubes below threshold lands in 0.08-0.15 and is closest to 10 %, and writes `data/dvfs/crops_3d/<brain>_moderate.npy` — all six landed at exactly 10.0 %, at (z, y, x) offsets B0032 (360, 216, 144), B0049 (72, 96, 216), B0053 (96, 60, 240), B0200 (12, 108, 228), B0213 (48, 240, 204), B0304 (348, 120, 408). One run per call (`--case <case> --method <method>`, `--objective` default `l2`), a JSON record each, `--table` rebuilds `h2h.md`; `auto_pick` in every record is what `auto_strategy` resolves for that artefact today, so the table shows the previous default beside every method.

- **The explicit alternatives stay reachable by label.** `strategy='m10_3d'` is the fast route that certifies roughly 3 in 4 (above), and `strategy='pipeline3d'` / `correct_dvf_3d` is the packaged pipeline; neither is reachable from `auto` below 5000 folds any more. **Recommended follow-up, not built here:** a composite `m10_3d` → windowed residual repair — `m10_3d`'s failures are 1-3 residual folds, i.e. a few small windows, so the composite should certify at roughly `m10_3d`'s wall and move.

- **GUI.** The simplex-3D method menu gains `I-SQP windowed 3D (no-damage cluster windows; needs osqp)` as its first row and the family's pinned default (`DEFAULT_METHOD_BY_CONSTRAINT[CONSTRAINT_TET3D]`), with the windowed knobs under Params → Strategy (`'isqp_windowed@tet3d'`). Without `osqp` that row is disabled and the default falls back to `m14` (`DEFAULT_METHOD_FALLBACK`), so Run never lands on a dead selection.

- **Process.** The head-to-head ran from a detached snapshot worktree pinned at the benchmark commit while the branch moved on, so no library edit could perturb a run mid-chain. Deferred driver defect: `sqp_iters` is taken from `SolveInfo.total_iter`, which double-counts the round-loop entries (`sliver` 1398 = 2 × 699; `twist` 916 against the phase-3 record's 468) — the fix is to read the last round entry's `n_iter`; until then the docs take iterations from the phase-3 records and the column is omitted above.

### Added — 3D windowed engine, phase 3: `DEFAULTS_BY_DIM` (the per-dimension defaults table), the 3D cap, the QP settings; the cubic line model measured and refuted

- The defaults table. `DEFAULTS_BY_DIM` (`dvfopt/core/windowed/_common.py`) maps knob → {dim: default} and is resolved once at the engine entry by `resolve_dim_defaults(dim, **knobs)`: a knob a caller leaves at its 2D default takes the column for the field's dimension, any other explicit value is honoured in every dimension (`giant_tile=12` works on a 3D field; `mop_margin=0` still disables the mop). The sharp edge is that the rule is value equality, so a value that IS the 2D default cannot be requested on 3D — `giant_tile=64` on a 3D field reads as "the 3D default" (pass 65 for a 64-voxel tile), or pass `dim_defaults=False`, the escape that takes every knob literally with no per-dimension resolution (`windowed_correct(..., dim_defaults=False)`, the field `dim_defaults` on `ISQPWindowedStrategy` / `WindowedWrapperStrategy`, and `--set dim_defaults=False` in `benchmarks/windowed_3d_sweep.py`). The phase-2 twin knobs `giant_tile_3d` / `mop_margin_3d` are **gone** (unreleased, no alias): pass `giant_tile=` / `mop_margin=`, which now mean what they say on a 3D field.

  | knob | 2D | 3D | why |
  |---|---|---|---|
  | `giant_tile` | 64 | 16 | 16³ voxels ≈ the 2D 64² tile by count (phase 2) |
  | `mop_margin` | 25 | 6 | residual + 6/side → a 13-17³ mop window (phase 2) |
  | `max_window_area` | 3000 | 8000 | sub20 whole 19 iterations / L2 19.1 vs tiled 437 / 23.3 |
  | `reanchor_tile` | 48 | 16 | the giant tile, as in 2D |
  | `reanchor_overlap` | 8 | 8 | module constant `_REANCHOR_OVERLAP`, not a kwarg; ov4 buys -20.5 % L2 at half the wall vs ov8's -24 % — outside the 2 % rule |
  | `ip_cold` | True | True | measured and rejected as `False`: the ADMM-only start wins the 17³ whole window (107 → 26 iterations) but breaks the tiled crops (sliver 699 → 1154 iterations, 2 rounds, mop + re-seed; moderate 672 → 1449) |
  | `qp_max_iter` | 1000 | 1000 | measured and rejected at 2000: wins the 17³ whole window (107 → 22 iterations) but breaks the moderate crop (672 → 1592 iterations, 2 rounds, mop + re-seed) and taxes twist / sliver (+5 / +22 % iterations, +36 / +18 % wall); pass `qp_max_iter=2000` explicitly for whole-window sizes |
  | `qp_max_iter_fallback` | 500 | 500 | tracks `qp_max_iter` (half); 1000 measured and rejected alongside it |
  | `ip_after_admm_iters` | 800 | 800 | 400 / 200 measured, no gain on either case |

- **Wall-time caveat, once, for every number below.** A `degu_atlas.release` job from another project loaded this box to 50-94 % from 13:17 on 2026-09-08, and the same run measured 625 s idle against 1201-1320 s contended. Every wall in the QP table and the line-model gate is therefore contended; SQP-iteration counts and the per-QP ADMM statistics (`admm_med` = median ADMM iterations per QP solve, `admm_at_cap` = fraction of solves hitting the cap, `ip_solves` = Clarabel legs) are the robust columns, and both are carried. Every row in every table below is 0 folds at threshold, 0 folds at 0, 0 best-diagonal floor and damage 0 unless stated.

- **The 3D window cap is 8000 voxels.** The artefact is `sub20` — a 20³ = 8000-voxel cut of the raw B0039 field at offset (132, 132, 180), 10.0 % of its cubes below threshold, 686 folds in / 636 at the floor (`benchmarks/windowed_3d_sweep.py --cut-sub20`) — sized to sit between the 17³ one window solves whole and the 24³ crops the tiler must split. Caps 3000 and 5000 tile it identically (27 tiles, same iterations and same move); cap 8000 solves it whole in 19 iterations, 18 % closer in L2 and 34 % closer in L1, at 1.2-1.5x the wall (inside the 2.5x rule). The same shape holds on the 16³ sub-volume, where phase 2's tiling cost +74 % L2. The cost if the cap is wrong: an 8000-voxel window is ~26 s per SQP iteration (`qp_s_med` below), so a slow-converging region of that size is expensive where tiles would not be.

  | case | tag | n_windows | sqp_iters | qp_s_med | wall_s | L2 | L1 |
  |---|---|---|---|---|---|---|---|
  | sub20 (20³, 686 folds) | cap3000 | 27 | 437 | 0.27 | 435 | 23.3 | 635 |
  | sub20 | cap5000 | 27 | 437 | 0.32 | 544 | 23.3 | 635 |
  | sub20 | cap8000 (whole) | 1 | 19 | 25.9 | 630 | 19.1 | 417 |
  | subvol16 (17³, 721 folds) | cap5000 (whole) | 1 | 107 | 3.38 | 625 | 55.8 | 1150 |
  | subvol16 | phase-2 defaults (tiled) | 8 | 299 | — | 267 | 96.9 | — |
  | twist (24³, 403 folds) | cap8000 (tiled) | 29 | 468 | 0.49 | 599 | 27.0 | 649 |
  | twist | cap14000 (whole) | 3 | 38 | 7.00 | 1893 | 21.3 | 424 |

  The `cap14000` extension solves the whole 24³ twist crop as three windows with no tiler: -92 % iterations, -21 % L2 and -35 % L1 against the tiled run, but 3.2x the wall with 75 % of its QP solves at the ADMM cap. It is the **fidelity setting**, not the default (the 2.5x wall rule), and a phase-4 candidate together with the QP backend — at ~14k voxels the run is QP-bound, so a faster QP at that size would make the whole-window solve win outright.

- **The QP settings on 3D: `qp_max_iter` / `qp_max_iter_fallback` stay at their 2D values (1000 / 500), `ip_cold` unchanged at True.** (`qp_max_iter=2000` / fallback 1000 is what the first pass below picked; the crop-pack gate's second verdict later rejected it too — see "The gate's second verdict" after the final-gate table.) The 17³ / twist tuning below picked `ip_cold=False` alongside the caps; the crop-pack gate then **rejected it** (paragraph after the tables). On the whole 17³ window the cold Clarabel solve's first step steers the SQP into a 107-iteration basin; an ADMM-only start at a higher accuracy clears it in 23 at the same move (L2 55.7 vs 55.8). On the tiled twist crop — small tiles, where the QP is cheap — the setting is neutral (468 → 479 iterations), and the combination removes the +20 % iteration penalty `ip_cold=False` alone carries there (560). Rejected: `qp_max_iter=500` (on the 17³ the 500-iteration ADMM solves derail the SQP — the window fails, the ladder grows and tiles it into 9 windows, L2 +33 %), plain OSQP with no IP legs at all (12 windows, L2 +42 % — the IP legs are load-bearing in 3D), and `ip_after_admm_iters` 400 / 200 (no gain on either case).

  | case | tag | sqp_iters | qp_n | admm_med | admm_at_cap | ip_solves | wall_s (contended) | L2 |
  |---|---|---|---|---|---|---|---|---|
  | subvol16 | qp_base | 107 | 107 | 250 | 0.14 | 16 | 1320 | 55.8 |
  | subvol16 | qp_cap2000 | 22 | 22 | 1575 | 0.45 | 11 | 738 | 55.9 |
  | subvol16 | qp_nocold | 26 | 26 | 1000 | 0.71 | 12 | 350 | 55.8 |
  | subvol16 | **qp_nocold_cap2000** | 23 | 23 | 1350 | 0.33 | 11 | 336 | 55.7 |
  | subvol16 | qp_ip400 | 96 | 96 | 350 | 0.20 | 32 | 1473 | 55.8 |
  | subvol16 | qp_ip200 | 104 | 104 | 500 | 0.21 | 48 | 1360 | 55.8 |
  | subvol16 | qp_cap500 (9 windows) | 660 | 660 | 250 | 0.26 | 17 | 3548 | 74.4 |
  | subvol16 | qp_osqp (12 windows) | 519 | 519 | 100 | 0.08 | 0 | 1858 | 79.1 |
  | twist | qp_base | 468 | 468 | 212.5 | 0.14 | 100 | 651 | 27.0 |
  | twist | qp_cap2000 | 492 | 492 | 150 | 0.05 | 107 | 656 | 26.6 |
  | twist | qp_nocold | 560 | 560 | 125 | 0.11 | 73 | 522 | 27.0 |
  | twist | **qp_nocold_cap2000** | 479 | 479 | 150 | 0.03 | 76 | 572 | 26.6 |
  | twist | qp_ip400 | 435 | 435 | 200 | 0.12 | 137 | 678 | 26.6 |
  | twist | qp_ip200 | 487 | 487 | 75 | 0.11 | 162 | 761 | 27.0 |
  | twist | qp_cap500 | 458 | 458 | 275 | 0.38 | 39 | 342 | 26.5 |
  | twist | qp_osqp | 603 | 603 | 300 | 0.14 | 0 | 401 | 26.3 |

  `qp_n` equals `sqp_iters` on every row (one QP per SQP iteration). Idle re-runs of the two baselines and the finalist (load ~20 %, 22:22-22:55; the subvol16 finalist row is its 22:07 run, near-idle at 336 s):

  | case | tag | sqp_iters | qp_n | admm_med | admm_at_cap | ip_solves | wall s (idle) | L2 |
  |---|---|---|---|---|---|---|---|---|
  | subvol16 | qp_base_idle | 107 | 107 | 250 | 0.14 | 16 | 660 | 55.8 |
  | twist | qp_base_idle | 468 | 468 | 212 | 0.14 | 100 | 658 | 27.0 |
  | twist | qp_nocold_cap2000_idle | 479 | 479 | 150 | 0.03 | 76 | 592 | 26.6 |
  | subvol16 | qp_nocold_cap2000 | 23 | 23 | 1350 | 0.33 | 11 | 336 | 55.7 |

  Idle, the 17^3 baseline is 660 s (107 it) against the finalist's 336 s (23 it): -49 % wall at the same L2; on twist the finalist is neutral idle too (658 -> 592 s, 468 -> 479 it).

  **The gate's verdict: `ip_cold` stays True, the cap 2000 stays.** Under the pair, the crop pack regressed on the tiled cases the 17³/twist tuning never covered — sliver 1 round / 27 windows / 699 iterations / L2 23.3 → 2 rounds / 61 windows / a mop window / a re-seed round / 1314 iterations / L2 25.2, and moderate 672 → 1449 iterations at L2 58.7 → 61.8 (twist neutral). Isolation on sliver (explicit knobs, `'tr'`): cap 8000 with the phase-2 QP settings reproduces phase 2 exactly (699 / 27 / 23.3); `ip_cold=False` alone reproduces the regression (1154 / 61 / 25.0); the cap alone keeps the phase-2 shape — 1 round / 27 windows / 855 iterations (+22 %) / L2 23.4 / 955 s vs 1199 s for the base under the same load. So `ip_cold=False` is the culprit: it wins the 17³ whole window but the ADMM-only first step goes wrong on the 12³ tiles and the round loop escalates. The `ip_cold` row stays in `DEFAULTS_BY_DIM` at `{2: True, 3: True}` to document the measurement.

- **Re-anchor overlap on 3D stays 8.** Twist with `reanchor='l2'`, one sweep: overlap 8 opens 27 tiles (25 accepted) and takes the L2 move 27.03 → 20.57 (-24 %) in 5034 s; overlap 4 opens 8 tiles (8 accepted) → 21.48 (-20.5 %) in 2517 s. 4 lands 4.4 % above 8's result, outside the 2 % rule, so `_REANCHOR_OVERLAP` keeps its 8 in both dimensions and 4 is documented as the half-cost alternative (the stage is opt-in either way).

- **The cubic exact line search: built, measured, REFUTED** (commit 9c85b00, reverted by 36db826; `'exact_ls'` keeps degrading to `'tr'` on 3D, so the patience rung (`step_rule == 'exact_ls'`-gated) stays off in 3D). `'exact_ls'` is exact in 2D because every row family there is bilinear in the displacements, hence quadratic along a step; a 6-tet volume row is trilinear, hence **cubic** along a step. The model pinned that cubic from what the ratio test already evaluates: `c0` and `J d` are exact, and two constraint samples — at the half step and the full step — give the remaining coefficients in closed form (`q2 = 8 r_h - r_1`, `q3 = 2 r_1 - 8 r_h`, with `r_h` / `r_1` the linear model's residuals at those two points). The merit along the step is then piecewise cubic, minimised on `[0, 1]` by a 64-point grid plus golden-section refinement, with the TRUE merit checked before stepping (so a window can never regress). It wins the sparse crops and loses the dense ones, at every setting of the a*-collapse bail:

  | case | tag | rounds | n_windows | mop | reseed | sqp_iters | rejected | wall_s | L2 |
  |---|---|---|---|---|---|---|---|---|---|
  | twist | ls_tr | 1 | 29 | 0 | 0 | 468 | 131 | 599 | 27.0 |
  | twist | ls_cubic | 1 | 29 | 0 | 0 | 246 | 27 | 469 | 26.4 |
  | sliver | ls_tr | 1 | 27 | 0 | 0 | 699 | 189 | 623 | 23.3 |
  | sliver | ls_cubic | 1 | 27 | 0 | 0 | 266 | 12 | 444 | 23.7 |
  | moderate | ls_tr | 1 | 27 | 0 | 0 | 672 | 150 | 896 | 58.7 |
  | moderate | ls_cubic | 2 | 61 | 1 | 1 | 612 | 62 | 2272 | 61.3 |
  | moderate | ls_cubic_fb0 | 2 | 61 | 1 | 1 | 1509 | 123 | 2359 | 61.4 |
  | moderate | ls_cubic_fb6 | 2 | 61 | 1 | 1 | 755 | 91 | 2004 | 61.4 |
  | subvol16 | ls_tr | 1 | 1 | 0 | 0 | 107 | 11 | 1201 | 55.8 |
  | subvol16 | ls_cubic | 2 | 12 | 2 | 1 | 527 | 18 | 4389 | 80.6 |
  | subvol16 | ls_cubic_fb0 | 2 | 12 | 2 | 1 | 1294 | 50 | 3531 | 80.6 |
  | subvol16 | ls_cubic_fb6 | 2 | 12 | 2 | 1 | 673 | 33 | 2936 | 80.6 |

  Sparse: twist 468 → 246 iterations at -22 % wall and a smaller move (27.0 → 26.4); sliver 699 → 266 at -29 % wall, move +1.7 % (23.3 → 23.7). Dense: moderate goes from one round / 27 windows to two rounds / 61 windows plus a mop and a re-seed round, 2.2-2.6x the wall at bail 3 / 0 / 6; and the 17³ whole window **fails** under the cubic model at every bail setting, so the ladder tiles it into 12 windows for L2 +44 %. The bail (`exact_ls_fallback_steps`, a 2D-measured setting) is not the cause — with it off the windows still fail (20 no-TR fallbacks on moderate) — the cubic model simply converges worse than the ratio test on a dense fold region. Gate: iterations 3/4, wall 2/4, L2 within +10 % 3/4 → refuted for the 3D default. The code is in git history (9c85b00) if a dense-region fix appears; ~25 % wall is left on the table for sparse 3D regions.

- **The final gate**, the crop pack and the 16³ sub-volume under the phase-3 table, against the phase-2 rows (twist 468 iterations / 576 s / L2 27.0; sliver 699 / 633 / 23.3; moderate 672 / 878 / 58.7; cluster 2818 / 8250 / 90.4; 16³ tiled 299 / 267 / 96.9 vs whole 107 / 638 / 55.8):

    | case | folds in → out | floor out | damage | rounds | windows | mop | re-seed | SQP iters | L2 move | phase 2 (iters / L2) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | twist | 403 → 0 | 0 | 0 | 1 | 29 | 0 | 0 | 468 | 27.0 | 468 / 27.0 |
  | sliver | 1094 → 0 | 0 | 0 | 1 | 27 | 0 | 0 | 699 | 23.3 | 699 / 23.3 |
  | moderate | 1217 → 0 | 0 | 0 | 1 | 27 | 0 | 0 | 672 | 58.7 | 672 / 58.7 |
  | cluster | 3038 → 0 | 0 | 0 | 2 | 94 | 2 | 1 | 2818 | 90.4 | 2818 / 90.4 |
  | subvol16 (17³) | 721 → 0 | 0 | 0 | 1 | 1 | 0 | 0 | 107 | 55.8 | tiled 299 / 96.9 |

  Every case reaches 0 fixed-6-tet folds at threshold AND at 0, 0 best-diagonal floor, damage 0. The four crops reproduce the phase-2 result to the SQP iteration and the move (the defaults table changes no crop behaviour, and the deterministic engine repeats every count); the one phase-3 gain is the 17³ B0039 sub-volume, which the raised `max_window_area` (8000) now solves as ONE window in 107 SQP iterations at L2 move 55.8, against phase 2's tiled 299 iterations / L2 96.9 — 64 % fewer iterations, 43 % smaller move. Walls are load-contended (the QP tables above carry the idle baselines); SQP iterations and L2 move are the deterministic columns.

- **The gate's second verdict: `qp_max_iter` reverts to 1000 / 500.** The sliver isolation above cleared `ip_cold`, but under `qp_max_iter=2000` / fallback 1000 (cap 8000, `ip_cold=True`, `'tr'`) the full crop pack still regresses: twist 492 SQP iterations / 784 s / L2 26.6 (phase 2: 468 / 576 / 27.0); sliver 855 / 748 / 23.4 (699 / 633 / 23.3); moderate 2 rounds / 61 windows / a mop window / a re-seed round / 1592 iterations / 2724 s / L2 61.4 (phase 2: 1 round / 27 / 672 / 878 / 58.7). The `ls_tr` row on moderate — cap 8000 with the phase-2 QP settings — is 672 / 896 / 58.7, reproducing phase 2 exactly, so the cap-8000/mop coupling is not the cause; the ADMM cap is. `qp_max_iter` / `qp_max_iter_fallback` revert to their 2D values (1000 / 500) in `DEFAULTS_BY_DIM`; cap 2000 wins only the 17³ whole window (107 → 22 iterations, 738 s contended) and is now the explicit opt-in for whole-window sizes (`qp_max_iter=2000` on a 3D field is honoured — 2000 is not the 2D default).

- Process notes. Every measurement chain runs from a **detached snapshot worktree** at the measured commit, never a live one: the first cap chain failed with a `TypeError` in `_InnerOpts` because it imported a worktree an implementer was editing mid-run. The driver is `benchmarks/windowed_3d_sweep.py` — one `windowed_correct` run per call on a phase-2 artefact under explicit overrides, with the QP / exit spies of `windowed_3d_gate.py`: `--case {subvol16, sub20, twist, cluster, sliver, moderate}`, `--tag`, `--set k=v` (any engine kwarg; `reanchor_overlap=N` patches the module constant), `--cut-sub20`, `--table`.

### Added — 3D windowed engine, phase 2: every stage runs on 3D fields

- The giant-region Schwarz tiler (serial sweeps and `giant_workers` RAS), the coarse-grid warm start, the terminal mop, the harmonic re-seed, the re-anchor stage and the per-window polish are dimension-agnostic (`itertools.product` over per-axis ranges through the phase-1 box helpers; byte-identical in 2D — `benchmarks/windowed_2d_identity.py`: 21 cases, `IDENTITY PASS`). Two 3D defaults, sized from phase 1's cost curve: `giant_tile_3d=16` per axis (16³ voxels is the 2D 64² tile by count) and `mop_margin_3d=6`; the 3D re-anchor tile is `giant_tile_3d`. The phase-1 gates (advisory cap, skipped stages, refused `reanchor` / `polish`) are gone.
- Found by re-measuring the phase-1 16³ sub-volume under the phase-2 defaults: the tiler's trailing start on an image border. On the 17³ volume the fitted tile is 12 with step 8, so the starts are 0, 8, 16 and the 16..17 strip's ring-padded patch clips to width 2 at the border, which `validate_dvf` refuses (`SolverConfigError`). `_giant_tiles` now drops a trailing start whose padded patch is thinner than 3 on that axis — the strip is already inside the previous tile (`tile - step >= 2`) — and `_ras_cores` gives the last start on each axis the core up to the inset edge, so the RAS sweep still pastes the strip. Latent in 2D too (a 1-px remainder strip exactly on an image border) but never hit; byte-identical wherever the old code did not raise (the identity A/B stays `IDENTITY PASS`).
- Measured (`benchmarks/make_hard_crops_3d.py`, four 24³ crops of the raw B0039 field, threshold 0.01, `'tr'`):

  twist, sliver and moderate clear in one round through the tiler (no mop / re-seed); `cluster` (3038 folds, 25% of its cubes) takes 2 rounds, 94 windows, a 2-window mop and one re-seed round (8250 s, L2 90.4) — the mop and re-seed fire on real data under the default config there. The rows-off arm on moderate (`l2_norows`) reaches 0 folds too but at 2 rounds / 67 windows / mop 3 / re-seed 1 / 2512 SQP iterations / 3640 s vs 672 / 878 s with the rows: the edge rows are worth 4x in wall on 3D real data, as in phase 1. `none_rows` on twist: 0 folds, 777 iterations / 942 s / L2 62.8 vs the L2 default's 468 / 576 s / 27.0 — in 3D the in-solve L2 is both faster and closer (unlike 2D, where `none` is the fastest formulation). The `giant_tile_3d` A/B on moderate: 12 and 16 both fit to a 12-voxel tile on the 24³ crop's region (`_fit_tile_nd`), hence identical runs; 20 fits to 15 — fewer SQP iterations (454 vs 672) and a smaller move (54.3 vs 58.7) but 1.7x the wall (1520 vs 878 s: the per-iteration QP cost grows faster than the iteration count falls), so 16 stays the default. The coarse warm start never fires on a 24³ crop (`min(shape) < 4 * giant_tile_3d`) — its 3D value is unmeasured (phase 3). The per-window polish (`dvfopt/core/windowed/_common.py`) is gated by `allow_grow`, which defaults `True` for round-loop windows (`_solve_window` at line 1055) and mop windows (line 1272) but is forced `False` for every giant-tile Schwarz sub-window (`_solve_giant_schwarz` calls `_solve_window(..., allow_grow=False)` at lines 1668/1721): `if ok and opts.polish and allow_grow:` (line 1900) means the polish never fires on a giant tile, only on windows the round loop (or the mop) solves directly.

  | case | cfg | giant_tile_3d | mop_margin_3d | folds_in | floor_in | folds_out | floor_out | min_out | damage | rounds | n_windows | giant_regions | mop_windows | reseed_rounds_run | sqp_iters | wall_s | l2_move |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | sliver | l2_rows | 16 | 6 | 1094 | 1071 | 0 | 0 | 0.0110 | 0 | 1 | 27 | 1 | 0 | 0 | 699 | 633 | 23.3 |
  | twist | l2_rows | 16 | 6 | 403 | 352 | 0 | 0 | 0.0109 | 0 | 1 | 29 | 1 | 0 | 0 | 468 | 576 | 27.0 |
  | cluster | l2_rows | 16 | 6 | 3038 | 2705 | 0 | 0 | 0.0110 | 0 | 2 | 94 | 1 | 2 | 1 | 2818 | 8250 | 90.4 |
  | moderate | l2_rows | 16 | 6 | 1217 | 1045 | 0 | 0 | 0.0109 | 0 | 1 | 27 | 1 | 0 | 0 | 672 | 878 | 58.7 |
  | twist | none_rows | 16 | 6 | 403 | 352 | 0 | 0 | 0.0110 | 0 | 1 | 29 | 1 | 0 | 0 | 777 | 942 | 62.8 |
  | moderate | l2_norows | 16 | 6 | 1217 | 1045 | 0 | 0 | 0.0110 | 0 | 2 | 67 | 1 | 3 | 1 | 2512 | 3640 | 64.3 |
  | moderate | l2_rows | 12 | 6 | 1217 | 1045 | 0 | 0 | 0.0109 | 0 | 1 | 27 | 1 | 0 | 0 | 672 | 913 | 58.7 |
  | moderate | l2_rows | 20 | 6 | 1217 | 1045 | 0 | 0 | 0.0108 | 0 | 1 | 27 | 1 | 0 | 0 | 454 | 1520 | 54.3 |

  `folds_out_zero` / `floor_out_zero` are 0 on every row above. The phase-1 16³ B0039 sub-volume under the phase-2 defaults: 0 folds / 0 floor / damage 0 through the tiler (fitted tile 12, 8 tiles, 1 round) in 299 SQP iterations / 267 s, L2 move 96.9 — 2.4x faster than phase 1's single 17³ window (107 iterations / 638 s) but a 74 % larger move: the eight tiles' seams cost fidelity a whole-region solve does not pay. `max_window_area` (3000 voxels ≈ 14.4³) is the 2D cap read as a voxel count; phase 1's cost curve puts a single 17³ window at 11 s per SQP iteration, so a 3D cap near 5000 (≈ 17³) is the phase-3 candidate (the twin-knob defaults table).

### Added — 3D windowed engine, phase 1: family plumbing + certificate (`SimplexConstraint3D` in `ISQPWindowedStrategy`)

- `windowed_correct` / `ISQPWindowedStrategy` accept `SimplexConstraint3D` on
  `(3, D, H, W)` fields: `LOCALITY[SimplexConstraint3D]` (ring 1, `six_tet_min_volume_3d`
  fold map, eight-corner influenced rule, shape-cached native tet Jacobian), n-D boxes
  through three helpers (`_box_slices` / `_box_size` / `_pad_box`, byte-identical in 2D —
  verified main-vs-branch by `benchmarks/windowed_2d_identity.py`: 21 cases, every field
  `array_equal`, every report identical), 3D axial edge rows (every edge with at least one
  free endpoint, all three axes, DX_FIRST; the 3D injectivity helper's both-endpoints-free
  filter is the wrong predicate for a frozen-ring window), the `'tr'` step rule
  (`'exact_ls'` degrades on 3D), and the certificate fields `folds_after_zero` /
  `best_diag_floor_after` / `best_diag_floor_after_zero`. Not yet ported (phase 2): the
  coarse warm start, mop and re-seed are skipped on 3D, the giant cap is advisory
  (over-cap regions are solved whole), `reanchor` / `polish` raise.
- Measured (`benchmarks/windowed_3d_gate.py`, `'tr'`, threshold 0.01):

  | case | cfg | folds_in | folds_out | folds_out_zero | floor_in | floor_out | floor_out_zero | min_out | damage | n_windows | sqp_iters | exits | wall_s | l2_move | max_abs_dz |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | slice090 | l2_norows | 61 | 0 | 0 | 53 | 0 | 0 | 0.0110 | 0 | 1 | 16 | 'ftol': 1 | 5.91 | 6.66 | 0.358 |
  | slice090 | l2_rows | 61 | 0 | 0 | 53 | 0 | 0 | 0.0110 | 0 | 1 | 18 | 'ftol': 1 | 7.84 | 7.28 | 0.358 |
  | slice090 | none_norows | 61 | 0 | 0 | 53 | 0 | 0 | 0.0110 | 0 | 1 | 7 | 'step-tol': 1 | 2.41 | 8.93 | 0.822 |
  | slice090 | none_rows | 61 | 0 | 0 | 53 | 0 | 0 | 0.0110 | 0 | 1 | 7 | 'ftol': 1 | 3.04 | 9.31 | 0.726 |
  | slice200 | l2_norows | 30 | 0 | 0 | 18 | 0 | 0 | 0.0110 | 0 | 1 | 32 | 'ftol': 1 | 8.01 | 4.88 | 0.744 |
  | slice200 | l2_rows | 30 | 0 | 0 | 18 | 0 | 0 | 0.0110 | 0 | 1 | 13 | 'ftol': 1 | 3.83 | 6.97 | 0.625 |
  | slice200 | none_norows | 30 | 0 | 0 | 18 | 0 | 0 | 0.0112 | 0 | 1 | 8 | 'step-tol': 1 | 2.61 | 7.00 | 0.998 |
  | slice200 | none_rows | 30 | 0 | 0 | 18 | 0 | 0 | 0.0110 | 0 | 1 | 6 | 'ftol': 1 | 1.99 | 7.72 | 0.773 |
  | slice350 | l2_norows | 70 | 0 | 0 | 57 | 0 | 0 | 0.0110 | 0 | 1 | 232 | 'tr-collapse': 1, 'maxiter': 1 | 12.8 | 9.71 | 0.692 |
  | slice350 | l2_rows | 70 | 0 | 0 | 57 | 0 | 0 | 0.0109 | 0 | 1 | 17 | 'ftol': 1 | 7.73 | 11.5 | 0.547 |
  | slice350 | none_norows | 70 | 0 | 0 | 57 | 0 | 0 | 0.0111 | 0 | 1 | 19 | 'model-flat': 1 | 5.37 | 17.6 | 1.34 |
  | slice350 | none_rows | 70 | 0 | 0 | 57 | 0 | 0 | 0.0110 | 0 | 1 | 8 | 'step-tol': 1 | 3.00 | 14.6 | 0.851 |
  | subvol16 | l2_norows | 721 | 11 | 11 | 702 | 11 | 11 | -0.00542 | 0 | 8 | 1108 | 'tr-collapse': 16, 'linesearch-stall': 16 | 3990 | 33.3 | 2.38 |
  | subvol16 | l2_rows | 721 | 0 | 0 | 702 | 0 | 0 | 0.0110 | 0 | 1 | 107 | 'tr-collapse': 1, 'linesearch-stall': 1 | 638 | 55.8 | 0.722 |
  | subvol16 | none_norows | 721 | 14 | 14 | 702 | 14 | 13 | -0.00168 | 0 | 5 | 677 | 'tr-collapse': 7, 'maxiter': 2, 'linesearch-stall': 8, 'model-flat': 3 | 2440 | 61.5 | 5.21 |
  | subvol16 | none_rows | 721 | 0 | 0 | 702 | 0 | 0 | 0.0110 | 0 | 1 | 14 | 'step-tol': 1 | 159 | 86.3 | 1.91 |

- Per-SQP-iteration cost vs window volume (one frozen-ring window, 8 SQP iterations,
  hybrid backend):

  | L | n_free | n_rows | qp_vars | nnz_jac | jac_s | cons_s | sqp_iters | s_per_sqp_iter | exit | qp_s_median | admm_median |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 9 | 1029 | 4248 | 5277 | 39216 | 0.00109 | 0.000218 | 8 | 0.289 | maxiter | 0.0944 | 338 |
  | 17 | 10125 | 35376 | 45501 | 316512 | 0.00686 | 0.000683 | 8 | 11.1 | maxiter | 9.45 | 510 |
  | 25 | 36501 | 121032 | 157533 | 1071504 | 0.0225 | 0.00211 | 8 | 68.6 | maxiter | 46.8 | 488 |
  | 33 | 89373 | 288864 | 378237 | 2543808 | 0.0608 | 0.00821 | 8 | 411 | maxiter | 233 | 513 |

  > **The two unknowns.** U1 (cost): per SQP iteration 0.29 / 11.1 / 68.6 / 411 s at 9³ / 17³ / 25³ / 33³ — 5.3k / 45k / 158k / 378k QP variables (one slack per row); at 33³ every warm OSQP solve hits its 1000-iteration cap and Clarabel takes 195-240 s per cold solve. The QP solve is the whole cost (the Jacobian build is ≤ 0.4 % of an iteration), so phase 2's tiler must keep windows near 17³ and phase 3 starts at the QP backend, not the line search. U2 (convergence): the 16³ B0039 sub-volume (721 folds at threshold, 702 negative under every diagonal on input, where `correct_dvf_3d` had left one residual) reaches 0 fixed-6-tet folds and 0 best-diagonal floor at damage 0 under the default config — in-solve L2 + edge rows — in 107 SQP iterations / 638 s (L2 move 55.8): the ratio test alone stalls once (`tr-collapse`) and the no-trust-region line-search rung finishes it; with `objective='none'` + rows it converges directly in 14 iterations / 159 s; without the rows every rung of the ladder plateaus at 11-14 residual folds (1108 / 677 SQP iterations, 3991 / 2437 s), so the edge rows are load-bearing in 3D exactly as in 2D. `exits` counts inner calls (ladder rungs), not windows.

### Measured — 2.5D `orientation_delta=0.01` on the full 528-slice B0039 volume (sweep stage)

- base → rows: folds 66 → 50, true fold floor (negative under every main
  diagonal) 23 → 12, min volume −0.281 → −0.187, L1 move −6.7 %; but
  sub-threshold cubes 2137 → 4384 (cells parked at the 0.01 row margin), which
  is the mop's workload. Stays opt-in; details in findings §10.

### Fixed — 2.5D mop: parallel batches by dependency level (was ~1 of 4 cores)

- The parallel mop grouped only CONSECUTIVE pairwise-disjoint boxes; because
  `find_objects` order is spatial, neighbours touch and nearly every batch
  held one box — measured on the full B0039 volume as 3.2 h of worker CPU
  over a 3.5 h pass (2.3 h of it on one worker), i.e. ~0.9 of 4 cores.
  Boxes are now scheduled by dependency level (`_levels`: one above every
  earlier overlapping box), which is still byte-identical to the serial loop
  and lets every spatially separated box run alongside the rest.

### Fixed — 2.5D mop: per-box futility stop (`stall_iters=4`, `stall_rtol=1e-2`)

- Tiling giant boxes (below) did not make the full-volume mop tractable: a
  pass still took ~7 h and one worker 5.4 h. Measured directly on the densest
  edge-layer region of the base sweep field, even a 3.7k-free-voxel box needs
  > 11 min for four SLP solves — on a near-floor region nearly every tet row
  is active, so the LP is ~50k rows at minutes per solve regardless of box
  size, and `elastic_trust_solve`'s accept-micro-step (trust doubles back) /
  reject (trust halves) alternation never reaches the trust floor, burning
  all 40 solves. The engine gains an opt-in futility stop (end when the exact
  violation has not dropped `stall_rtol` over the last `stall_iters` solves —
  the 2D engine's a*-collapse bail, measured load-bearing there); the mop
  passes 4 / 1 %, the sweep is untouched (`stall_iters=0`, byte-identical).

### Fixed — 2.5D mop: giant residual boxes are tiled (`max_box=90`)

- `mop_interior_3d` cropped each residual cluster into ONE box with no size
  cap. On the 528-slice B0039 volume a plane-spanning cluster became a box
  that a single worker solved whole for **5.6 h** (up to 40 SLP iterations
  of a huge HiGHS LP) while the other three workers idled — the parallel mop
  could not split it, and the serial mop was stuck on the same box. Boxes
  wider than `max_box` on y/x are now tiled at stride `max_box` (the sweep's
  `_cluster_boxes` idiom, phase-shifted by `max_box // 2` on even passes so a
  seam-locked residual is tile-interior next pass); the tiles of one giant box
  are pairwise disjoint, so they run in a single parallel batch. Boxes within
  `max_box` are untouched (byte-identical), `max_box=None` disables the cap.

### Added — 2.5D marching: parallel mop (`n_workers`) and segment-parallel sweep (`n_segments`)

- Motivation (full-res 528-slice B0039 volume, 4 workers): the sweep took
  16.5 h — interior layers ~14 s each but the 13 edge layers ~1 h each; the
  cluster-pool workers were busy only ~27 % of the wall (one big cluster per
  layer dominates) and the parent spent ~40 % in serial work. The mop ran
  serially over ~260 boxes per pass on one core — hours per pass.
- `mop_interior_3d(n_workers=, pool_map=)`: with `n_workers > 1` each pass's
  boxes are repaired on the shared spawn pool in batches of pairwise-disjoint
  (padded) boxes. Boxes are walked in the serial order and a batch is closed
  as soon as the next box overlaps one already in it, so every box is cropped
  from — and pasted back into, under the same `v_after < v_before` rule — the
  exact state the serial loop would have used: **byte-identical to
  `n_workers=1`** (tested, in-process seam and real pool). `correct_dvf_25d`
  threads its `n_workers` into the mop.
- `correct_dvf_25d(n_segments=)`: `> 1` splits z into that many contiguous
  near-equal segments, each swept from its own origin (the mildest inter-layer
  inside the segment) as one process on the shared pool (clusters run serially
  inside a segment — the pool refuses nesting — so `n_workers` counts both the
  segment workers and the cluster/mop workers; segments queue on it). Workers
  receive and return only their `[dy, dx]` slab. The `n_segments - 1` seams
  are then repaired in the parent, one down-sweep `march_slice` each (lower
  segment's top slice against the frozen upper segment's bottom slice), before
  the usual stats / mop / report. New `report.stages` entries `segments` and
  `seams` (replacing `sweep`). Checkpoints keep working per slice: a segment's
  slices are marked when its slab returns, seams as `seam:<z>`; a fully-marked
  volume skips the segment stage on resume. `n_segments=1` (default) is the
  single-origin sweep, byte-identical to before (tested); a pooled
  `n_segments=3, n_workers=3` run equals the serial `n_segments=3` run byte
  for byte (segments are deterministic given their slab). Needs
  `D >= 2 * n_segments`.

### Fixed — 2.5D mop: repair predicate = the report predicate

- `mop_interior_3d` clustered its boxes on its own LP target (`min_vol <
  thr3 - 1e-9`, thr3 = threshold + 1e-4) while the pipeline's `mop_max_folds`
  gate and `feasible` count at the report predicate (`< threshold - 1e-5`).
  On the 528-slice B0039 volume the sweep parks ~127k cubes AT thr3 within LP
  tolerance, so one mop pass ran its serial box LPs over ~700 clusters (>24 h,
  never finishing) instead of the ~260 clusters / ~2k cubes that are actually
  below the report threshold. The mop now clusters on the report predicate;
  thr3 remains the per-box LP target. `info['n_below_*']` count under that
  predicate (`n_below_report_after` is kept as an alias).

### Changed — `data/dvfs/` is the centralized DVF suite

- Every deformation field the repo consumes or produces now lives under one
  root: `origins/` (the `dvf_origins` per-mechanism benchmark fields, moved from
  `data/origins`), `cohort/` (the 7-brain RegTools cohort, renamed from
  `brain25_cohort_corrected`), `crops/` (the hard B0039 crops, moved from
  `benchmarks/output/testcases`), `testcases/` + `testcases_3d/` (fixture fields,
  moved from `data/test_cases{,_3d}`), the pre-existing `canonical_2tri_2d/`,
  `b0039/`, `b0036/`, `archive/`, and a designated `results/` for corrected-field
  artifacts. `data/dvfs/README.md` (tracked) is the map; `python -m dvf_origins
  index` inventories the suite into `data/dvfs/manifest.json`.
- **No old path breaks**: every pre-suite location is a directory junction onto
  its new home, so unmerged branches, notebooks and external tooling keep
  working; the Python entry points (`benchmark_utils.cohort_dir`,
  `dvf_origins`, the crop/testcase scripts) now name the suite paths, with a
  fallback to the pre-suite cohort name for checkouts on the old layout.

### Added — resumable runs: `checkpoint_dir` / `dvfopt correct --checkpoint DIR`

- `dvfopt/checkpoint.py` — `RunCheckpoint`: a memmap mirror of the output
  (`<dir>/field.npy`) plus an atomically rewritten `state.json` (validated meta
  — shape, input sha256, the run's knobs — the finished unit ids, per-unit
  report rows, `stage`). A checkpoint from a different input or options
  raises `ValueError` naming the differing keys; a finished one reloads.
- Wired into `correct_dvf_25d` (per sweep slice; the 2.5D helper from the
  previous commit is refactored onto it), `DVFoptConfig(checkpoint_dir=)` (per
  z-slice, serial and `n_workers>1` — the pool now marks slices as they land),
  `correct_dvf_3d` (per stage: bulk, every escape iteration with its loop
  state, the multiscale re-seed; tighten is not a separate resume point) and
  the CLI's `--checkpoint DIR` on every `--pipeline` (`solver` = a finished
  run reloads). `checkpoint_dir=None` is byte-identical to before.

### Added — GUI: File → *New random folded field…*

- `dvfopt.testdata.make_patch_folded_dvf(shape, n_patches=, patch=, amp=, base_amp=, seed=)`
  builds a seeded synthetic 2D field — Gaussian-smoothed noise plus square
  patches folded by a slope-`amp` flip along a random direction — and the GUI's
  new File-menu action opens a spinbox form over it and loads the result through
  the same path as a file (undo / metrics / fold strip behave identically).

## [0.6.0] — 2026-08-31

The zero-folds release. The windowed elastic-QP engine's default formulation is
now self-contained — bilinear certificate rows + linear edge-monotonicity rows
+ the in-solve L2 anchor — and certified fold-free at scale: the 528-slice
full-resolution B0039 volume twice (599,313 → 0 simplex/bilinear/finite folds,
damage 0), and the 7-brain cohort at **203/203 slices, 0 folds, damage 0, on
stock defaults** (worst case 37,525 folds → 0 in 1857 s; the terminal re-seed
never fired). Ordinary slices solve in seconds-to-~100 s; the campaign's
speed fixes (window feasibility #103, ADMM cap #106), the objective menu
(`auto`/`l2`/`none`/`polish`/`reanchor`), the RAS giant-tile sweeps, the
opt-in QPALM backend, and the dead-knob pruning are all in. Findings, tables
and dead ends: `docs/superpowers/notes/zero-folds-campaign-findings.md`.

### Removed — windowed engine: the measured-dead, default-off knobs

- `untangle_delta` (+ the `_monotone_untangle` QP pass and the `SliceReport`
  `untangle_*` counters), `reseed_before_mop`, and `orientation_scope` are gone.
  All three were default-off options whose measurements said "don't": the blanket
  untangle fails the fidelity gate on ordinary fields (raw z16 L2 596 vs 268;
  `z0_sliver` 422 vs 21.5), re-seed-before-mop is too blunt for sliver residual
  (L2 137.8 vs 21.5), and rows-scoped-to-folds is slower AND worse (z=240: 677
  iterations vs 328 — the rows on all cells are what conditions the QP). The
  findings report keeps the measurements; the engine no longer carries the code.

### Added — isqp: `qp_backend='qpalm'` (opt-in) and the QP-inner investigation

- 316 real window QPs were captured from a z=240 solve (OSQP encoding, shared
  patterns + per-iteration values; harness in the campaign scratchpad, capture
  in `benchmarks/output/isqp_campaign/qpcap_z240.*`) and replayed through
  candidate solvers: **QPALM 8.4 s vs OSQP 21.2 s (2.5×, faster on every one of
  9 patterns, 23–36 outer iterations where ADMM runs to its 1000 cap,
  comparable feasibility)**; PIQP 19.3 s (machine-precision feasible but one
  pattern at 0.87 s/solve); proxsuite has no Windows wheel.
- `qp_backend='qpalm'` wires QPALM behind the OSQP `setup`/`update`/`solve`
  surface (its binding cannot update matrices, so each solve rebuilds
  Data + Solver and warm-starts from the previous solution — still 2.5× in the
  replay). Degrades to plain OSQP when `qpalm` is not installed (logged once).
- **Slice gate** (vs the hybrid default; every entry 0 folds, damage 0,
  identical L2 move): z=240 **49 s vs 99 s (2.0×** — the L2 default at the
  `none` objective's wall); z16 75 vs 65 s (+15 %); full-res z=2 **3159 vs
  328 s (9.6× WORSE** — 7 rounds, 153 a*-collapses, the ladder's backend rung
  firing 432 OSQP retries); sliver crop +55 % iterations. So the backend is
  **opt-in for QP-bound ordinary fields only**; the default stays `'hybrid'`.
- The refined finding (findings §4.3f): the L2 per-QP cost is an OSQP-ADMM
  artifact, not intrinsic to the QPs — but the SQP's step dynamics co-evolved
  with OSQP's solution style, and a different (even better-converged) QP
  solution perturbs iterates enough to shatter trap-heavy trajectories. A
  qpalm-on-mild-windows / osqp-escalation policy is the natural follow-up.

### Added — windowed engine: `giant_workers` — RAS (Jacobi) giant-tile sweeps (opt-in)

- With `giant_workers > 1`, each giant-region Schwarz sweep runs as restricted
  additive Schwarz on the shared spawn pool: every tile of the sweep is solved
  concurrently from the same snapshot of the field, and each pastes back only
  its disjoint step-grid core (the partition is tested). Trades the serial
  multiplicative sweep's within-sweep propagation (more rounds) for parallel
  tiles; `0` (the default) is the serial sweep, byte-identical (tested).
- **Measured** (L2 default, serial references from the same engine, idle box;
  0 folds / damage 0 everywhere): z=240 (5 giant regions) **73 s vs 99 s
  (−26 %)** at move 30.1 vs 29.9; full-res z=2 (1 giant region) 335 s vs 328 s
  (neutral) at move 1966.1 vs 1977.8. The knob is the single-slice latency
  lever for giant-heavy slices; inside a per-slice pool worker it caps to 1
  (the pool never nests), so volume throughput is unaffected.
- Same investigation, measured and recorded (findings §4.3e): the terminal
  `reanchor` stage is the max-fidelity mode on ordinary slices only — it beats
  even the in-solve L2 anchor there (z=240 move 28.1 vs 29.9, z=440 54.5 vs
  67.8, at 3–4× the wall) but recovers NOTHING on trap-heavy slices (z=2: all
  tiles revert), where only the warm at-solve-time anchor works. No mode
  dominates; all are existing knobs.

### Added — windowed engine: `polish='l2'|'l1'` — per-window anchored polish (opt-in)

- After a window solves, the SAME box is re-solved on the current field against
  the distance to its pre-solve patch, from the warm feasible point
  (`polish_maxiter=30` inner iterations, the `ftol` stop), with the reanchor
  stage's verify-and-revert — so it can never cost feasibility or fidelity.
  Meant for `objective='none'`: the cheap pure-feasibility solve plus a short
  anchored polish, as a middle point between `none` and the in-solve L2 default.
- **Measured** (current engine, serial, idle box; wall s / L2 move; every entry
  0 folds, damage 0). L2 → `none` → `none`+polish:
  z=240 99/29.9 → 48/36.0 → **72/33.5** (41 % of the fidelity gap recovered);
  z=440 255/67.8 → 97/87.4 → 113/85.9 (8 %); z16 65/189.6 → 27/227.7 →
  36/222.0 (15 %); z=2 328/1977.8 → 411/2341.1 → **425/2025.9** (87 %, one
  round fewer than plain `none`); crops: z16_twist reaches L2's exact move
  (70.7 vs plain `none`'s 123.1), z0_cluster/z0_sliver unchanged (reverted).
- **Why it cannot fully close the gap** (and the in-solve L2 default stays):
  the polish recovers the WITHIN-window share of the anchor's fidelity; the
  rest is trajectory shaping during the solve — the anchor steering windows
  into different basins and negotiating with their frozen rings — which no
  post-hoc per-window polish can reproduce (z0_cluster: the polish runs 30
  iterations and reverts; L2's smaller move lives in a different basin).

### Changed — windowed engine: `qp_max_iter=1000` (was 2000)

- Under the in-solve L2 default the window QPs are harder than under the zero
  objective (ADMM median ~1060 iterations; most solves reach the hybrid backend's
  800-iteration Clarabel handoff). Capping ADMM at 1000 lets the SQP absorb the
  capped solves: same-slice, equal-contention A/Bs give z=240 **−13 %** wall
  (223 vs 257 s) and volume z16 **−37 %** (333 vs 529 s) at the identical SQP
  iteration count, L2 move and 0 folds / damage 0; the hardest slice (full-res
  z=2) is neutral (1235 vs 1214 iterations, L2 1977.8 vs 1978.8) and the crop
  pack holds (L2: z16_twist 58 → 29, z0_cluster 120 → 96, z0_sliver 617 → 637;
  `none`: identical or better, z0_sliver 620 → 286 at a smaller move).
- **Measured and rejected:** 500 (ordinary slices −25 %, but the hard slice
  2013 iterations and the L2 sliver crop +28 %), 750 (no better than 1000),
  4000, the Clarabel handoff threshold 400 / 1500, no cold IP solve, and
  OSQP-only (539 vs 481 s — the hybrid earns its keep).

### Fixed — windowed engine: a window landing within half the margin of the shifted target is solved (`feas_tol`); `ftol` stop

- **Why.** The in-solve L2 default (#99) cost ~3x on ORDINARY slices (151-slice
  interim of the full-resolution certification: <2000-fold slices median 171 s ->
  559 s at the SAME SQP iteration count overall; the hard slices went 78 623 s ->
  9 119 s). A traced 4-way A/B on z=440 (1828 folds, all four concurrent) put the
  whole cost on the objective, not the rows: rows + `none` 356 SQP iterations
  (old engine 433 — the edge rows *condition* the QP, median ADMM iterations
  637 -> 187), L2 + rows 915, and 59 % of those (544) inside 48 window calls
  that ended `a-collapse` FAILED and were then fed the escalation ladder
  (no-trust-region retries 32 vs 5, window calls 187 vs 110). The per-iteration
  trace of those windows shows them CONVERGING — max violation 3.3 -> 1e-5,
  merit / 700 — with 25–30 rows hovering 1e-5..1e-4 below the margin-SHIFTED
  target: a distance objective parks the solution ON the active rows at ADMM
  precision, where a zero objective steps off the boundary to exactly 0. The
  margin (`margin_delta=1e-3`) exists precisely so that "a solve landing a hair
  short of the active bound is still fold-free", but the isqp inner's own test
  was `cons >= -1e-6`, three orders stricter than that slack.
- **Change.** `solve_window_inner(feas_tol=)` re-derives `feasible` as
  `sub.cons(x).min() >= -feas_tol` for every inner, and `isqp_solve(feas_tol=)`
  uses the same slack for its own flag; `_solve_window` passes
  `0.5 * margin_delta` (rows at `threshold + margin/2` are fold-free by
  construction; the engine's final fold check against the real threshold is
  untouched). `ftol` (engine / `ISQPWindowedStrategy` knob, `isqp_solve`
  parameter; **default 1e-2**, 0 = off) is a relative objective-decrease stop for
  such feasible-within-slack iterates — the remaining L2 cost was in-window
  polishing along the active rows at a median relative merit decrease of 3e-4
  per iteration. Calibrated on z=440 under L2: feas_tol alone 790 SQP iterations,
  ftol 1e-3 647, **1e-2 591**, all at the identical L2 move 67.8, 0 folds,
  damage 0 (the default engine: 915).
- **Measured** (bilinear rows, edge rows, threshold 0.01, 0 simplex / 0 bilinear
  folds and damage 0 everywhere): z=440 under L2 **915 -> 790 SQP iterations,
  window calls 187 -> 147, window success 67 % -> 91 % (old engine 92 %),
  no-TR retries 32 -> 7, L2 move 67.8 unchanged**; volume z=16 (2131 folds)
  442 -> 385 iterations, 61 -> 35 calls, L2 190.0 -> 189.5; full-res z=2 (the
  hardest slice) unchanged (1239 vs 1224, L2 1980.5 vs 1978.8 — its collapses
  are genuine); crop pack under L2 byte-identical (z16_twist one call fewer);
  under `none` byte-identical except `z0_sliver` 890 -> 621 iterations at L2
  1109 -> 1151 (fewer retries, less polish on the reflected crop).
- **Measured and rejected** on the same slice (do not retry): the penalty
  parameter (`rho` 1e4: 873 iterations, ADMM median 1237; 1e5: 898, ADMM at the
  2000 cap, L2 worse), a 1.0 px initial trust region (845), the a\*-collapse bail
  off (1323) or at 6 (1177), and a "collapse needs a standing violation"
  predicate (905 — inert everywhere).

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
