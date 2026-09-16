# Canonical 2D benchmark — tracked results

This is the paper's tracked home for the canonical 2D I-SLSQP benchmark: one pinned engine, one
pre-registered protocol, every 2D source in the repo, nothing dropped. It holds the small,
reproducible artefacts (`results.csv`, `summary.json`, `manifest.json`, `table.md`, figures) for
each finished run; the corrected DVFs themselves are gitignored and live under
`data/dvfs/results/` (see Layout). Full protocol: `docs/superpowers/notes/2026-09-11-2d-canonical-benchmark-handoff.md`;
plan: `docs/superpowers/plans/2026-09-11-2d-canonical-benchmark.md`; ledger:
`.superpowers/sdd/2026-09-11-2d-canonical-benchmark/progress.md`.

## Protocol (pre-registered before the first run, verbatim in substance)

- **Engine:** `dvfopt.correct_dvf(phi, threshold=0.01, record_history=True, **config)` at the
  pinned commit. No per-case knobs. `err_tol` is the package default, `1e-5`.
  `constraint='bilinear'` is the 4-triangle-per-cell certificate (both diagonals); the simplex
  families' `'simplex_standard'` row is the fixed-diagonal 2-triangle one.
- **Seven configs** (`benchmarks/canonical_2d.py`'s `CONFIGS`), plus the `isqp_l1` amendment below:

  | config | constraint | strategy | objective |
  |---|---|---|---|
  | `isqp_none` | `bilinear` | `isqp_windowed` | `none` |
  | `isqp_l2` | `bilinear` | `isqp_windowed` | `l2` |
  | `auto` | `bilinear` | `auto` | `auto` |
  | `slp` | `simplex_standard` | `slp` | `l1` |
  | `barrier` | `simplex_standard` | `barrier` | `l2` |
  | `m14` | `simplex_standard` | `m14` | `l2` |
  | `slsqp_windowed` | `jdet` | `slsqp_windowed` | `l2` |

  The two `isqp_*` rows (the engine's own headline configs) run on **every** source; the rest of
  the taxonomy runs on the three small sources only (origins, crops, synthetic) so the paper's
  "across methods AND across registration sources on a common axis" table exists without paying
  for the full taxonomy on ~600 cohort/ANTs slices.
- **`isqp_l1` (ruling R19, amendment):** the windowed engine's L1 anchor (`bilinear` +
  `isqp_windowed` + `objective='l1'`), added alongside the taxonomy rows on the small sources only
  (origins, crops, synthetic) — not one of the two every-source engine headline rows. Status:
  pending, no numbers yet.
- **The three hard B0039 crops (`z0_sliver`, `z0_cluster`, `z16_twist`) are the engine's TUNING
  set** — every table below says so, and their numbers are never quoted as a held-out result.
- **Nothing is dropped:** every pair is a row in `results.csv`, in `manifest.json`, and in every
  aggregate denominator; only the median/IQR *distributions* skip sentinel rows (see Metric
  definitions). A pair that never produced a result (its input would not load, its pool worker
  died, or the watchdog cut it) is a full `-1` row with `feasible=False` and the error text. A
  solve that *raises* is not a `-1` row: it carries the error text, `feasible=False` and
  `certified=False`, its metrics are real values of the unchanged input field, and its `time_s` is
  the real time to the exception. A run exceeding its 2-hour soft cap (`--cap-s`, default 7200 s)
  is recorded as `hit_cap=True` but is not interrupted by that cap, and `time_budget_s` is left
  `None`. The one interruption is the pool path's **no-progress watchdog** (driver commits from
  37eecde on; recorded as kept `WatchdogTimeout` rows from 5904a0b on): when a pool completes no
  pair for `watchdog_timeout_s` (default 6 h, env
  `CANONICAL_2D_NO_PROGRESS_S`), the pairs its workers were running are cut and recorded as
  `WatchdogTimeout` rows (`-1`, `hit_cap=True`), which `--resume` keeps as measured "did not
  finish" outcomes. The tracked runs here were made by drivers cdbf5f4 / 90fccab, which predate
  the watchdog, so every one of them ran to the engine's own termination.
- **Walls:** throughput passes use `n_workers` (2-4); the paper's per-case wall column is meant to
  come from a serial pass on an idle box. See Walls below — this session's box was never idle.

## Engine pin

Driver commit **`cdbf5f4`** (branch `feat/2d-canonical-benchmark`, worktree
`../dvfopt-canonical-2d`), built on top of **main `4636af8`** (#125). `dvfopt/` itself is
byte-identical to that main commit — `git diff --stat main -- dvfopt/` is empty for the whole
branch — so every number here is the shipped engine's own behavior, not a benchmark-specific
tweak.

## Sources

All payloads are gitignored; `data/dvfs/README.md` is the tracked map.

| source | what | on-disk path | count in the sample |
|---|---|---|---|
| `origins` | fields generated per fold-origin mechanism by the `dvf_origins` harness (m1 interpolation, m2 dense optimization, m3 learned, m4 diffeomorphic) across 8 registration tools | `data/dvfs/origins/<mechanism>/<case>.npy` + `.json`, index `data/dvfs/origins/manifest.json` | 27 |
| `cohort` | 7 real brains' Laplacian-interpolated fields with correspondences | `data/dvfs/cohort/<brain>/laplacian_exterior/laplacian_deformation_field.npz` | ≈85 slices: `z ∈ range(0, 528, 48)` (11/brain) + the named hard slices (B0039 z1,2,11,16,264; B0032 z1; B0304 z128,181 — the cohort-sweep plateau slices) |
| `ants` | the same 7 brains' ANTs SyN warps — the "already injective" control | `data/dvfs/cohort/<brain>/laplacian_exterior/ants_warp_0.nii.gz` | same slice sample as `cohort`, ≈85 |
| `crops` | the three hard B0039 crops — **the tuning set** | `data/dvfs/crops/{z0_sliver, z0_cluster, z16_twist}.npy` | 3 |
| `synthetic` | `dvfopt.testdata.SYNTHETIC_CASES` + `RANDOM_DVF_CASES` (fixed seeds) + any `data/dvfs/canonical_2tri_2d/*.npz` not already covered by name | `dvfopt/testdata/`, `data/dvfs/canonical_2tri_2d/*.npz` | 13 |

## Metric definitions

Every record is one `(input, output, result)` triple, computed by `benchmarks/canonical_2d.py`'s
`metrics()`. Column groups:

- **The existing `cohort_benchmark` row schema**, unchanged names, on the central-difference Jdet
  (`dvfopt.jacobian.numpy_jdet.jacobian_det2D`): `n_neg_init`/`n_neg_final`,
  `neg_vol_init`/`neg_vol_final`, `n_clusters_init`/`n_clusters_final`,
  `min_jdet_init`/`min_jdet_final`, `l2_err`, `time_s`. **Gauge trap:** this schema's
  `n_neg_init`/`n_neg_final` count central-difference `jdet < 0.01` (the threshold itself, no
  `err_tol`), which is NOT the `<family>_n_neg_*` gauge below (`fold_stats`'s `n_neg`, values
  `<= 0`) nor the `<family>_n_below_*` gauge (`< 0.01 - 1e-5`). Same prefix, different predicate.
- **The four certificate families, before and after, each at two gauges:** `simplex` (2 triangles
  per cell, fixed BL-TR diagonal), `bilinear` (4 triangles per cell, both diagonals), `finite`
  (forward-difference Jdet, 1 triangle per cell), `jdet` (central-difference Jdet). Each family
  contributes `<family>_n_neg_<init|final>` (count `<= 0` — the gauge **at 0**),
  `<family>_n_below_<init|final>` (count `< threshold - err_tol`, i.e. `< 0.01 - 1e-5` — the gauge
  **at the 0.01 threshold**), and `<family>_min_<init|final>` (the minimum value, at that column's
  own scale). Counts are PER LOCATION (per cell for the three simplicial families, per pixel for
  `jdet`), never per constraint row. **`bilinear` is exactly `cell_min_jdet_2d / 2`** (triangle
  area is half the determinant, verified exact in the driver) — it is the same sub-pixel
  certificate the solver itself constrains against, evaluated at the solver's own scale. **A paper
  quoting the `bilinear` column must not halve it again** — the 0.01 threshold in this table is
  already the triangle-area gauge, not the determinant gauge. `certified` is the headline verdict:
  `bilinear_n_below_final == 0`; the other three families are reported beside it, never instead of
  it.
- **The registration-standard pair** (over ALL pixels of the central-difference Jdet, no
  foreground mask — the sources here carry none): `frac_nonpos_jdet = mean(jdet <= 0)` and
  `sdlogj = std(log(clip(jdet, 1e-3, None)))`, both before and after. **Clipping convention:**
  Learn2Reg computes SDlogJ on the foreground only; since none of these sources have a mask, this
  is whole-slice, and non-positive Jacobians (where the log is undefined) are clipped up to `1e-3`
  rather than dropped or masked out — stated once here so a paper number is never read as the
  foreground-only Learn2Reg convention.
- **Engine accounting**, from `res.info`, **windowed rows only** (`ISQPWindowedStrategy` or
  `WindowedWrapperStrategy` — including an `auto` run that resolved to one of them):
  `damage` (`res.info.extras['damage']`; must be exactly 0 on every windowed row — a nonzero value
  is a bug report, not a result), `n_windows`, `giant_regions`, `mop_cleared`, `rounds` (count of
  phases named `round*`), `sqp_iters` (sum of `n_iter` over phases NOT named `giant*` — a `giant`
  phase is nested inside its round entry, so `total_iter` would double-count it). **`sqp_iters`
  is not the total SQP count when the re-seed or re-anchor stage ran:** it counts the round,
  coarse and mop iterations, while the engine's history records the re-seed and re-anchor stages
  with `n_iter = 0` even though their polish solves iterate (`dvfopt/core/windowed/_common.py`);
  never quote it as a total on such a row. **`mop_cleared` is signed** — the engine's folded
  pixels before the mop minus after — so a `-1` on a windowed row is a measurement (the mop left
  one more folded pixel than it found; e.g. `origins_serial`'s `m2_ffd_brainpair_coarse` ×
  `isqp_none`), not a sentinel. **`-1` on every
  other strategy** (`barrier`, `slp`, `m14`, `slsqp_windowed`) — the other strategies report phases
  in their own units (L-BFGS iterations, named SLP stages with `n_iter=0`), and summing those under
  the same column name would mix three different quantities. `slsqp_windowed` is
  `SLSQPWindowedStrategy`, a different class from the windowed I-SLSQP engine — the driver's
  `WINDOWED_STRATEGY_NAMES = ("ISQPWindowedStrategy", "WindowedWrapperStrategy")` never names it —
  so its rows carry `-1` here unconditionally, not "when it did not resolve to the windowed engine".
  Apart from the signed `mop_cleared`, `-1` is a sentinel, never a measurement. The median/IQR
  aggregates apply the sentinel rule by row type: an engine column is aggregated over the
  windowed rows only (`damage >= 0`), sign kept, and every other column skips the never-ran rows
  (whose values are all `-1`). The count/rate denominators (`n`, `feasible_rate`,
  `certified_rate`) still include every row, including full-sentinel rows for a case that never
  produced a result at all (load failure, a dead pool worker, or a watchdog cut).
- **Move and locality:** `moved_frac` (fraction of pixels with any channel changed by more than
  `1e-9`), `l1_move` (`sum(|phi_out - phi_in|)`), `l2_move` (`norm(phi_out - phi_in)`), `max_move`
  (`max(|phi_out - phi_in|)`), `mean_move_moved` (`l1_move` divided by the count of moved pixels).
- **Correspondence fidelity — cohort rows only:** `corr_n`, `corr_resid_med_init/final` (median
  registration residual at the prescribed Laplacian boundary correspondences),
  `corr_resid_mad_init/final` (MAD, scaled by 1.4826), and the outlier counts
  `corr_n_outliers`/`corr_n_large`/`corr_n_high_resid`/`corr_n_incoherent` from
  `correspondence_analysis.analyze_slice`. **`-1` on every non-cohort row, and on cohort slices
  that carry no correspondences** — in `cohort/` that is 20 rows (10 slices × 2 configs: every
  brain's z0, plus B0304 z48, z96 and z480).
- **Injectivity diagnostics (optional, cheap):** `ift_min_radius_<init|final>` and
  `ift_frac_subpixel_<init|final>` from `dvfopt.metrics.injectivity_stats` — the quantitative-IFT
  radius **estimate**, orientation-blind and never a certificate; read beside the four certificate
  gauges, never instead of them.
- **Provenance, per run directory** (`summary.json`'s `provenance` block): git commit,
  `dvfopt.__version__`, the config dict, `n_workers`, python + OS, box CPU load at the start of the
  run, `time_budget_s` (always `null` — the engine is never given a soft budget internally),
  `cap_s` (the driver's own recorded-not-enforced cap), and `hit_cap` per row. Drivers after
  65c1b15 also record `git_dirty` (tracked-file changes in the driver's repo, `null` without
  git), `dvfopt_path` (the directory of the `dvfopt` actually imported — so `git_commit` can be
  matched to the engine that ran), `driver_path`, and `watchdog_timeout_s`. **The tracked
  summaries below predate these four keys** and do not carry them; their engine was verified
  separately (see Engine pin).

## Walls

The box was **not idle** for most of this session's runs — two unrelated user jobs (one already
running for 70+ CPU-hours, a second started mid-session) put the recorded start load at 24%
(`synthetic`), 39% (`cohort`) and 76% (`crops`); only `origins_serial` and `ants` started on a
near-idle box (2%). Every wall (`time_s`) in `crops/` and `synthetic/` is a **throughput** pass
(`timing_mode: "throughput"`, recorded `n_workers: 1` since both sources are small enough to run
sequentially without a pool) taken under that contention, not the paper's intended idle-box serial
column. The pre-registered idle-box serial subsets (origins: all 27 cases; cohort: z=0 and z=240 of
every brain) are the recorded follow-up, to be run once the box is confirmed idle — see Status.

The **origins serial-timing pass** (`origins_serial/`) is that follow-up for `origins`: it ran on a
near-idle box (load 2 % at its start; `timing_mode=serial`, one in-process solve at a time, no
pool) and supplies the paper's per-case wall column for this source. The throughput passes
(`n_workers=4` on a shared, contended box) measured the same pairs with inflated walls: observed
on identical pairs, `m2_demons_brainpair_weak` × `isqp_none` ran 147 s serial vs 394 s throughput,
and `m2_ffd_brainpair_coarse` × `isqp_none` ran 951 s serial vs 2,498 s throughput — about 2.6x.
Only the serial pass's `time_s` column should be quoted as this source's wall.

The `cohort` walls (`cohort/`) are likewise a **throughput** pass, not the paper's per-case wall
column: `n_workers=4`, run while the box also carried the `origins` recovery chain (an unrelated
job). The idle-box serial re-timing of a cohort subset (z=0 and z=240 of every brain) is the
recorded follow-up — see Status and Regeneration.

## Caveats

- **R4 — the on-disk crops do not reproduce CLAUDE.md's historical move/wall figures.** CLAUDE.md
  documents `z0_sliver` at L2 move ≈19-25 and walls of 32/22/106 s for the three crops; this run
  measures `z0_sliver` at L2 move 805.8-910.6 and very different walls (crops fold counts do match:
  645/598/0 → 0/0/0). The ledger bisected this across four historical engine commits and reproduced
  today's number bit-for-bit on all of them, and confirmed the crop's channel order and the
  engine's own `l2_move` extra agree with the driver's — there is **no engine regression**. The
  conclusion is that CLAUDE.md's historical figures describe an *earlier cut* of the crop file
  (rebuilt since from a different campaign output), not the current file on disk or a driver bug.
  **Only this directory's numbers are quotable for the crops** — do not cite CLAUDE.md's historical
  crop move/wall figures alongside them.
- **ANTs slice indices may be reversed.** `load_dvf` on an ANTs warp returns axes reoriented to the
  cohort grid up to a per-axis reversal (the three spatial axis lengths differ, so the permutation
  is forced, but the direction along each axis is not); an `ants_<brain>_z<k>` row may correspond
  to the cohort's `z = 527 - k`. A reversal negates the axis and its displacement component
  together, so every Jacobian sign — and hence every metric here — is invariant; do not pair
  `cohort_*` and `ants_*` rows by z.
- **`max_damage` is meaningful only on windowed rows.** It is `null` (not `-1`, which would read as
  "damage minus one is better than zero") whenever a group has no windowed-engine row at all
  (`barrier`/`slp`/`m14` groups, and every `slsqp_windowed` group — `slsqp_windowed` is
  `SLSQPWindowedStrategy`, a different class from the windowed I-SLSQP engine, so it never
  resolves to it); it is exactly 0 on every group that does contain windowed rows in the results so
  far.
- **Reproducibility between pooled and in-process runs is discrete, not bit-identical.** Two
  pooled runs, whose workers are pinned to one BLAS thread (the two throughput passes of
  `m2_ffd_brainpair_coarse` × `isqp_none`, before and after a machine restart), agree bit-for-bit
  in all 16 outcome fields. Between a pooled (pinned) run and an in-process (unpinned) run of the
  same pair the *discrete* outcome is identical — certified, feasible, fold counts under every
  gauge, damage, windows, rounds, SQP iterations — while continuous outputs (L1 / L2 / max move)
  differ by about 3e-5 relative and the worst residual value by about 2.5 %. Most plausibly this
  is the BLAS pinning (supported by the pinned-vs-pinned bit-identical check above, not proven
  further). `timing_mode` does NOT record pinning: the `crops/` and `synthetic/` rows are
  `timing_mode=throughput` but ran in-process and unpinned (`n_workers=1`). **Never describe
  results as bit-identical between pooled and in-process runs** — only within one kind.

## Layout

```
docs/paper/results/2d_canonical/
├── README.md                  # this file
├── crops/                     # FINAL — the 3-crop tuning-set row
│   ├── results.csv
│   ├── summary.json
│   ├── manifest.json
│   ├── table.md
│   └── figures/*.png
├── synthetic/                 # FINAL — the synthetic-case row
│   ├── results.csv
│   ├── summary.json
│   ├── manifest.json
│   └── table.md                # no figures/: this run was not passed --figures
├── origins_serial/             # FINAL — origins, serial-timing wall column
│   ├── results.csv
│   ├── summary.json
│   ├── manifest.json
│   └── table.md                # no figures/: this run was not passed --figures
├── ants/                       # FINAL — the ANTs "already injective" control
│   ├── results.csv
│   ├── summary.json
│   ├── manifest.json
│   └── table.md                # no figures/: this run was not passed --figures
└── cohort/                     # FINAL — the cohort sample (throughput walls, see Walls)
    ├── results.csv
    ├── summary.json
    ├── manifest.json
    ├── table.md
    └── figures/*.png
```

One subdirectory per source, matching the run-dir names in
`benchmarks/output/2d_canonical/<row>/` (gitignored) — `ants/` here holds `ants_isqp/`'s output.
`report/` (the `cohort_benchmark`-shared `report.html` + its own filtered CSV/JSON copy) is **not**
copied here — it duplicates `results.csv`/`summary.json` at a filtered column set and is meant to
be regenerated locally, not tracked. Corrected DVFs are never tracked: they live under
`data/dvfs/results/<run-name>/<source>/<case>__<config>.npz` (gitignored), and each source's
`manifest.json` here lists every one with its case, config, shape and sha256, plus the untouched
*input* path (inputs are never duplicated).

The pending source (`origins` full taxonomy) will get its own subdirectory here, in the same
shape, once its recovery chain finishes — see Status.

## Regeneration

Every command below runs from the repo root, against the pinned commit. `--figures` is included
only where the corresponding run directory actually has a `figures/` — that is a record of what
was run, not a recommendation to omit figures elsewhere.

```bash
# crops (the tuning set) — FINAL, this directory's crops/
python benchmarks/canonical_2d.py --source crops \
    --config isqp_none isqp_l2 auto slp barrier m14 slsqp_windowed \
    --figures --table --hist-case z0_cluster \
    --run-dir benchmarks/output/2d_canonical/crops_all

# synthetic — FINAL, this directory's synthetic/
python benchmarks/canonical_2d.py --source synthetic \
    --config isqp_none isqp_l2 auto slp barrier m14 slsqp_windowed \
    --table \
    --run-dir benchmarks/output/2d_canonical/synthetic_all

# origins, all 7 configs, throughput pass — being recovered, see Status (was origins_all, invalid;
# origins_all_recovered replaces it)
python benchmarks/canonical_2d.py --source origins \
    --config isqp_none isqp_l2 auto slp barrier m14 slsqp_windowed \
    --n-workers 4 --figures --table \
    --run-dir benchmarks/output/2d_canonical/origins_all

# origins, the paper's wall column (isqp_none / isqp_l2 only, serial) — FINAL, this directory's
# origins_serial/
python benchmarks/canonical_2d.py --source origins \
    --config isqp_none isqp_l2 --serial-timing --table \
    --run-dir benchmarks/output/2d_canonical/origins_serial

# origins recovery: reuse origins_all's measured rows, rerun exactly its 130 BrokenProcessPool
# losses, with slsqp_windowed pairs isolated one at a time (--resume / --isolate-config exist from
# driver commit 90fccab onward) — this run replaces origins_all; origins_all's own aggregates must
# not be used
python benchmarks/canonical_2d.py --source origins \
    --config isqp_none isqp_l2 auto slp barrier m14 slsqp_windowed \
    --n-workers 4 --resume benchmarks/output/2d_canonical/origins_all \
    --isolate-config slsqp_windowed --figures --table \
    --run-dir benchmarks/output/2d_canonical/origins_all_recovered

# ANTs controls, isqp_none only (expected 0 -> 0 at ~0 s) — FINAL, this directory's ants/
python benchmarks/canonical_2d.py --source ants \
    --config isqp_none --n-workers 4 --table \
    --run-dir benchmarks/output/2d_canonical/ants_isqp

# cohort sample, the two engine configs, throughput pass — FINAL, this directory's cohort/
python benchmarks/canonical_2d.py --source cohort --config isqp_none isqp_l2 \
    --n-workers 4 --figures --table \
    --run-dir benchmarks/output/2d_canonical/cohort_isqp
```

The cohort's idle-box **serial**-timing subset (z=0 and z=240 of every brain, 14 slices × 2
configs) is deliberately not listed above — per ledger ruling R5 it was not run this session (the
box was never idle) and is the recorded follow-up, run only once load is confirmed low:

```bash
python benchmarks/canonical_2d.py --source cohort --config isqp_none isqp_l2 \
    --serial-timing --run-dir benchmarks/output/2d_canonical/cohort_serial
    # then filter to z in {0, 240} per brain, or add a --case-filter if one exists by then
```

## Status

- **`crops`, `synthetic`, `origins` (serial-timing pass), `ants`, `cohort`: FINAL.** Tracked here
  in full (see the tables below).
- **`origins` (full 7-config taxonomy, throughput): being recovered.** The original
  `origins_all` run lost 130 of its 189 pairs to a dead pool worker
  (`BrokenProcessPool`) mid-run; a recovery run reuses the 59 rows it did measure and reruns
  exactly the 130 losses, with `slsqp_windowed` pairs isolated one at a time. It replaces
  `origins_all`, whose aggregates (`summary.json`/`table.md`, computed over the 130 sentinel rows)
  must not be used.
- The findings-note section, the CHANGELOG entry and the CLAUDE.md benchmarks-bullet update follow
  in a later commit, once the origins recovery also finishes.

## Results — `crops` (TUNING SET)

<!-- pasted verbatim from crops/table.md, including its legend comment -->

<!-- certificate gauges: simplex: 2 triangles per cell (fixed BL-TR diagonal), triangle area = det/2, per cell (last row/col are +inf); bilinear: 4 triangles per cell (both diagonals), triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2, per cell (last row/col are +inf); finite: forward-difference Jdet (1 triangle per cell), determinant, per cell (last row/col are +inf); jdet: central-difference Jdet, determinant, per pixel. certified = bilinear has 0 values < 0.01 - 1e-5 after. -1 is a sentinel (see summary.json notes), skipped by every median. -->
| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |
|---|---|---|---|---|---|---|---|---|---|---|
| crops (TUNING SET) | auto | 3 | 3/3 | 3/3 | 32.17 [31.35, 75.42] | 1.743e+04 [9596, 2.131e+04] | 690.5 [380.6, 748.2] | 2.66 -> 1.586 | 0.208 -> 0 | 0 |
| crops (TUNING SET) | barrier | 3 | 0/3 | 1/3 | 2.195 [1.115, 2.372] | 1331 [665.3, 4431] | 47.19 [23.6, 157.7] | 2.66 -> 1.788 | 0.208 -> 0.0264 | n/a |
| crops (TUNING SET) | isqp_l2 | 3 | 3/3 | 3/3 | 41.43 [35.68, 177.1] | 1.743e+04 [9596, 2.155e+04] | 690.5 [380.6, 800.5] | 2.66 -> 1.795 | 0.208 -> 0 | 0 |
| crops (TUNING SET) | isqp_none | 3 | 3/3 | 3/3 | 6.28 [4.813, 45.43] | 2.466e+04 [1.459e+04, 2.492e+04] | 787.5 [455.3, 796.6] | 2.66 -> 1.586 | 0.208 -> 0 | 0 |
| crops (TUNING SET) | m14 | 3 | 0/3 | 1/3 | 6.617 [5.973, 8.955] | 2050 [1025, 5997] | 76.68 [38.34, 193.8] | 2.66 -> 1.677 | 0.208 -> 0.0216 | n/a |
| crops (TUNING SET) | slp | 3 | 0/3 | 2/3 | 8.286 [7.491, 9.169] | 2114 [1057, 1.477e+04] | 87.11 [43.55, 514.8] | 2.66 -> 1.677 | 0.208 -> 0.004 | n/a |
| crops (TUNING SET) | slsqp_windowed | 3 | 0/3 | 3/3 | 915.6 [457.8, 4000] | 975.5 [487.8, 3057] | 34 [17, 113.8] | 2.66 -> 2.004 | 0.208 -> 0 | n/a |

Box load at start: 76%, contended (see Walls). Only the two `isqp_*` rows and `auto` (which
resolved to the windowed engine on all three crops) certify at 0 folds; `barrier`/`m14`/`slp`
leave residual bilinear folds and `slsqp_windowed` is feasible only under its own (central-Jdet)
gauge, never the bilinear one — it is the smallest-move, longest-wall row. Full figures
(fold-count bars + per-config jdet histograms for `z0_cluster`) in `crops/figures/`.

## Results — `synthetic`

<!-- pasted verbatim from synthetic/table.md, including its legend comment -->

<!-- certificate gauges: simplex: 2 triangles per cell (fixed BL-TR diagonal), triangle area = det/2, per cell (last row/col are +inf); bilinear: 4 triangles per cell (both diagonals), triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2, per cell (last row/col are +inf); finite: forward-difference Jdet (1 triangle per cell), determinant, per cell (last row/col are +inf); jdet: central-difference Jdet, determinant, per pixel. certified = bilinear has 0 values < 0.01 - 1e-5 after. -1 is a sentinel (see summary.json notes), skipped by every median. -->
| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic | auto | 13 | 13/13 | 13/13 | 0.2246 [0.05676, 2.428] | 46.12 [8.411, 405.1] | 6.475 [3.054, 31.45] | 2.31 -> 1.283 | 0.14 -> 0 | 0 |
| synthetic | barrier | 13 | 1/13 | 9/13 | 0.4013 [0.2098, 0.7546] | 75.02 [9.879, 272.1] | 7.386 [3.051, 20.39] | 2.31 -> 1.524 | 0.14 -> 0.0163 | n/a |
| synthetic | isqp_l2 | 13 | 13/13 | 13/13 | 0.3028 [0.06111, 1.483] | 46.13 [8.23, 409.9] | 6.001 [3.053, 31] | 2.31 -> 1.332 | 0.14 -> 0 | 0 |
| synthetic | isqp_none | 13 | 13/13 | 13/13 | 0.1276 [0.04434, 0.5276] | 47.42 [8.609, 681.1] | 6.537 [3.067, 41.36] | 2.31 -> 1.111 | 0.14 -> 0 | 0 |
| synthetic | m14 | 13 | 1/13 | 12/13 | 0.5038 [0.3326, 1.536] | 64.42 [10.6, 295.5] | 6.712 [3.051, 20.93] | 2.31 -> 1.562 | 0.14 -> 0.0213 | n/a |
| synthetic | slp | 13 | 4/13 | 13/13 | 0.7087 [0.4018, 1.778] | 34.42 [7.093, 240.5] | 6.605 [3.045, 25.39] | 2.31 -> 1.404 | 0.14 -> 0 | n/a |
| synthetic | slsqp_windowed | 13 | 0/13 | 13/13 | 0.06125 [0.01266, 1.681] | 34.04 [8.872, 242.4] | 3.867 [2.271, 14.37] | 2.31 -> 1.604 | 0.14 -> 0 | n/a |

Box load at start: 24% (much less contended than the crops row). Both `isqp_*` rows and `auto`
certify 13/13; `slsqp_windowed` again has the smallest move but 0/13 under the bilinear gauge
despite 13/13 feasibility under its own central-Jdet constraint. No `figures/` for this row (run
without `--figures`).

## Results — origins — serial timing pass

<!-- pasted verbatim from origins_serial/table.md, including its legend comment -->

<!-- certificate gauges: simplex: 2 triangles per cell (fixed BL-TR diagonal), triangle area = det/2, per cell (last row/col are +inf); bilinear: 4 triangles per cell (both diagonals), triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2, per cell (last row/col are +inf); finite: forward-difference Jdet (1 triangle per cell), determinant, per cell (last row/col are +inf); jdet: central-difference Jdet, determinant, per pixel. certified = bilinear has 0 values < 0.01 - 1e-5 after. -1 is a sentinel (see summary.json notes), skipped by every median. -->
| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |
|---|---|---|---|---|---|---|---|---|---|---|
| origins | isqp_l2 | 27 | 27/27 | 27/27 | 10.25 [0.497, 109] | 1034 [21.28, 5703] | 17.87 [1.833, 161.5] | 0.8696 -> 0.8042 | 0.00662 -> 0 | 0 |
| origins | isqp_none | 27 | 26/27 | 26/27 | 4.348 [0.3431, 19.02] | 1075 [23.75, 8891] | 18.12 [1.971, 194] | 0.8696 -> 0.7981 | 0.00662 -> 0 | 0 |

Box load at start: 2% (near-idle; see Walls). `isqp_l2` certifies 27/27; `isqp_none`'s single
non-certified, non-feasible row is `m2_ffd_brainpair_coarse` — a 39.5%-folded input where pure
feasibility plateaus a handful of bilinear cells a few 1e-4 short of the threshold (worst residual
about -5.1e-4) while the in-solve L2 objective (`isqp_l2`) clears it; this is the paper's per-case
wall column for `origins` (see Walls for the serial-vs-throughput inflation on this same pair).

## Results — ANTs controls

<!-- pasted verbatim from ants/table.md, including its legend comment -->

<!-- certificate gauges: simplex: 2 triangles per cell (fixed BL-TR diagonal), triangle area = det/2, per cell (last row/col are +inf); bilinear: 4 triangles per cell (both diagonals), triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2, per cell (last row/col are +inf); finite: forward-difference Jdet (1 triangle per cell), determinant, per cell (last row/col are +inf); jdet: central-difference Jdet, determinant, per pixel. certified = bilinear has 0 values < 0.01 - 1e-5 after. -1 is a sentinel (see summary.json notes), skipped by every median. -->
| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |
|---|---|---|---|---|---|---|---|---|---|---|
| ants | isqp_none | 85 | 85/85 | 85/85 | 0.4254 [0.3573, 0.5056] | 0 [0, 0] | 0 [0, 0] | 0.1395 -> 0.1395 | 0 -> 0 | 0 |

85 slices (7 brains × 11 sampled slices + 8 named hard slices), all certified, none with any
bilinear fold on input, and no pixel moved — the engine leaves an already-injective warp untouched.
Box load at start: 2% (near-idle). See the Caveats note on ANTs slice-index reversal before pairing
these rows with `cohort_*` rows by z.

## Results — cohort sample (7 brains, Laplacian-exterior)

<!-- pasted verbatim from cohort/table.md, including its legend comment -->

<!-- certificate gauges: simplex: 2 triangles per cell (fixed BL-TR diagonal), triangle area = det/2, per cell (last row/col are +inf); bilinear: 4 triangles per cell (both diagonals), triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2, per cell (last row/col are +inf); finite: forward-difference Jdet (1 triangle per cell), determinant, per cell (last row/col are +inf); jdet: central-difference Jdet, determinant, per pixel. certified = bilinear has 0 values < 0.01 - 1e-5 after. -1 is a sentinel (see summary.json notes), skipped by every median. -->
| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |
|---|---|---|---|---|---|---|---|---|---|---|
| cohort | isqp_l2 | 85 | 85/85 | 85/85 | 209.6 [164, 391.6] | 1037 [752, 2052] | 36.42 [29.87, 59.4] | 0.357 -> 0.1861 | 0.00214 -> 0 | 0 |
| cohort | isqp_none | 85 | 85/85 | 85/85 | 86.45 [58.76, 176.1] | 1511 [1094, 3539] | 44.32 [36.49, 75.61] | 0.357 -> 0.1706 | 0.00214 -> 0 | 0 |

### Hard slices

One row per slice × config, read directly from `cohort/results.csv` (not from the ledger).

| slice | config | certified | input bilinear folds | worst bilinear after | wall s | L2 move | moved frac | damage | corr n | corr resid median before -> after (px) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0039 z1 | isqp_l2 | yes | 3975 | 0.01046 | 1155.1 | 2038.2 | 0.0856 | 0 | 20 | 1.56 -> 132.7 |
| B0039 z1 | isqp_none | yes | 3975 | 0.01085 | 738.21 | 2416.9 | 0.086 | 0 | 20 | 1.56 -> 135.1 |
| B0039 z2 | isqp_l2 | yes | 3935 | 0.01053 | 1009.8 | 1977.8 | 0.0837 | 0 | 17 | 1.51 -> 124 |
| B0039 z2 | isqp_none | yes | 3935 | 0.01068 | 1338.9 | 2341.1 | 0.088 | 0 | 17 | 1.51 -> 127.2 |
| B0039 z11 | isqp_l2 | yes | 3135 | 0.01052 | 919.55 | 566.05 | 0.0945 | 0 | 131 | 0.153 -> 8.01 |
| B0039 z11 | isqp_none | yes | 3135 | 0.01088 | 230.01 | 736.93 | 0.0944 | 0 | 131 | 0.153 -> 7.861 |
| B0039 z16 | isqp_l2 | yes | 2203 | 0.01023 | 207.63 | 189.6 | 0.117 | 0 | 287 | 0.0356 -> 1.811 |
| B0039 z16 | isqp_none | yes | 2203 | 0.011 | 85.961 | 227.68 | 0.117 | 0 | 287 | 0.0356 -> 2.348 |
| B0039 z264 | isqp_l2 | yes | 1244 | 0.01 | 177.16 | 36.788 | 0.26 | 0 | 1127 | 0.00864 -> 0.04168 |
| B0039 z264 | isqp_none | yes | 1244 | 0.011 | 70.812 | 45.953 | 0.291 | 0 | 1127 | 0.00864 -> 0.06633 |
| B0032 z1 | isqp_l2 | yes | 4641 | 0.01053 | 876.73 | 1575.3 | 0.138 | 0 | 132 | 0.961 -> 77.51 |
| B0032 z1 | isqp_none | yes | 4641 | 0.01089 | 475.86 | 1817.8 | 0.141 | 0 | 132 | 0.961 -> 78.85 |
| B0304 z128 | isqp_l2 | yes | 9139 | 0.01086 | 1543.1 | 1091.5 | 0.507 | 0 | 461 | 0.0308 -> 20.44 |
| B0304 z128 | isqp_none | yes | 9139 | 0.011 | 1031.1 | 1418.5 | 0.506 | 0 | 461 | 0.0308 -> 19.19 |
| B0304 z181 | isqp_l2 | yes | 34359 | 0.01077 | 2497 | 882.87 | 0.67 | 0 | 497 | 0.486 -> 5.075 |
| B0304 z181 | isqp_none | yes | 34359 | 0.011 | 2398.2 | 1109.5 | 0.683 | 0 | 497 | 0.486 -> 5.278 |

### Findings

All 170 rows certified: 85 slices x {`isqp_none`, `isqp_l2`}, 0 error rows, damage 0 everywhere.
81 of the 85 slices folded on input. 75 of the 85 slices carry correspondences (`corr_n > 0`). All
8 named hard slices (B0039 z1, z2, z11, z16, z264; B0032 z1; B0304 z128, z181) certify under both
configs — see the Hard slices table above.

**Fidelity cost.** On the pin-collapsed volume-edge slices, certification is reached by moving the
field far from the registration's own landmarks. The median correspondence residual (px, at the
prescribed Laplacian boundary correspondences) rises from about 1.5 px to 124-135 px on B0039 z1
and z2, and from 0.96 px to about 78 px on B0032 z1: exactly, B0039 z1 1.56 -> 132.7 (`isqp_l2`) /
135.1 (`isqp_none`) px; B0039 z2 1.51 -> 124.0 (`isqp_l2`) / 127.2 (`isqp_none`) px; B0032 z1 0.96
-> 77.5 (`isqp_l2`) / 78.9 (`isqp_none`) px. It stays under about 2 px on the ordinary slices —
e.g. B0039 z16 0.0356 -> 1.81 (`isqp_l2`) / 2.35 (`isqp_none`) px, and B0039 z264 0.0086 -> 0.042
(`isqp_l2`) / 0.066 (`isqp_none`) px.

This was verified directly on B0039 z1 x `isqp_l2` from the saved corrected field
(`data/dvfs/results/cohort_isqp/cohort/cohort_B0039_z1__isqp_l2.npz`): all 20 landmarks lie on
moved pixels, and the median displacement there is 134.2 px, while only 8.6% of the slice moved.

The certificate therefore is not a fidelity claim on these slices, and the residual must be read
beside it.
