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
- **Seven configs** (`benchmarks/canonical_2d.py`'s `CONFIGS`):

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
- **The three hard B0039 crops (`z0_sliver`, `z0_cluster`, `z16_twist`) are the engine's TUNING
  set** — every table below says so, and their numbers are never quoted as a held-out result.
- **Nothing is dropped:** a case that fails to load or a solve that raises is still a full row
  (`feasible=False`, `-1` metrics, the error text) in `results.csv`, in `manifest.json`, and in
  every aggregate denominator; only the median/IQR *distributions* skip sentinel rows (see Metric
  definitions). A run exceeding its 2-hour soft cap (`--cap-s`, default 7200 s) is recorded as
  `hit_cap=True` but is **not** interrupted — `time_budget_s` is left `None` so the engine always
  runs to its own termination.
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
  `min_jdet_init`/`min_jdet_final`, `l2_err`, `time_s`.
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
  phase is nested inside its round entry, so `total_iter` would double-count it). **`-1` on every
  other strategy** (`barrier`, `slp`, `m14`, `slsqp_windowed`) — the other strategies report phases
  in their own units (L-BFGS iterations, named SLP stages with `n_iter=0`), and summing those under
  the same column name would mix three different quantities. `slsqp_windowed` is
  `SLSQPWindowedStrategy`, a different class from the windowed I-SLSQP engine — the driver's
  `WINDOWED_STRATEGY_NAMES = ("ISQPWindowedStrategy", "WindowedWrapperStrategy")` never names it —
  so its rows carry `-1` here unconditionally, not "when it did not resolve to the windowed engine".
  `-1` is always a sentinel, never a measurement: the median/IQR aggregates
  (`_quantiles`) skip rows where the value is negative, but the count/rate denominators (`n`,
  `feasible_rate`, `certified_rate`) still include every row, including full-sentinel rows for a
  case that never produced a result at all (load failure or a dead pool worker).
- **Move and locality:** `moved_frac` (fraction of pixels with any channel changed by more than
  `1e-9`), `l1_move` (`sum(|phi_out - phi_in|)`), `l2_move` (`norm(phi_out - phi_in)`), `max_move`
  (`max(|phi_out - phi_in|)`), `mean_move_moved` (`l1_move` divided by the count of moved pixels).
- **Correspondence fidelity — cohort rows only:** `corr_n`, `corr_resid_med_init/final` (median
  registration residual at the prescribed Laplacian boundary correspondences),
  `corr_resid_mad_init/final` (MAD, scaled by 1.4826), and the outlier counts
  `corr_n_outliers`/`corr_n_large`/`corr_n_high_resid`/`corr_n_incoherent` from
  `correspondence_analysis.analyze_slice`. **`-1` on every non-cohort row.**
- **Injectivity diagnostics (optional, cheap):** `ift_min_radius_<init|final>` and
  `ift_frac_subpixel_<init|final>` from `dvfopt.metrics.injectivity_stats` — the quantitative-IFT
  radius **estimate**, orientation-blind and never a certificate; read beside the four certificate
  gauges, never instead of them.
- **Provenance, per run directory** (`summary.json`'s `provenance` block): git commit,
  `dvfopt.__version__`, the config dict, `n_workers`, python + OS, box CPU load at the start of the
  run, `time_budget_s` (always `null` — the engine is never given a soft budget internally),
  `cap_s` (the driver's own recorded-not-enforced cap), and `hit_cap` per row.

## Walls

The box was **not idle** during this session's runs — two unrelated user jobs (one already
running for 70+ CPU-hours, a second started mid-session) held CPU load between 24% and 76%
throughout. Every wall (`time_s`) in `crops/` and `synthetic/` is a **throughput** pass
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
- **Reproducibility across timing modes is discrete, not bit-identical.** Two runs pinned to the
  same BLAS threading (the two throughput passes of `m2_ffd_brainpair_coarse` × `isqp_none`, before
  and after a machine restart) agree bit-for-bit in all 16 outcome fields. Across timing modes
  (serial vs throughput) on the same pair the *discrete* outcome is identical — certified, feasible,
  fold counts under every gauge, damage, windows, rounds, SQP iterations — while continuous outputs
  (L1 / L2 / max move) differ by about 3e-5 relative and the worst residual value by about 2.5 %.
  Most plausibly this is because pool workers are pinned to one BLAS thread and the in-process
  serial path is not (supported by the pinned-vs-pinned bit-identical check above, not proven
  further). **Never describe results as bit-identical across timing modes** — only within one.

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
└── ants/                       # FINAL — the ANTs "already injective" control
    ├── results.csv
    ├── summary.json
    ├── manifest.json
    └── table.md                # no figures/: this run was not passed --figures
```

One subdirectory per source, matching the run-dir names in
`benchmarks/output/2d_canonical/<row>/` (gitignored) — `ants/` here holds `ants_isqp/`'s output.
`report/` (the `cohort_benchmark`-shared `report.html` + its own filtered CSV/JSON copy) is **not**
copied here — it duplicates `results.csv`/`summary.json` at a filtered column set and is meant to
be regenerated locally, not tracked. Corrected DVFs are never tracked: they live under
`data/dvfs/results/<run-name>/<source>/<case>__<config>.npz` (gitignored), and each source's
`manifest.json` here lists every one with its case, config, shape and sha256, plus the untouched
*input* path (inputs are never duplicated).

Pending sources (`origins` full taxonomy, `cohort`) will each get their own subdirectory here, in
the same shape, once their chains finish — see Status.

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

# cohort sample, the two engine configs, throughput pass — PENDING
python benchmarks/canonical_2d.py --source cohort \
    --config isqp_none isqp_l2 --n-workers 4 --table \
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

- **`crops`, `synthetic`, `origins` (serial-timing pass), `ants`: FINAL.** Tracked here in full
  (see the tables below).
- **`origins` (full 7-config taxonomy, throughput): being recovered.** The original
  `origins_all` run lost 130 of its 189 pairs to a dead pool worker
  (`BrokenProcessPool`) mid-run; a recovery run reuses the 59 rows it did measure and reruns
  exactly the 130 losses, with `slsqp_windowed` pairs isolated one at a time. It replaces
  `origins_all`, whose aggregates (`summary.json`/`table.md`, computed over the 130 sentinel rows)
  must not be used.
- **`cohort`: running.** The cohort sample is still solving as of this commit; its subdirectory,
  the findings-note section, the CHANGELOG entry and the CLAUDE.md benchmarks-bullet update are
  appended by a later commit once it and the origins recovery finish.

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
Box load at start: 0% (near-idle). See the Caveats note on ANTs slice-index reversal before pairing
these rows with `cohort_*` rows by z.
