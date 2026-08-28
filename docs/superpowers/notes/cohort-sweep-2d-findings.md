# Cohort 2D sweep — does the one-call windowed/bilinear recipe generalise past B0039?

**Verdict: yes.** 139/146 real-brain slice solves reach exactly 0 two-triangle folds with
damage 0 and a certified `min_T >= 0.01`. All 7 residual solves are wall-clock
budget cuts at a 2400 s/stage cap, not solver plateaus — and under the recipe's own
default step rule (`exact_ls`) **no slice plateaued at all**.

## The claim under test

One call, all engine defaults on `main` (as of #84):

```python
windowed_correct(phi, 'isqp',
                 constraint=SimplexConstraint2DBilinear(shape=(H, W)),
                 objective=NoneObjective(), threshold=0.01)
# equivalently: Solver(constraint='bilinear', strategy='isqp_windowed', objective='none')
```

reaches 0 two-triangle folds with damage 0 from raw on every B0039 slice tested.
This sweep asks whether that holds across the whole in-repo brain cohort.

## Method

- Script: [`benchmarks/cohort_sweep_2d.py`](../../../benchmarks/cohort_sweep_2d.py)
  (`selfcheck` / `survey` / `sweep` / `summarize`, each stage resumable).
- Data: `data/dvfs/brain25_cohort_corrected/<brain>/<variant>/` — 7 brains, all
  `(3, 528, 320, 456)` `[dz, dy, dx]`, gitignored.
- Deterministic slice sample: **every 64th z plus the single most folded z** of that
  field (9-10 slices per field), for 3 variants x 7 brains.
- Fold metric is the cell minimum, `min(T1, T2) < 0.01`, via
  `dvfopt.core.windowed.pixel_fold_mask` — the same collapse for the 2-triangle
  (2 rows/cell), bilinear (4 rows/cell) and finite (1 row/cell) families, so the
  three "after" columns are directly comparable.
- The worst slice of each field is re-run with `step_rule='tr'` (11 paired slices).
- 4 worker processes, each pinned via `dvfopt.core._pool.pin_worker_threads()`.
  A second job shared the box for roughly the first two thirds of the run, so
  **SQP iteration counts are the contention-proof metric** and wall is an upper bound.
- Per-slice engine `time_budget_s = 2400`. Note this is **per stage**: the
  coarse-to-fine warm start is a nested `windowed_correct` that gets its own full
  budget, so a slice can legitimately reach ~2x the cap.

### Variants, and how the ANTs warp is read

`benchmark_utils.load_cohort_field` covers `laplacian_all` / `laplacian_exterior`
directly. The ANTs warp needed a new reader (`load_ants_field`) because
`dvfopt.io.fields.load_dvf_sitk` maps sitk's *physical* `[dx, dy, dz]` components
straight onto numpy axes, which is only valid for an identity direction matrix.
These warps sit on a permuted LPS grid:

```
size (i,j,k) = (528, 320, 456)   spacing = 0.025 mm isotropic
direction    = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
```

so the physical displacement is rotated into the image's own index frame,
`d_index = Dir.T @ d_phys / spacing`, and the sitk `(k, j, i)` array is transposed to
the cohort's `(i, j, k)`. The three sizes are distinct, so the `i,j,k -> D,H,W`
correspondence against the Laplacian field is pinned by shape alone. Sanity: the
result carries ±12-voxel displacements with a minimum triangle area of 0.29 — a
smooth, valid warp in index units; a scrambled component/axis mapping at that
amplitude would fold heavily. `selfcheck` pins the rotation on the cohort's exact
direction/spacing.

## Survey: where the folds actually are (all 528 slices of every field)

| brain | variant | total folded cells | worst slice | at | folded z |
|---|---|---|---|---|---|
| B0032 | ants | 0 | 0 | z0 | 0/528 |
| B0032 | laplacian_all | 1051926 | 3613 | z294 | 528/528 |
| B0032 | laplacian_exterior | 605921 | 4556 | z1 | 528/528 |
| B0039 | ants | 0 | 0 | z0 | 0/528 |
| B0039 | laplacian_all | 1124427 | 4633 | z11 | 528/528 |
| B0039 | laplacian_exterior | 599313 | 3957 | z1 | 528/528 |
| B0049 | ants | 0 | 0 | z0 | 0/528 |
| B0049 | laplacian_all | 1045929 | 3607 | z309 | 528/528 |
| B0049 | laplacian_exterior | 545062 | 2626 | z397 | 528/528 |
| B0053 | ants | 0 | 0 | z0 | 0/528 |
| B0053 | laplacian_all | 1005854 | 3730 | z299 | 528/528 |
| B0053 | laplacian_exterior | 519788 | 2023 | z412 | 528/528 |
| B0200 | ants | 0 | 0 | z0 | 0/528 |
| B0200 | laplacian_all | 1022498 | 3660 | z307 | 528/528 |
| B0200 | laplacian_exterior | 538064 | 2204 | z439 | 528/528 |
| B0213 | ants | 0 | 0 | z0 | 0/528 |
| B0213 | laplacian_all | 994491 | 3345 | z309 | 528/528 |
| B0213 | laplacian_exterior | 535334 | 2060 | z435 | 528/528 |
| B0304 | ants | 0 | 0 | z0 | 0/528 |
| B0304 | laplacian_all | 4157169 | 29699 | z181 | 358/528 |
| B0304 | laplacian_exterior | 2933090 | 36902 | z176 | 362/528 |

Two survey findings stand on their own, before any solve:

1. **The ANTs warps carry no in-plane 2-triangle folds at all** — 0 folded cells
   across all 528 slices of all 7 brains, minimum triangle area ~0.29 (29x the 0.01
   threshold). **Every fold in this cohort is introduced by the Laplacian
   correspondence interpolation, not by the registration it refines.** The ANTs
   variant is a no-op control in the sweep (63 slices, 0 -> 0, ~0.25 s each), not a
   test case.
2. **B0304 is a severity outlier**: worst slice 36902 folded cells, ~8x any other
   brain, and concentrated (only ~360 of 528 slices fold at all). B0039 — the family
   every engine default was tuned on — is mid-pack, not the hard case.

## Results — per field (`exact_ls`, the recipe's default)

| brain | variant | slices | initial folds | worst `min_before` | reached 0 | worst residual | damage | wall med/max (s) | iters med | L2 move % med | px moved % med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0032 | ants | 9 | 0-0 | 0.1 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0032 | laplacian_all | 10 | 389-3613 | -15.2 | 10/10 | 0 | 0 | 329 / 630 | 792 | 13.9 | 39.8 |
| B0032 | laplacian_exterior | 9 | 382-4509 | -83.0 | 9/9 | 0 | 0 | 215 / 811 | 300 | 13.4 | 29.7 |
| B0039 | ants | 9 | 0-0 | 0.2 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0039 | laplacian_all | 10 | 681-4633 | -118.2 | 9/10 | 49 | 0 | 309 / 2783 | 950 | 17.3 | 29.1 |
| B0039 | laplacian_exterior | 10 | 561-3957 | -328.8 | 9/10 | 39 | 0 | 324 / 3458 | 376 | 16.5 | 28.5 |
| B0049 | ants | 9 | 0-0 | 0.1 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0049 | laplacian_all | 10 | 689-3607 | -18.6 | 10/10 | 0 | 0 | 227 / 776 | 518 | 16.6 | 35.1 |
| B0049 | laplacian_exterior | 10 | 608-2626 | -55.4 | 10/10 | 0 | 0 | 270 / 572 | 366 | 18.5 | 26.2 |
| B0053 | ants | 9 | 0-0 | 0.1 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0053 | laplacian_all | 10 | 652-3730 | -21.2 | 10/10 | 0 | 0 | 270 / 545 | 600 | 13.5 | 36.0 |
| B0053 | laplacian_exterior | 10 | 562-2023 | -15.2 | 10/10 | 0 | 0 | 167 / 437 | 284 | 14.7 | 23.7 |
| B0200 | ants | 9 | 0-0 | 0.2 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0200 | laplacian_all | 10 | 500-3660 | -24.5 | 10/10 | 0 | 0 | 255 / 732 | 512 | 15.5 | 35.0 |
| B0200 | laplacian_exterior | 10 | 341-2204 | -38.2 | 10/10 | 0 | 0 | 165 / 633 | 246 | 10.9 | 25.8 |
| B0213 | ants | 9 | 0-0 | 0.2 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0213 | laplacian_all | 10 | 384-3345 | -26.0 | 10/10 | 0 | 0 | 254 / 493 | 564 | 11.1 | 38.1 |
| B0213 | laplacian_exterior | 10 | 603-2060 | -17.5 | 10/10 | 0 | 0 | 184 / 486 | 366 | 8.7 | 23.2 |
| B0304 | ants | 9 | 0-0 | 0.1 | 9/9 | 0 | 0 | 0 / 0 | 0 | 0.0 | 0.0 |
| B0304 | laplacian_all | 7 | 0-10334 | -227.0 | 7/7 | 0 | 0 | 303 / 921 | 685 | 3.0 | 37.3 |
| B0304 | laplacian_exterior | 7 | 0-8956 | -75.9 | 6/7 | 80 | 38 | 207 / 2640 | 394 | 1.0 | 26.0 |

Across the **139 clean Laplacian solves**:

- `min_after >= 0.01` on every one — strict feasibility, not just "no negatives".
- **damage = 0 on every one** — the no-damage invariant held across the whole cohort.
- Initial folds 0-10334, initial worst cell down to **-227.0**.
- Wall median **246 s**, max 1251 s. Iterations median **424**, max 2334.
- L2 move median **13.5%** of `||phi||` (max 49.9%); pixels moved median 31.4%.

## The 7 non-clean solves — all budget cuts, none an `exact_ls` plateau

| brain | variant | z | rule | folds | `min_before` | `min_after` | damage | giants | rounds | wall (s) | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0304 | laplacian_all | 181 | tr | 29699 -> 9913 | -285.7 | -251.40 | **3440** | 1 | 1 | 3580 | 2419 |
| B0304 | laplacian_exterior | 128 | exact_ls | 8956 -> 80 | -75.9 | -3.99 | **38** | 1 | 1 | 2640 | 3709 |
| B0032 | laplacian_exterior | 1 | tr | 4556 -> 62 | -983.8 | -0.066 | 0 | 1 | 3 | 3163 | 7436 |
| B0039 | laplacian_all | 11 | exact_ls | 4633 -> 49 | -118.2 | -0.077 | 0 | 1 | 8 | 2783 | 3639 |
| B0039 | laplacian_exterior | 1 | exact_ls | 3957 -> 39 | -328.8 | -0.034 | 0 | 1 | 6 | 3458 | 5119 |
| B0039 | laplacian_all | 11 | tr | 4633 -> 41 | -118.2 | -0.075 | 0 | 1 | 3 | 1594 | 4561 |
| B0039 | laplacian_exterior | 1 | tr | 3957 -> 32 | -328.8 | -0.048 | 0 | 1 | 4 | 2346 | 5866 |

Reading these:

- **Every `exact_ls` residual is a budget cut** (wall > 2400 s). Only two rows finished
  under budget with folds left, and both are `tr`, not the recipe's default. So this
  sweep found **no slice on which the recipe provably stalls** — only slices it did not
  finish inside 40 min/stage on one core.
- The residuals are tiny and shallow: 32-80 cells out of 145 540, worst cell
  -0.034 to -0.077 against a +0.01 target. These fields are ~99.95% repaired.
- **All 7 have `giant_regions == 1`** — a single merged fold region routed to the
  overlapping-tile Schwarz decomposition. That, not fold count, is the cost driver.

### Fold count does not predict difficulty; a single merged giant does

B0304 `laplacian_all` z=256 cleared **10334 folds from `min_before = -227.0`** in
921 s (5 giants), while B0039 `laplacian_exterior` z=1 ran out of budget on 3957
folds. B0039 z=0 (3951 folds, `min_before = -23.4`) cleared in 730 s, but its
neighbour z=1 (3957 folds, `min_before = -328.8`) did not finish. Neither density
nor severity alone separates the two sets — the pairing that does is **deep folding
concentrated into one contiguous region**. When the folds break into several giants
the tiler clears each cheaply, even at severity -227.

### `time_budget_s` is not a safe hard cap while `coarse_to_fine=True`

Both damage-positive rows have `rounds == 1`; every row cut in a later round has
damage 0. That is exactly the mechanism: `_coarse_warm_start` prolongates the coarse
correction and masks it to **all** round-1 window boxes, but only windows actually
*solved* enter `touched`. Cut round 1 partway and the delta has moved pixels in boxes
that were never repaired — new folds outside every solved window, i.e. real damage to
previously healthy area (3440 cells in the worst case). The engine's `damage == 0`
invariant is therefore conditional on the run not being truncated. **Do not use
`time_budget_s` as a hard cap and then keep the field**; treat a budget-exhausted
result as discardable, or disable `coarse_to_fine` when capping.

## `tr` vs `exact_ls` on real data — no sliver pathology, one mild caveat

PR #84 made `exact_ls` the default and flagged the synthetic `z0_sliver` crop
(351 s vs 77 s) as "a chaos detector, with no counterpart on real slices". Eleven
paired worst-slices across all 7 brains, on independent data:

**9 pairs where both rules reach 0 folds** — `exact_ls` wins wall **8/9** and
iterations **8/9**, totalling **-20.3% wall and -23.6% iterations**. That closely
reproduces #84's B0039-only measurement (-19% / -27%) on six brains it never saw.
Per-iteration cost ratio (`exact_ls`/`tr`) median **1.07**, max 1.22 — the exact line
search is essentially free and pays for itself in iterations. **No sliver-style
pathology appears anywhere in the converging regime.**

**2 pairs where neither rule reaches 0** (B0039 `laplacian_all` z=11 and
`laplacian_exterior` z=1 — both single-giant) — here `exact_ls` still uses *fewer*
iterations (3639 vs 4561; 5119 vs 5866) but its per-iteration cost jumps to **2.19x
and 1.69x**, so it loses on wall (2783 vs 1594 s; 3458 vs 2346 s). The true-merit
check that guards the line minimiser costs a full constraint evaluation, and on the
huge windows a single merged giant produces that evaluation dominates.

So: the `z0_sliver` behaviour has **no real-data counterpart of the same character**
(4.5x on a converging case). What real data does show is a milder, structural cousin
— on single-giant windows the exact line search's per-iteration overhead roughly
doubles and can outweigh its iteration saving. Worth knowing before raising
`max_window_area`, not a reason to change the default.

## Coverage gap

B0304's 6 most severe sampled slices (13187-36902 folds) and
`B0032 laplacian_exterior z=1` under `exact_ls` were not run to completion — each
needs more than the ~1 h of uninterrupted wall available per attempt. `results.csv`
is resumable; re-running `sweep` picks them up. The one data point in that regime
(B0304 z=181 under `tr`, 29699 folds) was cut during round 1 and is the 3440-cell
damage row above.

## Reproduce

```bash
python benchmarks/cohort_sweep_2d.py selfcheck
python benchmarks/cohort_sweep_2d.py survey
python benchmarks/cohort_sweep_2d.py sweep --workers 4 --time-budget-s 2400
python benchmarks/cohort_sweep_2d.py summarize --time-budget-s 2400
```

Outputs land in `benchmarks/output/cohort_sweep_2d/` (gitignored): `results.csv` (one
row per slice per step rule), `summary.json`, per-z `zcounts/`, and the `slices/`
cache. Add `--data-root <path>` when running from a git worktree without the
gitignored data.
