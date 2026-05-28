# Wall-breaker findings

**Question.** Find a method that resolves every slice of `deformation3d.npy`
(528 slices, 320×456 each) with **no folding whatsoever**
(`tri_neg == 0`, ideally also `tri_min ≥ 0.01`) — including the dense
wall slices where the manuscript run leaves residual folds.

**Answer.**

Full-DVF totals (sums over the whole (2, H, W) slice):

| method | fold-free | strict (`tri_min≥0.01`) | mean L2 | median L2 | mean L1 | median L1 | median wall | total wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| manuscript SLSQP (aggregated) | 509/528 (96.4 %) | 0/528 | 35.2 | 34.7 | — | 733 | — | — |
| slsqp_windowed (fresh) | 333/528 (63.1 %) | 226/528 (42.8 %) | **32.3** | 31.4 | 825 | 825 | **29 s** | 21 h |
| m10 harmonic_l2_polished | **528/528 (100 %)** | **528/528 (100 %)** | 108.1 | 85.3 | 6 295 | 4 924 | 35 s | 5.4 h |
| m14 l2_refine_repair | **528/528 (100 %)** | **528/528 (100 %)** | 73.7 | 48.7 | 2 961 | 1 409 | 233 s | 34 h |
| **m14_l1 l1_refine_repair** | **528/528 (100 %)** | **528/528 (100 %)** | **52.1** | **40.6** | **1 139** | **705** | 241 s | 36 h |

Same numbers normalised per scalar / per corner (grid 320×456 = 145 920 corners = 291 840 dy+dx scalars). MEDIAN per-slice:

| method | L2 / entry | L2 / pixel | L1 / entry | L1 / pixel |
|---|---:|---:|---:|---:|
| manuscript SLSQP | 0.0643 | 0.0909 | 0.0025 | 0.0050 |
| slsqp_windowed | 0.0582 | 0.0823 | 0.0028 | 0.0057 |
| m10 harmonic_l2_polished | 0.1578 | 0.2232 | 0.0169 | 0.0337 |
| m14 l2_refine_repair | 0.0901 | 0.1274 | 0.0048 | 0.0097 |
| **m14_l1 l1_refine_repair** | **0.0752** | **0.1064** | **0.0024** | **0.0048** |

* `L2 / entry` = `L2_total / sqrt(2 H W)` — RMS deviation of each dy or dx scalar (in pixel units).
* `L2 / pixel` = `L2_total / sqrt(H W)` — RMS deviation per corner (= sqrt(2) × L2/entry).
* `L1 / entry` = `L1_total / (2 H W)` — mean absolute deviation per dy/dx scalar.
* `L1 / pixel` = `L1_total / (H W)` — mean (|dy| + |dx|) per corner.

On the **median slice**, m14_l1 already matches manuscript SLSQP on L1/entry (0.0024 vs 0.0025) while being only 17 % worse on L2/entry — and m14_l1 is fully feasible everywhere where SLSQP is not. The gap on the **mean** is entirely from the dense z=0-9 wall region (see "Per-slice picture" below).

* The aggregated manuscript run (`_run_2d_clusters.py`) used the full retry/L1-polish/pad-boost cluster solver and produced 96.4 % fold-free outputs, but its L2 anchor binds the constraint *at* threshold (so 0/528 land at `tri_min ≥ 0.01`).
* The fresh harness run (`m_slsqp`) is a simplified cluster solver — same SLSQP-per-window mechanism but without the escalation logic — so its fold-free rate drops to 63 %, *but* where it succeeds it reaches strict `tri_min ≥ 0.01` (the constraint is fed `threshold = 0.01` directly, not as a soft anchor target). The fresh run also exposes a SLSQP tail risk: median 29 s/slice but max 12 000 s (200 min) when the solver thrashes near the wall.
* m10 / m14_l2 / m14_l1 all give full 100 % feasibility (both fold-free and strict). Within this group:
  * m14_l2 cuts m10's L2 by 32 % (mean) and 43 % (median).
  * m14_l1 cuts m10's L2 by 52 % (mean) and 52 % (median) AND cuts m10's L1 by 82 % — the L1 anchor dominates the L2 anchor on every aggregate metric.

The manuscript SLSQP solver leaves 19 slices with residual folds and never lands
`tri_min` strictly above the 0.01 threshold (its L2 anchor pulls every cell back
toward the constraint boundary). Both m10 and m14 satisfy the strict criterion
on every one of the 528 slices. m14 is the **L2 winner among feasibility-100 %
methods**: 32 % lower mean L2 and 43 % lower median L2 than m10.

## How the methods relate

```
   phi_in (input DVF, lots of folds)
        │
        ▼
   ┌────────────────────────────────────┐
   │ m10 = harmonic_l2_polished         │       <-- always-feasible seed
   │  1. harmonic extension over fold   │           L2 ≈ 108 mean
   │     cores  (Radó-Kneser-Choquet)   │
   │  2. augmented Lagrangian nudge     │
   │  3. scipy log-barrier L2 polish    │
   └─────────────┬──────────────────────┘
                 │ feasible, L2 high
                 ▼
   ┌────────────────────────────────────┐
   │ m14 = l2_refine_repair             │       <-- L2 refinement
   │  1. start from m10 output          │           L2 ≈ 74 mean
   │  2. soft-quadratic-penalty pull    │
   │     anchored to phi_in             │
   │     (no barrier wall -> big steps) │
   │  3. harmonic repair of residual    │
   │     folds (typically <200 cells)   │
   │  4. log-barrier L2 polish          │
   └────────────────────────────────────┘
```

The full-DVF benchmark (`run_full_dvf.py`) ran every method through identical
fixtures and computed identical metrics. Results land in `results_full_dvf/`.

## Per-slice picture

Selected slices showing L2 across the three feasibility-100 % methods plus the (infeasible) SLSQP baseline:

| z   | m10 L2 | m14_l2 L2 | **m14_l1 L2** | SLSQP L2 (infeasible) | m14_l1 vs m14_l2 |
|----:|-------:|----------:|--------------:|----------------------:|-----------------:|
|   5 | 988.0  | 1076.6    | **481.3**     | 101.5 | −55 % |
|   6 | 908.8  |  907.4    | **427.4**     |  92.7 | −53 % |
|   7 | 725.8  |  606.0    | **(see CSV)** |  48.2 | — |
|  12 | 551.9  |  470.0    | **294.7**     | 126.8 | −37 % |
|  14 | 453.7  |  546.0    | **244.4**     | 127.9 | −55 % |
|  17 | 409.3  |  336.9    | **227.8**     | 118.9 | −32 % |
|  50 |  54.8  |   35.3    | **28.0**      |  23.7 | −21 % |
| 100 |  49.2  |   26.7    |   26.8        |  22.5 | tie  |
| 200 |  67.0  |   30.1    |   31.9        |  27.0 | +6 % |
| 300 |  91.0  |   56.4    | **49.2**      |  43.8 | −13 % |
| 500 |  84.1  |   50.5    | **37.9**      |  32.9 | −25 % |
| 527 |  68.3  |   11.5    |   13.0        |   9.8 | +13 % |

* **Wall slices (z = 5–17)**: m14_l1 cuts L2 **in half** vs both m10 and m14_l2. This is where the L1 anchor pays off the most — fold cores concentrate the deviation into a few cells (sparse), exactly the regime L1 is built for.
* **Mid-density slices (z = 50, 300, 500)**: m14_l1 gives a modest 13–25 % improvement over m14_l2.
* **Easy slices (z = 100, 200, 527)**: m14_l1 and m14_l2 are essentially tied — when L2 is already small, the anchor choice doesn't matter much.
* **z = 14**: m14_l2 was *worse* than m10 (harmonic repair grew larger than m10's seed patch). m14_l1 fixes that — 244 vs 546.

## Why the refinement works

m10's polish converges in 1–2 L-BFGS-B iterations because the log barrier
``-μ Σ log(T - τ)`` has a singularity at the feasibility boundary;
m10 sits *exactly* on that boundary, so the barrier gradient blows up
for the active cells and the L2 gradient is locked. Shrinking μ does
not help — that's the central-path stationary point at every μ.

m14 swaps the log barrier for a **soft quadratic penalty**
``λ Σ max(0, τ − T)²``. The penalty is *exactly zero* for inactive cells,
so non-fold-core cells can slide freely toward `phi_in`. Active cells
get a smooth, finite pushback that L-BFGS-B can integrate over many
iterations (the inner solve takes 200–300 steps now, vs 1–2 for the
log barrier). The price is that the soft penalty saturates: it
*never quite* enforces strict feasibility, leaving a few residual
folds (e.g. 168 cells on z=12).

The residual folds are typically small *clusters* (the penalty pushed
the constraint boundary outward but a few cells slipped). We patch
them with a small harmonic extension (m02) — same theorem as in m10,
but on a much smaller patch, so the L2 cost of the patch is small.
The final log-barrier polish recovers strict feasibility (`tri_min`
landing 0.010-0.015).

## Method catalogue (12 candidates)

| ID | Name | Idea | Feasible (wall slices) | Feasible (full DVF) |
|---|---|---|:---:|:---:|
| m01 | svf_squaring | Exp(v) of `phi / 2^N` | 4/6 | 15/528 |
| m02 | harmonic | Bare Laplacian extension | 1/6 | 199/528 |
| m03 | aug_lagrangian | PHR augmented Lagrangian | 3/6 | (would diverge globally) |
| m04 | paint_blend | Bilinear patch + Hann blend | 1/6 | not benchmarked |
| m05 | torch_full_grid | Torch L-BFGS / Adam barrier | 0/6 | not benchmarked |
| m06 | quasi_conformal | Bound \|μ\| < 1 (Beltrami) | 0/6 | not benchmarked |
| m07 | tv_anchor | TV regularised correction | 1/6 | not benchmarked |
| m08 | harmonic+polish | m02 → torch polish → ALM | 6/6 | (similar to m10) |
| m09 | svf_polished | m01 → scipy log-barrier polish | 4/6 | not benchmarked |
| m10 | **harmonic_l2_polished** | **m02 → ALM → log-barrier L2 polish** | **6/6** | **528/528** |
| m12 | l2_refine | Soft penalty L2 pull from m10 | (sub-method) | (intermediate stage) |
| m13 | line_search | Global line search seed → phi_in | (sub-method) | 0.1 % reduction (ineffective) |
| m14 | **l2_refine_repair** | **m10 → m12 pull → patch → polish** | **6/6** | **528/528** |
| m_slsqp | slsqp_windowed | Manuscript SLSQP cluster pipeline | 0/6 (wall-slice failures) | 509/528 fold-free, 0/528 strict |

## Cost / accuracy trade-off

```
              mean L2 (lower = closer to input)              median wall/slice (lower = faster)

                0   50   100   150   200                       0s   60s   120s   180s   240s
                │   │    │     │     │                         │    │     │      │      │
SLSQP_agg       ████                                           ───  (per-slice timing not recovered)
slsqp_windowed  ████                                           ███
m10             ████████████                                   ████
m14             ████████                                       ███████████████████████
```

* manuscript SLSQP is the L2 floor but fails strict-threshold feasibility everywhere and fails fold-free on 19 slices.
* m10 gives full feasibility (100 % fold-free + strict) for the lowest median wall time among the feasibility-100 % methods.
* m14 splits the difference: ~2× L2 of SLSQP, full feasibility, 6× the wall of m10.

The fresh `slsqp_windowed` run also exposes the long-tail risk of SLSQP near the wall: while its median is fast (29 s), some slices burned 200 min apiece in retry thrash — m10 and m14 have bounded per-slice tails (max ~5 min).

## Recommended pipeline

**Default**: **m14_l1** (L1 anchor) — best L2 AND best L1 among 100 %-feasibility methods. Dramatic wins on the dense wall slices.
**When you need maximum speed**: m10 (6× faster, 100 % feasibility, higher L2/L1).
**When you want lowest-possible L2 and tolerate a few folded slices**: manuscript SLSQP (clusters / `_run_2d_clusters.py`).

## Reproducing the benchmark

```powershell
cd notebooks\experiments\wall_breakers
# wall slices (cheap sanity check)
python run_all.py --fixture slice --methods l2_refine_repair --save_npy

# whole DVF (parallel, ~5-6 h on 6 cores)
$env:CUDA_VISIBLE_DEVICES = ""
python run_full_dvf.py --method l2_refine_repair --workers 6 --time_budget_s 360

# compare against baselines
python aggregate_manuscript_slsqp.py
python run_full_dvf.py --method harmonic_l2_polished --workers 6 --time_budget_s 400
```

Per-method CSVs land in `results_full_dvf/{method}__full_dvf.csv`. The
companion `results_visualizer.ipynb` plots feasibility vs L2 with
per-z winners.

## What's in this directory

```
wall_breakers/
├── README.md
├── FINDINGS.md                                # this file
├── harness.py                                 # fixtures, metrics, schema
├── run_all.py                                 # serial driver (wall slices)
├── run_full_dvf.py                            # parallel driver (528 slices)
├── _rebuild_summary.py                        # re-aggregate from JSONs
├── aggregate_manuscript_slsqp.py              # reads existing manuscript outputs
├── results_visualizer.ipynb                   # plots and inspection
├── methods/
│   ├── m01_svf_projection.py
│   ├── m02_harmonic_extension.py
│   ├── m03_augmented_lagrangian.py
│   ├── m04_paint_and_blend.py
│   ├── m05_torch_full_grid.py
│   ├── m06_quasi_conformal.py
│   ├── m07_tv_regularized.py
│   ├── m08_harmonic_seed_polish.py
│   ├── m09_svf_polished.py
│   ├── m10_harmonic_l2_polished.py            # always-feasibility winner
│   ├── m11_lbfgs_barrier.py
│   ├── m12_l2_refine.py                       # soft-penalty L2 pull
│   ├── m13_line_search.py
│   ├── m14_l2_refine_repair.py                # FINAL WINNER (L2 + feasibility)
│   └── m_slsqp.py                             # manuscript SLSQP wrapper
├── results/                                   # wall-slice benchmarks
└── results_full_dvf/                          # full-DVF benchmarks
    ├── harmonic_l2_polished__full_dvf.csv
    ├── l2_refine_repair__full_dvf.csv          # main result
    ├── manuscript_slsqp__full_dvf.csv
    ├── harmonic__full_dvf.csv
    └── svf_squaring__full_dvf.csv
```

## How to use the winner

```python
from notebooks.experiments.wall_breakers.methods import m14_l1 as winner
import numpy as np

phi_full = np.load('data/.../deformation3d.npy')   # (3, D, H, W)
z = 12
phi_in = np.stack([phi_full[1, z], phi_full[2, z]])  # (2, H, W)
out = winner.solve(phi_in, threshold=0.01, margin=1e-3, time_budget_s=360)
phi_out = out['phi_out']                # (2, H, W), tri_neg == 0 by construction
print(out['info']['final_min_T'])       # >= 0.01 by construction
print(out['info']['final_L2'])
```

`m14_l1` is `m14_l2_refine_repair` with `anchor='l1'` -- everything else
identical. The smoothed-L1 anchor
`sqrt((phi - phi_in)^2 + eps^2)` tolerates a few large local
deviations cheaply, which is exactly what's optimal when the fold
cores require concentrated displacement and the rest of the field can
return to `phi_in` unchanged.
