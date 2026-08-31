# The 2D zero-folds recipe

The measured-robust way to take a raw 2D deformation slice to **zero folds**:
bilinear cell-min rows, on the windowed elastic-QP engine, with no fidelity
anchor.

```python
from dvfopt import correct_dvf

result = correct_dvf(
    phi,                        # (2, H, W) or (3, H, W) — dz passes through
    constraint='bilinear',      # 4 triangle rows / cell = sub-pixel injectivity
    strategy='isqp_windowed',   # cluster-windowed, no-damage by construction
    objective='l2',           # pure feasibility
)
assert result.feasible
```

CLI equivalent:

```bash
dvfopt correct in.npy out.npy --constraint bilinear --strategy isqp_windowed --objective none
```

`strategy='auto'` picks `isqp_windowed` for the `bilinear` constraint at any
objective, and for `simplex_standard` under `objective='l2'` — see the routing
table in `auto_strategy`'s docstring. It does **not** silently swap in this
recipe for an L1/L2 request: an anchor is a different fidelity ask, so
`simplex` + `l1` stays on the SLP champion (with a one-line hint on the `dvfopt`
logger pointing here).

**Measured basis.** On B0039 this reaches 0 simplex folds from the RAW field on
every slice tested — z16 goes 3890 folds → 0 in ~200 s, damage 0 — where the
2-triangle-row methods (SLP, m10/m14) stall on twisted cells. Nine real slices
(fold counts 835–3890) all clear at damage 0.

## Why each piece

- **`constraint='bilinear'`** — four triangle rows per cell (both diagonal
  splits), so `min(rows) = ½·cell_min_jdet_2d`: feasibility certifies the
  *bilinear interpolant* injective on every cell, and the extra rows give
  non-degenerate constraint gradients at bow-tie cells where the 2-row simplex
  gauge is blind and its gradients collapse. That is the actual reason the
  2-tri-row methods stall.
- **`strategy='isqp_windowed'`** — one small frozen-ring window per fold
  cluster. Only free pixels are written back, so healthy area is untouched *by
  construction* (damage 0 is structural, not measured luck).
- **`objective='l2'`** — the L1/L2 distance objective pins residual folds
  (the objective-basin trap: the same window clears the instant the anchor is
  off). Use `'l1'`/`'l2'` only if you need the anchor and accept the risk of a
  residual.

## What the engine defaults do, and what bought them

All numbers: raw B0039 z16 (bilinear rows, `objective='l2'`, threshold 0.01,
OMP/BLAS/RAYON pinned to 1), 0 simplex folds and damage 0 on every row unless
stated. Sources: [CHANGELOG.md](../CHANGELOG.md), [ARCHITECTURE.md](../ARCHITECTURE.md).

| Default | What it does | Measured basis |
|---|---|---|
| `qp_backend='hybrid'`, `ip_cold=True`, `ip_after_admm_iters=800` (#78) | Interior-point Clarabel on a window's cold first QP and after any ADMM solve that hit 800 iterations (the stale-warm-start signal); warm-started OSQP otherwise | 262 s vs 300 s all-OSQP, better fidelity (L2 move 325 vs 346). Clarabel-*always* is slower (381 s) — a warm ADMM solve averages 0.175 s. Policy sweep: cold-only 296 s, 400 → 289 s, **800 → 262 s**, 1500 → 269 s |
| `no_tr_fallback=True`, `fallback_maxiter=200` (#73) | A window that misses its target is retried once on the same box with the trust region off (backtracking line search), warm-started from the failed iterate | The TR ratio test freezes on sliver-scale violations (~1e-4, inside OSQP's own noise) that the line search clears. The retry keeps whichever iterate has the higher constraint minimum, so it can never be worse |
| backend fallback (#78) | A real window (never a giant tile) left *genuinely* folded is re-attempted whole on plain OSQP from its ORIGINAL start state, ahead of grow-on-failure | The IP legs can steer a window into a basin with no escape. Without this rung the `z0_cluster` crop finishes one triangle inverted at -1.2e-4; with it all three hard crops reach 0 |
| `qp_max_iter=2000`, `qp_max_iter_fallback=500` (#73) | Two-tier ADMM iteration cap per subproblem | ~2× faster at unchanged feasibility. A cap-*escalation* ladder over these was measured slower and no more feasible — do not add one |
| `giant_tile=64`, `giant_tile_fit=True` (#75, #77) | Over-`max_window_area` regions are cleared by overlapping-tile Schwarz; the tile is a *target*, fitted per region so an integer number of near-equal tiles covers its longest side | Tile 64 vs 32: 362 s / 22 windows / 1 round / no mop vs 685 s / 264 windows / 3 rounds / 4 mop (1.9×, and a smaller move: L2 316 vs 404). Size acts through grid **alignment**, not size: on the z16 giant (a 125×152 box) 64 aligns (1 round, 374 s) while 56 and 80 do not (2 rounds, ~600 s); the fitted 51 aligns by construction (1 round, 345 s) |
| `coarse_to_fine=True`, `coarse_factor=4` (#82, #83) | Solve the same problem on a 4× box-averaged field first, prolongate the correction back masked to the fine free boxes, start the fine solve warm | 205 s (incl. a 16 s coarse solve) / 909 SQP iterations vs 283 s / 1320 cold, at a slightly *smaller* move (L2 320.6 vs 325.1). Factor 4 vs 2: 182 s / L2 280 vs 189 s / L2 321. Masking to the free boxes is what keeps no-damage exact. Skipped — byte-identical to off — on a fold-free field or when `min(H, W) < 4 * giant_tile` |
| `step_rule='exact_ls'` (#84) | The QP step is accepted at the EXACT minimiser of the merit along it, instead of the trust-region ratio test's accept/reject | 200 s / 563 SQP iterations vs 244 s / 780 at `'tr'`, smaller move (L2 268 vs 280); **9/9 wall and iteration wins** over nine real slices (−19% wall, −27% iterations) with a smaller L2 move on every one. Free because 2D rows are bilinear in `(dy, dx)`, hence exactly quadratic along a line, and the model reuses the `cons(x + d)` the ratio test already evaluates. 2D only (a 6-tet row is cubic along a line). Guarded: the TRUE merit at `a*` is checked, and the iteration falls back to `'tr'` acceptance if it did not decrease |
| `tr_delta=2.0`, `tr_max=16.0` (#82) | isqp trust-region initial radius / cap, in grid units | `tr_delta=1.0` runs 267 s / 1022 iterations but at L2 move 344 — speed bought with fidelity, which coarse-to-fine is not. `tr_max` never binds on the measured windows |
| thread pinning + small `n_workers` (#79) | Every process pool pins its workers to one compute thread; `n_workers` / `--n-workers` of **2–4** | An unpinned worker carries 53 OS threads before doing any work (+23 from `import numpy`, +26 more from scipy). Pinning buys 9–25% at ≥12 workers. The real ceiling is memory bandwidth: throughput peaks at **2.6× around 4 workers** and declines past it; the physical (16) or logical (24) core count is 4–5× slower per solve *and* lower throughput |

`qp_backend='hybrid'` needs `clarabel>=0.9` (a core dependency); without it the
engine silently behaves as `'osqp'`, which reproduces the pre-hybrid path byte
for byte. `isqp_windowed` needs `osqp` (`pip install dvfopt[solvers]`).

## The fast crop pack

Full raw slices take minutes. `benchmarks/make_hard_crops.py` builds three small
crops of the RAW B0039 field around mapped pathologies, keeping the property that
ordinary solvers fail on them while running in seconds-to-minutes:

| crop | pathology |
|---|---|
| `z16_twist` | the bow-tie cell (collapsed edges, 65–160 px displacement ring) that defeated every 2-tri-row method, all ladder variants, and M14 |
| `z0_cluster` | the ~3×-compressed dense cluster (area transport + twists + a three-corners-coincident cell) |
| `z0_sliver` | cells pinned ~-4e-4 below threshold; simplex-clean on input, so the 2-tri rows are blind to it |

```bash
python benchmarks/make_hard_crops.py             # build + validate
python benchmarks/make_hard_crops.py --build-only
```

Validation runs `windowed_correct` **once** per gauge on engine defaults: the
discriminator (standard 2tri rows, objective none) leaves bilinear folds behind,
and the recipe (bilinear rows) clears the case to zero on both gauges. Crops land
in `data/dvfs/crops/` (gitignored; the script regenerates them).

## Measured dead ends

Each was built and measured. Do not re-litigate without new evidence.

- **float32 OSQP** (`proto-osqp-float32`) — a custom float32 build is 2.4–3.0×
  faster per ADMM iteration (confirming the bandwidth diagnosis) but exits at
  worse points: z16 157.6 s vs 192.5 s *at* L2 331 vs 280 and 1114 vs 780 SQP
  iterations; `z0_sliver` 175 s vs 66 s with 9 of 18 folds LEFT; the suite is net
  slower with quality regressions. (Landmine if ever revisited: float32 trips
  polish LDL and OSQP returns a stale `x` buffer with status `'solved'`.)
- **GPU batched ADMM** (`proto-gpu-admm`) — on 40 real captured QPs: CPU pool
  291 ms/QP vs GPU K=64 969 ms (0.30×). Jacobi-PCG needs ~25 CG iterations per
  ADMM iteration; batching amortises launches, not a 25:1 work ratio. Direct
  hybrid (CPU factorise + GPU triangular solve) is worse still.
- **Newton-SQP** (`proto-newton-sqp`) — exact row Hessians verified to 4.4e-12,
  but with `NoneObjective` the true KKT multipliers are zero, and away from
  feasibility the elastic form pins violated rows' duals at `-rho`, so the
  "Newton" term models big-M *penalty* curvature (indefinite). PSD-capped: z16
  176 s / 836 iterations / L2 295 vs 187 s / 762 / 280; sliver 131 s vs 59 s.
- **Dual warm starts** (`proto-dual-warmstart`) — the driver already carries
  OSQP `x` *and* `y` across SQP iterations (zeroing `y` costs 2.3× the ADMM
  work); coarse→fine dual prolongation is structurally vacuous (all 35,708
  coarse duals are ~0). Mapping Clarabel duals into OSQP cuts ADMM work 35% but
  adds 39% SQP iterations — wall no better.
- **Constraint-row pruning** — a slack-based active set is 3–8× slower (z16 at
  tau 0.1: 1246 s vs 374 s, 196 augmentations); even tau 1.0, keeping 88% of
  rows, is 3–7× slower. Dead at every tau: nodes move enough that slack rows
  activate, and every augmentation is a full re-solve.
- **OSQP settings** — fully closed. Polish is irrelevant; eps 1e-4 is 2.4×
  slower (386 s); `alpha=1.0` costs +62% iterations and `adaptive_rho=False`
  +23%; `rho=0.01` gives ~4× better per-QP violation at equal iterations but
  in-engine is worse (sliver 155 s / 1248 iterations vs 59 s / 540). The
  defaults are the sweet spot.
- **Stall detection / iteration caps** — `maxiter` 600/300/150/75 are identical
  on the crops and on z16; windows stop on tolerance, not on the cap.
- **Shared symbolic factorisation** — OSQP setup totals ~0 s of a 374 s run.
- **Maximal fold-free step cap** (the nonlinear analogue of monotone accept) —
  REFUTED alongside `exact_ls`: real windows admit `a_max ~ 1e-3–1e-1`, which
  strangles the elastic mechanism (end violations 40–84 vs the baseline's 0.027).

The recurring lesson: the loop's cost is governed by SQP **step count**, and it
is chaotic with respect to where the subproblem terminates. Per-iteration
arithmetic wins evaporate when subproblem accuracy shifts; the levers that
actually pay (coarse-to-fine, tile fit, hybrid IP cold start, exact line search)
are the ones that cut step count.

## Known cost

`step_rule='exact_ls'` costs on **one** case: the `z0_sliver` crop runs 350.5 s /
1684 SQP iterations against `'tr'`'s 76.7 s / 540 (both 0 folds, damage 0). It is
a chaos artifact, not a trend: that crop starts already simplex-clean (min
+0.0110 against a 0.01 threshold) with only ~1e-4-scale bilinear violations, so
every decision is made at OSQP's noise floor and the escalation ladder amplifies
the outcome ~10×. The mechanism is understood — `'tr'` discovers in 11 iterations
that it cannot clear the window and hands it to the escalation ladder, while an
exact minimiser always finds *some* decrease and grinds to `step-tol` first.
Nothing on real slices behaves this way (9/9 wins). Pass `step_rule='tr'` for a
sliver-dominated workload.
