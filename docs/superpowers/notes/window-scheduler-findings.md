# Continuous scheduler vs colour-group batches in the windowed engine — findings

Investigation branch `proto-window-scheduler` (based on `proto-parallel-windows`,
merged up to `origin/main` so everything is measured against the **current** engine
defaults: `giant_tile=64` + `giant_tile_fit=True` (#77) and `qp_backend='hybrid'`
(#78)). Prototype:
[`benchmarks/parallel_scheduler_proto.py`](../../../benchmarks/parallel_scheduler_proto.py).
Nothing under `dvfopt/` was changed.

**Question.** The previous prototype
([parallel-windows-findings.md](parallel-windows-findings.md)) batched each round's
windows into conflict-free **colour groups** and hit a hard **1.9 effective
concurrency plateau, flat from 4 workers to 16**. Its own recommendation named the
suspect and the lever: a colour group finishes in the time of its slowest member, so
"the lever is a continuous scheduler over a conflict graph (submit the next
non-conflicting window as soon as a worker frees) — the pattern
`dvfopt/core/slp/cluster_lp_2tri.py` already implements — not a bigger pool". Does
that lever break the plateau?

**Answer up front: no. Do not promote.** The plateau is the dependency structure,
not the batching, and this lever is now spent.

- At **equal tiling geometry** (`giant_tile=32`, what the colour-group prototype
  measured), the continuous scheduler reaches effective concurrency **1.79 / 1.85 /
  1.82** at 4 / 8 / 12 workers — level with, and if anything just under, the
  colour-group design's **1.84 / 1.89 / 1.89**. Three barriers removed; the number
  did not move.
- At **today's engine defaults** it is worse: **1.47 at every pool size**, because
  #77's geometry-fit tiling cut the giant from 30 tiles to 12 and its conflict
  graph's maximum independent set from **9 to 4**. Peak observed in-flight was
  exactly 4 at w=4, w=8 and w=12 — the pool never found a fifth admissible tile.
- Wall clock on a quiet box: **1.04-1.05x**, flat in pool size. Twelve workers buy
  what four do, and four buy almost nothing.
- The serial remainder is **not** the ceiling: 4.7% of stream wall, essentially all
  of it the serial grow-on-failure drain. **63-88% of worker-slot-seconds sit idle**
  because nothing is admissible. That is the ceiling.
- **#75/#77 already beat process parallelism at its own game.** Fitting the tile to
  the giant's geometry took the *serial* engine from **599 s to 263 s (2.28x)** on
  this slice — far more than any pool size delivered, on one core, deterministically,
  with a better move.

## Measured results

`z16_full` = raw B0039 z=16, `(2, 320, 456)`, **3890** simplex folds before.
`SimplexConstraint2DBilinear`, `objective=NoneObjective()`, `threshold=0.01`,
`maxiter=600`, 24-core Windows box, spawn start method, every thread pool pinned to
1. `folds` is the task's simplex metric `min(T1, T2) < 0.01`; `L2` is
`||phi_out - phi_raw||_2`. Pool start-up (0.8-1.3 s) is excluded from `wall`.
Effective concurrency = summed in-worker solve seconds / scheduler stream wall.

### Engine defaults (`giant_tile=64`, `giant_tile_fit=True`, `qp_backend='hybrid'`)

| run | wall | vs engine | **eff. conc.** | peak in flight | solves (win/tile) | rounds / giant sweeps / mop sweeps | folds after | damage | L2 move |
|---|---|---|---|---|---|---|---|---|---|
| **engine serial** | **262.64 s** | 1.00x | — | 1 | 40 | 1 / — / — | **0** | **0** | **325.1** |
| scheduler w=4 | 249.32 s | **1.05x** | **1.47** | 4 | 36 (7/29) | 2 / 2 / 0 | 0 | 0 | 384.9 |
| scheduler w=8 | 250.33 s | **1.05x** | **1.47** | 4 | 36 (7/29) | 2 / 2 / 0 | 0 | 0 | 384.9 |
| scheduler w=12 | 253.19 s | **1.04x** | **1.47** | 4 | 36 (7/29) | 2 / 2 / 0 | 0 | 0 | 384.9 |

One contiguous sweep on an idle box, reproduced by an independent bracket sweep to
within 0.5% (engine 263.82 s, w=4 248.46 s, eff. conc. 1.47). The work itself is
deterministic — identical solve counts and identical L2 at every pool size — so the
only thing pool size changes is how long the idle workers wait.

Every run is feasible with `damage == 0`: the no-damage invariant survives
continuous scheduling exactly as it survived colour groups, and structurally
(conflict-free admission means no worker reads what another in-flight worker writes).

**Scheduler serial (`w=1`, no pool, same ordering and halo-crop plumbing): 787.02 s
against a 335.60 s engine measured minutes earlier in the same window — 2.34x.** The
reordering penalty the previous note warned about ("re-ordering is not free", it
measured 64%) is larger here. It is not a clean overhead measurement: the trajectory
diverges (27 solves vs 40) and this one drew a few very expensive windows (29.1 s per
solve against 11.0 s at w=4). Read it as evidence that **trajectory noise on this
problem is several times larger than the entire parallel win.**

### Old geometry (`giant_tile=32`, fit off) — equal width to the colour-group prototype

| run | wall | vs its own engine | **eff. conc.** | peak in flight | solves (win/tile) | folds | damage | L2 |
|---|---|---|---|---|---|---|---|---|
| engine serial | 599.23 s | 1.00x | — | 1 | 259 | 0 | 0 | 409.6 |
| scheduler w=4 | 282.45 s | 2.12x | **1.79** | 4 | 79 (8/71) | 0 | 0 | 505.0 |
| scheduler w=8 | 329.12 s | 1.82x | **1.85** | 7 | 90 (6/84) | 0 | 0 | 414.7 |
| scheduler w=12 | 324.29 s | 1.85x | **1.82** | 7 | 90 (6/84) | 0 | 0 | 414.7 |
| *colour groups (prior note, same geometry)* | — | *1.57x / 1.61x / 1.63x* | ***1.84 / 1.89 / 1.89*** | — | — | *0* | *0* | — |

**This is the decisive row.** Handed the same geometry the colour-group prototype
had, the continuous scheduler lands at 1.79-1.85 effective concurrency against
1.84-1.89. Removing the intra-group straggler, the inter-sweep Schwarz barrier *and*
the round-vs-giant ordering changed nothing. The 1.9 plateau is a property of the
conflict graph, and load balancing was never what was binding.

(Wall columns are not comparable across the two notes — different serial baselines,
and this branch runs the hybrid QP backend. Effective concurrency is, which is why
the prior note nominated it as the decisive number.)

Note also that **more workers made it slower here**: w=8/12 (324-329 s) lose to w=4
(282 s) despite higher concurrency. Wider concurrency means staler frozen rings in
the Gauss-Seidel coupling, hence a different trajectory and more total work
(`worker_solve` 607 s at w=8 against 482 s at w=4 for the same zero-fold result).
Concurrency and convergence are in direct tension.

### Why the width collapsed — measured, not inferred

Round 1 on this slice emits **5 boxes**: one `125 x 152` giant (19 000 px) and four
small ones (1591, 266, 100, 56 px). Practically all exploitable parallelism lives
inside that one giant, so its tiling *is* the concurrency ceiling:

| tiling | effective tile | tiles | max independent set |
|---|---|---|---|
| `giant_tile=32` (old default, prior prototype) | 32 | **30** | **9** |
| `giant_tile=64`, fit off (#75) | 64 | 9 | 4 |
| `giant_tile=64` + `giant_tile_fit` (#77, **today's default**) | 51 | **12** | **4** |

Tiles step by `tile - (2*ring + 2)`, so neighbours (diagonal ones included) overlap
and conflict; an independent set is a spaced sub-lattice. Twelve tiles in a 3x4 grid
admit 4, and 4 is exactly what `max_inflight` measured at every pool size. The
prototype's greedy admission *is* an incremental maximal-independent-set
construction, so this is the graph's limit and not the heuristic's.

### Ceiling: where the wall actually goes (engine defaults)

| workers | stream wall | serial remainder | of which grow drain | idle worker-slot-seconds | idle fraction of pool |
|---|---|---|---|---|---|
| 4 | 237.6 s | 11.8 s (4.7%) | 11.5 s | 601.3 s | **63%** |
| 8 | 238.6 s | 11.8 s (4.7%) | 11.4 s | 1557.3 s | **82%** |
| 12 | 241.4 s | 11.8 s (4.7%) | 11.3 s | 2542.4 s | **88%** |

The serial tail is negligible and nothing in it is worth attacking: `min_field`
recomputation 0.1 s, `find_windows` + conflict ordering < 0.05 s, accounting
< 0.05 s. The entire serial remainder is the grow-on-failure drain (2 windows), and
Amdahl on 11.8 s of 249.3 s caps the extra win at 1.05x even if grow were
parallelised too — which it cannot be, being a second attempt that depends on the
first.

**The ceiling is the idle column.** At 12 workers, 88% of worker-slot-seconds have
nothing admissible to run: the round loop's 4 small boxes plus a 4-wide sweep front
inside the giant is all the parallelism that exists. The mop never fired in any
scheduler run (`mop_sweeps=0`, the round loop cleared the slice), so **these numbers
say nothing about the mop** — which the prior note measured as the worst case for
width (62 clusters collapsing to 21 groups, `mop_margin=25` inflating every box).

## Measurement hygiene — the first sweep was contaminated

Another timing job saturated the machine (43 python processes, 100% CPU) during the
first attempt. Same script, same case, same deterministic work:

| run | contended box | idle box | factor |
|---|---|---|---|
| engine serial | 1439.15 s | 262.64 s | **5.5x** |
| scheduler w=8 | 280.72 s | 250.33 s | 1.12x |

A single-thread integer benchmark showed *no* slowdown under that load, and a
process-pool capacity probe still measured 10.56 effective concurrency at 12 workers
— the contention was memory bandwidth, invisible to a register-bound probe and
brutal to the solver. It also hits the serial run about five times harder than the
parallel one (the parallel run is already sharing), so a contaminated reference
**inflates the apparent parallel win**: the contended sweep read as 1.20-1.25x where
the clean one reads 1.04-1.05x.

Every number in the tables above is from a verified-idle window. The contended sweep
was discarded, and two independent idle sweeps agree within 0.5% on every config.

This is also why the tables lead with effective concurrency: `worker_solve` is
wall-clock measured *inside* each worker, so preemption inflates numerator and
denominator together and the ratio still reports achieved parallel width. It read
1.47 in the contended sweep and 1.47 in both clean ones.

## How the prototype works

Engine building blocks reused verbatim — `find_windows`, `build_subproblem` (through
`_solve_window`), `_solve_window`, `min_field` / `pixel_fold_mask`, `_InnerOpts`,
`SliceReport`, and `_fit_tile` so the tile geometry (including `giant_tile_fit`)
matches the shipped `_solve_giant_schwarz`. Local to the prototype: the halo crop
(carried over from the previous prototype), the conflict predicate, an
incrementally-maintained fold map, and the scheduler.

### The three barriers it removes

1. **Intra-group straggler.** A colour group of 9 tiles finished in the time of its
   slowest tile. Here admission is per worker slot: when a worker frees, the
   scheduler admits the first pending item conflicting with nothing in flight,
   scanning most-conflicted-first then largest-first (degree so the blocking items
   go early, area as an LPT long-job defence).
2. **Inter-sweep (Schwarz).** The giant tiler's sweeps were strictly sequential.
   Here each tile carries its own sweep counter `k` and is re-queued the moment it
   completes still holding a fold in its free box, capped at `giant_max_sweeps`. The
   engine's plateau guard survives as a per-giant stall check at *virtual sweep*
   boundaries (every `n_tiles` completions in that giant), so worst-case work is
   still `giant_max_sweeps * n_tiles` solves.
3. **Round-loop vs giant.** The previous prototype ran a round's small boxes as
   batches, *then* the giants. Here a round's small windows and every giant's tiles
   enter **one stream**, so a giant's tail overlaps the next small window.

### Correctness carried over

- **No-damage is structural.** Two items run concurrently only if neither's free box
  meets the other's patch (free box ± ring), so no worker reads a region another
  in-flight worker writes. `touched` is marked exactly as the engine marks it, so any
  residual counts as residual, never damage. Every run reported `damage == 0`.
- **Crops are taken at admission time**, so an item admitted after a neighbour
  completed sees the updated field in its frozen ring — the same Gauss-Seidel
  coupling the serial engine has.
- **Grow-on-failure stays serial**, drained in the parent through the engine's own
  `_solve_window(..., _grow=1)` recursion.
- **`--selftest`** asserts (a) a worker's halo-crop solve is bit-equal to the serial
  engine's on interior / top-left / bottom-right boxes, (b) the incremental fold map
  yields the same fold *mask* as a full recompute (values agree to ~1e-16 — the same
  algebra over a strided crop rounds differently from the contiguous whole — and the
  map is re-seeded from a full recompute at every pass boundary so noise cannot
  accumulate), and (c) an end-to-end run reaches the engine's feasibility.

### Thread pinning — not optional at these defaults

Every worker pins `OMP`, `OPENBLAS`, `MKL`, `NUMEXPR`, `NUMBA`, **`RAYON`** and
`VECLIB` thread counts to 1, in the parent env before the pool is created *and*
again in the worker initializer, before any heavy import. `RAYON_NUM_THREADS` is the
one the previous prototype did not need and this one does: the engine's default
`qp_backend='hybrid'` runs Clarabel (Rust/rayon) on cold or stale warm starts, so
without the pin every worker spawns its own rayon pool and an N-way process pool
oversubscribes the box N-fold. **Any future per-process pool in this codebase needs
this line**, whatever happens to the scheduler.

The pool deliberately does **not** use `dvfopt.core._pool.get_pool`: its
`_warmup_worker` JIT-compiles the 3D tet numba kernels, which a 2D windowed run never
touches (~5-10 s/worker of pure waste). The prototype creates one persistent
`ProcessPoolExecutor` with a windowed-family initializer (pin, then one tiny
`_solve_window` to pay the osqp/clarabel import and the first CPR colouring). Pool
start-up is 0.8-1.3 s, paid once.

## Recommendation: do not promote

1. **The lever the prior note nominated is spent.** At equal geometry the continuous
   scheduler does not beat colour-group batching (1.79-1.85 vs 1.84-1.89 effective
   concurrency). There is no third scheduling idea worth trying: greedy admission
   already builds maximal independent sets, and the graph's maximum is 9 (old
   tiling) / 4 (current).
2. **Today's defaults make it worse, and that is the right trade.** #75/#77 cut the
   giant from 30 tiles to 12 and the achievable width from 9 to 4, and in exchange
   took the serial engine from 599 s to 263 s. A 2.28x serial win on one core beats
   the 1.05x this pool delivers on twelve. Widening the tiling to feed a pool would
   hand that back.
3. **The efficiency is poor even where it "wins".** 1.05x wall for ~349 s of worker
   CPU against 263 s of single-core serial — it spends **33% more total CPU** and
   idles 63-88% of the pool to save 13 s.
4. **Concurrency fights convergence.** At the old geometry w=8/12 were *slower* than
   w=4 (324-329 s vs 282 s) because wider concurrency means staler frozen rings, a
   different trajectory and more total work. A promoted knob would make wall clock
   non-monotone in `n_workers` — a bad knob.
5. **The output legitimately differs and is measurably worse here.** L2 move 384.9
   (parallel) vs 325.1 (serial) under `NoneObjective()`, both feasible with
   `damage == 0`. `L1Objective` / `L2Objective` would pin the two together, but that
   is a reason not to ship a knob whose only benefit is 1.05x.
6. **Slice-level parallelism remains the right axis**, unchanged from the prior note:
   `DVFoptConfig.n_workers` and the CLI's `--n-workers` fan whole slices out with no
   coupling and near-linear scaling. Window-level parallelism only matters with
   exactly one slice in flight, and must never nest with it.

If it is ever revisited, the *only* thing that would change the answer is a
constraint family or a fold geometry that produces many independent mid-size
clusters instead of one dominant giant — i.e. a different problem, not a better
scheduler.

### What is worth keeping from this investigation

- **`RAYON_NUM_THREADS=1`** in any worker initializer that can reach the hybrid QP
  backend. A live footgun today for anything that pools `dvfopt` work.
- **`dvfopt.core._pool`'s initializer is 3D-only.** Whatever eventually pools 2D work
  will waste 5-10 s/worker JIT-compiling tet kernels it never calls. Cheap fix when
  someone needs it; no reason to do it speculatively.
- **One serial hypothesis fell out of the scheduler and deserves its own A/B:**
  re-sweeping only the tiles that still hold a fold, instead of every tile every
  sweep. At `giant_tile=32` that took solve count from the engine's **259 to 79-90**
  for the same zero-fold result. It is confounded here (the trajectory also changed
  and per-solve cost rose, so wall did not fall proportionally), and
  `giant_tile_fit` already reaches 40 solves by a different route — so this is a
  hypothesis for a serial experiment in `_solve_giant_schwarz`, not a result.

## Reproduce

```bash
python -u benchmarks/parallel_scheduler_proto.py --selftest

# engine defaults (the headline table)
python -u benchmarks/parallel_scheduler_proto.py \
    --cases z16_full --workers 4,8,12 --reference

# equal width to the colour-group prototype (the decisive comparison)
python -u benchmarks/parallel_scheduler_proto.py \
    --cases z16_full --workers 4,8,12 --reference \
    --giant-tile 32 --giant-tile-fit 0
```

`z16_full` reads `data/dvfs/b0039/b0039_laplacian_deformation_field.npy` (mmapped,
`vol[1:, 16]` -> `(2, 320, 456)` float64; gitignored). **Check the box is idle
first** — a competing job cost the serial reference a 5.5x wall factor here and
inflated the apparent parallel win from 1.05x to 1.25x.
