# Parallel windows in the windowed isqp engine — findings

Investigation branch `proto-parallel-windows`. Prototype:
[`benchmarks/parallel_windows_proto.py`](../../../benchmarks/parallel_windows_proto.py).
Nothing under `dvfopt/` was changed.

**Question.** OSQP solves are ~90% of `windowed_correct`'s wall clock and each
round's windows are independent — how much wall clock does batching them across
processes actually give back, and what should a promoted `n_workers` look like?

**Answer up front.** Per-window process parallelism is *sound* — byte-identical
serial path, `damage == 0` preserved, ~1 ms IPC per window — but **structurally
starved**: the engine produces a handful of windows per round, not dozens.

- Full B0039 z=16 slice (3890 folds): **1.57x at 4 workers, 1.61x at 8, 1.63x at
  16** — and the win comes almost entirely from inside the **giant tiler**, not
  from the round loop.
- The three hard crops: **0.94x - 1.05x**, i.e. noise or a net loss. Every batch
  had width 1.
- Effective concurrency plateaus at **1.9** regardless of pool size. More workers
  buy nothing.

Recommendation and the cheaper alternatives are in
[Design recommendation](#design-recommendation).

## How the prototype works

It re-creates the round loop and reuses every engine building block verbatim —
`find_windows`, `build_subproblem` (through `_solve_window`), the inner solve +
no-TR fallback + paste-back + `WindowRec`, `min_field` / `pixel_fold_mask`,
`_InnerOpts`. Three things are local to the prototype:

1. **`_halo_crop`** — a worker gets a small crop, not the whole field.
   `build_subproblem` infers its `borders` tuple from `py0 == 0 / py1 == H` against
   the array it is handed, so a crop of *exactly* the patch would look like an
   all-image-border patch and enforce rows evaluated with the wrong one-sided
   difference. Cropping with a `ring + 1` halo on interior sides (and none on sides
   that reached the image border) reproduces the global flags exactly. The
   prototype's `--selftest` asserts a worker's window result equals the serial
   engine's on interior, top-left and bottom-right boxes.
2. **`_conflict` / `_groups`** — a window READS its patch (free box ± ring) and
   WRITES its free box, so two windows may run concurrently iff neither's free box
   meets the other's patch. Conflicting boxes are split into successive groups
   (greedy colouring); each group is one `executor.map` batch.
3. **Parallel copies of the giant tiler and the mop sweep**, because their
   tile/cluster geometry lives inline in `_common.py`.

Grow-on-failure stays serial: the parallel batch runs first attempts, then each
window the inner declared infeasible is grown and retried in-process through the
engine's own recursion (`_solve_window(..., _grow=1)`).

`--workers 1` short-circuits the pool and calls `_solve_window` in the engine's own
order (giants inline). That path is **byte-identical to `windowed_correct`** —
verified on every case (`identical_to_engine=True`).

BLAS/OMP are pinned to one thread per process before numpy is imported (spawn
workers re-execute the module top-level, so they inherit the pin).

## The structural problem: there are almost no independent windows

Measured with `SimplexConstraint2DBilinear`, `threshold=0.01`, `margin=3`,
`ring=1` — round-1 window counts and the conflict-free group decomposition:

| case | shape | folded px | round-1 boxes | small→groups | giants | giant tiles→groups | mop clusters→groups |
|---|---|---|---|---|---|---|---|
| `z0_cluster` | (2, 35, 42) | 605 | 1 | 1 → 1 | 0 | — | 1 → 1 |
| `z0_sliver` | (2, 53, 52) | 18 | 1 | 1 → 1 | 0 | — | 9 → 9 |
| `z16_twist` | (2, 50, 50) | 663 | 2 | 2 → 2 | 0 | — | 13 → 13 |
| `z16_full` | (2, 320, 456) | 4022 | 5 | 4 → 2 (3, 1) | 1 | 30 → 4 (9, 9, 6, 6) | 62 → 21 |

Two effects collapse the available width:

- **`find_windows` dilates by `margin + ring` before labelling**, deliberately
  merging any clusters whose windows could couple. On a real slice that merges
  4022 folded pixels into **5** windows, one of them a 19 000-px giant.
- **The mop uses `mop_margin=25`**, which inflates every residual cluster's box by
  50 px in each axis; on a 50×50 crop every mop box covers the whole crop, so all
  of them conflict (9 clusters → 9 serial groups). Even on a full slice, 62 mop
  clusters only decompose into 21 groups (mean width ≈ 3).

So the exploitable parallelism is: a handful of round windows, ~4-colour groups of
giant tiles (width 6-9), and mop groups of width ~3.

## Measured results

`SimplexConstraint2DBilinear`, `objective=NoneObjective()`, `threshold=0.01`,
`maxiter=600`, 24-core Windows box, spawn start method. `folds` is the task's
simplex metric `min(T1, T2) < 0.01`. Pool start-up is measured but excluded from
the wall column (it is amortised by a shared pool in any promotion).

### The three crops — parallelism is a net loss

| case | engine serial | proto w=1 | w=4 | w=8 | w=16 | best speedup | folds after (all runs) | damage |
|---|---|---|---|---|---|---|---|---|
| `z16_twist` (2, 50, 50) | 33.15 s | 33.61 s | 35.77 s | 35.48 s | 36.00 s | **0.94x** | 0 | 0 |
| `z0_cluster` (2, 35, 42) | 23.97 s | 23.86 s | 22.68 s | 22.89 s | 23.05 s | **1.05x** | 0 | 0 |
| `z0_sliver` (2, 53, 52) | 103.06 s | 101.89 s | 103.12 s | 106.17 s | 111.26 s | **0.99x** | 0 | 0 |

Every crop dispatched exactly as many batches as windows (`dispatched == n_batches`):
**every batch had width 1**, so there was no concurrency to have — only per-window
IPC and worker overhead, which shows as the 6% regression on `z16_twist` and the 9%
regression on `z0_sliver` at 16 workers (bigger pools cost more to feed for the same
serial work).

### The full B0039 z=16 slice

3890 simplex folds before, `(2, 320, 456)`:

| run | wall | speedup vs w=1 | windows | folds after (simplex / bilinear) | damage | effective concurrency |
|---|---|---|---|---|---|---|
| engine serial | 685.23 s | — | 264 | 0 / 0 | 0 | — |
| proto w=1 | 679.36 s | 1.00x | 264 | 0 / 0 | 0 | 1.00 |
| proto w=4 | 434.03 s | **1.57x** | 249 | 0 / 0 | 0 | 1.84 |
| proto w=8 | 420.99 s | **1.61x** | 249 | 0 / 0 | 0 | 1.89 |
| proto w=16 | 415.79 s | **1.63x** | 249 | 0 / 0 | 0 | 1.89 |

*Effective concurrency* = `sum(in-worker solve time) / sum(batch wall time)`. It
**plateaus at 1.9 whether the pool has 4, 8 or 16 workers** — the ceiling is the
dependency structure, not the pool. 247 windows were dispatched in only **36
batches** (mean group 6.9), but each colour group is dominated by its single
slowest tile, so a group of 9 finishes in the time of its worst member.

Two second-order effects visible in the same numbers:

- **The same work costs ~20% more in workers.** `worker_solve` sums to 747 s at
  w=4 against ~623 s of in-process solve time at w=1 — per-window process overhead
  plus a cold `_COLORING_CACHE` in each worker (the CPR colouring is cached per
  `(constraint type, ph, pw)` and every worker rebuilds it).
- **The reordered trajectory skipped the mop entirely** (`mop = 0.00 s` at every
  parallel setting, against 56.50 s serial) — worth ~8% of the win, and pure luck
  of the trajectory, not a property to rely on.

Run-to-run variance on this box is ~15% on identical deterministic work (the same
engine-serial run measured 794.81 s and 685.23 s in two sittings), so treat
anything under ~1.2x as noise. The 1.6x is comfortably outside it; the crops'
0.94-1.05x is not.

### Output identity

- **`n_workers = 1` is byte-identical to `windowed_correct`** on every case,
  including a giant-containing 140x140 synthetic (`identical_to_engine=True`,
  `max |diff| = 0`). This is structural: the serial branch calls the engine's
  `_solve_window` in the engine's own box order with giants inline.
- **`n_workers > 1` legitimately differs** where a pass has more than one box:
  `z0_sliver` at 4/8/16 workers ends at `max |diff| = 2.26e-1` from the serial
  result — same 0 folds, same `damage = 0`. Two mechanisms:
  1. grow-on-failure is deferred to the end of the batch instead of firing between
     windows, and
  2. with a giant present, the tiler becomes multi-colour Gauss-Seidel instead of
     sequential Gauss-Seidel.

  Both change the iterate path, not the feasibility guarantee.
- **The drift is not small.** On `z16_full` the 4/8/16-worker outputs differ from
  the serial one by up to `max |Δφ| = 27.0` px (the field's own p99 `|φ|` is 37.5).
  Both are feasible (0 folds, `damage = 0`), but they are genuinely different
  feasible points — expected with `objective=NoneObjective()`, which is pure
  feasibility with no anchor to the input. A promotion that enables reordering
  should report `l1_move` / `l2_move`, and callers who care about staying near the
  input should use `L1Objective` / `L2Objective`, which pin the two solutions
  together.
- **Ordering is not free.** An earlier version of the prototype that ran all small
  boxes before the giant (rather than in `find_windows` order) took **1302 s vs
  795 s** on `z16_full` for the same zero-fold result — a 64% penalty purely from
  re-ordering, because `find_windows` can emit boxes that touch. Any batching
  scheme pays some of this; on `z16_full` the parallel path happened to win the
  coin flip (it skipped the mop), which is not something to bank on.

## Amdahl: what stays serial

The intuition going in was that window finding, `min_field` recomputation and
write-back would be the serial tail. **They are not — they are free.** On the full
slice, `foldmap + find + grouping + accounting` totals **0.31 s of 679 s (0.05%)**.
Amdahl's serial fraction here is not bookkeeping, it is *dependency structure*:

| serial component | z16_full w=1 | what it is | parallelisable? |
|---|---|---|---|
| giant tiler | 553.5 s (81%) | 30 overlapping tiles x up to 8 sweeps | partly — 4 colour groups per sweep, sweeps are strictly sequential |
| round-loop windows | 125.7 s (19%) | 4 small boxes + rounds 2..n | barely — 4 boxes in 2 conflict groups |
| terminal mop | 56.5 s (of the above) | 62 clusters, `mop_margin=25` | poorly — 62 clusters collapse to 21 groups |
| grow-on-failure | 0 s serial / 26 s parallel | second attempt on infeasible windows | no (depends on the first attempt) |
| foldmap + find + grouping + accounting | 0.31 s (0.05%) | `min_field`, `find_windows`, colouring, damage accounting | irrelevant |

(`mop` overlaps the `round`/`giant` counters — the mop pass runs windows and giants
through the same timers.)

The hard serial chain is: **rounds are sequential** (each needs the previous
round's fold map), **sweeps inside a giant are sequential** (Schwarz), and inside
one sweep only ~1/4 of the tiles are mutually independent. Multiply that out and
1.9x effective concurrency is close to what the structure allows — a bigger pool
cannot help.

## IPC / pickling cost — negligible

Per-window round trip is **~1 ms**, against solves measured in seconds:

| case | payload pickle | dispatched | `batch_wall` (parent) | `worker_solve` (sum in workers) | overhead |
|---|---|---|---|---|---|
| `z0_sliver` w=4 | 11 445 B | 10 | 62.44 s | 62.43 s | 0.01 s total |
| `z0_cluster` w=4 | 17 205 B | 3 | 18.18 s | 18.17 s | 0.01 s total |
| `z16_twist` w=4 | 39 605 B | 2 | 35.64 s | 35.63 s | 0.01 s total |

The payload is the halo crop (`2 x ph x pw` float64) plus the constraint type,
threshold, objective and `_InnerOpts`; the return is the free region plus a
`WindowRec`. Pool start-up (spawn + `import dvfopt` + `import osqp` per worker) is
0.8 s at 4 workers, 1.1 s at 8, 1.6 s at 16 — paid once, and already amortised by
`dvfopt.core._pool`'s long-lived pool.

**So IPC is not the reason parallelism fails to pay here.** The reason is that
`worker_solve` is ~the same as `batch_wall`: the batches are width-1.

## Design recommendation

**1. Don't make it the default, and don't reach for it first.** 1.6x on a full
slice for a process pool, conflict analysis, a `borders` refactor and a
non-deterministic output is a thin trade; on the crops it is a net loss. Two
cheaper levers move the same wall clock further:

- The **mop pass** dominates sparse-residual work: **62 s of `z0_sliver`'s 102 s
  (61%), to clear 18 folds**. `mop_margin=25` both inflates that cost and destroys
  the mop's parallel width (every mop box on a 50x50 crop covers the whole crop, so
  all 9 clusters conflict). Sizing the mop box relative to its cluster, or capping
  `max_sweeps`, is a one-line change with a bigger payoff than a process pool.
- The **giant tiler**'s `tile=32` / `overlap=2*ring+2` / `max_sweeps=8` set both the
  serial cost and the achievable width. It is the only place with real width, so
  it is where tuning *and* parallelism both pay.

**2. Slice-level parallelism already exists and dominates.** `DVFoptConfig.n_workers`
([dvfopt/unified.py](../../../dvfopt/unified.py)) and `benchmarks/windowed_bench.py`
both fan whole slices out to processes: no coupling, no conflict analysis, near-linear
scaling. For a 528-slice volume that is the right axis and window-level parallelism
adds nothing but nested pools. Window-level parallelism only matters when there is
exactly **one** slice in flight — the GUI's "Run section", a single-slice CLI call,
or the last slice of a batch — and the two must never nest (the existing code already
guards this pattern: see `_coupled_kring_3d.py`'s "no nested pools" comment).

**3. If it is promoted anyway, this is the shape.**

- `n_workers: int = 1` on `windowed_correct`, surfaced on `WindowedWrapperStrategy` /
  `ISQPWindowedStrategy`. **`n_workers <= 1` must take today's code path verbatim** —
  not a pool of size 1 — so serial byte-identity is structural rather than tested-for.
  (The prototype does exactly this; `identical_to_engine=True` on every case.)
- **Reuse `dvfopt.core._pool.get_pool` / `pool_map`**, do not create a per-call
  `ProcessPoolExecutor`. It is already spawn-based, pre-warmed, grow-only, and falls
  back to a serial in-process map on `BrokenProcessPool`. One change needed: its
  `_warmup_worker` JIT-compiles the 3D tet numba kernels, which a 2D windowed run
  never touches — make the initializer family-aware, or add a windowed warm-up
  (import `osqp`, run one tiny `isqp_solve`).
- **Batch granularity = one conflict-free group.** Two windows may run concurrently
  iff neither's free box meets the other's patch (free box ± ring); greedy colouring
  over the round's boxes is ~10 lines (`_conflict` / `_groups` in the prototype).
  The same seam serves all three call sites — round loop, giant tiler sweep, mop sweep.
- **Cap the pool at ~8.** Effective concurrency plateaued at 1.9 from 4 workers up;
  16 workers were 1% faster than 8 and cost more to start. Sizing the pool by
  `min(n_workers, max group size)` avoids spawning workers that only idle.
- **Hoist the CPR colouring cache into the worker initializer.** `_COLORING_CACHE`
  is per-process, so each worker rebuilds it per patch shape; that (plus process
  overhead) is why the same windows cost ~20% more CPU in workers than in-process.
  Tiles within a giant share one shape, so a warm cache is nearly free to arrange.
- **Load balance, don't just batch.** A colour group of 9 tiles finishes in the time
  of its slowest tile, which is why 6.9 windows per batch yielded only 1.9x. If this
  is pursued further, the lever is a continuous scheduler over a conflict graph
  (submit the next non-conflicting window as soon as a worker frees) — the pattern
  `dvfopt/core/slp/cluster_lp_2tri.py` already implements — not a bigger pool.
- **Two small refactors in `_common.py`** the promotion needs:
  1. `build_subproblem(..., borders=None)` — let the caller pass the border flags so
     a worker can be handed a crop directly, instead of the prototype's `ring + 1`
     halo trick.
  2. Factor the tile list out of `_solve_giant_schwarz` into
     `_giant_tiles(box, ring, H, W, tile)`, so the tiler, the mop and the round loop
     all feed one `_run_boxes(phi, boxes, ctx, n_workers)` seam instead of three
     copies of the dispatch.
- **Grow-on-failure stays serial.** It is a second attempt that depends on the first's
  result and fires on a minority of windows; batching it would need a second round-trip
  for no measurable gain.
- **The no-damage invariant is preserved by construction**, not by luck: `touched` is
  marked exactly as today, and conflict-free grouping means no window in a batch reads
  a region another window in that batch writes. Every parallel run measured here
  reported `damage == 0`. Keep asserting it in tests.
- **Determinism:** with `n_workers > 1` the giant tiler becomes multi-colour
  Gauss-Seidel instead of sequential Gauss-Seidel, so the output legitimately differs
  from serial. Tests must assert fold counts / `damage == 0` / no-worse-than-serial,
  and reserve byte-equality for `n_workers == 1`. The gate file is
  [tests/test_windowed_isqp.py](../../../tests/test_windowed_isqp.py) — its
  `test_no_damage_*` cases parameterise cleanly over `n_workers`.

## Reproduce

```bash
python -u benchmarks/parallel_windows_proto.py --selftest
python -u benchmarks/parallel_windows_proto.py \
    --cases z16_twist,z0_cluster,z0_sliver --workers 1,4,8,16 --reference
python -u benchmarks/parallel_windows_proto.py \
    --cases z16_full --workers 1,4,8,16 --reference
```

`z16_full` reads `data/dvfs/b0039/b0039_laplacian_deformation_field.npy` (mmapped,
`vol[1:, 16]`); the crops come from `benchmarks/output/testcases/`. Both are
gitignored. Constraint `SimplexConstraint2DBilinear`, `objective=NoneObjective()`,
`threshold=0.01`, `maxiter=600` throughout. Measured on a 24-core Windows 11 box,
spawn start method, BLAS/OMP pinned to 1 thread per process.
