# Solver optimization analysis — 2D (tri) and 3D (tet) pipelines

Profiling-grounded assessment of where time actually goes in the fold-
elimination solvers, and which optimizations are worth the effort. Driven
by the question "are the proposed optimizations toward the best 2D tri
pipeline?" — answered with cProfile evidence, not intuition.

## TL;DR

Both pipelines are **L-BFGS-B-iteration-bound**, not kernel-bound. The
single Fortran routine `scipy.optimize._lbfgsb.setulb` is **~48% of a 3D
solve and ~35% of a 2D solve**. Therefore:

- **float32 kernels and barrier-gradient sparsification are LOW-ROI**
  (≤~4%): they speed the constraint kernels / adjoint, which are a small
  slice of the total. The kernels are *already* Numba-parallel (3D) /
  fused + sparsity-skipping (both).
- **The real levers reduce L-BFGS-B work:** (1) fewer iterations
  (schedule tuning, cheaper seed), (2) **cross-slice parallelism** (2D),
  (3) a structure-exploiting solver (big effort).
- For the **2D champion `auto_slp`**, the dominant per-slice cost is the
  **serial m14 *seed*** (ALM + L2-refine + barrier, all L-BFGS-B), NOT the
  LP solve. So the seed is the thing to attack, and running slices
  concurrently is the biggest single win because the seed leaves 23/24
  cores idle in the current sequential stage.
- **Accuracy (round 3):** the *local* cluster-SLP champion is near its L1
  ceiling, but a **global GPU untangle achieves ~2× lower L1 on dense
  slices** (95k vs 185k on the z=13 worst case) — at a wall cost. There is
  a real **global-vs-local L1 axis**, not just a speed dial. It is *not* a
  faster method: first-order flow stalls on sliver-folds and needs an
  LP-SLP mop-up. See round 3 below.

## Profiling evidence

### 3D — `correct_dvf_3d` on a 1,326-fold crop (8×80×80), 343 s wall

| component | tottime | % | nature |
|---|---:|---:|---|
| `scipy._lbfgsb.setulb` | 165 s | **48%** | L-BFGS-B optimizer loop |
| `_alm_objective_3d` (obj+grad) | 86 s | 25% | our objective |
| ↳ `tet_grad_T_v` (adjoint) | 14 s | 4% | kernel |
| ↳ `six_tet_volumes_3d` (forward) | 9 s | 3% | kernel |
| ↳ numpy glue (`astype`/`array`/`stack`/`copy`) | ~32 s | ~9% | mostly scipy-internal `x` copies; `np.stack` in `tet_volumes_flat` (~1.4%) is ours |
| sparse harmonic solve (`gssv`) | 2 s | <1% | one-time seed |

Kernels total ~7%. **float32 ceiling ≈ 3.5%.**

### 2D — `cluster_slp_iter` (z=300, n_workers=1, in-process), ~104 s wall

(Every B0039 slice is 320×456 > 5k px, so `auto_slp` always takes the
cluster path. Profiled with n_workers=1 so per-cluster work is captured.)

| component | tottime | % | nature |
|---|---:|---:|---|
| `scipy._lbfgsb.setulb` | 31 s | **35%** | L-BFGS-B (the m14 seed: ALM + L2-refine + barrier) |
| `_soft_pen_objective` (L2-refine / m12) | 4.6 s | 5% | seed stage |
| `_alm_objective` (2-tri ALM) | 1.6 s | 2% | seed stage |
| LP path: HiGHS `check_option` + `gssv` | ~7 s | ~8% | the actual SLP/LP solve |
| tri kernels (`tri_grad_T_v`+`tri_areas_flat`+`_triangle_areas_2d`) | ~2.2 s | ~2.5% | kernels |

**The m14 seed (L-BFGS-B) dominates; the LP solve is minor.** tri-kernel
float32 ceiling ≈ 1.3%.

## Per-optimization verdict (profiling-grounded)

| optimization | helps which path | measured ceiling | verdict |
|---|---|---:|---|
| **Cross-slice parallelism** (2D stage) | the serial seed leaves cores idle | see results below | **DO — top 2D win** |
| **ALM schedule tuning** | `setulb` count (3D M10Tet + 2D seed) | high (setulb is 35–48%) | **DO if feasibility holds** |
| **Cheaper m14 seed** (drop/limit stages) | the dominant 2D per-cluster cost | high | **investigate** |
| Structure-exploiting solver (Gauss-Newton / CyIpopt) | replaces `setulb` entirely | could be transformative | **high effort — plan only** |
| `np.stack` removal in `tet_volumes_flat` | 3D forward glue | ~1.4% | trivial, free, do it |
| **float32 kernels** | constraint kernels | ~3.5% (3D), ~1.3% (2D) | **low ROI — skip** |
| **barrier_grad_rtol** into M10Tet | adjoint (already sparsity-skips) | ~4% (3D) | **low ROI — skip** |

The last two were the headline items of the original (3D-framed) list;
the profile shows they target a part of the solve that is already small.

## Benchmark results

### Cross-slice parallelism (12 B0039 slices, 24 cores) — only 1.11×, NOT a win

| config | wall | feasible |
|---|---:|---:|
| A: sequential slices, inner `n_workers=16` (current) | 162.9 s | 12/12 |
| B: 12 slices concurrent, inner `n_workers=1` + numba pinned | 147.0 s | 12/12 |

**Speedup 1.11× — the hypothesis was wrong.** The n_workers=1 *profile*
made the m14 seed look serial-dominant, but in the real n_workers=16 path
`cluster_slp_iter` runs the seed **per-cluster inside its 16-worker pool**
— so the seed is already parallelized across clusters and the baseline
already fills the cores. Reorganizing parallelism across slices instead of
clusters is a wash. **auto_slp is already well-parallelized; the lever is
NOT more parallelism — it's fewer L-BFGS-B iterations (schedule/seed/
solver).**

### Within-slice parallelism scaling + L1 (the real auto_slp axis)

auto_slp already parallelizes *inside* a slice (`cluster_slp_iter` solves
fold-clusters concurrently). Sweeping `n_workers` on single slices,
tracking **L1 too**:

z=300 (2,360 folds): 1w 89.3s → 2w 1.73× → 4w 2.61× → 8w 3.72× → 16w
4.98× → 24w 5.11×. z=450: 1w 59.8s → 8w 3.60× → 24w 3.82×. **L1 invariant
across worker count** (≤0.41%) — the frozen-ring decomposition is
deterministic; parallelism costs nothing in quality.

Amdahl fit gives a **serial fraction ≈ 0.15–0.18 → hard ceiling ~6×**; the
pipeline reaches 85% of its own ceiling at 24 workers. Efficiency falls
87%(2w) → 31%(16w) → 21%(24w); 16→24 buys almost nothing. The default
`n_workers=16` sits at the efficiency knee — well-chosen. Parallelism is
bounded by cluster count: a single-big-cluster slice gets ~none.

### Continuous (as-completed) scheduler — prototyped, +1.01–1.16×

Added an opt-in `scheduler='continuous'` to `cluster_slp_iter`: instead of
barrier sub-rounds (a slow cluster idles workers that finished), admit any
non-conflicting cluster as a worker frees. Recovers the barrier-idle slice
of the serial fraction:

| slice | clusters | subround | continuous | speedup | L1 Δ |
|---|---:|---:|---:|---:|---:|
| z=300 | 263 | 17.8 s | 16.9 s | 1.05× | −0.04% |
| z=200 | 136 | 12.0 s | 10.3 s | 1.16× | −0.12% |
| z=12 | 13 | 55.8 s | 55.1 s | 1.01× | 0.00% |

Real but modest, L1-identical, feasibility preserved. Best on cluster-rich
slices; ~nil on dense slices (z=12 merges into 13 big clusters → little
idle to recover). It's the only *parallelism* win left, and it's small —
consistent with the pipeline already sitting near its Amdahl ceiling.

### ALM schedule tuning (small 3D crop, 180 folds) — inconclusive

| config | wall | feasible | min_T |
|---|---:|---:|---:|
| baseline (inner=200, outer=60, rho_g=5) | 26.8 s | ✓ n<thr=0 | +0.0100 |
| inner=50 | 21.7 s | ✓ | +0.0100 |
| inner=100 | 43.2 s | ✓ | +0.0100 |
| inner=100, rho_g=10 | 44.4 s | ✓ | +0.0100 |

inner=50 was nominally 1.24× faster with identical feasibility, but
inner=100 came out *slower than both* inner=50 and inner=200 — a
non-monotonic result that means **timing is noise-dominated** on this
crop. All configs hit the same `outer_used=60` cap and the same converged
min_T, i.e. the inner cap mostly doesn't bind and the outer loop runs to
budget regardless. So schedule tuning is at best a **modest ~1.2× nudge**,
not reliable or large; confirming it needs multi-trial timing on a larger
crop. The L-BFGS-B step count is set by the problem, not the caps —
reinforcing that only a different *step* (sparse GN/IPM) changes the 35–48%.

### Gauss-Newton ALM inner step — PROTOTYPED, negative result

Built `augmented_lagrangian_2d_gn` (`_gn_alm_proto.py`): same PHR outer
loop, but the inner L-BFGS-B replaced by a sparse GN step
`(I + ρ Jₐᵀ Jₐ) δ = −g` (active rows only) + Armijo. Head-to-head vs the
L-BFGS-B ALM on fold crops:

| crop | L-BFGS-B | GN | result |
|---|---|---|---|
| z=300 (14 folds, min_T −1.6) | 0.38 s, feasible, L1 10.29 | 5.1 s, feasible, L1 10.29 | GN **13× slower**, same quality |
| z=12 (4432 folds, min_T −46) | 50 s → n_neg 197 | 65 s → n_neg **2555 (diverged)** | GN **fails** on severe folds |

GN converges to the *identical* optimum on mild folds (L1 matches) but is
slow (no trust region → tiny Armijo steps on the bilinear area
constraint), and it **diverges on severely folded input** (the
linearization is invalid far from feasibility). Both pathologies are fixed
by a **trust region** — which is *exactly* what the project's **SLP /
`auto_slp` already is**: a trust-region linearized solve on this same
Jacobian (`build_sparse_jacobian_T` → HiGHS L1-LP step). 

**So the "sparse structure-exploiting solver" is not an untapped lever —
it is already deployed as the champion (auto_slp/SLP).** A naive
GN-on-ALM is strictly dominated by it. The L-BFGS-B that remains lives in
the m14 *seed*, performing robust penalty-homotopy from severe folds
(min_T −46) that neither the GN step nor the LP can start from — it is
there for robustness, not for lack of optimization.

## Overall conclusion

**The pipelines are genuinely well-optimized; there is no large untapped
speedup.** Every lever was prototyped/measured and came up small or
negative:

| lever | result |
|---|---|
| cross-slice (cross-volume) parallelism | 1.11× — auto_slp already within-slice parallel |
| within-slice scaling | ~5× at 16w, at 85% of a ~6× Amdahl ceiling; L1-invariant |
| continuous (as-completed) scheduler | **prototyped, +1.01–1.16×, L1-identical** (only parallelism win left) |
| ALM schedule tuning | ~1.2× nominal, noise-dominated |
| float32 kernels / barrier_grad_rtol | ≤3.5% / ≤4% — target the already-small kernel/adjoint |
| Gauss-Newton inner step | **prototyped, negative** — slower + diverges; SLP already is the trust-region structure solver |

The kernels are Numba-parallel/fused with sparsity skips; auto_slp routes,
cluster-parallelizes, and *is* the sparse structure-exploiting solver. The
recommended, shippable change is the **opt-in continuous scheduler**
(~1.1×, safe). Beyond that, materially faster 2D/3D would require a
genuinely better *trust-region/IPM* solver than the current SLP+seed combo
— and the prototype shows the obvious GN formulation does not clear that
bar; the seed's L-BFGS-B is doing necessary robustness work, not waste.

## Experimental round 2 — speed + accuracy levers

Shipped: the continuous scheduler is now the `auto_slp` default
(`_compare.py`, `scheduler='continuous'`; 32 tests pass).

### Seed-cost sweep — no free lunch (the seed buys L1 via basin selection)

`cluster_slp_iter(inner_seed=...)`, z=300, continuous, 16 workers:

| seed | wall | n_neg | L1 |
|---|---:|---:|---:|
| harmonic | 17.0 s | 0 | 17 960 |
| m10 | 21.4 s | 0 | 14 761 |
| m14_quick | 22.0 s | 0 | 5 748 |
| **m14_fast (default)** | 34.6 s | 0 | **2 077** |

Cheaper seeds are ~2× faster but **8×+ worse L1**. The final L1 is
basin-determined by the seed (the SLP loop only locally refines), so the
seed is doing the L1 work, not wasting time. **Default m14_fast is
accuracy-optimal; rejected.**

### L1-optimality gap — slice-dependent (and the saving is too)

Clustered `auto_slp` vs global `slp_iter`, both to strict feasibility:

| slice | folds | clustered | global | saving | L1 gap |
|---|---:|---:|---:|---:|---:|
| z=450 (moderate) | 1 841 | 22 s | 404 s | **18.2×** | +13.5% |
| z=12 (worst/dense) | 8 978 | 298 s | 921 s (**n_neg=4, infeasible!**) | **3.1×** | +1.9% |

**The gap AND the speedup are inversely related to fold density**, opposite
to the naive expectation:
- **Sparse/moderate** → folds in separated pockets → many small
  *independent parallel* LPs → huge speedup (18×) but frozen-ring
  boundaries cost L1 (+13.5%).
- **Dense (worst case)** → folds pervasive → few big clusters that nearly
  *are* the whole slice → modest speedup (3×) but near-optimal L1 (+1.9%),
  and the giant global LP fails to even converge (n_neg=4 after 921 s).

So clustering is **strictly better across the whole spectrum** — faster
everywhere, and the only one that reaches feasibility on the worst case.
The +13.5% is a sparse-slice phenomenon, and recovering it is not cheap:

- **Naive global L1 polish (warm-started): BREAKS feasibility.** +1 step
  L1 2372→2191 but n_neg 0→33; +3 steps L1→2083 (gap −0.3%) but n_neg=40.
  A fixed-trust linearized step overshoots the true constraint; a
  feasibility-preserving polish needs the full adaptive-trust SLP (≈ the
  404 s global solve). No cheap feasible polish exists.
- **merge_dilation sweep:** z=450 md=1 looked like a free win (L1
  2372→2180 at equal wall), but multi-slice verification showed it is
  **slice-dependent and noise-dominated** (md=1 wins z=300/450, loses
  1.5–3.7× on z=100/400/500; same-config wall varied 13.7 s↔55.7 s).
  Default md=2 stands. md=8 reaches global-optimal L1 but at 9× wall.

### Net: accuracy is a speed DIAL, not a free lever

The L1/speed tradeoff is a genuine frontier, not an inefficiency:

| operating point | wall (z=450) | L1 | vs global |
|---|---:|---:|---:|
| clustered (default) | ~22 s | 2 372 | +13.5% |
| merge_dilation=8 | ~120 s | 1 998 | −4.4% |
| global slp_iter | ~404 s | 2 090 | optimal |

Could be exposed as an `accuracy={'fast','optimal'}` knob on `auto_slp`
(cluster vs global), but there is **no point on the curve that is both
faster and more accurate** than the current default for a given slice.

## Experimental round 3 — independent-2D levers + global GPU untangler

Scope: **single, unrelated 2D slices** (no cross-slice/temporal tricks —
each slice an independent dataset). Two questions: cheap intra-slice levers,
and whether a fundamentally different *method* beats SLP+seed.

### Intra-slice tuning levers — all rejected

- **Elastic (seedless) SLP routing.** Route mild-fold clusters to the
  seedless elastic LP (skip the m14 seed). Rejected: elastic only reaches
  feasibility on **69% of mild clusters, ≤22% of medium/deep**
  (`_bench_depth_routing.py`) — too unreliable, needs a seed fallback
  anyway, and the wall saving is marginal.
- **Depth-adaptive seed** (m10 for shallow clusters, m14_fast for deep).
  A single-crop probe looked promising (m10 2× faster at negligible L1),
  but the whole-slice aggregate (`_bench_adaptive_seed.py`, 80 clusters ×
  3 slices) rejects it: **+123% to +436% L1 for only ~1.2× speedup**. m10
  lands a worse basin even on shallow-*depth* clusters once the cluster is
  non-trivial. Fold *depth* ≠ L1 cost.

Confirms round-2's finding from the other direction: the m14-seeded cluster
SLP's per-cluster basin is not cheaply reducible.

### Global GPU untangler — NOT faster, but ~2× lower L1 on dense slices

A first-order **whole-slice** untangler on the GPU (Adam on a smooth
penalty / PHR augmented-Lagrangian over all 2-tri areas at once, no
clustering — `algorithms/_gpu_untangle.py`, `_bench_gpu_untangle.py`,
`_bench_gpu_alm.py`). Torch areas verified against `tri_areas_flat`
(max|Δ|=4e-14).

| slice | method | wall | folds | L1 | vs champion |
|---|---|---:|---:|---:|---:|
| z=300 mild (2360) | GPU-ALM→mop | 86 s | 0 | 3 424 | +65% L1, 3× slower |
| | champion | 27 s | 0 | 2 080 | — |
| z=13 dense (9057) | GPU-penalty→mop | 283 s | 0 | 155 988 | −16% L1 |
| | GPU-ALM→mop | 341 s | 0 | **95 137** | **−49% L1**, 4.6× slower |
| | champion | 74 s | 0 | 185 131 | — |

Two firm conclusions:

1. **Not a faster method.** First-order flow (penalty *and* ALM) **stalls
   on sliver-folds** — near-collapsed triangles have degenerate area
   gradients, so Adam starves exactly where feasibility is hardest
   (worst_g plateaus at ≈−0.02 to −0.2). Feasibility always requires an
   LP-based SLP mop-up; the champion's SLP is the speed frontier because
   its LP linearization + exact-feasibility acceptance resolves slivers
   that gradient descent cannot.
2. **A better-L1 method on dense fields — substantially.** A *global*
   untangle lands ~**2× lower L1** than the *local* cluster-SLP champion
   on the dense worst case (95k vs 185k). This **updates round-2's
   conclusion**: the "no point both faster and more accurate" frontier and
   "L1 basin-fixed" claims hold for the *local* SLP family, but a global
   solve breaks the local ceiling on dense slices. The local cluster
   decomposition (frozen-ring boundaries, per-cluster basins) sacrifices
   global L1 optimality — invisible when folds are sparse, but worth half
   the L1 when they're pervasive. Note the earlier *CPU* global LP could
   not even reach feasibility on dense (n_neg=4 after 921 s); the GPU
   global untangler does, and at far lower L1.

**Can the global untangler reach strict feasibility standalone (no mop)?**
Tested four formulations (`algorithms/_gpu_untangle.py`), aiming to kill the
expensive SLP mop-up:

| formulation | start | 0 folds standalone? | dense L1 |
|---|---|---|---|
| quadratic penalty | φ_in | no — stalls ~2650 slivers | 156k basin |
| PHR aug-Lagrangian | φ_in | no — stalls ~1408 slivers | 95k basin |
| Adam→L-BFGS-ALM polish | φ_in | no — worst_g μ-invariant at −0.02 | — |
| **log-barrier (interior pt)** | **identity** | **YES — 0 folds, no mop** | 477k |
| shifted/relaxed barrier | φ_in | no — slack-floor kills the untangling gradient; un-floored freezes on stiff gradient | — |

**Conclusion — a structural wall, not a tuning gap.** Standalone GPU
feasibility *is* reachable, but only by the log-barrier from a **feasible
identity start**, which lands in a high-L1 basin (dense 477k ≫ champion
185k) — feasible but useless. Every formulation that starts from the
low-L1 φ_in basin **fails to reach strict feasibility**: the last
~1400 sliver-folds are near-collapsed triangles needing *coordinated
multi-vertex moves* that (a) the data term resists and (b) first-order /
quasi-Newton dynamics cannot organize. A log-barrier can't even start from
φ_in (infeasible → log undefined); the shifted-barrier relaxation to fix
that either freezes the guard (stiff gradient) or, once slacks are floored
to stay defined, makes the shifted margin constant in Aₖ so the barrier
exerts **zero** untangling force. The champion's SLP resolves these slivers
because an **LP step coordinates the whole cluster and accepts only
strictly-feasible moves** — that is exactly what a first-order untangler
lacks.

**So the division of labour is the answer, not a workaround:** the GPU
untangler is a superb **low-L1 seed**, and the SLP is the **feasibility
solver**. `GPU-ALM(from φ_in) → SLP mop` reaches strict feasibility at
**~2× lower L1** than the champion on dense (95k vs 185k), at ~4.6× wall —
a genuine **high-accuracy mode** (L1 is the standing objective), just not a
speedup. Making the mop cheap would require the GPU stage to leave fewer
residual slivers, which the wall above shows first-order methods cannot do.

## Final bottom line (after three experimental rounds)

The 2D/3D fold-elimination pipelines are at a **strong local optimum for
speed**, and the local SLP family is near its L1 ceiling — but a *global*
solve beats that ceiling on dense fields. Across ~13 prototyped/benchmarked
levers, the only clean speed win was the **continuous scheduler (~1.1×, now
default)**; the notable accuracy finding is the **global GPU untangler's
~2× dense-slice L1 reduction** (at a wall cost).

- speed: parallelism is saturated (within-slice cluster pool, ~85% of a
  ~6× Amdahl ceiling); seed/float32/grad_rtol/GN/elastic-routing/
  depth-adaptive-seed all rejected. First-order GPU untangling is slower
  (sliver stall → SLP mop-up).
- accuracy: the local cluster L1 gap is a genuine speed/accuracy dial
  (9–18× wall for the sparse-slice +13.5%); **separately, a global GPU
  untangle achieves ~2× lower L1 than the local champion on dense slices**
  — a global-vs-local axis, not a tuning knob.
- the one true speed step-change (a better trust-region/IPM solver than
  SLP+seed) is large and uncertain; the GN prototype was dominated, and
  osqp/cyipopt/clarabel are not installed to test a sparse IPM here.

## Full-volume per-slice validation (shipped continuous default)

Ran the shipped `auto_slp` (continuous scheduler) on **all 528 B0039
slices** (`_bench_all_slices.py` → `output/all_slices_auto_slp_continuous.csv`):

- **Feasibility: 528/528 strictly feasible (n_neg=0), 0 residual folds.**
- **Total wall: 4.91 h** (mean 33.5 s, median **16.1 s**, max 918.6 s).
- Total L1: 4 556 881 (mean 8 630 / slice).
- Slowest 10 slices are all the dense band z0–11 (533–919 s each); they
  dominate the run (~10 slices ≈ 1.9 h of the 4.91 h). The median 16 s
  shows the sparse/moderate majority is fast.

This validates the new default end-to-end: every slice corrected to strict
2D feasibility, with the dense band (z0–17) the only expensive region —
exactly the per-density behaviour the L1-gap analysis predicted. The
corrected per-slice DVFs are also saved fresh as
`b0039_FULL_stage1_continuous.npy` (3, 528, 320, 456; dz=0, corrected
[dy,dx]) — a drop-in for the 3D pipeline.
