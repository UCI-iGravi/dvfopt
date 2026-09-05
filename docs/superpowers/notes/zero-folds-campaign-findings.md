# Zero simplex folds from the DVF alone — campaign findings (2026-08-27 … 08-29)

Reference document for the paper. Everything here was measured on this repository
(PRs #85–#94 and the branches named below); numbers are quoted as measured, with
the box load noted where it inflates wall time. Data artefacts are gitignored
under `benchmarks/output/`; the paths are listed in §9.

## 1. Problem statement

Given a 2D displacement field `phi = [dy, dx]` (backward map) from a Laplacian
correspondence interpolation, produce a field with **0 simplex folds** — every
triangle of the piecewise-linear interpolant on the fixed grid triangulation has
area ≥ 0.01 — using only the field itself (no correspondences, no images),
moving as little as possible, and never creating folds in untouched area
(the *no-damage* invariant). The stronger certificate used throughout is the
**bilinear** gauge (both diagonals, 4 triangles per cell); bilinear feasibility
implies simplex feasibility.

The engine is `dvfopt.core.windowed.windowed_correct`: fold clusters are
solved in frozen-ring windows by an elastic-QP SQP (I-SLSQP: OSQP / Clarabel
subproblems, exact merit line search), with a giant-region Schwarz tiler, a
coarse-to-fine warm start and a terminal mop. Recipe:
`correct_dvf(phi, constraint='bilinear', strategy='isqp_windowed', objective='none')`.

## 2. Headline results

| data set | slices | 0 simplex folds | 0 bilinear folds | damage 0 | folds before → after |
|---|---|---|---|---|---|
| B0039 Laplacian-exterior, **full resolution** (320×456) | 528 | **528 / 528** | 528 / 528 | 528 / 528 | 599,313 → **0** |
| B0039 Laplacian-exterior, 2× downsampled (160×228) | 528 | **528 / 528** | 528 / 528 | 528 / 528 | 64,085 → **0** |
| 7-brain cohort, sampled Laplacian slices (pre-fix engine) | 146 | 139 / 146 | — | 146 / 146 | — |
| the 7 cohort residual slices, from raw, fixed engine | 6 run | **6 / 6** | 6 / 6 | 6 / 6 | e.g. 29,699 → 0 |

Full-resolution run: 10.6 h pool wall on 4 workers (42.4 h serial); per-slice
median 132 s, p90 243 s; L2 move median 46.9; 26 % of pixels moved (median).
The eight volume-edge slices (z 0–7, 3.3–4k folds) took 1–4.2 h each and are
42 % of the serial time (§6).

Engine used for those runs: `main` after PR #92 (re-seed stage) — commit
`39f661f`; PR #94 (orientation rows, off by default) does not change defaults.

### 2b. Certification of the final formulation (edge rows + in-solve L2)

Second full-resolution sweep of the B0039 Laplacian-exterior volume
(`benchmarks/output/b0039_ext_full_v2/`, engine = PR #99's default, 4-worker
pool): **528/528 slices at 0 simplex / 0 bilinear / 0 finite folds, damage 0**
(599,313 → 0), never a re-seed. Against the first certification (v1, the re-seed
path), per slice:

| | v1 (re-seed path) | v2 (edge rows + L2) |
|---|---|---|
| slices at 0 folds / damage 0 | 528 / 528 | 528 / 528 |
| L2 move, sum over slices | 51,835 | **39,047 (−25 %, smaller on 527/528)** |
| SQP iterations, total | 210,522 | 312,448 |
| wall, hard slices (≥ 2000 folds, n = 17) | 78,623 s (median 2725 s) | **9,119 s (539 s)** |
| wall, ordinary slices (500–2000 folds, n = 506) | 73,639 s (median 130 s) | 247,640 s (437 s) |
| pool wall | 10.6 h | 17.9 h |

The fidelity gain is uniform (median L2 ratio 0.75 in every fold tier) and the
hard slices are 8× faster, but ordinary slices cost 3.4× — the v2 walls were
also inflated by the diagnosis runs sharing the machine. That cost is the
subject of §4.3c: it was the L2 objective parking on the active rows at ADMM
precision and the inner's -1e-6 feasibility test failing fold-free windows into
the ladder; PR #103 (margin-consistent window feasibility + `ftol`) removes
35–36 % of the ordinary-slice iterations at identical fidelity.

**Speed of the merged engine** (main after #103), 22-slice sample (every 24th
z), 4-worker pool running alone, same slices from the v1 and v2 certifications
(`benchmarks/output/b0039_ext_full_v3_sample/`, `docs/paper_figures/v1_v2_v3_speed_sample.json`):

| 22 slices | v1 (re-seed path) | v2 (rows + L2, pre-#103) | **v3 (main)** |
|---|---|---|---|
| 0 folds / damage 0 | 22 / 22 | 22 / 22 | 22 / 22 |
| SQP iterations | 7,775 | 13,061 | **7,782** |
| L2 move (sum) | 3,697 | 2,700 | **2,697 (−27 %, smaller on 22/22)** |
| wall (sum of slices) | 4,997 s | 11,957 s | **6,932 s** (0.58× v2 on 22/22; 1.39× v1) |
| hard slice z=0 (3951 folds) | 1,938 s | 814 s | **711 s** |
| ordinary slices (n = 21), median | 128 s | 483 s | **281 s** |
| seconds per SQP iteration, ordinary | 0.53 | 0.95 | 0.94 |

So the iteration inflation is gone (v3 has v1's iteration count), the fidelity
gain is kept, the hard slices are 2.7× faster than the re-seed path, and the one
cost that remains is intrinsic to the objective: an L2 window's QPs are ~1.8×
dearer per iteration (ADMM median ~1060 vs ~640; most solves escalate to
Clarabel), which makes ordinary slices ~2× the re-seed path. `objective='none'`
with the edge rows remains the fastest formulation (z=440: 356 iterations vs
old 433) at the old fidelity.

That per-iteration cost has one lever (PR #106): the ADMM cap. With
`qp_max_iter` 2000 → 1000 the SQP absorbs the capped solves — equal-contention
A/Bs: z=240 −13 % wall (223 vs 257 s), volume z16 −37 % (333 vs 529 s), z=440
−9 %, the hardest slice z=2 −6 % — at the identical SQP-iteration count, L2 move
and 0 folds / damage 0 everywhere; the crop pack holds. 500 is faster still on
ordinary slices but inflates the hard slice to 2013 iterations (+66 %) and the
L2 sliver crop +28 %, so 1000 is the default. Measured dead ends on the same
slice: Clarabel handoff at 400 / 1500, no cold IP solve, caps 750 / 4000,
OSQP-only (the hybrid earns its keep: 539 vs 481 s).

## 3. The mechanism behind every residual: the rotated orientation branch

Before the fix, five cohort slices plateaued with 5–79 residual cells that no
solver setting could move (exact vs trust-region step rule, trust radius,
backend, patience, no bail, bigger windows, SLSQP inner, harmonic re-seed via the
library helper, pairwise/identity "kicks" — all measured, see §8).

**Where they are.** 29/29 residual clusters sit within 4 px of a *prescribed
Laplacian correspondence pin* whose displacement disagrees with its neighbours
(base rate for same-size random boxes: 5–9 %). Worked example, B0304
`laplacian_all` z=181: the cluster at y 111–114, x 387–389 sits on five pins,
three of them fixed (112,389), (113,389), (114,389) → the same moving pixel
(159,246), displacement `dx = −143` against a slice median of +30 (174 px off).
On B0039 `laplacian_all` z=11 the slice carries 37 pins, 14 of which (rows
223–228, cols 255–260) map to one moving pixel (101,288).

**What the solver does with them.** Per-iteration traces (`isqp_solve(trace=)`)
show every covering window call ending `a-collapse` / `linesearch-stall` /
`tr-collapse` with the exact line-search step `a* → 0`, even in 72×71 windows
with 9,660 free pixels. The geometry at the plateau (z=181, image coordinates
`y+dy`, `x+dx` of rows 110–116, cols 386–392):

```
image x: [293.7 294.6 298.0 300.1 305.1 307.4 309.5]     image y: [130.3 130.3 130.2 130.3 130.4 130.6 130.9]
         [286.7 290.2 299.8 299.8 311.0 311.1 311.3]              [130.3 130.6 130.8 130.8 131.1 131.1 131.1]
         [286.2 287.7 292.8 261.9 303.6 310.4 311.0]              [130.3 130.5 130.7 130.0 130.9 131.4 131.7]
         [279.4 279.4 279.4 257.1 303.0 310.2 312.3]              [130.4 130.4 130.4 129.9 130.9 131.5 132.6]
```

Four grid rows are squeezed into ~0.5 px of image y, and the pin column (389)
maps 20–45 px *left* of both neighbours. The bilinear area of such a cell is a
product `Δy·Δx` with **both factors negative** — the cell sits on the 180°-rotated
orientation branch: locally fold-free by the sign test, but not joinable to the
surrounding un-rotated field. The seam between the two branches is a *maximum*
of the feasibility merit (passing between them means passing through zero area).
Direct probes confirm it: splitting the glued pins in y, or shifting the pin
column into its neighbours' interval, raises the merit **linearly** — every
axis-aligned move is uphill. A local (descent) method therefore cannot cross;
this is the "twist-lock / disconnected feasible set" obstruction, and it is the
solver's own trap, not a property of the data (the correspondence pins are only
where the input is tangled enough to fall into it).

## 4. The fixes

### 4.1 Terminal harmonic re-seed (PR #92, default on) — *repair*

After the round loop plateaus, each residual cluster's neighbourhood (its cells'
corner pixels dilated by 2) is replaced by the discrete-harmonic interpolation of
its ring (one small sparse Laplace solve), and the engine polishes the re-seeded
field. The ring is on the correct branch, so the refilled interior is too.

| plateaued slice | residual (simplex / bilinear) | after re-seed + polish | wall |
|---|---|---|---|
| B0304 lap_all z181 | 5 / 7 | **0 / 0**, damage 0 | 16 s |
| B0304 lap_ext z128 | 8 / 8 | **0 / 0** | 12 s |
| B0039 lap_all z11 | 49 / 53 | **0 / 0** | 10 s |
| B0039 lap_ext z1 | 39 / 47 | **0 / 0** | 40 s |
| B0032 lap_ext z1 | 70 / 79 | **0 / 0** | 18 s |

From raw, the same six slices (incl. ds2 z1) reached 0 with 1 re-seed round of
34–357 pixels each. Fidelity vs the plateaued field: z11 L2 1017 vs 929
(+9.5 %); z181 1870 vs 1869; z128 1395 vs 1390. Byte-identical wherever the mop
already cleared (z16_twist, raw z16); it also removed the single bilinear row the
`z0_cluster` crop always ended short of.

Ordering: the re-seed stays *after* the mop — see §6 for why the reverse order was rejected.

### 4.2 Linear orientation rows (PR #94, `orientation_delta`, opt-in) — *prevention*

Every deformed grid edge keeps a projection ≥ δ on its own direction, plus the
anti-diagonal convexity rows of `dvfopt.jacobian.monotonicity`. A rotated cell
violates them, so the QP never enters that branch; the rows are linear, hence
exact in the QP (no thin-cell linearisation error). Measured from raw with the
re-seed off (loaded box):

| slice | plain engine | rows on every window | L2 vs re-seed path |
|---|---|---|---|
| B0304 ext z128 (8956) | 8 left | **0**, 2 rounds / 200 windows, 3643 s | +0.7 % |
| B0032 ext z1 (4556) | 70 left after 8902 s / 374 windows | **0**, 1 round / 31 windows, 1066 s | +0.8 % |
| B0039 lap_all z11 (4633) | 49 left | **0**, 1 round / 37 windows, 1604 s | +54 % |
| B0039 ext z1 (3957) | 39 left | **0**, 2 rounds / 47 windows, 2466 s | +1.7 % |
| B0304 lap_all z181 (29699) | 5 left | **0**, 1 round / 178 windows, 6319 s | +28 % |
| full-res B0039 z=2 (3909) | 0 in 10168 s / 149 windows | **0**, 2 rounds / 53 windows, 1577 s | +1 % |

Full-resolution edge slices z 0–7 (3.3–4k folds each) with the rows on every
window (`none` objective, re-seed off — it never fired):

| z | plain engine: wall / windows / SQP iters / L2 | rows: wall / windows / SQP iters / L2 |
|---|---|---|
| 0 | 1938 s / 78 / 1993 / 2686 | 2231 s / 81 / 1331 / 2766 |
| 1 | 15062 s / 187 / 6999 / 2878 | **2145 s** / 68 / 973 / 2847 |
| 2 | 10168 s / 149 / 4610 / 2732 | **1569 s** / 53 / 1024 / 2760 |
| 3 | 12212 s / 193 / 6854 / 2652 | **2159 s** / 78 / 1372 / 2707 |
| 4 | 13981 s / 181 / 7246 / 2382 | **4247 s** / 78 / 1236 / 2560 |
| 5 | 3947 s / 97 / 3392 / 2315 | 5111 s\* / 64 / 1269 / 2327 |
| 6 | 2896 s / 142 / 5312 / 2014 | 4945 s\* / 81 / 1320 / 2128 |
| 7 | 4037 s / 187 / 7759 / 1711 | 5415 s\* / 82 / 1554 / 1893 |

\* solved while ~20 other solver jobs shared the box; the iteration counts are the
load-independent signal. Edge total 17.8 → 7.7 serial hours (2.3×); the trapped
slices 3.3–7×; fidelity +1–10 % L2 with the `none` objective.

The rows exclude legitimately fold-free cells rotated by > 90°, which is the
fidelity cost on the hardest slices; hence opt-in. A *rung* variant (rows only on
windows the ladder fails) was measured and rejected (z128: 12,526 s, never solved
a window on its own).

### 4.3 Two-phase: monotone untangle then polish (branch `monotone-untangle`)

Because the orientation rows are linear, "solve for monotonicity first" is a
**convex** problem: one sparse QP, `min ½‖phi − phi_in‖²` s.t. the rows, no
windows, no SQP, cannot fail (the identity is feasible). Whole-slice QP on
full-resolution z=2: 581,354 rows × 291,840 variables, OSQP 775 iterations,
**304 s**; then the ordinary engine from that point: **45 s, 7 windows → 0
folds, damage 0**. Total 349 s vs 10,168 s for the plain engine (29×), at
**L2 2270 vs 2732 (−17 %)** — faster *and* closer to the input.

On the hardest cohort slice, B0039 lap_all z=11 (4633 folds; re-seed path from
raw: 14,372 s on a loaded box, L2 1017; rows-always 1604 s, L2 1566): phase 1 at
δ = 0.1 (215 s) + engine polish (362 s, 22 windows) = **577 s → 0 folds, damage 0,
L2 1350**; at δ = 0.05: 218 + 434 s, L2 1306. No re-seed fired in either polish.

Caveat found on the way: the monotonicity rows do **not** by themselves imply
positive cell areas. A *dart* cell (fourth corner pulled inside the triangle of
the other three) satisfies the row/column monotonicity and both anti-diagonal
conditions while one of its triangles is inverted — the library docstring's
convexity claim is wrong. Phase 1 therefore leaves 1–2k such cells (z=2:
3935 → 1076 bilinear folds, min −2.2), which are ordinary proper-basin problems
the SQP clears in seconds. A bounded-shear ("cone") variant of the rows —
projection ≥ δ *and* perpendicular component ≤ κ × projection — is a sufficient
linear condition (area ≥ δ²(1−κ²)/(1+κ²)). Measured on z11 at δ = 0.2, κ = 0.5:
the single whole-slice QP (873,192 rows, OSQP 3075 iterations, 2124 s on a
loaded box) returns **0 simplex / 0 bilinear folds directly** (min area 0.015),
phase 2 has nothing to do — guaranteed feasibility from one convex solve — at
L2 1643. A wider cone (δ = 0.25, κ = 0.8; 4725 iterations, 3172 s) also returns 0
folds directly at **L2 1274** — on par with the two-phase monotone path
(1306–1350), better than rows-always (1500–1566), 25 % above the re-seed path
(1017): guaranteed feasibility from a single convex solve.

**Not a blanket pre-pass.** Run on every field at δ = 0.1 it fails the fidelity
gate where the plain engine is already cheap: raw B0039 z16 (solved by the plain
engine in 807 s at L2 268) comes out at L2 596 in 833 s; the `z0_sliver` crop at
L2 422 vs 21.5; `z0_cluster` +58 %. The δ spacing it enforces is ten times the
threshold's own scale, and ordinary folds need far smaller moves. The count of
rotated cells in the *input* does not identify the trapped slices either (raw
z16 has 150, the trapped edge slices 22–50), so the stage is an opt-in for
trapped fields (`untangle_delta`), not a default; a δ at the threshold scale
(0.01–0.03) is being measured.

The engine integration restricts the QP's variables to the fold neighbourhoods
(`find_windows` boxes; everything else fixed), which keeps it tiny on ordinary
slices, and books the moved pixels as `touched` exactly like the coarse warm
start (no-damage unchanged).

### 4.3b Which rows, which objective — the fidelity ledger

Measured L2 move vs the raw input (0 folds in every cell of the table):

| formulation | raw z16 (plain 268) | full-res z=2 (plain 2732) | z11 (plain plateau 929, 49 folds) |
|---|---|---|---|
| re-seed path (default engine) | 268 | 2732 | 1017 |
| rows (all) + `none` | — | 2760 | 1566 |
| rows (all) + in-solve L2 | 524 | 2441 | 1306 |
| rows on folded cells only (+1 ring) + `none` | — | 2575 | 1315 |
| **edge rows only** + `none` | 317 | **2327** | 980 (re-seed fired once) |
| **edge rows only + in-solve L2** | **264** | **1979** (2039 s vs 10,168 s) | **783** (2595 s) |
| two-phase monotone QP + polish | — | 2270 | 1306–1350 |
| minimal engine (rows all) + L2 | — | 2502 | 1153 |
| whole-slice cone QP (δ .25, κ .8) | — | — | 1274 |

Reading: the **anti-diagonal convexity rows** are the fidelity cost of the rows
(z11 1566 → 980, z=2 2760 → 2327 when dropped); the monotone edge rows alone
reach a *better* basin than the plain engine on the trapped slices (z=2 −15 %)
and cost +18 % on an ordinary slice (raw z16) with `none`; the in-solve L2
objective pulls every rows variant back toward the input (z11 1566 → 1306 for
the full rows). The crop pack cannot gauge the rows' fidelity: `z0_sliver` was cut
from an engine *output* and contains 147 fold-free rotated cells (the
rotated-branch artefact); the rows un-rotate them, which reads as a large L2
against that artefactual start.
Measured against the *raw* z0 input over that box: the crop's start field is at
L2 1130, the plain engine's output 1128, the edge-rows + L2 output 1478, with
rotated cells 147 → 155 → 0. The raw region is reflected — a registration error —
and a fold-free completion that keeps the reflection (a 180°-rotated patch,
orientation-preserving by double inversion) is closer in L2 to that erroneous
input than the un-rotated one. On real full slices a rotated patch must join
un-rotated tissue and becomes the trap, which is why the rows win fidelity there
(z=2 −28 %, z11 −23 %, raw z16 −1.5 %); on a globally reflected region they force
the un-rotation at an L2 cost against the input. This is a documented behaviour
change of the default formulation, not a defect.

**From-raw robustness of the final formulation** (edge rows δ=0.01 + in-solve L2,
bilinear rows, threshold 0.01, engine defaults, serial; every row 0 simplex / 0
bilinear folds, damage 0, and the terminal re-seed stage never fired):

| slice | folds before | rounds / windows | patience | wall | L2 move | re-seed-path engine |
|---|---|---|---|---|---|---|
| B0032 lap_ext z1 | 4556 | 2 / 125 | 0 | 2052 s | 1575 | 70 folds left after 8902 s |
| B0039 lap_ext z1 | 3957 | 2 / 57 | 2 | 2344 s | 2038 | 15,062 s (certification run, pool) |
| B0304 lap_ext z128 | 8956 | 2 / 248 | 3 | 3789 s | 1092 | 3643 s with the full rows (+0.7 %) |
| B0304 lap_all z181 | 29,699 | 3 / 308 | 43 | 5863 s | 1466 | 6319 s with the full rows, L2 +28 % |

Patience fallbacks (the bail-free exact-LS continuation) are now the only rung
that fires on these slices; the backend and grow rungs and the re-seed are idle.

### 4.3c Where the in-solve L2 objective spends its time (and the one real fix)

The 151-slice interim of the full-resolution certification under the final
formulation showed a split: the hard slices (>= 2000 folds, n = 17) went 78,623 s
-> 9,119 s, but the ordinary ones (n = 134) went 25,436 s -> 77,408 s (median 171
-> 559 s) at the SAME total SQP iteration count. A contention-matched four-way A/B
on z=440 (1828 folds) separated the two ingredients:

| formulation (z=440) | SQP it | window calls | ok | L2 move | ADMM it / QP (median) |
|---|---|---|---|---|---|
| old engine (`none`, no rows) | 433 | 110 | 101 | 85.8 | 637 |
| edge rows + `none` | **356** | 108 | 101 | 87.4 | **187** |
| L2, no rows | 1457 | 214 | 136 | 62.1 | 500 |
| edge rows + L2 (the default) | 915 | 187 | 126 | 67.8 | 550 |

The rows are free — they *condition* the QP (ADMM iterations 637 -> 187). The
objective is the cost, and not where one would guess: successful windows cost the
same as before (328 iterations over 126 calls vs 361 / 101); 59 % of all
iterations (544) sat in 48 window calls that ended `a-collapse` FAILED and were
then fed the escalation ladder (no-trust-region retries 32 vs 5). Their
per-iteration traces show them *converging* — max violation 3.3 -> 1e-5, merit
/ 700 — with 25–30 rows hovering 1e-5..1e-4 below the margin-shifted target: a
distance objective parks the solution ON the active rows at ADMM precision,
where a zero objective steps off the boundary to exactly 0. The engine's margin
(1e-3) exists precisely so that a solve landing a hair short of the active bound
is still fold-free, but the inner's own feasibility test was `-1e-6`.

Fix: the window counts as solved when its rows end within half the margin of
the shifted target (`solve_window_inner(feas_tol=0.5 * margin_delta)`, and the
same slack in the isqp inner's flag). z=440: 915 -> 790 SQP iterations, calls
187 -> 147, window success 67 % -> 91 % (old engine 92 %), L2 unchanged;
volume z=16 442 -> 385 / calls 61 -> 35; full-res z=2 unchanged (its collapses
are genuine); crop pack byte-identical under L2. The remaining L2 cost is
in-window polishing along the active rows (median relative merit decrease 3e-4
per iteration) — the `ftol` stop (relative objective decrease, for
feasible-within-slack iterates) addresses that.

Measured dead ends for the same cost, all on z=440 under L2 + rows (do not
retry): penalty parameter `rho` 1e4 (873 it, ADMM median 1237) and 1e5 (898,
ADMM at the 2000 cap, L2 73.7); initial trust region 1.0 px (845); the
a*-collapse bail off (1323) or at 6 (1177); a "collapse needs a standing
violation" predicate (905, inert on every other case).

### 4.3d The remaining cost is ADMM convergence — and what that leaves

With the ladder waste (#103) and the cap (#106) gone, an ordinary slice is ~300
SQP iterations at ~0.7–0.9 s each in the 4-worker pool, 70 % of it inside the
QP solves. Everything cheaper was tried on z=240 (five concurrent, walls
relative; all 0 folds, damage 0, identical L2 unless stated):

| lever | result |
|---|---|
| OSQP polish off | 220 vs 222 s — nothing |
| OSQP eps 1e-3 → 1e-2 | 200 s (−10 %) but +8 % SQP iterations — a wash; not plumbed |
| orientation rows only near folds | 269 s, 677 iterations — worse (the rows condition the QP) |
| **lagged Jacobian** (reuse the KKT factorisation 3 / 5 / 10 iterations, update only q/l/u) | 344 / 368 / 383 s, 457–517 iterations — worse; the *reused* solves still ran ADMM to the cap at ~1.0 s/QP |

The last row is the diagnosis: the per-QP cost is ADMM **convergence** on these
QPs (a 2·I objective Hessian over ~9k free variables against ~20k bilinear
and edge rows), not the factorisation — so factorisation reuse, tolerances and
polishing cannot buy it back, and the interior-point handoff already earns its
keep (OSQP-only 539 vs 481 s).

Window-level parallelism inside a slice was then built and **measured out**: a
round's windows with pairwise-disjoint footprints solved concurrently on the
shared spawn pool reproduced the serial result byte for byte, but bought only
−5 % (z=240: 98 vs 103 s), −6 % (z16: 66 vs 70 s), −5 % (z=2: 325 vs 343 s) —
real slices are dominated by the giant-region Schwarz tiler, whose sweeps are
sequential by construction, so the change was dropped rather than shipped. The
one unexplored lever is a Jacobi-style (additive) Schwarz sweep — all tiles of
a sweep solved concurrently from the same iterate — which trades convergence
rate for parallelism and has an unknown sweep-count cost.

Where that leaves the engine on an idle box (serial, merged defaults after
#106): the hardest full-resolution slice z=2 (3909 folds) in **343 s** (10,168 s
at the start of the campaign), z=240 (988 folds) in 103 s, volume z16 (2131
folds) in 70 s; the 4-worker pool gives ~2.4× volume throughput on a
memory-bandwidth-bound box.

### 4.3e Best-of-both attempted: the per-window anchored polish (what is and is not recoverable)

`polish='l2'` re-solves each window, immediately after it solves, against the
distance to its pre-solve patch from the warm feasible point (verify-and-revert,
so it can never cost feasibility or fidelity). Three-way measurement (wall s /
L2 move; identical engine, serial, idle box; all 0 folds, damage 0):

| case | in-solve L2 (default) | `none` | `none` + polish | gap recovered |
|---|---|---|---|---|
| z=240 (988 folds) | 99 / **29.9** | **48** / 36.0 | 72 / 33.5 | 41 % |
| z=440 (1828) | 255 / **67.8** | **97** / 87.4 | 113 / 85.9 | 8 % |
| z16 (2131) | 65 / **189.6** | **27** / 227.7 | 36 / 222.0 | 15 % |
| z=2 (3909, hardest) | **328** / **1977.8** | 411 / 2341.1 | 425 / 2025.9 | 87 % |
| z16_twist crop | 14 / 70.7 | 4 / 123.1 | 18 / **70.7** | **100 %** |
| z0_cluster crop | 19 / **690.5** | 2 / 787.5 | 7 / 787.5 (reverted) | 0 % |

The split is the finding: the anchor's fidelity has a **within-window share**
(recoverable post hoc — 100 % on the twist crop, 87 % on z=2's big windows,
41 % on z=240) and a **trajectory share** — the anchor steering windows into
different basins and negotiating with their frozen rings during the solve —
which no post-hoc polish can reproduce (z0_cluster reverts after 30 iterations;
z=440/z16, dominated by many small coupled windows, recover 8–15 %). So no
single formulation dominates: the in-solve L2 default keeps the best fidelity
everywhere and wins outright on trap-heavy slices; `objective='none'` is the
2–2.5× fast lane; `polish='l2'` is the opt-in middle point.

**The terminal `reanchor` stage, re-measured on the final engine** (it predates
#103/#106; wall s / L2 move, all 0 folds / damage 0):

| case | in-solve L2 | `none` | `none`+polish | `none`+reanchor | `none`+polish+reanchor |
|---|---|---|---|---|---|
| z=240 | 99 / 29.9 | 48 / 36.0 | 72 / 33.5 | 356 / **28.1** | 354 / 28.1 |
| z=440 | 255 / 67.8 | 97 / 87.4 | 113 / 85.9 | 765 / **54.5** | 737 / 54.9 |
| z16 | 65 / **189.6** | 27 / 227.7 | 36 / 222.0 | 326 / 218.5 | 440 / 213.1 |
| z=2 | **328 / 1977.8** | 411 / 2341 | 425 / **2025.9** | 731 / 2340.6 (all tiles revert) | 700 / 2025.9 |

Two complementary facts: the terminal reanchor's overlapping-tile sweeps dig
past even the in-solve anchor on ordinary slices (its tiles escape the one-shot
frozen-ring limit) — the max-fidelity mode at 3–4× the wall — yet recover
nothing on the trap-heavy slice, where every tile reverts and only the warm
at-solve-time anchor (in-solve L2, or the per-window polish at 87 %) reaches the
good basin. The polish+reanchor combination is exactly the union of both gains
at both costs. No mode dominates; the L2 default remains the balanced choice,
and every mode is an existing knob.

### 4.3f The QP inner itself: the cost is OSQP's, the coupling is ours

316 real window QPs were captured from a z=240 solve and replayed through
candidate solvers on identical matrices (57 solves with full data, 9 patterns):

| solver | total | per-solve behaviour |
|---|---|---|
| OSQP (engine settings) | 21.2 s | ADMM at the 1000-iteration cap on most patterns; worst viol 3.4e-3 |
| **QPALM** (ALM; rebuilt + warm-started each solve) | **8.4 s (2.5×)** | 23–36 outer iterations; faster on every pattern; worst viol 1.7e-3 |
| PIQP (proximal IP) | 19.3 s | machine-precision feasible, but one pattern at 0.87 s/solve |

So the ~1.8× per-iteration cost of the L2 default (§4.3c–d) is **not intrinsic
to the QPs** — QPALM solves the same problems 2.5× faster. But wiring it in
(`qp_backend='qpalm'`) and gating on slices splits sharply: z=240 **49 s vs
99 s (2.0×, the L2 default at the `none` objective's wall)**, z16 +15 %, and
full-res z=2 **9.6× worse** (3159 vs 328 s: 7 rounds, 153 a*-collapses, the
backend rung firing 432 OSQP retries) — at identical fidelity and 0 folds /
damage 0 throughout, because the ladder nets correctness. The refined
statement: **the engine's step dynamics co-evolved with OSQP's solution
style** — trap-heavy trajectories depend on the particular near-solutions ADMM
returns, and a different, even better-converged QP answer perturbs the iterates
enough to shatter them (the same mechanism that made Clarabel-always lose,
§4.3c). `'qpalm'` ships opt-in for QP-bound ordinary fields; the follow-up with
real upside is a qpalm-on-mild-windows / osqp-escalation policy.

### 4.4 What is "bloat" and what is not — the minimal engine

With the orientation rows in every window and *every* fallback off (no no-TR /
backend / patience rungs, no grow, no mop, no re-seed, no coarse warm start),
the windowed I-SLSQP alone reaches 0 folds on the hard cases — z11 (2913 s,
L2 1382) and full-resolution z=2 (4651 s with `objective='none'`, L2 2557;
4372 s with an in-solve **L2** objective, L2 2502). Two lessons: (i) once the
feasible set is single-basin the ladder is no longer needed for *robustness*,
and the in-solve distance objective no longer traps residual folds — feasibility
and fidelity fit in one formulation; (ii) the coarse warm start and the ladder
are *speed*, not bloat: the full engine with rows solves z=2 in 1577 s, 3×
faster than the stripped one. Windowing itself is locality, not bloat: a
9,660-free-pixel window costs 3.7 s per SQP iteration.

The bounded-shear (cone) rows cannot be applied locally: restricted to the fold
neighbourhoods (margin 3) the QP is **primal infeasible** on both z11 and z=2 —
a free pixel next to a healthy-but-sheared fixed neighbour is asked to fit a cone
anchored on it, and healthy tissue routinely shears more than κ. The "identity is
feasible" guarantee needs the whole slice free (the 3172 s solve). Monotone
(edge-only) rows do not have this failure.

## 5. Other engine changes in the campaign (all merged)

- #85 `auto_strategy` routes `bilinear` → `isqp_windowed`; crop script de-monkeypatched; `docs/recipe-2d-zero-folds.md`.
- #86 opt-in re-anchor stage (`reanchor='l1'|'l2'`): raw z16 L2 −22 % / L1 −48 % at 0 folds; `'l1'` beats `'l2'` even in L2.
- #87 exact-LS a*-collapse bail (default 3): z0_sliver 1684 → 212 SQP iterations, five real slices −18 %.
- #88 budget-cut "damage" was an accounting artefact (warm-start boxes not in `touched`); giant tiler now honours the deadline.
- #89 patience rung (bail-free continuation): z128 8 → 0.
- #91 (closed): re-solving the patience rung from the original start — refuted; the divergence was basin sensitivity via grow-on-failure's failed-iterate paste.

## 6. Why the edge slices took hours (trace analysis) and the ordering fix

Trace of full-resolution z=1 (15,657 s of inner-solver time): 665 of 694 inner
calls fail; the 373 calls on windows with > 3000 free pixels take 15,173 s
(97 %) at **3.7 s per SQP iteration** (0.04 s on ordinary windows); the terminal
mop alone takes **12,367 s (79 %)** running the whole escalation ladder (10
calls per box) on the 50-cell rotated-branch residual that the re-seed then
clears in 7 s. Exit reasons: `a-collapse` 343 calls / 7,528 s, `tr-collapse`
69 / 5,370 s. Two fixes were measured. Running the re-seed *before* the mop removes the cost
but is too blunt for sliver-type residual (`z0_sliver`, 18 cells within ~1e-4 of
the threshold: L2 137.8 vs 21.5 for the mop) — rejected, kept as an opt-in knob.
The fix adopted gives the mop's windows above `max_window_area` a **single
attempt** (no retries, no grow; `_InnerOpts.ladder`) and leaves the ladder on
the small mop windows the sliver residual needs, so those stay byte-identical;
the re-seed after the mop handles what the big windows leave. Combined with the
monotone untangle (§4.3) this is the engine under validation on the edge slices.

## 7. Cohort facts useful for the paper

- Many-to-one correspondences are routine (median ~1000 fixed pixels per slice share a moving pixel) and are *not* the discriminator; local incoherence (> 25 px off the 15-NN median displacement) is: the six well-behaved brains have it on 1–15 of 526 slices per variant, and every residual slice among them is one of those. B0304 has it on 311/330 slices (the noisy cohort member).
- The cohort's ANTs warps have zero in-plane 2-tri folds (528 × 7 slices); all folding comes from the Laplacian interpolation.
- Filtering correspondences before the 3D Laplacian solve (drop > 25 px local outliers, merge many-to-one groups) halves z11's folds (4540 → 2474) — an upstream lever, no longer needed for feasibility.

## 8. Measured dead ends (do not retry)

Per-window: switching the rest of a window to `'tr'` after a bail (worse), scoping exact-LS out of the no-TR rung (worse), bigger trust region (`tr_delta=8`, fails), SLSQP inner (5.9 h, 17 folds, damage 15), harmonic re-seed via `harmonic_extension_2d` (no-op: it only accepts fold-free patches), pairwise / cluster-identity / targeted "de-collapse" kicks (blunt or catastrophic), orientation rows as a late rung. Engine-wide (earlier): float32 OSQP, GPU ADMM, Newton-SQP, dual warm starts, row pruning, OSQP settings, multilevel coarse-to-fine beyond one level, maximal fold-free step cap.

## 9. Artefacts

- Reports (self-contained HTML + CSV + figures): `benchmarks/output/b0039_ext_full/` (full resolution) and `benchmarks/output/b0039_ext_ds2/` (2×). Figures copied for the paper under `docs/paper_figures/`.
- Cohort sweep: `benchmarks/output/cohort_sweep_2d/` (results.csv, summary.json), note `docs/superpowers/notes/cohort-sweep-2d-findings.md` (branch `cohort-sweep-2d`).
- Traces and experiments: `benchmarks/output/isqp_campaign/residual_*/` (per-window traces, dvfopt.log, post-mortems), `orient_*.txt`, `phase1_*.log`, `minimal_*.log`, `reseed_probe.txt`, `decollapse_*.txt`, `rung_trace_ab.txt`.
- Fast crop pack (`benchmarks/make_hard_crops.py`): `z16_twist`, `z0_cluster`, `z0_sliver`.
- 3D: `research-3d-all-tets` branch — 24 distinct tets; a strictly 6-tet-feasible field hid 557 inverted cells on the other diagonals.


## 10. The 2.5D pipeline at full volume (2026-09-01 … 09-04): resumable runs, the mop's real cost, and the rows A/B

Everything below is on the 528-slice B0039 Laplacian-exterior volume after the
v2 per-slice 2D correction (`dz ≡ 0`), 4 workers, one pipeline at a time.

### 10.1 Resumable runs (`checkpoint_dir`, `dvfopt correct --checkpoint DIR`)

A full-volume 2.5D run is ~17 h of sweep; the first attempt was lost twice (a
harness without the `__main__` guard re-ran the pipeline in every pool worker
and exhausted the commit limit; a later 20 GB measurement next to two running
pipelines OOM'd the rows mop). `dvfopt/checkpoint.py` (`RunCheckpoint`) mirrors
the output to a memmap after every sweep slice / per-slice solve / 3D stage and
records progress in an atomically rewritten `state.json` validated against the
input hash and knobs; interrupt → resume is byte-identical to a cold run (nine
tests). Both sweeps of the A/B were reloaded from their checkpoints in seconds,
several times.

### 10.2 The sweep

| | residual folds after sweep | min volume | wall |
|---|---|---|---|
| base | 66 | −0.281 | 16.5 h |
| rows (`orientation_delta=0.01`) | 50 | −0.187 | ~20 h (edge layers slower, partly contended) |

Interior layers take ~14 s each; the 13 edge layers (z ≤ 12, 1000–1600 folds)
take ~1 h each and are 60 % of the sweep. On those layers the rows leave
~30 % fewer incoming folds for the next layer (z=0: 762 vs 1219) and slightly
fewer residuals (z=12…8: 7/10/7/4/4 vs 5/17/11/10/7). A cProfile of the z=5
layer: 5635 s wall, **20.8 s of Python (0.4 %)**, 409 HiGHS solves at ~14 s —
the sweep is LP-bound; Python micro-optimisation is worthless there.

### 10.3 The mop was the wall — four measured fixes

1. **Predicate bug.** `mop_interior_3d` clustered on its LP target
   (`min_vol < thr3 − 1e-9`, thr3 = threshold + 1e-4) while the pipeline's gate
   and `feasible` count at the report predicate (`< threshold − 1e-5`). The
   sweep parks ~127k cubes *at* thr3 within LP tolerance (579k at `< thr3`
   exactly), so one pass ran serial box LPs over ~700 clusters instead of the
   ~2.1k cubes / 261 clusters actually below the report threshold — a > 24 h
   pass that never finished. Fixed: the mop clusters on the report predicate.
2. **Parallel boxes + giant-box tiling.** Boxes are repaired on the shared spawn
   pool; boxes wider than `max_box=90` are tiled (the sweep's idiom). Neither
   made the pass tractable (still ~7 h, one worker 5.4 h), because:
3. **The cost is near-floor LP grind, not box size.** Measured directly on the
   densest edge region: a 3.7k-free-voxel box needs **> 11 min for four SLP
   solves** — nearly every tet row is active there, so it is a ~50k-row LP at
   minutes per solve regardless of size, and `elastic_trust_solve`'s
   accept-micro-step (trust doubles back) / reject (trust halves) alternation
   never reaches the trust floor, burning all 40 solves. The 2D engine's
   a*-collapse bail, ported: stop when the exact violation has not dropped 1 %
   over the last 4 solves (`stall_iters=4`, opt-in in the engine, default on in
   the mop, sweep byte-identical). Pass 1: **3.5 h vs 7 h**, same quality
   (66 → 30 folds vs 66 → 33).
4. **Scheduling.** Even so, the four workers used 3.2 h of CPU over that 3.5 h
   pass, 2.3 h of it on one worker (~0.9 of 4 cores), and pass 2 ran 100 %
   in the parent for 5 h with the workers untouched: batches held only
   *consecutive* disjoint boxes (`find_objects` order is spatial, so neighbours
   touch), and singleton batches were solved in-process — which is where the
   heavy, overlapping, near-floor boxes always landed. Boxes are now scheduled
   by dependency level (one above every earlier overlapping box; still
   byte-identical to serial, re-verified bit-for-bit on real data). Dry run on
   the edge band (159 boxes): **26 rounds instead of 57, mean 6.1 boxes per
   round (max 15) instead of 2.8**; on the real volume the parent now sits at
   0 % and the pool carries the mop — but at **1.24 cores averaged over the
   first 8 min** (599 s of worker CPU over 483 s), not 4. The heavy near-floor
   boxes overlap each other, so they form dependency *chains* (the size-1
   levels above), and a chain runs one box at a time on any number of
   workers; the light, spatially spread boxes are what parallelise.

The remaining floor is therefore two things: the per-solve price of a
near-floor box, and the chains those boxes form. Levers not taken this round:
a Jacobi / restricted-additive variant of the mop (overlapping boxes solved
from one snapshot, disjoint cores pasted back — the 2D engine's RAS tiler
idea, measured −26 % on one slice there) would break the chains at the cost of
the exact serial semantics; a cheaper LP formulation, or skipping cubes the
best-of-4-diagonals certificate proves are at the geometric floor, would cut
the per-solve price.

### 10.4 The rows verdict

<!-- PENDING: fill from the AB25D base / rows summary lines (n_neg_out,
n_neg_best_diag_out, n_below, min_T, L1 move; wall is soft — resumed and
contended). -->

### 10.5 Artefacts

`benchmarks/output/isqp_campaign/`: `ab25d_ck_base/`, `ab25d_ck_rows/`
(sweep-final checkpoints), `ab25d_v2*.log` (sweeps), `ab25d_v3…v6.log`
(the mop chain: parallel → tiled → futility → levels), `profile_z5.log`,
`time_mop_tile.log`, and the scratch measurement scripts named in the CHANGELOG.
