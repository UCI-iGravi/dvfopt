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

The windowed engine (fold *repair*, not the 2.5D marching prevention above)
gained the same resumability plus a z-banded full-volume driver in the 3D
port's phase 4 (`checkpoint_dir=` on `windowed_correct` / the new
`windowed_correct_banded`) — see the CHANGELOG's phase-4 entry for the
full-resolution B0039 rows as they land.

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

Measured at the sweep stage — where the knob acts; the mop is identical
machinery on both legs and chain-bound on this volume (§10.3) — band-wise from
the sweep-final checkpoints (`sweep_verdict.py`, `sweep_verdict_floor.py`):

| | folds (fixed diagonal) | folds under every diagonal (true floor) | sub-threshold (< 0.01 − 1e-5) | sub-threshold under every diagonal | min volume | L1 move from input | sweep wall |
|---|---|---|---|---|---|---|---|
| base | 66 | 23 | 2137 | 1345 | −0.281 | 306,088 | 16.5 h |
| rows (`orientation_delta=0.01`) | 50 | 12 | 4384 | 2463 | −0.187 | 285,521 | ~20 h |

The rows cut the fixed-diagonal folds by 24 % and halve their depth, and
the true fold floor (negative under all four main diagonals) goes 23 → 12, so the rows remove real folds, not just re-split artifacts. They also move the field **6.7 % less** from the input (a
prevention row is cheaper than a repair), so on fidelity and folds they win
outright. The price is margin: twice as many cubes end up parked just under the
0.01 threshold (the rows enforce edge projections ≥ δ = 0.01 and the LP lands
cells at that bound), and those are exactly the mop's workload — the chain-bound
near-floor boxes of §10.3. Recommendation: keep `orientation_delta` opt-in for
the 2.5D chain as it stands; make it the default once the mop no longer pays
per sub-threshold cube (a Jacobi/RAS mop, or a floor-skip on the
best-diagonal certificate). Walls are soft (resumed, partly contended).

### 10.5 Artefacts

`benchmarks/output/isqp_campaign/`: `ab25d_ck_base/`, `ab25d_ck_rows/`
(sweep-final checkpoints), `ab25d_v2*.log` (sweeps), `ab25d_v3…v6.log`
(the mop chain: parallel → tiled → futility → levels), `profile_z5.log`,
`time_mop_tile.log`, and the scratch measurement scripts named in the CHANGELOG.

## 12. The canonical 2D benchmark (2026-09-14 … 09-15)

One pinned engine, one pre-registered protocol, every 2D source in the repo, nothing
dropped. The driver is `benchmarks/canonical_2d.py` (branch `feat/2d-canonical-benchmark`).
`dvfopt/` is byte-identical to main `4636af8` for the whole branch, so every number is the
shipped engine's own behavior. The tracked results live in `docs/paper/results/2d_canonical/`:
one directory per source (`results.csv`, `summary.json`, `manifest.json`, `table.md`, figures)
and a README with the protocol, the metric and gauge definitions and the per-source tables.
Protocol, plan and ledger: `docs/superpowers/notes/2026-09-11-2d-canonical-benchmark-handoff.md`,
`docs/superpowers/plans/2026-09-11-2d-canonical-benchmark.md`,
`.superpowers/sdd/2026-09-11-2d-canonical-benchmark/progress.md`.

### 12.1 What was run

Engine call: `correct_dvf(phi, threshold=0.01, record_history=True, **config)`, no per-case
knobs. Seven configs: `isqp_none` and `isqp_l2` (bilinear rows, the windowed engine, objective
`none` / `l2`), `auto` (bilinear), `slp` (simplex, L1), `barrier` and `m14` (simplex, L2), and
`slsqp_windowed` (central-difference Jdet, L2). The two `isqp_*` configs run on every source.
The other five run on the three small sources (origins, crops, synthetic).

`certified` means 0 bilinear values below `0.01 - 1e-5` after the solve. The bilinear column is
exactly `cell_min_jdet_2d / 2`, the triangle-area scale the solver constrains, so a paper must not
halve it again. A method's own-gauge `feasible` flag is reported beside it, never instead of it.

| source | pairs | config | certified | wall median, s | wall kind |
|---|---|---|---|---|---|
| crops (TUNING SET) | 3 | `isqp_none` / `isqp_l2` / `auto` | 3/3 each | 6.28 / 41.43 / 32.17 | throughput, load 76 % |
| crops (TUNING SET) | 3 | `barrier` / `m14` / `slp` | 0/3 each | 2.195 / 6.617 / 8.286 | throughput, load 76 % |
| crops (TUNING SET) | 3 | `slsqp_windowed` | 0/3 (3/3 own gauge) | 915.6 | throughput, load 76 % |
| synthetic | 13 | `isqp_none` / `isqp_l2` / `auto` | 13/13 each | 0.1276 / 0.3028 / 0.2246 | throughput, load 24 % |
| synthetic | 13 | `barrier` / `m14` / `slp` | 1/13, 1/13, 4/13 | 0.4013 / 0.5038 / 0.7087 | throughput, load 24 % |
| synthetic | 13 | `slsqp_windowed` | 0/13 (13/13 own gauge) | 0.06125 | throughput, load 24 % |
| origins | 27 | `isqp_none` | 26/27 | 4.348 | serial, load 2 % |
| origins | 27 | `isqp_l2` | 27/27 | 10.25 | serial, load 2 % |

This serial-timing pass predates the re-seed fix (PR #127, `a5a3a51`) and was never rerun, so its
`isqp_none` column still reads 26/27 — see §12.11 for the fix and pass 5's re-measurement to
27/27.
| ANTs controls | 85 | `isqp_none` | 85/85 (0 pixels moved) | 0.4254 | throughput, load 2 % |
| cohort (7 brains) | 85 | `isqp_none` | 85/85 | 86.45 | throughput, load 39 % |
| cohort (7 brains) | 85 | `isqp_l2` | 85/85 | 209.6 | throughput, load 39 % |

**Final origins taxonomy** (`origins_all_v5`, all 8 configs, pooled/throughput across the recovery,
amendment and re-seed-fix re-measurement passes; `isqp_none`'s 26/27 below is pass 4's number,
superseded — see §12.11):

| source | pairs | config | certified | wall median, s | wall kind |
|---|---|---|---|---|---|
| origins | 27 | `isqp_l2` | 27/27 | 32.13 | throughput, pooled across passes |
| origins | 27 | `auto` | 27/27 | 21.01 | throughput, pooled across passes |
| origins | 27 | `isqp_none` | 27/27 (pass 4: 26/27) | 7.765 | throughput, pooled across passes |
| origins | 27 | `isqp_l1` | 24/27 | 45.66 | throughput, pooled across passes |
| origins | 27 | `slp` | 3/27 | 20.13 | throughput, pooled across passes |
| origins | 27 | `m14` | 4/27 | 17.19 | throughput, pooled across passes |
| origins | 27 | `barrier` | 3/27 | 26.23 | throughput, pooled across passes |
| origins | 27 | `slsqp_windowed` | 3/27 (19/27 feasible, own gauge) | 0.3216 | throughput, but this config's pairs each ran isolated, one at a time |

Damage is 0 on every windowed row. The box was shared with unrelated jobs for most of the
session, so only the origins serial pass supplies a per-case wall column. On identical pairs
the throughput walls ran about 2.6x slower: `m2_demons_brainpair_weak` × `isqp_none` 147 s
serial vs 394 s throughput, `m2_ffd_brainpair_coarse` × `isqp_none` 951 s vs 2,498 s.

The one uncertified origins row under pass 4 was `m2_ffd_brainpair_coarse` × `isqp_none`. Its
input is 39.5 % folded. Pure feasibility stopped 6 bilinear cells short of the gauge (worst
−5.09e-4 in the serial pass, −5.22e-4 pooled) after moving about 87 % of the pixels by up to 89 px.
`isqp_l2` certified the same field. The residual cluster sat in a patch the input had folded
completely, on cells that were x-slivers: the vertical edge projection sat exactly at
`orientation_delta` = 0.01 while the horizontal one had collapsed, so the linear edge rows bound in
y and the cells could not open in x.

Root cause (PR #127, `a5a3a51`, merged after this pass): the giant-tile Schwarz sweep had left a
180°-rotated column strip at col 193, rows ~8-29. A rotated cell has both axial edge projections
negative but a POSITIVE determinant, so the strip was area-feasible and invisible to the fold mask
the terminal harmonic re-seed stage keyed on; the re-seed ring it dilated to was itself rotated, so
the fill inherited the rotation and the stage stopped on no progress, leaving the sliver folds at
the strip's seam uncleared. The fix folds orientation-row violations into the re-seed mask, not
only fold cells. Pass 5 (`origins_all_v5`, §12.11) re-measured this one pair alone under the fix:
**certified**, 0 bilinear cells below gauge (was 6), worst value +0.01099 (was −0.00052), damage 0,
5 rounds / 147 windows / 2,572 SQP iterations, 982.9 s wall (that pair ran alone; the earlier
2,497.8 s pooled figure was throughput contention, not a slower fix). `isqp_none` is now 27/27 on
origins.

**Crops.** The on-disk crops reproduce the documented fold counts (simplex 645 / 598 / 0 → 0)
but not the historical move and wall figures in CLAUDE.md. Four historical engine commits
reproduce today's `z0_sliver` result bit for bit, so there is no engine regression. The
historical figures came from an earlier cut of the crop file. Measured here: `z0_sliver` L2 move
805.8 (`isqp_none`) and 910.6 (`isqp_l2`). Under `isqp_none` the sliver moves 97 % of its pixels,
RMS about 11 px per component, to clear 18 near-threshold bilinear folds. The crops are the
engine's tuning set and are never quoted as a held-out result.

### 12.2 The certificate costs the landmarks on pin-collapsed cohort slices

All 170 cohort rows certify (85 slices × 2 configs, 0 error rows, damage 0). This includes the
collapsed-pin slices the earlier cohort sweep had called a data obstruction: B0039 z1, B0039 z11
and B0032 z1. The price is landmark fidelity. The median registration residual at the prescribed
Laplacian correspondences changes as follows (px):

| slice | before | after `isqp_l2` | after `isqp_none` |
|---|---|---|---|
| B0039 z1 | 1.56 | 132.7 | 135.1 |
| B0039 z2 | 1.51 | 124.0 | 127.2 |
| B0032 z1 | 0.96 | 77.5 | 78.9 |
| B0304 z128 | 0.0308 | 20.44 | 19.19 |
| B0039 z11 | 0.153 | 8.01 | 7.861 |
| B0304 z181 | 0.486 | 5.075 | 5.278 |
| B0039 z16 | 0.0356 | 1.811 | 2.348 |
| B0039 z264 | 0.00864 | 0.04168 | 0.06633 |

The B0039 z1 × `isqp_l2` figure was recomputed from the saved field. All 20 landmarks lie on
moved pixels. The median displacement there is 134.2 px, while only 8.6 % of the slice moved. On
these slices the certificate is not a fidelity claim, and the residual must be reported beside it.

### 12.3 Reproducibility and verdict stability

Two pooled runs agree bit for bit. Their workers are pinned to one BLAS thread. The two
throughput passes of `m2_ffd_brainpair_coarse` × `isqp_none`, before and after a machine
restart, match in all 16 outcome fields. A pooled run and an in-process run of the same pair
agree only discretely. Certificates, fold counts under every gauge, damage, windows, rounds and
SQP iterations match. On that pair the L1 and L2 moves differ by about 3e-5 relative, and the
worst bilinear residual by 2.4 % (−5.22e-4 pooled vs −5.09e-4 in-process). The most plausible
cause is the BLAS thread count. That is supported, not proven.

Over all 54 origins pairs (27 fields × `isqp_none` / `isqp_l2`), serial vs throughput: **0
certified-verdict flips and 0 pairs with any discrete-field difference.** Two pairs are
gauge-marginal, with the worst bilinear value within 2.5 % of 0.01. Both are `isqp_l2` and both
certify in both passes: `m1_laplacian_cohort_B0039_z264` at 0.010002 and
`m2_ffd_brainpair_coarse` at 0.010216. No verdict depends on BLAS threading. Results are never
described as bit-identical between pooled and in-process runs.

### 12.4 The scale limit of `slsqp_windowed`

`m2_ffd_brainpair_coarse` × `slsqp_windowed` failed in 0.015 s with
`MemoryError: Unable to allocate 429. GiB for an array with shape (57520089628,)`. The traced
SLSQP driver (`dvfopt/core/primitives/slsqp.py`) allocates scipy's worst-case dense workspace:

`buffer_size = n(n+1)/2 + 3mn − (m+5n+7)·meq + 9m + 8n² + 35n + meq² + 28`

(plus `2n(n+1)` when there are no inequality rows). A windowed Jacobian solve on k free pixels
has n = 2k, m = k and meq = 0, so the buffer is about 40·k² float64 values. With k = 37,920 the
formula gives 57,520,089,628, equal to the error's array length to the element. The failing
window therefore held exactly 37,920 free pixels, 89.8 % of the 160×264 slice, with one Jacobian
row per free pixel. The workspace is quadratic in window area, so `slsqp_windowed` does not apply
to dense folding at slice scale. The row is a measured failure, kept in every denominator.

**The full origins outcome legend for `slsqp_windowed`** (27 pairs, each run isolated one at a
time under a 21,600 s / 6 h no-progress watchdog, ruling R17): 3/27 bilinear-certified
(`m1_laplacian_cohort_B0039_z264`, `m2_demons_brainpair_smooth`, `m4_ants_cohort_B0039_z264`), 17
finished but were not bilinear-certified (feasible only under the config's own central-difference
Jdet gauge, or not even that), 1 `MemoryError` (`m2_ffd_brainpair_coarse`, 429 GiB, 16,319
central-diff folds on 160×264 — the workspace-size failure above), 1 `WorkerCrash`
(`m2_ffd_brainpair_fine`, 9,289 folds on 160×264 — see below), and 5 `WatchdogTimeout` at the 6 h
cap: `m2_demons_brainpair_weak` (11,827 folds, 160×264), `m2_tvl1_brainpair_a60` (7,019 folds,
160×264), `m3_proxy_synthetic_strong` (4,110 folds, 192×192), `m2_ilk_synthetic_r3` (1,425 folds,
192×192) and `m3_voxelmorph_ellipses_direct` (800 folds, 64×64). Every finished-in-seconds pair sat
at a few hundred folds or fewer; every pair that did not finish in 6 h sat at ~1,400 folds or
above, a clean split apart from the crash.

**The `WorkerCrash` (ruling R18).** `m2_ffd_brainpair_fine` × `slsqp_windowed` crashed with
`WorkerCrash: worker process terminated abruptly` seconds after its worker spawned, in both its
pass-3 attempt and its single pass-4 isolated rerun — the same text, no output field either time.
Under R18 this is now recorded as a MEASURED failure of the baseline config on this field, not an
infrastructure loss, and there is no further rerun. At the time of the first crash the evidence
ruled out both the 429 GiB `MemoryError` path (this crash raises nothing; the process simply dies)
and a box-wide memory kill (40.8 GB RAM free, no Windows application-error report and no
resource-exhaustion event in the crash window). The **working hypothesis** — unproven — is that
the vendored traced SLSQP core's dense workspace (the same `_slsqplib` allocation derived above)
takes an element count that exceeds an addressable index range at this window's size, so the
allocation succeeds and the C core then indexes out of range and kills the process instead of
raising a catchable error. This is a hypothesis, stated as such, and is NOT independently
verified.

### 12.5 Recovery provenance

The origins taxonomy run (27 fields × 7 configs = 189 pairs) did not finish in one pass.

1. A machine restart cut the first attempt at 48 rows. That partial run is kept apart and not
   merged. The rerun started fresh.
2. The rerun's pool broke after 58 measured rows and the MemoryError row. All four workers died
   at once and 130 pairs became `BrokenProcessPool` rows. Those are infrastructure losses, not
   measurements. The driver gained `--resume` (reuse measured rows and their sha256-verified
   fields, rerun only losses), pool-break recovery (rerun the break's suspects alone) and
   `--isolate-config` (run a config one pair at a time in its own single-worker pool).
3. The first recovery measured 110 of the 130 and then hung (defect D1, §12.6). A second pass
   started on the last 20 pairs, all `slsqp_windowed`.
4. Under ruling R17, pass 3 reruns those pairs from the code-final driver `e1d3a88`, whose
   isolated path gives each pair a 6 h cap.

The table's rows come from three driver commits. 59 rows were measured at `cdbf5f4` and 110 at
`90fccab`. The two commits differ only in the runner (`--resume`, pool-break recovery,
`--isolate-config`). `run_case` and every metric it calls are byte-identical between them. Pass 3
runs at `e1d3a88`. It is measurement-equivalent (R17(4)): `git diff 90fccab e1d3a88 --
benchmarks/canonical_2d.py` touches the measurement path in one hunk (`run_case` sets `certified =
feasible = False` when the solve raises, which affects only error rows — the reused `MemoryError`
row already reads `False`/`False`); everything else in the diff is runner plumbing (spawn context,
pool close, the watchdog), resume guards, aggregation and provenance keys. `dvfopt/` is untouched
on the branch throughout, so pass-3 rows are measurement-equivalent to the 169 reused rows.

Pass 3 ran `origins_all_v3`: `--resume origins_all_final --isolate-config slsqp_windowed`, giving
each of the 20 remaining `slsqp_windowed` pairs its own 6 h watchdog cap, one at a time (uncontended
— no other pair or pool shared the box while a `slsqp_windowed` pair ran). It ended cleanly at 189
rows: 169 reused, 20 rerun, 0 pool breaks, 1 `WorkerCrash`. The full 27-pair `slsqp_windowed`
outcome legend (3 certified, 17 finished-but-uncertified, 1 `MemoryError` carried over from the
169 reused rows, 5 `WatchdogTimeout` and 1 `WorkerCrash` newly measured by this pass) is in §12.4.

Pass 4 (ruling R19) ran `origins_all_v4` from a fresh snapshot at `c0f8af0` (the commit that adds
the `isqp_l1` config): `--resume origins_all_v3 --config <the 7> isqp_l1 --isolate-config
slsqp_windowed --n-workers 4`, reusing the 188 rows `origins_all_v3` had measured and running only
the 27 new `isqp_l1` pairs plus the single R18 rerun of the crashed `WorkerCrash` pair — 28 reruns
in total, 216/216 rows, 0 pool breaks, git commit `c0f8af0`, `dvfopt/` still identical to main. The
27 new `isqp_l1` pairs ran pooled at `n_workers=4` (a throughput pass), unlike pass 3's isolated
`slsqp_windowed` pairs. `origins_all_v4` measured `isqp_none` at 26/27 (§12.4-style miss on
`m2_ffd_brainpair_coarse`; the fix and its re-measurement are in §12.11).

Pass 5 (2026-09-21, driver `2edd8ff`, after the re-seed fix PR #127 / `a5a3a51` merged to main) ran
`origins_all_v5`: `--resume origins_all_v4_minus1` (a copy of `origins_all_v4` with the
`m2_ffd_brainpair_coarse` × `isqp_none` row removed), reusing the other 214 rows and re-measuring
only that one — now certified. `--resume`'s rule reruns every `WorkerCrash` row unconditionally, so
the isolated `m2_ffd_brainpair_fine` × `slsqp_windowed` pair also ran a third time and crashed a
third time with the identical error text; ruling R18's "single rerun, measured failure, not an
infrastructure loss" verdict is unchanged by this third occurrence. The CSV's 27 `isqp_l1` rows
also change position between v4 and v5 (the driver writes rows in config-table order) — a
reordering, not a content change; a text diff of the two CSVs otherwise touches only the
re-measured row. `origins_all_v5` is the FINAL origins run directory; its tracked copy is
`docs/paper/results/2d_canonical/origins/`.

### 12.6 Driver defect D1 and the no-progress watchdog

**D1, the nested-pool shutdown hang.** The first recovery finished its parallel pass and never
started its isolated phase. Every process in its tree sat at 0 CPU. Each pool worker held a
nested process pool, spawned inside the worker by the `slp` or `m14` strategy, and the driver
blocked shutting the outer pool down. Killing the grandchildren let all four workers exit on
their own, which supports the diagnosis. The fix (`37eecde`) closes every pool without waiting
on worker exit. It shuts the pool down without waiting, then terminates lingering workers and
their descendants after a short grace. A test-only hook reproduces the nested pool: before the
fix `run()` never returned (killed at 240 s), after it the run ends in about 10 s.

**The watchdog (R12).** `37eecde` also added a no-progress watchdog, and `5904a0b` fixed what it
records. When a pool completes no pair for `CANONICAL_2D_NO_PROGRESS_S` seconds (default 6 h),
the driver records the first `n_workers` running pairs in submission order. Each becomes a
`WatchdogTimeout: no result within <T> s (parallel pass | isolated run)` row with `hit_cap=True`.
Every other unfinished pair, including the call the executor pre-queues, is resubmitted to a
fresh pool. A recorded pair stalled for the full timeout and would stall again, so it is a
measured "did not finish within T" outcome, not an infrastructure loss. `--resume` keeps it and
refuses an old run made under a different `watchdog_timeout_s`. Summaries from `e1d3a88` record
`watchdog_timeout_s` in their provenance. The tracked runs were made by `cdbf5f4` and `90fccab`,
which predate the watchdog, so each of them ran to the engine's own termination.

A related chain-script lesson: a dead worker's orphaned grandchild inherits the stdout pipe, so
a `| grep` stage never sees EOF and the chain blocks until the orphan is killed.

### 12.7 Rotated-branch census

The certificate is a per-cell gauge, and a cell rotated by 180° still has a positive
determinant. So the certified outputs were checked for cells with both edge projections
negative. **Across all 399 certified outputs measured, the engine never introduced a cell
rotated past 90°: every such cell in a certified output was already present, and untouched, in
its input.** The 399 cover `origins_serial`, `ants_isqp`, `cohort_isqp` and the first origins
recovery, all configs. The one output that carries such cells is `m4_svf_synthetic_subpixel`
(75 cells under `isqp_none`, `isqp_l2` and `auto`, identically in both passes). All 75 are input
cells the engine did not move. The engine removed the input's other 50 rotated cells, which lay
inside fold windows. The 75 are orientation-preserving (forward-difference Jdet 0.255 to 8.707),
with a median x-edge rotation of 112°. They are large local rotations of the synthetic SVF, not
folds, and the certificate correctly passes them.

Proxy caveat: "both edge projections < 0" flags any cell rotated past 90°. A smooth large
rotation is a valid injective deformation, so the proxy is not a fold indicator. The claim is
about cells the engine introduced, never global injectivity.

**The final origins census.** Over the pass-4 origins directory (`origins_all_v4`, every certified
row across all 8 configs): **117 certified origins outputs, 0 introduced rotated cells** past 90°.
This is the origins-final number, separate from the 399-output figure above, which covers the
other tracked sources (`origins_serial`, `ants_isqp`, `cohort_isqp`) plus the pre-final origins
rows measured before the taxonomy closed — the two figures are not summed, since the 399 already
counts an earlier, superseded slice of the origins rows. The same proxy caveat applies to both.
This census was not rerun for pass 5 (`origins_all_v5`, §12.11): the one newly-certified row,
`m2_ffd_brainpair_coarse` × `isqp_none`, adds a 118th certified origins output to the count above,
but a full re-census of that row's cells for introduced rotations past 90° was not performed.

### 12.8 Engine follow-up: `sqp_iters` misses the polish iterations

The driver's `sqp_iters` sums `n_iter` over the engine's history, skipping nested `giant*`
entries. `windowed_correct` records its re-seed and re-anchor stages with `n_iter = 0`, although
their polish solves iterate (`dvfopt/core/windowed/_common.py`). So `sqp_iters` counts the round,
coarse and mop iterations only. It must never be quoted as a total on a row where the re-seed or
re-anchor stage ran. The fix belongs in the engine (record the real inner iteration counts). It
is outside this benchmark's branch, which leaves `dvfopt/` untouched.

### 12.9 Artefacts

Tracked: `docs/paper/results/2d_canonical/{crops, synthetic, origins, origins_serial, ants,
cohort}/` — `origins/` (the FINAL 8-config taxonomy, from `origins_all_v5`) joins the previously
tracked sources; `crops/` and `synthetic/` now carry the `isqp_l1` row alongside their original
seven. Gitignored run directories: `benchmarks/output/2d_canonical/<row>/`, including the pass
history `origins_all` (invalid, pre-`--resume`) / `origins_all_recovered` / `origins_all_v3` /
`origins_all_v4` / `origins_all_v4_minus1` / `origins_all_v5` and `crops_all_v2` /
`synthetic_all_v2`. Corrected fields: `data/dvfs/results/<run>/<source>/<case>__<config>.npz`,
each listed with its sha256 in the source's `manifest.json`. `origins_all/` and
`origins_all_partial_restart/` hold aggregates over sentinel rows and must not be used;
`origins_all_v3`'s and `origins_all_v4`'s own aggregates are likewise superseded, by
`origins_all_v4`'s and `origins_all_v5`'s respectively.

### 12.10 The `isqp_l1` amendment (ruling R19)

The user asked, after the pre-registered protocol had closed, for the windowed engine's own L1
anchor as a benchmark row: `isqp_l1 = dict(constraint="bilinear", strategy="isqp_windowed",
objective="l1")`. This is documented engine behaviour, not a new code path — the exact-line-search
step rule fits the eps-smoothed L1 approximation and falls back to the `'tr'` ratio-test
acceptance when the true merit did not decrease. Scope was the three small sources only (origins,
crops, synthetic — 43 pairs); cohort and ANTs kept their two pre-registered engine rows.

**Numbers** (`origins_all_v4` / `crops_all_v2` / `synthetic_all_v2`, verified against
`results.csv` and `summary.json`): origins 24/27 certified (median wall 45.7 s, L2 move 22.35, L1
move 883.5, moved fraction 0.108), against `isqp_none` (26/27 as measured here in `origins_all_v4`,
now 27/27 after the pass-5 re-seed-fix re-measurement — §12.11 — which left the median wall/move
figures below unchanged; 7.8 s, 18.12, 1075.1, 0.191) and
`isqp_l2` (27/27, 32.1 s, 17.87, 1033.8, 0.181); `auto` medians 21.0 s. Crops 3/3 (L2 move 681.5,
the smallest of the four engine rows against `isqp_none` 787.5 and `isqp_l2` 690.5; L1 move
16,981, also the smallest; wall 19.6 s). Synthetic 13/13 (moved fraction 0.56 vs 0.81 for
`isqp_l2` and 1.0 for `isqp_none`).

**The three origins misses** are the three heaviest brain-pair fields: `m2_demons_brainpair_weak`
(10 bilinear cells below the gauge after 3,352 s), `m2_ffd_brainpair_fine` (3 below after 3,837 s)
and `m2_ffd_brainpair_coarse` (57 below, worst −0.0068, after 16,483 s — the only capped engine
row, `hit_cap=True`). On this evidence the eps-smoothed L1 anchor is less robust than `none` /
`l2` on dense fold fields.

Over the 24 origins cases where all three engine objectives (`none`, `l1`, `l2`) certify: total L2
move 2,968 / 2,629 / 2,420; total L1 move 239,194 / 157,690 / 151,646; total wall 1,655 s / 3,564 s
/ 5,311 s. Pairwise against `isqp_l2`, `isqp_l1` wins the L1 move on 16/24 cases but never the L2
move (0/24), and loses the L1 total only because the heaviest cases dominate the sum. The real
signature of the L1 anchor is **sparsity** — a smaller moved fraction (0.108 vs 0.181) — not a
smaller total move, and it is not a faster route than `none` or `l2` either. `m2_ffd_brainpair_coarse`
is one of `isqp_l1`'s three misses and stays uncertified under that objective — the pass-5 fix
(§12.11) is specific to `isqp_none`'s re-seed path and does not change this set of 24.

### 12.11 Pass 5: `m2_ffd_brainpair_coarse` × `isqp_none` re-measured under the re-seed fix — origins `isqp_none` 27/27

2026-09-21. After the terminal-re-seed fix (PR #127, `a5a3a51` — §4.1 root cause: a 180°-rotated
column strip left by the giant-tile Schwarz sweep is area-feasible and so invisible to the fold
mask the re-seed stage keyed on; the fix ORs orientation-row violations into that mask) merged to
`main` at `2edd8ff`, the one pair it addresses was re-measured: `m2_ffd_brainpair_coarse` ×
`isqp_none`, by resuming the final origins run `origins_all_v4` with that single row removed
(`origins_all_v4_minus1`) and rerunning only it.

**Result: certified False -> True.** Bilinear cells below the gauge 6 -> 0, worst value −0.00052
-> +0.01099. Damage 0. 5 rounds, 147 windows, 2,572 SQP iterations. Wall 982.9 s (that pair ran
alone; the earlier 2,497.8 s figure was throughput contention on a shared box, not a slower fix).
L2 move 4,530.7 -> 4,751.4 — a larger move, since the field now has to clear the last few cells the
plateau had left short instead of stopping there.

The resulting run, `origins_all_v5`, has 216 rows: the other 215 are identical in content to
`origins_all_v4` (a CSV text diff also shows the 27 `isqp_l1` rows changing position, because the
driver writes rows in config-table order — a reordering from `--resume`, not a content change).
Origins `isqp_none` is now **27/27** certified (was 26/27); `isqp_l2` 27/27, `auto` 27/27 and
`isqp_l1` 24/27 are unchanged, as are `slp` 3/27, `m14` 4/27, `barrier` 3/27 and `slsqp_windowed`
3/27.

**Disclosed side effect.** `--resume` reruns every `WorkerCrash` row unconditionally, so the
isolated `m2_ffd_brainpair_fine` × `slsqp_windowed` pair ran a third time and crashed a third time
with the identical error text. Ruling R18 said this pair's crash is a single, measured failure of
the config on this field, not an infrastructure loss, with no further rerun — the outcome of this
unplanned third occurrence is unchanged from that ruling.

The full-taxonomy table in §12.1, the origins table in `docs/paper/results/2d_canonical/README.md`
and `origins/table.md` were updated to this run's numbers; see also the root-cause paragraph in
§12.1 above.

## 13. The 3D windowed engine at band scale: what was measured and what was refuted

Two spikes (2026-09-18, 2026-09-19) and one full-volume route attempt (2026-09-19 to 09-21)
asked whether the 3D windowed engine (`dvfopt.core.windowed`, `SimplexConstraint3D`) that
certifies crops and sub-volumes can be pushed to band or full-volume scale. Snapshots:
`../dvfopt-3d-spike` @ `6412a51`, `../dvfopt-3d-spike2` @ `a5a3a51`, `../dvfopt-3d-route` @
`2edd8ff`. No tracked file was changed by any spike; every artefact lives under
`benchmarks/output/spike_3d/`, `benchmarks/output/spike_3d_2/` and `benchmarks/output/route_3d/`
(all gitignored). Protocols pre-registered pass/fail rules before each run
(`.superpowers/sdd/2026-09-18-3d-spike/progress.md`,
`.superpowers/sdd/2026-09-18-3d-spike-2/progress.md`,
`.superpowers/sdd/2026-09-19-3d-defaults/progress.md`,
`.superpowers/sdd/2026-09-19-3d-route/progress.md`). A standing user rule, R1, held throughout:
no new engine logic — every spike arm turns an existing knob, runs an existing stage, or
composes existing pipelines.

### 13.1 Scope: what the engine certifies today

Every finished 3D `windowed_correct` / `ISQPWindowedStrategy` run reaches 0 fixed-diagonal
folds, 0 best-diagonal floor and damage 0 (`benchmarks/output/windowed_3d/h2h.md`, `crops.md`,
`gate.md`):

| artefact | size | folds in | folds out | floor out | wall s | L2 move | damage |
|---|---|---|---|---|---|---|---|
| slice090 (testcases_3d) | 5x10x10 | 61 | 0 | 0 | 7.8 | 7.28 | 0 |
| slice200 (testcases_3d) | 5x10x10 | 30 | 0 | 0 | 3.8 | 6.97 | 0 |
| slice350 (testcases_3d) | 5x10x10 | 70 | 0 | 0 | 7.7 | 11.50 | 0 |
| subvol16 | 16^3 cut | 721 | 0 | 0 | 660.7 | 55.81 | 0 |
| sub20 | 20^3 cut | 686 | 0 | 0 | 611.1 | 19.12 | 0 |
| twist (24^3 crop) | 24^3 | 403 | 0 | 0 | 719 | 27.03 | 0 |
| sliver (24^3 crop) | 24^3 | 1094 | 0 | 0 | 703.5 | 23.28 | 0 |
| moderate (24^3 crop) | 24^3 | 1217 | 0 | 0 | 1020 | 58.73 | 0 |
| cluster (24^3 crop) | 24^3 | 3038 | 0 | 0 | 6449 | 90.44 | 0 |
| B0032_moderate (cohort 24^3) | 24^3 | 1217 | 0 | 0 | 1042 | 64.22 | 0 |
| B0049_moderate (cohort 24^3) | 24^3 | 1217 | 0 | 0 | 934.7 | 43.97 | 0 |
| B0053_moderate (cohort 24^3) | 24^3 | 1217 | 0 | 0 | 931 | 50.34 | 0 |
| B0200_moderate (cohort 24^3) | 24^3 | 1216 | 0 | 0 | 1901 | 55.85 | 0 |
| B0213_moderate (cohort 24^3) | 24^3 | 1217 | 0 | 0 | 1791 | 40.38 | 0 |
| B0304_moderate (cohort 24^3) | 24^3 | 1217 | 0 | 0 | 704 | 60.85 | 0 |

That is 15 of 15 finished 3D windowed-engine runs at 0 folds, 0 best-diagonal floor and damage 0
(gate.md's `l2_rows` / `isqp_windowed` rows, taking the single-window row where subvol16 was run
both tiled and whole). None of the 15 exceeds 24 voxels on a side.

**L2-move caveat (PR #128).** Clarabel's internal thread pool changes the SQP trajectory on 3D
windows large enough to engage it (never on 2D windows). Pinning it to one thread (below) moved
the certified L2 on the three 24^3 B0039 crops: twist 27.0306 -> 27.0550, sliver 23.2803 ->
23.3012, moderate 58.7325 -> 58.7958 (+0.09 % / +0.09 % / +0.11 %), at 0 folds and damage 0 on
both sides of the pin. The pinned numbers are the current baseline and are what this section and
the table above report; any comparison against pre-PR-#128 figures must use the un-pinned column.

### 13.2 Where it stops

Nothing above 24^3 has certified. Three escalating attempts stopped short:

- **Phase 4's pre-registered full-volume rows** (`.superpowers/sdd/2026-09-11-3d-windowed-engine-port-phase4/progress.md`):
  two banded chains were stopped with no band finished — `slab_banded2` (2 workers) at 38 h wall
  / 37 CPU-h per worker, and `ds2_b0039` (4 workers) at 33 h wall / 33 CPU-h per worker. A 30-min
  instrumented probe on the ds2 band-0 slab (`(3, 40, 160, 228)`, 1.46 M voxels) measured 4 tile
  windows solved in 1,881 s, median 493 s / max 854 s per window, 29-61 SQP iterations, roughly
  17 s per SQP iteration on a contended box (11 s at 17^3 idle in phase 1); folds only 12,832 ->
  12,759 in that window.
- **Spike 1's sub-slab** (48x64x64 cut of `slab_240_288.npy`, 17,722 of 186,543 cubes below 0.01,
  9.50 % density, best-diagonal floor 13,973 / 7,615, min tet volume -8.4258): three concurrent
  3 h arms at `giant_tile` in {9, 12, 16}, all budget-cut. Folds cleared: 7,657 / 7,709 / 5,492
  out of 17,722 (43 % / 43 % / 31 %); SQP iterations 27,371 / 7,934 / 1,598; L2 move 148.0 / 127.2
  / 82.8; damage 0 in every arm; the worst cell (-8.4258) never moved in any arm.
- **The full-volume route** (2.5D marching output -> `windowed_correct_banded`, band 24 /
  overlap 8 / 4 workers, `benchmarks/windowed_3d_full.py --run`, unchanged driver): input 517
  folds below 0.01, 33 below 0, best-diagonal floor 352 / 14, min -0.035020533. Fold distribution
  over the 22 z-bands put 489 of the 517 folds and all 33 below-zero cells in band 0 (z in
  [0, 24)). Band 0 did not finish. The controller cut the run at **35 h 48 m** (the watchdog was
  meant to fire at 12 h but the run agent that would have enforced it had died; three of four
  workers sat idle after about 6 h, having finished their own bands, while the fourth stayed on
  band 0 for the full 35 h 48 m). Because the driver retrieves band results with `ex.map` in
  submission order and band 0 was submitted first, none of the 21 finished bands was ever
  committed, checkpointed or logged; all were lost with the kill (`ck/state.json` shows
  `done: []`).

### 13.3 Levers refuted

| lever | pre-registered rule | measured | verdict |
|---|---|---|---|
| QP backend (17^3 window, 8 SQP iterations) | >=5x faster QP at equal-or-better SQP progress -> candidate; >=10x -> worth engineering | median QP solve s: OSQP 8.10 (fastest), hybrid 9.37 (1.16x), qpalm 14.95 (1.8x), PIQP 29.00 (3.6x); OSQP hits its 1000-iteration cap on nearly every solve and still ends at the worst constraint minimum (-13.18); PIQP converges properly (27 IP iterations, `PIQP_SOLVED` every time) and ends best (-11.06) at 3.6x the wall | REFUTED (no backend is faster; PIQP's quality note is kept as a candidate if window quality, not wall, ever becomes binding) |
| Factorization reuse | setup+update >=50% of QP wall under OSQP -> the lever | 17^3: 10.9% (0.864 s setup + 5.76 s update against 54.3 s solve); 9^3: 4.7% | REFUTED |
| Tile size (sub-slab, 3 h budget) | tile 9 certifies where 16 does not, or in <=1/2 the wall at <=+25% L2 move -> 9^3 is the lever | nothing certified at any tile size; tiles 9/12 clear ~40% more folds than 16 at equal wall (7,657/7,709 vs 5,492) but at 1.5-1.8x the L2 move; tile 12 reaches that in 3.4x fewer SQP iterations than tile 9 | REFUTED (a <1.5x rate gain against a ~100x requirement) |
| `qp_max_iter=2000` at band scale (sub-slab, `giant_tile=16`) | cleared/hour >=1.5x baseline -> document as the band-scale setting | ratio **0.94** (1,154 cleared/h vs 1,225 baseline); more SQP iterations on fewer windows for slightly fewer folds cleared | REFUTED |
| 3D coarse warm start (64^3 cut, `giant_tile=16`) | worse fold count at equal wall -> unsafe; introduced negative edges in a certified output -> the 2D seam mechanism exists in 3D; -20% or better SQP iterations at the same fold count -> lever | `coarse_factor=2`: 21,694 folds vs 20,457 off at 98% of the wall, half the 3 h budget (5,565 s) spent on the coarse solve, worst cell -8.4258 -> -9.5413 (the only arm to worsen it), and it introduces 568 negative axial edge projections / 95 rotated cells; `coarse_factor=4` (old 2D default): neutral on iterations (1,015 vs 1,028, -1.3%) but introduces 12 negative axial edge projections / 4 rotated cells where the stage off introduces 0; on the 2.5D residual's densest box (65x128x152, 493 folds) the default warm start drove it to **23,277 folds**, minimum -0.035 -> -13.1, `damage` still 0 because every new fold sits inside the set the solve was entitled to move | REFUTED at both factors; factor 2 unsafe, factor 4 neutral-to-harmful and the source of introduced rotated cells; now off by default on 3D (`DEFAULTS_BY_DIM`, PR #128) |
| Crop-and-paste composition (2.5D residual, 11 boxes) | folds after <=14 (the true floor), damage 0 inside every box -> the route composes | full volume 33 -> **43** folds below zero after pasting (best-diagonal floor 14 -> 29); every box reported `damage 0` locally; 16 of the 31 new folded cells sit on a box boundary; boxes 4, 7, 9 and 10 moved their ENTIRE rim while reporting 0 folds and damage 0 | REFUTED. Cause: `windowed_correct` freezes each WINDOW's own ring, not the rim of the sub-volume it is handed, so a window reaching the box border moves border voxels whose remaining corners were never in that box's constraint set |
| Banded full-volume route | folds after <=14 and every remaining cell a documented true-floor cell, damage 0 | band 0 (489/517 folds, all 33 below-zero cells) did not finish in 35 h 48 m; run cut, no band's result retrievable | REFUTED (not certified; the dense z[0,24) band is the wall) |

### 13.4 The one lever found

The RAS tile pool (`giant_workers`) is the one arm in either spike that changed the answer.
On the sub-slab (`giant_tile=16`, 3 h budget): serial (`giant_workers=0`) cleared 3,781 folds
(1,225 cleared/h); `giant_workers=4` cleared **14,883** folds (3,387-3,409 cleared/h), a
**2.77x** rate improvement, and moved the worst cell from -8.4258 to -4.5949 — the only setting
in either spike to improve it. `giant_workers=8` produced byte-identical output to `giant_workers=4`
at the same wall (same 2,839/1,809 folds, 11,169 SQP iterations, L2 move 173.7339731911813 to
every recorded digit, min -4.594925564585521): a sweep is bound by its slowest tile, so doubling
the pool past 4 buys nothing on this region. Damage stayed 0 in all three arms, meeting the
pre-registered "RAS must not raise damage above 0" rule. Both pool arms overran their 3 h budget
by about 46% (15,817 / 15,718 s), because `time_budget_s` is checked only between stages, not
inside a RAS sweep.

Pinning Clarabel to one thread (`max_threads = 1` in `_HybridQP._solve_ip`, or
`RAYON_NUM_THREADS=1` in the environment for spawned pool workers, which
`dvfopt.core._pool` already sets) is the second lever, though it changes resource use rather
than throughput. On a single 17^3 frozen-ring window: **7.49 -> 1.08 cores**, an identical SQP
iteration count, constraint minima agreeing to 1e-12, and **-33%** wall on an idle box —
Clarabel's own internal parallelism was a net loss at this window size. The pin is what let 8
RAS workers fit on a 24-core box at all: unpinned, each worker would have demanded roughly 6-7
cores, i.e. an 8-worker pool would need on the order of 50 cores. The pin does change 3D
numerics on windows large enough for Clarabel's internal parallelism to engage (accepted by
ruling, see 13.1's L2-move caveat); it does not change 2D numerics, and a serial 3D run and a
pooled 3D run now agree bit for bit where they used to disagree.

### 13.5 Engine facts recorded and deliberately not fixed

Per the standing no-new-engine-logic rule, the following are documented as known behavior, not
bugs to patch in this campaign:

- **`time_budget_s` is not checked inside a RAS sweep.** A `giant_workers > 1` run finishes the
  sweep it is in progress on regardless of the budget; both pool arms in 13.4 overran by ~46%.
- **`damage` cannot see folds created inside the touched set by the warm start.** The 2.5D
  residual's box 0 went from 493 to 23,277 folds under the default coarse warm start while the
  engine reported `damage == 0` throughout, because every newly folded cell lies inside the set
  of voxels the solve was entitled to move.
- **`windowed_correct_banded` retrieves bands in submission order** (`ex.map`), so a slow first
  band blocks the checkpointing, logging and recoverability of every band that finishes after
  it, however quickly those finish. In the route attempt, 21 of 22 bands finished inside their
  workers within about 6 h; none was ever committed, because band 0 (submitted first) had not
  returned when the run was cut at 35 h 48 m.
- **A band has no time budget of its own.** The banded driver's overall budget governs the whole
  run, not any individual band, so one pathological band can consume the entire budget while
  every other band sits finished and unretrievable.

### 13.6 Reading

Across both spikes the binding constraint is SQP iteration count on dense bands, not QP solve
seconds. Q1's 17^3 window moves its constraint minimum by about 1.4 out of roughly 12.4 in 8
iterations under every QP backend tested; the sub-slab's 3 h arms spent 1,598 to 27,371 SQP
iterations and cleared 31% to 43% of 17,722 folds regardless of backend, tile size or iteration
cap. Every full-volume attempt in this section stalls on the same region: the 2.5D marching
pipeline itself stops at it, spike 2's box 0 could not clear it in a 2 h budget, and the banded
route's band 0 did not finish it in 35 h 48 m. That region is B0039's z[0, 24) band (489 of the
2.5D residual's 517 folds, all 33 of its below-zero cells).

A read-only, solver-free look at that band (correspondences vs. the 517 residual fold cells)
found it sparsely and inconsistently constrained: z-slices 1-10 carry 8-29 Laplacian
correspondence pins each and z-slices 11-23 carry 96-348, against a volume median of 904 pins
per slice; the prescribed displacement magnitude at z < 24 has median 8.1 px and p95 126.8 px,
against 1.4 / 8.1 px over the whole volume; and among the 2,913 pins at z < 24 there are 6,335
moving-point pairs within 1 px of each other (many-to-one collapse). Unlike the 2D cohort
precedent, the residual folds are not concentrated at the pins themselves: only 18 of 489 (4%)
lie within 4 px of a pin, against an 8% base rate in the same bounding box.

Whether this is a solver problem or a data problem is an open question, not a conclusion. The
2D cohort precedent (section 12.2, and the campaign's earlier correspondence-analysis findings)
showed that filtering correspondences before the Laplacian solve — dropping >25 px local
outliers, merging many-to-one groups — halved one hard slice's fold count before any solver ran.
Whether the same filtering applied to B0039's z[0, 24) band would shrink or remove the 517-fold
residual, as opposed to the windowed engine needing a cheaper or differently-formulated inner
solve to clear it as-is, has not been tested.

### 13.7 Artefacts

All gitignored unless noted.

- `benchmarks/output/spike_3d/REPORT.md`, `q1_cost.md`, `q1_cost.json`, `q2_tile{9,12,16}.json`,
  `q3_census.json` and related throwaway scripts/logs — spike 1 (QP backends, factorization
  share, tile size, seam census).
- `benchmarks/output/spike_3d_2/REPORT.md` and its JSON records (`q4a_threads*.json`,
  `q2_*.json`, `q3_c2f_*.json`, `census.json`, `q4b_gw{4,8}.json`, `q5_*.json`) — spike 2
  (Clarabel thread pin, `qp_max_iter` at band scale, 3D coarse warm start, the tile pool, the
  crop-and-paste pipeline route).
- `benchmarks/output/route_3d/` (`census.py`, `input_census.json`, `input_mask.npy`,
  `input_cells.json`, `mem.log`, `ck/state.json`) — the full-volume route attempt.
- `.superpowers/sdd/2026-09-18-3d-spike/progress.md`,
  `.superpowers/sdd/2026-09-18-3d-spike-2/progress.md`,
  `.superpowers/sdd/2026-09-19-3d-defaults/progress.md`,
  `.superpowers/sdd/2026-09-19-3d-route/progress.md`,
  `.superpowers/sdd/2026-09-11-3d-windowed-engine-port-phase4/progress.md` — pre-registered
  protocols, rulings and per-step logs (tracked in the repo, not gitignored).
- `benchmarks/output/windowed_3d/h2h.md`, `crops.md`, `cost.md`, `gate.md` — the certified 3D
  artefact table in section 13.1 (gitignored).
- CHANGELOG.md, "Changed — 3D windowed-engine defaults from spike 2" (PR #128) and "Added — 3D
  windowed engine, phase 4 (Scale)" (PR #124/#125) — the changes and measurements this section
  draws on (tracked).
