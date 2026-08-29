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

Ordering (branch `reseed-before-mop`): re-seed *before* the terminal mop — see §6.

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
linear condition (area ≥ δ²(1−κ²)/(1+κ²)); measured separately.

The engine integration restricts the QP's variables to the fold neighbourhoods
(`find_windows` boxes; everything else fixed), which keeps it tiny on ordinary
slices, and books the moved pixels as `touched` exactly like the coarse warm
start (no-damage unchanged).

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
69 / 5,370 s. Fix: run the re-seed as soon as the round loop plateaus, mop only
for leftovers (`reseed_before_mop=True`, branch `reseed-before-mop`).

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
