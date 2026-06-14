# Strict-feasibility 2D — LP/SLP for the 2-triangle constraint

**Date:** 2026-06-14
**Status:** Design approved; ready for implementation plan.
**Goal owner:** Andy Thai

## Goal

Achieve, on every worst-case 2D deformation field we have, **strict 2-triangle
feasibility** — `min(T1, T2)(phi*) ≥ τ = 0.01` for every cell — **with minimised
L1 deviation** `‖phi* − phi_in‖₁` from the input.

The existing methods (M10, M14, M14-Schwarz, the per-cluster SLSQP pipeline at
`notebooks/manuscript/_run_2d_clusters.py`) reach feasibility on the real B0039
field but at L1/L2 costs that are heuristically chosen, not provably minimal.
The goal of this research thread is to close that gap.

**Pareto target.** No-compromise feasibility AND best-known L1 — both. A method
that reaches feasibility at higher L1 than the cluster pipeline does not solve
the goal; neither does a method that minimises L1 but leaves one fold.

## Approach summary

Lead with the most theoretically motivated single idea — that the 2-triangle
constraint, after fixing the orientation per cell to the natural "positive"
convention, becomes affine in `phi`, so

```
min  ‖phi − phi_in‖₁
s.t. T_k(phi) ≥ τ   ∀k
```

is a **linear program** under the orientation-fix (with one linearization for
the bilinear term). Two variants are implemented and compared:

* `lp_oneshot_2tri` — single LP, linearised around a feasible harmonic seed.
* `slp_iter_2tri`  — sequential LP loop with adaptive trust region; iterates
  to the NLP KKT point.

Compared against five existing methods on a curated worst-case suite. Fallback
paths are pre-specified but built only on observed failure.

## Folder structure

New top-level folder `research/strict_feasibility_2d/`, parallel to
`notebooks/`, `benchmarks/`, etc. Name signals "active research toward a
specific manuscript goal," distinct from the exploratory `notebooks/experiments/`.

```
research/strict_feasibility_2d/
├── README.md                          # goal, success criteria, status board
├── DESIGN.md                          # link to this spec
├── worst_cases/
│   ├── catalog.md                     # selected worst cases + per-case rationale
│   ├── synthetic/                     # links to bowtie + 03c/03d + 1-2 adversarial NPZs
│   └── b0039/                         # canonical slice indices (z=12 + empirically worst)
├── analysis/
│   ├── 01_baseline_l1_gap.ipynb      # measure L1/L2 of each existing method per case
│   └── 02_lp_certifies_optimum.ipynb # LP optimum vs each baseline → "gap" number
├── algorithms/
│   ├── __init__.py
│   ├── lp_direct_2tri.py              # both lp_oneshot and slp_iter
│   ├── orientation_fix.py             # sign convention helper
│   └── highs_solver.py                # scipy.optimize.linprog(method='highs') wrapper
├── runners/
│   ├── _run_lp_synthetic.py           # batch LP on canonical + adversarial NPZs
│   ├── _run_lp_b0039.py               # batch LP on selected B0039 slices
│   └── output/
│       ├── comparison.csv             # the headline-table data source
│       └── corrected/                 # per-case corrected (2, H, W) NPZs
└── (fallback/ — only created if A/B fail criteria below)
```

## Worst-case catalog

| Bucket | Cases | Rationale |
|---|---|---|
| Synthetic minimal | `bowtie_7x7_shoelace` | Manuscript "Jdet-blind to 2-tri" demo; 2 folds, mild deviation needed |
| Synthetic dense | `03c_20x20_opposite` (58 folds), `03d_20x20_crossing` (72 folds) | Densest of the 6 canonical 2-tri cases |
| Synthetic adversarial (new, built in `_build_adversarial.py`) | dense-bowtie cluster, near-degenerate tiny-margin folds | Stress-test the orientation-fix assumption |
| B0039 pinned | `z=12` | Famously hard manuscript slice |
| B0039 empirical | 2–4 additional slices ranked highest residual L1 from the cluster pipeline | Manuscript-relevance; tests scaling |

## Algorithm spec

### Shared infrastructure (`algorithms/`)

* `orientation_fix.py::canonical_signs(H, W)` — returns the per-triangle
  positive-orientation sign vector. Used to set up constraint inequalities so
  every `T_k ≥ τ` becomes a single-sided affine inequality after linearisation.
* `highs_solver.py::solve_lp(c, A_ub, b_ub, A_eq=None, b_eq=None, bounds=None)`
  — thin wrapper over `scipy.optimize.linprog(method='highs')` returning
  `(phi_star, status_dict)`. Handles sparse `A_ub` (essential at B0039 scale)
  and the L1-epigraph reformulation:

  ```
  min sum(t)   s.t.   -t ≤ phi - phi_in ≤ t,   linearised T_k ≥ τ
  ```

* `lp_direct_2tri.py` — exposes:
  * `lp_oneshot(phi_in, *, threshold=0.01, seed='harmonic') -> (phi_out, info)`
  * `slp_iter(phi_in, *, threshold=0.01, trust_radius_0=0.5, max_iter=20, ftol=1e-6) -> (phi_out, info)`

### Algorithm A: `lp_oneshot_2tri`

1. Compute feasible seed `phi^(0)` via `dvfopt.core.wallbreakers.harmonic_extension_2d(phi_in)`.
2. Linearise each `T_k` around `phi^(0)`: `c_k + G_k·(phi − phi^(0)) ≥ τ`.
3. Solve LP (L1 epigraph + linearised constraints) via `highs_solver`.
4. Compute exact `min_k T_k(phi_LP)` using
   `dvfopt.jacobian.triangle_sign._triangle_areas_2d`. Record residual.
5. Return `phi_LP, info_dict` regardless of strict-feasibility outcome — the
   residual is itself a measured quantity.

### Algorithm B: `slp_iter_2tri`

```
phi^(0) = harmonic_extension(phi_in)
Δ = 0.5      # trust radius, in cell units
for it in range(max_iter):
    G, c = linearise_T_around(phi^(it))
    phi^(it+1) = solve_lp(
        min ‖phi − phi_in‖₁
        s.t. c + G·(phi − phi^(it)) ≥ τ,
             ‖phi − phi^(it)‖∞ ≤ Δ
    )
    T_exact = exact_T(phi^(it+1))
    if min(T_exact) < τ - safety_tol:    # safety_tol = 1e-5, slack to absorb LP/exact numerics
        Δ *= 0.5                         # shrink trust region on violation
        continue
    if ‖phi^(it+1) − phi^(it)‖∞ < ε:
        return phi^(it+1)                # converged
    if step at trust-region boundary:
        Δ *= 1.5                         # grow trust region if step was good
return phi^(it+1), with `converged=False` flag
```

`info` includes per-iter `L1_dev`, `min_T_exact`, `lp_status`, trust-radius
trajectory, and total wall-clock.

### Risks + mitigations

| Risk | Mitigation |
|---|---|
| `phi^(0)` infeasible → LP infeasible | Always start from `harmonic_extension(phi_in)` (feasible by construction); fall back to `m10` seed |
| Trust region too tight → slow | Adaptive grow/shrink per Nocedal-Wright SLP § 18.5; start at Δ=0.5 cell-units |
| LP scale at B0039 (~290k vars × ~290k constraints × 290k slacks) | HiGHS via scipy handles this in 1–10 s with sparse constraints |
| Linearisation error makes `lp_oneshot` infeasible at exact-eval | Documented as expected; SLP variant exists precisely to absorb this |

## Comparison plan

### Methods in the bake-off

| Method | Implementation | Role |
|---|---|---|
| `harmonic_only` (m02) | `harmonic_extension_2d` | Feasibility-floor (large L1, no polish) |
| `m10` | `iterative_2d_tri_harmonic_polished` | Current best feasibility-guaranteed wallbreaker |
| `m14` | `iterative_2d_tri_refine_repair` | Current best L1/L2 winner per CLAUDE.md |
| `m14_schwarz` | `iterative_2d_tri_refine_repair_schwarz` | Best on large slices per WRITEUP |
| `cluster_pipeline` | `notebooks/manuscript/_run_2d_clusters.py` | Claims 100% feasibility on real B0039 |
| **`lp_oneshot`** (new) | `algorithms/lp_direct_2tri.py::lp_oneshot` | Headline experiment A |
| **`slp_iter`** (new) | `algorithms/lp_direct_2tri.py::slp_iter` | Headline experiment B |

### Metrics (per (case, method) row in `runners/output/comparison.csv`)

| Column | Computed from |
|---|---|
| `case_id` | e.g. `bowtie_7x7_shoelace`, `b0039_z012`, `03d_20x20_crossing` |
| `method` | one of the seven above |
| `init_n_neg_2tri`, `init_min_T` | inputs from `_triangle_areas_2d` |
| `final_n_neg_2tri`, `final_min_T` | exact post-correction triangle stats |
| `feasible` | `final_n_neg == 0 AND final_min_T ≥ τ` |
| `L1_dev`, `L2_dev`, `Linf_dev` | `‖phi_out − phi_in‖_p` |
| `wall_s` | total run wall-clock |
| `n_lp_iters` (LP/SLP only) | SLP iteration count |
| `lp_solver_status` (LP/SLP only) | HiGHS termination code |

### Headline-table analysis (in `analysis/01_baseline_l1_gap.ipynb`)

Rows where `feasible=False` are **excluded from the L1 ranking**. No averaging
over infeasible solutions. The user's "no compromise" rule is enforced by row
exclusion, not by penalising infeasible L1.

The notebook answers five questions:

1. Does LP/SLP achieve strict feasibility on every case? If not, what's the
   failure mode (LP infeasibility / linearisation error / trust-region stall)?
2. Per case: `L1_dev(method) − L1_dev(lp_or_slp)` — the deviation gap of each
   existing method.
3. Is LP wall-time competitive with the cluster pipeline at B0039 scale?
4. Where does iteration matter? `slp_iter.L1_dev < lp_oneshot.L1_dev` ⇒ by
   how much per case?
5. Does the harmonic seed matter? Run `lp_oneshot` from `phi=0` and from
   `harmonic_extension(phi_in)`; report the gap.

### Adversarial validation

To catch a method that "wins on numbers but cheats geometrically":

* **Visual check** — bowtie wireframe should look smooth, not snap-to-edge.
* **Exact-T re-check** — `_triangle_areas_2d(lp_out)` strictly, not the
  linearisation. Failures here are the SLP trigger for trust-region shrink.
* **Monotonicity check** — L1 must decrease monotonically across SLP iters
  (a regression signals a bug).

## Fallback plan

Each fallback is keyed to a specific failure of A or B. Lands in
`research/strict_feasibility_2d/fallback/` only on observed need.

| Trigger | Fallback algorithm | Why it should help |
|---|---|---|
| LP infeasible from harmonic seed | `lp_from_m10_seed` — replace seed with full m10 pipeline | m10's ALM nudge pushes strictly into interior |
| LP infeasible from m10 seed too | `signed_lp` — disjunctive LP allowing either sign per cell with parsimony penalty | Pathological inputs where canonical sign fix breaks |
| SLP oscillates between iterates | `slp_with_filter` — Fletcher-style filter line-search | Classical SLP remedy |
| SLP converges but `min_T < τ` (linearisation error) | `sqp_2tri` — sequential QP with exact quadratic constraint | Kills the linearisation-error term |
| LP/SLP correct but slow on B0039 (>30 s/slice) | `cluster_lp` — apply LP per fold cluster | Scales by per-cluster sparsity |
| Cluster_lp also slow | `admm_split` — split into (project-feasible) + (shrink-to-input) | ADMM scales + parallelises trivially |
| None of the above wins on L1 | `active_set_kkt` — identify binding constraints from LP dual; project onto KKT manifold | The L1 NLP optimum lies on a feasibility-polytope face |

### Fallback trigger criteria

A fallback gets built only if A/B fail on either:

1. `feasible=False` on any case → blocks the strict-feasibility goal.
2. `L1_dev > L1_dev(cluster_pipeline)` on >50% of B0039 cases → fails the
   "Pareto-best L1" goal.

If both criteria are cleared by A or B, **no fallback is built**; LP-direct +
SLP variants are the answer and the work proceeds to writeup.

Order of exploration if triggered: row 1 → 2 → 5 → 3 → 6 → 4 → 7.

## Success criteria (for the whole research thread)

1. **Algorithm correctness.** At least one of `lp_oneshot` or `slp_iter` (or a
   fallback) achieves `feasible=True` on every case in the worst-case catalog.
2. **L1 dominance.** That same method achieves `L1_dev ≤ L1_dev(cluster_pipeline)`
   on every B0039 case in the catalog.
3. **Reproducibility.** Running `_run_lp_synthetic.py` and `_run_lp_b0039.py`
   end-to-end on a fresh checkout reproduces the headline-table numbers
   bit-for-bit.

## Out of scope (for this design)

* 3D 6-tetrahedron constraint (separate follow-on; same algorithm extends
  naturally but tetrahedra add disjunctive-orientation cases).
* GPU LP solvers — HiGHS on CPU is the targeted backend for now.
* Wall-time competitiveness with the SLSQP-windowed live-render GUI path
  (this is a research thread, not a user-facing path).
* Notebook reorganisation of `notebooks/experiments/` — separate cleanup.
