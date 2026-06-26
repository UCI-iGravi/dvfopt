# TVCG manuscript pivot — design spec

**Date:** 2026-06-25
**Target venue:** IEEE TVCG
**Status:** design approved; spec for review before `.tex` edits begin
**Scope of this spec:** restructuring of `writing/` (the `dvfopt.tex` body + `tvcg_manuscript.tex` front matter). Does *not* cover code changes to `dvfopt/`, only the new benchmark/analysis *runs* the manuscript needs.

---

## 1. Premise pivot (what changes and why)

**Old premise (current draft):** a hybrid Laplacian-registration paper with three loosely-coupled threads — (1) interpolation-space analysis (fixed vs. moving), (2) Laplacian correspondence augmentation, (3) post-hoc DVF correction via SLSQP. The draft's Methods section already leans heavily into the geometric measure + SLSQP, but the abstract/intro/related-work/conclusion still pitch the Laplacian threads, and the solver story is years behind the actual code (only SLSQP + windowed SLSQP are described).

**New premise:** a geometry-centric paper on **detecting and correcting folding in deformation vector fields (DVFs)**. Folding is recast as a *geometric* property — the sign of triangulated (2D) / tetrahedralized (3D) cell areas/volumes — and the standard discrete-Jacobian (central-difference) check is shown to be provably blind to intra-cell folds. Once folding is posed geometrically, fold-free correction is a constrained optimization with an exact feasible region, solvable efficiently and registration-method-agnostically — up to a geometric ceiling beyond which no correction exists.

**Why this direction:**
- It aligns the paper with where the research actually landed (LP/SLP, `auto_slp`, m10/m14, cluster decomposition, 2D+3D, strict-feasibility studies, profiling-grounded speed/accuracy analysis).
- The geometric framing (triangulation / Freudenthal–Kuhn tetrahedralization, signed areas/volumes) fits TVCG better than a registration-methods framing.
- The strongest novel result — geometrically untangleable cells as a hard feasibility ceiling — becomes a genuine theoretical contribution with a classical analog (Schönhardt polyhedron).

**Working title:** *Beyond the Discrete Jacobian: Triangulated Fold Detection and Feasible Correction of Deformation Vector Fields.*

## 2. Key decisions (locked)

| Decision | Choice |
|---|---|
| Laplacian threads (interpolation space, correspondence augmentation) | **Cut entirely.** Laplacian becomes one DVF *source* among several. |
| DVFopt framing | **Balanced three contributions** (measure + method + benchmark/limits). |
| Evaluation breadth | **Multi-source DVFs** — correct fields from several registration backends (Laplacian + VoxelMorph + Elastix/ANTs). |
| Dimensionality | **2D is the spine; 3D is a self-contained extension section.** |
| Geometric-ceiling result | Lives in the 3D extension as its capstone; mentioned in abstract/intro as "the 3D extension reveals a hard geometric limit." |

## 3. Contributions (the balanced three)

- **C1 — Exact geometric fold measure.** 2D triangle signed area / 3D Freudenthal–Kuhn tet signed volume; exact within each cell (equals the determinant of the piecewise-affine map on each simplex); central differences proven and *quantified* (see Exp-6) to miss intra-cell folds. Includes the diagonal/tetrahedralization-dependence analysis.
- **C2 — DVFopt correction methods.** Constrained-optimization formulation (objectives L1/L2/L2,1; constraint = signed area/volume ≥ τ), a curated solver family, and the **cluster-LP / `auto_slp` (2D)** and **tet-LP (3D)** as a new method that Pareto-dominates prior barrier/ALM approaches. Windowed/cluster decomposition is the novel scaling mechanism.
- **C3 — Benchmark + feasibility limits.** Head-to-head across methods *and* registration sources on a common axis (feasibility, L1 deviation, wall-time), with distributional statistics; the speed/accuracy frontier; strict-feasibility study; and the geometric ceiling (untangleable cells) with diagonal-flip recovery and density-dependent regimes.

## 4. Section outline (cut / keep / new vs. current `dvfopt.tex`)

| # | Section | Status | Source in current draft |
|---|---|---|---|
| 1 | Introduction | **Rewrite** — geometric thesis + C1–C3; drop interpolation-space/augmentation pitch | §"Restructure Introduction" |
| 2 | Related work | **Restructure** — trim registration survey; **add** locally-injective-maps / mesh-untangling literature (must-add #1); fold detection & Jacobian computation | §"Restructure Related work" |
| 3 | Background: folding & the discrete Jacobian | **Keep (2D-focused)** — registration/invertibility defs, central-difference Jacobian, two failure modes | §4.1 (solid as written) |
| 4 | Geometric measure (2D triangle) | **Keep + expand** — signed area, exactness, + 2D diagonal-choice note | §4.2 |
| 5 | Correction as constrained optimization | **Keep** — objectives L1/L2/L2,1 + constraint family | §4.4 |
| 6 | Solvers (DVFopt) | **Rewrite/expand** — taxonomy: baselines (full/windowed SLSQP, NMVF) → barrier/ALM (m10/m14) → LP family (LP-direct → SLP → cluster/`auto_slp` champion) | §"SLSQP" + §"Windowed SLSQP" (expand heavily) |
| 7 | Experiments (2D) | **New** — datasets, metrics, benchmark, multi-source correction, Pareto frontier, distributions | replaces thin SLSQP table |
| 8 | Extension to 3D | **New (merged)** — tet measure (Freudenthal–Kuhn), 3D tet-LP, B0039 3D strict-feasibility, **the geometric ceiling + Schönhardt analog** | §4.3 (3D fold detection) + research/strict_feasibility_3d |
| 9 | Discussion / limitations / conclusion | **Rewrite** | §"Conclusion" + §"Summary and Future Work" |

**Cut entirely:** interpolation-space (fixed-vs-moving) analysis, correspondence augmentation, and the Laplacian-specific conclusion paragraphs. The `laplacian.tex` input stays disabled.

## 5. Required additions reviewers will expect (must-adds 1–5)

1. **Locally-injective-maps / mesh-untangling literature.** The constraint (every simplex signed area/volume ≥ τ > 0) is the foldover-free PL-map constraint from geometry processing. Cite and differentiate: Schüller et al. *Locally Injective Mappings* (SGP 2013), Smith & Schaefer *Bijective Parameterization with Free Boundaries* (2015), Du et al. *Lifting Simplices to Find Injectivity* (SIGGRAPH 2020), Garanzha et al. *Foldover-free maps in 50 lines* (SIGGRAPH 2021); FEM untangling (Knupp; Toulorge et al. *Robust untangling of curvilinear meshes*). Differentiation: DVF-specific, registration-agnostic, medical-volume scale, windowed/cluster decomposition for scalability, real-data feasibility ceiling. **Goes in §2 + a positioning paragraph in §6.**
2. **Schönhardt-polyhedron analog for the geometric ceiling.** Frame the untangleable cells (fold under all 4 cube diagonals) as deformation-induced non-tetrahedralizable configurations, classically embodied by the Schönhardt polyhedron (a twisted prism requiring Steiner points). **Goes in §8.**
3. **Downstream registration-quality impact.** Show a task-level metric before/after correction — label Dice and/or intensity similarity (NCC/SSD) on the warped image (TRE where landmarks exist). Demonstrates correction removes folds without degrading the registration. **Goes in §7 (and §8 for 3D).**
4. **Standard registration fold metrics.** Report % non-positive-Jacobian voxels and **SDlogJ** (std. of log-Jacobian) alongside the tri/tet measure, so results are comparable to published (e.g., Learn2Reg) numbers. **Goes in §7 metrics.**
5. **At least one standard public dataset.** One public benchmark beyond mouse brain (OASIS/Learn2Reg, LPBA40, or IXI — sliced for the 2D spine) to demonstrate generalization and anchor against known baselines. **Goes in §7 datasets.**

## 6. Committed discretionary analyses (6–9)

6. **Central-difference blindness statistics.** On real DVFs, count cells where central differences report J>0 but the triangle measure reports a fold (false negatives), and vice versa. Table: "central differences miss X% of true folds on real data." Nearly free (both quantities already computed). **§7 / §3.**
7. **Speed/accuracy Pareto frontier plot.** Consolidate SOLVER_OPTIMIZATION.md data (cluster vs global L1 gap, merge_dilation, seed sweep) into a frontier figure framing the tradeoff as a deliberate dial. **§7.**
8. **Distributional statistics.** Replace single-slice anecdotes with mean±std / box plots across all 528 B0039 slices and across sources. Re-runs of existing pipelines, no new method work. **§7.**
9. **Inverse-consistency check.** Verify corrected fields are globally invertible: compose with a numerical inverse and report inverse-consistency/composition error. Addresses the local-vs-global injectivity gap. **§7 + a sentence in §4/§5 noting local positivity is necessary but not globally sufficient.**

**Space-permitting (not committed):** diffeomorphic-by-construction baseline comparison (#10 — address in text regardless), threshold-τ sweep (#11), quantified objective ablation (#12).

## 7. Figures & tables

**Reuse (already drawn):** signed-areas figure incl. the central-diff-misses-fold panel (teaser candidate); quad/triangle assignment; Freudenthal–Kuhn tet decomposition; per-vertex tetrahedra; geometric primitives; objective L1/L2 comparison; runtime-vs-gridsize / runtime-vs-#folds.

**New figures:**
- F-new-1: a cell that folds under all four cube diagonals — visual proof of the untangleable ceiling (§8).
- F-new-2: speed/accuracy Pareto scatter, L1 deviation vs wall-time across methods (§7, Exp-7).
- F-new-3: density-dependent feasibility ceiling (§8).
- (teaser) folded DVF → central-diff says fine → tri measure catches → corrected output.

**Tables:**
- T1: method taxonomy (method / measure / feasibility guarantee / parallelizable / 2D-3D).
- T2: main 2D benchmark (methods × feasibility, min-T, L1, wall) on B0039 + synthetic, with distributions (Exp-8).
- T3: multi-source correction (across VoxelMorph / Elastix / Laplacian DVFs), incl. downstream Dice/NCC (must-add #3) and SDlogJ / %non-pos-J (#4).
- T4: strict-feasibility & geometric-ceiling results (§8), incl. diagonal-flip recovery and threshold relaxation.
- T5 (optional): central-diff blindness false-negative rates (Exp-6).

## 8. New runs required (work plan, given multi-source)

- **Generate DVFs from VoxelMorph + Elastix/ANTs** on B0039 (and the chosen public dataset) using `benchmarks/registration/` scaffolding → detect folds (tri/tet) → correct with the curated method set → record T2/T3 metrics + downstream quality + SDlogJ.
- **Public dataset** acquisition + 2D slicing pipeline (must-add #5).
- **Curated head-to-head benchmark** on the common axis. Most 2D/3D numbers already exist in research folders — consolidate, do not re-derive.
- **Distributional sweep** across 528 B0039 slices (Exp-8) — re-run existing `auto_slp` pipeline, aggregate.
- **Inverse-consistency computation** on corrected fields (Exp-9) — new small analysis script.
- **Already in hand (consolidate only):** central-diff blindness counts (Exp-6), Pareto-frontier data (Exp-7), diagonal-flip + density-ceiling numbers (§8, from `strict_feasibility_3d`).

## 9. Scope risks & fallbacks

- **C2 "champion" claim** + full benchmark + multi-source data is a lot. Fallback if a reviewer challenges per-method novelty: demote C2 from "new champion method" to "the LP/cluster formulation," keeping C1+C3 as the spine.
- **Multi-source runs are the dominant new effort.** If backend integration stalls, the minimum viable version is Laplacian + one learned method (VoxelMorph), with Elastix/ANTs as "additional sources" added later.
- **3D extension must earn its place** beyond "we also ran 3D" — the geometric ceiling + Schönhardt framing is what does that.

## 10. Out of scope

- Code changes to `dvfopt/` (the package already implements everything the manuscript describes).
- The interpolation-space / correspondence-augmentation research (cut; could become a separate future paper, not tracked here).
- Notebook/benchmark refactors beyond what the new runs require.
