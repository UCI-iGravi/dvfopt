# TVCG Folding-Correction Pivot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the TVCG manuscript in `writing/` from a Laplacian-registration hybrid into a geometry-centric paper on triangulated/tetrahedralized fold detection and feasible post-hoc correction of deformation vector fields (DVFs).

**Architecture:** Two work streams. (1) *Result generation* — produce/consolidate the new benchmark numbers, figures, and tables the new framing needs (multi-source DVFs, public dataset, distributional stats, inverse-consistency, blindness stats, Pareto frontier, 3D ceiling). (2) *Writing* — restructure `dvfopt.tex` section-by-section and rewrite the front matter in `tvcg_manuscript.tex`. Results precede the sections that cite them; front matter is written last.

**Tech Stack:** LaTeX (IEEEtran), Python (`dvfopt` package + `benchmarks/` + `research/strict_feasibility_*`), matplotlib for figures, BibTeX.

## Global Constraints

- **Git commits are performed by the user.** Do NOT run `git commit` or `git push`. At each "Commit checkpoint", stage the relevant files (`git add ...`) and notify the user that the task is ready to commit. (Project rule.)
- **Design spec is authoritative:** `docs/superpowers/specs/2026-06-25-tvcg-folding-correction-pivot-design.md`. Every task references a spec section; if writing and spec disagree, stop and reconcile.
- **Verification for writing tasks (NO local TeX — Overleaf-managed):** there is no local `pdflatex`. A writing task is "done" only when (a) the LaTeX **sanity checker** below passes (no structural errors: balanced environments/braces, no undefined `\ref`/`\cite`, no duplicate labels), (b) the assigned spec bullet(s) are covered, (c) the task introduces no new undefined refs/cites for content it adds. The author runs the real compile in Overleaf.
- **Check command (run from repo root):** `python tools/latex_check.py writing/tvcg_manuscript.tex` — expected final line `[latex_check] OK` (or only `WARN` lines). Any `ERROR` line fails the task.
- **Data conventions (from CLAUDE.md):** 2-tri solvers use `phi[:N]=dy, phi[N:]=dx` (DY_FIRST); Jdet-SLSQP uses DX_FIRST. Jacobian threshold τ = 0.01. Do not cross-mix phi packings.
- **Prose adaptation:** for writing tasks, each step specifies the *claims, structure, and exact source material* to include (a content checklist), not ghostwritten final prose — final wording is produced at execution and reviewed. This is intentional and is NOT a placeholder; the substance is fully specified.
- **Laplacian content stays cut.** `\input{laplacian}` remains commented out in `tvcg_manuscript.tex`. Do not re-enable interpolation-space or correspondence-augmentation material.

---

## Execution decisions (2026-06-25, locked at kickoff)

- **Working location: main repo dir (NO worktree).** `writing/` is **gitignored** (Overleaf-managed), so it does not exist in a git worktree and branch isolation cannot apply to it. The worktree was created and removed at kickoff once this was discovered. All edits happen in place at `c:\Users\Andy\Documents\GitHub\UCI-iGravi\deformation-field-processing\writing\`. Task file paths are relative to the main repo root. Because the manuscript is gitignored, the "stage + user commits" rule applies only to tracked artifacts (the result scripts, `docs/`, `tools/`); manuscript edits are saved to disk and synced via Overleaf by the author, not committed.
- **Prose mode = HYBRID.** Writing subagents produce *full prose* for the technical/mechanical sections (§2 Related work, §3 Background, §4 Geometric measure, §5 Optimization formulation, §6 Solvers, §7 Experiments body, §8 Extension to 3D) and *skeleton drafts* (claims + structure + citations; author writes final wording) for the high-stakes voice sections (§1 Introduction, §9 Discussion, and the abstract in `tvcg_manuscript.tex`). Each Phase-C task states which mode applies.
- **DEFERRED to the user (later work) — do NOT execute now:**
  - **Task B1, B3, B4** (backend-dependent DVF generation: VoxelMorph/ANTs runs, public-dataset acquisition) — user sets eval scope later. Probed available: VoxelMorph ✅, ANTs/antspyx ✅ (also diffeomorphic baseline #10), Laplacian DVF ✅, second subject `b0036` ✅; itk-elastix ❌, no public dataset present.
  - **Benchmark scale** (representative subset vs full volume) — user decides later.
  - **All of Phase B (B2, B5, B6, B7, B8)** result-generation is ALSO deferred: it consolidates artifacts that live as *untracked* files in the main working dir (`research/strict_feasibility_3d/runners/output/*`, `research/SOLVER_OPTIMIZATION.md`), absent from this worktree. These run in the main repo during the user's result pass.
- **Consequence for Phase C:** result tables (T2/T3/T4/T5) and result figures (F-new-1/2/3, distribution box plots, Pareto) are inserted as **clearly-marked LaTeX placeholders** — a `% RESULT PLACEHOLDER (deferred): <what fills it, which Task>` comment plus a stub `\begin{table}`/`\begin{figure}` with a TODO caption — so the document compiles and the author drops numbers in later. Writing subagents must NOT invent numbers. Where a result already exists in the *committed* manuscript (e.g. the old `tab:jac_results` windowed-SLSQP numbers), it may be reused and relabeled.
- **Now-executable scope:** Phase A (A1 bib, A2 asset inventory), Phase C (all writing tasks, with placeholders per above), Phase D (integration/compile/coverage). Phase B is the deferred user pass.

---

## File map

**Writing (edit):**
- `writing/tvcg_manuscript.tex` — front matter: title, authors, abstract, keywords, `\input` order.
- `writing/dvfopt.tex` — the paper body (all numbered sections).
- `writing/dvfopt.bib` — citations (new entries for geometry-processing / untangling / metrics / diffeomorphic refs).

**Results (create under a new results area):**
- `research/manuscript_results/` — new home for the consolidation/analysis scripts this plan adds.
  - `blindness_stats.py` (Exp-6)
  - `multisource_benchmark.py` (Exp-6/T2/T3 driver)
  - `public_dataset_prep.py` (must-add #5)
  - `distributional_sweep.py` (Exp-8)
  - `inverse_consistency.py` (Exp-9)
  - `pareto_consolidate.py` (Exp-7)
  - `ceiling_consolidate.py` (§8, from `research/strict_feasibility_3d`)
  - `output/` — generated `.txt`/`.npy`/`.csv` result tables (gitignored like sibling `runners/output/`).
- `writing/images/dvfopt/` — figure PDFs/PNGs (existing reuse assets live here; new figures land here).

**Read-only sources (consolidate from, don't modify):**
- `research/SOLVER_OPTIMIZATION.md`, `research/strict_feasibility_3d/README.md` and its `runners/output/*.txt`.
- `benchmarks/registration/` scaffolding + `benchmarks/benchmark_utils.py`.
- `dvfopt/` package (Solver, strategies, jacobian primitives).

---

## Phase A — References & figure scaffolding

### Task A1: Bibliography additions

**Files:**
- Modify: `writing/dvfopt.bib`

**Interfaces:**
- Produces: BibTeX keys later writing tasks cite — `schuller2013lim`, `smith2015bijective`, `du2020lifting`, `garanzha2021foldfree`, `knupp2001untangling`, `toulorge2013robust`, `schonhardt1928`, `learn2reg2023`, `dalca2018diffeomorphic`, `ants_syn` (reuse existing `AVANTS200826` if present).

- [ ] **Step 1: Inventory existing keys**

Run: `grep -E "^@" writing/dvfopt.bib writing/laplacian.bib` and confirm which of the keys above already exist (e.g., `AVANTS200826`, `VoxelMorph2019`, `Klein2010elastixAT` likely exist). Only add missing ones; reuse existing keys to avoid duplicates.

- [ ] **Step 2: Add missing entries**

Add BibTeX entries for the geometry-processing / untangling / Schönhardt / metrics / diffeomorphic references (must-add #1, #2, #4; spec §5). Use canonical citations:
- Schüller et al., "Locally Injective Mappings", Computer Graphics Forum / SGP 2013.
- Smith & Schaefer, "Bijective Parameterization with Free Boundaries", SIGGRAPH 2015.
- Du et al., "Lifting Simplices to Find Injectivity", ACM TOG (SIGGRAPH) 2020.
- Garanzha et al., "Foldover-free maps in 50 lines of code", ACM TOG (SIGGRAPH) 2021.
- Knupp, "Hexahedral and tetrahedral mesh untangling", Engineering with Computers 2001.
- Toulorge et al., "Robust untangling of curvilinear meshes", JCP 2013.
- Schönhardt, "Über die Zerlegung von Dreieckspolyedern in Tetraeder", Math. Annalen 1928.
- Hering et al. / Learn2Reg benchmark (for SDlogJ + fold metrics), IEEE TMI 2023.
- Dalca et al., "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration", MICCAI 2018 / MedIA 2019.

- [ ] **Step 3: Verify compile**

Run the build command. Expected: no `Citation undefined` errors *for keys already used*; new keys won't warn until cited. No BibTeX syntax errors.

- [ ] **Step 4: Commit checkpoint**

Stage `writing/dvfopt.bib`; notify user: "Task A1 (bibliography) ready to commit."

---

### Task A2: Figure & table asset inventory

**Files:**
- Create: `research/manuscript_results/ASSET_INVENTORY.md`

**Interfaces:**
- Produces: a checklist mapping every figure/table in spec §7 to either an existing asset path or a "to-generate" task ID. Consumed by all writing tasks (so they reference real paths, not invented ones).

- [ ] **Step 1: List existing figure assets**

Run: `ls writing/images/dvfopt/` and `ls writing/images/laplacian_analysis/registration/`. Record which spec-§7 "reuse" figures exist (signed_areas.pdf, corner_point_notation.pdf, tet_decomposition.pdf, per_vertex_tetrahedra.pdf, geometric_primitives.pdf, objective_comparison2.png, and the runtime plots).

- [ ] **Step 2: Write inventory file**

Create `ASSET_INVENTORY.md` with three columns: asset (T1–T5, F-new-1..3, reuse figs) | status (exists / to-generate by Task __) | path. Mark F-new-1 (untangleable cell), F-new-2 (Pareto), F-new-3 (density ceiling), and tables T2/T3/T5 as to-generate, pointing at the Phase-B tasks below.

- [ ] **Step 3: Commit checkpoint**

Stage `research/manuscript_results/ASSET_INVENTORY.md`; notify user.

---

## Phase B — Result generation

### Task B1: Registration-backend & dataset availability check

**Files:**
- Create: `research/manuscript_results/output/backend_availability.txt`

**Interfaces:**
- Produces: a recorded decision on which DVF sources are runnable (drives B3/B4 scope and the §9 fallback).

- [ ] **Step 1: Probe backends**

Run: `python -c "import voxelmorph"`, `python -c "import itk; import itk.elastix" 2>/dev/null || python -c "import SimpleITK"`, and check for ANTs (`python -c "import ants"` or CLI `which antsRegistration`). Record import successes/failures.

- [ ] **Step 2: Probe public dataset access**

Decide the public dataset (OASIS via Learn2Reg, LPBA40, or IXI). Check whether data is locally available under `data/` or must be downloaded. Record the chosen dataset + access path.

- [ ] **Step 3: Write availability + fallback decision**

Write `backend_availability.txt`: list runnable sources. If fewer than two of {VoxelMorph, Elastix, ANTs} are runnable, record that the §9 fallback applies (Laplacian + VoxelMorph minimum) and note which backends are deferred.

- [ ] **Step 4: Commit checkpoint**

Stage the output file; notify user with the availability summary so they can confirm the eval scope before downstream tasks run.

---

### Task B2: Central-difference blindness statistics (Exp-6)

**Files:**
- Create: `research/manuscript_results/blindness_stats.py`
- Create: `research/manuscript_results/output/blindness_stats.csv`

**Interfaces:**
- Consumes: `dvfopt.jacobian.numpy_jdet.jacobian_det2D` (central-diff), `dvfopt.jacobian` triangle signed-area primitives (`tri_areas_flat` via `dvfopt.core.tri_primitives`).
- Produces: per-DVF counts → CSV columns `source, slice, n_cells, n_fold_tri, n_fold_centraldiff, n_falseneg, n_falsepos`. Feeds Table T5 and a sentence in §3/§7.

- [ ] **Step 1: Write the analysis script**

Implement `blindness_stats.py`: load a set of real B0039 DVF slices (raw, pre-correction), compute (a) central-difference Jdet sign per grid point, (b) triangle signed-area sign per cell. Define a cell as "truly folded" if either of its two triangles has negative signed area. Count false negatives (central-diff says all-positive in a cell the triangle measure flags) and false positives. Aggregate to CSV.

- [ ] **Step 2: Run it**

Run: `python research/manuscript_results/blindness_stats.py`
Expected: CSV written with non-zero `n_falseneg` total (the headline "central differences miss X% of folds" number).

- [ ] **Step 3: Sanity check**

Verify on the synthetic case from the manuscript (the `(+1.2,0)/(-1.2,0)` opposed-displacement cell, Fig. signed_areas panel c): central diff should report positive everywhere while the triangle measure flags the fold. Assert this single known case in the script's `__main__` or a tiny adjacent test.

- [ ] **Step 4: Commit checkpoint**

Stage `blindness_stats.py` + CSV; notify user with the headline false-negative rate.

---

### Task B3: Multi-source DVF correction benchmark (T2/T3, must-add #3/#4)

**Files:**
- Create: `research/manuscript_results/multisource_benchmark.py`
- Create: `research/manuscript_results/output/multisource_benchmark.csv`

**Interfaces:**
- Consumes: B1 backend list; `benchmarks/registration/` helpers + `benchmarks/benchmark_utils.py`; `dvfopt.Solver` / `correct_dvf` for the curated method set; the strict-feasibility `auto_slp` path.
- Produces: CSV with `source, method, feasible, min_T, L1, wall_s, pct_nonpos_J, SDlogJ, dice_before, dice_after, ncc_before, ncc_after`. Feeds T2 (methods axis) and T3 (sources axis).

- [ ] **Step 1: Fix the curated method set**

Pin the methods for the benchmark (spec §6 taxonomy): `NMVF`, `SLSQPWindowedStrategy` (Jdet baseline), `M10Strategy`, `M14Strategy`, and the champion `auto_slp` (cluster-LP). Document the list at the top of the script so T1/T2 match.

- [ ] **Step 2: Generate DVFs from each available source**

For each runnable backend from B1, register the standard B0039 moving/fixed pairs and save the resulting DVFs as `.npy` under `output/dvfs/<source>/`. For Laplacian, reuse existing B0039 DVF checkpoints (no regeneration). Cache so reruns skip regeneration.

- [ ] **Step 3: Correct + measure**

For each (source, method): run correction, then compute feasibility/min_T/L1/wall via the geometric measure, plus standard metrics — `pct_nonpos_J` and `SDlogJ` (std of log central-diff Jacobian over positive cells), and downstream Dice (on available segmentation labels) and NCC (warped-vs-fixed intensity) before/after correction. Write rows to CSV.

- [ ] **Step 4: Run it**

Run: `python research/manuscript_results/multisource_benchmark.py`
Expected: CSV populated for all runnable sources × methods; champion `auto_slp` shows feasible=True with lowest or near-lowest L1 and dice_after ≈ dice_before (correction doesn't break registration).

- [ ] **Step 5: Commit checkpoint**

Stage script + CSV (+ note cached DVF location); notify user with the T2/T3 highlights.

---

### Task B4: Public dataset preparation + correction (must-add #5)

**Files:**
- Create: `research/manuscript_results/public_dataset_prep.py`
- Modify: `research/manuscript_results/multisource_benchmark.py` (add the public dataset as an additional `source`)
- Create: `research/manuscript_results/output/public_dataset_benchmark.csv`

**Interfaces:**
- Consumes: chosen public dataset from B1; the same correction+metrics harness as B3.
- Produces: rows appended for `source = <public_dataset>` into the benchmark CSV.

- [ ] **Step 1: Acquire + slice**

Implement `public_dataset_prep.py`: load the chosen public dataset, produce 2D slices matching the 2D spine, and either (a) use provided DVFs or (b) register with one available backend to obtain DVFs. Save under `output/dvfs/<public>/`.

- [ ] **Step 2: Run correction on public DVFs**

Re-run `multisource_benchmark.py` restricted to the public source. Expected: public-dataset rows in CSV; demonstrates generalization beyond mouse brain.

- [ ] **Step 3: Commit checkpoint**

Stage scripts + CSV; notify user.

---

### Task B5: Distributional sweep across B0039 slices (Exp-8)

**Files:**
- Create: `research/manuscript_results/distributional_sweep.py`
- Create: `research/manuscript_results/output/distribution_b0039.csv`
- Create: `writing/images/dvfopt/distribution_boxplots.pdf`

**Interfaces:**
- Consumes: the full 528-slice B0039 volume + `auto_slp` per-slice path.
- Produces: per-slice metrics CSV + a box-plot figure (L1, wall, fold-count distributions). Replaces single-slice anecdotes in T2.

- [ ] **Step 1: Aggregate per-slice metrics**

Implement `distributional_sweep.py`: run (or load cached Stage-1 results from `research/strict_feasibility_3d`) per-slice `auto_slp`, recording per-slice `n_folds_in, min_T_in, L1, wall, feasible`. Write CSV.

- [ ] **Step 2: Plot distributions**

Produce `distribution_boxplots.pdf` (box plots / violin of L1 and wall across slices, and the fold-count histogram). Use matplotlib; match the repo's existing figure style.

- [ ] **Step 3: Run + sanity check**

Run: `python research/manuscript_results/distributional_sweep.py`
Expected: 528 rows; mean±std reported to console for the §7 prose; feasible=True for the overwhelming majority (consistent with README's full-volume Stage 1).

- [ ] **Step 4: Commit checkpoint**

Stage script + CSV + figure; notify user with mean±std headline.

---

### Task B6: Inverse-consistency check (Exp-9)

**Files:**
- Create: `research/manuscript_results/inverse_consistency.py`
- Create: `research/manuscript_results/output/inverse_consistency.csv`

**Interfaces:**
- Consumes: corrected DVFs from B3 (champion method) + raw DVFs.
- Produces: CSV `source, slice, ic_error_before, ic_error_after, max_residual` — composition error of the corrected map with its numerical inverse.

- [ ] **Step 1: Implement inverse + composition error**

Implement `inverse_consistency.py`: for a corrected field φ, compute a numerical inverse (fixed-point iteration or grid inversion), then measure ‖(φ ∘ φ⁻¹) − id‖ over the grid (mean + max). Compare before vs. after correction.

- [ ] **Step 2: Run it**

Run: `python research/manuscript_results/inverse_consistency.py`
Expected: ic_error_after is small and finite where the corrected field is locally feasible; document any cases where local positivity does not yield global injectivity (the spec §6/#9 caveat).

- [ ] **Step 3: Commit checkpoint**

Stage script + CSV; notify user.

---

### Task B7: Pareto-frontier consolidation (Exp-7)

**Files:**
- Create: `research/manuscript_results/pareto_consolidate.py`
- Create: `writing/images/dvfopt/pareto_frontier.pdf`

**Interfaces:**
- Consumes: numbers already in `research/SOLVER_OPTIMIZATION.md` (cluster vs global L1 gap; merge_dilation sweep; seed sweep) + B3 CSV.
- Produces: F-new-2, a scatter of L1 deviation vs wall-time across methods/operating points with the frontier highlighted.

- [ ] **Step 1: Encode the data points**

Implement `pareto_consolidate.py` with the operating points from SOLVER_OPTIMIZATION.md (clustered default ~22 s / L1 2372; merge_dilation=8 ~120 s / L1 1998; global ~404 s / L1 2090; plus seed-sweep points) and B3 method points.

- [ ] **Step 2: Plot the frontier**

Produce `pareto_frontier.pdf` (log-x wall-time vs L1, Pareto-optimal points connected). Annotate the champion default as the knee.

- [ ] **Step 3: Run it**

Run: `python research/manuscript_results/pareto_consolidate.py`
Expected: figure written; champion default lies on/near the frontier knee.

- [ ] **Step 4: Commit checkpoint**

Stage script + figure; notify user.

---

### Task B8: 3D ceiling & diagonal-flip consolidation (§8)

**Files:**
- Create: `research/manuscript_results/ceiling_consolidate.py`
- Create: `writing/images/dvfopt/untangleable_cell.pdf` (F-new-1)
- Create: `writing/images/dvfopt/density_ceiling.pdf` (F-new-3)
- Create: `research/manuscript_results/output/ceiling_table.csv` (T4)

**Interfaces:**
- Consumes: results already in `research/strict_feasibility_3d/README.md` + `runners/output/*.txt` (diagonal counts, density-regime table, threshold-relaxation table).
- Produces: T4 table data + the two §8 figures.

- [ ] **Step 1: Consolidate the ceiling tables**

Implement `ceiling_consolidate.py`: encode the diagonal-choice counts (default (0,7): 173 folds → best-per-cell: 94 unavoidable), the density-regime table (sparse ✅ / moderate ✅ / dense ⚠️ 19 residual), and the threshold-relaxation table. Emit `ceiling_table.csv`.

- [ ] **Step 2: Render F-new-1 (untangleable cell)**

Produce `untangleable_cell.pdf`: a single deformed cube cell shown with each of the 4 main-diagonal tetrahedralizations, all containing at least one negative-volume tet — the visual proof + Schönhardt analog. Use a representative untangleable cell from the residual set (extract coordinates via the existing `focused_lp_6tet.py` / diagonal-flip code path).

- [ ] **Step 3: Render F-new-3 (density ceiling)**

Produce `density_ceiling.pdf`: fold-reduction % (or residual fold count) vs fold density, marking the sparse/moderate/dense regimes and the feasibility cliff.

- [ ] **Step 4: Run it**

Run: `python research/manuscript_results/ceiling_consolidate.py`
Expected: CSV + both PDFs written.

- [ ] **Step 5: Commit checkpoint**

Stage script + figures + CSV; notify user.

---

## Phase C — Writing (body first, front matter last)

> Each writing task: edit `writing/dvfopt.tex` for that section, run the build command, confirm spec coverage, stage for commit. Keep the existing equations/figures noted as "keep". Delete the giant red `TODO LIST` block (lines ~4–19) only in Task D1.

### Task C1: §3 Background — folding & the discrete Jacobian

**Files:**
- Modify: `writing/dvfopt.tex` (current §"Method"→"Registration and invertibility" block, ~lines 178–258)

**Interfaces:**
- Produces: the `\label{sec:problems_with_discrete_jacobian_determinant}` and central-difference equations that §4 builds on. Keeps Figs `fig:triangle_assignment`, `fig:geometric_primitives`.

- [ ] **Step 1: Trim to 2D-primary**

Keep the registration/invertibility definitions, `eq:jacobian_decomposition`, `eq:2d_jacobian_matrix_det`, `eq:central_differences_2d`, and the two-failure-modes subsection. Keep `eq:3d_jacobian_matrix_det` but reduce surrounding 3D prose to one sentence ("the 3D case extends analogously; we return to it in §8"). Spec §4 row 3.

- [ ] **Step 2: Add the local-vs-global injectivity caveat**

Add one sentence (spec §6/#9): positive Jacobian everywhere is necessary but not globally sufficient for invertibility without boundary conditions; we verify global injectivity empirically in §7 (cite the inverse-consistency result, Task B6).

- [ ] **Step 3: Wire the blindness statistic**

Replace "Part (c)... illustrates an example" with a forward reference to the *quantified* blindness result (Task B2 / Table T5): central differences miss a measurable fraction of real folds, not just a synthetic one.

- [ ] **Step 4: Verify compile + coverage**

Run the build command. Expected: compiles; `fig:triangle_assignment` still referenced; no new undefined refs.

- [ ] **Step 5: Commit checkpoint** — stage `writing/dvfopt.tex`; notify user.

---

### Task C2: §4 Geometric measure (2D triangle)

**Files:**
- Modify: `writing/dvfopt.tex` (§"A geometric measure for fold detection", ~lines 290–340)

**Interfaces:**
- Consumes: §3 central-difference setup.
- Produces: `sec:geometric_measure_for_fold_detection`, `eq:signed_area`, `eq:vertex_signed_areas`. The 3D subsection (current ~342–412) MOVES to §8 in Task C6.

- [ ] **Step 1: Keep the 2D construction**

Retain the quad→two-triangle split, `eq:quad_corners`, `eq:vertex_triangles`, `eq:signed_area`, `eq:vertex_signed_areas`, Figs `fig:quad_setup`/`fig:triangle_assignment`, and the fold-free criterion.

- [ ] **Step 2: Add the exactness statement**

Add a short proposition: restricted to each triangle, φ is affine, so the signed area equals (½)·det of the linear map sending reference vertices to deformed positions — i.e. the per-triangle measure is the *exact* Jacobian determinant of the piecewise-affine interpolant, not an approximation. Spec §3/C1.

- [ ] **Step 3: Add the 2D diagonal-choice note**

Add one short paragraph: the quad admits two diagonal splits; a cell may be fold-free under one and folded under the other, foreshadowing the 3D tetrahedralization-dependence and the geometric ceiling in §8. Spec §4 row 4.

- [ ] **Step 4: Cut the 3D subsection from here**

Remove `\subsubsection{3D fold detection}` and everything through `fig:vertex_tetrahedra` from §4 (it relocates to §8 in C6). Leave a `% moved to Section 8` marker.

- [ ] **Step 5: Verify compile + coverage** — build; ensure no equation labels referenced elsewhere are now dangling (they move with the text in C6; if C6 not yet written, temporarily expect 2 undefined refs and note them for C6). Commit checkpoint.

---

### Task C3: §5 Correction as constrained optimization

**Files:**
- Modify: `writing/dvfopt.tex` (§"Optimization formulation" + §"Objective functions", ~lines 414–495)

**Interfaces:**
- Produces: `eq:opt_problem`, the objective definitions, `sec:objective_functions`.

- [ ] **Step 1: Keep the formulation**

Retain `eq:opt_problem`, the L1/L2/L2,1 objective block, τ=0.01, and the L1 rationale (concentrates corrections). Fix the inconsistency where the summary currently writes `\|\phi-\phi'\|_2` after arguing for L1 — make the chosen objective consistent (L1) throughout.

- [ ] **Step 2: Generalize the constraint to the geometric measure**

Reframe `eq:opt_constraint` from `det(J_φ)>0` (central diff) to the signed-area/volume ≥ τ constraint family from §4 — this is the exact feasible region. Keep a note that the central-diff constraint is the (blind) baseline.

- [ ] **Step 3: Add the locally-injective-maps anchor**

Add one or two sentences (must-add #1): this constraint set is the foldover-free / locally-injective PL-map constraint studied in geometry processing; cite `schuller2013lim`, `du2020lifting`, `garanzha2021foldfree`. Full positioning paragraph lives in §6.

- [ ] **Step 4: Verify compile + coverage** — build; commit checkpoint.

---

### Task C4: §6 Solvers (DVFopt) — taxonomy + champion

**Files:**
- Modify: `writing/dvfopt.tex` (§"The SLSQP Method" + §"Windowed SLSQP", ~lines 497–613)

**Interfaces:**
- Consumes: §5 formulation. Produces: Table T1 (method taxonomy), `tab:jac_results` replacement context.

- [ ] **Step 1: Restructure into the method taxonomy**

Replace the SLSQP-only narrative with the spec §6 taxonomy in four tiers: (a) baselines — full-grid & windowed SLSQP, NMVF; (b) barrier/ALM — m10/m14 (harmonic extension → PHR-ALM → barrier polish); (c) the LP family — LP-direct → SLP → cluster/`auto_slp` champion; (d) decomposition — windowed (Jdet) and cluster (frozen-ring) for scale. Add Table T1 (method / measure / feasibility guarantee / parallelizable / 2D-3D) from `ASSET_INVENTORY.md`.

- [ ] **Step 2: Keep windowed SLSQP mechanics**

Retain the windowed formulation (`eq:iter_objective_function`, frozen-edge constraints, window-growth) as the baseline scaling mechanism, but frame cluster decomposition as its generalization.

- [ ] **Step 3: State the champion claim + LIM positioning**

Write the C2 claim: cluster-LP/`auto_slp` Pareto-dominates barrier/ALM (forward-ref T2/Pareto F-new-2). Add the positioning paragraph (must-add #1): differentiate from geometry-processing untangling — DVF-specific, registration-agnostic, medical-volume scale, the cluster decomposition as the scaling novelty; note SLP is itself the trust-region structure-exploiting solver (per SOLVER_OPTIMIZATION.md's GN finding).

- [ ] **Step 4: Verify compile + coverage** — build; commit checkpoint.

---

### Task C5: §7 Experiments (2D)

**Files:**
- Modify: `writing/dvfopt.tex` (replace the old `tab:jac_results` region, ~lines 577–613, and add the experiments section)

**Interfaces:**
- Consumes: B2 (T5), B3/B4 (T2/T3), B5 (distributions), B6 (inverse-consistency), B7 (F-new-2).
- Produces: `sec:experiments`, Tables T2/T3/T5, Figs `pareto_frontier`, `distribution_boxplots`.

- [ ] **Step 1: Datasets & metrics subsection**

Write datasets (synthetic, B0039 multi-source, public dataset) and metrics: geometric feasibility/min_T/L1/wall + standard `%non-pos-J` and SDlogJ (must-add #4) + downstream Dice/NCC (must-add #3) + inverse-consistency (Exp-9).

- [ ] **Step 2: Benchmark tables**

Insert T2 (methods axis, with distributions from B5 — report mean±std, not single slices) and T3 (sources axis, with downstream metrics). Pull numbers from the B3/B4/B5 CSVs. Add T5 (blindness, B2).

- [ ] **Step 3: Frontier + distributions figures + prose**

Insert F-new-2 (Pareto) and `distribution_boxplots`. Write the speed/accuracy-is-a-dial narrative (spec §6/Exp-7) and the "correction does not degrade registration" result (Dice_after ≈ Dice_before).

- [ ] **Step 4: Diffeomorphic-baseline paragraph**

Address "why not just use SyN/VoxelMorph-diff" in text (spec §6/#10): post-hoc is agnostic, cheaper, and works on *any* field incl. learned ones that fold; cite `dalca2018diffeomorphic`, `AVANTS200826`.

- [ ] **Step 5: Verify compile + coverage** — build; ensure all new figure/table refs resolve. Commit checkpoint.

---

### Task C6: §8 Extension to 3D

**Files:**
- Modify: `writing/dvfopt.tex` (insert the relocated 3D measure from C2 + new ceiling material)

**Interfaces:**
- Consumes: the 3D subsection text removed in C2; B8 (T4, F-new-1, F-new-3).
- Produces: `sec:3d_fold_detection` (relocated), the strict-feasibility + ceiling subsections, Table T4.

- [ ] **Step 1: Reinsert the tet measure**

Paste the 3D fold-detection construction removed in C2 (parallelepiped, Freudenthal–Kuhn, `eq:signed_volume`, `eq:oriented_volume`, Figs `fig:kuhn_decomposition`, `fig:vertex_tetrahedra`) here as the opening of §8. Resolve the C2 dangling refs.

- [ ] **Step 2: 3D tet-LP + B0039 results**

Summarize the 3D tet-LP path and the B0039 strict-feasibility pipeline numbers (from `strict_feasibility_3d/README.md`): Stage-1 2D auto_slp → global M10Tet @0.01/@0.015, density-dependent outcomes.

- [ ] **Step 3: The geometric ceiling + Schönhardt**

Write the capstone (spec §5/#2, §8): cells untangleable under all 4 cube diagonals are deformation-induced non-tetrahedralizable configurations, classically the Schönhardt polyhedron (cite `schonhardt1928`). Insert F-new-1. Report the diagonal-flip recovery (173→94, 46%) and the density-regime table T4 (F-new-3). State this is a hard mathematical ceiling independent of solver.

- [ ] **Step 4: Verify compile + coverage** — build; all 3D refs resolve; no leftover dangling labels from C2. Commit checkpoint.

---

### Task C7: §2 Related work

**Files:**
- Modify: `writing/dvfopt.tex` (§"Restructure Related work", ~lines 114–175)

- [ ] **Step 1: Trim registration survey**

Keep rigid/non-rigid/landmark/learning paragraphs but tighten; remove text that only served the cut Laplacian-augmentation story.

- [ ] **Step 2: Add the geometry-processing/untangling subsection**

New paragraph (must-add #1): locally-injective maps & mesh untangling (`schuller2013lim`, `smith2015bijective`, `du2020lifting`, `garanzha2021foldfree`, `knupp2001untangling`, `toulorge2013robust`) — the home-field prior art for the signed-area/volume constraint — and how this work differs (post-hoc DVF correction, registration-agnostic, scale, ceiling).

- [ ] **Step 3: Add fold-detection/metrics paragraph**

Cover central-difference Jacobian as the standard diagnostic + standard fold metrics (`learn2reg2023`), motivating the exact geometric measure.

- [ ] **Step 4: Rewrite "Our method" paragraph** to match the new C1–C3 framing (drop augmentation language).

- [ ] **Step 5: Verify compile + coverage** — build; commit checkpoint.

---

### Task C8: §9 Discussion / limitations / conclusion

**Files:**
- Modify: `writing/dvfopt.tex` (§"Conclusion" + §"Summary and Future Work", ~lines 616–640)

- [ ] **Step 1: Rewrite conclusion** around C1–C3; delete the interpolation-space and correspondence-augmentation paragraphs entirely.

- [ ] **Step 2: Honest limitations** (spec §9): density-dependent feasibility ceiling (strict 100% on sparse/moderate; 99.94% with unavoidable folds on dense bands), the cluster L1 gap as a speed/accuracy trade, local-vs-global injectivity.

- [ ] **Step 3: Future work**: better trust-region/IPM solver for the dense regime; Steiner-point remeshing to break untangleable cells; extend downstream-task evaluation.

- [ ] **Step 4: Verify compile + coverage** — build; commit checkpoint.

---

### Task C9: §1 Introduction

**Files:**
- Modify: `writing/dvfopt.tex` (§"Restructure Introduction" + §"Main contributions", ~lines 64–112)

- [ ] **Step 1: Rewrite the intro** to lead with the geometric thesis (folding = signed-area/volume sign; discrete Jacobian is blind) and the post-hoc-correction goal; keep `fig:registration-example` as motivation only.

- [ ] **Step 2: Rewrite the contributions list** to exactly C1 (measure), C2 (methods + champion), C3 (benchmark + ceiling). Mention the 3D ceiling as a teased highlight. Delete the old augmentation/interpolation-space contributions.

- [ ] **Step 3: Verify compile + coverage** — build; commit checkpoint.

---

### Task C10: Front matter — title, abstract, keywords

**Files:**
- Modify: `writing/tvcg_manuscript.tex` (title ~line 26, abstract ~lines 50–63, keywords ~lines 65–67)

- [ ] **Step 1: Update title** to the working title (or a refined variant) — geometry-framed.

- [ ] **Step 2: Rewrite the abstract** around C1–C3: discrete-Jacobian blindness → exact triangulated measure → constrained-optimization correction across registration sources → 3D extension reveals a hard geometric (Schönhardt-type) feasibility ceiling. Remove all interpolation-space / augmentation sentences.

- [ ] **Step 3: Update keywords** (drop "correspondence, interpolation"; add "fold detection, local injectivity, mesh untangling, constrained optimization").

- [ ] **Step 4: Verify compile + coverage** — build; commit checkpoint.

---

## Phase D — Integration & final pass

### Task D1: Remove dead content

**Files:**
- Modify: `writing/dvfopt.tex` (TODO list ~lines 4–19; `\begin{comment}` dead blocks), `writing/tvcg_manuscript.tex` (confirm `\input{laplacian}` stays commented).

- [ ] **Step 1: Delete the red TODO list block** and the stale `\begin{comment}...\end{comment}` Laplacian/registration dead text that the new structure supersedes. Keep any commented block still referenced as a deliberate alternative.

- [ ] **Step 2: Confirm Laplacian stays cut** — `\input{laplacian}` remains commented; bibliography still includes `laplacian.bib` only for shared citation keys actually used.

- [ ] **Step 3: Verify compile** — build; commit checkpoint.

---

### Task D2: Full compile & cross-reference audit

**Files:**
- Read: `writing/tvcg_manuscript.tex` build log.

- [ ] **Step 1: Clean build** — run the build command twice; capture warnings.

- [ ] **Step 2: Resolve all undefined refs/citations** — grep the `.log` for `undefined`, `multiply defined`, `Citation ... undefined`. Fix each (missing `\label`, wrong key, figure path typo).

- [ ] **Step 3: Verify every figure/table is referenced** — each of T1–T5 and every figure has at least one `\autoref`/`\ref`. Add references where missing.

- [ ] **Step 4: Commit checkpoint.**

---

### Task D3: Spec-coverage final pass

**Files:**
- Read: spec + `writing/dvfopt.tex`.

- [ ] **Step 1: Walk the spec** — for each of C1–C3, must-adds #1–5, and discretionary #6–9, point to the section/table/figure that delivers it. List gaps.

- [ ] **Step 2: Fill any gap** found, then rebuild.

- [ ] **Step 3: Final commit checkpoint** — stage all; notify user the manuscript pivot is complete and ready for their read-through.

---

## Self-review notes (author of plan)

- **Spec coverage:** C1→C2,C3 / Tasks C2,C4,C5,C6; must-adds #1 (A1,C3,C4,C7), #2 (B8,C6), #3 (B3,C5), #4 (B3,C5), #5 (B4,C5); discretionary #6 (B2,C5), #7 (B7,C5), #8 (B5,C5), #9 (B6,C5/C3). All covered.
- **Dependency order:** Phase B precedes the §7/§8 writing that cites it; §4's 3D-cut (C2) is resolved by §8's reinsert (C6) — flagged explicitly so the dangling refs are expected and closed.
- **Backend risk:** B1 gates B3/B4 scope; §9 fallback (Laplacian + VoxelMorph) is encoded.
- **Commit rule:** every task ends at a stage-and-notify checkpoint; no auto-commit (project rule).
