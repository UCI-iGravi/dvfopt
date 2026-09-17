# Canonical 2D I-SLSQP Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the canonical, reproducible 2D results for the paper: the I-SLSQP windowed engine (and, on the small set, the full method taxonomy) run on every 2D benchmark source in the repo — four fold-origin mechanisms across eight registration tools, the seven-brain cohort with its correspondences, the ANTs controls, the hard crops, the synthetic cases — with the run-dir outputs the code already defines, every metric the paper needs, and the corrected DVFs saved.

**Architecture:** One driver `benchmarks/canonical_2d.py` builds a case registry from the four on-disk sources, runs a named config through `dvfopt.correct_dvf` (or a registry strategy label), computes one metric block on `(input, output)` — the existing cohort-benchmark row schema plus the paper's additions — and writes the standard run directory (`results.csv`, `summary.json`, `figures/`, `report.html`) through the shared writer in `benchmarks/cohort_benchmark.py`, saving each corrected field with `dvfopt.io.fields.save_dvf` under `data/dvfs/results/` (gitignored, the README's "paper deliverables" home) with a manifest, and copying the small tracked artefacts (CSV, JSON, PNG) to a new tracked directory `docs/paper/results/2d_canonical/`. No engine change; no routing change.

**Tech Stack:** numpy / scipy, the dvfopt certificate and metric kernels, matplotlib via `dvfopt.viz.theme.apply_theme('paper')`, pytest for the metric block and the registry.

**Spec:** `docs/superpowers/notes/2026-09-11-2d-canonical-benchmark-handoff.md` (§2 is the pre-registered protocol, §3 the sources and their paths, §4 the metric definitions); the paper's contribution C3 in `docs/superpowers/specs/2026-06-25-tvcg-folding-correction-pivot-design.md` ("head-to-head across methods and registration sources on a common axis — feasibility, L1 deviation, wall-time — with distributional statistics; the speed/accuracy frontier"; its must-add #4: report % non-positive-Jacobian voxels and SDlogJ alongside the simplicial measure).

## Global Constraints

- **`dvfopt/` is untouched.** The engine's defaults at the pinned commit ARE the method; no per-case tuning; the tuning set (the three hard crops) is labelled as such in every table.
- **The protocol is pre-registered in the ledger before the first run** (handoff §2) and any later deviation is a separate, labelled row.
- **Nothing is dropped:** a failed or capped run is a row with its failure recorded; `feasible=False` rows stay in every aggregate.
- Every run records provenance: git commit, `dvfopt.__version__`, the config dict, python + OS, `n_workers`, and the box's load at start.
- Walls: throughput runs may use `n_workers` (2–4), but the paper's per-case wall column comes from a SERIAL pass on an idle box over a stated subset; both are reported and labelled.
- Python `>=3.10`; ruff 0.16.3; `mypy` clean; tests under `tests/` use synthetic fields only (no gitignored data).
- Long runs are the controller's, in the background from the main checkout root with `PYTHONPATH=<detached snapshot worktree>`; subagents never start them.

---

### Task 1: The driver — case registry, metric block, run-dir + DVF writer

**Files:**
- Create: `benchmarks/canonical_2d.py`
- Reuse (import, do not edit): `benchmarks/cohort_benchmark.py` (the shared writer of `results.csv` / `summary.json` / `report.html` — read its `_write_run_outputs`-style function near line 767 and the row schema near line 296; the `_neg_volume` / `_n_clusters` helpers), `benchmarks/benchmark_utils.py` (`load_cohort_field`, `load_cohort_section`, `load_cohort_correspondences`, `list_cohort`, `save_results_csv`, `save_summary_json`, `plot_jdet_histograms`, `plot_jac_heatmaps`), `benchmarks/correspondence_analysis.py` (`analyze_slice`, `slice_correspondences`), `dvfopt.metrics` (`fold_stats`, `injectivity_stats`), `dvfopt.jacobian.numpy_jdet.jacobian_det2D`, `dvfopt.jacobian.injectivity_radius.cell_min_jdet_2d`, `dvfopt.core.primitives.tri.tri_areas_flat`, `dvfopt.io.fields.save_dvf / load_dvf`, `dvfopt.viz.theme.apply_theme`.
- Test: `tests/test_canonical_2d.py`

**Interfaces:**
- `SOURCES = ('origins', 'cohort', 'ants', 'crops', 'synthetic')`; `cases(source, sample='canonical') -> list[Case]` with `Case(id, source, mechanism, tool, params, path_or_builder, shape)` — origins from `data/dvfs/origins/manifest.json` (27 cases; `mechanism`, `tool`, `source` fields), cohort from `list_cohort()` × the handoff's slice sample (`z in range(0, D, 48)` plus the named hard slices), ANTs from the cohort's `ants_warp_0.nii.gz` at the same z (via `load_dvf`), crops from `data/dvfs/crops/*.npy` (converted `(2,H,W)` → `(3,1,H,W)` with `dz=0`), synthetic from `dvfopt.testdata.SYNTHETIC_CASES` (+ `data/dvfs/canonical_2tri_2d/*.npz` if present).
- `CONFIGS = {'isqp_none': dict(constraint='bilinear', strategy='isqp_windowed', objective='none'), 'isqp_l2': dict(constraint='bilinear', strategy='isqp_windowed', objective='l2'), 'auto': dict(constraint='bilinear', strategy='auto', objective='auto'), 'slp': dict(constraint='simplex_standard', strategy='slp', objective='l1'), 'barrier': dict(constraint='simplex_standard', strategy='barrier', objective='l2'), 'm14': dict(constraint='simplex_standard', strategy='m14', objective='l2'), 'slsqp_windowed': dict(constraint='jdet', strategy='slsqp_windowed', objective='l2')}` — every value passes straight to `correct_dvf(phi, threshold=0.01, record_history=True, **cfg)`.
- `metrics(phi_in, phi_out, res, threshold=0.01) -> dict` — the handoff §4 block (the existing schema keys keep their names).
- CLI: `--source S [--config C ...] [--sample canonical|smoke] [--n-workers N] [--serial-timing] [--run-dir D] [--figures] [--table]`; each run writes `<run-dir>/results.csv`, `summary.json`, `report.html`, `figures/`, and `<run-dir>/manifest.json` listing every saved DVF `data/dvfs/results/<run-name>/<source>/<case>__<config>.npz`.

- [ ] **Step 1: Tests first** (`tests/test_canonical_2d.py`): the metric block on `tests/conftest.planted_fold` → the four certificates agree with `fold_stats` on the same values, `frac_nonpos_jdet` and `sdlogj` match a hand computation on a tiny field, `moved_frac` / `l1_move` / `l2_move` / `max_move` match numpy; the registry's synthetic source enumerates `SYNTHETIC_CASES`; a smoke run of `--source synthetic --config isqp_none --sample smoke --run-dir <tmp>` writes `results.csv`, `summary.json`, `manifest.json` and one `.npz` per case (needs `osqp`; skip otherwise).
- [ ] **Step 2: Implement** the registry, the configs, the metric block, the runner (serial and `ProcessPoolExecutor` with `pin_worker_threads`, spawn-safe module-level worker — copy `benchmarks/cohort_benchmark.py`'s worker pattern), the writers, `--figures` (fold counts before/after per source on a log axis; the wall-vs-move frontier per config on the origins set; jdet histograms before/after for a named case; `apply_theme('paper')`), `--table` (markdown per source with medians and IQR).
- [ ] **Step 3: Run the tests** (foreground, 900000 ms): `pytest tests/test_canonical_2d.py -q -p no:cacheprovider`.
- [ ] **Step 4: Lint, mypy, commit:** `benchmarks: canonical 2D benchmark driver — registry over origins/cohort/ANTs/crops/synthetic, the paper metric block, run-dir + DVF outputs`.

---

### Task 2 (controller): pre-registration and the runs

- [ ] Paste the handoff's §2 protocol into the ledger with the date and the pinned commit.
- [ ] Chains (each from a detached snapshot worktree; log under `benchmarks/output/canonical_2d/chain_*.log`; check the box is idle first):
  1. **origins × all seven configs** (27 cases, single slices ≤ 320×456): `--source origins --config isqp_none isqp_l2 auto slp barrier m14 slsqp_windowed --n-workers 4` (throughput), then `--serial-timing` for `isqp_none` and `isqp_l2` on all 27 (the wall column).
  2. **cohort sample × isqp_none, isqp_l2** (7 brains × 11 slices + the named hard slices ≈ 90): `--n-workers 4`; the serial-timing pass on the 22 z=0/240 slices.
  3. **ANTs controls × isqp_none** (the same slices): expected 0 → 0 at ~0 s.
  4. **crops × all configs** (the tuning set, labelled).
  5. **synthetic × all configs**.
- [ ] After each chain: `--table`, `--figures`; ledger the medians and any `feasible=False` row with its exits.

---

### Task 3: The tracked results, figures, findings, PR

- [ ] Docs subagent: create `docs/paper/results/2d_canonical/README.md` (the protocol, the pinned commit, the sources, the metric definitions, how to regenerate), copy `results.csv` / `summary.json` / `figures/*.png` per source there (small files only; DVFs stay under `data/dvfs/results/` with the manifest), add "§12 Canonical 2D results" to `docs/superpowers/notes/zero-folds-campaign-findings.md` with the per-source tables (certification rate, median wall, median L1/L2 move, SDlogJ before/after, non-positive fraction before/after, damage), the frontier figure, and the explicit scope (the origins set is single slices; the cohort is a slice sample; the full B0039 528/528 certification is cited from `benchmarks/output/b0039_ext_full_v2/`), a `CHANGELOG.md` entry, `CLAUDE.md`'s benchmarks bullet gains the driver and the results directory, `data/dvfs/README.md`'s `results/` row names the run.
- [ ] Controller: full suite; ruff / mypy; push; PR to `UCI-iGravi/dvfopt` (body: the protocol, the per-source summary table, the provenance); squash-merge on all-green; sync main; remove the worktrees; ledger to the main checkout's `.superpowers/sdd/`; memory (`paper-2d-canonical-results.md` + the MEMORY.md line).
