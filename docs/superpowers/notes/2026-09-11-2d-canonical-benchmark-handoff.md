# Handoff — the canonical 2D I-SLSQP benchmark for the paper

Written 2026-09-11. Purpose: a fresh session produces the paper's canonical 2D results — the I-SLSQP windowed
engine on every 2D benchmark source in the repo, spanning fold-origin mechanisms and registration tools — with
the outputs the code already defines, all the metrics the paper needs, and the corrected DVFs saved. Read this,
then the plan `docs/superpowers/plans/2026-09-11-2d-canonical-benchmark.md`.

## 1. What "canonical" means here

One pinned engine (the defaults at a named commit of `main`), one pre-registered protocol, every source, nothing
dropped, everything reproducible from the tracked CSV/JSON + the gitignored DVFs + the driver. The headline
rows are the engine's two configs — the measured robust recipe `bilinear + isqp_windowed + none` and the
in-solve-L2 default `bilinear + isqp_windowed + l2` — and, on the small sources, the full method taxonomy
(`slp`, `barrier`, `m14`, `slsqp_windowed`, `auto`) so the paper's "across methods AND across registration
sources on a common axis" claim (contribution C3 of the TVCG pivot spec) is one table.

## 2. The protocol (pre-register verbatim in the ledger before the first run)

- **Engine:** `dvfopt.correct_dvf(phi, threshold=0.01, record_history=True, **config)` at the pinned commit; no
  per-case knobs; `err_tol` 1e-5 (the package default). `constraint='bilinear'` is the 4-triangle cell certificate
  (both diagonals); the simplex families' `'simplex_standard'` row is the fixed-diagonal 2-triangle one.
- **Configs:** `isqp_none` (bilinear, isqp_windowed, none), `isqp_l2` (bilinear, isqp_windowed, l2), `auto`
  (bilinear, auto, auto — what `correct_dvf` picks by itself), `slp` (simplex_standard, slp, l1),
  `barrier` (simplex_standard, barrier, l2), `m14` (simplex_standard, m14, l2), `slsqp_windowed` (jdet,
  slsqp_windowed, l2). The two `isqp_*` rows run on every source; the rest on origins, crops, synthetic.
- **Sources and samples:** see §3. The three hard crops are the engine's TUNING set — every table says so.
- **Runs:** one run per (case, config); throughput passes with `n_workers=4`; the per-case wall column from a
  serial pass on an idle box over the stated subset (origins: all 27; cohort: the z=0 and z=240 slices of every
  brain). Check the box's load before every chain (an unrelated job once loaded it 50–94 % for hours and
  doubled every wall); record the load in `summary.json`.
- **Nothing dropped:** a run that fails or exceeds its cap (`time_budget_s` is NOT set — the engine runs to its
  own termination; the chain's per-run cap is 2 h, `hit_cap` recorded) is a row with `feasible=False` and its
  exit reasons; every aggregate includes it.
- **Reported statistics:** per source and config — certification rate (fraction of cases at 0 folds under
  each gauge), median / IQR of wall, L1 move, L2 move, SDlogJ before→after, non-positive-Jacobian fraction
  before→after, damage (must be 0 on every windowed row — a nonzero value is a bug report, not a result), moved
  fraction; per-mechanism panels for the origins set.

## 3. Sources on disk (all gitignored payloads; `data/dvfs/README.md` is the tracked map)

| source | what | where | count | notes |
|---|---|---|---|---|
| `origins` | fields generated per fold-origin mechanism by the `dvf_origins` harness | `data/dvfs/origins/<mechanism>/<case>.npy` + `.json`, index `data/dvfs/origins/manifest.json` (27 entries: `file, mechanism, source, tool, shape, params`) | 27 | m1 interpolation ×5 (Laplacian of sparse / corrupted correspondences, incl. the real B0039 z264 slice), m2 dense optimization ×8 (SimpleITK demons weak/smooth, B-spline FFD fine/coarse, TV-L1, ILK; synthetic + the brain pair), m3 learned ×10 (VoxelMorph / TransMorph, direct and diffeomorphic, on ellipses and on the cohort; two proxies), m4 diffeomorphic ×4 (discretized SVF ×3 + the real ANTs B0039 z264 slice). Shapes `(3,1,H,W)`: 192², 160×264, 320×456. `python -m dvf_origins list` prints the registry; `generate` rebuilds (m3 real networks need `.venv-torch` — the fields are already on disk, do not regenerate). |
| `cohort` | 7 real brains' Laplacian fields with correspondences | `data/dvfs/cohort/<brain>/laplacian_exterior/laplacian_deformation_field.npz` (key `arr`, `(3,528,320,456)` float32), `fpoints.npz` / `mpoints.npz` | B0032, B0039, B0049, B0053, B0200, B0213, B0304 | slice sample: `z in range(0, 528, 48)` (11 per brain) + the named hard slices: B0039 z1, z2, z11, z16, z264; B0032 z1; B0304 z128, z181 (the cohort-sweep plateau slices, CLAUDE.md). Loaders: `benchmarks/benchmark_utils.load_cohort_section(brain, z)` → `(3,1,H,W)`, `load_cohort_correspondences(brain)` → `(mp, fp)`; `benchmarks/correspondence_analysis.slice_correspondences(mp, fp, z)` + `analyze_slice(sec_init, sec_out, mp_slice, fp_slice)` → residual before/after. NB `cohort_dir()` resolves relative to the module's own `__file__`, so a worktree needs a junction `data/dvfs/cohort` → the main checkout's (as #122 did). **Check `list_cohort()` returns 14 pairs before planning a chain**: on 2026-09-11 `data/dvfs/cohort/` was found EMPTY (an empty directory shadows the `brain25_cohort_corrected` fallback in `cohort_dir()`, which is itself a junction to it) and was restored by copying, per brain × variant, `03_laplacian_refinement/parameters/laplacian_deformation_field.npz`, `02_nonlinear/parameters/fwd_transforms/ants_warp_0.nii.gz` and `03_laplacian_refinement/parameters/boundary_conditions/{fpoints,mpoints}.npz` from `C:/Users/Andy/Documents/GitHub/UCI-XuLab/UCI-XuLab-RegTools/output/brain25_cohort_corrected/<brain>/<variant>/` into the flattened `data/dvfs/cohort/<brain>/<variant>/` (~17 GB, gitignored). |
| `ants` | the cohort's ANTs warps (in-plane fold-free per CLAUDE.md) | `data/dvfs/cohort/<brain>/laplacian_exterior/ants_warp_0.nii.gz` via `dvfopt.io.fields.load_dvf` | 7 | the "already injective" control: expect 0 → 0, damage 0, near-zero wall; the same slice sample. |
| `crops` | the three hard B0039 crops (the tuning set) | `data/dvfs/crops/{z0_sliver (2,53,52), z0_cluster (2,35,42), z16_twist (2,50,50)}.npy` (`[dy,dx]`; wrap to `(3,1,H,W)` with `dz=0`) | 3 | known result: `bilinear + isqp_windowed + none` → 0/0/0 folds, damage 0, 32 / 22 / 106 s (CLAUDE.md). `benchmarks/make_hard_crops.py` recuts. |
| `synthetic` | controlled cases | `dvfopt.testdata.SYNTHETIC_CASES` (+ `make_random_dvf`, `make_patch_folded_dvf`), `data/dvfs/canonical_2tri_2d/*.npz`, `data/dvfs/testcases/*.npy` | a dozen | fixed seeds; state them. |
| (cited, not rerun) | the full-resolution B0039 528-slice certification | `benchmarks/output/b0039_ext_full_v2/` (528/528 at 0 simplex / bilinear / finite folds, damage 0, 599,313 → 0) and the cohort sweep `benchmarks/output/cohort_sweep_2d/` (139/146 slices at 0 folds, median 246 s) | — | cite as the full-volume rows; do not spend the session re-running 528 slices. |
| (not on disk) | a public dataset (OASIS / Learn2Reg / LPBA40) | — | — | the pivot spec's must-add #5; out of scope unless someone stages it — say so in the findings. |

## 4. Metric definitions (one block for every `(input, output, result)`; the existing schema keys keep their names)

- From `benchmarks/cohort_benchmark.py`'s row (keep): `n_neg_init`, `n_neg_final`, `neg_vol_init/final`,
  `n_clusters_init/final`, `min_jdet_init/final`, `l2_err`, `time_s` — all on the central-difference Jdet
  (`dvfopt.jacobian.numpy_jdet.jacobian_det2D`) at threshold 0.01.
- **Certificates, before and after, each at threshold 0.01 AND at 0:** `simplex_standard` (2 triangles per cell,
  `dvfopt.core.primitives.tri.tri_areas_flat`), `bilinear` (4 per cell, `dvfopt.jacobian.injectivity_radius.cell_min_jdet_2d`),
  `finite` (forward-difference, one triangle), central `jdet` — counts via `dvfopt.metrics.fold_stats(values, threshold)`
  (`n_neg` = `<= 0`, `n_below` = `< threshold − err_tol`, `min_val`). `certified = (bilinear n_below == 0)` is the
  headline; report the others beside it.
- **Registration-standard metrics** (pivot spec #4): `frac_nonpos_jdet` = mean(`jdet <= 0`) over all pixels;
  `sdlogj` = std(log(clip(`jdet`, 1e-3, None))) over all pixels — state this clipping convention in the README
  (Learn2Reg computes SDlogJ on the foreground; we have no masks, so whole-slice) — both before and after.
- **Engine accounting** (from `res.info`, windowed rows only, `-1` elsewhere): `damage` (`res.info.extras['damage']`,
  must be 0), `n_windows`, `giant_regions`, `mop_cleared`, `rounds` (count phases named `round*`), `sqp_iters`
  (sum `n_iter` over phases NOT named `giant*` — a `giant` entry is nested in its `round` entry and `total_iter`
  double-counts it), `exits` if available, `feasible` (`res.feasible`).
- **Move and locality:** `moved_frac` (fraction of pixels with any channel changed by > 1e-9), `l1_move`,
  `l2_move`, `max_move`, `mean_move_moved` (mean over moved pixels).
- **Correspondence fidelity (cohort only):** `analyze_slice`'s residual statistics before and after (median /
  MAD of the registration residual at the prescribed correspondences; the outlier flags) — the paper's "does the
  correction respect the registration's own landmarks" number.
- **Injectivity diagnostics (optional, cheap):** `dvfopt.metrics.injectivity_stats(phi)` before / after (the IFT
  radius estimate — labelled an estimate, never a certificate).
- **Provenance per run:** git commit, `dvfopt.__version__`, config, `n_workers`, python, OS, box load at start,
  `time_budget_s=None`, `hit_cap`.

## 5. Outputs

- Per run directory (the existing convention, via the shared writer in `benchmarks/cohort_benchmark.py`):
  `results.csv` (one row per case × config), `summary.json` (aggregates + provenance), `report.html`
  (self-contained), `figures/` (fold counts before/after per source on a log axis; the wall-vs-L1/L2-move frontier
  per config on the origins set; jdet histograms before/after for named cases; per-mechanism panels;
  `dvfopt.viz.theme.apply_theme('paper')`), `manifest.json` (every saved DVF with its case, config, shape, sha256).
- Corrected DVFs: `data/dvfs/results/2d_canonical_<YYYYMMDD>/<source>/<case>__<config>.npz` via
  `dvfopt.io.fields.save_dvf` (gitignored; the README's "paper deliverables" row), input copies NOT duplicated
  (the manifest points at the source path).
- Tracked copy for the paper: `docs/paper/results/2d_canonical/` — `README.md` (protocol, pinned commit, metric
  definitions, regeneration command), the per-source `results.csv` / `summary.json`, `figures/*.png`. Small files
  only. (`benchmarks/results/` and `benchmarks/output/` are gitignored — do not put the canonical numbers there.)
- The findings note: `docs/superpowers/notes/zero-folds-campaign-findings.md` §12; a CHANGELOG entry; CLAUDE.md's
  benchmarks bullet; memory.

## 6. Environment and process (standing rules from the last sessions)

- Python: `C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing/.venv/Scripts/python.exe` (3.12,
  osqp + clarabel installed; it is a launcher that execs the Anaconda base interpreter). Run chains from the main
  checkout root with `PYTHONPATH=<detached snapshot worktree>` so the gitignored data resolves and the code is
  pinned. `pytest` in the venv; `-n 4` can oversubscribe (some tests spawn their own pools) — the full suite is
  the controller's, and a summary line containing `failed` is a failure regardless of the pipeline's exit code.
- Worktree per branch (`../dvfopt-canonical-2d`, branch `feat/2d-canonical-benchmark`); plan → subagent-driven
  development with task reviews; measurement chains from a DETACHED snapshot worktree; kill by PID, never by
  command-line pattern; never nudge an agent after taking over its task (stop it); PR to `UCI-iGravi/dvfopt`
  (never the `heemmanshuu` fork); squash-merge on all-green; memory dir
  `C:/Users/Andy/.claude/projects/c--Users-Andy-Documents-GitHub-UCI-iGravi-deformation-field-processing/memory/`;
  follow the session's system-provided commit/PR attribution trailer.
- Compute budget (estimates): origins 27 × 7 configs — the wallbreakers seconds-minutes, `isqp_*` minutes; the
  320×456 slices are the long ones (raw z16-class slices run 200–600 s under `isqp_none`). Cohort ~90 slices ×
  2 configs at a median of ~4 min → ~12 h serial, ~3–4 h at 4 workers. Run the cohort chain overnight; the
  serial-timing subsets are the only walls the paper quotes.

## 7. Session prompt (copy-paste)

```
Read docs/superpowers/notes/2026-09-11-2d-canonical-benchmark-handoff.md and then
docs/superpowers/plans/2026-09-11-2d-canonical-benchmark.md in this repo. Execute the plan autonomously via
subagents in an isolated worktree (../dvfopt-canonical-2d, branch feat/2d-canonical-benchmark): first the
driver with its tests, then the pre-registered runs from a detached snapshot worktree (pin the commit in
every summary.json; check the box is idle before each chain; nothing dropped — failed or capped runs stay as
rows), then the tracked results directory docs/paper/results/2d_canonical/, the findings section, CHANGELOG,
memory, and one PR to UCI-iGravi/dvfopt squash-merged on all-green. The engine is not changed and not tuned
per case; the three hard crops are labelled as the tuning set in every table. Save every corrected DVF under
data/dvfs/results/2d_canonical_<date>/ with a manifest, and every metric in handoff §4 — the existing
cohort-benchmark row schema plus the four certificates at both thresholds, SDlogJ and the non-positive-Jacobian
fraction before/after, damage (must be 0 on every windowed row), moved fraction, L1/L2/max move, SQP iterations
excluding nested giant phases, and the correspondence residual on cohort slices. Report per-source medians and
IQR, certification rates, and the wall-vs-move frontier on the origins set. End with a report listing every
ruling you made with its cost if wrong.
```
