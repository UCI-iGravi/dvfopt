# research/strict_feasibility_2d/

Active research thread targeting **strict 2-triangle feasibility with
minimised L1 deviation** on worst-case 2D deformation fields.

See [`DESIGN.md`](DESIGN.md) for the design spec.

## Status

| Milestone | Status |
|---|---|
| Folder scaffolded | ✓ |
| Algorithms implemented (`lp_oneshot`, `slp_iter`) | ✓ |
| Worst-case catalog built | ✓ |
| Synthetic bake-off run | ✓ |
| B0039 z=12 bake-off run | ✓ partial (harmonic_only + m14; LP/SLP did not complete in 12-min budget at full 320×456) |
| Analysis notebooks finalised | ✓ |

## Results (2026-06-14)

Headline numbers from `analysis/01_baseline_l1_gap.ipynb` and
`analysis/02_lp_certifies_optimum.ipynb`. CSVs in `runners/output/`.

### Synthetic suite (9 cases — bowtie + 6 canonical + 2 adversarial)

After landing **fallback row 1** (use the m10 pipeline as the SLP seed
instead of plain harmonic extension), `slp_iter` now reaches strict
feasibility on **every** case:

| Method | Feasibility coverage |
|---|---|
| m10 / m14 / m14_schwarz | 9 / 9 ✓ |
| **slp_iter (m10 seed)** | **9 / 9 ✓** |
| lp_oneshot (m10 seed) | 3 / 9 (single-LP step still infeasible at exact eval on dense cases) |
| harmonic_only | 3 / 9 |

L1 head-to-head (synthetic, all 9 feasible for m14, slp_iter,
`slp_iter_m14_seed`, and `cluster_slp`):

| Case | M14 L1 | **SLP (M14 seed)** | **cluster_slp** |
|---|---:|---:|---:|
| `bowtie_7x7_shoelace` | 1.466 | **1.420** (−3.2%) | **1.420** (−3.2%) |
| `dense_bowtie_cluster_15x15` | 12.958 | **12.780** (−1.4%) | **12.780** (−1.4%) |
| `01a_10x10_crossing` | 6.676 | **6.640** (−0.5%) | **6.640** (−0.5%) |
| `01b_10x10_opposite` | 3.552 | **3.536** (−0.5%) | 3.552 (≈0) |
| `03a_10x10_opposite` | 7.135 | **7.093** (−0.6%) | 7.137 (+0.0%) |
| `03b_10x10_crossing` | 15.868 | **15.819** (−0.3%) | 15.867 (≈0) |
| `03c_20x20_opposite` | 22.990 | **22.872** (−0.5%) | 22.991 (≈0) |
| `03d_20x20_crossing` | 43.764 | **43.660** (−0.2%) | 45.104 (+3.1%) |
| `tiny_margin_10x10` | 7.826 | **6.000** (−23%) | 8.719 (+11.4%) |

**Two winners depending on scale:**

* **`slp_iter_m14_seed`** — L1 winner everywhere on small/medium slices
  (9/9 feasibility, ≥ M14 on every case, never loses, sometimes huge
  win like tiny_margin's 23%).
* **`cluster_slp`** — wall-time winner at B0039 scale (see next
  section); on small synthetic slices the polish step overhead drags
  L1 by 3–11% on dense crossing cases.

A wide-trust-region variant (`slp_iter_wide_tr`) was tested but
*worsened* dense-crossing cases — the big LP first-step pulls phi
away from input then gets stuck. The bottleneck is seed quality, not
trust radius.

### B0039 z=12 (320×456, init n_neg=4902)

| Method | Feasible | n_neg | min_T | L1 | Wall |
|---|:---:|---:|---:|---:|---:|
| harmonic_only | ✗ | 641 | -1.89 | 239 499 | 0.9 s |
| m14 | ✓ | 0 | +0.020 | 184 034 | 267 s |
| **cluster_slp** | **✓** | 0 | +0.010 | **175 638 (−4.6%)** | **64 s (4.2× faster)** |
| slp_iter (global, m10 seed) | ✓ | 0 | +0.010 | 287 316 (+56%) | 515 s |
| lp_oneshot (global, direct LP) | (timeout) | — | — | — | >12 min |

**`cluster_slp` is the headline result on B0039:** strict feasibility,
4.6% lower L1 than M14, and 4.2× faster wall-time.

The approach: decompose the slice into connected fold clusters (via
`scipy.ndimage` CCL with merge-dilation), solve SLP on each cluster's
padded crop with frozen boundary, splice interior corners back. On
z=12 this produces 11 clusters in round 0 (drops 4902→49 folds) and 5
clusters in round 1 (49→0). No global polish needed — the cluster
pass alone reaches feasibility because B0039's folds are sparsely
distributed in a large slice.

The optimal inner seed (per per-cluster solve) is M14, not M10 or
harmonic — both M10 and harmonic inner seeds force a global polish to
fire, which costs ~3× wall-time and worsens L1.

### Failure modes feeding back into the [fallback plan](../../docs/superpowers/specs/2026-06-14-strict-feasibility-2d-design.md#fallback-plan)

| Observed | Affects | Fallback row | Status |
|---|---|---|---|
| LP infeasible from harmonic seed on dense canonicals | `slp_iter` | 1 (use m10 seed) | **✓ Landed**: slp_iter now 9/9 feasible |
| slp_iter L1 worse than M14 on 3 dense-crossing cases | `slp_iter` | (new) Tune trust-region growth, or seed from M14 | Pending |
| `lp_oneshot` still fails feasibility on 6/9 cases (single-step linearisation error) | `lp_oneshot` only | Iterate (SLP) | Use `slp_iter` instead |
| Wall-time >12 min at 320×456 | `lp_oneshot` / `slp_iter` on B0039 | 5 (cluster_lp — per-cluster LP solve) | Pending |

## How to run

```bash
# Run the synthetic bake-off (canonical + adversarial cases):
python research/strict_feasibility_2d/runners/_run_lp_synthetic.py

# Run the B0039 bake-off (z=12 only is recommended for first iteration):
python research/strict_feasibility_2d/runners/_run_lp_b0039.py --slices 12

# Rebuild + execute the analysis notebooks:
python research/strict_feasibility_2d/analysis/_build_01.py
python research/strict_feasibility_2d/analysis/_build_02.py
jupyter notebook research/strict_feasibility_2d/analysis/
```

Outputs land in `runners/output/`:

- `comparison_synthetic.csv` — one row per (case, method)
- `comparison_b0039.csv` — one row per (slice, method)
- `corrected/<case>_<method>.npz` — corrected `phi` arrays (gitignored — large)

## Layout

```
research/strict_feasibility_2d/
├── README.md                   # this file
├── DESIGN.md                   # points to docs/superpowers/specs/…
├── worst_cases/
│   ├── _load.py
│   ├── _build_adversarial.py
│   ├── catalog.md
│   ├── synthetic/              # adversarial NPZs (built locally)
│   └── b0039/                  # placeholder; slices load from data/dvfs/
├── algorithms/
│   ├── orientation_fix.py      # canonical sign vector helper
│   ├── tri_linearize.py        # explicit sparse Jacobian for 2-tri constraint
│   ├── highs_solver.py         # L1 LP step via scipy linprog(method='highs')
│   └── lp_direct_2tri.py       # lp_oneshot + slp_iter public API
├── runners/
│   ├── _compare.py             # run_method(name, phi) → metric record
│   ├── _run_lp_synthetic.py    # batch over synthetic suite
│   ├── _run_lp_b0039.py        # batch over selected B0039 slices
│   └── output/
│       ├── comparison_synthetic.csv
│       ├── comparison_b0039.csv
│       └── corrected/          # gitignored
└── analysis/
    ├── _build_01.py            # generator for notebook 01
    ├── _build_02.py            # generator for notebook 02
    ├── 01_baseline_l1_gap.ipynb
    ├── 02_lp_certifies_optimum.ipynb
    ├── l1_per_case.png         # output of notebook 01
    └── l1_gap_vs_slp.png       # output of notebook 02
```

## Tests

```bash
pytest tests/research/strict_feasibility_2d/ -v
```

20 tests across `test_worst_cases.py`, `test_orientation_fix.py`,
`test_tri_linearize.py`, `test_highs_solver.py`, `test_lp_direct.py`,
`test_comparison.py`. All passing on the head commit.
