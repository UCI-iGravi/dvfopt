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

LP/SLP **wins on L1 over M14 on every case where they reach feasibility:**

| Case | M14 L1 | LP-oneshot L1 | LP wins by |
|---|---:|---:|---:|
| `bowtie_7x7_shoelace` | 1.466 | **1.420** | 3% |
| `dense_bowtie_cluster_15x15` | 12.958 | **12.780** | 1.4% |
| `tiny_margin_10x10` | 7.826 | **6.000** | **23%** |

LP/SLP **fail strict feasibility** on the canonical dense fold cases
(`03b`/`03c`/`03d`/`01a`/`01b`/`03a`) — linearisation error from the
harmonic seed is too large when displacements are large (~1 cell). M14
still wins by construction on those.

### B0039 z=12 (320×456, init n_neg=4902)

| Method | Feasible | n_neg | min_T | L1 | Wall |
|---|:---:|---:|---:|---:|---:|
| harmonic_only | ✗ | 641 | -1.89 | 239 499 | 0.9 s |
| **m14** | **✓** | 0 | +0.020 | **174 131** | 379 s |
| lp_oneshot | (timeout) | — | — | — | >12 min |
| slp_iter | (timeout) | — | — | — | >12 min |

LP/SLP at full B0039 scale need per-fold-cluster decomposition (spec
fallback row 5: `cluster_lp`) — the direct ~290k-variable solve is
too slow for practical use.

### Failure modes feeding back into the [fallback plan](../../docs/superpowers/specs/2026-06-14-strict-feasibility-2d-design.md#fallback-plan)

| Observed | Affects | Fallback row |
|---|---|---|
| LP infeasible from harmonic seed on dense 10×10/20×20 canonicals | `lp_oneshot` / `slp_iter` | 1 (replace harmonic seed with m10 seed) |
| `slp_iter` returns the seed when no LP step is exact-feasible | `slp_iter` | 1 + improve "best iterate" tracking |
| Wall-time >12 min at 320×456 | `lp_oneshot` / `slp_iter` on B0039 | 5 (cluster_lp — per-cluster LP solve) |

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
