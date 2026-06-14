# research/strict_feasibility_2d/

Active research thread targeting **strict 2-triangle feasibility with
minimised L1 deviation** on worst-case 2D deformation fields.

See [`DESIGN.md`](DESIGN.md) for the design spec.

## Status

| Milestone | Status |
|---|---|
| Folder scaffolded | ☑ |
| Algorithms implemented (`lp_oneshot`, `slp_iter`) | ☐ |
| Worst-case catalog built | ☐ |
| Synthetic bake-off run | ☐ |
| B0039 z=12 bake-off run | ☐ |
| Analysis notebooks finalised | ☐ |

## How to run

```bash
# Run the synthetic bake-off (canonical + adversarial cases):
python research/strict_feasibility_2d/runners/_run_lp_synthetic.py

# Run the B0039 bake-off (z=12 + selected slices):
python research/strict_feasibility_2d/runners/_run_lp_b0039.py

# Then open the analysis notebooks:
#   research/strict_feasibility_2d/analysis/01_baseline_l1_gap.ipynb
#   research/strict_feasibility_2d/analysis/02_lp_certifies_optimum.ipynb
```

Outputs land in `runners/output/`:
- `comparison.csv` — one row per (case, method)
- `corrected/<case>_<method>.npz` — corrected `phi` arrays
