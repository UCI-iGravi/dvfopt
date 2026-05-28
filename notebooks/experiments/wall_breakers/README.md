# Wall-breaker experiments

A test suite for finding a method that achieves **100% 2-triangle feasibility**
(`tri_min >= 0.01`, no folded cells) on the 6 dense slices of
`data/corrected_correspondences_count_touching/registered_output/deformation3d.npy`
that the manuscript run (`_run_2d_clusters.py`) leaves at the wall
(`tri_min ≈ −0.002 to −0.006`).

## Quick start

```powershell
# crops only (cheap; ~5 min)
python run_all.py --fixture crop --save_npy

# full slices (1-2 hours total at default time budgets)
python run_all.py --fixture slice --time_budget_s 300 --save_npy

# subset of methods
python run_all.py --methods svf_polished harmonic+polish --fixture slice

# one slice
python run_all.py --fixture slice --z 12 --save_npy
```

Outputs land in `results/`:

* `summary.csv` and `summary.md` — leaderboard
* `{method}__{fixture}__z{Z}.json` — per-result metrics and method-specific info
* `{method}__{fixture}__z{Z}.npy` — corrected `(2, H, W)` slice (only with `--save_npy`)

The companion notebook `results_visualizer.ipynb` plots feasibility vs L2-cost and shows the warped-grid comparison for the best method.

## Method catalogue

| ID | Name | Idea | Guarantees feasibility? |
|---|---|---|---|
| m01 | `svf_squaring` | Exp(v) of `phi / 2^N`, scaling-and-squaring | **Yes** (Jdet > 0 in continuum) |
| m02 | `harmonic` | Laplacian extension over dilated fold cores | Often; needs convex ring (RKC theorem) |
| m03 | `aug_lagrangian` | PHR augmented Lagrangian, L-BFGS-B inner | No — but no active-set fragility |
| m04 | `paint_blend` | Bilinear patch over each core + Hann blend | Yes if ring quad convex |
| m05 | `torch_full_grid` | Penalty → log-barrier in PyTorch (CUDA) | No — same wall as scipy barrier |
| m06 | `quasi_conformal` | Bound Beltrami coefficient `|μ| < 1` | Equivalent to Jdet > 0 in continuum |
| m07 | `tv_anchor` | TV-regularised correction (drop L2 minimum) | No |
| m08 | `harmonic+polish` | m02 seed → torch polish → ALM cleanup | Inherits best stage |
| m09 | `svf_polished` | **m01 seed → scipy log-barrier L2 polish** | **Yes** (barrier preserves feasibility) |

## What we are measuring

Each method takes a `(2, H, W)` input slice (channels `[dy, dx]`, pull-back convention) and returns a corrected `(2, H, W)` slice. The harness reports three feasibility metrics, the L2 correction size, and wall time:

```
init  -> tri_neg=724 tri_min=-59.4    (input z=12)
```

The unambiguous success criterion is `tri_neg == 0 and tri_min >= 0.01`. Among methods satisfying that, lowest `l2_delta` wins.
