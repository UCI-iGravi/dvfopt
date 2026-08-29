# dvf_origins — sample DVFs by fold-origin mechanism

Standalone harness (not part of the `dvfopt` package) that generates the
paper's §4 cases: one displacement field per (mechanism, tool, severity), all
in `dvfopt`'s convention (`(3, 1, H, W)`, `[dz, dy, dx]`, `dz == 0`,
pull-back, voxel units), plus the fold-morphology table over them.

| mechanism | synthetic (seeded, no data) | real tool / data |
|---|---|---|
| 1 interpolation of sparse correspondences | `synthetic.interp_sparse` — Laplacian of corrupted correspondences (outliers, many-to-one collapses, jitter) | cohort Laplacian slice (`real.laplacian_slice`) |
| 2 dense weakly-regularized optimization | `synthetic.dense_weak_reg` — skimage TV-L1 / ILK on a textured pair | SimpleITK demons / B-spline FFD, skimage TV-L1 on the `data/mouse_brain` pair (`registered.py`) |
| 3 learned displacement field | `synthetic.learned_proxy` — **proxy** (smooth warp + grid-scale noise) | saved VoxelMorph / TransMorph outputs (`real.saved_field`; make them with the `benchmarks/registration/` notebooks, needs torch) |
| 4 discretized diffeomorphic warp | `synthetic.diffeo_discretized` — SVF scaling-and-squaring, then decimation | cohort ANTs SyN warp slice (`real.ants_slice`) |

```bash
python -m dvf_origins list                     # the case registry (dvf_origins/__init__.py: CASES)
python -m dvf_origins generate                 # -> data/origins/<case>.npy + <case>.json (gitignored)
python -m dvf_origins generate --mechanism 1 4 # subset; cases whose data/deps are absent are skipped, with the reason
python -m dvf_origins sweep                    # -> output/origins/<timestamp>/results.csv
pytest dvf_origins                             # self-check (~20 s; CI runs it too, data-gated tests skip there)
```

`sweep` columns (`morphology.py`): size, displacement magnitude, then the
same field under three metrics — central-difference Jdet per pixel, the
simplex (piecewise-linear) certificate per cell / per triangle, the bilinear
sub-pixel certificate per cell — with `bilinear_only_cells` (cells the
bilinear certificate folds that the simplex metric passes), fold fraction,
and cluster count / median / max area (8-connected on the simplex cell mask).

Add a case: one row in `CASES`. Add a mechanism variant: a function returning
`(phi, meta)`; put `source` and `tool` in `meta`.

Conventions worth knowing before trusting a row: skimage optical flow and
SimpleITK displacement fields are already pull-back (fixed → moving), so they
are used unchanged; the ANTs warp is converted from physical (mm, LPS) to
index-space voxels through the NIfTI **direction matrix** (`D⁻¹·phys/spacing` —
the cohort files carry a signed permutation, so dividing by spacing alone mixes
components: 4667 spurious 3D folds vs 0) and re-laid-out onto the Laplacian
field's `(i, j, k)` grid, so `z` means the same plane in both real rows; a
large in-plane fold count on it means a convention mismatch, not folds. The
mechanism-3 proxy reproduces the morphology of an unregularized network
output, not the mechanism — label it as such in any table.
