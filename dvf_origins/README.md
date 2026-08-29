# dvf_origins — sample DVFs by fold-origin mechanism

Standalone harness (not part of the `dvfopt` package) that generates the
paper's §4 cases: one displacement field per (mechanism, tool, severity), all
in `dvfopt`'s convention (`(3, 1, H, W)`, `[dz, dy, dx]`, `dz == 0`,
pull-back, voxel units), plus the fold-morphology table over them.

| mechanism | synthetic (seeded, no data) | real tool / data |
|---|---|---|
| 1 interpolation of sparse correspondences | `synthetic.interp_sparse` — Laplacian of corrupted correspondences (outliers, many-to-one collapses, jitter) | cohort Laplacian slice (`real.laplacian_slice`) |
| 2 dense weakly-regularized optimization | `synthetic.dense_weak_reg` — skimage TV-L1 / ILK on a textured pair | SimpleITK demons / B-spline FFD, skimage TV-L1 on the `data/mouse_brain` pair (`registered.py`) |
| 3 learned displacement field | `synthetic.learned_proxy` — **proxy** (smooth warp + grid-scale noise) | `learned.voxelmorph` / `learned.transmorph` — the `benchmarks/registration/` notebooks' networks trained here on synthetic images (direct and diffeo variants; needs the torch venv below), or any saved field via `real.saved_field` |
| 4 discretized diffeomorphic warp | `synthetic.diffeo_discretized` — SVF scaling-and-squaring, then decimation | cohort ANTs SyN warp slice (`real.ants_slice`) |

```bash
python -m dvf_origins list                     # the case registry (dvf_origins/__init__.py: CASES)
python -m dvf_origins generate                 # -> data/origins/<case>.npy + <case>.json (gitignored)
python -m dvf_origins generate --mechanism 1 4 # subset; cases whose data/deps are absent are skipped, with the reason
python -m dvf_origins sweep                    # -> output/origins/<timestamp>/results.csv
pytest dvf_origins                             # self-check (~3 s in the main venv, ~15 s in the torch venv; ~20 s more with the gitignored data; CI runs it too)
```

The learned rows (mechanism 3) train small networks and need torch, which the
main venv deliberately does not carry. A separate CPU venv is enough (the
models are 64×64 toys; VoxelMorph rows build in ~3-4 min, TransMorph in ~15-30):

```bash
uv venv .venv-torch --python 3.12
uv pip install --python .venv-torch --torch-backend=cpu \
    -e . torch timm "voxelmorph @ git+https://github.com/voxelmorph/voxelmorph.git"
.venv-torch/Scripts/python -m dvf_origins generate --mechanism 3   # POSIX: .venv-torch/bin/python
python -m dvf_origins sweep                                          # any venv; reads data/origins/
```

Each learned row records `warp_rmse` — pull-back-warping the source by the
returned field must reproduce the network's own warped output — next to the
same number with the channels swapped, so the `[dy, dx]` / `moving(x + u(x))`
convention is checked rather than assumed (it caught a ±0.5 px identity stretch
in the TransMorph notebook's sampler, fixed in both). `off_image_frac` is the
collapse detector: the notebook's 2×2 Swin bottleneck learns to shift the whole
source off-image (border padding returns black, MSE = mean(target²)), which
reads as a fold-free constant translation — the harness reads the encoder's
stage-0 (16×16) features instead (`feature_stage=0`; `None` = the notebook).

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
index-space voxels by the library's `dvfopt.io.fields.dvf_from_sitk_image`
(`D⁻¹·phys/spacing` through the NIfTI **direction matrix** — the cohort files
carry a signed permutation, and ignoring the geometry reads 4667 spurious 3D
folds vs 0) and re-laid-out onto the Laplacian field's `(i, j, k)` grid, so `z`
means the same plane in both real rows; a large in-plane fold count on it means
a convention mismatch, not folds. The
mechanism-3 proxy reproduces the morphology of an unregularized network
output, not the mechanism — label it as such in any table.
