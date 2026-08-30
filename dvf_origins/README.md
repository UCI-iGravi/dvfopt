# dvf_origins — sample DVFs by fold-origin mechanism

Standalone harness (not part of the `dvfopt` package) that generates the
paper's §4 cases: one displacement field per (mechanism, tool, data, variant), all
in `dvfopt`'s convention (`(3, 1, H, W)`, `[dz, dy, dx]`, `dz == 0`,
pull-back, voxel units), plus the fold-morphology table over them.

| mechanism | synthetic (seeded, no data) | real tool / data |
|---|---|---|
| 1 interpolation of sparse correspondences | `synthetic.interp_sparse` — Laplacian of corrupted correspondences (outliers, many-to-one collapses, jitter) | cohort Laplacian slice (`real.laplacian_slice`) |
| 2 dense weakly-regularized optimization | `synthetic.dense_weak_reg` — skimage TV-L1 / ILK on a textured pair | SimpleITK demons / B-spline FFD, skimage TV-L1 on the `data/mouse_brain` pair (`registered.py`) |
| 3 learned displacement field | `synthetic.learned_proxy` — **proxy** (smooth warp + grid-scale noise) | `learned.voxelmorph` / `learned.transmorph` — the `benchmarks/registration/` notebooks' networks trained here (direct and diffeo variants; needs the torch venv below) on the notebooks' synthetic images **or on real data** (`data='cohort'`: six cohort brains affinely aligned onto the template, coronal slices paired with the template's, test pair = B0039 z=264 like the m1/m4 rows; needs the RegTools outputs, `DVF_ORIGINS_REGTOOLS`), or any saved field via `real.saved_field` |
| 4 discretized diffeomorphic warp | `synthetic.diffeo_discretized` — SVF scaling-and-squaring, then decimation | cohort ANTs SyN warp slice (`real.ants_slice`) |

```bash
python -m dvf_origins list                     # the case registry (dvf_origins/__init__.py: CASES)
python -m dvf_origins generate                 # -> data/origins/<mechanism>/<case>.npy + .json, manifest.json (gitignored)
python -m dvf_origins generate --mechanism 1 4 # subset; cases whose data/deps are absent are skipped, with the reason
python -m dvf_origins sweep                    # -> output/origins/<timestamp>/results.csv + results_latest.csv
pytest dvf_origins                             # self-check (~3 s in the main venv, ~15 s in the torch venv; ~20 s more with the gitignored data; CI runs it too)
```

On disk (everything gitignored and regenerable; the learned rows cost minutes
of CPU, the cohort ones also need the external RegTools volumes):

```
data/origins/
  manifest.json                  case -> file, mechanism, tool, source, shape, build time
  m1_interpolation/              m1_laplacian_synthetic_{clean,outliers,collapse,mixed}, m1_laplacian_cohort_B0039_z264
  m2_dense_optimization/         m2_tvl1_synthetic_{weak,strong}, m2_ilk_synthetic_r3, m2_{demons,ffd}_brainpair_*, m2_tvl1_brainpair_a60
  m3_learned/                    m3_proxy_synthetic_{strong,mild}, m3_{voxelmorph,transmorph}_{ellipses,cohort}_{direct,diffeo}
  m4_diffeomorphic/              m4_svf_synthetic_{decimated,subpixel,coarse_steps}, m4_ants_cohort_B0039_z264
  external/learned.npy           optional hand-dropped field for the m3_external_saved_field row
  cache/                         real-slice cache for the cohort learned rows (hash-keyed)
output/origins/
  <timestamp>/results.csv        one sweep
  results_latest.csv             copy of the most recent FULL sweep of data/origins (stable path
                                 for the paper build; a sweep of another root needs --latest)
```

Each field is `<case>.npy` in `dvfopt`'s `(3, 1, H, W)` layout with a `<case>.json`
sidecar (tool, parameters, seed, timings, the convention/collapse checks for the
learned rows, grid info for the cohort rows), so a file is self-describing
without the registry. `manifest.json` is rebuilt from the tree by every
`generate` and `sweep` (never merged), so it cannot drift from what is on disk.

Names are `m<k>_<tool>_<data>_<variant>` — mechanism, the tool that made the
field, the data it was made from, the variant — and the self-check enforces
the shape. `<data>` is one of `synthetic` (generated images or pins),
`ellipses` (the notebooks' toy images), `brainpair` (the in-repo B0039/template
slice pair), `cohort` (the 7-brain RegTools cohort), `saved` (dropped in by
hand). A field at a path no `CASES` row maps to is reported by `sweep` and never
tabulated.

The learned rows (mechanism 3) train small networks and need torch, which the
main venv deliberately does not carry. A separate CPU venv is enough (the
models are 64×64 toys; VoxelMorph rows build in ~3-4 min, TransMorph in ~15-30):

```bash
uv venv .venv-torch --python 3.12
uv pip install --python .venv-torch --torch-backend=cpu \
    -e . torch "timm>=1.0" "voxelmorph @ git+https://github.com/voxelmorph/voxelmorph.git"
.venv-torch/Scripts/python -m dvf_origins generate --mechanism 3   # POSIX: .venv-torch/bin/python
python -m dvf_origins sweep                                          # any venv; reads data/origins/
```

The `*_cohort` rows train on real brains: `learned.cohort_data` resamples each
RegTools brain (`01_axis_alignment/axisAlignedData.nii.gz`) onto the template
grid through its ANTs affine (`fwd_transforms/ants_affine_1.mat` — verified:
B0039-vs-template slice correlation 0.18 identity → 0.87 affine → 0.94 SyN, so
the network learns the nonlinear residual SyN solved), takes coronal planes
`z = 60, 72, …, 468` of the six training brains paired with the template's, and
holds out B0039 at z=264. Planes are block-mean downsampled ×3 and centre-cropped
to 96×128 (the VoxelMorph UNet needs multiples of 32; ~85 % of the field of view
survives), so these rows live on a different grid than the native 320×456 m1/m4
rows of the same plane — compare fold *fractions*, not counts. The slice cache
lands in `data/origins/cache/` keyed by a hash of every input. The brain roster
is whatever `DVF_ORIGINS_REGTOOLS` (default: the sibling RegTools checkout)
holds with both files; the JSON meta records which brains trained.

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
