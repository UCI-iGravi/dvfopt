# data/dvfs — the DVF suite

Every deformation field this repo consumes (benchmark inputs) or is meant to
keep (results) lives under this root. Payloads are gitignored — regenerable or
copied data; this README is the tracked map. `python -m dvf_origins index`
inventories the tree into `manifest.json` (per-section file counts, bytes, and
the per-case manifest for `origins/`).

All fields are dvfopt-native unless a converter is named: `(3, 1, H, W)` /
`(3, D, H, W)` float arrays, channels `[dz, dy, dx]`, pull-back convention,
voxel units.

| section | what | provenance / regeneration |
|---|---|---|
| `origins/` | the fold-origin benchmark suite: one field per (mechanism, tool, data, variant), `m1_interpolation/` … `m4_diffeomorphic/`, each `.npy` with a `.json` sidecar, plus its own `manifest.json` and the learned rows' `cache/` | `python -m dvf_origins generate` (learned rows need the torch venv — `dvf_origins/README.md`) |
| `cohort/<brain>/<variant>/` | 7 real brains × {`laplacian_all`, `laplacian_exterior`}: `laplacian_deformation_field.npz` (3, 528, 320, 456), `ants_warp_0.nii.gz` (SyN, physical units — load via `dvfopt.io.fields`), `mpoints/fpoints.npz` correspondences (~17 GB) | copied from the sibling RegTools project (`output/brain25_cohort`); accessor: `benchmarks/benchmark_utils.cohort_dir()` |
| `crops/` | the hard B0039 benchmark crops (`z0_sliver`, `z0_cluster`, `z16_twist`, …) | `python benchmarks/make_hard_crops.py` |
| `testcases/` | small 2D fixture fields (`01a_*` synthetic, `02a_*` real-data slices) + preview PNGs | `notebooks/generate_test_cases.ipynb` era; consumed by legacy comparison scripts |
| `testcases_3d/` | small 3D crops for `benchmarks/two_triangle` | `scripts/_save_3d_crops.py` |
| `canonical_2tri_2d/` | the curated canonical 2-tri worst cases | `scripts/_save_canonical_2tri_2d.py` |
| `b0039/`, `b0036/` | older single-brain full-volume Laplacian fields (`b0039/…npy` is the benchmark scripts' `DEFAULT_VOL`; NOT byte-identical to `cohort/B0039`) | RegTools, earlier runs |
| `archive/` | superseded fields kept for provenance | — |
| `results/` | corrected-field artifacts you want to keep (campaign outputs, paper deliverables). Run tables/reports stay under `output/` and `benchmarks/output/` | produced by `dvfopt correct` / campaign scripts |

Compatibility: the pre-suite paths — `data/origins`, `data/test_cases`,
`data/test_cases_3d`, `data/dvfs/brain25_cohort_corrected`,
`benchmarks/output/testcases` — are directory **junctions** onto the sections
above, so old branches, notebooks and external tooling keep working. New code
should name the suite paths (`dvf_origins._common.DVFS`, `cohort_dir()`).
