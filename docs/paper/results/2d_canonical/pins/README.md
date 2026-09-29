# Canonical 2D benchmark with pin preprocessing (run 2026-09-27/28)

`benchmarks/canonical_2d.py` (branch `feat/canonical-pins`) over every 2D source x 9 configs,
`--n-workers 4`, one run dir, 1,917 rows, 0 errors, 6 `hit_cap`, ~29 h wall on the dev box (4 pool
workers; the box was otherwise idle). Configs: the windowed I-SQP engine with objective none / l2 / l1
(`isqp_*`, the baseline, no pins) and the same three with pin preprocessing in front: `pins_*` (harmonic
re-fill, the shipped `correct_dvf_pins` re-fill) and `pinss_*` (source-preserving re-fill, the comparison
arm). Metrics are against the ORIGINAL input; correspondence files are read only for the landmark-residual
column.

Guards (added after the first pass and re-measured for the 54 rows they change, 9 cases x 6 pins configs,
in a side run merged here): a fold-free input is never touched; a 2D-stencil pin read with fewer than 50
pins passes through untouched; cohort slices (Laplacian by construction) keep their 3D-stencil read at any
count.

`table.md` carries the per-source medians, the paired ratios against the baseline, the ground-truth error on
the m1 synthetic origins (clean = same seed, uncorrupted), and every non-certified row. `results.csv` is the
merged row set without the local `out_path` / `sha256` columns (the corrected DVFs are not tracked).

Headline (cohort, 85 slices, paired): harmonic pins wall x0.027 (l2) / x0.053 (none) / x0.030 (l1) at
unchanged certification (l1 83 -> 85 / 85), L2 move x9-11, landmark residual 0.16 -> 0.56 px.
Source-preserving loses on every column (wall x0.26-0.32, residual ~0.9-1.0 px, l1 83 -> 80). On the m1
synthetic origins the pins output is 6-8x closer to the clean field on outliers / mixed (RMS 3.02 -> 0.38,
2.46 -> 0.53; the baseline moves 0.02-0.09) and worse on collapse (0.57 -> 0.95). Every other source passes
through unchanged (pins_n 0 or below the guard). One documented false positive remains:
m2_tvl1_synthetic_weak (191 "pins" on a non-Laplacian field, L2 x2.5, still certified).
