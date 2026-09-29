# Canonical 2D benchmark with pin preprocessing — canonical_pins_2026-09-27 (54 rows re-measured under the guards from canonical_pins_2026-09-27_regard)

Configs: `isqp_*` = the windowed I-SQP engine (bilinear rows) with objective none/l2/l1 (baseline, no pins); `pins_*` = pin read → pairwise drop → HARMONIC re-fill, then the same engine; `pinss_*` = same with the SOURCE-PRESERVING re-fill. Metrics are against the ORIGINAL input. Cohort slices read pins with the 3D stencil from the parent volume; other sources with the 2D stencil. Guards: fold-free inputs and 2D reads with < 50 pins pass through untouched.

## Per source × config (medians; certified = fraction; t = wall s; L2 = move from input; resid = landmark residual px, cohort only)

### cohort

| config | n | certified | t median s | t sum h | L2 median | resid px | pins med | dropped med | hit_cap |
|---|---|---|---|---|---|---|---|---|---|
| isqp_none | 85 | 1.000 | 80.3 | 4.06 | 44.3 | 0.21 | -1 | -1 | 0 |
| pins_isqp_none | 85 | 1.000 | 4.0 | 0.12 | 446.2 | 0.56 | 752 | 397 | 0 |
| pinss_isqp_none | 85 | 1.000 | 18.1 | 3.04 | 421.3 | 0.91 | 752 | 397 | 0 |
| isqp_l2 | 85 | 1.000 | 199.8 | 7.75 | 36.4 | 0.16 | -1 | -1 | 0 |
| pins_isqp_l2 | 85 | 1.000 | 5.4 | 0.20 | 446.2 | 0.56 | 752 | 397 | 0 |
| pinss_isqp_l2 | 85 | 1.000 | 45.3 | 5.26 | 425.2 | 0.90 | 752 | 397 | 0 |
| isqp_l1 | 85 | 0.976 | 563.1 | 23.86 | 42.4 | 0.15 | -1 | -1 | 2 |
| pins_isqp_l1 | 85 | 1.000 | 15.4 | 0.63 | 446.2 | 0.59 | 752 | 397 | 0 |
| pinss_isqp_l1 | 85 | 0.941 | 140.0 | 15.72 | 420.5 | 1.02 | 752 | 397 | 1 |

### origins

| config | n | certified | t median s | t sum h | L2 median | resid px | pins med | dropped med | hit_cap |
|---|---|---|---|---|---|---|---|---|---|
| isqp_none | 27 | 1.000 | 7.1 | 0.97 | 18.1 |  | -1 | -1 | 0 |
| pins_isqp_none | 27 | 1.000 | 2.9 | 0.96 | 47.4 |  | 0 | 0 | 0 |
| pinss_isqp_none | 27 | 1.000 | 3.6 | 0.96 | 47.4 |  | 0 | 0 | 0 |
| isqp_l2 | 27 | 1.000 | 17.5 | 3.09 | 17.9 |  | -1 | -1 | 0 |
| pins_isqp_l2 | 27 | 1.000 | 4.8 | 2.99 | 39.5 |  | 0 | 0 | 0 |
| pinss_isqp_l2 | 27 | 1.000 | 8.4 | 3.11 | 39.5 |  | 0 | 0 | 0 |
| isqp_l1 | 27 | 0.889 | 51.1 | 14.12 | 22.3 |  | -1 | -1 | 1 |
| pins_isqp_l1 | 27 | 0.889 | 21.3 | 13.92 | 43.3 |  | 0 | 0 | 1 |
| pinss_isqp_l1 | 27 | 0.889 | 25.9 | 14.16 | 43.3 |  | 0 | 0 | 1 |

### crops

| config | n | certified | t median s | t sum h | L2 median | resid px | pins med | dropped med | hit_cap |
|---|---|---|---|---|---|---|---|---|---|
| isqp_none | 3 | 1.000 | 6.1 | 0.02 | 787.5 |  | -1 | -1 | 0 |
| pins_isqp_none | 3 | 1.000 | 5.6 | 0.02 | 787.5 |  | 0 | 0 | 0 |
| pinss_isqp_none | 3 | 1.000 | 5.0 | 0.02 | 787.5 |  | 0 | 0 | 0 |
| isqp_l2 | 3 | 1.000 | 23.8 | 0.08 | 690.5 |  | -1 | -1 | 0 |
| pins_isqp_l2 | 3 | 1.000 | 23.7 | 0.08 | 690.5 |  | 0 | 0 | 0 |
| pinss_isqp_l2 | 3 | 1.000 | 23.5 | 0.08 | 690.5 |  | 0 | 0 | 0 |
| isqp_l1 | 3 | 1.000 | 30.9 | 0.09 | 681.5 |  | -1 | -1 | 0 |
| pins_isqp_l1 | 3 | 1.000 | 29.9 | 0.09 | 681.5 |  | 0 | 0 | 0 |
| pinss_isqp_l1 | 3 | 1.000 | 32.9 | 0.09 | 681.5 |  | 0 | 0 | 0 |

### synthetic

| config | n | certified | t median s | t sum h | L2 median | resid px | pins med | dropped med | hit_cap |
|---|---|---|---|---|---|---|---|---|---|
| isqp_none | 13 | 1.000 | 0.1 | 0.00 | 6.5 |  | -1 | -1 | 0 |
| pins_isqp_none | 13 | 1.000 | 0.1 | 0.00 | 6.5 |  | 0 | 0 | 0 |
| pinss_isqp_none | 13 | 1.000 | 0.2 | 0.00 | 6.5 |  | 0 | 0 | 0 |
| isqp_l2 | 13 | 1.000 | 0.3 | 0.00 | 6.0 |  | -1 | -1 | 0 |
| pins_isqp_l2 | 13 | 1.000 | 0.4 | 0.00 | 6.0 |  | 0 | 0 | 0 |
| pinss_isqp_l2 | 13 | 1.000 | 0.3 | 0.00 | 6.0 |  | 0 | 0 | 0 |
| isqp_l1 | 13 | 1.000 | 0.6 | 0.01 | 7.0 |  | -1 | -1 | 0 |
| pins_isqp_l1 | 13 | 1.000 | 0.7 | 0.01 | 7.0 |  | 0 | 0 | 0 |
| pinss_isqp_l1 | 13 | 1.000 | 0.7 | 0.01 | 7.0 |  | 0 | 0 | 0 |

### ants

| config | n | certified | t median s | t sum h | L2 median | resid px | pins med | dropped med | hit_cap |
|---|---|---|---|---|---|---|---|---|---|
| isqp_none | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | -1 | -1 | 0 |
| pins_isqp_none | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | 0 | 0 | 0 |
| pinss_isqp_none | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | 0 | 0 | 0 |
| isqp_l2 | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | -1 | -1 | 0 |
| pins_isqp_l2 | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | 0 | 0 | 0 |
| pinss_isqp_l2 | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | 0 | 0 | 0 |
| isqp_l1 | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | -1 | -1 | 0 |
| pins_isqp_l1 | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | 0 | 0 | 0 |
| pinss_isqp_l1 | 85 | 1.000 | 0.2 | 0.01 | 0.0 |  | 0 | 0 | 0 |

## Paired against the baseline (same case): median wall ratio, median L2 ratio, certification, residual

| twin vs baseline | source | n | wall × | L2 × | certified | resid px |
|---|---|---|---|---|---|---|
| pins_isqp_none vs isqp_none | cohort | 85 | 0.053 | 8.92 | 85 → 85 | 0.21 → 0.56 |
| pins_isqp_none vs isqp_none | origins | 27 | 0.986 | 1.00 | 27 → 27 |  |
| pins_isqp_none vs isqp_none | crops | 3 | 0.949 | 1.00 | 3 → 3 |  |
| pins_isqp_none vs isqp_none | synthetic | 13 | 0.987 | 1.00 | 13 → 13 |  |
| pins_isqp_none vs isqp_none | ants | 85 | 1.057 | 0.00 | 85 → 85 |  |
| pins_isqp_l2 vs isqp_l2 | cohort | 85 | 0.027 | 11.24 | 85 → 85 | 0.16 → 0.56 |
| pins_isqp_l2 vs isqp_l2 | origins | 27 | 0.983 | 1.00 | 27 → 27 |  |
| pins_isqp_l2 vs isqp_l2 | crops | 3 | 0.991 | 1.00 | 3 → 3 |  |
| pins_isqp_l2 vs isqp_l2 | synthetic | 13 | 1.039 | 1.00 | 13 → 13 |  |
| pins_isqp_l2 vs isqp_l2 | ants | 85 | 1.064 | 0.00 | 85 → 85 |  |
| pins_isqp_l1 vs isqp_l1 | cohort | 85 | 0.030 | 9.78 | 83 → 85 | 0.15 → 0.59 |
| pins_isqp_l1 vs isqp_l1 | origins | 27 | 0.946 | 1.00 | 24 → 24 |  |
| pins_isqp_l1 vs isqp_l1 | crops | 3 | 0.989 | 1.00 | 3 → 3 |  |
| pins_isqp_l1 vs isqp_l1 | synthetic | 13 | 0.986 | 1.00 | 13 → 13 |  |
| pins_isqp_l1 vs isqp_l1 | ants | 85 | 1.085 | 0.00 | 85 → 85 |  |
| pinss_isqp_none vs isqp_none | cohort | 85 | 0.260 | 8.88 | 85 → 85 | 0.21 → 0.91 |
| pinss_isqp_none vs isqp_none | origins | 27 | 0.950 | 1.00 | 27 → 27 |  |
| pinss_isqp_none vs isqp_none | crops | 3 | 0.954 | 1.00 | 3 → 3 |  |
| pinss_isqp_none vs isqp_none | synthetic | 13 | 1.025 | 1.00 | 13 → 13 |  |
| pinss_isqp_none vs isqp_none | ants | 85 | 1.035 | 0.00 | 85 → 85 |  |
| pinss_isqp_l2 vs isqp_l2 | cohort | 85 | 0.283 | 10.64 | 85 → 85 | 0.16 → 0.90 |
| pinss_isqp_l2 vs isqp_l2 | origins | 27 | 0.969 | 1.00 | 27 → 27 |  |
| pinss_isqp_l2 vs isqp_l2 | crops | 3 | 0.985 | 1.00 | 3 → 3 |  |
| pinss_isqp_l2 vs isqp_l2 | synthetic | 13 | 1.014 | 1.00 | 13 → 13 |  |
| pinss_isqp_l2 vs isqp_l2 | ants | 85 | 1.094 | 0.00 | 85 → 85 |  |
| pinss_isqp_l1 vs isqp_l1 | cohort | 85 | 0.319 | 9.27 | 83 → 80 | 0.15 → 1.02 |
| pinss_isqp_l1 vs isqp_l1 | origins | 27 | 0.985 | 1.00 | 24 → 24 |  |
| pinss_isqp_l1 vs isqp_l1 | crops | 3 | 1.018 | 1.00 | 3 → 3 |  |
| pinss_isqp_l1 vs isqp_l1 | synthetic | 13 | 0.988 | 1.00 | 13 → 13 |  |
| pinss_isqp_l1 vs isqp_l1 | ants | 85 | 1.060 | 0.00 | 85 → 85 |  |

## Ground truth on the m1 synthetic origins (clean = same seed, uncorrupted correspondences): RMS / max error of the output to the clean field

| case | config | RMS err | max err | L2 move from input |
|---|---|---|---|---|
| m1_laplacian_synthetic_collapse | (input) | 0.567 | 9.39 | |
| | isqp_none | 0.564 | 7.93 | 6.9 |
| | isqp_l2 | 0.564 | 8.17 | 6.4 |
| | isqp_l1 | 0.557 | 7.75 | 7.6 |
| | pins_isqp_none | 0.950 | 3.92 | 211.9 |
| | pins_isqp_l2 | 0.950 | 3.93 | 211.9 |
| | pins_isqp_l1 | 0.950 | 3.62 | 211.9 |
| m1_laplacian_synthetic_outliers | (input) | 3.024 | 29.97 | |
| | isqp_none | 2.983 | 18.39 | 134.2 |
| | isqp_l2 | 2.940 | 19.49 | 95.2 |
| | isqp_l1 | 2.888 | 18.87 | 104.7 |
| | pins_isqp_none | 0.375 | 2.12 | 594.5 |
| | pins_isqp_l2 | 0.375 | 2.12 | 594.5 |
| | pins_isqp_l1 | 0.375 | 2.12 | 594.5 |
| m1_laplacian_synthetic_mixed | (input) | 2.458 | 30.34 | |
| | isqp_none | 2.442 | 17.23 | 100.9 |
| | isqp_l2 | 2.406 | 18.59 | 69.0 |
| | isqp_l1 | 2.370 | 16.40 | 77.5 |
| | pins_isqp_none | 0.526 | 4.98 | 441.7 |
| | pins_isqp_l2 | 0.526 | 5.20 | 441.7 |
| | pins_isqp_l1 | 0.524 | 4.30 | 441.8 |

## Non-certified rows

| case | config | folds init | folds final | t s | hit_cap | pins | dropped |
|---|---|---|---|---|---|---|---|
| m2_demons_brainpair_weak | isqp_l1 | 15169 | 0 | 2872 | False | -1 | -1 |
| m2_demons_brainpair_weak | pins_isqp_l1 | 15169 | 0 | 2909 | False | 0 | 0 |
| m2_demons_brainpair_weak | pinss_isqp_l1 | 15169 | 0 | 3056 | False | 0 | 0 |
| m2_ffd_brainpair_fine | isqp_l1 | 9884 | 0 | 4952 | False | -1 | -1 |
| m2_ffd_brainpair_fine | pins_isqp_l1 | 9884 | 0 | 4788 | False | 0 | 0 |
| m2_ffd_brainpair_fine | pinss_isqp_l1 | 9884 | 0 | 5012 | False | 0 | 0 |
| m2_ffd_brainpair_coarse | isqp_l1 | 16324 | 1 | 39075 | True | -1 | -1 |
| m2_ffd_brainpair_coarse | pins_isqp_l1 | 16324 | 1 | 39246 | True | 0 | 0 |
| m2_ffd_brainpair_coarse | pinss_isqp_l1 | 16324 | 1 | 39541 | True | 0 | 0 |
| cohort_B0032_z1 | pinss_isqp_l1 | 4578 | 1 | 3245 | False | 356 | 313 |
| cohort_B0039_z1 | pinss_isqp_l1 | 3892 | 0 | 812 | False | 99 | 83 |
| cohort_B0304_z128 | pinss_isqp_l1 | 9013 | 0 | 2265 | False | 461 | 228 |
| cohort_B0304_z144 | isqp_l1 | 20340 | 0 | 25456 | True | -1 | -1 |
| cohort_B0304_z144 | pinss_isqp_l1 | 20340 | 0 | 23179 | True | 438 | 304 |
| cohort_B0304_z432 | pinss_isqp_l1 | 11873 | 0 | 3286 | False | 724 | 521 |
| cohort_B0304_z432 | isqp_l1 | 11873 | 0 | 5184 | False | -1 | -1 |
