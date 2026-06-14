# Worst-case catalog

## Synthetic
| Case | Source | Shape | init n_neg | init min_T |
|---|---|---|---|---|
| `bowtie_7x7_shoelace` | data/dvfs/canonical_2tri_2d/ | 7×7 | 2 | -0.70 |
| `01a_10x10_crossing` | data/dvfs/canonical_2tri_2d/ | 10×10 | 24 | -0.74 |
| `01b_10x10_opposite` | data/dvfs/canonical_2tri_2d/ | 10×10 | 10 | -0.59 |
| `03a_10x10_opposite` | data/dvfs/canonical_2tri_2d/ | 10×10 | 23 | -0.81 |
| `03b_10x10_crossing` | data/dvfs/canonical_2tri_2d/ | 10×10 | 28 | -0.70 |
| `03c_20x20_opposite` | data/dvfs/canonical_2tri_2d/ | 20×20 | 58 | -0.81 |
| `03d_20x20_crossing` | data/dvfs/canonical_2tri_2d/ | 20×20 | 72 | -0.74 |

## Synthetic — adversarial (built by `_build_adversarial.py`)
| Case | Shape | init n_neg | init min_T | Purpose |
|---|---|---|---|---|
| `dense_bowtie_cluster_15x15` | 15×15 | 12 | -0.70 | Dense single-cluster bowtie field |
| `tiny_margin_10x10` | 10×10 | 45 | -0.05 | Stress linearisation: every interior cell barely infeasible |

## B0039
| Case | Source | Shape | Status |
|---|---|---|---|
| `b0039_z012` | data/dvfs/b0039/b0039_laplacian_deformation_field.npy z=12 | 320×456 | Manuscript-canonical hardest slice |

Empirical worst slices (from cluster-pipeline residuals) discovered
after the synthetic suite is settled — added inline.
