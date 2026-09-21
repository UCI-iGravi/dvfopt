<!-- certificate gauges: simplex: 2 triangles per cell (fixed BL-TR diagonal), triangle area = det/2, per cell (last row/col are +inf); bilinear: 4 triangles per cell (both diagonals), triangle area = det/2, i.e. exactly cell_min_jdet_2d / 2, per cell (last row/col are +inf); finite: forward-difference Jdet (1 triangle per cell), determinant, per cell (last row/col are +inf); jdet: central-difference Jdet, determinant, per pixel. certified = bilinear has 0 values < 0.01 - 1e-5 after. -1 is a sentinel (see summary.json notes), skipped by every median. -->
| source | config | n | certified | feasible | wall s (IQR) | L1 move (IQR) | L2 move (IQR) | SDlogJ before -> after | frac<=0 before -> after | max damage |
|---|---|---|---|---|---|---|---|---|---|---|
| origins | auto | 27 | 27/27 | 27/27 | 21.01 [2.093, 132.7] | 1042 [21.77, 7359] | 17.93 [1.844, 186.2] | 0.8696 -> 0.8042 | 0.00662 -> 0 | 0 |
| origins | barrier | 27 | 3/27 | 16/27 | 26.23 [2.852, 42.29] | 263.9 [31.51, 2599] | 8.517 [1.975, 70.26] | 0.8696 -> 0.8042 | 0.00662 -> 8.14e-05 | n/a |
| origins | isqp_l1 | 27 | 24/27 | 24/27 | 45.66 [5.882, 272.6] | 883.5 [19.34, 5199] | 22.35 [2.217, 176.2] | 0.8696 -> 0.8041 | 0.00662 -> 0 | 0 |
| origins | isqp_l2 | 27 | 27/27 | 27/27 | 32.13 [1.045, 401] | 1034 [21.28, 5703] | 17.87 [1.833, 161.5] | 0.8696 -> 0.8042 | 0.00662 -> 0 | 0 |
| origins | isqp_none | 27 | 27/27 | 27/27 | 7.765 [0.78, 58.77] | 1075 [23.75, 8891] | 18.12 [1.971, 194] | 0.8696 -> 0.7981 | 0.00662 -> 0 | 0 |
| origins | m14 | 27 | 4/27 | 21/27 | 17.19 [6.711, 121.1] | 280.7 [10.76, 4407] | 8.571 [1.133, 151] | 0.8696 -> 0.763 | 0.00662 -> 2.71e-05 | n/a |
| origins | slp | 27 | 3/27 | 24/27 | 20.13 [2.987, 195] | 232.6 [8.936, 3942] | 10.33 [1.353, 156.2] | 0.8696 -> 0.8041 | 0.00662 -> 5.43e-05 | n/a |
| origins | slsqp_windowed | 27 | 3/27 | 19/27 | 0.3216 [0.004323, 6.698] | 13.46 [0, 170.2] | 1.548 [0, 7.955] | 0.7427 -> 0.646 | 0.00208 -> 0 | n/a |