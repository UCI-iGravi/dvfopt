| arm | pins | viol pairs | dropped | raw folds | raw min | final folds | <0 | floor | final min | resid px | <=10px % | L2 move | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0032_laplacian_all | 734772 | 17230775 | 457998 | 1785745 | -21 | 2 | 0 | 2 | +0.0099 | 1.15 | 98.2 | 9193 | 54 |
| B0032_laplacian_exterior | 400398 | 13161879 | 254169 | 988339 | -352 | 0 | 0 | 0 | +0.0101 | 1.20 | 98.2 | 14911 | 35 |
| B0039_laplacian_all | 820167 | 34109919 | 530603 | 1868747 | -154 | 0 | 0 | 0 | +0.0101 | 1.02 | 98.9 | 20036 | 47 |
| B0039_laplacian_exterior | 424682 | 10927126 | 275315 | 967234 | -312 | 0 | 0 | 0 | +0.0101 | 1.01 | 98.8 | 17904 | 0 |
| B0049_laplacian_all | 827187 | 16888340 | 476778 | 1793834 | -33 | 0 | 0 | 0 | +0.0101 | 0.87 | 99.0 | 8778 | 43 |
| B0049_laplacian_exterior | 427868 | 8109768 | 273593 | 917298 | -42 | 0 | 0 | 0 | +0.0101 | 0.98 | 98.9 | 6604 | 25 |
| B0053_laplacian_all | 762062 | 16266468 | 419788 | 1736898 | -19 | 0 | 0 | 0 | +0.0101 | 0.79 | 98.5 | 9282 | 55 |
| B0053_laplacian_exterior | 411476 | 7783920 | 260846 | 864550 | -23 | 0 | 0 | 0 | +0.0101 | 0.96 | 98.7 | 7651 | 26 |
| B0200_laplacian_all | 770814 | 16798714 | 459367 | 1769104 | -19 | 0 | 0 | 0 | +0.0101 | 1.02 | 98.6 | 8241 | 46 |
| B0200_laplacian_exterior | 410957 | 8339279 | 266556 | 895706 | -17 | 0 | 0 | 0 | +0.0101 | 1.02 | 98.6 | 8420 | 27 |
| B0213_laplacian_all | 751062 | 15805350 | 486304 | 1686027 | -15 | 0 | 0 | 0 | +0.0101 | 1.03 | 98.6 | 10823 | 40 |
| B0213_laplacian_exterior | 410002 | 7904681 | 228597 | 896404 | -36 | 0 | 0 | 0 | +0.0101 | 0.83 | 98.5 | 10599 | 37 |
| B0304_laplacian_exterior_c0.5 | 309833 | 650897023 | 287265 | 3738194 | -399 | 25 | 1 | 15 | -0.0004 | 9.43 | 51.6 | 110842 | 271 |
| B0304_laplacian_exterior_c0.5_tau2.0 | 205424 | 253987118 | 193721 | 3738194 | -399 | 8 | 0 | 3 | +0.0028 | 9.65 | 51.0 | 125656 | 151 |

B0304 exterior arms: `_c0.5` = tau 0.7 / c 0.5; `_c0.5_tau2.0` = tau 2 / c 0.5 (best: 8 sub-margin cells, none negative); tau 0.7 / c 1 and tau 2 / c 1 never finished 2.5D (6 h caps). B0304 laplacian_all (auto tau 2.38, 20 % pins kept) hit the 4 h cap in 2.5D. laplacian_all rows ran with tau=auto (0.70 on every clean brain). Walls contended by an unrelated job for the laplacian_exterior rows; laplacian_all rows ran alone except B0304 (an unrelated degu_atlas job from 16:31).
