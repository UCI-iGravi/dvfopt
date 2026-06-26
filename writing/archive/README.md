# writing/archive/

Assets and `.tex` no longer used by the current TVCG manuscript
(`writing/tvcg/tvcg_manuscript.tex`). Archived 2026-06-25 during the
geometry-centric pivot (folding measures + DVFopt correction). Kept here in
case any are needed later; none are referenced by the active manuscript.

| Item | Why archived |
|---|---|
| `backup.tex` | Old manuscript backup; not `\input` anywhere. |
| `laplacian.tex` | The cut Laplacian-analysis paper text (interpolation-space + correspondence-augmentation threads). `\input{laplacian}` is commented out in the current manuscript and stays that way. |
| `signed_areas.pdf` | Stray duplicate that sat at the manuscript root; the figure actually used is `images/dvfopt/signed_areas.pdf`. |
| `images/dvfopt_old/` | Superseded older versions of the dvfopt figures. |
| `images/laplacian_analysis/ants/` | Figures for the cut ANTs comparison. |
| `images/laplacian_analysis/interpolation_space/` | Figures for the cut fixed-vs-moving interpolation-space analysis. |
| `images/laplacian_analysis/mouse_brain/` | Figures for the cut Laplacian mouse-brain analysis. |
| `images/laplacian_analysis/augmented_points_*.png`, `expanding_points_*.png`, `initial_correspondences.png`, `mapping_change.png` | Loose figures for the cut correspondence-augmentation analysis. |

Archived after consolidating into a single `.tex` + single `.bib` (2026-06-25):

| Item | Why archived |
|---|---|
| `dvfopt.tex` | Inlined into `tvcg/tvcg_manuscript.tex` (now self-contained; no `\input`). |
| `dvfopt.bib`, `laplacian.bib` | Merged + pruned into `tvcg/references.bib`, which keeps only the 45 works actually cited. Kept here as the full source bibs in case more citations are added later. |

NOT archived (still used by the current manuscript):
- `tvcg/dvfopt.bib`, `tvcg/laplacian.bib` — `laplacian.bib` still supplies cited keys (e.g. `AVANTS200826`, `Dalca2018`).
- `tvcg/images/dvfopt/` — active figures, plus the commented-out runtime PNGs
  (`grids_combined_row.png`, `slsqp_time_benchmark.png`, `runtime_plot.png`) that
  belong to the §7 runtime/scaling figure floats and may be re-enabled.
- `tvcg/images/laplacian_analysis/registration/` — the `fig:registration-example` images.
