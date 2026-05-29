## 4. Visualization: fold-cluster decomposition + bounding boxes

For the multi-cluster real-data case (`z12_60x60`), show:

- Left: initial `min(T1, T2)` heatmap with fold cells outlined.
- Center: cluster decomposition (each cluster a different color) with
  bounding-box overlays for the crops m14-Schwarz processes.
- Right: final state after m14-Schwarz.

If clusters are well-separated, the crops cover only a small fraction
of the slice — that's where the wall-clock win comes from.
