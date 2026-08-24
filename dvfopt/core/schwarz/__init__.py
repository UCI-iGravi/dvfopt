"""Schwarz domain decomposition — one home for the 2-triangle/simplex (3D) split.

Modules: ``_common`` (generic ``cluster_schwarz_2d_tri`` /
``cluster_schwarz_3d_tet`` decomposition core, shared by the wallbreaker
m14-Schwarz drivers and :class:`SchwarzWrapperStrategy`), ``tri2d``
(hybrid overlapping-tile Schwarz + per-cluster SLSQP,
``iterative_2d_tri_schwarz``), ``_cluster`` (per-cluster 2-triangle
SLSQP with a frozen-edge interior mask, ``solve_cluster_2tri_2d``).
"""
