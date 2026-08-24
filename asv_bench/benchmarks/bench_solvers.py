"""Airspeed Velocity benchmarks for dvfopt solvers.

Run:  asv run          (build isolated envs per commit — slow)
      asv check -E existing   (validate benchmarks import/discover — fast)
      asv continuous main HEAD   (compare a branch to main)

Keep these cheap and deterministic (fixed seeds, small grids) so a full sweep
across commits stays tractable — they exist to catch *regressions*, not to
measure absolute solver performance (the benchmarks/ notebooks do that).
"""

import numpy as np

from dvfopt import correct_dvf, jacobian_det2D


def _planted_fold(size, seed=0, scale=0.4):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, scale, (size, size)), rng.normal(0, scale, (size, size))])


class Correct2D:
    """Wall-time of the auto simplex (2D) / L1 correction at a couple of grid sizes."""

    params = [12, 24]
    param_names = ["size"]

    def setup(self, size):
        self.phi = _planted_fold(size)

    def time_correct_2tri_l1_auto(self, size):
        correct_dvf(self.phi, constraint="simplex", objective="l1", strategy="auto")

    def time_correct_jdet_barrier(self, size):
        correct_dvf(self.phi, constraint="jdet", objective="l1", strategy="barrier")


class Jacobian2D:
    """Wall-time of the fast numpy Jacobian determinant (a hot inner kernel)."""

    params = [64, 128]
    param_names = ["size"]

    def setup(self, size):
        phi = _planted_fold(size)
        self.field = np.stack([np.zeros_like(phi[0]), phi[0], phi[1]])[:, None]

    def time_jacobian_det2d(self, size):
        jacobian_det2D(self.field)
