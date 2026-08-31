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


class WindowedEngine:
    """The zero-folds campaign's engine, pinned to its own metrics: wall time plus
    the contention-proof SQP-iteration and L2-move counters. A regression in the
    trajectory (more iterations), the QP cost (more wall at equal iterations) or
    the fidelity (a larger move) each fails a separate metric."""

    timeout = 300

    def setup(self):
        from dvfopt.constraints import SimplexConstraint2DBilinear
        from dvfopt.testdata import make_random_dvf

        patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
        self.phi = np.zeros((2, 64, 64))
        self.phi[:, 24 : 24 + patch.shape[1], 26 : 26 + patch.shape[2]] = patch
        self.constraint = SimplexConstraint2DBilinear(shape=(64, 64))
        # one CONNECTED region over max_window_area -> exercises the giant tiler
        self.giant = np.zeros((2, 120, 120))
        for by in range(6):
            for bx in range(6):
                y, x = 25 + by * 10, 25 + bx * 10
                self.giant[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
        self.giant_constraint = SimplexConstraint2DBilinear(shape=(120, 120))

    def _run(self, phi, constraint, objective):
        from dvfopt.core.windowed import windowed_correct

        out, rep = windowed_correct(
            phi.copy(),
            "isqp",
            constraint=constraint,
            objective=objective,
            threshold=0.01,
            verbose=0,
        )
        assert rep.folds_after == 0 and rep.damage == 0
        return out, rep

    def time_window_l2(self):
        from dvfopt.objectives import L2Objective

        self._run(self.phi, self.constraint, L2Objective())

    def time_window_none(self):
        from dvfopt.objectives import NoneObjective

        self._run(self.phi, self.constraint, NoneObjective())

    def time_giant_tiler_l2(self):
        from dvfopt.objectives import L2Objective

        self._run(self.giant, self.giant_constraint, L2Objective())

    def track_sqp_iters_l2(self):
        from dvfopt.objectives import L2Objective

        _out, rep = self._run(self.phi, self.constraint, L2Objective())
        return float(sum(w.inner_iters for w in rep.windows) + rep.coarse_iters)

    def track_l2_move(self):
        from dvfopt.objectives import L2Objective

        out, _rep = self._run(self.phi, self.constraint, L2Objective())
        return float(np.linalg.norm((out - self.phi).ravel()))

    track_sqp_iters_l2.unit = "iterations"
    track_l2_move.unit = "L2"
