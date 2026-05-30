"""SLSQP strategies — full-grid (2-tri) and windowed (Jdet)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvfopt.constraints import (
    JdetConstraint2D,
    JdetConstraint3D,
    TriConstraint2D,
    TriConstraint2DFullCoverage,
)
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


@register_strategy('slsqp')
@dataclass
class SLSQPFullGridStrategy(Strategy):
    """Full-grid SLSQP with reactive warm-restart (notebook 14).

    2-triangle constraints only; the underlying full-grid Jacobian
    builder lives in :mod:`dvfopt.core.iterative2d_tri_slsqp` and
    assumes the 2-tri constraint structure.
    """

    max_iter: int = 50
    warm_max_iter: int = 1200
    warm_ftol: float = 1e-10
    warm_sigma: float = 0.01
    warm_seed: int = 123

    supports_3d: bool = False
    accepts_constraints = (TriConstraint2D, TriConstraint2DFullCoverage)

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core.iterative2d_tri_slsqp import iterative_2d_tri_slsqp

        self._check_constraint(constraint)
        full_coverage = isinstance(constraint, TriConstraint2DFullCoverage)
        out = iterative_2d_tri_slsqp(
            phi_in,
            threshold=threshold,
            max_iter=self.max_iter,
            warm_max_iter=self.warm_max_iter,
            warm_ftol=self.warm_ftol,
            warm_sigma=self.warm_sigma,
            warm_seed=self.warm_seed,
            anchor=objective.label or 'l2',
            eps_l1=getattr(objective, 'eps', 1e-4),
            full_coverage=full_coverage,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, hist = out
            return phi_out, _build_solve_info('SLSQPFullGridStrategy', {'history': hist}, threshold)
        return out, _build_solve_info('SLSQPFullGridStrategy', {}, threshold)


@register_strategy('slsqp_windowed')
@dataclass
class SLSQPWindowedStrategy(Strategy):
    """Windowed iterative SLSQP — the legacy Jdet path.

    Finds the worst-Jdet voxel/pixel, builds a bbox + 1-cell ring,
    solves the local SLSQP subproblem with frozen edges, repeats. Also
    supports the 2-triangle constraint via ``enforce_triangles=True``.
    """

    max_iterations: int = 80
    max_minimize_iter: int = 120

    supports_3d: bool = True

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        if isinstance(constraint, JdetConstraint2D):
            from dvfopt.core import iterative_serial

            H, W = constraint.shape
            deformation = np.zeros((3, 1, H, W), dtype=np.float64)
            deformation[1, 0] = phi_in[0]  # dy
            deformation[2, 0] = phi_in[1]  # dx
            out = iterative_serial(
                deformation,
                threshold=threshold,
                verbose=verbose,
                max_iterations=self.max_iterations,
                max_minimize_iter=self.max_minimize_iter,
                enforce_triangles=False,
            )
            return self._coerce_2d(out), _build_solve_info('SLSQPWindowedStrategy', {}, threshold)
        if isinstance(constraint, JdetConstraint3D):
            from dvfopt.core import iterative_3d

            D, H, W = constraint.shape
            deformation = np.zeros((3, D, H, W), dtype=np.float64)
            deformation[0] = phi_in[0]
            deformation[1] = phi_in[1]
            deformation[2] = phi_in[2]
            out = iterative_3d(
                deformation,
                threshold=threshold,
                verbose=verbose,
                max_iterations=self.max_iterations,
                max_minimize_iter=self.max_minimize_iter,
            )
            return out, _build_solve_info('SLSQPWindowedStrategy', {}, threshold)
        if isinstance(constraint, (TriConstraint2D, TriConstraint2DFullCoverage)):
            from dvfopt.core import iterative_serial

            H, W = constraint.shape
            deformation = np.zeros((3, 1, H, W), dtype=np.float64)
            deformation[1, 0] = phi_in[0]
            deformation[2, 0] = phi_in[1]
            out = iterative_serial(
                deformation,
                threshold=threshold,
                verbose=verbose,
                max_iterations=self.max_iterations,
                max_minimize_iter=self.max_minimize_iter,
                enforce_triangles=True,
            )
            return self._coerce_2d(out), _build_solve_info('SLSQPWindowedStrategy', {}, threshold)
        raise TypeError(
            f'SLSQPWindowedStrategy: unsupported constraint {type(constraint).__name__}'
        )

    @staticmethod
    def _coerce_2d(out):
        if out.ndim == 4:
            return np.stack([out[1, 0], out[2, 0]])
        if out.ndim == 3 and out.shape[0] == 3:
            return np.stack([out[1], out[2]])
        return out


__all__ = ['SLSQPFullGridStrategy', 'SLSQPWindowedStrategy']
