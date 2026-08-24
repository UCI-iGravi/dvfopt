"""SLSQP strategies — full-grid (simplex (2D) / simplex (3D)) and windowed (Jdet)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvfopt.constraints import (
    JdetConstraint2D,
    JdetConstraint3D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
    SimplexConstraint2DFullCoverage,
    SimplexConstraint3D,
)
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


def _lift_slsqp_trace(raw_history, info) -> None:
    """Surface per-major-iteration SLSQP traces into ``info.extras``.

    ``raw_history`` entries (Tasks 7-8) carry a ``'trace'`` dict with
    per-major-iteration records under ``'trace'['iters']``. Lifted to a
    stable ``info.extras['slsqp_trace']`` path so callers (GUI, reports)
    don't have to reach into per-phase ``PhaseInfo.extras``.
    """
    if not raw_history:
        return
    traces = [
        {'phase': h.get('phase', f'run{i}'), **h['trace']}
        for i, h in enumerate(raw_history)
        if isinstance(h, dict) and h.get('trace')
    ]
    if traces:
        info.extras['slsqp_trace'] = traces


@register_strategy('slsqp')
@dataclass
class SLSQPFullGridStrategy(Strategy):
    """Full-grid SLSQP with reactive warm-restart (notebook 14).

    2-triangle constraints only; the underlying full-grid Jacobian
    builder lives in :mod:`dvfopt.core.slsqp_fullgrid.tri2d` and
    assumes the simplex (2D) constraint structure.
    """

    max_iter: int = 50
    warm_max_iter: int = 1200
    warm_ftol: float = 1e-10
    warm_sigma: float = 0.01
    warm_seed: int = 123

    supports_3d: bool = False
    accepts_constraints = (SimplexConstraint2D, SimplexConstraint2DFullCoverage)

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        from dvfopt.core.slsqp_fullgrid.tri2d import iterative_2d_tri_slsqp

        self._check_constraint(constraint)
        full_coverage = isinstance(constraint, SimplexConstraint2DFullCoverage)
        out = iterative_2d_tri_slsqp(
            phi_in,
            threshold=threshold,
            max_iter=self.max_iter,
            warm_max_iter=self.warm_max_iter,
            warm_ftol=self.warm_ftol,
            warm_sigma=self.warm_sigma,
            warm_seed=self.warm_seed,
            objective=objective,
            full_coverage=full_coverage,
            verbose=verbose,
            record_history=record_history,
        )
        raw_history = out[1] if record_history else None
        phi_out, info = self._finish(out, record_history, threshold, wrap_history=True)
        _lift_slsqp_trace(raw_history, info)
        return phi_out, info


@register_strategy('slsqp_windowed')
@dataclass
class SLSQPWindowedStrategy(Strategy):
    """Windowed iterative SLSQP — the legacy Jdet path.

    Finds the worst-Jdet voxel/pixel, builds a bbox + 1-cell ring,
    solves the local SLSQP subproblem with frozen edges, repeats. The
    2D triangle constraints run its ``enforce_triangles=True`` mode,
    which enforces all four triangles of every cell (both diagonals) —
    exactly :class:`~dvfopt.constraints.SimplexConstraint2DBilinear`, and a
    superset of the simplex (2D) pair.

    The composed :class:`~dvfopt.objectives.Objective` is plumbed all
    the way down to the per-window ``scipy.optimize.minimize`` call and
    evaluated on ``phi - phi_init`` (``None`` means L2, the historical
    behaviour).
    """

    max_iterations: int = 80
    max_minimize_iter: int = 120
    # Extra constraint modes. 2D supports both flags; 3D supports
    # enforce_injectivity (axial monotonicity, linear rows). The 3D
    # analogue of enforce_shoelace is the simplex (3D) constraint family.
    # NOTE on injectivity_threshold=None semantics: the 2D path runs the
    # adaptive tau-doubling loop (doubling until globally injective);
    # the 3D path simply defaults the gap bound to `threshold` — no
    # adaptive loop and no global-injectivity certificate.
    enforce_shoelace: bool = False
    enforce_injectivity: bool = False
    injectivity_threshold: float | None = None

    supports_3d: bool = True
    accepts_constraints = (
        JdetConstraint2D,
        JdetConstraint3D,
        SimplexConstraint2D,
        SimplexConstraint2DFullCoverage,
        SimplexConstraint2DBilinear,
    )

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        self._check_constraint(constraint)
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
                enforce_shoelace=self.enforce_shoelace,
                enforce_injectivity=self.enforce_injectivity,
                injectivity_threshold=self.injectivity_threshold,
                objective=objective,
            )
            return self._coerce_2d(out), _build_solve_info('SLSQPWindowedStrategy', {}, threshold)
        if isinstance(constraint, JdetConstraint3D):
            from dvfopt.core import iterative_3d

            D, H, W = constraint.shape
            deformation = np.zeros((3, D, H, W), dtype=np.float64)
            deformation[0] = phi_in[0]
            deformation[1] = phi_in[1]
            deformation[2] = phi_in[2]
            if self.enforce_shoelace:
                raise ValueError(
                    'enforce_shoelace is 2D-only; in 3D the geometric cell-volume '
                    "condition is served by the simplex (3D) constraint family ('simplex_3d')."
                )
            out = iterative_3d(
                deformation,
                threshold=threshold,
                verbose=verbose,
                max_iterations=self.max_iterations,
                max_minimize_iter=self.max_minimize_iter,
                enforce_injectivity=self.enforce_injectivity,
                injectivity_threshold=self.injectivity_threshold,
                objective=objective,
            )
            return out, _build_solve_info('SLSQPWindowedStrategy', {}, threshold)
        if isinstance(
            constraint,
            (SimplexConstraint2D, SimplexConstraint2DFullCoverage, SimplexConstraint2DBilinear),
        ):
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
                enforce_shoelace=self.enforce_shoelace,
                enforce_injectivity=self.enforce_injectivity,
                injectivity_threshold=self.injectivity_threshold,
                objective=objective,
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


@register_strategy('slsqp_3d_tet')
@dataclass
class SLSQPFullGrid3DStrategy(Strategy):
    """Full-grid SLSQP for the 3D 6-tetrahedron constraint.

    3D analogue of :class:`SLSQPFullGridStrategy` — uses
    ``SimplexConstraint3D.jacobian`` (sparse forward Jacobian wired in PR
    #12) to drive ``scipy.optimize.minimize(method='SLSQP')``.

    .. warning::
        **Scaling.** 3D SLSQP does not scale to realistic registration
        problems. The constraint vector grows as
        ``6 * (D-1)(H-1)(W-1)`` — at a 32×32×32 voxel grid that's
        178k constraints, and SLSQP's active-set QP step becomes the
        bottleneck. Prefer :class:`dvfopt.strategies.BarrierStrategy`
        on any non-tiny 3D problem. This strategy exists for symmetry
        with the 2D path and for tiny-grid debugging where KKT
        semantics matter.
    """

    max_iter: int = 50
    ftol: float = 1e-8

    supports_3d: bool = True
    accepts_constraints = (SimplexConstraint3D,)

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        **_,
    ):
        from dvfopt.core.slsqp_fullgrid.tet3d import iterative_3d_tet_slsqp

        self._check_constraint(constraint)
        out = iterative_3d_tet_slsqp(
            phi_in,
            threshold=threshold,
            max_iter=self.max_iter,
            ftol=self.ftol,
            objective=objective,
            verbose=verbose,
            record_history=record_history,
        )
        raw_history = out[1] if record_history else None
        phi_out, info = self._finish(out, record_history, threshold, wrap_history=True)
        _lift_slsqp_trace(raw_history, info)
        return phi_out, info


__all__ = ['SLSQPFullGrid3DStrategy', 'SLSQPFullGridStrategy', 'SLSQPWindowedStrategy']
