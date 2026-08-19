"""SLSQP strategies — full-grid (2-tri / 6-tet) and windowed (Jdet)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from dvfopt.constraints import (
    JdetConstraint2D,
    JdetConstraint3D,
    Tet6Constraint3D,
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
        return self._finish(out, record_history, threshold, wrap_history=True)


@register_strategy('slsqp_windowed')
@dataclass
class SLSQPWindowedStrategy(Strategy):
    """Windowed iterative SLSQP — the legacy Jdet path.

    Finds the worst-Jdet voxel/pixel, builds a bbox + 1-cell ring,
    solves the local SLSQP subproblem with frozen edges, repeats. Also
    supports the 2-triangle constraint via ``enforce_triangles=True``.

    .. note::
        **Objective contract.** The delegated windowed solvers
        (``iterative_serial`` / ``iterative_3d``) hard-code an **L2**
        anchor objective (``objective_euc``); the composed
        :class:`~dvfopt.objectives.Objective` is **not** plumbed
        through. Composing anything other than
        :class:`~dvfopt.objectives.L2Objective` /
        :class:`~dvfopt.objectives.NoneObjective` with this strategy
        emits a :class:`UserWarning` and the composed objective is
        ignored. Use :class:`SLSQPFullGridStrategy`,
        :class:`~dvfopt.strategies.SLPStrategy`, or
        :class:`~dvfopt.strategies.BarrierStrategy` if you need an L1
        anchor.
    """

    max_iterations: int = 80
    max_minimize_iter: int = 120
    # Extra constraint modes. 2D supports both flags; 3D supports
    # enforce_injectivity (axial monotonicity, linear rows). The 3D
    # analogue of enforce_shoelace is the 6-tet constraint family.
    # NOTE on injectivity_threshold=None semantics: the 2D path runs the
    # adaptive tau-doubling loop (doubling until globally injective);
    # the 3D path simply defaults the gap bound to `threshold` — no
    # adaptive loop and no global-injectivity certificate.
    enforce_shoelace: bool = False
    enforce_injectivity: bool = False
    injectivity_threshold: float | None = None

    supports_3d: bool = True

    def solve(
        self, phi_in, *, constraint, objective, threshold, verbose=0, record_history=False, **_
    ):
        _label = getattr(objective, 'label', None) if objective is not None else None
        if objective is not None and _label not in ('l2', 'none'):
            warnings.warn(
                'SLSQPWindowedStrategy optimises an L2 objective; the composed '
                f"'{_label or type(objective).__name__}' objective is ignored.",
                UserWarning,
                stacklevel=2,
            )
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
                    "condition is served by the 6-tet constraint family ('6tet')."
                )
            out = iterative_3d(
                deformation,
                threshold=threshold,
                verbose=verbose,
                max_iterations=self.max_iterations,
                max_minimize_iter=self.max_minimize_iter,
                enforce_injectivity=self.enforce_injectivity,
                injectivity_threshold=self.injectivity_threshold,
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
                enforce_shoelace=self.enforce_shoelace,
                enforce_injectivity=self.enforce_injectivity,
                injectivity_threshold=self.injectivity_threshold,
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
    ``Tet6Constraint3D.jacobian`` (sparse forward Jacobian wired in PR
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
    accepts_constraints = (Tet6Constraint3D,)

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
        from dvfopt.core.iterative3d_tet_slsqp import iterative_3d_tet_slsqp

        self._check_constraint(constraint)
        out = iterative_3d_tet_slsqp(
            phi_in,
            threshold=threshold,
            max_iter=self.max_iter,
            ftol=self.ftol,
            anchor=objective.label or 'l2',
            eps_l1=getattr(objective, 'eps', 1e-4),
            verbose=verbose,
            record_history=record_history,
        )
        return self._finish(out, record_history, threshold, wrap_history=True)


__all__ = ['SLSQPFullGrid3DStrategy', 'SLSQPFullGridStrategy', 'SLSQPWindowedStrategy']
