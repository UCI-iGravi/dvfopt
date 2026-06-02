"""NMVF heuristic strategy — Neighborhood Mean Vector Filter.

The original method this package was built around (see README): a fast
non-optimisation heuristic that iteratively smooths the displacement
field around fold cores. It's a lossy first-pass tool; for accuracy
prefer the parameterized solvers.

The Strategy wrapper is provided so NMVF composes cleanly with the
``correct_dvf`` / ``Solver`` API like any other strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.constraints import JdetConstraint2D
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


@register_strategy('nmvf')
@dataclass
class NMVFStrategy(Strategy):
    """Heuristic Neighborhood Mean Vector Filter — 2D Jdet fold smoother.

    Iteratively replaces each pixel within a 3x3 neighborhood of every
    folded pixel (Jdet ≤ 0) with the local mean displacement vector,
    until ``min(J) > 0`` or ``max_iter`` is reached.

    Characteristics
    ---------------

    * **Fast**: each iteration is a handful of array slices + a
      mean, no optimisation.
    * **Lossy**: doesn't minimise anything. L2 / L1 deviation from the
      input is typically much larger than what the SLSQP / barrier /
      wallbreaker strategies produce.
    * **2D only**: uses :func:`jacobian_det2D` to detect folds.
    * **No convergence guarantee**: dense folds can keep regenerating
      after each smoothing pass.

    Use cases
    ---------

    * Quick first-pass smoother when speed dominates accuracy.
    * Cases with sparse, isolated folds caused by single-pixel outliers.
    * As a baseline comparator against the optimisation-based strategies.
    """

    max_iter: int = 1000

    supports_3d: bool = False
    accepts_constraints = (JdetConstraint2D,)

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        step_callback=None,
        **_,
    ):
        from dvfopt.core._nmvf import nmvf_correct_2d

        self._check_constraint(constraint)
        # NMVF doesn't use the objective — it's heuristic, no anchor.
        # ``threshold`` is also unused (it just chases min_J > 0). We
        # accept both for API uniformity.
        out = nmvf_correct_2d(
            phi_in,
            max_iter=self.max_iter,
            verbose=verbose,
            record_history=record_history,
            step_callback=step_callback,
        )
        if record_history:
            phi_out, info = out
            return phi_out, _build_solve_info('NMVFStrategy', info, threshold)
        return out, _build_solve_info('NMVFStrategy', {}, threshold)


__all__ = ['NMVFStrategy']
