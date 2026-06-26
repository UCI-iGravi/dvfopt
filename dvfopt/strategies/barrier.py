"""Penalty -> log-barrier L-BFGS-B strategy.

The "workhorse" for moderate-density problems. Works for every concrete
:class:`Constraint` subclass — 2D 2-tri, 2D Jdet, 3D Jdet — because the
underlying :func:`dvfopt.core._barrier_core.run_penalty_barrier_lbfgs`
only requires ``constraint.values`` + ``constraint.adjoint``.
"""

from __future__ import annotations

from dataclasses import dataclass

from dvfopt.exceptions import IncompatibleConstraintError
from dvfopt.strategies.base import Strategy, _build_solve_info, register_strategy


@register_strategy('barrier')
@dataclass
class BarrierStrategy(Strategy):
    """Penalty -> log-barrier L-BFGS-B."""

    margin: float = 1e-3
    lam_schedule: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
    mu_schedule: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)
    max_iter: int = 300
    # Sparsify the barrier-phase gradient co-vector by dropping tets whose
    # barrier pressure is below this fraction of the max (far above the
    # threshold). Lets the adjoint early-exit fire (~5-9x faster gradient).
    # Feasibility is unaffected (the inf-guard uses full slack). 0 = exact.
    barrier_grad_rtol: float = 0.0

    supports_3d: bool = True

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
        from dvfopt.core._barrier_core import run_penalty_barrier_lbfgs

        self._check_constraint(constraint)
        phi_flat = constraint.flatten(phi_in)
        phi_anchor = phi_flat.copy()
        anchor_kind = objective.label or 'l2'
        anchor_eps = getattr(objective, 'eps', 1e-4)

        # Wrap the caller's ``step_callback`` so the inner barrier core
        # (which works in flat decision-vector layout) emits ``{'phi':
        # (C, *shape), 'stage': name}`` like the other strategies. The
        # constraint owns the unflatten convention.
        if step_callback is None:
            inner_cb = None
        else:
            def inner_cb(state):
                phi_arr = constraint.unflatten(state['phi_flat'])
                step_callback({'phi': phi_arr, 'stage': state['stage']})

        out_flat, info = run_penalty_barrier_lbfgs(
            phi_flat,
            phi_anchor,
            constraint_values=constraint.values,
            constraint_adjoint=constraint.adjoint,
            threshold=threshold,
            margin=self.margin,
            lam_schedule=self.lam_schedule,
            mu_schedule=self.mu_schedule,
            max_iter=self.max_iter,
            anchor=anchor_kind,
            eps_l1=anchor_eps,
            barrier_grad_rtol=self.barrier_grad_rtol,
            verbose=verbose,
            record_history=record_history,
            step_callback=inner_cb,
        )
        return constraint.unflatten(out_flat), _build_solve_info('BarrierStrategy', info, threshold)


@register_strategy('barrier_tet_3d_torch')
@dataclass
class BarrierTet3DTorchStrategy(Strategy):
    """GPU (CUDA) penalty → log-barrier solver for the 3D 6-tet constraint.

    Wraps :func:`dvfopt.core.iterative3d_tet_barrier_torch.iterative_3d_tet_barrier_torch`
    — an on-device (torch.optim.LBFGS) version of the penalty→barrier
    homotopy. Use it for LARGE dense-fold bands where the active-band crop
    trick can't help (folds span the region) and the CPU barrier is slow.

    Measured ~1.4-1.9x vs the (already parallel-kernel) CPU barrier on a
    16x64x64 / 9182-fold field (CPU 53.6 s → GPU f64 39.5 s → GPU f32
    28.9 s), growing on larger fields; the torch.optim.LBFGS Python loop
    caps the win below the raw-kernel ratio. float32 matched float64
    feasibility at the 0.01 threshold in testing, but float64 is the
    safe default here.

    Falls back to CPU automatically if CUDA is unavailable (the torch
    solver resolves ``device`` itself).

    Parameters
    ----------
    margin : float, default 1e-3
    max_iter : int, default 200       per-phase L-BFGS cap
    anchor_override : str | None      anchor kind; None uses the objective's label
    device : str | None               'cuda'/'cpu'; None = auto
    dtype : {'float64', 'float32'}, default 'float64'
        float64 is safest near the threshold; float32 is faster and
        matched feasibility in testing.
    windowed : bool, default False     per-fold-cluster patches with frozen ring
    pad : int, default 2               patch pad when windowed
    """

    margin: float = 1e-3
    max_iter: int = 200
    anchor_override: str | None = None
    device: str | None = None
    dtype: str = 'float64'
    windowed: bool = False
    pad: int = 2

    supports_3d: bool = True

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
        import numpy as np

        from dvfopt.constraints import Tet6Constraint3D
        from dvfopt.core.iterative3d_tet_barrier_torch import (
            iterative_3d_tet_barrier_torch,
        )

        if not isinstance(constraint, Tet6Constraint3D):
            raise IncompatibleConstraintError(
                'BarrierTet3DTorchStrategy requires Tet6Constraint3D, got '
                f'{type(constraint).__name__}'
            )
        import torch

        torch_dtype = torch.float64 if self.dtype == 'float64' else torch.float32
        anchor_kind = self.anchor_override or objective.label or 'l2'
        out = iterative_3d_tet_barrier_torch(
            np.asarray(phi_in, dtype=np.float64),
            threshold=threshold,
            margin=self.margin,
            max_iter=self.max_iter,
            anchor=anchor_kind,
            eps_l1=getattr(objective, 'eps', 1e-4),
            device=self.device,
            dtype=torch_dtype,
            windowed=self.windowed,
            pad=self.pad,
            verbose=verbose,
            record_history=record_history,
        )
        if record_history:
            phi_out, history = out
        else:
            phi_out, history = out, {}
        phi_out = np.asarray(phi_out, dtype=np.float64)
        return phi_out, _build_solve_info(
            'BarrierTet3DTorchStrategy', history, threshold
        )


__all__ = ['BarrierStrategy', 'BarrierTet3DTorchStrategy']
