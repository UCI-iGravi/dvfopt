"""Shared penalty -> log-barrier homotopy core for the L-BFGS-B solvers.

All four CPU barrier solvers (``iterative2d_barrier``, ``iterative2d_tri_barrier``,
``iterative3d_barrier``, plus their patch-level helpers) run the same
two-phase continuation:

* Phase 1 (penalty)::

      F_pen(phi) = anchor(phi - phi_anchor) + lam * sum_active max(0, target - T_k)^2

  iterated over an increasing ``lam_schedule`` until ``min T_k >= target``.

* Phase 2 (barrier)::

      F_bar(phi) = anchor(phi - phi_anchor) - mu * sum_active log(T_k - threshold)

  iterated over a decreasing ``mu_schedule`` to polish the iterate inside
  the feasible interior.

The constraint ``T(phi)`` and its adjoint ``J^T @ v`` differ per solver,
so this core takes them as callables. The anchor term is an
:class:`~dvfopt.objectives.Objective` (``L2`` / ``L1`` / ``None``), so
every solver can trivially gain a non-L2 anchor.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._logging import log_info, log_warning
from dvfopt.objectives import L2Objective, Objective

# Schedules shared across all CPU barrier solvers.
DEFAULT_LAM_SCHEDULE: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
DEFAULT_MU_SCHEDULE: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)


def _penalty_objective(
    phi_flat,
    lam,
    *,
    phi_anchor,
    constraint_values,
    constraint_adjoint,
    target,
    active_mask,
    objective,
):
    """Smooth quadratic exterior penalty on negative-T cells."""
    diff = phi_flat - phi_anchor
    val, grad = objective(diff)
    T = constraint_values(phi_flat)
    viol = np.maximum(0.0, target - T)
    if active_mask is not None:
        viol = viol * active_mask
    if np.any(viol > 0):
        val += lam * float(np.dot(viol, viol))
        # dF/dT_i = -2 lam viol_i; chain via J^T.
        grad = grad - 2.0 * lam * constraint_adjoint(phi_flat, viol)
    return val, grad


def _barrier_objective(
    phi_flat,
    mu,
    *,
    phi_anchor,
    constraint_values,
    constraint_adjoint,
    threshold,
    active_mask,
    objective,
    grad_rtol=0.0,
):
    """Log-barrier interior penalty. Returns ``(+inf, zeros)`` on infeasible iterates
    so L-BFGS-B's line search rejects the step.

    ``grad_rtol`` (default 0 = exact): if > 0, the barrier gradient
    co-vector ``dF_dT = -mu/slack`` is sparsified by zeroing entries whose
    magnitude is below ``grad_rtol * max|dF_dT|`` — i.e. tets far above the
    threshold (large slack → negligible barrier pressure). This lets the
    adjoint kernel's per-cell early-exit fire, speeding the gradient up to
    ~5-9x on near-feasible polishes. Feasibility is NOT affected: the
    ``slack <= 0 -> +inf`` guard above uses the FULL (un-sparsified) slack,
    so no tet can cross the threshold undetected; only the descent
    direction (and thus the final L2/L1, negligibly) changes."""
    diff = phi_flat - phi_anchor
    val, grad = objective(diff)
    T = constraint_values(phi_flat)
    slack = T - threshold
    if active_mask is not None:
        slack_active = slack[active_mask]
        if np.any(slack_active <= 0.0):
            return np.inf, np.zeros_like(phi_flat)
        val += -mu * float(np.log(slack_active).sum())
        dF_dT = np.zeros_like(slack)
        dF_dT[active_mask] = -mu / slack_active
    else:
        if np.any(slack <= 0.0):
            return np.inf, np.zeros_like(phi_flat)
        val += -mu * float(np.log(slack).sum())
        dF_dT = -mu / slack
    if grad_rtol > 0.0:
        amax = np.abs(dF_dT).max()
        if amax > 0.0:
            dF_dT = np.where(np.abs(dF_dT) >= grad_rtol * amax, dF_dT, 0.0)
    grad = grad + constraint_adjoint(phi_flat, dF_dT)
    return val, grad


def _min_active(T: np.ndarray, active_mask: Optional[np.ndarray]) -> float:
    if active_mask is None:
        return float(T.min())
    return float(T[active_mask].min())


def run_penalty_barrier_lbfgs(
    phi_init_flat: np.ndarray,
    phi_anchor: np.ndarray,
    *,
    constraint_values: Callable[[np.ndarray], np.ndarray],
    constraint_adjoint: Callable[[np.ndarray, np.ndarray], np.ndarray],
    threshold: float,
    margin: float = 1e-3,
    lam_schedule: tuple[float, ...] = DEFAULT_LAM_SCHEDULE,
    mu_schedule: tuple[float, ...] = DEFAULT_MU_SCHEDULE,
    max_iter: int = 300,
    active_mask: Optional[np.ndarray] = None,
    objective: Optional[Objective] = None,
    bounds=None,
    barrier_grad_rtol: float = 0.0,
    verbose: int = 0,
    record_history: bool = False,
    log_prefix: str = '',
    step_callback=None,
):
    """Run the penalty -> log-barrier L-BFGS-B homotopy.

    Parameters
    ----------
    phi_init_flat
        Initial (and default-anchor) flat decision vector.
    phi_anchor
        Anchor for the data term — usually a copy of the initial phi, but
        callers can supply a different reference (e.g. the *outer* iter
        anchor in a multi-pass solver).
    constraint_values
        Callable ``phi_flat -> T`` (1-D array of constraint values).
    constraint_adjoint
        Callable ``(phi_flat, v) -> J^T @ v`` returning a vector of the
        same shape as ``phi_flat``. Used by both phases.
    threshold, margin
        Penalty phase pushes ``T >= threshold + margin``; barrier requires
        ``T > threshold`` strictly.
    active_mask
        Optional bool mask over the constraint vector — only ``True``
        entries are penalised / barrier'd (the windowed 3D solver uses this
        to ignore the patch rim whose one-sided-difference Jdet does not
        match the global field).
    objective
        Anchor term, called as ``objective(phi_flat - phi_anchor)`` ->
        ``(value, gradient)``. ``None`` (default) means
        :class:`~dvfopt.objectives.L2Objective`.

    Returns
    -------
    phi_flat : ndarray
    info : dict with keys ``feasible``, ``lam_steps``, ``mu_steps``, ``history``.
    """
    objective = objective or L2Objective()
    target = threshold + margin
    phi_flat = phi_init_flat.copy()
    history = []
    lam_steps = 0
    mu_steps = 0

    def _fire(stage: str, phi_flat_now: np.ndarray) -> None:
        """Forward a flat-phi snapshot to ``step_callback`` if a caller
        passed one. The barrier core works in the constraint's flat
        decision-vector layout; the caller (typically a Strategy) is
        responsible for any reshape it wants — we pass ``phi_flat`` in
        the payload so the consumer can unflatten with the constraint
        it owns. Buggy callbacks are swallowed; KeyboardInterrupt is
        the documented stop signal and propagates."""
        if step_callback is None:
            return
        try:
            step_callback({'phi_flat': phi_flat_now.copy(), 'stage': stage})
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log_warning(f'step_callback raised {type(exc).__name__}: {exc}; continuing')

    obj_kwargs = dict(
        phi_anchor=phi_anchor,
        constraint_values=constraint_values,
        constraint_adjoint=constraint_adjoint,
        active_mask=active_mask,
        objective=objective,
    )

    # Initial feasibility check.
    T0 = constraint_values(phi_flat)
    feasible = _min_active(T0, active_mask) >= target

    # Phase 1: penalty
    for lam in lam_schedule:
        if feasible:
            break
        t0 = time.time()
        res = minimize(
            lambda p, lam_=lam: _penalty_objective(p, lam_, target=target, **obj_kwargs),
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            # No verbosity option: scipy's 1.15 C rewrite of L-BFGS-B
            # removed 'disp'/'iprint' (passing either raises
            # OptimizeWarning on every call). Progress goes through our
            # own per-phase log lines instead.
            options={'maxiter': max_iter, 'gtol': 1e-6},
        )
        phi_flat = res.x
        lam_steps += 1
        T = constraint_values(phi_flat)
        cur_min = _min_active(T, active_mask)
        cur_neg = int((T <= 0).sum())
        if record_history:
            history.append(
                dict(
                    phase='penalty',
                    step=lam_steps,
                    lam=float(lam),
                    n_neg=cur_neg,
                    min_T=cur_min,
                    wall_s=time.time() - t0,
                )
            )
        if verbose >= 1:
            log_info(
                f'{log_prefix}[penalty {lam_steps}] lam={lam:g}  '
                f'neg={cur_neg}  min_T={cur_min:+.6f}  '
                f'({time.time() - t0:.2f}s)',
            )
        _fire(f'penalty_lam={lam:g}', phi_flat)
        if cur_min >= target:
            feasible = True
            break

    # Phase 2: barrier
    if feasible:
        for mu in mu_schedule:
            t0 = time.time()
            res = minimize(
                lambda p, mu_=mu: _barrier_objective(
                    p, mu_, threshold=threshold, grad_rtol=barrier_grad_rtol, **obj_kwargs
                ),
                phi_flat,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                # No verbosity option: scipy's 1.15 C rewrite of L-BFGS-B
                # removed 'disp'/'iprint' (passing either raises
                # OptimizeWarning on every call). Progress goes through our
                # own per-phase log lines instead.
                options={'maxiter': max_iter, 'gtol': 1e-6},
            )
            # Only accept the step if the barrier objective is finite; an
            # infeasible iterate returns inf and would silently corrupt phi.
            if np.isfinite(res.fun):
                phi_flat = res.x
            mu_steps += 1
            T = constraint_values(phi_flat)
            cur_min = _min_active(T, active_mask)
            cur_neg = int((T <= 0).sum())
            if record_history:
                history.append(
                    dict(
                        phase='barrier',
                        step=mu_steps,
                        mu=float(mu),
                        n_neg=cur_neg,
                        min_T=cur_min,
                        wall_s=time.time() - t0,
                    )
                )
            if verbose >= 1:
                log_info(
                    f'{log_prefix}[barrier {mu_steps}] mu={mu:g}  '
                    f'neg={cur_neg}  min_T={cur_min:+.6f}  '
                    f'({time.time() - t0:.2f}s)',
                )
            _fire(f'barrier_mu={mu:g}', phi_flat)

    return phi_flat, {
        'feasible': feasible,
        'lam_steps': lam_steps,
        'mu_steps': mu_steps,
        'history': history,
    }
