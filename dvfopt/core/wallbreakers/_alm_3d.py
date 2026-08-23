"""Augmented Lagrangian method (ALM) for the 3D 6-tetrahedron constraint.

3D analog of :mod:`dvfopt.core.wallbreakers._alm` (the 2D 2-triangle
ALM). Uses the PHR (Powell-Hestenes-Rockafellar) augmented Lagrangian
with L-BFGS-B inner — unlike SLSQP, ALM does **not** solve a QP and is
therefore not subject to the active-set degeneracy at the constraint
wall.

For inequality constraints ``V_k(phi) >= tau`` rewrite as slacks
``V_k - tau >= 0`` and minimise::

    L_rho(phi, mu) = f(phi) + (1/(2 rho)) * sum (psi^2 - mu^2)
    psi_i = max(0, mu_i - rho (V_i - tau))

with ``f = 0.5 * ||phi - phi_in||^2`` (or smoothed L1). The outer loop
updates the Lagrange multiplier ``mu_i <- max(0, mu_i - rho (V_i - tau))``;
``rho`` grows only when the constraint violation fails to halve
(Birgin-Martinez safeguard).

Same approach + the same shaped helpers as the 2D version, just with
the 6-tet primitives in place of 2-tri.

Used as step 2 of the full m10-3D pipeline (harmonic seed → ALM →
barrier polish). For now it's also useful as a standalone alternative
to barrier on dense-fold 3D cases — wherever the barrier's penalty
phase stalls.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._logging import log_info
from dvfopt.jacobian.tetrahedron_sign import tet_grad_T_v, tet_volumes_flat
from dvfopt.objectives import L2Objective, Objective, _kind_eps


def _alm_objective_3d(
    phi_flat, phi_init_flat, D, H, W, threshold, mu, rho, anchor='l2', eps_l1=1e-4
):
    """PHR-ALM objective + gradient on the 6-tet constraint."""
    diff = phi_flat - phi_init_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    elif anchor == 'l1':
        s_anc = np.sqrt(diff * diff + eps_l1 * eps_l1)
        val = float((s_anc - eps_l1).sum())
        grad = diff / s_anc
    else:
        val = 0.0
        grad = np.zeros_like(diff)

    V = tet_volumes_flat(phi_flat, D, H, W)
    slack = V - threshold
    psi = np.maximum(0.0, mu - rho * slack)
    val += float((psi * psi - mu * mu).sum()) / (2.0 * rho)
    grad = grad - tet_grad_T_v(phi_flat, D, H, W, psi)
    return val, grad


def augmented_lagrangian_3d(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    objective: Objective | None = None,
    phi_anchor: Optional[np.ndarray] = None,
    rho_init: float = 1.0,
    rho_growth: float = 5.0,
    rho_max: float = 1e8,
    outer_max: int = 60,
    inner_maxiter: int = 200,
    ftol_inner: float = 1e-10,
    gtol_inner: float = 1e-7,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
):
    """PHR augmented Lagrangian; stops when feasible at ``threshold + margin``.

    Parameters
    ----------
    phi_in : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    threshold : float, optional
        Lower bound on per-tet volume.
    margin : float
        Safety margin above ``threshold``.
    objective : Objective or None
        Data term; ``None`` (default) means
        :class:`~dvfopt.objectives.L2Objective`.
        :class:`~dvfopt.objectives.NoneObjective` skips the anchor — just push for
        feasibility (cheapest L2 unconstrained → harmonic-like result).
    phi_anchor : ndarray or None
        The reference field for the anchor term — distance is measured
        against this, not ``phi_in``. Default ``None`` uses ``phi_in``
        for both starting iterate and anchor (the standalone case).
        Set explicitly when chaining stages (m10-3D harmonic→ALM:
        ALM starts FROM the harmonic seed but anchors TO the original
        input).
    rho_init, rho_growth, rho_max : float
        Penalty parameter schedule (Birgin-Martinez safeguard).
    outer_max : int
        Maximum outer ALM iterations.
    inner_maxiter, ftol_inner, gtol_inner : float
        L-BFGS-B inner-solver knobs.
    time_budget_s : float
        Wall-clock budget; early-exit if exceeded.
    verbose, record_history : int / bool
        Logging + history controls.

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``.
    info : dict, only if ``record_history=True``.
    """
    from dvfopt._defaults import DEFAULT_PARAMS

    objective = objective or L2Objective()
    anchor, eps_l1 = _kind_eps(objective)
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    _, D, H, W = phi_in.shape
    # Phi pack: [dx, dy, dz] (DX_FIRST) — matches Tet6Constraint3D.
    phi_init = phi_in.copy()
    phi_init_flat = np.concatenate([phi_init[2].ravel(), phi_init[1].ravel(), phi_init[0].ravel()])
    # Anchor reference (distance is measured against this, not phi_init).
    # Defaults to phi_init for the standalone case; M10TetStrategy passes
    # the user's original input when chaining harmonic → ALM.
    if phi_anchor is None:
        phi_anchor_flat = phi_init_flat
    else:
        phi_anchor_flat = np.concatenate(
            [phi_anchor[2].ravel(), phi_anchor[1].ravel(), phi_anchor[0].ravel()]
        )
    n_constr = 6 * (D - 1) * (H - 1) * (W - 1)

    mu = np.zeros(n_constr)
    rho = rho_init
    phi_flat = phi_init_flat.copy()

    target = threshold + margin
    last_min = -np.inf
    log = []
    t0 = time.time()

    for outer in range(outer_max):
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _alm_objective_3d,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_anchor_flat, D, H, W, threshold, mu, rho, anchor, eps_l1),
            options=dict(maxiter=inner_maxiter, ftol=ftol_inner, gtol=gtol_inner),
        )
        phi_flat = res.x

        V = tet_volumes_flat(phi_flat, D, H, W)
        slack = V - threshold
        min_T = float(V.min())
        # ``n_neg`` is the canonical "tets with V <= 0" — same definition
        # the other wallbreaker stages use. Including it on every log entry
        # lets ``SolveInfo.from_legacy_history`` detect the
        # feasibility-transition outer round (which appears as
        # ``feasible_after_phase`` on the resulting :class:`SolveInfo`).
        n_neg = int((V <= 0).sum())
        mu = np.maximum(0.0, mu - rho * slack)
        viol_now = float(np.maximum(0.0, target - V).max())
        if outer > 0 and viol_now > 0.5 * log[-1]['viol']:
            rho = min(rho_max, rho * rho_growth)

        log.append(
            dict(
                outer=outer,
                inner_nit=int(res.nit),
                n_neg=n_neg,
                min_T=min_T,
                viol=viol_now,
                rho=rho,
                mu_max=float(mu.max()),
                wall=time.time() - t0,
            )
        )
        if verbose:
            log_info(
                f'  ALM out={outer:3d}  inner={res.nit:4d}  '
                f'min_V={min_T:+.5f}  viol={viol_now:.3e}  '
                f'rho={rho:.1e}  ({time.time() - t0:.1f}s)',
            )
        last_min = min_T
        if min_T >= target:
            break

    # Unpack flat [dx, dy, dz] back to (3, D, H, W) [dz, dy, dx].
    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    phi_out = np.stack([dz, dy, dx])

    info = dict(
        outer_used=len(log),
        rho_final=rho,
        min_T_final=last_min,
        feasible=last_min >= target,
        log_first5=log[:5],
        log_last5=log[-5:],
    )

    if record_history:
        return phi_out, info
    return phi_out


__all__ = ['augmented_lagrangian_3d']
