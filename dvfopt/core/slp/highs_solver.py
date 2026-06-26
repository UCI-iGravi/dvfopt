"""LP step solver: ``min ||phi - phi_in||_1`` subject to linearised
2-tri constraint + optional trust region. Backend is HiGHS via
``scipy.optimize.linprog(method='highs')``.

L1 epigraph reformulation
-------------------------
Decision vector: ``x = [phi (2HW), t (2HW)]``.

Objective:      ``min c^T x``,  ``c = [zeros(2HW), ones(2HW)]``.
L1 epigraph:    ``phi - t <= phi_in``  and  ``-phi - t <= -phi_in``.
Triangle (lin): ``-J @ phi <= -threshold + T_lin - J @ phi_lin``.
Trust region:   ``phi - phi_lin in [-trust, +trust]``  (only if set).
``t`` bounds:   ``t >= 0`` (no upper bound).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def solve_l1_lp_step(
    *,
    phi_in_flat: np.ndarray,
    phi_lin_flat: np.ndarray,
    T_lin: np.ndarray,
    J_sparse: sp.spmatrix,
    threshold: float,
    trust_radius: float | None = None,
):
    """One LP iteration.

    Parameters
    ----------
    phi_in_flat : (2HW,) float64
        Anchor field for the L1 objective.
    phi_lin_flat : (2HW,) float64
        Linearisation point. Equal to ``phi_in_flat`` for ``lp_oneshot``;
        equal to the current SLP iterate for ``slp_iter``.
    T_lin : (K,) float64
        ``T(phi_lin_flat)`` from ``linearize_T_2tri``.
    J_sparse : (K, 2HW) sparse
        ``dT/dphi`` at ``phi_lin_flat``.
    threshold : float
        Lower bound on each triangle area.
    trust_radius : float or None
        L-inf box around ``phi_lin`` for SLP. ``None`` means unbounded
        (lp_oneshot path).

    Returns
    -------
    phi_out_flat : (2HW,)
    status : dict with keys ``success``, ``message``, ``fun``,
        ``status_code``, ``nit``.
    """
    n = phi_in_flat.size  # 2HW
    K = T_lin.size
    J_csr = J_sparse.tocsr() if not sp.isspmatrix_csr(J_sparse) else J_sparse

    # Objective: min sum(t).
    c = np.concatenate([np.zeros(n), np.ones(n)])

    blocks = []
    b_ub_blocks = []

    # 1) L1 epigraph upper:  phi - t <= phi_in
    A1 = sp.hstack([sp.eye(n), -sp.eye(n)])
    blocks.append(A1); b_ub_blocks.append(phi_in_flat)
    # 2) L1 epigraph lower: -phi - t <= -phi_in
    A2 = sp.hstack([-sp.eye(n), -sp.eye(n)])
    blocks.append(A2); b_ub_blocks.append(-phi_in_flat)

    # 3) Linearised triangle: -J @ phi <= -threshold + T_lin - J @ phi_lin
    if K > 0:
        rhs_tri = -threshold + T_lin - J_csr @ phi_lin_flat
        A3 = sp.hstack([-J_csr, sp.csr_matrix((K, n))])
        blocks.append(A3); b_ub_blocks.append(rhs_tri)

    # 4) Optional trust region: -trust <= phi - phi_lin <= +trust
    if trust_radius is not None:
        A4 = sp.hstack([sp.eye(n), sp.csr_matrix((n, n))])
        blocks.append(A4); b_ub_blocks.append(phi_lin_flat + trust_radius)
        A5 = sp.hstack([-sp.eye(n), sp.csr_matrix((n, n))])
        blocks.append(A5); b_ub_blocks.append(-phi_lin_flat + trust_radius)

    A_ub = sp.vstack(blocks).tocsr()
    b_ub = np.concatenate(b_ub_blocks)

    # Bounds: phi unbounded, t >= 0.
    bounds = [(None, None)] * n + [(0.0, None)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    status = {
        'success': bool(result.success),
        'message': str(result.message),
        'fun': float(result.fun) if result.fun is not None else None,
        'status_code': int(result.status),
        'nit': int(getattr(result, 'nit', -1)),
    }
    if result.success:
        phi_out = result.x[:n]
    else:
        phi_out = phi_lin_flat.copy()
    return phi_out, status
