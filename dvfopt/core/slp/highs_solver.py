"""LP step solver: ``min ||phi - phi_in||_1`` subject to linearised
simplex (2D) constraint + optional trust region. Backend is HiGHS via
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

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Constant LP components, cached per problem size. Across slp_iter iterations
# on a fixed crop only the -J data and the b_ub vector change; the objective,
# the L1-epigraph blocks, the trust identity blocks, the tri zero-pad block
# and the variable bounds are all shape-only. Cached ndarrays are marked
# read-only (linprog copies its inputs; nothing downstream may mutate them).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _cost_and_bounds(n: int):
    """Objective vector and variable bounds for size ``n`` (= 2HW).

    ``bounds`` is an ``(2n, 2)`` float ndarray using ``+/-inf`` for the
    unbounded sides — verified to produce results identical to the
    list-of-``(None, None)`` tuples form with ``method='highs'``.
    """
    c = np.concatenate([np.zeros(n), np.ones(n)])
    bounds = np.empty((2 * n, 2), dtype=np.float64)
    bounds[:n, 0] = -np.inf  # phi unbounded
    bounds[:n, 1] = np.inf
    bounds[n:, 0] = 0.0  # t >= 0
    bounds[n:, 1] = np.inf
    c.flags.writeable = False
    bounds.flags.writeable = False
    return c, bounds


@lru_cache(maxsize=32)
def _epigraph_blocks(n: int):
    """L1-epigraph rows: ``[I, -I]`` and ``[-I, -I]``."""
    A1 = sp.hstack([sp.eye(n), -sp.eye(n)])
    A2 = sp.hstack([-sp.eye(n), -sp.eye(n)])
    return A1, A2


@lru_cache(maxsize=32)
def _trust_blocks(n: int):
    """Trust-region rows: ``[I, 0]`` and ``[-I, 0]``."""
    A4 = sp.hstack([sp.eye(n), sp.csr_matrix((n, n))])
    A5 = sp.hstack([-sp.eye(n), sp.csr_matrix((n, n))])
    return A4, A5


@lru_cache(maxsize=32)
def _tri_zero_pad(K: int, n: int):
    """The ``(K, n)`` zero pad for the triangle block's t-columns."""
    return sp.csr_matrix((K, n))


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

    # Objective (min sum(t)) + bounds (phi unbounded, t >= 0): shape-only.
    c, bounds = _cost_and_bounds(n)

    blocks = []
    b_ub_blocks = []

    # 1) L1 epigraph upper:  phi - t <= phi_in
    # 2) L1 epigraph lower: -phi - t <= -phi_in
    A1, A2 = _epigraph_blocks(n)
    blocks.append(A1)
    b_ub_blocks.append(phi_in_flat)
    blocks.append(A2)
    b_ub_blocks.append(-phi_in_flat)

    # 3) Linearised triangle: -J @ phi <= -threshold + T_lin - J @ phi_lin
    if K > 0:
        rhs_tri = -threshold + T_lin - J_csr @ phi_lin_flat
        A3 = sp.hstack([-J_csr, _tri_zero_pad(K, n)])
        blocks.append(A3)
        b_ub_blocks.append(rhs_tri)

    # 4) Optional trust region: -trust <= phi - phi_lin <= +trust
    if trust_radius is not None:
        A4, A5 = _trust_blocks(n)
        blocks.append(A4)
        b_ub_blocks.append(phi_lin_flat + trust_radius)
        blocks.append(A5)
        b_ub_blocks.append(-phi_lin_flat + trust_radius)

    A_ub = sp.vstack(blocks).tocsr()
    b_ub = np.concatenate(b_ub_blocks)

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
