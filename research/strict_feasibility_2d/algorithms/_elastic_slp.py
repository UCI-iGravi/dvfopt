"""PROTOTYPE (Part XXI option C): elastic Sl1LP — seedless trust-region SLP.

Fletcher-style exact-penalty SLP: slack the linearized 2-tri constraints
(`T >= tau - s, s >= 0`) and minimize `mu * 1^T s + ||phi - phi_in||_1`
inside a trust region. The LP is feasible BY CONSTRUCTION, so it can start
from the raw folded input — no m14 seed (the profiled per-cluster
bottleneck). With `mu` large enough this is an exact penalty method: once
a feasible point exists inside the trust region the solution drives s to 0
and reduces to the ordinary L1 LP step.

Acceptance is on EXACT (non-linearized) areas: accept iff total exact
violation strictly decreases (or, once feasible, iff L1 decreases while
staying feasible); shrink the trust radius otherwise.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from dvfopt.core.slp.tri_linearize import linearize_T_2tri
from dvfopt.core.tri_primitives import tri_areas_flat


def solve_elastic_l1_lp_step(
    *,
    phi_in_flat,
    phi_lin_flat,
    T_lin,
    J_sparse,
    threshold,
    trust_radius,
    mu=100.0,
    active_margin=1.0,
):
    """One elastic LP step. Variables ``x = [phi (n), t (n), s (Ka)]``.

    Only rows with ``T_lin < threshold + active_margin`` get slacks/rows
    (inactive triangles can't approach the bound within a bounded step).
    """
    n = phi_in_flat.size
    act = np.where(T_lin < threshold + active_margin)[0]
    Ka = act.size
    J_csr = J_sparse.tocsr()[act]

    # Objective: min 1^T t + mu * 1^T s.
    c = np.concatenate([np.zeros(n), np.ones(n), mu * np.ones(Ka)])

    blocks, rhs = [], []
    # L1 epigraph.
    blocks.append(sp.hstack([sp.eye(n), -sp.eye(n), sp.csr_matrix((n, Ka))]))
    rhs.append(phi_in_flat)
    blocks.append(sp.hstack([-sp.eye(n), -sp.eye(n), sp.csr_matrix((n, Ka))]))
    rhs.append(-phi_in_flat)
    # Elastic linearized triangles:  -J phi - s <= -thr + T_lin - J phi_lin.
    if Ka > 0:
        rhs_tri = -threshold + T_lin[act] - J_csr @ phi_lin_flat
        blocks.append(sp.hstack([-J_csr, sp.csr_matrix((Ka, n)), -sp.eye(Ka)]))
        rhs.append(rhs_tri)
    A_ub = sp.vstack(blocks).tocsr()
    b_ub = np.concatenate(rhs)
    # Trust region via variable bounds (tighter than extra rows).
    lo = phi_lin_flat - trust_radius
    hi = phi_lin_flat + trust_radius
    bounds = (
        [(float(lo[i]), float(hi[i])) for i in range(n)]
        + [(0.0, None)] * n
        + [(0.0, None)] * Ka
    )
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        return None, {'success': False, 'message': str(res.message)}
    return res.x[:n], {'success': True, 'n_active': int(Ka)}


def elastic_slp_iter(
    phi_in_2hw,
    *,
    threshold=0.01,
    mu=100.0,
    trust_radius_0=2.0,
    max_iter=30,
    verbose=0,
):
    """Seedless elastic SLP from the RAW (folded) input.

    Returns ``(phi_out_2hw, info)`` with the same conventions as
    ``slp_iter``. ``info['feasible']`` is on exact areas at ``threshold``.
    """
    H, W = phi_in_2hw.shape[1:]
    anchor = np.concatenate([phi_in_2hw[0].ravel(), phi_in_2hw[1].ravel()])
    cur = anchor.copy()
    thr_in = threshold + 1e-4  # inner margin, same as slp_iter

    def _viol(flat):
        T = tri_areas_flat(flat, H, W)
        return float(np.maximum(0.0, thr_in - T).sum()), float(T.min())

    viol_cur, min_cur = _viol(cur)
    l1_cur = 0.0
    trust = trust_radius_0
    n_lp = 0
    for it in range(max_iter):
        T_lin, J = linearize_T_2tri(cur, H, W)
        cand, st = solve_elastic_l1_lp_step(
            phi_in_flat=anchor, phi_lin_flat=cur, T_lin=T_lin, J_sparse=J,
            threshold=thr_in, trust_radius=trust, mu=mu,
        )
        n_lp += 1
        if cand is None:
            trust *= 0.5
            if trust < 1e-4:
                break
            continue
        viol_new, min_new = _viol(cand)
        l1_new = float(np.abs(cand - anchor).sum())
        if viol_cur > 0:
            ok = viol_new < viol_cur * (1 - 1e-9)
        else:
            ok = viol_new == 0.0 and l1_new < l1_cur - 1e-9
        if verbose:
            print(f'  [elastic {it}] viol {viol_cur:.4e}->{viol_new:.4e} '
                  f'minT {min_cur:+.4f}->{min_new:+.4f} L1 {l1_new:.1f} '
                  f'trust={trust:.3f} {"ACC" if ok else "rej"}', flush=True)
        if ok:
            cur, viol_cur, min_cur, l1_cur = cand, viol_new, min_new, l1_new
            trust = min(trust * 2.0, trust_radius_0)
            if viol_cur == 0.0 and it > 0:
                # feasible; one more L1-shrink round then stop on stagnation
                if l1_new >= l1_cur - 1e-6:
                    break
        else:
            trust *= 0.5
            if trust < 1e-4:
                break

    out = np.stack([cur[: H * W].reshape(H, W), cur[H * W:].reshape(H, W)])
    T = tri_areas_flat(cur, H, W)
    info = {
        'feasible': bool(T.min() >= threshold),
        'n_neg': int((T <= 0).sum()),
        'min_T': float(T.min()),
        'L1_dev': float(np.abs(cur - anchor).sum()),
        'n_lp': n_lp,
    }
    return out, info


__all__ = ['elastic_slp_iter', 'solve_elastic_l1_lp_step']
