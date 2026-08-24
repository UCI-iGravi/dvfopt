"""Shared elastic trust-region SLP engine for the 2.5D marching repairs.

Factored out of ``_marching_25d._repair_cluster`` (sweep: one frozen plane,
inter-layer simplex (3D) + per-slice simplex (2D) LP blocks) and
``_mop_interior_3d._repair_box`` (mop: frozen rim, simplex (3D)-only LP block with a
simplex (2D) term in the acceptance oracle). The engine owns the trust loop, the LP
assembly from linearized constraint blocks, the ``linprog`` call, the exact
acceptance test and the trust bookkeeping; callers own free-column selection,
Jacobian construction/slicing, the exact-violation definition and the
application of a candidate free vector into their crop.

Deliberately import-light (numpy / scipy.sparse / scipy.optimize.linprog
only): this module is imported by ``ProcessPoolExecutor`` workers on Windows
spawn, so it must stay side-effect-free and cheap.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

#: Active-row window: constraint rows with ``T < thr + ACTIVE_WINDOW`` enter
#: the LP as linearized elastic rows. Callers use this when selecting the
#: active rows they hand to ``blocks_fn``.
ACTIVE_WINDOW = 0.5


def elastic_trust_solve(
    x0,
    anchor,
    blocks_fn,
    viol_fn,
    apply_fn,
    *,
    state,
    mu,
    max_iters,
    trust0=0.5,
    trust_cap=0.5,
    trust_floor=1e-3,
    accept_rtol=1e-9,
    stop_viol=1e-12,
):
    """Shared elastic trust-region SLP engine.

    x0 : (nf,) initial free vector.  anchor : (nf,) L1 anchor.
    blocks_fn(state) -> list of (J_active csr (m, nf), T_active (m,), thr float)
        linearized active constraint rows at the current state. Block order is
        preserved in the LP slack layout (one elastic slack group per block,
        in the order given).
    viol_fn(state) -> float   exact total violation (acceptance oracle; may
        include families that have no LP rows).
    apply_fn(state, x) -> new_state   applies a candidate free vector.
    state : the caller's crop array (required keyword). Opaque to the engine;
        it is threaded through ``blocks_fn``/``viol_fn``/``apply_fn`` and the
        final accepted state is returned as ``(state, viol)``.

    Each iteration solves the elastic LP

        min  1^T t + mu * 1^T s
        s.t. |x - anchor| <= t            (L1 epigraph)
             J_k x >= thr_k - s_k rows    (per active block k, linearized)
             x in [x_cur - trust, x_cur + trust],  t >= 0,  s >= 0

    with ``linprog(method='highs')``. A candidate is accepted only if the
    EXACT violation strictly decreases (``v_new < viol * (1 - accept_rtol)``);
    on accept the trust radius doubles (capped at ``trust_cap``), otherwise it
    halves (loop breaks below ``trust_floor``). LP failure also halves the
    trust radius. The loop stops once ``viol <= stop_viol``.
    """
    x = np.asarray(x0)
    anchor = np.asarray(anchor)
    nf = x.size
    viol = viol_fn(state)
    trust = trust0
    for _ in range(max_iters):
        if viol <= stop_viol:
            break
        blocks = blocks_fn(state)
        sizes = [J.shape[0] for (J, _T, _thr) in blocks]
        Ka = int(sum(sizes))
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), mu * np.ones(Ka)])
        A1 = sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        A2 = sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        rows = [A1, A2]
        rhs = [anchor, -anchor]
        off = 0
        for (J, T, thr), m in zip(blocks, sizes):
            parts = [-J, sp.csr_matrix((m, nf))]
            if off:
                parts.append(sp.csr_matrix((m, off)))
            parts.append(-sp.eye(m))
            tail = Ka - off - m
            if tail:
                parts.append(sp.csr_matrix((m, tail)))
            rows.append(sp.hstack(parts))
            rhs.append(-thr + T - J @ x)
            off += m
        A_ub = sp.vstack(rows).tocsr()
        b_ub = np.concatenate(rhs)
        bounds = (
            [(float(x[i] - trust), float(x[i] + trust)) for i in range(nf)]
            + [(0.0, None)] * nf
            + [(0.0, None)] * Ka
        )
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            trust *= 0.5
            if trust < trust_floor:
                break
            continue
        x_new = res.x[:nf]
        cand = apply_fn(state, x_new)
        v_new = viol_fn(cand)
        if v_new < viol * (1 - accept_rtol):
            state, viol, x = cand, v_new, x_new
            trust = min(trust * 2, trust_cap)
        else:
            trust *= 0.5
            if trust < trust_floor:
                break
    return state, viol
