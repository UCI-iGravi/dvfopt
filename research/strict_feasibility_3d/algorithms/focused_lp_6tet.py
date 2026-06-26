"""Focused active-set LP for closing the last few stubborn 3D folds.

When M10Tet/cluster_slp_3d plateaus at a small number of residual
folds (the "convergence ceiling" on dense B0039 bands), the problem
becomes a tiny LP at heart:

  - the input phi is ALREADY feasible-or-near-feasible globally
  - only ~24 cells out of 3.48M have V_k < threshold
  - those 24 cells touch ~100-500 phi vars (a tiny subset of 7M)

This module solves the focused LP directly:

  min ||phi - phi_in||_1   s.t.   J_active @ (phi - phi_lin)
                                  >= -threshold + T_active

where J_active and T_active include ONLY the rows for cells where
V_k < threshold + buffer (an extended active set). Phi vars outside
the "any-active-tet" set are frozen at their current values.

For 24 below-threshold cells with ~150 active phi vars, the LP has
~300 rows and ~300 decision vars — solves in milliseconds.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from dvfopt.jacobian.tetrahedron_sign import (
    build_tet_sparse_jac,
    tet_volumes_flat,
)


def _phi3hw_to_flat(phi_3dhw: np.ndarray) -> np.ndarray:
    dz, dy, dx = phi_3dhw
    return np.concatenate([dx.ravel(), dy.ravel(), dz.ravel()])


def _flat_to_phi3dhw(phi_flat: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    return np.stack([dz, dy, dx])


def focused_lp_step(
    phi_in_3dhw: np.ndarray,
    phi_cur_3dhw: np.ndarray,
    *,
    threshold: float = 0.01,
    active_buffer: float = 0.05,
    trust_radius: float = 0.1,
):
    """One focused-active-set LP step.

    Parameters
    ----------
    phi_in_3dhw : (3, D, H, W) — L1 anchor (original input).
    phi_cur_3dhw : (3, D, H, W) — current iterate, linearisation point.
    threshold : minimum tet volume to satisfy.
    active_buffer : include constraints where V_k < threshold + buffer.
        The buffer should exceed `trust_radius * |grad|_max` so the
        LP can't break previously-satisfied constraints within the
        trust region.
    trust_radius : L-inf step bound for phi - phi_cur.

    Returns
    -------
    phi_new_3dhw : (3, D, H, W)
    info : dict
    """
    t0 = time.time()
    D, H, W = phi_in_3dhw.shape[1:]
    phi_in_flat = _phi3hw_to_flat(phi_in_3dhw)
    phi_cur_flat = _phi3hw_to_flat(phi_cur_3dhw)

    # Active constraint rows: those near violation.
    T_lin = tet_volumes_flat(phi_cur_flat, D, H, W)
    active_mask = T_lin < threshold + active_buffer
    n_active = int(active_mask.sum())
    if n_active == 0:
        return phi_cur_3dhw.copy(), {
            'n_active': 0,
            'success': True,
            'reason': 'no active constraints',
            'wall_s': time.time() - t0,
        }

    # Build full J then subset to active rows.
    jac = build_tet_sparse_jac(D, H, W)
    J_full = jac(phi_cur_flat).tocsr()
    J_active = J_full[active_mask, :]  # (n_active, 3*D*H*W) CSR

    # Active phi vars: columns with any nonzero in J_active.
    col_nnz = np.diff(J_active.tocsc().indptr)
    active_cols = np.where(col_nnz > 0)[0]
    n_vars = active_cols.size

    # Subset J to active columns. Rows already filtered.
    J_small = J_active[:, active_cols]  # (n_active, n_vars)
    T_active = T_lin[active_mask]
    phi_in_small = phi_in_flat[active_cols]
    phi_cur_small = phi_cur_flat[active_cols]

    # ---- Focused LP ----
    # Decision vector: [phi_small (n_vars), t (n_vars, L1 epigraph)]
    n = n_vars
    c = np.concatenate([np.zeros(n), np.ones(n)])

    # L1 epigraph:  phi - t <= phi_in,  -phi - t <= -phi_in.
    A1 = sp.hstack([sp.eye(n), -sp.eye(n)])
    A2 = sp.hstack([-sp.eye(n), -sp.eye(n)])
    # Linearised constraint:  J_small @ phi >= threshold - T_active + J_small @ phi_cur_small.
    # Equivalently: -J_small @ phi <= -threshold + T_active - J_small @ phi_cur_small.
    rhs_tri = -threshold + T_active - J_small @ phi_cur_small
    A3 = sp.hstack([-J_small, sp.csr_matrix((n_active, n))])
    # Trust region: phi - phi_cur in [-trust, trust]  =>  phi <= phi_cur + trust, -phi <= -phi_cur + trust.
    A4 = sp.hstack([sp.eye(n), sp.csr_matrix((n, n))])
    A5 = sp.hstack([-sp.eye(n), sp.csr_matrix((n, n))])

    A_ub = sp.vstack([A1, A2, A3, A4, A5]).tocsr()
    b_ub = np.concatenate([
        phi_in_small,
        -phi_in_small,
        rhs_tri,
        phi_cur_small + trust_radius,
        -phi_cur_small + trust_radius,
    ])
    bounds = [(None, None)] * n + [(0.0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        return phi_cur_3dhw.copy(), {
            'n_active': n_active,
            'n_vars': n_vars,
            'success': False,
            'lp_status': str(res.message),
            'wall_s': time.time() - t0,
        }

    # Apply update to active vars only.
    phi_new_flat = phi_cur_flat.copy()
    phi_new_flat[active_cols] = res.x[:n]
    phi_new = _flat_to_phi3dhw(phi_new_flat, D, H, W)
    return phi_new, {
        'n_active': n_active,
        'n_vars': n_vars,
        'success': True,
        'lp_obj': float(res.fun),
        'wall_s': time.time() - t0,
    }


def focused_slp(
    phi_in_3dhw: np.ndarray,
    *,
    threshold: float = 0.01,
    safety_tol: float = 1e-5,
    active_buffer: float = 0.05,
    trust_radius_0: float = 0.1,
    max_iter: int = 20,
    trust_grow: float = 1.5,
    trust_shrink: float = 0.5,
    seed: np.ndarray | None = None,
    verbose: int = 0,
):
    """Sequential focused-active-set LP. Starts from `seed` (or
    `phi_in_3dhw` if None) and iterates until strict feasibility
    or step convergence."""
    t0 = time.time()
    phi_cur = (seed if seed is not None else phi_in_3dhw).astype(np.float64).copy()
    trust = float(trust_radius_0)
    history = []
    for it in range(max_iter):
        from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d
        V = six_tet_volumes_3d(phi_cur)
        n_neg = int((V <= 0).sum())
        n_below = int((threshold - safety_tol > V).sum())
        if verbose:
            print(
                f'[focused iter {it}] n_neg={n_neg}  n<thresh={n_below}  '
                f'min_T={float(V.min()):+.6f}  trust={trust:.4f}',
                flush=True,
            )
        history.append({'iter': it, 'n_neg': n_neg, 'n_below': n_below,
                        'min_T': float(V.min()), 'trust': trust})
        if n_neg == 0 and n_below == 0:
            break
        phi_new, info = focused_lp_step(
            phi_in_3dhw, phi_cur,
            threshold=threshold,
            active_buffer=active_buffer,
            trust_radius=trust,
        )
        if not info['success']:
            if verbose:
                print(
                    f'    LP FAIL: n_active={info.get("n_active")}  '
                    f'n_vars={info.get("n_vars")}  status={info.get("lp_status", "n/a")}',
                    flush=True,
                )
            trust *= trust_shrink
            if trust < 1e-6:
                break
            continue
        V_new = six_tet_volumes_3d(phi_new)
        new_n_neg = int((V_new <= 0).sum())
        new_n_below = int((V_new < threshold - safety_tol).sum())
        if verbose:
            print(
                f'    LP: n_active={info["n_active"]}  n_vars={info["n_vars"]}  '
                f'wall={info["wall_s"]:.2f}s  '
                f'-> n_neg={new_n_neg}  n<thresh={new_n_below}',
                flush=True,
            )
        # Accept if (n_neg, n_below) lex-decreased; else shrink trust.
        if (new_n_neg, new_n_below) <= (n_neg, n_below):
            phi_cur = phi_new
            trust = min(trust * trust_grow, 1.0)
        else:
            trust *= trust_shrink
            if trust < 1e-6:
                break
    final_V = six_tet_volumes_3d(phi_cur)
    return phi_cur, {
        'iters': it + 1,
        'final_n_neg': int((final_V <= 0).sum()),
        'final_n_below_threshold': int((final_V < threshold - safety_tol).sum()),
        'final_min_T': float(final_V.min()),
        'wall_s': time.time() - t0,
        'history': history,
    }
