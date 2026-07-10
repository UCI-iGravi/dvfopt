"""Per-cluster 2-triangle solver with a frozen-edge interior mask.

Promoted from ``notebooks/manuscript/_bench_worker.solve_cluster_inline``,
the workhorse the manuscript runner used per fold cluster. Same logic,
exposed as a package entry point so other dvfopt code (notably the
Schwarz hybrid in :mod:`dvfopt.core.iterative2d_tri_schwarz`) can call
it directly instead of re-implementing the L2-multi-pass + L1-polish
loop inline.

The cluster solve takes a crop of the field, an ``interior_mask``
selecting which voxel corners are movable, the original (full-grid)
anchor for the same crop, plus solver budgets, and returns the
corrected crop plus a dictionary of per-pass statistics.

The constraint is the 2-triangle-per-cell scheme. The analytical
Jacobian (vectorised over the crop) is provided to SLSQP, which roughly
10-100× the per-iteration cost over scipy's default finite-difference
column sweep for crops of a few hundred variables.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

# ---------------------------------------------------------------------------
# Interior pack/unpack
# ---------------------------------------------------------------------------


def _interior_pack_unpack_2d(phi_win, interior_mask):
    """Build pack/unpack closures restricted to ``interior_mask`` corners.

    ``pack(phi)`` -> 1-D array of length 2*n_int with ``[dy_int, dx_int]``.
    ``unpack(z, base)`` -> ``base`` with the interior corners overwritten
    from ``z``; frozen-edge corners stay at ``base``'s values.
    """
    int_idx = np.argwhere(interior_mask)
    n_int = len(int_idx)
    iy, ix = int_idx[:, 0], int_idx[:, 1]

    def pack(phi):
        return np.concatenate([phi[0][iy, ix], phi[1][iy, ix]])

    def unpack(z, base):
        out = base.copy()
        out[0][iy, ix] = z[:n_int]
        out[1][iy, ix] = z[n_int:]
        return out

    return pack, unpack, n_int


# ---------------------------------------------------------------------------
# Analytical 2-tri constraint Jacobian (interior vars only)
# ---------------------------------------------------------------------------


def _make_2tri_jac_2d(phi_win, interior_mask):
    """Return a callable ``jac(z) -> (n_constr, n_vars)`` Jacobian.

    Hands SLSQP the analytical Jacobian of the 2-triangle constraint,
    restricted to the interior-only variable layout. The shape is
    ``(2*Hc*Wc, 2*n_int)`` where ``Hc=H-1``, ``Wc=W-1``.

    See :func:`_make_2tri_jac_2d` in the original ``_bench_worker.py``
    for the per-vertex partial derivations. Triangles use the TR-BL
    diagonal split:
      ``T1 = -0.5 det([BL-TR, BR-TR])``  with vertices (TR, BL, BR)
      ``T2 = -0.5 det([BL-TL, TR-TL])``  with vertices (TL, BL, TR)
    """
    _, H, W = phi_win.shape
    Hc, Wc = H - 1, W - 1
    n_cells = Hc * Wc
    n_constr = 2 * n_cells

    int_idx = np.argwhere(interior_mask)
    n_int = len(int_idx)
    iy = int_idx[:, 0].copy()
    ix = int_idx[:, 1].copy()
    int_pos = np.full((H, W), -1, dtype=np.int64)
    int_pos[iy, ix] = np.arange(n_int)
    n_vars = 2 * n_int

    cy_idx = np.arange(Hc, dtype=np.int64)[:, None]
    cx_idx = np.arange(Wc, dtype=np.int64)[None, :]
    col_TL_dy = int_pos[cy_idx, cx_idx]
    col_TR_dy = int_pos[cy_idx, cx_idx + 1]
    col_BL_dy = int_pos[cy_idx + 1, cx_idx]
    col_BR_dy = int_pos[cy_idx + 1, cx_idx + 1]
    col_TL_dx = np.where(col_TL_dy >= 0, col_TL_dy + n_int, -1)
    col_TR_dx = np.where(col_TR_dy >= 0, col_TR_dy + n_int, -1)
    col_BL_dx = np.where(col_BL_dy >= 0, col_BL_dy + n_int, -1)
    col_BR_dx = np.where(col_BR_dy >= 0, col_BR_dy + n_int, -1)

    rows_T1 = (cy_idx * Wc + cx_idx).astype(np.int64) * np.ones((Hc, Wc), dtype=np.int64)
    rows_T2 = rows_T1 + n_cells

    partials = []
    for rows_arr, col_arr in [
        (rows_T1, col_TR_dy),
        (rows_T1, col_TR_dx),
        (rows_T1, col_BL_dy),
        (rows_T1, col_BL_dx),
        (rows_T1, col_BR_dy),
        (rows_T1, col_BR_dx),
        (rows_T2, col_TL_dy),
        (rows_T2, col_TL_dx),
        (rows_T2, col_TR_dy),
        (rows_T2, col_TR_dx),
        (rows_T2, col_BL_dy),
        (rows_T2, col_BL_dx),
    ]:
        col_flat = col_arr.ravel()
        valid = col_flat >= 0
        partials.append(
            {
                'rows': rows_arr.ravel()[valid],
                'cols': col_flat[valid],
                'valid': valid,
            }
        )

    iy_local = iy
    ix_local = ix
    ref_y = np.arange(H, dtype=np.float64)[:, None]
    ref_x = np.arange(W, dtype=np.float64)[None, :]
    phi_base = phi_win.copy()

    # The sparsity pattern is constant — precompute the concatenated
    # (row, col) fancy index ONCE at build time, and preallocate the dense
    # Jacobian buffer that every ``jac(z)`` call rewrites in place.
    rows_concat = np.concatenate([p['rows'] for p in partials])
    cols_concat = np.concatenate([p['cols'] for p in partials])
    J_buf = np.zeros((n_constr, n_vars), dtype=np.float64)

    def jac(z):
        phi_base[0][iy_local, ix_local] = z[:n_int]
        phi_base[1][iy_local, ix_local] = z[n_int:]
        def_x = ref_x + phi_base[1]
        def_y = ref_y + phi_base[0]
        TL_x = def_x[:-1, :-1]
        TL_y = def_y[:-1, :-1]
        TR_x = def_x[:-1, 1:]
        TR_y = def_y[:-1, 1:]
        BL_x = def_x[1:, :-1]
        BL_y = def_y[1:, :-1]
        BR_x = def_x[1:, 1:]
        BR_y = def_y[1:, 1:]

        dT1_TR_x = 0.5 * (BR_y - BL_y)
        dT1_TR_y = 0.5 * (BL_x - BR_x)
        dT1_BL_x = 0.5 * (TR_y - BR_y)
        dT1_BL_y = 0.5 * (BR_x - TR_x)
        dT1_BR_x = 0.5 * (BL_y - TR_y)
        dT1_BR_y = 0.5 * (TR_x - BL_x)
        dT2_TL_x = 0.5 * (TR_y - BL_y)
        dT2_TL_y = 0.5 * (BL_x - TR_x)
        dT2_BL_x = 0.5 * (TL_y - TR_y)
        dT2_BL_y = 0.5 * (TR_x - TL_x)
        dT2_TR_x = 0.5 * (BL_y - TL_y)
        dT2_TR_y = 0.5 * (TL_x - BL_x)

        vals = [
            dT1_TR_y,
            dT1_TR_x,
            dT1_BL_y,
            dT1_BL_x,
            dT1_BR_y,
            dT1_BR_x,
            dT2_TL_y,
            dT2_TL_x,
            dT2_TR_y,
            dT2_TR_x,
            dT2_BL_y,
            dT2_BL_x,
        ]
        # Scatter the per-call values into the PREALLOCATED dense buffer
        # via the constant fancy index. Returning a sparse matrix here is
        # counterproductive: scipy's ``new_constraint_to_old.j_ineq``
        # (scipy/optimize/_constraints.py) calls ``.toarray()`` on it AND
        # allocates its own dense ``zeros((n_constr, n_vars))`` + row-copy
        # on EVERY jac call — so the COO build + CSR sort was pure
        # overhead on top of the dense array scipy materialised anyway.
        # The (row, col) pairs are unique (3 distinct corners x 2 channels
        # per triangle row), so plain assignment is exact; entries off the
        # constant sparsity pattern stay 0 from the initial allocation.
        J_buf[rows_concat, cols_concat] = np.concatenate(
            [v.ravel()[p['valid']] for p, v in zip(partials, vals)]
        )
        return J_buf

    return jac


def _seed_perturb(z_init, z_anchor, sigma=1e-3, seed=42):
    """Small reproducible perturbation away from the anchor — gives SLSQP
    a non-trivial first iterate when the anchor itself is feasible."""
    rng = np.random.default_rng(seed)
    direction = z_init - z_anchor
    norm = float(np.linalg.norm(direction))
    if norm > 1e-12:
        return z_init  # already off-anchor
    return z_init + rng.normal(scale=sigma, size=z_init.shape)


# ---------------------------------------------------------------------------
# Per-cluster solver
# ---------------------------------------------------------------------------


def solve_cluster_2tri_2d(
    phi_win: np.ndarray,
    phi_anchor_win: np.ndarray,
    interior_mask: np.ndarray,
    *,
    threshold: Optional[float] = None,
    eps_l1: float = 1e-4,
    l2_max_passes: int = 12,
    l2_max_iter: int = 80,
    l1_max_iter: int = 120,
) -> tuple[np.ndarray, dict]:
    """Multi-pass L2-SLSQP + L1-polish on one fold cluster (2D).

    Parameters
    ----------
    phi_win : ndarray, shape ``(2, Hw, Ww)``
        Crop of the field (channels [dy, dx]). Frozen-edge corners on the
        boundary stay at their input values; interior corners are
        optimized.
    phi_anchor_win : ndarray, same shape as ``phi_win``
        Anchor for the L2/L1 objective — usually the original (pre-fold)
        field's crop.
    interior_mask : ndarray of bool, shape ``(Hw, Ww)``
        ``True`` for corners that are movable; ``False`` for frozen-edge
        corners. The shape matches the per-voxel corner grid (one more
        than the cell grid on each axis).
    threshold : float, optional
        Lower bound for both T1 and T2 areas (the 2-triangle constraint).
        Defaults to ``DEFAULT_PARAMS['threshold']`` (0.01) — matching every
        other 2D triangle solver in the package.
    eps_l1 : float
        Smoothing for the L1 polish step.
    l2_max_passes : int
        Maximum number of L2-SLSQP passes (each pass restarts from the
        current iterate; the first pass uses a small Gaussian perturb to
        kick SLSQP off the anchor). Aborts early after
        ``STALL_PERTURB_LIMIT = 3`` consecutive non-improving passes.
    l2_max_iter : int
        SLSQP ``maxiter`` per L2 pass.
    l1_max_iter : int
        SLSQP ``maxiter`` for the L1 polish (only runs if L2 reaches
        ``n_neg = 0``).

    Returns
    -------
    phi_out : ndarray, same shape as ``phi_win``
        Corrected crop. Frozen-edge corners are unchanged; interior
        corners are updated. The caller is responsible for splicing
        ``phi_out[:, interior_mask]`` back into the full slice.
    info : dict with keys ``feasible``, ``after_l2_n_neg``,
        ``after_l2_min``, ``after_l1_n_neg``, ``after_l1_min``,
        ``l2_passes_run``, ``l2_total_nit``, ``l2_total_t``,
        ``l1_polished``, ``l1_nit``, ``l1_t``, ``cluster_t``.

    Notes
    -----
    All ``*_n_neg`` counts (and the ``feasible`` flag derived from them)
    count triangles **below the constraint's lower bound** — i.e.
    ``T < threshold - err_tol`` with the package-wide ``err_tol = 1e-5``
    SLSQP slack — NOT merely inverted triangles (``T <= 0``). This keeps
    the feasibility gate consistent with what SLSQP actually enforces
    (``lb=threshold``): a crop whose min area lies in ``(0, threshold)``
    is *not* feasible, is not early-returned unsolved, and cannot be
    reported ``feasible`` after a failed solve.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    # Feasibility predicate matching the SLSQP constraint lb=threshold,
    # with the package error tolerance as slack for solver round-off.
    feas_lb = threshold - DEFAULT_PARAMS['err_tol']

    def _n_below(T1, T2):
        return int((T1 < feas_lb).sum() + (T2 < feas_lb).sum())

    t0 = time.time()
    T1, T2 = _triangle_areas_2d(phi_win[0], phi_win[1])
    init_n_neg = _n_below(T1, T2)
    init_min_tri = float(min(T1.min(), T2.min()))

    info = {
        'init_n_neg': init_n_neg,
        'init_min_tri': init_min_tri,
    }

    if init_n_neg == 0:
        info.update(
            {
                'after_l2_n_neg': 0,
                'after_l2_min': init_min_tri,
                'after_l1_n_neg': 0,
                'after_l1_min': init_min_tri,
                'l2_passes_run': 0,
                'l2_total_nit': 0,
                'l2_total_t': 0.0,
                'l1_polished': False,
                'l1_nit': 0,
                'l1_t': 0.0,
                'cluster_t': time.time() - t0,
                'feasible': True,
            }
        )
        return phi_win.copy(), info

    pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, interior_mask)
    if n_int == 0:
        # No movable corners — can't fix anything.
        info.update(
            {
                'after_l2_n_neg': init_n_neg,
                'after_l2_min': init_min_tri,
                'after_l1_n_neg': init_n_neg,
                'after_l1_min': init_min_tri,
                'l2_passes_run': 0,
                'l2_total_nit': 0,
                'l2_total_t': 0.0,
                'l1_polished': False,
                'l1_nit': 0,
                'l1_t': 0.0,
                'cluster_t': time.time() - t0,
                'feasible': False,
            }
        )
        return phi_win.copy(), info

    z_anchor = pack(phi_anchor_win)

    def obj_l2(z):
        d = z - z_anchor
        return 0.5 * float(np.dot(d, d)), d

    def constr(z):
        phi = unpack(z, phi_win)
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        return np.concatenate([T1.ravel(), T2.ravel()])

    jac_func = _make_2tri_jac_2d(phi_win, interior_mask)
    nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf, jac=jac_func)

    # ----- L2 multi-pass with perturb-on-stall -----
    STALL_PERTURB_LIMIT = 3
    phi_work = phi_win.copy()
    l2_total_nit = 0
    l2_total_t = 0.0
    l2_passes_run = 0
    stall_count = 0
    perturb_seed = 0
    for pass_idx in range(l2_max_passes):
        T1, T2 = _triangle_areas_2d(phi_work[0], phi_work[1])
        cur_n_neg = _n_below(T1, T2)
        if cur_n_neg == 0:
            break
        z_init = pack(phi_work)
        if pass_idx == 0:
            z_init = _seed_perturb(z_init, z_anchor)
        elif stall_count > 0:
            rng = np.random.default_rng(101 + perturb_seed)
            sigma = 0.005 * stall_count
            z_init = z_init + rng.normal(scale=sigma, size=z_init.shape)
            perturb_seed += 1
        t_pass = time.time()
        res = minimize(
            obj_l2,
            z_init,
            jac=True,
            method='SLSQP',
            constraints=[nl],
            options={'maxiter': l2_max_iter, 'disp': False},
        )
        l2_total_t += time.time() - t_pass
        l2_passes_run += 1
        phi_new = unpack(res.x, phi_work)
        T1_new, T2_new = _triangle_areas_2d(phi_new[0], phi_new[1])
        new_n_neg = _n_below(T1_new, T2_new)
        if new_n_neg < cur_n_neg:
            phi_work = phi_new
            l2_total_nit += int(res.nit)
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= STALL_PERTURB_LIMIT:
                break

    T1_f, T2_f = _triangle_areas_2d(phi_work[0], phi_work[1])
    after_l2_n_neg = _n_below(T1_f, T2_f)
    after_l2_min = float(min(T1_f.min(), T2_f.min()))

    # ----- L1 polish (only if L2 reached feasibility) -----
    l1_nit = 0
    l1_t = 0.0
    l1_polished = False
    after_l1_n_neg = after_l2_n_neg
    after_l1_min = after_l2_min
    phi_out = phi_work
    if after_l2_n_neg == 0:
        z_init = pack(phi_work)

        def obj_l1(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps_l1 * eps_l1)
            return float(s.sum()), d / s

        t_pass = time.time()
        res = minimize(
            obj_l1,
            z_init,
            jac=True,
            method='SLSQP',
            constraints=[nl],
            options={'maxiter': l1_max_iter, 'ftol': 1e-9, 'disp': False},
        )
        l1_t = time.time() - t_pass
        l1_nit = int(res.nit)
        phi_candidate = unpack(res.x, phi_work)
        T1c, T2c = _triangle_areas_2d(phi_candidate[0], phi_candidate[1])
        # Threshold-consistent gate: the L1 polish may only replace a
        # threshold-feasible L2 result with another threshold-feasible one.
        n_neg_c = _n_below(T1c, T2c)
        L1_l2 = float(np.abs(phi_work - phi_anchor_win).sum())
        L1_c = float(np.abs(phi_candidate - phi_anchor_win).sum())
        if n_neg_c == 0 and L1_c < L1_l2 - 1e-9:
            phi_out = phi_candidate
            after_l1_n_neg = n_neg_c
            after_l1_min = float(min(T1c.min(), T2c.min()))
            l1_polished = True

    info.update(
        {
            'after_l2_n_neg': after_l2_n_neg,
            'after_l2_min': after_l2_min,
            'l2_passes_run': l2_passes_run,
            'l2_total_nit': l2_total_nit,
            'l2_total_t': l2_total_t,
            'after_l1_n_neg': after_l1_n_neg,
            'after_l1_min': after_l1_min,
            'l1_polished': l1_polished,
            'l1_nit': l1_nit,
            'l1_t': l1_t,
            'cluster_t': time.time() - t0,
            'feasible': bool(after_l1_n_neg == 0),
        }
    )
    return phi_out, info
