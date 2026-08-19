"""Full-grid SLSQP for the 2-triangle constraint, with smoothed-L1 / L2
anchor and a reactive warm-restart.

A sister to :func:`dvfopt.core.iterative2d_tri_barrier.iterative_2d_tri_barrier`
that uses sequential quadratic programming (``scipy.optimize.minimize``
with ``method='SLSQP'``) and the two-triangle ``NonlinearConstraint`` —
the formulation explored in ``notebooks/two-triangle-check/`` (esp.
``11_l1-vs-l2-canonical-cases.ipynb`` and
``14_l1-warmstart-2d-cases.ipynb``) and promoted here into the dvfopt
public surface so downstream code can call it without re-implementing
the SLSQP setup inline.

The constraint is the 2-triangle-per-cell scheme:

    T1, T2 = _triangle_areas_2d(dy, dx)           shape (H-1, W-1) each

Optionally augmented with the two corner patches
(:func:`dvfopt.jacobian.triangle_sign._corner_patch_areas_2d`) so every
grid vertex — including the two diagonally-opposite corners that the
standard TR-BL split leaves with only one triangle — is covered by at
least two constraints. Enable via ``full_coverage=True``.

An analytical constraint Jacobian is supplied to scipy so SLSQP does
not fall back to forward-difference column sweeps (which scaled
``O(n_vars * n_iter)`` and dominated the wall-clock on crops ≥ 20×20).
It is returned as a preallocated DENSE buffer rewritten in place per
call: scipy's SLSQP path (``new_constraint_to_old.j_ineq`` in
``scipy/optimize/_constraints.py``) materialises a dense
``(n_constr, n_vars)`` array every call anyway — calling ``.toarray()``
on sparse input on top of allocating its own dense zeros + row-copy —
so a sparse return only added COO-build/CSR-sort/toarray overhead.

The smoothed-L1 anchor ``F = sum sqrt(diff^2 + eps^2) - eps`` is C¹ and
plays nicely with SLSQP's active-set search. The L2 anchor is also
available as a baseline.

The reactive warm-restart matches the recipe from notebook 14: on
``not res.success`` after the initial budget, either perturb the
iterate (Gaussian sigma, for ``status==8`` line-search stalls) or
resume from it (other statuses, e.g. ``status==9`` max-iter) with a
larger budget and tighter ``ftol``.

Output shape: ``(2, H, W)`` with channels ``[dy, dx]`` matching
:func:`iterative_2d_tri_barrier`.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import log_info
from dvfopt.core._barrier_core import anchor_term
from dvfopt.core.tri_primitives import (
    tri_areas_flat,
    tri_areas_flat_full_coverage,
)
from dvfopt.jacobian.shoelace import _ref_grid


def _build_full_grid_tri_jac(H, W, full_coverage):
    """Build a callable ``jac(z) -> (n_constr, n_vars) ndarray`` for the
    full-grid 2-triangle constraint.

    Variable layout: ``[dy.ravel(), dx.ravel()]`` (length ``2*H*W``).
    Constraint layout: ``[T1.ravel(), T2.ravel()]`` (length
    ``2*(H-1)*(W-1)``); optionally with two corner-patch rows appended.

    The sparsity pattern is constant — only the entries change per call —
    so we precompute the (row, col) index arrays once at build time and
    scatter the per-iteration values into ONE preallocated dense buffer.
    The buffer is large — ``(2*(H-1)*(W-1), 2*H*W)`` — but scipy's SLSQP
    constraint adapter was already materialising exactly that dense array
    internally on every jac call (``j_ineq`` allocates dense zeros and
    ``.toarray()``s sparse input, scipy/optimize/_constraints.py), so the
    reused buffer strictly reduces allocation versus the old CSR return.
    """
    Hc, Wc = H - 1, W - 1
    n_cells = Hc * Wc
    n_constr = 2 * n_cells + (2 if full_coverage else 0)
    n_vars = 2 * H * W

    HW = H * W

    cy_idx = np.arange(Hc, dtype=np.int64)[:, None]
    cx_idx = np.arange(Wc, dtype=np.int64)[None, :]
    # Pixel indices for each corner of cell (cy, cx).
    pix_TL = cy_idx * W + cx_idx
    pix_TR = cy_idx * W + (cx_idx + 1)
    pix_BL = (cy_idx + 1) * W + cx_idx
    pix_BR = (cy_idx + 1) * W + (cx_idx + 1)

    # Column = pixel for dy channel; col + HW for dx channel.
    rows_T1 = (cy_idx * Wc + cx_idx) * np.ones((Hc, Wc), dtype=np.int64)
    rows_T2 = rows_T1 + n_cells

    # 12 per-cell triplets ordered to match the partial-derivative ordering
    # used inside ``jac()`` below.
    triplets = [
        # T1 partials
        (rows_T1, pix_TR, 'dT1_TR_y'),  # dy(TR)
        (rows_T1, pix_TR + HW, 'dT1_TR_x'),  # dx(TR)
        (rows_T1, pix_BL, 'dT1_BL_y'),
        (rows_T1, pix_BL + HW, 'dT1_BL_x'),
        (rows_T1, pix_BR, 'dT1_BR_y'),
        (rows_T1, pix_BR + HW, 'dT1_BR_x'),
        # T2 partials
        (rows_T2, pix_TL, 'dT2_TL_y'),
        (rows_T2, pix_TL + HW, 'dT2_TL_x'),
        (rows_T2, pix_TR, 'dT2_TR_y'),
        (rows_T2, pix_TR + HW, 'dT2_TR_x'),
        (rows_T2, pix_BL, 'dT2_BL_y'),
        (rows_T2, pix_BL + HW, 'dT2_BL_x'),
    ]
    rows_flat = np.concatenate([t[0].ravel() for t in triplets])
    cols_flat = np.concatenate([t[1].ravel() for t in triplets])
    key_order = [t[2] for t in triplets]

    if full_coverage:
        # Patch TL: A=(0,0), B=(1,1), C=(0,1). Patch BR: A=(H-2,W-2),
        # B=(H-1,W-2), C=(H-1,W-1). Each patch contributes 6 partials
        # (dy/dx for each of its 3 vertices).
        row_p_tl = 2 * n_cells
        row_p_br = 2 * n_cells + 1
        pTL_A = 0 * W + 0
        pTL_B = 1 * W + 1
        pTL_C = 0 * W + 1
        pBR_A = (H - 2) * W + (W - 2)
        pBR_B = (H - 1) * W + (W - 2)
        pBR_C = (H - 1) * W + (W - 1)

        patch_triplets = [
            (row_p_tl, pTL_A, 'dPTL_A_y'),
            (row_p_tl, pTL_A + HW, 'dPTL_A_x'),
            (row_p_tl, pTL_B, 'dPTL_B_y'),
            (row_p_tl, pTL_B + HW, 'dPTL_B_x'),
            (row_p_tl, pTL_C, 'dPTL_C_y'),
            (row_p_tl, pTL_C + HW, 'dPTL_C_x'),
            (row_p_br, pBR_A, 'dPBR_A_y'),
            (row_p_br, pBR_A + HW, 'dPBR_A_x'),
            (row_p_br, pBR_B, 'dPBR_B_y'),
            (row_p_br, pBR_B + HW, 'dPBR_B_x'),
            (row_p_br, pBR_C, 'dPBR_C_y'),
            (row_p_br, pBR_C + HW, 'dPBR_C_x'),
        ]
        rows_flat = np.concatenate(
            [rows_flat, np.array([t[0] for t in patch_triplets], dtype=np.int64)]
        )
        cols_flat = np.concatenate(
            [cols_flat, np.array([t[1] for t in patch_triplets], dtype=np.int64)]
        )
        key_order = key_order + [t[2] for t in patch_triplets]

    ref_y, ref_x = _ref_grid(H, W)

    # Preallocated dense Jacobian, rewritten in place each call. Entries
    # off the (constant) sparsity pattern stay 0 from this allocation; the
    # (row, col) pairs are unique so plain fancy-index assignment is exact.
    J_buf = np.zeros((n_constr, n_vars), dtype=np.float64)

    def jac(z):
        dy = z[:HW].reshape(H, W)
        dx = z[HW:].reshape(H, W)
        def_x = ref_x + dx
        def_y = ref_y + dy
        x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
        x_tr, y_tr = def_x[:-1, 1:], def_y[:-1, 1:]
        x_bl, y_bl = def_x[1:, :-1], def_y[1:, :-1]
        x_br, y_br = def_x[1:, 1:], def_y[1:, 1:]

        d = {
            'dT1_TR_x': 0.5 * (y_br - y_bl),
            'dT1_TR_y': 0.5 * (x_bl - x_br),
            'dT1_BL_x': 0.5 * (y_tr - y_br),
            'dT1_BL_y': 0.5 * (x_br - x_tr),
            'dT1_BR_x': 0.5 * (y_bl - y_tr),
            'dT1_BR_y': 0.5 * (x_tr - x_bl),
            'dT2_TL_x': 0.5 * (y_tr - y_bl),
            'dT2_TL_y': 0.5 * (x_bl - x_tr),
            'dT2_BL_x': 0.5 * (y_tl - y_tr),
            'dT2_BL_y': 0.5 * (x_tr - x_tl),
            'dT2_TR_x': 0.5 * (y_bl - y_tl),
            'dT2_TR_y': 0.5 * (x_tl - x_bl),
        }

        if full_coverage:
            # Patch TL: A=(0,0), B=(1,1), C=(0,1).
            Ax = def_x[0, 0]
            Ay = def_y[0, 0]
            Bx = def_x[1, 1]
            By = def_y[1, 1]
            Cx = def_x[0, 1]
            Cy = def_y[0, 1]
            d['dPTL_A_x'] = 0.5 * (Cy - By)
            d['dPTL_A_y'] = 0.5 * (Bx - Cx)
            d['dPTL_B_x'] = 0.5 * (Ay - Cy)
            d['dPTL_B_y'] = 0.5 * (Cx - Ax)
            d['dPTL_C_x'] = 0.5 * (By - Ay)
            d['dPTL_C_y'] = 0.5 * (Ax - Bx)
            # Patch BR: A=(H-2,W-2), B=(H-1,W-2), C=(H-1,W-1).
            Ax = def_x[H - 2, W - 2]
            Ay = def_y[H - 2, W - 2]
            Bx = def_x[H - 1, W - 2]
            By = def_y[H - 1, W - 2]
            Cx = def_x[H - 1, W - 1]
            Cy = def_y[H - 1, W - 1]
            d['dPBR_A_x'] = 0.5 * (Cy - By)
            d['dPBR_A_y'] = 0.5 * (Bx - Cx)
            d['dPBR_B_x'] = 0.5 * (Ay - Cy)
            d['dPBR_B_y'] = 0.5 * (Cx - Ax)
            d['dPBR_C_x'] = 0.5 * (By - Ay)
            d['dPBR_C_y'] = 0.5 * (Ax - Bx)

        parts = []
        for key in key_order:
            arr = d[key]
            parts.append(
                np.ravel(arr) if isinstance(arr, np.ndarray) else np.array([arr], dtype=np.float64)
            )
        data_flat = np.concatenate(parts)
        J_buf[rows_flat, cols_flat] = data_flat
        return J_buf

    return jac


def iterative_2d_tri_slsqp(
    deformation_2hw,
    *,
    threshold=None,
    max_iter=50,
    warm_max_iter=1200,
    warm_ftol=1e-10,
    warm_sigma=0.01,
    warm_seed=123,
    anchor='l1',
    eps_l1=1e-4,
    full_coverage=False,
    verbose=1,
    record_history=False,
):
    """Full-grid SLSQP solver enforcing ``T1, T2 >= threshold`` on every cell.

    Parameters
    ----------
    deformation_2hw : ndarray
        Input field. Shape ``(2, H, W)`` with channels ``[dy, dx]``, or
        ``(3, 1, H, W)`` (dz channel ignored).
    threshold : float, optional
        Lower bound on triangle areas. Defaults to
        ``DEFAULT_PARAMS['threshold']`` (0.01).
    max_iter : int
        SLSQP iteration cap for the initial (cold) run.
    warm_max_iter : int
        SLSQP iteration cap for the warm-restart run (only fires if the
        cold run exits with ``res.success == False``).
    warm_ftol : float
        Tighter ``ftol`` used by the warm restart.
    warm_sigma : float
        Gaussian perturbation scale applied to the cold iterate on
        ``status == 8`` (line-search stall).
    warm_seed : int
        Seed for the warm-restart perturbation RNG (deterministic).
    anchor : {'l1', 'l2', 'none'}
        Anchor objective against ``deformation_2hw``. Default ``'l1'``
        (smoothed) — produces concentrated corrections; see notebook 14.
    eps_l1 : float
        Smoothing constant for the L1 anchor.
    full_coverage : bool
        When True, the constraint is augmented with two corner-patch
        triangles so vertices ``(0, 0)`` and ``(H-1, W-1)`` — which the
        standard TR-BL per-cell scheme leaves under-covered — each
        participate in at least two constraints.
    verbose : int
        ``0`` = silent, ``1`` = one-line summary, ``>=3`` enables scipy's
        own ``disp=True``.
    record_history : bool
        If True, returns ``(phi, history)`` where history records the
        cold + warm run statistics.

    Returns
    -------
    phi_corrected : ndarray, shape ``(2, H, W)`` — channels ``[dy, dx]``.
    history : list of dict, only if ``record_history=True``.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    # Coerce input to (2, H, W) [dy, dx].
    if deformation_2hw.ndim == 4:  # (3, 1, H, W) or (2, 1, H, W)
        if deformation_2hw.shape[0] == 3:
            deformation_2hw = np.stack([deformation_2hw[1, 0], deformation_2hw[2, 0]])
        else:
            deformation_2hw = deformation_2hw[:, 0]
    H, W = deformation_2hw.shape[1], deformation_2hw.shape[2]
    pixels = H * W

    # Tri-barrier phi-pack convention: [dy.ravel(), dx.ravel()]. Matches
    # every other 2-triangle solver in the package.
    z_anchor = np.concatenate([deformation_2hw[0].ravel(), deformation_2hw[1].ravel()])

    def _obj(z):
        return anchor_term(z - z_anchor, anchor, eps_l1)

    constraint_values = tri_areas_flat_full_coverage if full_coverage else tri_areas_flat

    def _constr(z):
        return constraint_values(z, H, W)

    jac_func = _build_full_grid_tri_jac(H, W, full_coverage)
    nlc = NonlinearConstraint(_constr, lb=threshold, ub=np.inf, jac=jac_func)

    t0 = time.time()
    history = []

    # Cold run.
    res = minimize(
        _obj,
        z_anchor.copy(),
        jac=True,
        method='SLSQP',
        constraints=[nlc],
        options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': verbose >= 3},
    )
    cold_nit = int(res.nit)
    cold_status = int(res.status)
    cold_success = bool(res.success)
    cold_wall = time.time() - t0
    if record_history:
        history.append(
            dict(
                phase='cold',
                nit=cold_nit,
                status=cold_status,
                success=cold_success,
                wall_s=cold_wall,
            )
        )

    warm_fired = False
    warm_reason = ''
    if not res.success:
        warm_fired = True
        if res.status == 8:
            rng = np.random.default_rng(warm_seed)
            z_warm = res.x + rng.normal(scale=warm_sigma, size=res.x.shape)
            warm_reason = f'status=8 (line-search stall); perturbed sigma={warm_sigma}'
        else:
            z_warm = res.x.copy()
            warm_reason = f'status={res.status} (resume with extra budget)'

        t1 = time.time()
        res = minimize(
            _obj,
            z_warm,
            jac=True,
            method='SLSQP',
            constraints=[nlc],
            options={'maxiter': warm_max_iter, 'ftol': warm_ftol, 'disp': verbose >= 3},
        )
        if record_history:
            history.append(
                dict(
                    phase='warm',
                    nit=int(res.nit),
                    status=int(res.status),
                    success=bool(res.success),
                    wall_s=time.time() - t1,
                    reason=warm_reason,
                )
            )

    dy_out = res.x[:pixels].reshape(H, W)
    dx_out = res.x[pixels:].reshape(H, W)
    phi_out = np.stack([dy_out, dx_out])

    if verbose >= 1:
        T = _constr(res.x)
        n_neg = int((T <= 0).sum())
        scheme = 'full-coverage' if full_coverage else 'per-cell'
        log_info(
            f'[2d-tri-slsqp done] grid {H}x{W}  anchor={anchor}  '
            f'scheme={scheme}  '
            f'cold_nit={cold_nit} status={cold_status}  '
            f'warm_fired={warm_fired}  '
            f'final_neg={n_neg}  min_T={float(T.min()):+.5f}  '
            f'total_t={time.time() - t0:.2f}s'
        )

    if record_history:
        return phi_out, history
    return phi_out
