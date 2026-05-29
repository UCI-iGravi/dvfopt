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
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import (
    _triangle_areas_2d,
    _corner_patch_areas_2d,
)


def _anchor_objective(z, z_anchor, anchor, eps_l1):
    """Smooth anchor objective + gradient.

    ``anchor='l1'`` returns ``(sum sqrt(d^2+eps^2), d/sqrt(d^2+eps^2))``,
    ``'l2'`` returns ``(0.5 d.d, d)``, and ``'none'`` returns ``(0, 0)``.
    """
    diff = z - z_anchor
    if anchor == 'l1':
        s = np.sqrt(diff * diff + eps_l1 * eps_l1)
        return float(s.sum()), diff / s
    if anchor == 'l2':
        return 0.5 * float(diff @ diff), diff.copy()
    if anchor == 'none':
        return 0.0, np.zeros_like(diff)
    raise ValueError(f"unknown anchor kind: {anchor!r}")


def _tri_constraint_values(z, H, W, full_coverage):
    """Stacked T1, T2 (and optionally the two corner patches).

    ``z`` follows the tri-barrier phi-pack convention: ``z[:H*W] = dy``,
    ``z[H*W:] = dx``. This matches every other 2-triangle solver in
    the package so a flat phi from one module can be passed to another.
    """
    pixels = H * W
    dy = z[:pixels].reshape(H, W)
    dx = z[pixels:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    if full_coverage:
        patches = _corner_patch_areas_2d(dy, dx)
        return np.concatenate([T1.ravel(), T2.ravel(), patches])
    return np.concatenate([T1.ravel(), T2.ravel()])


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
    if deformation_2hw.ndim == 4:                # (3, 1, H, W) or (2, 1, H, W)
        if deformation_2hw.shape[0] == 3:
            deformation_2hw = np.stack([deformation_2hw[1, 0],
                                        deformation_2hw[2, 0]])
        else:
            deformation_2hw = deformation_2hw[:, 0]
    H, W = deformation_2hw.shape[1], deformation_2hw.shape[2]
    pixels = H * W

    # Tri-barrier phi-pack convention: [dy.ravel(), dx.ravel()]. Matches
    # every other 2-triangle solver in the package.
    z_anchor = np.concatenate([deformation_2hw[0].ravel(),
                                deformation_2hw[1].ravel()])

    def _obj(z):
        return _anchor_objective(z, z_anchor, anchor, eps_l1)

    def _constr(z):
        return _tri_constraint_values(z, H, W, full_coverage)

    nlc = NonlinearConstraint(_constr, lb=threshold, ub=np.inf)

    t0 = time.time()
    history = []

    # Cold run.
    res = minimize(
        _obj, z_anchor.copy(), jac=True, method='SLSQP',
        constraints=[nlc],
        options={'maxiter': max_iter, 'ftol': 1e-9,
                 'disp': verbose >= 3},
    )
    cold_nit = int(res.nit)
    cold_status = int(res.status)
    cold_success = bool(res.success)
    cold_wall = time.time() - t0
    if record_history:
        history.append(dict(phase='cold', nit=cold_nit, status=cold_status,
                            success=cold_success, wall_s=cold_wall))

    warm_fired = False
    warm_reason = ''
    if not res.success:
        warm_fired = True
        if res.status == 8:
            rng = np.random.default_rng(warm_seed)
            z_warm = res.x + rng.normal(scale=warm_sigma, size=res.x.shape)
            warm_reason = (f'status=8 (line-search stall); '
                           f'perturbed sigma={warm_sigma}')
        else:
            z_warm = res.x.copy()
            warm_reason = (f'status={res.status} '
                           f'(resume with extra budget)')

        t1 = time.time()
        res = minimize(
            _obj, z_warm, jac=True, method='SLSQP',
            constraints=[nlc],
            options={'maxiter': warm_max_iter, 'ftol': warm_ftol,
                     'disp': verbose >= 3},
        )
        if record_history:
            history.append(dict(phase='warm', nit=int(res.nit),
                                status=int(res.status),
                                success=bool(res.success),
                                wall_s=time.time() - t1,
                                reason=warm_reason))

    dy_out = res.x[:pixels].reshape(H, W)
    dx_out = res.x[pixels:].reshape(H, W)
    phi_out = np.stack([dy_out, dx_out])

    if verbose >= 1:
        T = _constr(res.x)
        n_neg = int((T <= 0).sum())
        scheme = 'full-coverage' if full_coverage else 'per-cell'
        print(f'[2d-tri-slsqp done] grid {H}x{W}  anchor={anchor}  '
              f'scheme={scheme}  '
              f'cold_nit={cold_nit} status={cold_status}  '
              f'warm_fired={warm_fired}  '
              f'final_neg={n_neg}  min_T={float(T.min()):+.5f}  '
              f'total_t={time.time()-t0:.2f}s')

    if record_history:
        return phi_out, history
    return phi_out
