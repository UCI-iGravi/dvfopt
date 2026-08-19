"""Full-grid SLSQP for the 3D 6-tetrahedron constraint.

3D analogue of :mod:`dvfopt.core.iterative2d_tri_slsqp`. Drives
``scipy.optimize.minimize(method='SLSQP')`` with:

* ``Tet6Constraint3D.values`` as the constraint vector
  (length ``6 * (D-1) * (H-1) * (W-1)``).
* ``Tet6Constraint3D.jacobian`` as the analytical sparse forward
  Jacobian (built via
  :func:`dvfopt.jacobian.tetrahedron_sign.build_tet_sparse_jac`).
* A smoothed-L1 or L2 anchor against the input field, computed via
  :func:`dvfopt.core._barrier_core.anchor_term`.

Practical note
--------------

3D SLSQP scales poorly. The constraint vector grows as
``6 * (D-1) * (H-1) * (W-1)`` — for a 32×32×32 voxel grid that's
178k constraints, and SLSQP's active-set QP step becomes the bottleneck
long before any reasonable wall-clock target. Use
:class:`dvfopt.strategies.BarrierStrategy` for any non-trivial 3D
problem; this entry point exists for symmetry with the 2D path and for
tiny-grid debugging where KKT semantics matter.

Phi pack convention: ``[dx.ravel(), dy.ravel(), dz.ravel()]`` (DX_FIRST),
matching :class:`dvfopt.constraints.Tet6Constraint3D` and the existing
3D barrier / Jdet paths.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import log_info
from dvfopt.core._barrier_core import anchor_term
from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac, tet_volumes_flat


def iterative_3d_tet_slsqp(
    deformation,
    *,
    threshold=None,
    max_iter=50,
    ftol=1e-8,
    anchor='l2',
    eps_l1=1e-4,
    verbose=1,
    record_history=False,
):
    """Full-grid SLSQP enforcing ``V_k(phi) >= threshold`` on every tet.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``, channels ``[dz, dy, dx]``.
    threshold : float, optional
        Lower bound on per-tet signed volume. Defaults to
        ``DEFAULT_PARAMS['threshold']`` (0.01).
    max_iter : int
        SLSQP iteration cap.
    ftol : float
        SLSQP convergence tolerance.
    anchor : {'l1', 'l2', 'none'}
        Anchor objective against ``deformation``.
    eps_l1 : float
        Smoothing constant for the L1 anchor.
    verbose : int
        0 = silent, 1 = one-line summary, >=3 enables scipy's ``disp=True``.
    record_history : bool
        If True, returns ``(phi, history)`` where history records SLSQP
        run statistics.

    Returns
    -------
    phi_corrected : ndarray, shape ``(3, D, H, W)`` — channels ``[dz, dy, dx]``.
    history : list of dict, only if ``record_history=True``.

    Notes
    -----
    See module docstring for the scaling caveat: prefer
    :class:`dvfopt.strategies.BarrierStrategy` on anything non-tiny.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    deformation = np.asarray(deformation, dtype=np.float64)
    if deformation.ndim != 4 or deformation.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W) input; got shape {deformation.shape}')
    _, D, H, W = deformation.shape

    # Pack as [dx, dy, dz] (DX_FIRST) — matches Tet6Constraint3D.flatten.
    z_anchor = np.concatenate(
        [deformation[2].ravel(), deformation[1].ravel(), deformation[0].ravel()]
    )

    def _obj(z):
        return anchor_term(z - z_anchor, anchor, eps_l1)

    def _constr(z):
        return tet_volumes_flat(z, D, H, W)

    jac_func = build_tet_sparse_jac(D, H, W)
    nlc = NonlinearConstraint(_constr, lb=threshold, ub=np.inf, jac=jac_func)

    if verbose >= 1:
        V_init = _constr(z_anchor)
        log_info(
            f'[3d-tet-slsqp init] grid {D}x{H}x{W}  threshold={threshold}  '
            f'anchor={anchor}  n_constraints={V_init.size}  '
            f'n_neg={int((V_init <= 0).sum())}  min_V={float(V_init.min()):+.5f}'
        )

    t0 = time.time()
    res = minimize(
        _obj,
        z_anchor,
        method='SLSQP',
        jac=True,
        constraints=[nlc],
        options={'maxiter': max_iter, 'ftol': ftol, 'disp': verbose >= 3},
    )
    wall = time.time() - t0

    V_final = _constr(res.x)
    n_neg = int((V_final <= 0).sum())
    min_V = float(V_final.min())
    if verbose >= 1:
        log_info(
            f'[3d-tet-slsqp done] success={res.success}  nit={res.nit}  '
            f'n_neg={n_neg}  min_V={min_V:+.6f}  ({wall:.2f}s)'
        )

    # Unpack flat [dx, dy, dz] back to (3, D, H, W) [dz, dy, dx].
    n = D * H * W
    dx = res.x[:n].reshape(D, H, W)
    dy = res.x[n : 2 * n].reshape(D, H, W)
    dz = res.x[2 * n :].reshape(D, H, W)
    phi_corr = np.stack([dz, dy, dx])

    if record_history:
        # ``min_T`` is the canonical history key across the package — it's
        # what ``_build_solve_info`` / ``SolveInfo.from_legacy_history``
        # read to populate ``PhaseInfo.min_T`` and to detect feasibility.
        # The legacy name is preserved across constraint families (2-tri
        # uses ``T`` for triangle areas, 3D-tet uses ``V`` for volumes;
        # the schema treats them uniformly as "the minimum constraint
        # value reached at this phase").
        history = [
            dict(
                phase='slsqp',
                success=bool(res.success),
                status=int(res.status),
                nit=int(res.nit),
                n_neg=n_neg,
                min_T=min_V,
                wall_s=wall,
            )
        ]
        return phi_corr, history
    return phi_corr


__all__ = ['iterative_3d_tet_slsqp']
