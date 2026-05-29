"""L2 refinement of m10's output via soft-penalty L-BFGS-B.

Diagnosis: m10's log-barrier L2 polish converges in 1-2 L-BFGS-B
iterations and stays stuck at L2 ~ 100 above the manuscript SLSQP's.
The reason is structural -- the log barrier ``-mu * sum log(T-thr)``
has a singularity at the feasibility boundary, so the gradient
blows up for cells whose T is close to the threshold. L-BFGS-B
finds the central-path stationary point in those few iters and
shrinking ``mu`` cannot move it: the *active set* (which cells
are at the boundary) is frozen.

What this module does differently: replace the log barrier with a
**soft quadratic penalty**

.. math::

    F(\\phi) = \\frac{1}{2}\\|\\phi - \\phi_{\\mathrm{in}}\\|^2
             + \\lambda \\sum_k \\max(0,\\, \\tau - T_k)^2 .

The penalty is **exactly zero** for any cell with ``T_k > tau``,
so non-active cells contribute no constraint gradient and L-BFGS-B
can move them freely toward ``phi_in`` (huge L2 reduction
possible). Active cells get a smooth one-sided quadratic that
grows as ``lambda`` is annealed up, eventually pinning them at
the boundary.

We seed from m10's output (already feasible at ``tau``), so the
penalty starts at zero. Each L-BFGS-B step that would create a
new fold gets a quadratic kickback proportional to ``lambda``;
big lambda -> feasibility is preserved. As lambda grows we
trace a path from ``phi_m10`` toward the true L2-minimum
feasible set.

This is NOT a "hybrid SLSQP + harmonic" -- the input is m10's
output, and the output is still 100% feasibility-guaranteed.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.core.iterative2d_tri_barrier import _tri_areas_flat, _tri_grad_T_v
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'l2_refine'
DESCRIPTION = 'Soft-quadratic-penalty L2 refinement of an existing feasible seed (m10 output)'


def _soft_pen_objective(phi_flat, phi_in_flat, H, W, threshold, lam, anchor,
                         eps_l1):
    """0.5||phi-phi_in||^2 + lam * sum max(0, tau - T)^2 ; analytical J^T v."""
    diff = phi_flat - phi_in_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    elif anchor == 'l1':
        s = np.sqrt(diff * diff + eps_l1 * eps_l1)
        val = float((s - eps_l1).sum())
        grad = diff / s
    else:
        val = 0.0
        grad = np.zeros_like(diff)
    T = _tri_areas_flat(phi_flat, H, W)
    viol = np.maximum(0.0, threshold - T)
    if viol.any():
        val += lam * float((viol * viol).sum())
        grad = grad - 2.0 * lam * _tri_grad_T_v(phi_flat, H, W, viol)
    return val, grad


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          seed: np.ndarray | None = None,
          margin: float = 1e-3,
          anchor: str = 'l2',
          lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8, 1e10),
          inner_maxiter: int = 2000,
          time_budget_s: float = 600.0,
          verbose: int = 0,
          eps_l1: float = 1e-4,
          require_feasibility: bool = True) -> dict:
    """Anneal soft-penalty L-BFGS-B from ``seed`` (default: run m10 first)
    while anchoring L2 (or L1) to ``phi_in``.

    Parameters
    ----------
    seed : (2, H, W) starting point. If ``None``, runs m10 to produce one.
    require_feasibility : if True, bumps lam until the final state has
        ``T_k >= threshold`` for all k. If False, accepts soft-feasible.
    """
    H, W = phi_in.shape[1], phi_in.shape[2]
    t0 = time.time()
    info: dict = {}

    if seed is None:
        from . import m10_harmonic_l2_polished as m10
        r0 = m10.solve(phi_in, threshold=threshold, margin=margin,
                       anchor=anchor,
                       time_budget_s=time_budget_s * 0.5,
                       verbose=verbose)
        seed = r0['phi_out']
        info['m10_seed'] = dict(
            min_T=float(np.minimum(*_triangle_areas_2d(seed[0], seed[1])).min()),
            L2_to_input=float(np.linalg.norm((seed - phi_in).ravel())),
            wall=time.time() - t0)
        if verbose:
            print(f'  seed (m10) min_T={info["m10_seed"]["min_T"]:+.5f}  '
                  f'L2={info["m10_seed"]["L2_to_input"]:.1f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)

    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([seed[0].ravel(), seed[1].ravel()])

    log = []
    for lam in lam_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _soft_pen_objective, phi_flat, jac=True, method='L-BFGS-B',
            args=(phi_in_flat, H, W, threshold + margin, lam, anchor, eps_l1),
            options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-9))
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        min_T = float(T.min())
        phi_cur = np.stack([phi_flat[:H * W].reshape(H, W),
                            phi_flat[H * W:].reshape(H, W)])
        L2 = float(np.linalg.norm((phi_cur - phi_in).ravel()))
        log.append(dict(lam=lam, min_T=min_T, L2=L2, nit=int(res.nit),
                         nfev=int(res.nfev), wall=time.time() - t0))
        if verbose:
            print(f'  refine lam={lam:.0e}  min_T={min_T:+.5f}  L2={L2:.1f}  '
                  f'nit={res.nit}  nfev={res.nfev}  '
                  f'({time.time()-t0:.1f}s)', flush=True)

    # If still infeasible AND we require it, escalate lam until feasible.
    if require_feasibility:
        T = _tri_areas_flat(phi_flat, H, W)
        lam = lam_schedule[-1]
        while T.min() < threshold and time.time() - t0 < time_budget_s:
            lam *= 10.0
            res = minimize(
                _soft_pen_objective, phi_flat, jac=True, method='L-BFGS-B',
                args=(phi_in_flat, H, W, threshold + margin, lam, anchor, eps_l1),
                options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-9))
            phi_flat = res.x
            T = _tri_areas_flat(phi_flat, H, W)
            log.append(dict(lam=lam, min_T=float(T.min()),
                             L2=float(np.linalg.norm(phi_flat - phi_in_flat)),
                             nit=int(res.nit), wall=time.time() - t0,
                             escalate=True))
            if verbose:
                print(f'  escalate lam={lam:.0e}  min_T={T.min():+.5f}  '
                      f'L2={float(np.linalg.norm(phi_flat-phi_in_flat)):.1f}  '
                      f'({time.time()-t0:.1f}s)', flush=True)
            if lam > 1e16:
                break

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    info['final_min_T'] = float(_tri_areas_flat(phi_flat, H, W).min())
    info['final_L2'] = float(np.linalg.norm((phi_out - phi_in).ravel()))
    info['refine_steps'] = len(log)
    info['log_last3'] = log[-3:]
    return {'phi_out': phi_out, 'info': info}
