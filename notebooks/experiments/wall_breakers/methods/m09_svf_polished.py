"""SVF seed + scipy log-barrier L2 polish (anchored to original input).

Two facts established by the per-method tests:

* SVF projection (m01) is guaranteed-feasible in < 1 s on full slices,
  but produces a large L2 deviation because exp(v)-fields are a strict
  subset of all DVFs.
* ALM (m03) is the L2 winner on small crops but DIVERGES on full slices
  when no feasible solution is close to the input.

The fix: use the SVF as a feasible STARTING POINT for a pure log-barrier
interior-point minimisation of ``0.5 ||phi - phi_in||^2``. The
log-barrier is +inf the moment any T_k <= threshold, so it CANNOT leave
the feasible set; scipy L-BFGS-B's line search backs off on +inf and
shrinks the step until feasibility is preserved.

This combines the *guarantee* of SVF (feasibility) with the *objective*
of ALM (closeness to phi_in), and runs on a feasible manifold rather
than chasing one. Expected on z=12 full slice: L2 << 600 with min_T at
threshold + margin.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.core.iterative2d_tri_barrier import _tri_areas_flat, _tri_grad_T_v
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from . import m01_svf_projection as svf

NAME = 'svf_polished'
DESCRIPTION = 'SVF seed (guaranteed feasible) then scipy log-barrier L2 polish anchored to phi_in'


def _min_tri(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(np.minimum(T1, T2).min())


def _barrier_anchored(phi_flat, phi_init_flat, H, W, threshold, mu, anchor):
    """0.5 ||phi - phi_init||^2  -  mu * sum log(T_k - threshold).

    Returns (val, grad). If any slack <= 0 returns (+inf, zeros) so
    scipy's line search shrinks the step.
    """
    diff = phi_flat - phi_init_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff)
        grad = diff.copy()
    elif anchor == 'l1':
        s = np.sqrt(diff * diff + 1e-8)
        val = float((s - 1e-4).sum())
        grad = diff / s
    else:
        val = 0.0
        grad = np.zeros_like(diff)

    T = _tri_areas_flat(phi_flat, H, W)
    s = T - threshold
    if (s <= 0).any():
        return np.inf, grad  # scipy will reject and shrink step
    val += -mu * float(np.log(s).sum())
    grad = grad - mu * _tri_grad_T_v(phi_flat, H, W, 1.0 / s)
    return val, grad


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          svf_kwargs: dict | None = None,
          mu_schedule: tuple = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
          inner_maxiter: int = 200,
          anchor: str = 'l2',
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    t0 = time.time()
    H, W = phi_in.shape[1], phi_in.shape[2]

    # Stage 1: SVF projection (cheap, guaranteed feasible).
    r1 = svf.solve(phi_in, threshold=threshold, **dict(svf_kwargs or {}))
    seed = r1['phi_out']
    seed_min = _min_tri(seed)
    info: dict = {'stage1_svf': dict(min_T=seed_min,
                                     L2_to_input=float(np.linalg.norm(
                                         (seed - phi_in).ravel())),
                                     wall=time.time() - t0,
                                     **r1['info'])}
    if verbose:
        print(f'  svf seed min_T={seed_min:+.5f}  '
              f'L2={info["stage1_svf"]["L2_to_input"]:.1f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    if seed_min <= threshold:
        info['final_min_T'] = seed_min
        info['stage2'] = {'skipped': 'seed-infeasible'}
        return {'phi_out': seed, 'info': info}

    # Stage 2: log-barrier polish, ANCHORED TO phi_in (not the seed).
    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([seed[0].ravel(), seed[1].ravel()])

    polish_log = []
    for mu in mu_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(_barrier_anchored, phi_flat, jac=True,
                       method='L-BFGS-B',
                       args=(phi_in_flat, H, W, threshold, mu, anchor),
                       options=dict(maxiter=inner_maxiter, ftol=1e-11,
                                    gtol=1e-8))
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        min_T = float(T.min())
        phi_cur = np.stack([phi_flat[:H * W].reshape(H, W),
                            phi_flat[H * W:].reshape(H, W)])
        L2 = float(np.linalg.norm((phi_cur - phi_in).ravel()))
        polish_log.append(dict(mu=mu, min_T=min_T, L2=L2,
                                nit=int(res.nit), wall=time.time() - t0))
        if verbose:
            print(f'  bar mu={mu:.1e}  min_T={min_T:+.5f}  L2={L2:.1f}  '
                  f'nit={res.nit}  ({time.time()-t0:.1f}s)', flush=True)
        # Continue through the full mu schedule -- the barrier itself
        # prevents infeasibility; if a step would lose feasibility scipy
        # rejects it. Earlier early-stopping on min_T < threshold+margin
        # was too conservative (the central path naturally approaches
        # threshold as mu -> 0).

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    info['stage2_polish'] = dict(steps=len(polish_log),
                                  final_min_T=_min_tri(phi_out),
                                  final_L2=float(np.linalg.norm(
                                      (phi_out - phi_in).ravel())),
                                  wall=time.time() - t0,
                                  log_last3=polish_log[-3:])
    info['final_min_T'] = _min_tri(phi_out)
    return {'phi_out': phi_out, 'info': info}
