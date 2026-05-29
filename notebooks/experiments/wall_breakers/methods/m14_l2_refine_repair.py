"""L2 refinement of m10 with residual-fold repair.

Diagnosis after m12 and m13:

* m12 (soft-penalty L-BFGS-B from m10 seed) reduces L2 by 90 % on
  z = 12 but leaves 168 residual folds (the quadratic penalty
  saturates and never reaches strict feasibility in finite steps).
* m13 (global line search from m10 toward phi_in) reduces L2 by
  0.1 % because m10 sits exactly at the boundary of the feasible
  set; any global step toward ``phi_in`` immediately violates
  somewhere.

The m12 residual folds are LOCALISED (small clusters of cells where
sliding back to ``phi_in`` would over-stretch). Repairing them with
a small harmonic patch costs negligible L2 because the patch is
small. So the pipeline:

1. ``stage_seed``    : m10 -> globally feasible field, L2 ~ 500 on dense.
2. ``stage_l2_pull`` : soft-penalty L-BFGS-B from the seed, anchored to
                       ``phi_in`` -> L2 ~ 50 on dense, slightly infeasible.
3. ``stage_repair``  : harmonic extension over each residual fold
                       cluster -> feasible, L2 ~ 60-80 (small bump).
4. ``stage_polish``  : final log-barrier L2 polish anchored to phi_in.

Expected: full feasibility + L2 cost much lower than m10 alone.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.core.iterative2d_tri_barrier import _tri_areas_flat, _tri_grad_T_v
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from . import m02_harmonic_extension as harmonic
from . import m10_harmonic_l2_polished as m10
from . import m12_l2_refine as m12

NAME = 'l2_refine_repair'
DESCRIPTION = ('m10 seed -> soft-penalty L2 pull -> harmonic repair of '
               'residual folds -> log-barrier L2 polish')


def _min_tri(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(np.minimum(T1, T2).min())


def _barrier_anchored(phi_flat, phi_in_flat, H, W, threshold, mu, anchor,
                       eps_l1):
    diff = phi_flat - phi_in_flat
    if anchor == 'l2':
        val = 0.5 * float(diff @ diff); grad = diff.copy()
    elif anchor == 'l1':
        s = np.sqrt(diff * diff + eps_l1 * eps_l1)
        val = float((s - eps_l1).sum()); grad = diff / s
    else:
        val = 0.0; grad = np.zeros_like(diff)
    T = _tri_areas_flat(phi_flat, H, W)
    s = T - threshold
    if (s <= 0).any():
        return np.inf, grad
    val += -mu * float(np.log(s).sum())
    grad = grad - mu * _tri_grad_T_v(phi_flat, H, W, 1.0 / s)
    return val, grad


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          anchor: str = 'l2',
          seed: np.ndarray | None = None,
          # m12 stage
          lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8),
          inner_maxiter: int = 300,
          # m02 stage
          ring_pad: int = 2,
          # polish stage
          polish_mu: tuple = (1e-2, 1e-4, 1e-6),
          polish_maxiter: int = 200,
          time_budget_s: float = 600.0,
          verbose: int = 0,
          eps_l1: float = 1e-4) -> dict:
    """Full pipeline. Caller may pass an existing m10 ``seed`` to skip stage 1."""
    H, W = phi_in.shape[1], phi_in.shape[2]
    t0 = time.time()
    info: dict = {}

    # Stage 1: m10 feasible seed.
    if seed is None:
        r10 = m10.solve(phi_in, threshold=threshold, margin=margin,
                        anchor=anchor,
                        time_budget_s=time_budget_s * 0.4, verbose=verbose)
        seed = r10['phi_out']
    seed_L2 = float(np.linalg.norm((seed - phi_in).ravel()))
    seed_min = _min_tri(seed)
    info['stage1'] = dict(min_T=seed_min, L2=seed_L2, wall=time.time() - t0)
    if verbose:
        print(f'  stage1 m10  min_T={seed_min:+.5f}  L2={seed_L2:.1f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    # Stage 2: soft-penalty L2 pull (m12-style).
    remaining = max(60.0, time_budget_s - (time.time() - t0))
    r12 = m12.solve(phi_in, seed=seed, threshold=threshold, margin=margin,
                    anchor=anchor, lam_schedule=lam_schedule,
                    inner_maxiter=inner_maxiter,
                    time_budget_s=remaining * 0.5,
                    verbose=verbose, eps_l1=eps_l1,
                    require_feasibility=False)
    pulled = r12['phi_out']
    pulled_L2 = float(np.linalg.norm((pulled - phi_in).ravel()))
    pulled_min = _min_tri(pulled)
    pulled_neg = int((np.minimum(*_triangle_areas_2d(pulled[0], pulled[1])) <= 0).sum())
    info['stage2'] = dict(min_T=pulled_min, L2=pulled_L2, n_neg=pulled_neg,
                          wall=time.time() - t0)
    if verbose:
        print(f'  stage2 pull min_T={pulled_min:+.5f}  L2={pulled_L2:.1f}  '
              f'n_neg={pulled_neg}  ({time.time()-t0:.1f}s)', flush=True)

    # Stage 3: harmonic repair of residual folds.
    if pulled_min < threshold:
        r_repair = harmonic.solve(pulled, threshold=threshold,
                                   ring_pad=ring_pad, max_grow_iters=8,
                                   margin=margin)
        repaired = r_repair['phi_out']
    else:
        repaired = pulled
    repaired_min = _min_tri(repaired)
    repaired_L2 = float(np.linalg.norm((repaired - phi_in).ravel()))
    info['stage3'] = dict(min_T=repaired_min, L2=repaired_L2,
                          wall=time.time() - t0)
    if verbose:
        print(f'  stage3 patch min_T={repaired_min:+.5f}  L2={repaired_L2:.1f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    # If repair didn't reach feasibility, ALM nudge it.
    if repaired_min < threshold + 1e-6:
        from . import m03_augmented_lagrangian as alm
        r_alm = alm.solve(repaired, threshold=threshold + 0.005, margin=1e-4,
                          anchor='none', outer_max=20, inner_maxiter=150,
                          time_budget_s=max(30.0, time_budget_s - (time.time() - t0)),
                          verbose=0)
        if _min_tri(r_alm['phi_out']) > repaired_min:
            repaired = r_alm['phi_out']
            repaired_min = _min_tri(repaired)
            repaired_L2 = float(np.linalg.norm((repaired - phi_in).ravel()))
            info['stage3b_alm'] = dict(min_T=repaired_min, L2=repaired_L2)

    if repaired_min <= threshold:
        info['final_min_T'] = repaired_min
        info['final_L2'] = repaired_L2
        info['stage4'] = {'skipped': 'still-not-strict-interior'}
        return {'phi_out': repaired, 'info': info}

    # Stage 4: log-barrier L2 polish anchored to phi_in.
    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([repaired[0].ravel(), repaired[1].ravel()])
    polish_log = []
    for mu in polish_mu:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(_barrier_anchored, phi_flat, jac=True,
                       method='L-BFGS-B',
                       args=(phi_in_flat, H, W, threshold, mu, anchor, eps_l1),
                       options=dict(maxiter=polish_maxiter, ftol=1e-12,
                                    gtol=1e-9))
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        polish_log.append(dict(mu=mu, min_T=float(T.min()),
                                L2=float(np.linalg.norm(phi_flat - phi_in_flat)),
                                nit=int(res.nit), wall=time.time() - t0))
        if verbose:
            print(f'  stage4 mu={mu:.1e}  min_T={T.min():+.5f}  '
                  f'L2={float(np.linalg.norm(phi_flat - phi_in_flat)):.1f}  '
                  f'nit={res.nit}  ({time.time()-t0:.1f}s)', flush=True)

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    info['stage4'] = dict(steps=len(polish_log), log_last3=polish_log[-3:],
                          final_min_T=_min_tri(phi_out),
                          final_L2=float(np.linalg.norm((phi_out - phi_in).ravel())))
    info['final_min_T'] = _min_tri(phi_out)
    info['final_L2'] = float(np.linalg.norm((phi_out - phi_in).ravel()))
    return {'phi_out': phi_out, 'info': info}
