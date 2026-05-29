"""Harmonic seed + scipy log-barrier L2 polish anchored to phi_in.

This is m08 (harmonic + ALM cleanup) with stage 3 replaced by a true
L2-minimising interior-point polish anchored to the *original input*
(not the harmonic seed). The log-barrier prevents the polish from ever
losing feasibility (scipy L-BFGS-B's line search shrinks the step at
+inf), so we get to keep the harmonic guarantee while the polish
sliders the non-core parts of the field back to phi_in.

Pipeline:
    1.  Harmonic extension (m02): produces a feasible seed by replacing
        dilated fold cores with smooth Laplacian completions of the ring.
        Cost: L2 deviation in the cores; gain: full feasibility.
    2.  If the seed is not yet feasible (rare on full slices but common
        on tight crops), run a short ALM (m03) to push it across.
    3.  scipy L-BFGS-B with objective
            0.5 ||phi - phi_in||^2  -  mu * sum log(T_k - threshold)
        and mu schedule 1e-1 -> 1e-5. The barrier cannot lose feasibility;
        all the polish does is slide along the feasible manifold toward
        the input.

Expected: 100% feasibility (inherited from stage 1+2), L2 substantially
lower than m08 because stage 3 actively minimises L2 instead of just
cleaning up infeasibility.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from dvfopt.core.iterative2d_tri_barrier import _tri_areas_flat, _tri_grad_T_v
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from . import m02_harmonic_extension as harmonic
from . import m03_augmented_lagrangian as alm

NAME = 'harmonic_l2_polished'
DESCRIPTION = 'Harmonic seed -> ALM if needed -> scipy log-barrier L2 polish anchored to phi_in'


def _min_tri(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return float(np.minimum(T1, T2).min())


def _barrier_anchored(phi_flat, phi_in_flat, H, W, threshold, mu, anchor):
    diff = phi_flat - phi_in_flat
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
        return np.inf, grad
    val += -mu * float(np.log(s).sum())
    grad = grad - mu * _tri_grad_T_v(phi_flat, H, W, 1.0 / s)
    return val, grad


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          margin: float = 1e-3,
          ring_pad: int = 2,
          mu_schedule: tuple = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
          inner_maxiter: int = 300,
          anchor: str = 'l2',
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    H, W = phi_in.shape[1], phi_in.shape[2]
    t0 = time.time()
    info: dict = {}

    # Stage 1: harmonic extension seed.
    r1 = harmonic.solve(phi_in, threshold=threshold,
                        ring_pad=ring_pad, max_grow_iters=8, margin=0.0)
    seed = r1['phi_out']
    seed_min = _min_tri(seed)
    seed_L2 = float(np.linalg.norm((seed - phi_in).ravel()))
    info['stage1_harmonic'] = dict(min_T=seed_min, L2=seed_L2,
                                    wall=time.time() - t0,
                                    patches=r1['info'].get('patches'))
    if verbose:
        print(f'  stage1 harmonic min_T={seed_min:+.5f}  L2={seed_L2:.1f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    # Stage 2: ALM cleanup if seed not feasible at threshold+margin.
    # Use a LARGER margin here so we get a strictly-interior seed for the
    # polish in stage 3. ALM binds to its target margin; if we ask for
    # margin=0.001 it ends at min_T=0.011, which is right at the polish
    # barrier's log(0) singularity. Asking for 10x margin keeps us in
    # safe interior.
    # ALM enforces T >= threshold_arg (the margin arg is just early-stop).
    # Feed a TIGHTER threshold so ALM lands strictly above the polish
    # log barrier's threshold.
    safety_margin = max(margin * 10.0, 0.005)
    if seed_min < threshold + safety_margin:
        remaining = max(60.0, time_budget_s * 0.35)
        r2 = alm.solve(seed, threshold=threshold + safety_margin, margin=1e-4,
                       anchor=anchor, outer_max=30,
                       inner_maxiter=150,
                       time_budget_s=remaining, verbose=0)
        seed = r2['phi_out']
        info['stage2_alm'] = dict(min_T=_min_tri(seed),
                                   L2=float(np.linalg.norm((seed - phi_in).ravel())),
                                   wall=time.time() - t0)
        if verbose:
            print(f'  stage2 ALM     min_T={_min_tri(seed):+.5f}  '
                  f'L2={info["stage2_alm"]["L2"]:.1f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
    else:
        info['stage2_alm'] = {'skipped': 'seed-already-feasible'}

    cur_min = _min_tri(seed)
    # Need strict interior (T_k > threshold) for the log-barrier polish.
    # If ALM ended at the boundary, do a tiny ALM bump.
    if cur_min <= threshold + 1e-6:
        r2b = alm.solve(seed, threshold=threshold + safety_margin,
                        margin=1e-4,
                        anchor='none', outer_max=10,
                        inner_maxiter=100,
                        time_budget_s=60.0, verbose=0)
        if _min_tri(r2b['phi_out']) > _min_tri(seed):
            seed = r2b['phi_out']
        cur_min = _min_tri(seed)
        info['stage2b_alm_bump'] = dict(min_T=cur_min,
                                         L2=float(np.linalg.norm(
                                             (seed - phi_in).ravel())))

    if cur_min <= threshold:
        # Polish can't start -- need strict feasibility for the barrier.
        info['stage3_polish'] = {'skipped': f'min_T={cur_min:.5f} not strict-feasible'}
        info['final_min_T'] = cur_min
        return {'phi_out': seed, 'info': info}

    # Stage 3: log-barrier L2 polish anchored to phi_in.
    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([seed[0].ravel(), seed[1].ravel()])
    polish_log = []
    for mu in mu_schedule:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(_barrier_anchored, phi_flat, jac=True,
                       method='L-BFGS-B',
                       args=(phi_in_flat, H, W, threshold, mu, anchor),
                       options=dict(maxiter=inner_maxiter, ftol=1e-12,
                                    gtol=1e-9))
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        min_T = float(T.min())
        phi_cur = np.stack([phi_flat[:H * W].reshape(H, W),
                            phi_flat[H * W:].reshape(H, W)])
        L2 = float(np.linalg.norm((phi_cur - phi_in).ravel()))
        polish_log.append(dict(mu=mu, min_T=min_T, L2=L2,
                                nit=int(res.nit), wall=time.time() - t0))
        if verbose:
            print(f'  stage3 mu={mu:.1e}  min_T={min_T:+.5f}  L2={L2:.1f}  '
                  f'nit={res.nit}  ({time.time()-t0:.1f}s)', flush=True)

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    info['stage3_polish'] = dict(steps=len(polish_log),
                                  final_min_T=_min_tri(phi_out),
                                  final_L2=float(np.linalg.norm(
                                      (phi_out - phi_in).ravel())),
                                  log_last3=polish_log[-3:])
    info['final_min_T'] = _min_tri(phi_out)
    return {'phi_out': phi_out, 'info': info}
