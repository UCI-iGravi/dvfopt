"""4-stage refinement of an m10 seed with residual-fold repair.

This is the **L2/L1 winner** (``m14`` in the wall_breakers experiment
suite). On the original B0039 DVF it achieves 528/528 feasibility AND
cuts the mean L2 by ~50% vs m10 alone (and the L1 by ~80% with the L1
anchor).

The reasoning behind each stage:

* m10 produces a feasible seed but it sits exactly at the constraint
  boundary, so any global step toward ``phi_in`` immediately violates.
* A soft-quadratic penalty (m12) can move non-active cells back toward
  ``phi_in`` (huge L2 reduction) while leaving active cells weakly
  pinned; this is fast and stable but does not reach strict
  feasibility in finite ``lambda``.
* The residual folds are tiny LOCALISED clusters where sliding back
  would over-stretch. A harmonic patch fixes them at negligible L2.
* A final log-barrier polish anchored to ``phi_in`` lands strictly
  inside the feasible interior at the L2-optimum of the central path.

Pipeline
--------
1. m10 seed (already feasible).
2. Soft-quadratic-penalty L-BFGS-B (:func:`l2_refine_2d`) anchored to
   ``phi_in`` — L2 ≈ 50 on dense, slightly infeasible.
3. Harmonic repair of residual folds (negligible L2 bump).
4. Final log-barrier L2 (or L1) polish anchored to ``phi_in``.

Promoted from ``notebooks/experiments/wall_breakers/methods/m14_l2_refine_repair.py``
and ``m14_l1.py``. With ``anchor='l1'`` the entire pipeline uses a
smoothed-L1 anchor — this is the **m14_l1** variant in the manuscript.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core.tri_primitives import tri_areas_flat as _tri_areas_flat
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from dvfopt.core.wallbreakers._harmonic import harmonic_extension_2d
from dvfopt.core.wallbreakers._alm import augmented_lagrangian_2d
from dvfopt.core.wallbreakers._harmonic_polished import (
    iterative_2d_tri_harmonic_polished)
from dvfopt.core.wallbreakers._l2_refine import l2_refine_2d
from dvfopt.core.wallbreakers._common import (
    min_tri as _min_tri,
    barrier_anchored_objective as _barrier_anchored,
    resolved_safety_margin,
)


def iterative_2d_tri_refine_repair(
    phi_in: np.ndarray,
    *,
    threshold: float = None,
    margin: float = 1e-3,
    anchor: str = 'l2',
    seed: np.ndarray = None,
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8),
    inner_maxiter: int = 300,
    ring_pad: int = 2,
    polish_mu: tuple = (1e-2, 1e-4, 1e-6),
    polish_maxiter: int = 200,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    eps_l1: float = 1e-4,
    record_history: bool = False,
):
    """Full 4-stage refine-repair pipeline.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)`` or ``(3, 1, H, W)``
        Input deformation field.
    anchor : {'l2', 'l1'}
        Objective anchor. ``'l1'`` is the **m14_l1** variant (smoothed-L1,
        concentrates corrections into a few cells; far smaller L1
        deviation than L2).
    seed : ndarray, optional
        Pre-computed m10 seed (skips stage 1 if provided).
    lam_schedule, polish_mu : sequences
        Continuation schedules for stages 2 and 4.
    eps_l1 : float
        Smoothing constant for the L1 anchor.

    Returns
    -------
    dict with keys ``phi_out`` (shape ``(2, H, W)``) and ``info``
    (per-stage statistics).
    """
    # Coerce input shape.
    if phi_in.ndim == 4:
        if phi_in.shape[0] == 3:
            phi_in = np.stack([phi_in[1, 0], phi_in[2, 0]])
        else:
            phi_in = phi_in[:, 0]
    if phi_in.dtype != np.float64:
        phi_in = phi_in.astype(np.float64)

    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    H, W = phi_in.shape[1], phi_in.shape[2]
    t0 = time.time()
    info: dict = {}

    # Stage 1: m10 seed.
    if seed is None:
        seed = iterative_2d_tri_harmonic_polished(
            phi_in, threshold=threshold, margin=margin,
            anchor=anchor, eps_l1=eps_l1,
            time_budget_s=time_budget_s * 0.4,
            verbose=verbose)
    seed_L2 = float(np.linalg.norm((seed - phi_in).ravel()))
    seed_min = _min_tri(seed)
    info['stage1_seed'] = dict(min_T=seed_min, L2=seed_L2,
                                wall=time.time() - t0)
    if verbose:
        print(f'  stage1 seed  min_T={seed_min:+.5f}  L2={seed_L2:.1f}  '
              f'({time.time()-t0:.1f}s)', flush=True)

    # Stage 2: soft-penalty L2 pull.
    remaining = max(60.0, time_budget_s - (time.time() - t0))
    pulled = l2_refine_2d(
        phi_in, seed=seed, threshold=threshold, margin=margin,
        anchor=anchor, lam_schedule=lam_schedule,
        inner_maxiter=inner_maxiter,
        time_budget_s=remaining * 0.5,
        verbose=verbose, eps_l1=eps_l1,
        require_feasibility=False)
    pulled_L2 = float(np.linalg.norm((pulled - phi_in).ravel()))
    pulled_min = _min_tri(pulled)
    T1p, T2p = _triangle_areas_2d(pulled[0], pulled[1])
    pulled_neg = int((np.minimum(T1p, T2p) <= 0).sum())
    info['stage2_pull'] = dict(min_T=pulled_min, L2=pulled_L2,
                                n_neg=pulled_neg, wall=time.time() - t0)
    if verbose:
        print(f'  stage2 pull  min_T={pulled_min:+.5f}  L2={pulled_L2:.1f}  '
              f'n_neg={pulled_neg}  ({time.time()-t0:.1f}s)', flush=True)

    # Stage 3: harmonic repair of residual folds.
    if pulled_min < threshold:
        repaired = harmonic_extension_2d(
            pulled, threshold=threshold, ring_pad=ring_pad,
            max_grow_iters=8, margin=margin)
    else:
        repaired = pulled
    repaired_min = _min_tri(repaired)
    repaired_L2 = float(np.linalg.norm((repaired - phi_in).ravel()))
    info['stage3_repair'] = dict(min_T=repaired_min, L2=repaired_L2,
                                  wall=time.time() - t0)
    if verbose:
        print(f'  stage3 patch min_T={repaired_min:+.5f}  '
              f'L2={repaired_L2:.1f}  ({time.time()-t0:.1f}s)', flush=True)

    # If repair didn't reach feasibility, ALM nudge. Use the same
    # safety_margin formula as m10 so the two pipelines agree on the
    # ALM target when ``margin`` is non-default.
    safety_margin = resolved_safety_margin(margin)
    if repaired_min < threshold + 1e-6:
        bumped = augmented_lagrangian_2d(
            repaired, threshold=threshold + safety_margin, margin=1e-4,
            anchor='none', outer_max=20, inner_maxiter=150,
            time_budget_s=max(30.0, time_budget_s - (time.time() - t0)),
            verbose=0)
        if _min_tri(bumped) > repaired_min:
            repaired = bumped
            repaired_min = _min_tri(repaired)
            repaired_L2 = float(np.linalg.norm((repaired - phi_in).ravel()))
            info['stage3b_alm'] = dict(min_T=repaired_min, L2=repaired_L2)

    if repaired_min <= threshold:
        info['final_min_T'] = repaired_min
        info['final_L2'] = repaired_L2
        info['stage4_polish'] = {'skipped': 'still-not-strict-interior'}
        return (repaired, info) if record_history else repaired

    # Stage 4: log-barrier polish anchored to phi_in.
    phi_in_flat = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    phi_flat = np.concatenate([repaired[0].ravel(), repaired[1].ravel()])
    polish_log = []
    for mu in polish_mu:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _barrier_anchored, phi_flat, jac=True, method='L-BFGS-B',
            args=(phi_in_flat, H, W, threshold, mu, anchor, eps_l1),
            options=dict(maxiter=polish_maxiter, ftol=1e-12, gtol=1e-9))
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        polish_log.append(dict(
            mu=mu, min_T=float(T.min()),
            L2=float(np.linalg.norm(phi_flat - phi_in_flat)),
            nit=int(res.nit), wall=time.time() - t0))
        if verbose:
            print(f'  stage4 mu={mu:.1e}  min_T={T.min():+.5f}  '
                  f'L2={float(np.linalg.norm(phi_flat - phi_in_flat)):.1f}  '
                  f'nit={res.nit}  ({time.time()-t0:.1f}s)', flush=True)

    phi_out = np.stack([phi_flat[:H * W].reshape(H, W),
                        phi_flat[H * W:].reshape(H, W)])
    info['stage4_polish'] = dict(
        steps=len(polish_log), log_last3=polish_log[-3:],
        final_min_T=_min_tri(phi_out),
        final_L2=float(np.linalg.norm((phi_out - phi_in).ravel())))
    info['final_min_T'] = _min_tri(phi_out)
    info['final_L2'] = float(np.linalg.norm((phi_out - phi_in).ravel()))
    return (phi_out, info) if record_history else phi_out
