"""4-stage refine-repair pipeline for the 6-tet constraint (m14-3D).

3D analog of :mod:`dvfopt.core.wallbreakers._refine_repair`. Same staged
structure, just with the 3D primitives in place of 2D:

1. **Seed**: m10-3D = harmonic patch + ALM tightening. Always feasible
   (Radó-Kneser-Choquet for the harmonic patch; ALM smoothly pulls toward
   ``phi_in`` from there).
2. **Soft-penalty pull** (``l2_refine_3d``): anneal a one-sided quadratic
   penalty while anchoring to ``phi_in``. Massive L2 reduction on the
   non-active cells; weakly pins active cells.
3. **Harmonic repair**: if stage 2 created residual folds, patch them
   with a harmonic Dirichlet fill. Cheap and L2-negligible.
4. **Barrier polish**: log-barrier L-BFGS-B from the feasible repaired
   iterate, anchored to ``phi_in``, lands strictly in the feasible
   interior at the L2-optimum of the central path.

Used directly by :class:`dvfopt.strategies.wallbreakers.M14TetStrategy`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import log_info
from dvfopt.core.wallbreakers._alm_3d import augmented_lagrangian_3d
from dvfopt.core.wallbreakers._harmonic_3d import harmonic_extension_3d
from dvfopt.core.wallbreakers._l2_refine_3d import l2_refine_3d
from dvfopt.jacobian.tetrahedron_sign import tet_grad_T_v, tet_volumes_flat
from dvfopt.objectives import L2Objective, Objective, _kind_eps


def _barrier_anchored_objective_3d(
    phi_flat, phi_anchor_flat, D, H, W, threshold, mu, anchor, eps_l1
):
    """Log-barrier objective anchored to ``phi_anchor_flat`` (stage 4 polish)."""
    diff = phi_flat - phi_anchor_flat
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
    V = tet_volumes_flat(phi_flat, D, H, W)
    slack = V - threshold
    if (slack <= 0).any():
        # Iterate fell out of the feasible interior — return +inf so
        # L-BFGS-B rejects this step (safer than evaluating log(<=0)).
        return np.inf, grad
    val -= mu * float(np.log(slack).sum())
    grad = grad - mu * tet_grad_T_v(phi_flat, D, H, W, 1.0 / slack)
    return val, grad


def _min_V(phi):
    """Convenience: minimum per-tet signed volume over the whole field.

    Uses the fused per-cube min kernel — ``min(min over tets per cube)``
    equals the global per-tet minimum, without materialising the full
    ``(6, Dc, Hc, Wc)`` volume array.
    """
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    return float(six_tet_min_volume_3d(phi).min())


def iterative_3d_tet_refine_repair(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    objective: Objective | None = None,
    seed: Optional[np.ndarray] = None,
    # stage 1 (harmonic + ALM seed)
    ring_pad: int = 2,
    max_grow_iters: int = 6,
    merge_dilation: int = 2,
    alm_outer_max: int = 30,
    alm_inner_maxiter: int = 200,
    # stage 2 (l2 refine)
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8),
    inner_maxiter: int = 300,
    # stage 4 (polish)
    polish_mu: tuple = (1e-2, 1e-4, 1e-6),
    polish_maxiter: int = 200,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
    step_callback=None,
):
    """Full 4-stage 3D m14 refine-repair pipeline.

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``.
    info : dict, only if ``record_history=True`` — per-stage stats.
    """
    objective = objective or L2Objective()
    anchor, eps_l1 = _kind_eps(objective)
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']

    phi_in = np.asarray(phi_in, dtype=np.float64)
    if phi_in.ndim != 4 or phi_in.shape[0] != 3:
        raise ValueError(f'expected (3, D, H, W) input; got shape {phi_in.shape}')
    _, D, H, W = phi_in.shape

    def _emit(phi_stage, stage):
        if step_callback is not None:
            step_callback({'phi': phi_stage, 'stage': stage})

    t0 = time.time()
    info: dict = {}

    # -----------------------------------------------------------------
    # Stage 1: m10-3D seed (harmonic patch + ALM tightening).
    # -----------------------------------------------------------------
    if seed is None:
        phi_h = harmonic_extension_3d(
            phi_in,
            threshold=threshold,
            ring_pad=ring_pad,
            max_grow_iters=max_grow_iters,
            merge_dilation=merge_dilation,
            margin=margin,
        )
        seed = augmented_lagrangian_3d(
            phi_h,
            threshold=threshold,
            margin=margin,
            objective=objective,
            phi_anchor=phi_in,
            outer_max=alm_outer_max,
            inner_maxiter=alm_inner_maxiter,
            time_budget_s=time_budget_s * 0.4,
            verbose=0,
        )
    seed_min = _min_V(seed)
    seed_L2 = float(np.linalg.norm((seed - phi_in).ravel()))
    info['stage1_seed'] = dict(min_T=seed_min, L2=seed_L2, wall=time.time() - t0)
    _emit(seed, 'seed')
    if verbose:
        log_info(
            f'  stage1 seed  min_V={seed_min:+.5f}  L2={seed_L2:.3f}  ({time.time() - t0:.1f}s)',
        )

    # -----------------------------------------------------------------
    # Stage 2: soft-penalty pull (l2_refine_3d).
    # -----------------------------------------------------------------
    remaining = max(60.0, time_budget_s - (time.time() - t0))
    pulled = l2_refine_3d(
        phi_in,
        seed=seed,
        threshold=threshold,
        margin=margin,
        objective=objective,
        lam_schedule=lam_schedule,
        inner_maxiter=inner_maxiter,
        time_budget_s=remaining * 0.5,
        verbose=0,
        require_feasibility=False,
    )
    pulled_min = _min_V(pulled)
    pulled_L2 = float(np.linalg.norm((pulled - phi_in).ravel()))
    info['stage2_pull'] = dict(min_T=pulled_min, L2=pulled_L2, wall=time.time() - t0)
    _emit(pulled, 'pull')
    if verbose:
        log_info(
            f'  stage2 pull  min_V={pulled_min:+.5f}  L2={pulled_L2:.3f}  '
            f'({time.time() - t0:.1f}s)',
        )

    # -----------------------------------------------------------------
    # Stage 3: harmonic repair of residual folds (if any).
    # -----------------------------------------------------------------
    if pulled_min < threshold:
        repaired = harmonic_extension_3d(
            pulled,
            threshold=threshold,
            ring_pad=ring_pad,
            max_grow_iters=max_grow_iters,
            merge_dilation=merge_dilation,
            margin=margin,
        )
    else:
        repaired = pulled
    repaired_min = _min_V(repaired)
    repaired_L2 = float(np.linalg.norm((repaired - phi_in).ravel()))
    info['stage3_repair'] = dict(
        triggered=pulled_min < threshold,
        min_T=repaired_min,
        L2=repaired_L2,
        wall=time.time() - t0,
    )
    _emit(repaired, 'repair')
    if verbose:
        log_info(
            f'  stage3 patch min_V={repaired_min:+.5f}  L2={repaired_L2:.3f}  '
            f'({time.time() - t0:.1f}s)',
        )

    if repaired_min <= threshold:
        # Not strictly inside the feasible interior — barrier would
        # produce inf in stage 4. Return the best we have.
        info['final_min_T'] = repaired_min
        info['final_L2'] = repaired_L2
        info['stage4_polish'] = {'skipped': 'still-not-strict-interior'}
        _emit(repaired, 'polish')
        return (repaired, info) if record_history else repaired

    # -----------------------------------------------------------------
    # Stage 4: log-barrier polish anchored to phi_in.
    # -----------------------------------------------------------------
    phi_anchor_flat = np.concatenate([phi_in[2].ravel(), phi_in[1].ravel(), phi_in[0].ravel()])
    phi_flat = np.concatenate([repaired[2].ravel(), repaired[1].ravel(), repaired[0].ravel()])
    polish_log = []
    for mu in polish_mu:
        if time.time() - t0 > time_budget_s:
            break
        res = minimize(
            _barrier_anchored_objective_3d,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_anchor_flat, D, H, W, threshold, mu, anchor, eps_l1),
            options=dict(maxiter=polish_maxiter, ftol=1e-12, gtol=1e-9),
        )
        # If the inner returned inf (iterate stepped out of the interior),
        # keep the last good phi_flat.
        if np.isfinite(res.fun):
            phi_flat = res.x
        V = tet_volumes_flat(phi_flat, D, H, W)
        polish_log.append(
            dict(
                mu=float(mu),
                min_T=float(V.min()),
                L2=float(np.linalg.norm(phi_flat - phi_anchor_flat)),
                nit=int(res.nit),
                wall=time.time() - t0,
            )
        )
        if verbose:
            log_info(
                f'  stage4 mu={mu:.1e}  min_V={V.min():+.5f}  '
                f'L2={float(np.linalg.norm(phi_flat - phi_anchor_flat)):.3f}  '
                f'nit={res.nit}  ({time.time() - t0:.1f}s)',
            )

    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    phi_out = np.stack([dz, dy, dx])
    final_min = _min_V(phi_out)
    final_L2 = float(np.linalg.norm((phi_out - phi_in).ravel()))
    info['stage4_polish'] = dict(
        steps=len(polish_log),
        log_last3=polish_log[-3:],
        final_min_T=final_min,
        final_L2=final_L2,
    )
    info['final_min_T'] = final_min
    info['final_L2'] = final_L2
    _emit(phi_out, 'polish')

    if record_history:
        return phi_out, info
    return phi_out


__all__ = ['iterative_3d_tet_refine_repair']
