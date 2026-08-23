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
and ``m14_l1.py``. With ``objective=L1Objective()`` the entire pipeline uses a
smoothed-L1 anchor — this is the **m14_l1** variant in the manuscript.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt._logging import log_info, log_warning
from dvfopt.core.primitives.tri import tri_areas_flat as _tri_areas_flat
from dvfopt.core.wallbreakers._alm import augmented_lagrangian_2d
from dvfopt.core.wallbreakers._common import (
    barrier_anchored_objective as _barrier_anchored,
)
from dvfopt.core.wallbreakers._common import (
    min_tri as _min_tri,
)
from dvfopt.core.wallbreakers._common import (
    resolved_safety_margin,
)
from dvfopt.core.wallbreakers._harmonic import harmonic_extension_2d
from dvfopt.core.wallbreakers._harmonic_polished import iterative_2d_tri_harmonic_polished
from dvfopt.core.wallbreakers._l2_refine import l2_refine_2d
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.objectives import L2Objective, NoneObjective, Objective, _kind_eps


def iterative_2d_tri_refine_repair(
    phi_in: np.ndarray,
    *,
    threshold: Optional[float] = None,
    margin: float = 1e-3,
    objective: Objective | None = None,
    seed: np.ndarray = None,
    lam_schedule: tuple = (1e2, 1e4, 1e6, 1e8),
    inner_maxiter: int = 300,
    ring_pad: int = 2,
    max_grow_iters: int = 8,
    polish_mu: tuple = (1e-2, 1e-4, 1e-6),
    polish_maxiter: int = 200,
    stage1_mu_schedule: Optional[tuple] = None,
    time_budget_s: float = 600.0,
    verbose: int = 1,
    record_history: bool = False,
    step_callback=None,
):
    """Full 4-stage refine-repair pipeline.

    Parameters
    ----------
    phi_in : ndarray, shape ``(2, H, W)`` or ``(3, 1, H, W)``
        Input deformation field.
    objective : Objective or None
        Anchor objective; ``None`` (default) means
        :class:`~dvfopt.objectives.L2Objective`.
        :class:`~dvfopt.objectives.L1Objective` is the **m14_l1** variant (smoothed-L1,
        concentrates corrections into a few cells; far smaller L1
        deviation than L2).
    seed : ndarray, optional
        Pre-computed m10 seed (skips stage 1 if provided).
    lam_schedule, polish_mu : sequences
        Continuation schedules for stages 2 and 4.
    stage1_mu_schedule : tuple or None
        Barrier ``mu_schedule`` forwarded to the stage-1 m10 call
        (:func:`iterative_2d_tri_harmonic_polished`). ``None`` (default)
        keeps m10's own default schedule (legacy behavior). ``()`` skips
        m10's internal log-barrier polish entirely — useful when m14
        runs as a seed stage whose stage-2 ``l2_refine_2d`` (anchored to
        the same ``phi_in``) immediately redoes that slide-toward-input
        work. Ignored when ``seed`` is provided.

    Returns
    -------
    dict with keys ``phi_out`` (shape ``(2, H, W)``) and ``info``
    (per-stage statistics).
    """
    objective = objective or L2Objective()
    anchor, eps_l1 = _kind_eps(objective)
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

    def _fire(stage: str, phi):
        """Forward an intermediate phi snapshot to ``step_callback`` if
        a caller passed one. Each fired stage corresponds to a major
        pipeline boundary — the GUI consumes these for the history
        slider so the user can scrub through M14's internal steps.
        Buggy callbacks are silently swallowed (except KeyboardInterrupt,
        which is the documented stop-the-solver mechanism)."""
        if step_callback is None:
            return
        try:
            step_callback({'phi': np.asarray(phi).copy(), 'stage': stage})
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log_warning(f'step_callback raised {type(exc).__name__}: {exc}; continuing')

    # Stage 1: m10 seed.
    if seed is None:
        # Only override m10's mu_schedule when explicitly requested so
        # the default path stays byte-identical to the legacy behavior.
        _m10_kwargs = {}
        if stage1_mu_schedule is not None:
            _m10_kwargs['mu_schedule'] = tuple(stage1_mu_schedule)
        seed = iterative_2d_tri_harmonic_polished(
            phi_in,
            threshold=threshold,
            margin=margin,
            objective=objective,
            max_grow_iters=max_grow_iters,
            time_budget_s=time_budget_s * 0.4,
            verbose=verbose,
            **_m10_kwargs,
        )
    seed_L2 = float(np.linalg.norm((seed - phi_in).ravel()))
    seed_min = _min_tri(seed)
    info['stage1_seed'] = dict(min_T=seed_min, L2=seed_L2, wall=time.time() - t0)
    if verbose:
        log_info(
            f'  stage1 seed  min_T={seed_min:+.5f}  L2={seed_L2:.1f}  ({time.time() - t0:.1f}s)',
        )
    _fire('stage1_seed', seed)

    # Stage 2: soft-penalty L2 pull.
    remaining = max(60.0, time_budget_s - (time.time() - t0))
    pulled = l2_refine_2d(
        phi_in,
        seed=seed,
        threshold=threshold,
        margin=margin,
        objective=objective,
        lam_schedule=lam_schedule,
        inner_maxiter=inner_maxiter,
        time_budget_s=remaining * 0.5,
        verbose=verbose,
        require_feasibility=False,
    )
    pulled_L2 = float(np.linalg.norm((pulled - phi_in).ravel()))
    pulled_min = _min_tri(pulled)
    T1p, T2p = _triangle_areas_2d(pulled[0], pulled[1])
    pulled_neg = int((np.minimum(T1p, T2p) <= 0).sum())
    info['stage2_pull'] = dict(
        min_T=pulled_min, L2=pulled_L2, n_neg=pulled_neg, wall=time.time() - t0
    )
    if verbose:
        log_info(
            f'  stage2 pull  min_T={pulled_min:+.5f}  L2={pulled_L2:.1f}  '
            f'n_neg={pulled_neg}  ({time.time() - t0:.1f}s)',
        )
    _fire('stage2_pull', pulled)

    # Stage 3: harmonic repair of residual folds.
    if pulled_min < threshold:
        repaired = harmonic_extension_2d(
            pulled,
            threshold=threshold,
            ring_pad=ring_pad,
            max_grow_iters=max_grow_iters,
            margin=margin,
        )
    else:
        repaired = pulled
    repaired_min = _min_tri(repaired)
    repaired_L2 = float(np.linalg.norm((repaired - phi_in).ravel()))
    info['stage3_repair'] = dict(min_T=repaired_min, L2=repaired_L2, wall=time.time() - t0)
    if verbose:
        log_info(
            f'  stage3 patch min_T={repaired_min:+.5f}  '
            f'L2={repaired_L2:.1f}  ({time.time() - t0:.1f}s)',
        )
    _fire('stage3_repair', repaired)

    # If repair didn't reach feasibility, ALM nudge. Use the same
    # safety_margin formula as m10 so the two pipelines agree on the
    # ALM target when ``margin`` is non-default.
    safety_margin = resolved_safety_margin(margin)
    if repaired_min < threshold + 1e-6:
        bumped = augmented_lagrangian_2d(
            repaired,
            threshold=threshold + safety_margin,
            margin=1e-4,
            objective=NoneObjective(),
            outer_max=20,
            inner_maxiter=150,
            time_budget_s=max(30.0, time_budget_s - (time.time() - t0)),
            verbose=0,
        )
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
            _barrier_anchored,
            phi_flat,
            jac=True,
            method='L-BFGS-B',
            args=(phi_in_flat, H, W, threshold, mu, anchor, eps_l1),
            options=dict(maxiter=polish_maxiter, ftol=1e-12, gtol=1e-9),
        )
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        polish_log.append(
            dict(
                mu=mu,
                min_T=float(T.min()),
                L2=float(np.linalg.norm(phi_flat - phi_in_flat)),
                nit=int(res.nit),
                wall=time.time() - t0,
            )
        )
        if verbose:
            log_info(
                f'  stage4 mu={mu:.1e}  min_T={T.min():+.5f}  '
                f'L2={float(np.linalg.norm(phi_flat - phi_in_flat)):.1f}  '
                f'nit={res.nit}  ({time.time() - t0:.1f}s)',
            )
        # Emit one snapshot per μ-step so the GUI can scrub through the
        # polish loop. Unflatten phi_flat back to (2, H, W).
        _fire(
            f'stage4_polish_mu={mu:g}',
            np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)]),
        )

    phi_out = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
    info['stage4_polish'] = dict(
        steps=len(polish_log),
        log_last3=polish_log[-3:],
        final_min_T=_min_tri(phi_out),
        final_L2=float(np.linalg.norm((phi_out - phi_in).ravel())),
    )
    info['final_min_T'] = _min_tri(phi_out)
    info['final_L2'] = float(np.linalg.norm((phi_out - phi_in).ravel()))
    return (phi_out, info) if record_history else phi_out
