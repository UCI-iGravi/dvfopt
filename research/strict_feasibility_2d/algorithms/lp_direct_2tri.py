"""LP-direct strategies for strict 2-tri feasibility.

Two variants:

* :func:`lp_oneshot` -- single LP linearised around a feasible harmonic seed.
  May leave a small residual fold at exact eval due to linearisation error.
* :func:`slp_iter`  -- sequential LP loop with adaptive trust region.
  Iterates until exact-T feasibility holds. Guaranteed feasible at
  termination (or returns the best iterate with a non-converged flag).

Both minimise ``||phi - phi_in||_1`` and return ``(phi_out, info)``.
"""
from __future__ import annotations

import time

import numpy as np

from research.strict_feasibility_2d.algorithms.highs_solver import solve_l1_lp_step
from research.strict_feasibility_2d.algorithms.tri_linearize import linearize_T_2tri


def _exact_min_T(phi_2hw: np.ndarray) -> float:
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    return float(np.minimum(T1, T2).min())


def _flatten(phi_2hw: np.ndarray) -> np.ndarray:
    return np.concatenate([phi_2hw[0].ravel(), phi_2hw[1].ravel()])


def _unflatten(phi_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    HW = H * W
    return np.stack([phi_flat[:HW].reshape(H, W), phi_flat[HW:].reshape(H, W)])


def _harmonic_seed(phi_in_2hw: np.ndarray, threshold: float) -> np.ndarray:
    """Cheap feasible-extension seed via Laplacian extension of fold cores.

    Works on small-displacement / mild-fold cases but leaves residual
    folds on dense canonical 10x10 / 20x20 inputs -- use ``_m10_seed``
    for guaranteed feasibility there.
    """
    from dvfopt.core.wallbreakers import harmonic_extension_2d
    return harmonic_extension_2d(phi_in_2hw, threshold=threshold)


def _m10_seed(phi_in_2hw: np.ndarray, threshold: float) -> np.ndarray:
    """Strict-interior feasible seed via the full m10 pipeline
    (harmonic + ALM + barrier polish).

    Slower than ``_harmonic_seed`` but guarantees ``min_T >= threshold``
    on every case where m10 itself reaches feasibility -- i.e. every
    case in the worst-case catalog. Triggers spec fallback row 1.
    """
    from dvfopt import (
        HarmonicALMBarrierStrategy,
        L1Objective,
        Solver,
        TriConstraint2DFullCoverage,
    )

    H, W = phi_in_2hw.shape[1:]
    solver = Solver(
        constraint=TriConstraint2DFullCoverage(shape=(H, W)),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMBarrierStrategy(),
        threshold=threshold,
    )
    return solver.fit(phi_in_2hw).corrected


def _m14_seed(phi_in_2hw: np.ndarray, threshold: float) -> np.ndarray:
    """Strict-interior seed via the full m14 pipeline (m10 seed +
    L2-refine + repair + barrier polish).

    The closest-to-``phi_in`` seed available — m14's L2-refine stage
    pulls back to the input as much as feasibility allows. SLP from
    this seed should match or improve on M14's L1, never worse.
    """
    from dvfopt import (
        HarmonicALMRefineRepairStrategy,
        L1Objective,
        Solver,
        TriConstraint2DFullCoverage,
    )

    H, W = phi_in_2hw.shape[1:]
    solver = Solver(
        constraint=TriConstraint2DFullCoverage(shape=(H, W)),
        objective=L1Objective(eps=1e-4),
        strategy=HarmonicALMRefineRepairStrategy(),
        threshold=threshold,
    )
    return solver.fit(phi_in_2hw).corrected


def _build_seed(phi_in_2hw: np.ndarray, threshold: float, seed) -> np.ndarray:
    """Dispatch on the ``seed`` kwarg shared by ``lp_oneshot`` + ``slp_iter``.

    Accepts a string kind (``'harmonic'`` / ``'m10'`` / ``'m14'`` /
    ``'zero'``) or a ndarray to use as the seed directly (useful for
    chaining: ``cluster_slp`` passes its output here as the seed for a
    global polish step)."""
    if isinstance(seed, np.ndarray):
        if seed.shape != phi_in_2hw.shape:
            raise ValueError(
                f'seed ndarray shape {seed.shape} != phi_in shape {phi_in_2hw.shape}'
            )
        return seed.astype(np.float64).copy()
    if seed == 'harmonic':
        return _harmonic_seed(phi_in_2hw, threshold)
    if seed == 'm10':
        return _m10_seed(phi_in_2hw, threshold)
    if seed == 'm14':
        return _m14_seed(phi_in_2hw, threshold)
    if seed == 'zero':
        return np.zeros_like(phi_in_2hw)
    raise ValueError(f'unknown seed: {seed!r}')


def lp_oneshot(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    seed: str = 'm10',
):
    """Single-LP linearised around ``seed``.

    Parameters
    ----------
    phi_in_2hw : (2, H, W) float64
    threshold : float
    seed : {'m10', 'harmonic', 'zero'}
        ``'m10'`` (default) — full m10 pipeline (harmonic + ALM +
        barrier polish). Strict-interior feasibility on every case in
        the worst-case catalog. Closes the linearisation-error gap on
        dense canonical cases where ``'harmonic'`` alone falls short.
        ``'harmonic'`` — cheap Laplacian extension. Faster but leaves
        residual folds on dense inputs.
        ``'zero'`` — linearise around ``phi = 0``. Ablation only.

    Returns
    -------
    phi_out_2hw : (2, H, W) float64
    info : dict
    """
    t0 = time.time()
    H, W = phi_in_2hw.shape[1:]
    seed_phi = _build_seed(phi_in_2hw, threshold, seed)

    phi_in_flat = _flatten(phi_in_2hw)
    phi_lin_flat = _flatten(seed_phi)
    T_lin, J = linearize_T_2tri(phi_lin_flat, H, W)
    phi_out_flat, status = solve_l1_lp_step(
        phi_in_flat=phi_in_flat,
        phi_lin_flat=phi_lin_flat,
        T_lin=T_lin,
        J_sparse=J,
        threshold=threshold,
        trust_radius=None,
    )
    phi_out = _unflatten(phi_out_flat, H, W)
    info = {
        'seed': seed,
        'seed_min_T_exact': _exact_min_T(seed_phi),
        'final_min_T_exact': _exact_min_T(phi_out),
        'L1_dev': float(np.abs(phi_out - phi_in_2hw).sum()),
        'lp_status': status,
        'wall_s': time.time() - t0,
    }
    return phi_out, info


def slp_iter(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    safety_tol: float = 1e-5,
    trust_radius_0: float = 0.5,
    max_iter: int = 20,
    ftol: float = 1e-6,
    trust_grow: float = 1.5,
    trust_shrink: float = 0.5,
    seed: str = 'm10',
):
    """Sequential LP loop with adaptive trust region.

    Termination: returns when either
      a) max-norm step < ``ftol`` AND exact ``min_T >= threshold - safety_tol``
         (converged), or
      b) ``max_iter`` reached (returns the best feasible iterate seen, or
         the seed if no LP step was accepted).

    On exact-T infeasibility after an LP step the trust radius is halved
    and the iterate is rejected. On feasibility + step at the trust
    boundary the radius is grown by ``trust_grow``.
    """
    t0 = time.time()
    H, W = phi_in_2hw.shape[1:]
    phi_in_flat = _flatten(phi_in_2hw)

    seed_phi = _build_seed(phi_in_2hw, threshold, seed)
    phi_cur_flat = _flatten(seed_phi)
    trust_radius = float(trust_radius_0)

    best_phi_flat = phi_cur_flat.copy()
    best_L1 = float(np.abs(seed_phi - phi_in_2hw).sum())
    best_feasible = _exact_min_T(seed_phi) >= threshold - safety_tol

    iters = 0
    converged = False
    statuses = []

    for it in range(max_iter):
        iters = it + 1
        T_lin, J = linearize_T_2tri(phi_cur_flat, H, W)
        phi_new_flat, status = solve_l1_lp_step(
            phi_in_flat=phi_in_flat,
            phi_lin_flat=phi_cur_flat,
            T_lin=T_lin,
            J_sparse=J,
            threshold=threshold,
            trust_radius=trust_radius,
        )
        statuses.append(status)
        if not status['success']:
            trust_radius *= trust_shrink
            if trust_radius < 1e-8:
                break
            continue

        phi_new_2hw = _unflatten(phi_new_flat, H, W)
        exact_min = _exact_min_T(phi_new_2hw)
        new_L1 = float(np.abs(phi_new_2hw - phi_in_2hw).sum())

        if exact_min < threshold - safety_tol:
            # Linearisation error: shrink trust region, reject step.
            trust_radius *= trust_shrink
            if trust_radius < 1e-8:
                break
            continue

        # Accept step.
        step_inf = float(np.max(np.abs(phi_new_flat - phi_cur_flat)))
        at_boundary = step_inf >= 0.99 * trust_radius
        phi_cur_flat = phi_new_flat
        if new_L1 <= best_L1 + 1e-12:
            best_phi_flat = phi_cur_flat.copy()
            best_L1 = new_L1
            best_feasible = True
        if step_inf < ftol:
            converged = True
            break
        if at_boundary:
            trust_radius *= trust_grow

    phi_out = _unflatten(best_phi_flat, H, W)
    info = {
        'iters': iters,
        'converged': converged,
        'L1_dev': best_L1,
        'final_min_T_exact': _exact_min_T(phi_out),
        'feasible_at_exact_eval': best_feasible,
        'lp_statuses': statuses,
        'trust_radius_final': trust_radius,
        'wall_s': time.time() - t0,
    }
    return phi_out, info
