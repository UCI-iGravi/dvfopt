"""LP-direct strategies for strict 6-tetrahedron feasibility (3D analog
of :mod:`dvfopt.core.slp.lp_direct_2tri`), promoted from
``research/strict_feasibility_3d``.

Public API:
  * :func:`lp_oneshot` — single LP linearised around a feasible seed.
  * :func:`slp_iter`  — sequential LP loop with adaptive trust region.

Both minimise ``||phi - phi_in||_1`` over the 3D phi pack
(``[dx, dy, dz]`` flat, length ``3 * D * H * W``) subject to:

  - ``T_k(phi) >= threshold`` for every per-cell tetrahedron (linearised)
  - optional trust region ``||phi - phi_lin||_inf <= trust_radius`` (SLP)

The HiGHS L1 solver (:func:`dvfopt.core.slp.highs_solver.solve_l1_lp_step`)
is dimension-agnostic; we reuse it directly here.
"""

from __future__ import annotations

import time

import numpy as np

from dvfopt.core.slp.highs_solver import solve_l1_lp_step
from dvfopt.jacobian.tetrahedron_sign import (
    build_tet_sparse_jac,
    six_tet_min_volume_3d,
    tet_volumes_flat,
)

# ---------- phi pack helpers ----------


def _phi3hw_to_flat(phi_3dhw: np.ndarray) -> np.ndarray:
    """``(3, D, H, W)`` ``[dz, dy, dx]`` -> flat ``[dx, dy, dz]`` (DX_FIRST)."""
    dz, dy, dx = phi_3dhw
    return np.concatenate([dx.ravel(), dy.ravel(), dz.ravel()])


def _flat_to_phi3dhw(phi_flat: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
    n = D * H * W
    dx = phi_flat[:n].reshape(D, H, W)
    dy = phi_flat[n : 2 * n].reshape(D, H, W)
    dz = phi_flat[2 * n :].reshape(D, H, W)
    return np.stack([dz, dy, dx])


def _exact_min_T(phi_3dhw: np.ndarray) -> float:
    # Fused per-cell-min kernel (never materialises the (6, ...) array).
    return float(six_tet_min_volume_3d(phi_3dhw).min())


# ---------- seeds ----------


def _tet_seed(phi_in_3dhw: np.ndarray, threshold: float, strategy) -> np.ndarray:
    """Feasible seed via a 3D wallbreaker pipeline (one helper for every
    seed kind — the recipe lives in ONE place)."""
    from dvfopt import L1Objective, Solver, Tet6Constraint3D

    D, H, W = phi_in_3dhw.shape[1:]
    solver = Solver(
        constraint=Tet6Constraint3D(shape=(D, H, W)),
        objective=L1Objective(eps=1e-4),
        strategy=strategy,
        threshold=threshold,
    )
    return solver.fit(phi_in_3dhw).corrected


def _seed_strategy(kind: str):
    """Strategy instance for a named seed kind.

    * ``'m10'`` — harmonic + ALM + barrier polish (the validated 3D
      default; m14 catastrophically overshoots on dense 3D folds).
    * ``'m10_fast'`` — m10 without the barrier polish (~50% of m10 wall
      on small crops; the outer SLP re-polishes L1 anyway).
    * ``'m14'`` — full refine-repair pipeline (closest-to-input seed).
    """
    from dvfopt import HarmonicALMBarrier3DStrategy, HarmonicALMRefineRepair3DStrategy

    return {
        'm10': HarmonicALMBarrier3DStrategy(),
        'm10_fast': HarmonicALMBarrier3DStrategy(polish=False),
        'm14': HarmonicALMRefineRepair3DStrategy(),
    }[kind]


def _build_seed(phi_in_3dhw: np.ndarray, threshold: float, seed) -> np.ndarray:
    if isinstance(seed, np.ndarray):
        if seed.shape != phi_in_3dhw.shape:
            raise ValueError(f'seed shape {seed.shape} != phi_in shape {phi_in_3dhw.shape}')
        return seed.astype(np.float64)  # astype always copies
    if seed == 'zero':
        return np.zeros_like(phi_in_3dhw)
    try:
        strategy = _seed_strategy(seed)
    except KeyError:
        raise ValueError(f'unknown seed: {seed!r}') from None
    return _tet_seed(phi_in_3dhw, threshold, strategy)


# ---------- LP / SLP public API ----------


def lp_oneshot(
    phi_in_3dhw: np.ndarray,
    *,
    threshold: float = 0.01,
    seed: str = 'm10',
):
    """Single-LP linearised around ``seed``.

    Parameters
    ----------
    phi_in_3dhw : (3, D, H, W) float64
    threshold : float
    seed : {'m10', 'm14', 'zero', ndarray}
    """
    t0 = time.time()
    D, H, W = phi_in_3dhw.shape[1:]
    seed_phi = _build_seed(phi_in_3dhw, threshold, seed)

    phi_in_flat = _phi3hw_to_flat(phi_in_3dhw)
    phi_lin_flat = _phi3hw_to_flat(seed_phi)
    T_lin = tet_volumes_flat(phi_lin_flat, D, H, W)
    jac = build_tet_sparse_jac(D, H, W)
    J = jac(phi_lin_flat)
    phi_out_flat, status = solve_l1_lp_step(
        phi_in_flat=phi_in_flat,
        phi_lin_flat=phi_lin_flat,
        T_lin=T_lin,
        J_sparse=J,
        threshold=threshold,
        trust_radius=None,
    )
    phi_out = _flat_to_phi3dhw(phi_out_flat, D, H, W)
    info = {
        'seed': seed,
        'seed_min_T_exact': _exact_min_T(seed_phi),
        'final_min_T_exact': _exact_min_T(phi_out),
        'L1_dev': float(np.abs(phi_out - phi_in_3dhw).sum()),
        'lp_status': status,
        'wall_s': time.time() - t0,
    }
    return phi_out, info


def slp_iter(
    phi_in_3dhw: np.ndarray,
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
    """Sequential LP loop with adaptive trust region (3D)."""
    t0 = time.time()
    D, H, W = phi_in_3dhw.shape[1:]
    phi_in_flat = _phi3hw_to_flat(phi_in_3dhw)

    seed_phi = _build_seed(phi_in_3dhw, threshold, seed)
    phi_cur_flat = _phi3hw_to_flat(seed_phi)
    trust_radius = float(trust_radius_0)

    jac = build_tet_sparse_jac(D, H, W)
    best_phi_flat = phi_cur_flat.copy()
    best_L1 = float(np.abs(seed_phi - phi_in_3dhw).sum())
    best_feasible = _exact_min_T(seed_phi) >= threshold - safety_tol

    iters = 0
    converged = False
    statuses = []

    for it in range(max_iter):
        iters = it + 1
        T_lin = tet_volumes_flat(phi_cur_flat, D, H, W)
        J = jac(phi_cur_flat)
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

        phi_new_3dhw = _flat_to_phi3dhw(phi_new_flat, D, H, W)
        exact_min = _exact_min_T(phi_new_3dhw)
        new_L1 = float(np.abs(phi_new_3dhw - phi_in_3dhw).sum())

        if exact_min < threshold - safety_tol:
            trust_radius *= trust_shrink
            if trust_radius < 1e-8:
                break
            continue

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

    phi_out = _flat_to_phi3dhw(best_phi_flat, D, H, W)
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
