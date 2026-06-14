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
    """Feasible-by-construction seed via Laplacian extension of fold cores."""
    from dvfopt.core.wallbreakers import harmonic_extension_2d
    return harmonic_extension_2d(phi_in_2hw, threshold=threshold)


def lp_oneshot(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    seed: str = 'harmonic',
):
    """Single-LP linearised around ``seed``.

    Parameters
    ----------
    phi_in_2hw : (2, H, W) float64
    threshold : float
    seed : {'harmonic', 'zero'}
        ``'harmonic'`` uses the Laplacian-extension feasible seed (default).
        ``'zero'`` linearises around ``phi = 0`` -- used in ablation runs.

    Returns
    -------
    phi_out_2hw : (2, H, W) float64
    info : dict
    """
    t0 = time.time()
    H, W = phi_in_2hw.shape[1:]
    if seed == 'harmonic':
        seed_phi = _harmonic_seed(phi_in_2hw, threshold)
    elif seed == 'zero':
        seed_phi = np.zeros_like(phi_in_2hw)
    else:
        raise ValueError(f'unknown seed: {seed!r}')

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
