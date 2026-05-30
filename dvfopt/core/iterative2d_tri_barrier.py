"""2-triangle penalty -> log-barrier L-BFGS-B solver (2D).

Sibling of ``iterative2d_barrier`` (which enforces the Jacobian determinant)
but enforces the manuscript's 2-triangle areas T1, T2 >= threshold. Used by
``dvfopt.unified.DVFopt`` when ``constraint='2tri'`` and ``solver='barrier'``.

Phase 1 - exterior quadratic penalty:
    F_pen(phi) = anchor(phi - phi_init) + lam * sum_k max(0, target - T_k)^2
Phase 2 - log-barrier interior point (only after every T_k > threshold):
    F_bar(phi) = anchor(phi - phi_init) - mu * sum_k log(T_k - threshold)

Both phases minimised with scipy L-BFGS-B. Full-grid (no windowing, no
frozen edges). The constraint Jacobian J^T @ v is computed analytically
via vectorised scatter-add (no dense Jacobian materialised), so this
scales cleanly to full 2D slices.

The penalty/barrier loop itself lives in :mod:`dvfopt.core._barrier_core`
so the same homotopy is shared with the Jdet barrier solvers.
"""

import time

import numpy as np

from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.core._barrier_core import (
    DEFAULT_LAM_SCHEDULE,
    DEFAULT_MU_SCHEDULE,
    run_penalty_barrier_lbfgs,
)

# The 2-triangle constraint primitives live in tri_primitives.py. Underscore
# aliases here preserve back-compat for the ~16 callers that still import
# them from this module under the old private names.
from dvfopt.core.tri_primitives import (
    tri_areas_flat as _tri_areas_flat,
)
from dvfopt.core.tri_primitives import (
    tri_areas_flat_full_coverage as _tri_areas_flat_full_coverage,
)
from dvfopt.core.tri_primitives import (
    tri_grad_T_v as _tri_grad_T_v,
)
from dvfopt.core.tri_primitives import (
    tri_grad_T_v_full_coverage as _tri_grad_T_v_full_coverage,
)


# ----------------------------------------------------------------- main entry
def iterative_2d_tri_barrier(
    deformation_2hw,
    *,
    threshold=None,
    margin=1e-3,
    lam_schedule=DEFAULT_LAM_SCHEDULE,
    mu_schedule=DEFAULT_MU_SCHEDULE,
    max_minimize_iter=300,
    anchor='l2',
    eps_l1=1e-4,
    verbose=1,
    record_history=False,
    full_coverage=False,
):
    """Penalty -> log-barrier L-BFGS-B solver enforcing T1, T2 >= threshold.

    Parameters
    ----------
    deformation_2hw : ndarray
        Shape (2, H, W) -- [dy, dx]. Or (3, 1, H, W) -- will be coerced.
    threshold : float, optional
        Lower bound on triangle areas. Default ``DEFAULT_PARAMS['threshold']``.
    margin : float
        Safety margin above ``threshold`` used by the penalty phase.
    lam_schedule, mu_schedule : sequence of float
        Continuation schedules for penalty (lam) and barrier (mu).
    max_minimize_iter : int
        Inner L-BFGS-B iteration cap per (lam, mu) step.
    anchor : {'l2', 'l1', 'none'}
        Anchor norm against ``deformation_2hw`` itself.
    eps_l1 : float
        Smoothing for the L1 anchor (only used when ``anchor='l1'``).
    verbose : int
        0 = silent, 1 = step-level, 2 = step-level + scipy.
    record_history : bool
        If True, returns ``(phi, history)`` where history is a list of
        per-step dicts. Otherwise returns ``phi``.
    full_coverage : bool
        When True, also enforces two patch triangles using the opposite
        (TL-BR) diagonal at cells ``(0, 0)`` and ``(H-2, W-2)``. This
        closes the coverage gap of the standard TR-BL per-cell scheme:
        without it, vertices ``(0, 0)`` and ``(H-1, W-1)`` are touched by
        exactly one triangle each; with it, every vertex is in at least
        two triangles. Adds 2 constraint values; negligible cost.

    Returns
    -------
    phi_corrected : ndarray, shape (2, H, W)
    history : list, only if ``record_history=True``
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    if deformation_2hw.ndim == 4:  # (3, 1, H, W)
        if deformation_2hw.shape[0] == 3:
            deformation_2hw = np.stack([deformation_2hw[1, 0], deformation_2hw[2, 0]])
        else:
            deformation_2hw = deformation_2hw[:, 0]
    H, W = deformation_2hw.shape[1], deformation_2hw.shape[2]
    phi_init_flat = np.concatenate([deformation_2hw[0].ravel(), deformation_2hw[1].ravel()])

    constraint_values_fn = _tri_areas_flat_full_coverage if full_coverage else _tri_areas_flat
    constraint_adjoint_fn = _tri_grad_T_v_full_coverage if full_coverage else _tri_grad_T_v

    T_init = constraint_values_fn(phi_init_flat, H, W)
    init_neg = int((T_init <= 0).sum())
    init_min = float(T_init.min())
    if verbose >= 1:
        scheme = '2-tri full-coverage' if full_coverage else '2-tri'
        print(
            f'[2d-tri-barrier init] grid {H}x{W}  threshold={threshold}  '
            f'margin={margin}  anchor={anchor}  scheme={scheme}'
        )
        print(f'[init] tri neg={init_neg}  min={init_min:+.5f}')

    t_start = time.time()
    phi_flat, info = run_penalty_barrier_lbfgs(
        phi_init_flat,
        phi_init_flat,
        constraint_values=lambda p: constraint_values_fn(p, H, W),
        constraint_adjoint=lambda p, v: constraint_adjoint_fn(p, H, W, v),
        threshold=threshold,
        margin=margin,
        lam_schedule=lam_schedule,
        mu_schedule=mu_schedule,
        max_iter=max_minimize_iter,
        anchor=anchor,
        eps_l1=eps_l1,
        verbose=verbose,
        record_history=record_history,
    )

    if verbose >= 1:
        T = constraint_values_fn(phi_flat, H, W)
        print(
            f'[2d-tri-barrier done] neg={int((T <= 0).sum())}  '
            f'min={float(T.min()):+.6f}  feasible={info["feasible"]}  '
            f'({time.time() - t_start:.1f}s)'
        )

    phi_corr = np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])
    if record_history:
        # Map core's 'min_T' key back to 'min_tri' the existing callers expect.
        # Non-mutating: the comprehension below copies each dict before
        # renaming, so info['history'] itself stays intact.
        history = [{**h, 'min_tri': h['min_T']} for h in info['history']]
        for h in history:
            del h['min_T']
        return phi_corr, history
    return phi_corr
