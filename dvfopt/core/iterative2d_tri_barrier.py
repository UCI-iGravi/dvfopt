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
from dvfopt.jacobian.shoelace import _ref_grid
from dvfopt.jacobian.triangle_sign import (
    _triangle_areas_2d,
    _corner_patch_areas_2d,
)
from dvfopt.core._barrier_core import (
    run_penalty_barrier_lbfgs,
    DEFAULT_LAM_SCHEDULE,
    DEFAULT_MU_SCHEDULE,
)


# --------------------------------------------------------------------- internals
def _tri_areas_flat(phi_flat, H, W):
    """Concatenated [T1.ravel, T2.ravel] of length 2*(H-1)*(W-1)."""
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    return np.concatenate([T1.ravel(), T2.ravel()])


def _tri_grad_T_v(phi_flat, H, W, v):
    """J^T @ v for the 2-triangle constraint Jacobian, analytically via
    vectorised scatter-add. ``v`` length 2*(H-1)*(W-1) (T1 then T2).
    Returns length 2*H*W ordered [dy.ravel(), dx.ravel()]."""
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy
    n_cells = (H - 1) * (W - 1)
    v1 = v[:n_cells].reshape(H - 1, W - 1)
    v2 = v[n_cells:].reshape(H - 1, W - 1)
    x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
    x_tr, y_tr = def_x[:-1, 1:],  def_y[:-1, 1:]
    x_bl, y_bl = def_x[1:, :-1],  def_y[1:, :-1]
    x_br, y_br = def_x[1:, 1:],   def_y[1:, 1:]

    g_dy = np.zeros((H, W))
    g_dx = np.zeros((H, W))

    # T1 (A=TR, B=BL, C=BR).
    g_dx[:-1, 1:]  +=  v1 * 0.5 * (y_br - y_bl)
    g_dy[:-1, 1:]  +=  v1 * 0.5 * (x_bl - x_br)
    g_dx[1:,  :-1] += -v1 * 0.5 * (y_br - y_tr)
    g_dy[1:,  :-1] +=  v1 * 0.5 * (x_br - x_tr)
    g_dx[1:,  1:]  +=  v1 * 0.5 * (y_bl - y_tr)
    g_dy[1:,  1:]  += -v1 * 0.5 * (x_bl - x_tr)
    # T2 (A=TL, B=BL, C=TR).
    g_dx[:-1, :-1] +=  v2 * 0.5 * (y_tr - y_bl)
    g_dy[:-1, :-1] +=  v2 * 0.5 * (x_bl - x_tr)
    g_dx[1:,  :-1] += -v2 * 0.5 * (y_tr - y_tl)
    g_dy[1:,  :-1] +=  v2 * 0.5 * (x_tr - x_tl)
    g_dx[:-1, 1:]  +=  v2 * 0.5 * (y_bl - y_tl)
    g_dy[:-1, 1:]  += -v2 * 0.5 * (x_bl - x_tl)
    return np.concatenate([g_dy.ravel(), g_dx.ravel()])


# --- Full-coverage variants: add two corner-patch triangles so every grid
# vertex (incl. the two diagonally-opposite corners (0,0) and (H-1, W-1))
# is enforced by at least two triangles. The standard scheme above leaves
# those two corners with only ONE constraint each.

def _tri_areas_flat_full_coverage(phi_flat, H, W):
    """Standard T1, T2 stack plus two corner patches.

    Output layout: ``[T1.ravel, T2.ravel, patch_TL, patch_BR]`` — length
    ``2*(H-1)*(W-1) + 2``.
    """
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    patches = _corner_patch_areas_2d(dy, dx)
    return np.concatenate([T1.ravel(), T2.ravel(), patches])


def _tri_grad_T_v_full_coverage(phi_flat, H, W, v):
    """J^T @ v for the full-coverage 2-triangle Jacobian.

    Layout of ``v``: first ``2*(H-1)*(W-1)`` entries are the standard T1/T2
    constraints, last 2 are the corner patches ``[patch_TL, patch_BR]``.
    """
    n_cells = (H - 1) * (W - 1)
    HW = H * W

    # Standard contribution.
    g = _tri_grad_T_v(phi_flat, H, W, v[:2 * n_cells])

    # Patch contributions are tiny — only 6 vertices touched total — but
    # we still write them into the dy/dx grids for a clean concat.
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy

    g_dy = g[:HW].reshape(H, W).copy()
    g_dx = g[HW:].reshape(H, W).copy()

    v_tl = v[2 * n_cells]      # patch at corner (0, 0)
    v_br = v[2 * n_cells + 1]  # patch at corner (H-1, W-1)

    # patch_TL: A=TL=(0,0), B=BR=(1,1), C=TR=(0,1).
    # Derived analytically from T = -0.5 * ((Bx-Ax)(Cy-Ay) - (By-Ay)(Cx-Ax)).
    g_dx[0, 0] += v_tl * 0.5 * (def_y[0, 1] - def_y[1, 1])    # ∂T/∂Ax
    g_dy[0, 0] += v_tl * 0.5 * (def_x[1, 1] - def_x[0, 1])    # ∂T/∂Ay
    g_dx[1, 1] += v_tl * -0.5 * (def_y[0, 1] - def_y[0, 0])   # ∂T/∂Bx
    g_dy[1, 1] += v_tl * 0.5 * (def_x[0, 1] - def_x[0, 0])    # ∂T/∂By
    g_dx[0, 1] += v_tl * 0.5 * (def_y[1, 1] - def_y[0, 0])    # ∂T/∂Cx
    g_dy[0, 1] += v_tl * -0.5 * (def_x[1, 1] - def_x[0, 0])   # ∂T/∂Cy

    # patch_BR: A=TL=(H-2, W-2), B=BL=(H-1, W-2), C=BR=(H-1, W-1).
    Hm2, Wm2 = H - 2, W - 2
    g_dx[Hm2, Wm2]     += v_br * 0.5 * (def_y[H - 1, W - 1] - def_y[H - 1, Wm2])
    g_dy[Hm2, Wm2]     += v_br * 0.5 * (def_x[H - 1, Wm2] - def_x[H - 1, W - 1])
    g_dx[H - 1, Wm2]   += v_br * -0.5 * (def_y[H - 1, W - 1] - def_y[Hm2, Wm2])
    g_dy[H - 1, Wm2]   += v_br * 0.5 * (def_x[H - 1, W - 1] - def_x[Hm2, Wm2])
    g_dx[H - 1, W - 1] += v_br * 0.5 * (def_y[H - 1, Wm2] - def_y[Hm2, Wm2])
    g_dy[H - 1, W - 1] += v_br * -0.5 * (def_x[H - 1, Wm2] - def_x[Hm2, Wm2])

    return np.concatenate([g_dy.ravel(), g_dx.ravel()])


# ----------------------------------------------------------------- main entry
def iterative_2d_tri_barrier(deformation_2hw, *,
                             threshold=None, margin=1e-3,
                             lam_schedule=DEFAULT_LAM_SCHEDULE,
                             mu_schedule=DEFAULT_MU_SCHEDULE,
                             max_minimize_iter=300,
                             anchor='l2', eps_l1=1e-4,
                             verbose=1,
                             record_history=False,
                             full_coverage=False):
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
    if deformation_2hw.ndim == 4:                # (3, 1, H, W)
        if deformation_2hw.shape[0] == 3:
            deformation_2hw = np.stack([deformation_2hw[1, 0],
                                        deformation_2hw[2, 0]])
        else:
            deformation_2hw = deformation_2hw[:, 0]
    H, W = deformation_2hw.shape[1], deformation_2hw.shape[2]
    phi_init_flat = np.concatenate([deformation_2hw[0].ravel(),
                                    deformation_2hw[1].ravel()])

    constraint_values_fn = (_tri_areas_flat_full_coverage
                            if full_coverage else _tri_areas_flat)
    constraint_adjoint_fn = (_tri_grad_T_v_full_coverage
                             if full_coverage else _tri_grad_T_v)

    T_init = constraint_values_fn(phi_init_flat, H, W)
    init_neg = int((T_init <= 0).sum())
    init_min = float(T_init.min())
    if verbose >= 1:
        scheme = '2-tri full-coverage' if full_coverage else '2-tri'
        print(f'[2d-tri-barrier init] grid {H}x{W}  threshold={threshold}  '
              f'margin={margin}  anchor={anchor}  scheme={scheme}')
        print(f'[init] tri neg={init_neg}  min={init_min:+.5f}')

    t_start = time.time()
    phi_flat, info = run_penalty_barrier_lbfgs(
        phi_init_flat, phi_init_flat,
        constraint_values=lambda p: constraint_values_fn(p, H, W),
        constraint_adjoint=lambda p, v: constraint_adjoint_fn(p, H, W, v),
        threshold=threshold, margin=margin,
        lam_schedule=lam_schedule, mu_schedule=mu_schedule,
        max_iter=max_minimize_iter,
        anchor=anchor, eps_l1=eps_l1,
        verbose=verbose, record_history=record_history,
    )

    if verbose >= 1:
        T = constraint_values_fn(phi_flat, H, W)
        print(f'[2d-tri-barrier done] neg={int((T <= 0).sum())}  '
              f'min={float(T.min()):+.6f}  feasible={info["feasible"]}  '
              f'({time.time()-t_start:.1f}s)')

    phi_corr = np.stack([phi_flat[:H * W].reshape(H, W),
                         phi_flat[H * W:].reshape(H, W)])
    if record_history:
        # Map core's 'min_T' key back to 'min_tri' the existing callers expect.
        # Non-mutating: the comprehension below copies each dict before
        # renaming, so info['history'] itself stays intact.
        history = [{**h, 'min_tri': h['min_T']} for h in info['history']]
        for h in history:
            del h['min_T']
        return phi_corr, history
    return phi_corr
