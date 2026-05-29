"""2-triangle penalty -> log-barrier L-BFGS-B solver.

Modeled on dvfopt.core.iterative2d_barrier but enforces the manuscript's
2-triangle constraint (T1, T2 >= threshold) instead of the Jacobian
determinant. The barrier *method* was already proven to crack a full-grid
2D feasibility problem in 130 s in the solver_comparison run; this swaps
the constraint function and its analytical gradient for the 2-triangle
pair so the result is *feasible by the manuscript's check*.

Phase 1 - exterior quadratic penalty:
    F_pen(phi) = 0.5 ||phi - anchor||^2 + lam * sum_k max(0, target - T_k)^2
Phase 2 - log-barrier interior point (only after every T_k > threshold):
    F_bar(phi) = 0.5 ||phi - anchor||^2 - mu * sum_k log(T_k - threshold)

Both minimised under L-BFGS-B (scipy). Full-grid: no windowing, no frozen
edges, the whole region is a single coupled problem.

Usage: python _run_tri_barrier_test.py [z ...]
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO)

import numpy as np
from scipy.optimize import minimize
from dvfopt.jacobian.shoelace import _ref_grid
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

THRESHOLD = 0.01
MARGIN = 1e-3
LAM_SCHEDULE = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7,
                1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11)
MU_SCHEDULE = (1e-1, 1e-2, 1e-3, 1e-4)
MAX_MINIMIZE_ITER = 800
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

phi_full = np.load(DATA_PATH)
H_g, W_g = phi_full.shape[2], phi_full.shape[3]


def _tri_areas_flat(phi_flat, H, W):
    """Concatenated [T1.ravel, T2.ravel] of length 2*(H-1)*(W-1)."""
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    T1, T2 = _triangle_areas_2d(dy, dx)
    return np.concatenate([T1.ravel(), T2.ravel()])


def _tri_grad_T_v(phi_flat, H, W, v):
    """J^T @ v where J is the dense 2-triangle constraint Jacobian.

    Analytical scatter-add over triangles. ``v`` has length 2*(H-1)*(W-1)
    (first half = weights for T1, second half = weights for T2). Returns
    a length-2*H*W vector ordered [dy.ravel(), dx.ravel()].
    """
    HW = H * W
    dy = phi_flat[:HW].reshape(H, W)
    dx = phi_flat[HW:].reshape(H, W)
    ref_y, ref_x = _ref_grid(H, W)
    def_x = ref_x + dx
    def_y = ref_y + dy
    n_cells = (H - 1) * (W - 1)
    v1 = v[:n_cells].reshape(H - 1, W - 1)
    v2 = v[n_cells:].reshape(H - 1, W - 1)
    # Deformed corners per cell.
    x_tl, y_tl = def_x[:-1, :-1], def_y[:-1, :-1]
    x_tr, y_tr = def_x[:-1, 1:],  def_y[:-1, 1:]
    x_bl, y_bl = def_x[1:, :-1],  def_y[1:, :-1]
    x_br, y_br = def_x[1:, 1:],   def_y[1:, 1:]

    g_dy = np.zeros((H, W))
    g_dx = np.zeros((H, W))

    # T1 = -0.5 * (AB_x * AC_y - AB_y * AC_x) with A=TR, B=BL, C=BR.
    # Partials -- one term per cell, scatter to the (H,W) corner grids.
    g_dx[:-1, 1:]  +=  v1 * 0.5 * (y_br - y_bl)   # dT1/dx_TR
    g_dy[:-1, 1:]  +=  v1 * 0.5 * (x_bl - x_br)   # dT1/dy_TR
    g_dx[1:,  :-1] += -v1 * 0.5 * (y_br - y_tr)   # dT1/dx_BL
    g_dy[1:,  :-1] +=  v1 * 0.5 * (x_br - x_tr)   # dT1/dy_BL
    g_dx[1:,  1:]  +=  v1 * 0.5 * (y_bl - y_tr)   # dT1/dx_BR
    g_dy[1:,  1:]  += -v1 * 0.5 * (x_bl - x_tr)   # dT1/dy_BR

    # T2 = -0.5 * (AB_x * AC_y - AB_y * AC_x) with A=TL, B=BL, C=TR.
    g_dx[:-1, :-1] +=  v2 * 0.5 * (y_tr - y_bl)   # dT2/dx_TL
    g_dy[:-1, :-1] +=  v2 * 0.5 * (x_bl - x_tr)   # dT2/dy_TL
    g_dx[1:,  :-1] += -v2 * 0.5 * (y_tr - y_tl)   # dT2/dx_BL
    g_dy[1:,  :-1] +=  v2 * 0.5 * (x_tr - x_tl)   # dT2/dy_BL
    g_dx[:-1, 1:]  +=  v2 * 0.5 * (y_bl - y_tl)   # dT2/dx_TR
    g_dy[:-1, 1:]  += -v2 * 0.5 * (x_bl - x_tl)   # dT2/dy_TR

    return np.concatenate([g_dy.ravel(), g_dx.ravel()])


EPS_L1 = 1e-4


def _anchor_l2(diff):
    """L2 anchor: F = 0.5 ||d||^2, grad = d."""
    return 0.5 * float(diff @ diff), diff


def _anchor_l1(diff, eps=EPS_L1):
    """Smoothed L1 anchor: F = Sigma sqrt(d^2 + eps^2) - eps*N.
    Promotes sparse corrections (few large moves vs many small)."""
    s = np.sqrt(diff * diff + eps * eps)
    val = float((s - eps).sum())
    grad = diff / s
    return val, grad


def _penalty_tri(phi_flat, phi_init_flat, H, W, threshold, margin, lam,
                 anchor='l2'):
    """F_pen and its gradient with anchor in {'l2', 'l1'}."""
    diff = phi_flat - phi_init_flat
    if anchor == 'l1':
        val, grad = _anchor_l1(diff)
    else:
        val, grad = _anchor_l2(diff)
    grad = grad.copy()
    T = _tri_areas_flat(phi_flat, H, W)
    target = threshold + margin
    viol = np.maximum(0.0, target - T)            # >= 0
    if viol.any():
        val += lam * float((viol * viol).sum())
        # d/dphi pen = lam * 2 * viol_k * (-dT_k/dphi) = -2*lam * J^T @ viol
        grad -= 2.0 * lam * _tri_grad_T_v(phi_flat, H, W, viol)
    return val, grad


def _barrier_tri(phi_flat, phi_init_flat, H, W, threshold, mu, anchor='l2'):
    """F_bar and its gradient. Requires every T_k > threshold."""
    diff = phi_flat - phi_init_flat
    if anchor == 'l1':
        val, grad = _anchor_l1(diff)
    else:
        val, grad = _anchor_l2(diff)
    grad = grad.copy()
    T = _tri_areas_flat(phi_flat, H, W)
    s = T - threshold
    if (s <= 0).any():
        return np.inf, grad                       # L-BFGS-B will reject
    val += -mu * float(np.log(s).sum())
    # d/dphi (-mu * sum log(s_k)) = -mu * sum (1/s_k) * dT_k/dphi
    grad += -mu * _tri_grad_T_v(phi_flat, H, W, 1.0 / s)
    return val, grad


def fd_check(z=126):
    """Sanity: analytic vs finite-difference gradient on a tiny region."""
    H, W = 8, 8
    rng = np.random.default_rng(0)
    phi_flat = rng.normal(scale=0.05, size=2 * H * W)
    phi_init = np.zeros_like(phi_flat)
    v0, g0 = _penalty_tri(phi_flat, phi_init, H, W, THRESHOLD, MARGIN, 10.0)
    eps = 1e-6
    g_fd = np.zeros_like(g0)
    for i in range(0, len(phi_flat), max(1, len(phi_flat) // 30)):
        e = np.zeros_like(phi_flat); e[i] = eps
        vp, _ = _penalty_tri(phi_flat + e, phi_init, H, W, THRESHOLD, MARGIN, 10.0)
        vm, _ = _penalty_tri(phi_flat - e, phi_init, H, W, THRESHOLD, MARGIN, 10.0)
        g_fd[i] = (vp - vm) / (2 * eps)
    sel = g_fd != 0
    rel = np.abs(g0[sel] - g_fd[sel]) / (np.abs(g_fd[sel]) + 1e-12)
    print(f'penalty FD-check (sampled): max rel err = {rel.max():.2e}',
          flush=True)


def slsqp_polish(phi_2hw, anchor_2hw, *, threshold=THRESHOLD, pad=3,
                 max_outer=8, l2_max_iter=120, l1_max_iter=150,
                 verbose=True):
    """Windowed SLSQP per-component polish. Run after the barrier has
    driven the field near-feasible: residuals are now sparse and shallow,
    which is SLSQP's good regime."""
    from scipy.ndimage import label as cc_label, binary_dilation, find_objects
    from scipy.optimize import minimize, NonlinearConstraint
    # Late import: this module lives next to the experiment helpers.
    sys.path.insert(0, os.path.join(_REPO, 'notebooks', 'manuscript'))
    from _bench_worker import _make_2tri_jac_2d, _interior_pack_unpack_2d

    phi = phi_2hw.copy()
    H, W = phi.shape[1], phi.shape[2]
    for outer in range(1, max_outer + 1):
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        fold = np.minimum(T1, T2) <= 0
        if not fold.any():
            break
        labels, _ = cc_label(binary_dilation(fold, iterations=1))
        objs = find_objects(labels)
        comps = [(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
                 for sl in objs if sl is not None]
        if verbose:
            print(f'  polish outer {outer}: {len(comps)} components',
                  flush=True)
        for cy0, cy1, cx0, cx1 in comps:
            y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
            x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
            sy, sx = y1 - y0, x1 - x0
            if sy < 4 or sx < 4:
                continue
            im = np.zeros((sy + 1, sx + 1), dtype=bool)
            im[1:-1, 1:-1] = True
            phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
            anc_win = anchor_2hw[:, y0:y1 + 1, x0:x1 + 1].copy()
            pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, im)
            if n_int == 0:
                continue
            z_anchor = pack(anc_win)
            jac_func = _make_2tri_jac_2d(phi_win, im)

            def obj(z):
                d = z - z_anchor
                return 0.5 * float(d @ d), d

            def constr(z):
                ph = unpack(z, phi_win)
                t1, t2 = _triangle_areas_2d(ph[0], ph[1])
                return np.concatenate([t1.ravel(), t2.ravel()])

            nl = NonlinearConstraint(constr, threshold, np.inf, jac=jac_func)
            res = minimize(obj, pack(phi_win), jac=True, method='SLSQP',
                           constraints=[nl],
                           options={'maxiter': l2_max_iter, 'ftol': 1e-10,
                                    'disp': False})
            phi_new = unpack(res.x, phi_win)
            yy, xx = np.where(im)
            phi[:, y0 + yy, x0 + xx] = phi_new[:, yy, xx]
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        min_tri = float(min(T1.min(), T2.min()))
        if verbose:
            print(f'    -> n_neg={n_neg}  min_tri={min_tri:+.6f}', flush=True)
        if n_neg == 0:
            break
    return phi


def solve_2d_tri_barrier(deformation_2hw, *, threshold=THRESHOLD, margin=MARGIN,
                         lam_schedule=LAM_SCHEDULE, mu_schedule=MU_SCHEDULE,
                         max_minimize_iter=MAX_MINIMIZE_ITER, anchor='l2',
                         verbose=True):
    """Penalty -> log-barrier full-grid L-BFGS-B solver enforcing
    T1, T2 >= threshold. deformation_2hw shape (2, H, W) = [dy, dx]."""
    H, W = deformation_2hw.shape[1], deformation_2hw.shape[2]
    phi_init_flat = np.concatenate([deformation_2hw[0].ravel(),
                                    deformation_2hw[1].ravel()])
    phi_flat = phi_init_flat.copy()
    T = _tri_areas_flat(phi_flat, H, W)
    init_neg = int((T <= 0).sum())
    init_min = float(T.min())
    if verbose:
        print(f'[init] grid {H}x{W}  threshold={threshold}  margin={margin}',
              flush=True)
        print(f'[init] 2-tri neg={init_neg}  min={init_min:+.5f}', flush=True)
    target = threshold + margin
    feasible = init_min >= target
    cur_min = init_min
    # Phase 1: penalty
    for k, lam in enumerate(lam_schedule):
        if feasible:
            break
        t0 = time.time()
        res = minimize(_penalty_tri, phi_flat,
                       args=(phi_init_flat, H, W, threshold, margin, lam,
                             anchor),
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': max_minimize_iter, 'gtol': 1e-6})
        phi_flat = res.x
        T = _tri_areas_flat(phi_flat, H, W)
        cur_neg = int((T <= 0).sum())
        cur_min = float(T.min())
        if verbose:
            l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
            print(f'[penalty {k+1}] lam={lam:g}  neg={cur_neg:5d}  '
                  f'min={cur_min:+.6f}  L2={l2:.3f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
        if cur_min >= target:
            feasible = True
    # Phase 2: barrier polish
    if feasible:
        for k, mu in enumerate(mu_schedule):
            t0 = time.time()
            res = minimize(_barrier_tri, phi_flat,
                           args=(phi_init_flat, H, W, threshold, mu, anchor),
                           jac=True, method='L-BFGS-B',
                           options={'maxiter': max_minimize_iter, 'gtol': 1e-6})
            if np.isfinite(res.fun):
                phi_flat = res.x
            T = _tri_areas_flat(phi_flat, H, W)
            cur_neg = int((T <= 0).sum())
            cur_min = float(T.min())
            if verbose:
                l2 = float(np.linalg.norm(phi_flat - phi_init_flat))
                print(f'[barrier {k+1}] mu={mu:g}  neg={cur_neg:5d}  '
                      f'min={cur_min:+.6f}  L2={l2:.3f}  '
                      f'({time.time()-t0:.1f}s)', flush=True)
    phi_corr = np.stack([phi_flat[:H * W].reshape(H, W),
                         phi_flat[H * W:].reshape(H, W)])
    return phi_corr, dict(init_neg=init_neg, init_min=init_min,
                          final_neg=cur_neg, final_min=cur_min)


def _run_case(phi0, label, threshold, anchor):
    print(f'\n----- {label}: threshold={threshold}  anchor={anchor} -----',
          flush=True)
    t0 = time.time()
    phi_c, stats = solve_2d_tri_barrier(phi0, threshold=threshold,
                                        anchor=anchor)
    wall = time.time() - t0
    T1, T2 = _triangle_areas_2d(phi_c[0], phi_c[1])
    n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    min_tri = float(min(T1.min(), T2.min()))
    print(f'{label}: n_neg {stats["init_neg"]} -> {n_neg}  '
          f'min_tri {stats["init_min"]:+.3f} -> {min_tri:+.6f}  '
          f'wall={wall:.0f}s', flush=True)
    return dict(label=label, threshold=threshold, anchor=anchor,
                n_neg=n_neg, min_tri=min_tri, wall=wall)


def main():
    fd_check()
    targets = [int(z) for z in sys.argv[1:]] or [12]
    print(flush=True)
    rows = []
    for z in targets:
        phi = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
        print(f'\n====================== z={z} ======================',
              flush=True)
        # Case A: L1 anchor + standard +0.01 margin (path 2 from the
        # discussion -- can sparse corrections cross the +0.01 wall?).
        r_a = _run_case(phi, label='L1, thr=0.01', threshold=0.01,
                        anchor='l1')
        # Case B: L2 anchor + thr=0 (path 1: relaxed margin -- can we at
        # least get n_neg=0 / min_tri > 0 on the dense slices?).
        r_b = _run_case(phi, label='L2, thr=0',    threshold=0.0,
                        anchor='l2')
        rows.append((z, r_a, r_b))

    print(f'\n{"=" * 78}', flush=True)
    print(f'{"z":>4s} | {"L1 thr=0.01":>30s} | {"L2 thr=0 (relaxed)":>30s}',
          flush=True)
    print(f'{"":>4s} | {"n_neg":>9s} {"min_tri":>11s} {"sec":>7s} '
          f'| {"n_neg":>9s} {"min_tri":>11s} {"sec":>7s}', flush=True)
    for z, ra, rb in rows:
        print(f'{z:4d} | {ra["n_neg"]:9d} {ra["min_tri"]:+11.6f} '
              f'{ra["wall"]:7.0f} | {rb["n_neg"]:9d} {rb["min_tri"]:+11.6f} '
              f'{rb["wall"]:7.0f}', flush=True)


if __name__ == '__main__':
    main()
