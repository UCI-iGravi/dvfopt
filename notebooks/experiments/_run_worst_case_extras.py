"""Probes on z=12's largest fold component (the same 15x24 / 1472-
constraint crop the continuation diagnostic showed degenerates at
SLSQP status 8). All probes are direct SLSQP solves on this single crop
so we isolate the constraint formulation and the Jacobian mode as the
variables. (iterative_serial on the full slice OOMs at 320x456.)

Probe A -- constraint combinations:
    triangle, shoelace, triangle+shoelace, shoelace+monotonic,
    triangle+shoelace+monotonic.
Tests whether geometric extras (shoelace = signed quad area;
monotonic = warped corners stay in original-grid order) help SLSQP
push past the wall.

Probe B -- analytical vs FD Jacobian on the triangle-only baseline.
Tests whether the analytical Jacobian's structure (rank-deficient at
T=0) is itself the cause of the status-8 collapse or whether scipy's
finite-difference fallback gives a different verdict.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
_MANU = os.path.join(_REPO, 'notebooks', 'manuscript')
sys.path.insert(0, _REPO)
sys.path.insert(0, _MANU)

import numpy as np
from scipy.ndimage import label as cc_label, binary_dilation, find_objects
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.jacobian.shoelace import _shoelace_areas_2d
from _bench_worker import _make_2tri_jac_2d, _interior_pack_unpack_2d

THRESHOLD = 0.01
INJ_THR = 0.05            # injectivity (monotonic) threshold per
                          # ``iterative_serial``'s sensible range 0.05-0.3
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')
Z = 12


def stats(phi2):
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    J = np.squeeze(jacobian_det2D(phi2))
    return dict(
        tri_neg=int((T1 <= 0).sum() + (T2 <= 0).sum()),
        tri_min=float(min(T1.min(), T2.min())),
        sho_neg=int(((T1 + T2) <= 0).sum()),
        sho_min=float((T1 + T2).min()),
        jdet_neg=int((J <= 0).sum()),
        jdet_min=float(J.min()),
    )


def _get_worst_component_crop(phi_full):
    phi2 = np.stack([phi_full[1, Z].copy(), phi_full[2, Z].copy()])
    H, W = phi2.shape[1], phi2.shape[2]
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    fold = np.minimum(T1, T2) <= 0
    labels, _ = cc_label(binary_dilation(fold, iterations=1))
    comps = []
    for sl in find_objects(labels):
        if sl is not None:
            comps.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    comps.sort(key=lambda c: (c[1] - c[0]) * (c[3] - c[2]), reverse=True)
    cy0, cy1, cx0, cx1 = comps[0]
    pad = 4
    y0 = max(0, cy0 - pad); y1 = min(H - 1, cy1 + pad)
    x0 = max(0, cx0 - pad); x1 = min(W - 1, cx1 + pad)
    return phi2[:, y0:y1 + 1, x0:x1 + 1].copy()


def _build_constraints(phi_win, im, kinds):
    """Return a list of NonlinearConstraint with the requested constraints
    stacked. ``kinds`` is a set/list of {'tri', 'sho', 'mon'}."""
    pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, im)
    cons = []

    if 'tri' in kinds:
        jac_func = _make_2tri_jac_2d(phi_win, im)

        def tri_constr(z, _u=unpack, _w=phi_win):
            ph = _u(z, _w)
            T1, T2 = _triangle_areas_2d(ph[0], ph[1])
            return np.concatenate([T1.ravel(), T2.ravel()])

        cons.append(NonlinearConstraint(tri_constr, THRESHOLD, np.inf,
                                        jac=jac_func))

    if 'sho' in kinds:
        def sho_constr(z, _u=unpack, _w=phi_win):
            ph = _u(z, _w)
            return _shoelace_areas_2d(ph[0], ph[1]).ravel()
        # FD Jacobian for shoelace (no analytical builder readily available)
        cons.append(NonlinearConstraint(sho_constr, THRESHOLD, np.inf,
                                        jac='2-point'))

    if 'mon' in kinds:
        # def_x[i,j+1] - def_x[i,j] = 1 + dx[i,j+1]-dx[i,j] >= INJ_THR
        # def_y[i+1,j] - def_y[i,j] = 1 + dy[i+1,j]-dy[i,j] >= INJ_THR
        def mon_constr(z, _u=unpack, _w=phi_win):
            ph = _u(z, _w)
            dy, dx = ph[0], ph[1]
            # horizontal x-monotonic
            dxh = (1.0 + dx[:, 1:] - dx[:, :-1]).ravel()
            # vertical y-monotonic
            dyv = (1.0 + dy[1:, :] - dy[:-1, :]).ravel()
            return np.concatenate([dxh, dyv])

        cons.append(NonlinearConstraint(mon_constr, INJ_THR, np.inf,
                                        jac='2-point'))

    return pack, unpack, n_int, cons


def _run_slsqp(phi_win, kinds, *, jac_mode='analytical', maxiter=80,
               label=None):
    """Solve SLSQP on the crop with the given constraint kinds. ``jac_mode``
    only affects the *triangle* constraint (it's the analytical-vs-FD knob)."""
    H, W = phi_win.shape[1], phi_win.shape[2]
    im = np.zeros((phi_win.shape[1], phi_win.shape[2]), dtype=bool)
    # phi_win is (2,H,W) corners -- cells are (H-1,W-1)
    im[1:-1, 1:-1] = True
    pack, unpack, n_int, cons = _build_constraints(phi_win, im, kinds)

    # If jac_mode='fd' AND 'tri' in kinds, replace the analytical tri jac.
    if jac_mode == 'fd' and 'tri' in kinds:
        for i, c in enumerate(cons):
            if hasattr(c, 'jac') and callable(c.jac) and c.jac not in ('2-point', '3-point'):
                # Wrap fresh: same constr, FD jac.
                def tri_constr_fd(z, _u=unpack, _w=phi_win):
                    ph = _u(z, _w)
                    T1, T2 = _triangle_areas_2d(ph[0], ph[1])
                    return np.concatenate([T1.ravel(), T2.ravel()])
                cons[i] = NonlinearConstraint(tri_constr_fd, THRESHOLD,
                                              np.inf, jac='2-point')
                break

    anc_win = phi_win.copy()
    z_anchor = pack(anc_win)

    def obj(z):
        d = z - z_anchor
        return 0.5 * float(d @ d), d

    t0 = time.time()
    try:
        res = minimize(obj, pack(phi_win), jac=True, method='SLSQP',
                       constraints=cons,
                       options=dict(maxiter=maxiter, ftol=1e-10, disp=False))
    except Exception as exc:
        wall = time.time() - t0
        return dict(label=label, kinds=kinds, jac_mode=jac_mode,
                    error=str(exc), wall=wall)
    wall = time.time() - t0
    phi_new = unpack(res.x, phi_win)
    s = stats(phi_new)
    return dict(label=label, kinds=kinds, jac_mode=jac_mode,
                status=res.status, nit=res.nit, msg=str(res.message),
                wall=wall, **s)


def main():
    phi_full = np.load(DATA_PATH)
    phi_win = _get_worst_component_crop(phi_full)
    print(f'z={Z}: worst-component crop shape (corners) = {phi_win.shape}',
          flush=True)
    s0 = stats(phi_win)
    print(f'  crop init:  2-tri n_neg={s0["tri_neg"]} min={s0["tri_min"]:+.4f}  '
          f'shoelace n_neg={s0["sho_neg"]} min={s0["sho_min"]:+.4f}  '
          f'jdet n_neg={s0["jdet_neg"]} min={s0["jdet_min"]:+.4f}',
          flush=True)

    # --- Probe A: constraint combinations (analytical jac for tri) ---
    probe_a_configs = [
        ('triangle',              {'tri'}),
        ('shoelace',              {'sho'}),
        ('tri+shoelace',          {'tri', 'sho'}),
        ('shoelace+monotonic',    {'sho', 'mon'}),
        ('tri+shoelace+monotonic', {'tri', 'sho', 'mon'}),
    ]

    print(f'\n=== probe A: constraint combos (SLSQP, analytical jac) ===')
    rows_a = []
    for label, kinds in probe_a_configs:
        print(f'  running: {label}', flush=True)
        r = _run_slsqp(phi_win, kinds, jac_mode='analytical', maxiter=100,
                       label=label)
        rows_a.append(r)
        if 'error' in r and r.get('error'):
            print(f'    -> ERROR: {r["error"]}', flush=True)
            continue
        feas = (r['tri_neg'] == 0 and r['tri_min'] >= THRESHOLD - 1e-4)
        print(f'    -> SLSQP status={r["status"]}  nit={r["nit"]}  '
              f'tri n_neg={r["tri_neg"]} min={r["tri_min"]:+.5f}  '
              f'sho min={r["sho_min"]:+.5f}  '
              f'jdet min={r["jdet_min"]:+.5f}  '
              f'({r["wall"]:.1f}s)  '
              f'{"TRI-FEASIBLE" if feas else "still folded"}',
              flush=True)

    # --- Probe B: analytical vs FD Jacobian (tri only) ---
    print(f'\n=== probe B: analytical vs FD Jacobian (tri only) ===')
    rows_b = []
    for jac_mode in ('analytical', 'fd'):
        print(f'  running: jac={jac_mode}', flush=True)
        r = _run_slsqp(phi_win, {'tri'}, jac_mode=jac_mode, maxiter=80,
                       label=f'tri / jac={jac_mode}')
        rows_b.append(r)
        if 'error' in r and r.get('error'):
            print(f'    -> ERROR: {r["error"]}', flush=True)
            continue
        feas = (r['tri_neg'] == 0 and r['tri_min'] >= THRESHOLD - 1e-4)
        print(f'    -> SLSQP status={r["status"]}  nit={r["nit"]}  '
              f'tri n_neg={r["tri_neg"]} min={r["tri_min"]:+.5f}  '
              f'msg="{r["msg"][:50]}"  ({r["wall"]:.1f}s)  '
              f'{"CONVERGED" if feas else "still folded"}',
              flush=True)

    # --- summary ---
    print('\n' + '=' * 90)
    print('SUMMARY  (z=12 worst component, threshold=0.01)', flush=True)
    print('\n[Probe A] constraint combinations')
    print(f'{"flags":>28s} | {"status":>6s} {"nit":>4s} | '
          f'{"tri n_neg":>10s} {"tri min":>10s} | '
          f'{"sho min":>9s} {"jdet min":>10s} | {"wall":>6s}  result')
    for r in rows_a:
        if 'error' in r and r.get('error'):
            print(f'{r["label"]:>28s} | ERROR: {r["error"][:60]}')
            continue
        feas = (r['tri_neg'] == 0 and r['tri_min'] >= THRESHOLD - 1e-4)
        print(f'{r["label"]:>28s} | {r["status"]:6d} {r["nit"]:4d} | '
              f'{r["tri_neg"]:10d} {r["tri_min"]:+10.5f} | '
              f'{r["sho_min"]:+9.5f} {r["jdet_min"]:+10.5f} | '
              f'{r["wall"]:6.1f}  {"FEAS" if feas else "still folded"}')
    print('\n[Probe B] Jacobian mode (tri only)')
    for r in rows_b:
        if 'error' in r and r.get('error'):
            print(f'  {r["label"]}: ERROR: {r["error"][:60]}')
            continue
        feas = (r['tri_neg'] == 0 and r['tri_min'] >= THRESHOLD - 1e-4)
        print(f'  {r["label"]:>20s}: status={r["status"]}  nit={r["nit"]}  '
              f'n_neg={r["tri_neg"]} min={r["tri_min"]:+.5f}  '
              f'({r["wall"]:.1f}s)  '
              f'{"FEAS" if feas else "still folded"}')


if __name__ == '__main__':
    main()
