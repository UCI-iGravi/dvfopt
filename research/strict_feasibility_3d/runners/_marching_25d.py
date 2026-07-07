"""Part XXI option A prototype: 2.5D MARCHING inter-layer correction.

Hypothesis: the 728k 3D folds are manufactured by z-blind per-slice 2D
correction; since dz==0, every inter-layer 6-tet volume depends only on
the dy/dx of the two adjacent slices. So sweep z sequentially: for each
slice z (starting from its already-2D-corrected state), repair the cube
layer (z-1, z) against the FROZEN, already-repaired slice z-1 by solving
small elastic LPs over slice-z's dy/dx in fold-cluster crops, with
  rows = linearized inter-layer 6-tet volumes (target 0.01)
       + linearized intra-slice 2-tri areas (preserve 2D feasibility)
  anchor = the slice's post-2D values (minimize ADDED L1)
  frozen 1-ring, exact-value acceptance, trust-region.

Prevention instead of the ~2.4-day 3D repair. Measures, per layer:
folds before/after, residual (cells unfixable with dy/dx only -> the
dz-freedom question), added L1, wall.
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parents[3]))

from dvfopt.core.slp.tri_linearize import build_sparse_jacobian_T  # noqa: E402
from dvfopt.core.tri_primitives import tri_areas_flat  # noqa: E402
from dvfopt.jacobian.tetrahedron_sign import (  # noqa: E402
    build_tet_sparse_jac,
    tet_volumes_flat,
)

THR3 = 0.01 + 1e-4   # inter-layer 6-tet inner target
THR2 = 0.01          # intra-slice 2-tri preserve level
MU = 1000.0          # elastic violation weight


def _stack_flat(prev, cur):
    """phi_flat [dx, dy, dz] for the (3, 2, Hc, Wc) two-slice stack, dz=0."""
    dx = np.concatenate([prev[1].ravel(), cur[1].ravel()])
    dy = np.concatenate([prev[0].ravel(), cur[0].ravel()])
    dz = np.zeros_like(dx)
    return np.concatenate([dx, dy, dz])


def _layer_min_v(prev, cur):
    """Min-of-6 tet volume per cell of the (prev, cur) cube layer."""
    Hc, Wc = cur.shape[1:]
    V = tet_volumes_flat(_stack_flat(prev, cur), 2, Hc, Wc)
    return V.reshape(6, Hc - 1, Wc - 1).min(axis=0)


def _repair_cluster(prev_c, cur_c, anchor_c, max_iter=8, trust0=0.5):
    """Elastic SLP over the crop's slice-z interior dy/dx. Returns cur_c'."""
    Hc, Wc = cur_c.shape[1:]
    n_pix = Hc * Wc
    jac3 = build_tet_sparse_jac(2, Hc, Wc)

    # Free columns: slice-1 interior dy/dx in the 3D pack [dx, dy, dz].
    inner = np.zeros((Hc, Wc), dtype=bool)
    inner[1:-1, 1:-1] = True
    ii = np.where(inner.ravel())[0]
    n2 = 2 * n_pix
    cols_dx3 = n_pix + ii             # dx block, plane 1
    cols_dy3 = n2 + n_pix + ii        # dy block, plane 1
    free3 = np.concatenate([cols_dx3, cols_dy3])   # order [dx_f, dy_f]
    # 2-tri pack is [dy, dx]: map the same free vars.
    cols_dy2 = ii
    cols_dx2 = n_pix + ii
    free2 = np.concatenate([cols_dx2, cols_dy2])   # J2 cols in [dx_f, dy_f] order

    def _free_vec(c):
        return np.concatenate([c[1].ravel()[ii], c[0].ravel()[ii]])

    def _apply(c, v):
        out = c.copy()
        nf = ii.size
        dx = out[1].ravel(); dx[ii] = v[:nf]; out[1] = dx.reshape(Hc, Wc)
        dy = out[0].ravel(); dy[ii] = v[nf:]; out[0] = dy.reshape(Hc, Wc)
        return out

    def _exact_viol(c):
        v3 = _layer_min_v(prev_c, c).ravel()
        t2 = tri_areas_flat(np.concatenate([c[0].ravel(), c[1].ravel()]), Hc, Wc)
        return (float(np.maximum(0, THR3 - v3).sum())
                + float(np.maximum(0, THR2 - t2).sum()))

    anchor_f = _free_vec(anchor_c)
    cur = cur_c.copy()
    viol = _exact_viol(cur)
    trust = trust0
    for _ in range(max_iter):
        if viol <= 1e-12:
            break
        pf = _stack_flat(prev_c, cur)
        T3 = tet_volumes_flat(pf, 2, Hc, Wc)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        p2 = np.concatenate([cur[0].ravel(), cur[1].ravel()])
        T2 = tri_areas_flat(p2, Hc, Wc)
        J2 = build_sparse_jacobian_T(p2, Hc, Wc).tocsc()[:, free2].tocsr()

        xf = _free_vec(cur)
        nf = xf.size
        # Active rows only (near/below target).
        a3 = np.where(T3 < THR3 + 0.5)[0]
        a2 = np.where(T2 < THR2 + 0.5)[0]
        J3a, J2a = J3[a3], J2[a2]
        Ka = a3.size + a2.size
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), MU * np.ones(Ka)])
        blocks, rhs = [], []
        blocks.append(sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))]))
        rhs.append(anchor_f)
        blocks.append(sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))]))
        rhs.append(-anchor_f)
        # rows: -J phi - s <= -thr + T - J phi_lin
        E3 = sp.hstack([-J3a, sp.csr_matrix((a3.size, nf)),
                        -sp.eye(a3.size), sp.csr_matrix((a3.size, a2.size))])
        rhs3 = -THR3 + T3[a3] - J3a @ xf
        E2 = sp.hstack([-J2a, sp.csr_matrix((a2.size, nf)),
                        sp.csr_matrix((a2.size, a3.size)), -sp.eye(a2.size)])
        rhs2 = -THR2 + T2[a2] - J2a @ xf
        A_ub = sp.vstack(blocks + [E3, E2]).tocsr()
        b_ub = np.concatenate(rhs + [rhs3, rhs2])
        bounds = ([(float(xf[i] - trust), float(xf[i] + trust)) for i in range(nf)]
                  + [(0.0, None)] * nf + [(0.0, None)] * Ka)
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            trust *= 0.5
            if trust < 1e-3:
                break
            continue
        cand = _apply(cur, res.x[:nf])
        viol_new = _exact_viol(cand)
        if viol_new < viol * (1 - 1e-9):
            cur, viol = cand, viol_new
            trust = min(trust * 2, trust0)
        else:
            trust *= 0.5
            if trust < 1e-3:
                break
    return cur


def march_slice(prev_sl, cur_sl, pad=4, dil=2, max_rounds=4, verbose=0):
    """Repair slice ``cur`` against frozen ``prev``. Returns (cur', stats)."""
    H, W = cur_sl.shape[1:]
    cur = cur_sl.copy()
    anchor = cur_sl.copy()
    t0 = time.time()
    n0 = int((_layer_min_v(prev_sl, cur) < 0).sum())
    for _ in range(max_rounds):
        mv = _layer_min_v(prev_sl, cur)
        bad = mv < THR3 - 1e-9
        if not bad.any():
            break
        merged = binary_dilation(bad, iterations=dil)
        labels, _ = cc_label(merged)
        for bbox in find_objects(labels):
            if bbox is None:
                continue
            y0 = max(0, bbox[0].start - pad)
            y1 = min(H - 1, bbox[0].stop + pad)
            x0 = max(0, bbox[1].start - pad)
            x1 = min(W - 1, bbox[1].stop + pad)
            sl_crop = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            fixed = _repair_cluster(
                prev_sl[:, sl_crop[0], sl_crop[1]].copy(),
                cur[:, sl_crop[0], sl_crop[1]].copy(),
                anchor[:, sl_crop[0], sl_crop[1]].copy(),
            )
            # frozen-ring splice (interior only)
            cur[:, y0 + 1:y1, x0 + 1:x1] = fixed[:, 1:-1, 1:-1]
    mv = _layer_min_v(prev_sl, cur)
    t2 = tri_areas_flat(np.concatenate([cur[0].ravel(), cur[1].ravel()]), H, W)
    stats = dict(
        n3_before=n0,
        n3_after=int((mv < 0).sum()),
        n3_below=int((mv < 0.01 - 1e-5).sum()),
        n2_after=int((t2 <= 0).sum()),
        l1_added=float(np.abs(cur - cur_sl).sum()),
        wall=time.time() - t0,
    )
    return cur, stats


def main():
    OUT = Path(__file__).parent / 'output'
    vol = np.load(
        Path('research/strict_feasibility_2d/runners/output') /
        'b0039_FULL_stage1_continuous.npy', mmap_mode='r')

    for label, z0, Z in (('moderate z200-205', 200, 6), ('dense z2-7', 2, 6)):
        print(f'\n===== {label} =====', flush=True)
        slices = [np.asarray(vol[1:3, z]).astype(np.float64)
                  for z in range(z0, z0 + Z)]
        # Baseline: per-layer 3D folds in the plain stage-1 stack.
        base = [int((_layer_min_v(slices[i], slices[i + 1]) < 0).sum())
                for i in range(Z - 1)]
        print(f'  baseline per-layer folds: {base}  (total {sum(base)})', flush=True)
        # Marching sweep.
        t0 = time.time()
        cur_prev = slices[0]
        tot_l1 = 0.0
        after = []
        for i in range(1, Z):
            fixed, st = march_slice(cur_prev, slices[i])
            after.append(st)
            tot_l1 += st['l1_added']
            print(f"  layer {i - 1}->{i}: folds {st['n3_before']}->{st['n3_after']} "
                  f"(n<0.01 {st['n3_below']})  2D n_neg={st['n2_after']}  "
                  f"L1+={st['l1_added']:.1f}  {st['wall']:.1f}s", flush=True)
            cur_prev = fixed
        res = sum(s['n3_after'] for s in after)
        print(f'  MARCHING: total folds {sum(base)} -> {res}  '
              f'L1 added={tot_l1:.1f}  wall={time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
