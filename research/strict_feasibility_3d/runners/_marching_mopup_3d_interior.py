"""Correct residual mop-up: per-cluster 3D-interior elastic SLP.

The marching sweep left 97 negative 6-tet volumes in the compact dense band
(z0-18). They survived because marching freezes ONE whole slice and moves
the other's interior -- but these folds need BOTH slices of the pair to
move. This mop crops a small box around each fold cluster, freezes the
ENTIRE rim (all 6 faces -> safe paste, no boundary breakage), and solves
the FREE INTERIOR dy/dx of all box slices with the same elastic-SLP the
sweep used (inter-layer 6-tet + intra-slice 2-tri rows, trust region,
exact-violation acceptance). dz stays 0 (2.5D structure preserved).

Non-destructive: reads b0039_FULL_marching25d.npy, writes
b0039_FULL_marching25d_mop3d.npy. Verifies globally before saving.
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as cc_label
from scipy.ndimage import find_objects
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parents[3]))

from dvfopt.core.slp.tri_linearize import build_sparse_jacobian_T  # noqa: E402
from dvfopt.core.tri_primitives import tri_areas_flat  # noqa: E402
from dvfopt.jacobian.tetrahedron_sign import (  # noqa: E402
    build_tet_sparse_jac, tet_volumes_flat, six_tet_volumes_3d,
)

OUT = 'research/strict_feasibility_3d/runners/output'
SRC = f'{OUT}/b0039_FULL_marching25d.npy'
DST = f'{OUT}/b0039_FULL_marching25d_mop3d.npy'

THR3 = 0.01 + 1e-4
THR2 = 0.01
MU = 1000.0
DIL = 1
MAXITERS = 40
# escalating (zpad, pad) per outer pass — bigger boxes give stubborn folds
# more free interior to work with
PASS_PADS = [(2, 4), (3, 6), (4, 9), (6, 12), (8, 16)]


def _stack(box):
    """box (2,D,H,W)=[dy,dx] -> [dx_flat, dy_flat, dz=0] slice-major."""
    dx = box[1].reshape(-1)
    dy = box[0].reshape(-1)
    return np.concatenate([dx, dy, np.zeros_like(dx)])


def _viol(box, D, H, W):
    v3 = tet_volumes_flat(_stack(box), D, H, W)
    tot = float(np.maximum(0.0, THR3 - v3).sum())
    for s in range(D):
        t2 = tri_areas_flat(np.concatenate([box[0, s].ravel(),
                                            box[1, s].ravel()]), H, W)
        tot += float(np.maximum(0.0, THR2 - t2).sum())
    return tot


def _repair_box(box):
    """Elastic SLP over interior dy/dx of a (2,D,H,W) box; rim frozen."""
    _, D, H, W = box.shape
    N = D * H * W
    jac3 = build_tet_sparse_jac(D, H, W)

    # interior node ids (not on any of the 6 faces)
    ss, ii, jj = np.meshgrid(np.arange(1, D - 1), np.arange(1, H - 1),
                             np.arange(1, W - 1), indexing='ij')
    nodes = (ss * H * W + ii * W + jj).ravel()
    if nodes.size == 0:
        return box, _viol(box, D, H, W)
    free3 = np.concatenate([nodes, N + nodes])          # dx cols, dy cols
    nf = free3.size

    def anchor_vec(b):
        return np.concatenate([b[1].reshape(-1)[nodes], b[0].reshape(-1)[nodes]])

    def apply(b, v):
        out = b.copy()
        dx = out[1].reshape(-1); dx[nodes] = v[:nodes.size]
        dy = out[0].reshape(-1); dy[nodes] = v[nodes.size:]
        out[1] = dx.reshape(D, H, W); out[0] = dy.reshape(D, H, W)
        return out

    anchor = anchor_vec(box)
    cur = box.copy()
    viol = _viol(cur, D, H, W)
    trust = 0.5
    for _ in range(MAXITERS):
        if viol <= 1e-12:
            break
        pf = _stack(cur)
        T3 = tet_volumes_flat(pf, D, H, W)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        a3 = np.where(T3 < THR3 + 0.5)[0]
        J3a = J3[a3]
        xf = anchor_vec(cur)

        # 6-tet-only elastic LP; the exact-violation acceptance (which
        # includes intra-slice 2-tri) rejects any step that breaks a 2D area.
        Ka = a3.size
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), MU * np.ones(Ka)])
        A1 = sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        A2 = sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        E3 = sp.hstack([-J3a, sp.csr_matrix((Ka, nf)), -sp.eye(Ka)])
        A_ub = sp.vstack([A1, A2, E3]).tocsr()
        b_ub = np.concatenate([anchor, -anchor, -THR3 + T3[a3] - J3a @ xf])
        bounds = ([(float(xf[i] - trust), float(xf[i] + trust)) for i in range(nf)]
                  + [(0.0, None)] * nf + [(0.0, None)] * Ka)
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            trust *= 0.5
            if trust < 1e-3:
                break
            continue
        cand = apply(cur, res.x[:nf])
        vnew = _viol(cand, D, H, W)
        if vnew < viol * (1 - 1e-9):
            cur, viol = cand, vnew
            trust = min(trust * 2, 0.5)
        else:
            trust *= 0.5
            if trust < 1e-3:
                break
    return cur, viol


def _pass(full, zpad, pad):
    """One repair pass over all current fold clusters. Returns n_fixed."""
    V0 = six_tet_volumes_3d(full)
    bad = V0.min(axis=0) <= 0
    lab, n = cc_label(binary_dilation(bad, iterations=DIL))
    boxes = find_objects(lab)
    dyx = full[1:3]
    fixed = 0
    for bb in boxes:
        if bb is None:
            continue
        z0 = max(0, bb[0].start - zpad); z1 = min(full.shape[1], bb[0].stop + zpad)
        y0 = max(0, bb[1].start - pad); y1 = min(full.shape[2], bb[1].stop + pad + 1)
        x0 = max(0, bb[2].start - pad); x1 = min(full.shape[3], bb[2].stop + pad + 1)
        box = dyx[:, z0:z1 + 1, y0:y1, x0:x1].copy()
        D, H, W = box.shape[1:]
        if D < 3 or H < 3 or W < 3:
            continue
        v_before = _viol(box, D, H, W)
        if v_before <= 1e-12:
            continue
        box2, v_after = _repair_box(box)
        if v_after < v_before:
            dyx[:, z0:z1 + 1, y0:y1, x0:x1] = box2
            fixed += 1
    full[1:3] = dyx
    return fixed


def main():
    src = DST if Path(DST).exists() else SRC
    full = np.array(np.load(src))
    n0 = int((six_tet_volumes_3d(full) <= 0).sum())
    print(f'GLOBAL start (from {Path(src).name}): n_neg={n0}', flush=True)
    t0 = time.time()
    for pi, (zpad, pad) in enumerate(PASS_PADS):
        before = int((six_tet_volumes_3d(full) <= 0).sum())
        if before == 0:
            break
        fixed = _pass(full, zpad, pad)
        V = six_tet_volumes_3d(full)
        after = int((V <= 0).sum())
        print(f'  pass {pi+1} (zpad={zpad},pad={pad}): n_neg {before}->{after} '
              f'min_T={float(V.min()):.5f} (fixed {fixed}, {time.time()-t0:.0f}s)',
              flush=True)
        np.save(DST, full)                    # checkpoint each pass
        if after == before:
            continue                          # escalate pad next pass
    V = six_tet_volumes_3d(full)
    print(f'GLOBAL final : n_neg={int((V<=0).sum())} '
          f'n<0.01={int((V<0.01).sum())} min_T={float(V.min()):.5f}', flush=True)
    print(f'saved {DST}', flush=True)


if __name__ == '__main__':
    main()
