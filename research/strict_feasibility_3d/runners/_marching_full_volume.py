"""Part XXI option A — PRODUCTIZED 2.5D marching sweep, full volume.

Prevents the ~728k stage-1 3D folds by repairing each slice against its
already-repaired neighbour, sweeping OUTWARD from the mildest layer (so
no layer is cold-started against raw data). Per layer: cluster the
violating cells, solve elastic LPs (inter-layer 6-tet + intra-slice 2-tri
rows, frozen ring, exact-value acceptance) over the free slice's dy/dx —
cluster solves run in PARALLEL via the shared pre-warmed pool.

RESUMABLE: output volume is an on-disk memmap; progress JSON records the
next slice per direction. Final phase: verify + light 3D mop-up of the
small residual with active-band recovery.

GUARDED for Windows spawn (workers re-import this module; top level is
imports only).
"""

import json
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

THR3 = 0.01 + 1e-4
THR2 = 0.01
MU = 1000.0
N_WORKERS = 12
PAD, DIL, MAX_ROUNDS = 4, 2, 6
MAX_BOX = 90            # split larger cluster boxes into tiles


def _stack_flat(lower, upper):
    dx = np.concatenate([lower[1].ravel(), upper[1].ravel()])
    dy = np.concatenate([lower[0].ravel(), upper[0].ravel()])
    return np.concatenate([dx, dy, np.zeros_like(dx)])


def _layer_min_v(lower, upper):
    Hc, Wc = lower.shape[1:]
    V = tet_volumes_flat(_stack_flat(lower, upper), 2, Hc, Wc)
    return V.reshape(6, Hc - 1, Wc - 1).min(axis=0)


def _repair_cluster(args):
    """Worker: elastic SLP over the FREE plane's interior dy/dx.

    args = (frozen_c, cur_c, anchor_c, cur_is_upper)
    Stack is built in geometric order (lower, upper); the free columns are
    plane 1 if cur_is_upper else plane 0.
    """
    frozen_c, cur_c, anchor_c, cur_is_upper = args
    Hc, Wc = cur_c.shape[1:]
    n_pix = Hc * Wc
    jac3 = build_tet_sparse_jac(2, Hc, Wc)

    inner = np.zeros((Hc, Wc), dtype=bool)
    inner[1:-1, 1:-1] = True
    ii = np.where(inner.ravel())[0]
    n2 = 2 * n_pix
    plane_off = n_pix if cur_is_upper else 0
    cols_dx3 = plane_off + ii
    cols_dy3 = n2 + plane_off + ii
    free3 = np.concatenate([cols_dx3, cols_dy3])
    free2 = np.concatenate([n_pix + ii, ii])     # J2 [dy, dx] -> [dx_f, dy_f]

    def _geo(c):
        return (frozen_c, c) if cur_is_upper else (c, frozen_c)

    def _free_vec(c):
        return np.concatenate([c[1].ravel()[ii], c[0].ravel()[ii]])

    def _apply(c, v):
        out = c.copy()
        nf = ii.size
        dx = out[1].ravel(); dx[ii] = v[:nf]; out[1] = dx.reshape(Hc, Wc)
        dy = out[0].ravel(); dy[ii] = v[nf:]; out[0] = dy.reshape(Hc, Wc)
        return out

    def _exact_viol(c):
        lo, up = _geo(c)
        v3 = tet_volumes_flat(_stack_flat(lo, up), 2, Hc, Wc)
        t2 = tri_areas_flat(np.concatenate([c[0].ravel(), c[1].ravel()]), Hc, Wc)
        return (float(np.maximum(0, THR3 - v3).sum())
                + float(np.maximum(0, THR2 - t2).sum()))

    anchor_f = _free_vec(anchor_c)
    cur = cur_c.copy()
    viol = _exact_viol(cur)
    trust = 0.5
    for _ in range(12):
        if viol <= 1e-12:
            break
        lo, up = _geo(cur)
        pf = _stack_flat(lo, up)
        T3 = tet_volumes_flat(pf, 2, Hc, Wc)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        p2 = np.concatenate([cur[0].ravel(), cur[1].ravel()])
        T2 = tri_areas_flat(p2, Hc, Wc)
        J2 = build_sparse_jacobian_T(p2, Hc, Wc).tocsc()[:, free2].tocsr()
        xf = _free_vec(cur)
        nf = xf.size
        a3 = np.where(T3 < THR3 + 0.5)[0]
        a2 = np.where(T2 < THR2 + 0.5)[0]
        J3a, J2a = J3[a3], J2[a2]
        Ka = a3.size + a2.size
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), MU * np.ones(Ka)])
        A1 = sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        A2 = sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        E3 = sp.hstack([-J3a, sp.csr_matrix((a3.size, nf)),
                        -sp.eye(a3.size), sp.csr_matrix((a3.size, a2.size))])
        E2 = sp.hstack([-J2a, sp.csr_matrix((a2.size, nf)),
                        sp.csr_matrix((a2.size, a3.size)), -sp.eye(a2.size)])
        A_ub = sp.vstack([A1, A2, E3, E2]).tocsr()
        b_ub = np.concatenate([anchor_f, -anchor_f,
                               -THR3 + T3[a3] - J3a @ xf,
                               -THR2 + T2[a2] - J2a @ xf])
        bounds = ([(float(xf[i] - trust), float(xf[i] + trust)) for i in range(nf)]
                  + [(0.0, None)] * nf + [(0.0, None)] * Ka)
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            trust *= 0.5
            if trust < 1e-3:
                break
            continue
        cand = _apply(cur, res.x[:nf])
        v_new = _exact_viol(cand)
        if v_new < viol * (1 - 1e-9):
            cur, viol = cand, v_new
            trust = min(trust * 2, 0.5)
        else:
            trust *= 0.5
            if trust < 1e-3:
                break
    return cur


def _boxes_conflict(a, b):
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _cluster_boxes(bad, H, W):
    """Padded, size-capped boxes (y0, y1, x0, x1) inclusive corner coords."""
    merged = binary_dilation(bad, iterations=DIL)
    labels, _ = cc_label(merged)
    out = []
    for bbox in find_objects(labels):
        if bbox is None:
            continue
        y0 = max(0, bbox[0].start - PAD)
        y1 = min(H - 1, bbox[0].stop + PAD)
        x0 = max(0, bbox[1].start - PAD)
        x1 = min(W - 1, bbox[1].stop + PAD)
        # tile oversized boxes (overlapping seams get later rounds)
        ys = list(range(y0, y1, MAX_BOX)) or [y0]
        xs = list(range(x0, x1, MAX_BOX)) or [x0]
        for ty in ys:
            for tx in xs:
                out.append((ty, min(y1, ty + MAX_BOX), tx, min(x1, tx + MAX_BOX)))
    return out


def march_slice(frozen_sl, cur_sl, cur_is_upper, pool_map):
    """Repair ``cur`` against frozen neighbour. Returns (cur', n_before, n_after)."""
    H, W = cur_sl.shape[1:]
    cur = cur_sl.copy()
    anchor = cur_sl.copy()

    def _mv(c):
        lo, up = (frozen_sl, c) if cur_is_upper else (c, frozen_sl)
        return _layer_min_v(lo, up)

    n_before = int((_mv(cur) < 0).sum())
    for _ in range(MAX_ROUNDS):
        bad = _mv(cur) < THR3 - 1e-9
        if not bad.any():
            break
        boxes = _cluster_boxes(bad, H, W)
        # greedy non-conflicting batches
        batches = []
        for b in boxes:
            placed = False
            for batch in batches:
                if not any(_boxes_conflict(b, o) for o in batch):
                    batch.append(b)
                    placed = True
                    break
            if not placed:
                batches.append([b])
        for batch in batches:
            args = [(frozen_sl[:, y0:y1 + 1, x0:x1 + 1].copy(),
                     cur[:, y0:y1 + 1, x0:x1 + 1].copy(),
                     anchor[:, y0:y1 + 1, x0:x1 + 1].copy(),
                     cur_is_upper) for (y0, y1, x0, x1) in batch]
            if len(batch) == 1:
                results = [_repair_cluster(args[0])]
            else:
                results = pool_map(_repair_cluster, args, min(N_WORKERS, len(batch)))
            for (y0, y1, x0, x1), fixed in zip(batch, results):
                if fixed is None:
                    continue
                cur[:, y0 + 1:y1, x0 + 1:x1] = fixed[:, 1:-1, 1:-1]
    return cur, n_before, int((_mv(cur) < 0).sum())


def main():
    from dvfopt.core._pool import pool_map
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    SRC = Path('research/strict_feasibility_2d/runners/output') / \
        'b0039_FULL_stage1_continuous.npy'
    VOL = OUT / 'b0039_FULL_marching25d.npy'
    PROG = OUT / 'b0039_FULL_marching25d_progress.json'

    src = np.load(SRC).astype(np.float64)
    _, D, H, W = src.shape

    if VOL.exists() and PROG.exists():
        vol = np.lib.format.open_memmap(VOL, mode='r+')
        st = json.loads(PROG.read_text())
        print(f'RESUME origin={st["origin"]} up_next={st["up_next"]} '
              f'down_next={st["down_next"]}', flush=True)
    else:
        vol = np.lib.format.open_memmap(VOL, mode='w+', dtype=np.float64,
                                        shape=src.shape)
        vol[:] = src
        vol.flush()
        # mildest layer as sweep origin (cold-start fix)
        counts = [int((_layer_min_v(src[1:3, z], src[1:3, z + 1]) < 0).sum())
                  for z in range(D - 1)]
        origin = int(np.argmin(counts))
        st = {'origin': origin, 'up_next': origin + 1, 'down_next': origin,
              'layer_counts_total': int(sum(counts))}
        PROG.write_text(json.dumps(st))
        print(f'total inter-layer folds={sum(counts)}  '
              f'origin layer z*={origin} (count {counts[origin]})', flush=True)

    t0 = time.time()
    tot_l1 = 0.0

    def _step(z, frozen_z, cur_is_upper, key):
        nonlocal tot_l1
        cur = np.zeros((2, H, W))
        cur[:] = vol[1:3, z]
        frz = np.zeros((2, H, W))
        frz[:] = vol[1:3, frozen_z]
        fixed, nb, na = march_slice(frz, cur, cur_is_upper, pool_map)
        vol[1:3, z] = fixed
        vol.flush()
        tot_l1 += float(np.abs(fixed - cur).sum())
        st[key] = z + 1 if cur_is_upper else z - 1
        PROG.write_text(json.dumps(st))
        print(f'[{"up" if cur_is_upper else "dn"} z={z:3d}] folds {nb}->{na}  '
              f'(elapsed {(time.time() - t0) / 3600:.2f}h)', flush=True)

    for z in range(st['up_next'], D):        # repair z against z-1 (upper free)
        _step(z, z - 1, True, 'up_next')
    for z in range(st['down_next'], -1, -1):  # repair z against z+1 (lower free)
        _step(z, z + 1, False, 'down_next')

    arr = np.asarray(vol)
    mv = six_tet_min_volume_3d(arr)
    n_neg = int((mv <= 0).sum())
    print(f'\nMARCHING SWEEP DONE: 3D n_neg={n_neg}  '
          f'n<0.01={int((mv < 0.01 - 1e-5).sum())}  min_T={float(mv.min()):+.4f}  '
          f'added L1={tot_l1:.0f}  wall={(time.time() - t0) / 3600:.2f}h', flush=True)

    if 0 < n_neg <= 20000:
        print('light 3D mop-up (active-band)...', flush=True)
        from dvfopt.core.wallbreakers._coupled_kring_3d import (
            active_band_alm_recovery_3d,
        )
        t1 = time.time()
        arr2, _ = active_band_alm_recovery_3d(arr, threshold=0.012,
                                              n_workers=1, verbose=1)
        mv = six_tet_min_volume_3d(arr2)
        print(f'mop-up: n_neg={int((mv <= 0).sum())}  '
              f'min_T={float(mv.min()):+.4f}  (+{(time.time() - t1) / 3600:.2f}h)',
              flush=True)
        vol[:] = arr2
        vol.flush()
    print(f'saved {VOL}', flush=True)


if __name__ == '__main__':
    main()
