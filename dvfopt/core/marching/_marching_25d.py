"""2.5D marching sweep core — inter-layer 6-tet fold repair primitives.

Ported from ``research/strict_feasibility_3d/runners/_marching_full_volume.py``
(the productized "Part XXI option A" experiment), with the former module
constants (``THR3``/``THR2``/``MU``/``PAD``/``DIL``/``MAX_ROUNDS``/``MAX_BOX``)
turned into function parameters.

Data conventions
----------------
- The full field is ``(3, D, H, W)`` = ``[dz, dy, dx]``.
- A "slice" for marching is ``phi[1:3, z]`` → a ``(2, H, W)`` array = ``[dy, dx]``.
- ``_stack_flat`` builds a DX_FIRST flat pack ``[dx, dy, dz=zeros]`` for the
  6-tet primitives. The hardcoded ``dz = zeros`` IS the 2.5D precondition.
- The 2-tri primitives use the opposite (DY_FIRST) pack:
  ``[:HW] = dy, [HW:] = dx``. ``_repair_cluster`` bridges the two packs.

Windows spawn note: this module's top level is imports only, and
``_repair_cluster`` is a module-top-level function taking a single
picklable ``args`` tuple, so it is a valid ``ProcessPoolExecutor`` worker
(workers re-import this module). Solver-specific thresholds are carried
INSIDE the args tuple rather than read from module globals.
"""

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label
from scipy.optimize import linprog

from dvfopt.core.slp.tri_linearize import build_sparse_jacobian_T
from dvfopt.core.tri_primitives import tri_areas_flat
from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac, tet_volumes_flat


def _stack_flat(lower, upper):
    """DX_FIRST flat pack ``[dx, dy, dz=zeros]`` for the two-layer 6-tet stack.

    ``lower``/``upper`` are ``(2, H, W)`` ``[dy, dx]`` slices; the returned
    vector feeds ``tet_volumes_flat(flat, D=2, H, W)`` /
    ``build_tet_sparse_jac(2, H, W)``. The zero dz block is the 2.5D
    precondition (in-plane displacement only across the layer pair).
    """
    dx = np.concatenate([lower[1].ravel(), upper[1].ravel()])
    dy = np.concatenate([lower[0].ravel(), upper[0].ravel()])
    return np.concatenate([dx, dy, np.zeros_like(dx)])


def layer_min_v(lower, upper):
    """Per-cell minimum 6-tet volume across the ``(lower, upper)`` layer pair.

    Returns an ``(H-1, W-1)`` array; negative entries are inter-layer folds.
    """
    Hc, Wc = lower.shape[1:]
    V = tet_volumes_flat(_stack_flat(lower, upper), 2, Hc, Wc)
    return V.reshape(6, Hc - 1, Wc - 1).min(axis=0)


def _repair_cluster(args):
    """Pool worker: elastic SLP over the FREE plane's interior dy/dx.

    ``args = (frozen_c, cur_c, anchor_c, cur_is_upper, thr3, thr2, mu,
    max_lp_iters)``. The stack is built in geometric order (lower, upper);
    the free columns are plane 1 if ``cur_is_upper`` else plane 0.

    Must remain a module-top-level function taking a single picklable
    tuple so it is usable as a ``ProcessPoolExecutor`` worker.
    """
    frozen_c, cur_c, anchor_c, cur_is_upper, thr3, thr2, mu, max_lp_iters = args
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
        return (float(np.maximum(0, thr3 - v3).sum())
                + float(np.maximum(0, thr2 - t2).sum()))

    anchor_f = _free_vec(anchor_c)
    cur = cur_c.copy()
    viol = _exact_viol(cur)
    trust = 0.5
    for _ in range(max_lp_iters):
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
        a3 = np.where(T3 < thr3 + 0.5)[0]
        a2 = np.where(T2 < thr2 + 0.5)[0]
        J3a, J2a = J3[a3], J2[a2]
        Ka = a3.size + a2.size
        c_obj = np.concatenate([np.zeros(nf), np.ones(nf), mu * np.ones(Ka)])
        A1 = sp.hstack([sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        A2 = sp.hstack([-sp.eye(nf), -sp.eye(nf), sp.csr_matrix((nf, Ka))])
        E3 = sp.hstack([-J3a, sp.csr_matrix((a3.size, nf)),
                        -sp.eye(a3.size), sp.csr_matrix((a3.size, a2.size))])
        E2 = sp.hstack([-J2a, sp.csr_matrix((a2.size, nf)),
                        sp.csr_matrix((a2.size, a3.size)), -sp.eye(a2.size)])
        A_ub = sp.vstack([A1, A2, E3, E2]).tocsr()
        b_ub = np.concatenate([anchor_f, -anchor_f,
                               -thr3 + T3[a3] - J3a @ xf,
                               -thr2 + T2[a2] - J2a @ xf])
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
    """True if inclusive boxes ``(y0, y1, x0, x1)`` overlap."""
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _cluster_boxes(bad, H, W, pad, dil, max_box):
    """Padded, size-capped boxes ``(y0, y1, x0, x1)`` (inclusive corners).

    Dilates ``bad`` by ``dil`` to merge nearby violations, labels connected
    components, pads each bbox by ``pad`` (clipped to the grid), then tiles
    boxes larger than ``max_box`` on either axis (overlapping seams are
    healed by later rounds).
    """
    merged = binary_dilation(bad, iterations=dil)
    labels, _ = cc_label(merged)
    out = []
    for bbox in find_objects(labels):
        if bbox is None:
            continue
        y0 = max(0, bbox[0].start - pad)
        y1 = min(H - 1, bbox[0].stop + pad)
        x0 = max(0, bbox[1].start - pad)
        x1 = min(W - 1, bbox[1].stop + pad)
        # tile oversized boxes (overlapping seams get later rounds)
        ys = list(range(y0, y1, max_box)) or [y0]
        xs = list(range(x0, x1, max_box)) or [x0]
        for ty in ys:
            for tx in xs:
                out.append((ty, min(y1, ty + max_box), tx, min(x1, tx + max_box)))
    return out


def march_slice(frozen_sl, cur_sl, cur_is_upper, *, thr3=0.01 + 1e-4, thr2=0.01,
                mu=1000.0, pad=4, dil=2, max_rounds=6, max_box=90,
                n_workers=1, pool_map=None, max_lp_iters=12):
    """Repair ``cur_sl`` against the frozen neighbour. Returns (cur', n_before, n_after).

    ``frozen_sl``, ``cur_sl`` are ``(2, H, W)`` ``[dy, dx]`` slices. The
    returned ``cur'`` is a fresh array (inputs are never mutated). Only the
    interior of each repaired box is pasted back (the box rim stays frozen),
    so the outer rim of ``cur'`` always equals ``cur_sl``'s rim.

    ``n_before``/``n_after`` are inter-layer fold counts
    (``int((layer_min_v(lo, up) < 0).sum())`` with the correct geometric
    order per ``cur_is_upper``).

    Parallelism seam: if ``pool_map`` is None, ``n_workers == 1``, or a
    batch has a single box, clusters run serially via ``_repair_cluster``;
    otherwise ``pool_map(_repair_cluster, args_list, min(n_workers, len(batch)))``
    dispatches the batch. The caller injects ``pool_map`` (this module never
    imports the process pool, keeping it import-light and unit-testable).
    """
    H, W = cur_sl.shape[1:]
    cur = cur_sl.copy()
    anchor = cur_sl.copy()

    def _mv(c):
        lo, up = (frozen_sl, c) if cur_is_upper else (c, frozen_sl)
        return layer_min_v(lo, up)

    n_before = int((_mv(cur) < 0).sum())
    for _ in range(max_rounds):
        bad = _mv(cur) < thr3 - 1e-9
        if not bad.any():
            break
        boxes = _cluster_boxes(bad, H, W, pad, dil, max_box)
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
                     cur_is_upper, thr3, thr2, mu, max_lp_iters)
                    for (y0, y1, x0, x1) in batch]
            if pool_map is None or n_workers == 1 or len(batch) == 1:
                results = [_repair_cluster(a) for a in args]
            else:
                results = pool_map(_repair_cluster, args, min(n_workers, len(batch)))
            for (y0, y1, x0, x1), fixed in zip(batch, results):
                if fixed is None:
                    continue
                cur[:, y0 + 1:y1, x0 + 1:x1] = fixed[:, 1:-1, 1:-1]
    return cur, n_before, int((_mv(cur) < 0).sum())
