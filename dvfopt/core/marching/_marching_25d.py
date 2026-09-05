"""2.5D marching sweep core — inter-layer simplex (3D) fold repair primitives.

Ported from ``research/strict_feasibility_3d/runners/_marching_full_volume.py``
(the productized "Part XXI option A" experiment), with the former module
constants (``THR3``/``THR2``/``MU``/``PAD``/``DIL``/``MAX_ROUNDS``/``MAX_BOX``)
turned into function parameters.

Data conventions
----------------
- The full field is ``(3, D, H, W)`` = ``[dz, dy, dx]``.
- A "slice" for marching is ``phi[1:3, z]`` → a ``(2, H, W)`` array = ``[dy, dx]``.
- ``_stack_flat`` builds a DX_FIRST flat pack ``[dx, dy, dz=zeros]`` for the
  simplex (3D) primitives. The hardcoded ``dz = zeros`` IS the 2.5D precondition.
- The simplex (2D) primitives use the opposite (DY_FIRST) pack:
  ``[:HW] = dy, [HW:] = dx``. ``_repair_cluster`` bridges the two packs.

Windows spawn note: ``_repair_cluster`` is a module-top-level function taking
a single picklable ``args`` tuple, so it is a valid ``ProcessPoolExecutor``
worker. Spawned workers re-import this module — which also executes every
ancestor package ``__init__`` (``dvfopt``, ``dvfopt.core``,
``dvfopt.core.marching``) — so the real requirement is that this module's top
level AND those package ``__init__``s stay side-effect-safe and import-light.
Solver-specific thresholds are carried INSIDE the args tuple rather than read
from module globals.
"""

import functools

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

from dvfopt.core.marching._elastic_engine import ACTIVE_WINDOW, elastic_trust_solve
from dvfopt.core.primitives.tri import tri_areas_flat
from dvfopt.core.slp.tri_linearize import build_sparse_jacobian_T
from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac, tet_volumes_flat


@functools.lru_cache(maxsize=8)
def _get_jac(D, H, W):
    """Cached ``build_tet_sparse_jac`` (per-process; pool workers build their own)."""
    return build_tet_sparse_jac(D, H, W)


def _stack_flat(lower, upper):
    """DX_FIRST flat pack ``[dx, dy, dz=zeros]`` for the two-layer simplex (3D) stack.

    ``lower``/``upper`` are ``(2, H, W)`` ``[dy, dx]`` slices; the returned
    vector feeds ``tet_volumes_flat(flat, D=2, H, W)`` /
    ``build_tet_sparse_jac(2, H, W)``. The zero dz block is the 2.5D
    precondition (in-plane displacement only across the layer pair).
    """
    dx = np.concatenate([lower[1].ravel(), upper[1].ravel()])
    dy = np.concatenate([lower[0].ravel(), upper[0].ravel()])
    return np.concatenate([dx, dy, np.zeros_like(dx)])


def layer_min_v(lower, upper):
    """Per-cell minimum simplex (3D) volume across the ``(lower, upper)`` layer pair.

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
    tuple so it is usable as a ``ProcessPoolExecutor`` worker (the pool
    pickles the function reference by name plus the args tuple, so the
    closures defined inside are fine).

    The elastic trust-region SLP loop itself lives in
    ``_elastic_engine.elastic_trust_solve``; this worker owns the free-column
    selection, the two linearized constraint blocks (inter-layer simplex (3D) rows
    FIRST, then per-slice simplex (2D) — the slack layout depends on this order) and
    the exact-violation acceptance oracle (both families).
    """
    frozen_c, cur_c, anchor_c, cur_is_upper, thr3, thr2, mu, max_lp_iters, orientation_delta = args
    Hc, Wc = cur_c.shape[1:]
    n_pix = Hc * Wc
    jac3 = _get_jac(2, Hc, Wc)

    inner = np.zeros((Hc, Wc), dtype=bool)
    inner[1:-1, 1:-1] = True
    ii = np.where(inner.ravel())[0]
    n2 = 2 * n_pix
    plane_off = n_pix if cur_is_upper else 0
    cols_dx3 = plane_off + ii
    cols_dy3 = n2 + plane_off + ii
    free3 = np.concatenate([cols_dx3, cols_dy3])
    free2 = np.concatenate([n_pix + ii, ii])  # J2 [dy, dx] -> [dx_f, dy_f]

    def _geo(c):
        return (frozen_c, c) if cur_is_upper else (c, frozen_c)

    # ---- Pack-layout guards (CLAUDE.md: never bridge DX_FIRST/DY_FIRST
    # silently). The simplex (3D) side is the DX_FIRST flat pack [dx | dy | dz] of
    # length 3*(2*n_pix) whose dz third is identically zero (2.5D
    # precondition); the simplex (2D) side is the DY_FIRST pack [dy | dx] of length
    # 2*n_pix. free3/free2 must index only inside those layouts — a
    # channel-order regression shifts these offsets/lengths and trips the
    # asserts once per cluster, before any LP work.
    _pf0 = _stack_flat(*_geo(cur_c))
    assert _pf0.size == 3 * n2, 'DX_FIRST tet pack must be [dx|dy|dz] of length 3*(2*HW)'
    assert not _pf0[2 * n2 :].any(), 'dz third of the DX_FIRST tet pack must be zero (2.5D)'
    assert np.concatenate([cur_c[0].ravel(), cur_c[1].ravel()]).size == 2 * n_pix, (
        'DY_FIRST simplex (2D) pack must be [dy|dx] of length 2*HW'
    )
    if ii.size:
        assert free3.max() < 2 * n2, 'free3 must stay inside the dx/dy thirds (never dz)'
        assert free2.max() < 2 * n_pix, 'free2 out of DY_FIRST pack range'

    def _free_vec(c):
        return np.concatenate([c[1].ravel()[ii], c[0].ravel()[ii]])

    def _apply(c, v):
        out = c.copy()
        nf = ii.size
        dx = out[1].ravel()
        dx[ii] = v[:nf]
        out[1] = dx.reshape(Hc, Wc)
        dy = out[0].ravel()
        dy[ii] = v[nf:]
        out[0] = dy.reshape(Hc, Wc)
        return out

    def _exact_viol(c):
        lo, up = _geo(c)
        v3 = tet_volumes_flat(_stack_flat(lo, up), 2, Hc, Wc)
        t2 = tri_areas_flat(np.concatenate([c[0].ravel(), c[1].ravel()]), Hc, Wc)
        return float(np.maximum(0, thr3 - v3).sum()) + float(np.maximum(0, thr2 - t2).sum())

    def _blocks(c):
        lo, up = _geo(c)
        pf = _stack_flat(lo, up)
        T3 = tet_volumes_flat(pf, 2, Hc, Wc)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        p2 = np.concatenate([c[0].ravel(), c[1].ravel()])
        T2 = tri_areas_flat(p2, Hc, Wc)
        J2 = build_sparse_jacobian_T(p2, Hc, Wc).tocsc()[:, free2].tocsr()
        a3 = np.where(thr3 + ACTIVE_WINDOW > T3)[0]
        a2 = np.where(thr2 + ACTIVE_WINDOW > T2)[0]
        # Block order matters: simplex (3D) rows first, then simplex (2D) (slack layout);
        # the optional monotonicity block is APPENDED (never inserted).
        blocks = [(J3[a3], T3[a3], thr3), (J2[a2], T2[a2], thr2)]
        if orientation_delta is not None:
            from dvfopt.core.marching._mono_rows import axial_mono_rows, mono_block

            # current plane's [dx | dy] flat; free2 is [dy, dx] of the DY_FIRST
            # pack — the mono layout is [dx | dy], so free cols are [ii, n+ii]
            dxdy = np.concatenate([c[1].ravel(), c[0].ravel()])
            mb = mono_block(
                axial_mono_rows(1, Hc, Wc),
                dxdy,
                np.concatenate([ii, n_pix + ii]),
                orientation_delta,
                ACTIVE_WINDOW,
            )
            if mb is not None:
                blocks.append(mb)
        return blocks

    cur, _ = elastic_trust_solve(
        _free_vec(cur_c),
        _free_vec(anchor_c),
        _blocks,
        _exact_viol,
        _apply,
        state=cur_c.copy(),
        mu=mu,
        max_iters=max_lp_iters,
    )
    return cur


def _boxes_conflict(a, b):
    """True if inclusive boxes ``(y0, y1, x0, x1)`` overlap."""
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _cluster_boxes(bad, H, W, pad, dil, max_box, phase=0):
    """Padded, size-capped boxes ``(y0, y1, x0, x1)`` (inclusive corners).

    Dilates ``bad`` by ``dil`` to merge nearby violations (``dil == 0`` means
    NO dilation — never scipy's repeat-until-convergence), labels connected
    components, pads each bbox by ``pad`` (clipped to the grid), then tiles
    boxes larger than ``max_box`` on either axis. ``phase`` shifts the tile
    grid backwards by that many cells: a node sitting on a tile seam at one
    phase is tile-interior at another, so alternating the phase across rounds
    deterministically heals seam-locked residuals.
    """
    if dil < 0:
        raise ValueError(f'dil must be >= 0, got {dil}')
    merged = binary_dilation(bad, iterations=dil) if dil >= 1 else bad
    labels, _ = cc_label(merged)
    out = []
    for bbox in find_objects(labels):
        if bbox is None:
            continue
        y0 = max(0, bbox[0].start - pad)
        y1 = min(H - 1, bbox[0].stop + pad)
        x0 = max(0, bbox[1].start - pad)
        x1 = min(W - 1, bbox[1].stop + pad)
        # tile oversized boxes on a phase-shifted grid, clipping each tile
        # back to the padded bbox and skipping tiles emptied by the clip
        ys = list(range(y0 - phase, y1, max_box)) or [y0]
        xs = list(range(x0 - phase, x1, max_box)) or [x0]
        for ty in ys:
            ty0, ty1 = max(y0, ty), min(y1, ty + max_box)
            if ty0 > ty1:
                continue
            for tx in xs:
                tx0, tx1 = max(x0, tx), min(x1, tx + max_box)
                if tx0 > tx1:
                    continue
                out.append((ty0, ty1, tx0, tx1))
    return out


def march_slice(
    frozen_sl,
    cur_sl,
    cur_is_upper,
    *,
    thr3=0.01 + 1e-4,
    thr2=0.01,
    mu=1000.0,
    pad=4,
    dil=2,
    max_rounds=6,
    max_box=90,
    n_workers=1,
    pool_map=None,
    max_lp_iters=12,
    orientation_delta=None,
):
    """Repair ``cur_sl`` against the frozen neighbour. Returns (cur', n_before, n_after).

    ``frozen_sl``, ``cur_sl`` are ``(2, H, W)`` ``[dy, dx]`` slices. The
    returned ``cur'`` is a fresh array (inputs are never mutated). Only the
    interior of each repaired box is pasted back (the box rim stays frozen),
    so the outer rim of ``cur'`` always equals ``cur_sl``'s rim.

    ``n_before``/``n_after`` are inter-layer fold counts
    (``int((layer_min_v(lo, up) < 0).sum())`` with the correct geometric
    order per ``cur_is_upper``).

    Parallelism seam: if ``pool_map`` is None with ``n_workers == 1``, or a
    batch has a single box, clusters run serially via ``_repair_cluster``;
    otherwise ``pool_map(_repair_cluster, args_list, n_workers)`` dispatches
    the batch (always the full ``n_workers`` — the shared pool is sized once
    and torn down/rebuilt whenever a different size is requested, so passing
    a per-batch size would thrash the warm pool; idle workers are free).
    When ``pool_map`` is None but ``n_workers > 1``, the shared
    ``dvfopt.core._pool.pool_map`` is imported lazily; an explicitly provided
    ``pool_map`` always wins (keeping the module import-light and
    unit-testable).
    """
    H, W = cur_sl.shape[1:]
    cur = cur_sl.copy()
    anchor = cur_sl.copy()

    if pool_map is None and n_workers > 1:
        from dvfopt.core._pool import pool_map

    def _mv(c):
        lo, up = (frozen_sl, c) if cur_is_upper else (c, frozen_sl)
        return layer_min_v(lo, up)

    n_before = int((_mv(cur) < 0).sum())
    for rnd in range(max_rounds):
        bad = _mv(cur) < thr3 - 1e-9
        if not bad.any():
            break
        # Alternate the tile-grid phase between rounds so seam-locked nodes
        # become tile-interior on the next round (round 0 keeps phase 0).
        boxes = _cluster_boxes(bad, H, W, pad, dil, max_box, phase=(rnd % 2) * (max_box // 2))
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
            args = [
                (
                    frozen_sl[:, y0 : y1 + 1, x0 : x1 + 1].copy(),
                    cur[:, y0 : y1 + 1, x0 : x1 + 1].copy(),
                    anchor[:, y0 : y1 + 1, x0 : x1 + 1].copy(),
                    cur_is_upper,
                    thr3,
                    thr2,
                    mu,
                    max_lp_iters,
                    orientation_delta,
                )
                for (y0, y1, x0, x1) in batch
            ]
            if pool_map is None or n_workers == 1 or len(batch) == 1:
                results = [_repair_cluster(a) for a in args]
            else:
                results = pool_map(_repair_cluster, args, n_workers)
            for (y0, y1, x0, x1), fixed in zip(batch, results):
                if fixed is None:
                    continue
                cur[:, y0 + 1 : y1, x0 + 1 : x1] = fixed[:, 1:-1, 1:-1]
    return cur, n_before, int((_mv(cur) < 0).sum())
