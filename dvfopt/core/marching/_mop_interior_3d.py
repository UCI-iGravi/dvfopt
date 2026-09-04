"""Frozen-rim 3D-interior elastic-SLP mop of residual simplex (3D) folds.

The 2.5D marching sweep freezes one WHOLE slice and moves the other's
interior, so it cannot fix folds that need BOTH slices of a folded pair to
move. This mop crops a small box around each residual fold cluster, freezes
the ENTIRE rim (all six faces -> safe paste, no boundary-seam folds), and
frees the true 3D interior ``(1:D-1, 1:H-1, 1:W-1)`` of the box so both
slices move together. On a real volume it reduced the residual fold count
from 97 -> 33; the packaged ``active_band_alm_recovery_3d`` was tried for
this and FAILED on the same residuals -- this module is the working tool.

Field layout is ``(3, D, H, W)`` = ``[dz, dy, dx]``. This module operates on
``phi[1:3]`` -> a ``(2, D, H, W)`` box ``[dy, dx]``. **dz stays exactly 0
throughout** (the 2.5D precondition) -- ``phi[0]`` is never written.

Ported from ``research/strict_feasibility_3d/runners/_marching_mopup_3d_interior.py``;
all B0039 file-path / checkpoint / resume specifics removed and module
constants turned into parameters.
"""

import functools
import time

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

from dvfopt._logging import log_info
from dvfopt.core.marching._elastic_engine import ACTIVE_WINDOW, elastic_trust_solve
from dvfopt.core.marching._precondition import require_25d_input
from dvfopt.core.primitives.tri import tri_areas_flat
from dvfopt.jacobian.tetrahedron_sign import (
    build_tet_sparse_jac,
    six_tet_min_volume_3d,
    tet_volumes_flat,
)


@functools.lru_cache(maxsize=8)
def _get_jac(D, H, W):
    """Cached ``build_tet_sparse_jac`` (per-process cache)."""
    return build_tet_sparse_jac(D, H, W)


def _stack(box):
    """box (2,D,H,W)=[dy,dx] -> [dx_flat, dy_flat, dz=0] slice-major (DX_FIRST)."""
    dx = box[1].reshape(-1)
    dy = box[0].reshape(-1)
    return np.concatenate([dx, dy, np.zeros_like(dx)])


def _viol(box, D, H, W, thr3, thr2):
    """Exact hinge violation: inter-layer simplex (3D) term + intra-slice simplex (2D) term.

    Bridges the two phi-pack conventions (CLAUDE.md): the simplex (3D) term consumes
    the DX_FIRST pack ``[dx | dy | dz]`` built by ``_stack``; the simplex (2D) term
    consumes the DY_FIRST pack ``[dy | dx]``. The length asserts guard a
    channel-order regression (a swapped/dropped channel changes the pack
    length); they are O(1) size comparisons — no per-iteration cost.
    """
    flat = _stack(box)
    assert flat.size == 3 * D * H * W, 'DX_FIRST tet pack must be [dx|dy|dz] of length 3*DHW'
    v3 = tet_volumes_flat(flat, D, H, W)
    tot = float(np.maximum(0.0, thr3 - v3).sum())
    for s in range(D):
        # tri_areas_flat takes the DY_FIRST pack ([dy, dx]) -> concat order correct.
        p2 = np.concatenate([box[0, s].ravel(), box[1, s].ravel()])
        assert p2.size == 2 * H * W, 'DY_FIRST simplex (2D) pack must be [dy|dx] of length 2*HW'
        t2 = tri_areas_flat(p2, H, W)
        tot += float(np.maximum(0.0, thr2 - t2).sum())
    return tot


def _repair_box(box, thr3, thr2, mu, max_iters, orientation_delta=None):
    """Elastic SLP over interior dy/dx of a (2,D,H,W) box; rim frozen.

    The elastic trust-region SLP loop lives in
    ``_elastic_engine.elastic_trust_solve``; this wrapper owns the
    interior-column selection, the single linearized simplex (3D) constraint block,
    and the exact-violation oracle — which ALSO includes the intra-slice
    simplex (2D) term (acceptance-only, no LP rows: any step that breaks a 2D area
    is rejected).
    """
    _, D, H, W = box.shape
    N = D * H * W
    jac3 = _get_jac(D, H, W)

    # interior node ids (not on any of the 6 faces)
    ss, ii, jj = np.meshgrid(
        np.arange(1, D - 1),
        np.arange(1, H - 1),
        np.arange(1, W - 1),
        indexing="ij",
    )
    nodes = (ss * H * W + ii * W + jj).ravel()
    if nodes.size == 0:
        return box, _viol(box, D, H, W, thr3, thr2)
    free3 = np.concatenate([nodes, N + nodes])  # dx cols, dy cols

    def anchor_vec(b):
        return np.concatenate([b[1].reshape(-1)[nodes], b[0].reshape(-1)[nodes]])

    def apply(b, v):
        out = b.copy()
        dx = out[1].reshape(-1)
        dx[nodes] = v[: nodes.size]
        dy = out[0].reshape(-1)
        dy[nodes] = v[nodes.size :]
        out[1] = dx.reshape(D, H, W)
        out[0] = dy.reshape(D, H, W)
        return out

    def blocks(b):
        # simplex (3D)-only elastic LP block; the exact-violation acceptance (which
        # includes intra-slice simplex (2D)) rejects any step that breaks a 2D area.
        pf = _stack(b)
        T3 = tet_volumes_flat(pf, D, H, W)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        a3 = np.where(thr3 + ACTIVE_WINDOW > T3)[0]
        out = [(J3[a3], T3[a3], thr3)]
        if orientation_delta is not None:
            from dvfopt.core.marching._mono_rows import axial_mono_rows, mono_block

            dxdy = np.concatenate([b[1].reshape(-1), b[0].reshape(-1)])  # [dx | dy]
            mb = mono_block(axial_mono_rows(D, H, W), dxdy, free3, orientation_delta, ACTIVE_WINDOW)
            if mb is not None:
                out.append(mb)
        return out

    def viol_fn(b):
        return _viol(b, D, H, W, thr3, thr2)

    return elastic_trust_solve(
        anchor_vec(box),
        anchor_vec(box),
        blocks,
        viol_fn,
        apply,
        state=box.copy(),
        mu=mu,
        max_iters=max_iters,
    )


def _repair_box_worker(args):
    """Pool worker: ``(box, thr3, thr2, mu, max_iters, orientation_delta) ->
    (box2, v_after)``. Module-level + single picklable tuple (spawn-safe)."""
    box, thr3, thr2, mu, max_iters, orientation_delta = args
    return _repair_box(box, thr3, thr2, mu, max_iters, orientation_delta=orientation_delta)


def _tiles(lo, hi, max_box, phase):
    """Half-open ``[lo, hi)`` as one interval when it fits ``max_box`` (or
    ``max_box`` is None), else clipped tiles of stride ``max_box`` starting at
    ``lo - phase``."""
    if max_box is None or hi - lo <= max_box:
        return [(lo, hi)]
    out = []
    for t in range(lo - phase, hi, max_box):
        a, b = max(lo, t), min(hi, t + max_box)
        if a < b:
            out.append((a, b))
    return out


def _extents_overlap(a, b):
    """True if two ``(z0, z1, y0, y1, x0, x1)`` half-open extents intersect."""
    return all(a[i] < b[i + 1] and b[i] < a[i + 1] for i in (0, 2, 4))


def _pass(
    full,
    mv,
    zpad,
    pad,
    thr3,
    thr2,
    mu,
    dil,
    max_iters,
    rep_thr,
    orientation_delta=None,
    n_workers=1,
    pool_map=None,
    max_box=90,
    phase=0,
):
    """One repair pass over all current below-threshold clusters.

    ``mv`` is the caller's already-measured per-cube min simplex (3D) volume of
    ``full`` (no recompute here). Clusters on the REPORT predicate
    ``mv < rep_thr`` (= ``threshold - 1e-5``, the pipeline's ``n_below``) —
    sub-threshold cubes are repaired, not just negatives; ``thr3`` is only the
    per-box LP target. Clustering on the LP target itself (``thr3 - 1e-9``)
    was measured to sweep up every cube the sweep parked AT its target within
    LP tolerance — 127k cubes / 700 clusters on the 528-slice B0039 volume
    against 2k / 260 under the report predicate — and turned one mop pass
    into a >24 h serial job. Returns n_fixed.

    ``n_workers > 1`` runs the boxes on ``pool_map`` in batches of pairwise-
    disjoint extents, byte-identical to the serial loop: boxes are walked in
    ``find_objects`` order and a batch is closed as soon as the next box
    overlaps one already in it, so every box in batch ``j+1`` comes AFTER every
    box in batches ``0..j`` — each box is cropped from exactly the state the
    serial loop would have cropped it from (its batch-mates are disjoint, so
    their paste-backs cannot reach it), and paste-back keeps the serial order
    and the ``v_after < v_before`` rule.
    """
    if dil < 0:
        raise ValueError(f'dil must be >= 0, got {dil}')
    bad = mv < rep_thr
    merged = binary_dilation(bad, iterations=dil) if dil >= 1 else bad
    lab, _ = cc_label(merged)
    dyx = full[1:3]
    exts = []
    for bb in find_objects(lab):
        if bb is None:
            continue
        z0 = max(0, bb[0].start - zpad)
        z1 = min(full.shape[1], bb[0].stop + zpad)
        y0 = max(0, bb[1].start - pad)
        y1 = min(full.shape[2], bb[1].stop + pad + 1)
        x0 = max(0, bb[2].start - pad)
        x1 = min(full.shape[3], bb[2].stop + pad + 1)
        # Tile boxes wider than ``max_box`` on y/x (the sweep's
        # ``_cluster_boxes`` idiom, phase-shifted across passes so a seam-
        # locked residual is tile-interior next pass). Without this a plane-
        # spanning residual cluster is ONE box that a single worker solves
        # whole — measured 5.6 h on the 528-slice B0039 volume while the
        # other workers idled. Boxes within ``max_box`` are left as-is, so the
        # small-box path (and every test) is byte-identical to the uncapped mop.
        for ty0, ty1 in _tiles(y0, y1, max_box, phase):
            for tx0, tx1 in _tiles(x0, x1, max_box, phase):
                exts.append((z0, z1 + 1, ty0, ty1, tx0, tx1))  # half-open, as sliced

    def _crop(e):
        # (box, v_before), or None when the crop is too small / already clean.
        z0, z1, y0, y1, x0, x1 = e
        box = dyx[:, z0:z1, y0:y1, x0:x1].copy()
        D, H, W = box.shape[1:]
        if D < 3 or H < 3 or W < 3:
            return None
        v_before = _viol(box, D, H, W, thr3, thr2)
        return None if v_before <= 1e-12 else (box, v_before)

    def _paste(e, box2, v_before, v_after):
        z0, z1, y0, y1, x0, x1 = e
        if v_after < v_before:
            dyx[:, z0:z1, y0:y1, x0:x1] = box2
            return 1
        return 0

    fixed = 0
    if n_workers <= 1 or pool_map is None:
        for e in exts:
            c = _crop(e)
            if c is None:
                continue
            box, v_before = c
            box2, v_after = _repair_box(
                box, thr3, thr2, mu, max_iters, orientation_delta=orientation_delta
            )
            fixed += _paste(e, box2, v_before, v_after)
        return fixed

    # Contiguous runs of pairwise-disjoint extents, processed in order.
    batches = [[]]
    for e in exts:
        if any(_extents_overlap(e, o) for o in batches[-1]):
            batches.append([])
        batches[-1].append(e)
    for batch in batches:
        crops = [(e, c) for e in batch if (c := _crop(e)) is not None]
        if not crops:
            continue
        args = [(box, thr3, thr2, mu, max_iters, orientation_delta) for _, (box, _) in crops]
        results = (
            [_repair_box_worker(a) for a in args]
            if len(args) == 1
            else pool_map(_repair_box_worker, args, n_workers)
        )
        for (e, (_, v_before)), (box2, v_after) in zip(crops, results):
            fixed += _paste(e, box2, v_before, v_after)
    return fixed


def mop_interior_3d(
    phi,
    *,
    threshold=0.01,
    thr3=None,
    thr2=0.01,
    mu=1000.0,
    pass_pads=((2, 4), (3, 6)),
    dil=1,
    max_iters=40,
    dz_tol=1e-12,
    copy=True,
    verbose=0,
    orientation_delta=None,
    n_workers=1,
    pool_map=None,
    max_box=90,
):
    """Frozen-rim 3D-interior elastic-SLP mop of residual simplex (3D) folds.

    Crops a small box around each residual below-threshold cluster (the
    pipeline-report predicate ``min_vol < threshold - 1e-5`` — sub-threshold
    cubes are repaired, not just negatives; ``thr3`` is the per-box LP
    target), freezes the entire rim (all six faces,
    giving a seam-safe paste), and frees the true 3D interior
    ``(1:D-1, 1:H-1, 1:W-1)`` so both slices of a folded pair move together. Each cropped box is repaired with an elastic sequential-LP
    (inter-layer simplex (3D) linearized rows + intra-slice simplex (2D) exact-violation
    acceptance, trust-region). The 2.5D precondition ``dz == 0`` is preserved:
    only ``phi[1:3]`` (``[dy, dx]``) is ever written.

    Parameters
    ----------
    phi : ndarray, shape (3, D, H, W)
        Field ``[dz, dy, dx]``. ``phi[0]`` must be (and remains) zero —
        validated up front (raises ``ValueError`` on nonzero dz or any
        non-finite value). Operated on via a copy by default -- see ``copy``.
    threshold : float, default 0.01
        The strict feasibility target for simplex (3D) volumes.
    thr3 : float or None, default None
        simplex (3D) feasibility target with margin. Defaults to
        ``threshold + 1e-4`` when None (strict target with a small margin).
    thr2 : float, default 0.01
        Intra-slice simplex (2D) area feasibility target.
    mu : float, default 1000.0
        Elastic-slack penalty weight in the per-box LP objective.
    pass_pads : sequence of (zpad, pad), default ((2, 4), (3, 6))
        Escalating ``(zpad, pad)`` schedule -- one outer pass per entry, each
        expanding cropped boxes by ``zpad`` on z and ``pad`` on y/x.
        **Kept deliberately short.** Measured on a real volume: pad=4 fixed
        most clusters, pad=6 fixed a couple more, and pads >= 9 cost
        exponentially more wall time for ZERO additional fixes (the remaining
        folds are at the geometric floor). Do NOT "helpfully" escalate this.
    dil : int, default 1
        Binary-dilation iterations used to merge nearby fold voxels into
        clusters before bounding-box extraction.
    max_iters : int, default 40
        Trust-region SLP iterations per box repair.
    dz_tol : float, default 1e-12
        Tolerance for the ``dz == 0`` precondition check.
    copy : bool, default True
        When True (the public default), operate on a float64 copy — the
        caller's array is never mutated. When False, operate on the caller's
        array directly (the caller relinquishes it; it must already be a
        float64 ``(3, D, H, W)`` array).
    verbose : int, default 0
        ``>= 1`` prints one bracketed line per pass.
    orientation_delta : float or None, default None
        Opt-in axial edge-monotonicity rows in every box LP (``None`` = off).
    n_workers : int, default 1
        ``> 1`` repairs a pass's boxes on the shared spawn pool
        (``dvfopt.core._pool.pool_map``, imported lazily unless ``pool_map``
        is given) in batches of pairwise-disjoint boxes, processed in the
        serial order — byte-identical to ``n_workers=1`` (see ``_pass``).
        ``<= 1`` is the unchanged serial loop.
    max_box : int or None, default 90
        Boxes wider than this on y or x are tiled (stride ``max_box``,
        phase-shifted by ``max_box // 2`` on even passes so seam-locked
        residuals are tile-interior next pass — the sweep's idiom). The tiles
        of one giant box are pairwise disjoint, so they run in ONE parallel
        batch instead of one worker solving the whole box for hours. Boxes
        within ``max_box`` are untouched; ``None`` disables the cap.

    Returns
    -------
    phi_out : ndarray
        The corrected field (a copy unless ``copy=False``).
    info : dict
        Keys: ``n_neg_before``/``n_neg_after`` (per-cube fold counts,
        ``mv <= 0``), ``n_below_before``/``n_below_after`` (per-cube
        sub-threshold counts under the repair predicate ``mv < threshold -
        1e-5``), ``n_below_report_after`` (the same predicate — kept as an
        alias of ``n_below_after`` for callers of the old key), ``min_T_after``, ``n_fixed``,
        ``passes`` (list of per-pass dicts), ``wall_s``.
    """
    if thr3 is None:
        thr3 = threshold + 1e-4
    rep_thr = threshold - 1e-5  # the repair predicate (pipeline-report n_below)
    if pool_map is None and n_workers > 1:
        from dvfopt.core._pool import pool_map

    phi = np.asarray(phi)
    require_25d_input(phi, dz_tol)

    phi_out = phi if not copy else np.array(phi, dtype=np.float64, copy=True)

    t0 = time.time()
    # Single fused measurement per state (per-cube min volume); each pass's
    # post-measurement is carried forward as the next pass's "before".
    mv = six_tet_min_volume_3d(phi_out)
    n_neg_before = int((mv <= 0).sum())
    n_below_before = int((mv < rep_thr).sum())

    passes = []
    total_fixed = 0
    for i, (zpad, pad) in enumerate(pass_pads, start=1):
        before_neg = int((mv <= 0).sum())
        before_below = int((mv < rep_thr).sum())
        if before_below == 0:
            break
        fixed = _pass(
            phi_out,
            mv,
            zpad,
            pad,
            thr3,
            thr2,
            mu,
            dil,
            max_iters,
            rep_thr,
            orientation_delta,
            n_workers=n_workers,
            pool_map=pool_map,
            max_box=max_box,
            phase=((i - 1) % 2) * (max_box // 2) if max_box else 0,
        )
        total_fixed += fixed
        mv = six_tet_min_volume_3d(phi_out)
        after_neg = int((mv <= 0).sum())
        after_below = int((mv < rep_thr).sum())
        mn = float(mv.min())
        passes.append(
            {
                "pass": i,
                "zpad": zpad,
                "pad": pad,
                "n_neg_before": before_neg,
                "n_neg_after": after_neg,
                "n_below_before": before_below,
                "n_below_after": after_below,
                "min_T": mn,
                "n_fixed": fixed,
            }
        )
        if verbose >= 1:
            log_info(
                f"  [mop pass {i} zpad={zpad} pad={pad}] "
                f"n_neg {before_neg}->{after_neg} "
                f"n_below {before_below}->{after_below} min_T={mn:.5f}",
            )

    # `mv` already measures the final state (post last pass, or the initial
    # measurement when no pass ran) — no extra full-volume measurement.
    info = {
        "n_neg_before": n_neg_before,
        "n_neg_after": int((mv <= 0).sum()),
        "n_below_before": n_below_before,
        "n_below_after": int((mv < rep_thr).sum()),
        "n_below_report_after": int((mv < rep_thr).sum()),
        "min_T_after": float(mv.min()),
        "n_fixed": total_fixed,
        "passes": passes,
        "wall_s": time.time() - t0,
    }
    return phi_out, info
