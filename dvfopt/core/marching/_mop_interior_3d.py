"""Frozen-rim 3D-interior elastic-SLP mop of residual 6-tet folds.

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
from dvfopt.core.tri_primitives import tri_areas_flat
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
    """Exact hinge violation: inter-layer 6-tet term + intra-slice 2-tri term.

    Bridges the two phi-pack conventions (CLAUDE.md): the 6-tet term consumes
    the DX_FIRST pack ``[dx | dy | dz]`` built by ``_stack``; the 2-tri term
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
        assert p2.size == 2 * H * W, 'DY_FIRST 2-tri pack must be [dy|dx] of length 2*HW'
        t2 = tri_areas_flat(p2, H, W)
        tot += float(np.maximum(0.0, thr2 - t2).sum())
    return tot


def _repair_box(box, thr3, thr2, mu, max_iters):
    """Elastic SLP over interior dy/dx of a (2,D,H,W) box; rim frozen.

    The elastic trust-region SLP loop lives in
    ``_elastic_engine.elastic_trust_solve``; this wrapper owns the
    interior-column selection, the single linearized 6-tet constraint block,
    and the exact-violation oracle — which ALSO includes the intra-slice
    2-tri term (acceptance-only, no LP rows: any step that breaks a 2D area
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
        # 6-tet-only elastic LP block; the exact-violation acceptance (which
        # includes intra-slice 2-tri) rejects any step that breaks a 2D area.
        pf = _stack(b)
        T3 = tet_volumes_flat(pf, D, H, W)
        J3 = jac3(pf).tocsc()[:, free3].tocsr()
        a3 = np.where(thr3 + ACTIVE_WINDOW > T3)[0]
        return [(J3[a3], T3[a3], thr3)]

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


def _pass(full, mv, zpad, pad, thr3, thr2, mu, dil, max_iters):
    """One repair pass over all current below-threshold clusters.

    ``mv`` is the caller's already-measured per-cube min 6-tet volume of
    ``full`` (no recompute here). Clusters on the sweep's predicate
    ``mv < thr3 - 1e-9`` — sub-threshold cubes are repaired, not just
    negatives. Returns n_fixed.
    """
    if dil < 0:
        raise ValueError(f'dil must be >= 0, got {dil}')
    bad = mv < thr3 - 1e-9
    merged = binary_dilation(bad, iterations=dil) if dil >= 1 else bad
    lab, _ = cc_label(merged)
    boxes = find_objects(lab)
    dyx = full[1:3]
    fixed = 0
    for bb in boxes:
        if bb is None:
            continue
        z0 = max(0, bb[0].start - zpad)
        z1 = min(full.shape[1], bb[0].stop + zpad)
        y0 = max(0, bb[1].start - pad)
        y1 = min(full.shape[2], bb[1].stop + pad + 1)
        x0 = max(0, bb[2].start - pad)
        x1 = min(full.shape[3], bb[2].stop + pad + 1)
        box = dyx[:, z0 : z1 + 1, y0:y1, x0:x1].copy()
        D, H, W = box.shape[1:]
        if D < 3 or H < 3 or W < 3:
            continue
        v_before = _viol(box, D, H, W, thr3, thr2)
        if v_before <= 1e-12:
            continue
        box2, v_after = _repair_box(box, thr3, thr2, mu, max_iters)
        if v_after < v_before:
            dyx[:, z0 : z1 + 1, y0:y1, x0:x1] = box2
            fixed += 1
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
):
    """Frozen-rim 3D-interior elastic-SLP mop of residual 6-tet folds.

    Crops a small box around each residual below-threshold cluster (the
    sweep's predicate ``min_vol < thr3 - 1e-9`` — sub-threshold cubes are
    repaired, not just negatives), freezes the entire rim (all six faces,
    giving a seam-safe paste), and frees the true 3D interior
    ``(1:D-1, 1:H-1, 1:W-1)`` so both slices of a folded pair move together. Each cropped box is repaired with an elastic sequential-LP
    (inter-layer 6-tet linearized rows + intra-slice 2-tri exact-violation
    acceptance, trust-region). The 2.5D precondition ``dz == 0`` is preserved:
    only ``phi[1:3]`` (``[dy, dx]``) is ever written.

    Parameters
    ----------
    phi : ndarray, shape (3, D, H, W)
        Field ``[dz, dy, dx]``. ``phi[0]`` must be (and remains) zero —
        validated up front (raises ``ValueError`` on nonzero dz or any
        non-finite value). Operated on via a copy by default -- see ``copy``.
    threshold : float, default 0.01
        The strict feasibility target for 6-tet volumes.
    thr3 : float or None, default None
        6-tet feasibility target with margin. Defaults to
        ``threshold + 1e-4`` when None (strict target with a small margin).
    thr2 : float, default 0.01
        Intra-slice 2-tri area feasibility target.
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

    Returns
    -------
    phi_out : ndarray
        The corrected field (a copy unless ``copy=False``).
    info : dict
        Keys: ``n_neg_before``/``n_neg_after`` (per-cube fold counts,
        ``mv <= 0``), ``n_below_before``/``n_below_after`` (per-cube
        sub-threshold counts, ``mv < thr3 - 1e-9`` — the repair predicate),
        ``n_below_report_after`` (per-cube ``mv < threshold - 1e-5`` — the
        pipeline-report semantics), ``min_T_after``, ``n_fixed``,
        ``passes`` (list of per-pass dicts), ``wall_s``.
    """
    if thr3 is None:
        thr3 = threshold + 1e-4

    phi = np.asarray(phi)
    require_25d_input(phi, dz_tol)

    phi_out = phi if not copy else np.array(phi, dtype=np.float64, copy=True)

    t0 = time.time()
    # Single fused measurement per state (per-cube min volume); each pass's
    # post-measurement is carried forward as the next pass's "before".
    mv = six_tet_min_volume_3d(phi_out)
    n_neg_before = int((mv <= 0).sum())
    n_below_before = int((mv < thr3 - 1e-9).sum())

    passes = []
    total_fixed = 0
    for i, (zpad, pad) in enumerate(pass_pads, start=1):
        before_neg = int((mv <= 0).sum())
        before_below = int((mv < thr3 - 1e-9).sum())
        if before_below == 0:
            break
        fixed = _pass(phi_out, mv, zpad, pad, thr3, thr2, mu, dil, max_iters)
        total_fixed += fixed
        mv = six_tet_min_volume_3d(phi_out)
        after_neg = int((mv <= 0).sum())
        after_below = int((mv < thr3 - 1e-9).sum())
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
        "n_below_after": int((mv < thr3 - 1e-9).sum()),
        "n_below_report_after": int((mv < threshold - 1e-5).sum()),
        "min_T_after": float(mv.min()),
        "n_fixed": total_fixed,
        "passes": passes,
        "wall_s": time.time() - t0,
    }
    return phi_out, info
