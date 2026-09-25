"""Read a Laplacian-interpolated DVF's pins back out of the DVF itself.

A field built by Laplacian interpolation is discrete-harmonic everywhere except at its
Dirichlet pins, so the 6- (or 4-) neighbour graph Laplacian ``A u`` — the stencil of
:mod:`dvfopt.laplacian.utils` applied to the field instead of solved against — is ~0 at
every free voxel and spikes at the pins. Nothing here reads a correspondence file: the pins,
their displacements and their pairwise consistency all come from the field.

Measured on B0039 z[0,40) (2026-09-22, ``notebooks/experiments/source_space_pin_probe.ipynb``):
on a field solved to ``rtol`` 1e-4 the pins are 1.5 decades above the residual and ``tau=0.1``
recovers 8,379 of 8,381 with no false positives; the production field (``rtol`` 1e-2) is NOT
readable this way (8 % precision) — tighten the solve first.

Threshold rule (:func:`auto_tau`, measured on the 7-brain cohort 2026-09-25):
``tau = max(0.7, 10 * p98(source_strength(phi)))``. The non-pin residual scales with the
prescribed ``|d|`` (p99.9 0.12 px on B0039, 0.78 px on B0304), so a fixed tau over-detects on
large-displacement brains (B0304 at 0.7: 309,833 "pins" for 210,625 landmarks, precision 0.675).
p98 sits below any plausible pin fraction (pins are ~0.5 % of voxels); the rule gives 0.7 on
B0039 / B0213 and 2.35 on B0304 (precision 0.977).

Caveat: the pairwise drop fixes CONTRADICTORY pins, not per-slice landmark offsets. B0304's
sections each carry a different rigid offset (per-slice landmark medians jump 10-30 px between
adjacent slices), so every inter-layer pair contradicts over a 1-px z gap and no in-plane edit
reconciles them; the chain (:mod:`dvfopt.pipeline_pins`) does not certify that brain.

Diagnosis only: nothing here edits the field (the re-fill lives in :mod:`dvfopt.dvf.refill`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def source_map(u: np.ndarray) -> np.ndarray:
    """``A u`` for the full 2·ndim-neighbour graph Laplacian (degree-corrected at the faces):
    ``deg_i * u_i - sum of present neighbours``. Any rank; matches ``laplacianA3D`` with no
    Dirichlet rows and unit spacing."""
    f = np.zeros_like(u)
    for ax in range(u.ndim):
        d = np.diff(u, axis=ax)
        lo = [slice(None)] * u.ndim
        hi = [slice(None)] * u.ndim
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        f[tuple(lo)] -= d
        f[tuple(hi)] += d
    return f


def source_strength(phi: np.ndarray) -> np.ndarray:
    """Per-voxel ``||A u||`` over the field's channels. ``phi`` is ``(C, *shape)``."""
    return np.sqrt(sum(source_map(phi[c]) ** 2 for c in range(phi.shape[0])))


def auto_tau(phi: np.ndarray) -> float:
    """Self-calibrating pin threshold: ``max(0.7, 10 * p98(source_strength(phi)))`` (see the
    module docstring for the measurement behind it)."""
    return max(0.7, 10.0 * float(np.percentile(source_strength(phi), 98)))


def detect_pins(phi: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Boolean mask of pin voxels: ``source_strength(phi) > tau`` (``tau`` in displacement
    units; 0.1 sits in the measured gap for a tightly solved field)."""
    return source_strength(phi) > tau


def inconsistent_pins(
    phi: np.ndarray, pins: np.ndarray, c: float = 1.0, radius: float = 60.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise Lipschitz test on the detected pins, from the field's own displacements.

    Two pins ``p, q`` within ``radius`` are inconsistent when ``|d(p) - d(q)| > c * |p - q|``
    (they would have to cross). Returns ``(coords, drop, pairs)``: pin voxel coordinates
    ``(n, ndim)``, a boolean ``drop`` marking a greedy max-degree vertex cover of the violation
    graph (a small set whose removal leaves no violating pair — an upper bound on the true
    minimum), and the violating pairs ``(m, 2)`` as indices into ``coords``.
    """
    coords = np.argwhere(pins)
    d = phi[(slice(None), *coords.T)].T  # (n, C) displacement at each pin
    pts = coords.astype(np.float64)
    bad = violating_pairs_pruned(pts, d, c, radius)
    return coords, greedy_cover(len(coords), bad), bad


def violating_pairs(
    pts: np.ndarray, d: np.ndarray, c: float, radius: float, n_workers: int = 1
) -> np.ndarray:
    """All pairs within ``radius`` with ``|d_p - d_q| > c |p - q|``, as ``(m, 2)`` int32 with ``p < q``.
    Chunked along the first coordinate so the (much larger) set of ALL near pairs is never held at
    once: ~230M near pairs on a 425k-pin volume vs 11M violating ones. ``n_workers > 1`` runs the
    chunks on a process pool (the caller guards ``__main__`` on spawn platforms)."""
    order = np.argsort(pts[:, 0], kind="stable")
    pts, d = pts[order], d[order]
    step = max(1, len(pts) // 64)
    jobs = [(pts, d, order, s, step, c, radius) for s in range(0, len(pts), step)]
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(n_workers) as ex:
            out = list(ex.map(_violating_chunk, jobs))
    else:
        out = [_violating_chunk(j) for j in jobs]
    return np.vstack(out) if out else np.zeros((0, 2), np.int32)


def _violating_chunk(job):
    pts, d, order, s, step, c, radius = job
    lo, hi = pts[s, 0] - radius, pts[min(s + step, len(pts)) - 1, 0] + radius
    ctx = np.arange(np.searchsorted(pts[:, 0], lo), np.searchsorted(pts[:, 0], hi, side="right"))
    ii, jj = _near(pts[s : s + step], pts[ctx], radius)
    a, b = order[s + ii], order[ctx[jj]]
    sep = np.linalg.norm(pts[s + ii] - pts[ctx[jj]], axis=1)
    dd = np.linalg.norm(d[s + ii] - d[ctx[jj]], axis=1)
    keep = (dd > c * sep) & (sep > 0) & (a < b)
    return np.column_stack([a[keep], b[keep]]).astype(np.int32)


def crossing_pairs(pts: np.ndarray, d: np.ndarray, c: float, radius: float) -> np.ndarray:
    """MEASURED AND REJECTED — kept for the record, not used by the chain.

    Pairs within ``radius`` whose moved images collapse or cross ALONG their own separation:
    ``(dx + dd) . dx < c |dx|^2`` (post-move spacing along the line under a fraction ``c`` of the
    original; ``c = 0`` = strict order reversal). A subset of :func:`violating_pairs` that is blind
    to pure rotation / expansion, which the norm test flags as violations at |grad d| > 1. Exact
    per-pin radius bound: such a pair needs ``|dd| >= (1 - c) |dx|``.

    On the B0039 slab it is WORSE than the norm test (8,082 re-fill folds at ``c=0``, 5,593 at
    0.5, vs 1,272): the norm test's extra pairs are shear differences, and shear folds the
    interpolant too."""
    mag = np.linalg.norm(d, axis=1)
    order = np.argsort(-mag, kind="stable")
    pts_o, d_o, mag_o = pts[order], d[order], mag[order]
    tree = cKDTree(pts_o)
    r = np.minimum(radius, 2.0 * mag_o / max(1.0 - c, 1e-6) + 1e-9)
    out = []
    step = 20000
    for s in range(0, len(pts_o), step):
        lists = tree.query_ball_point(pts_o[s : s + step], r[s : s + step])
        ii = np.repeat(np.arange(s, s + len(lists)), [len(x) for x in lists])
        jj = np.fromiter((j for x in lists for j in x), dtype=np.int64, count=len(ii))
        keep = jj > ii
        ii, jj = ii[keep], jj[keep]
        dx = pts_o[jj] - pts_o[ii]
        dd = np.zeros_like(dx)
        dd[:, -d.shape[1] :] = d_o[jj] - d_o[ii]  # d has the trailing (in-plane) components
        sep2 = (dx * dx).sum(axis=1)
        keep = (((dx + dd) * dx).sum(axis=1) < c * sep2) & (sep2 > 0)
        a, b = order[ii[keep]], order[jj[keep]]
        out.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]).astype(np.int32))
    return np.vstack(out) if out else np.zeros((0, 2), np.int32)


def violating_pairs_pruned(pts: np.ndarray, d: np.ndarray, c: float, radius: float) -> np.ndarray:
    """Same pairs as :func:`violating_pairs`, found with a per-pin radius. A violating pair has
    ``c |p - q| < |d_p - d_q| <= |d_p| + |d_q| <= 2 max(|d_p|, |d_q|)``, so it is found from its
    larger-|d| end with radius ``min(radius, 2 max|d| / c)`` — tiny for the ~1 px pins that are
    most of a real volume. Exact, not approximate."""
    if not c > 0:
        raise ValueError(
            f"c must be > 0 for the pruned search (got {c}); use violating_pairs for c <= 0"
        )
    mag = np.linalg.norm(d, axis=1)
    order = np.argsort(-mag, kind="stable")  # larger |d| first
    pts_o, d_o, mag_o = pts[order], d[order], mag[order]
    tree = cKDTree(pts_o)
    r = np.minimum(radius, 2.0 * mag_o / c + 1e-9)
    out = []
    step = 20000
    for s in range(0, len(pts_o), step):
        lists = tree.query_ball_point(pts_o[s : s + step], r[s : s + step])
        ii = np.repeat(np.arange(s, s + len(lists)), [len(x) for x in lists])
        jj = np.fromiter((j for x in lists for j in x), dtype=np.int64, count=len(ii))
        keep = jj > ii  # each pair once, from its larger-|d| end (ties by order)
        ii, jj = ii[keep], jj[keep]
        sep = np.linalg.norm(pts_o[ii] - pts_o[jj], axis=1)
        dd = np.linalg.norm(d_o[ii] - d_o[jj], axis=1)
        keep = (dd > c * sep) & (sep > 0)
        a, b = order[ii[keep]], order[jj[keep]]
        out.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]).astype(np.int32))
    return np.vstack(out) if out else np.zeros((0, 2), np.int32)


def _near(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    m = cKDTree(a).sparse_distance_matrix(cKDTree(b), radius, output_type="coo_matrix")
    return m.row, m.col


def greedy_cover(n: int, pairs: np.ndarray) -> np.ndarray:
    """Greedy max-degree vertex cover of the graph on ``n`` nodes with edges ``pairs``: repeatedly
    drop the node with the most remaining edges until none remain. Bucket queue, O(m + n)."""
    drop = np.zeros(n, bool)
    if len(pairs) == 0:
        return drop
    a, b = pairs[:, 0], pairs[:, 1]
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    o = np.argsort(src, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
    _cover_loop(deg.astype(np.int64), indptr.astype(np.int64), dst[o].astype(np.int64), drop)
    return drop


def _cover_loop(
    deg, indptr, nbr, drop
):  # ponytail: numba if importable, else the same loop in Python
    try:
        from numba import njit
    except ImportError:  # pragma: no cover
        njit = lambda f: f  # noqa: E731
    _cover_loop_impl = njit(_cover_loop_py)
    _cover_loop_impl(deg, indptr, nbr, drop)


def _cover_loop_py(deg, indptr, nbr, drop):
    n = len(deg)
    maxd = int(deg.max())
    # bucket lists by degree: head[d] -> node, nxt/prv links
    head = np.full(maxd + 1, -1, np.int64)
    nxt = np.full(n, -1, np.int64)
    prv = np.full(n, -1, np.int64)
    for v in range(n):
        if deg[v] > 0:
            nxt[v] = head[deg[v]]
            if head[deg[v]] >= 0:
                prv[head[deg[v]]] = v
            head[deg[v]] = v
    cur = maxd
    while cur > 0:
        v = head[cur]
        if v < 0:
            cur -= 1
            continue
        # unlink v, drop it
        head[cur] = nxt[v]
        if nxt[v] >= 0:
            prv[nxt[v]] = -1
        drop[v] = True
        deg[v] = 0
        for k in range(indptr[v], indptr[v + 1]):
            u = nbr[k]
            if deg[u] == 0:
                continue
            # unlink u from its bucket
            du = deg[u]
            if prv[u] >= 0:
                nxt[prv[u]] = nxt[u]
            else:
                head[du] = nxt[u]
            if nxt[u] >= 0:
                prv[nxt[u]] = prv[u]
            deg[u] = du - 1
            nxt[u] = -1
            prv[u] = -1
            if deg[u] > 0:
                nxt[u] = head[deg[u]]
                if head[deg[u]] >= 0:
                    prv[head[deg[u]]] = u
                head[deg[u]] = u
