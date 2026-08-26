"""Prototype: a CONTINUOUS scheduler over the windowed engine's conflict graph.

Investigation only — nothing under ``dvfopt/`` is touched. Follow-up to
``benchmarks/parallel_windows_proto.py`` (branch ``proto-parallel-windows``),
which batched each round's windows into conflict-free COLOUR GROUPS and measured
1.63x wall / **1.9 effective concurrency, flat from 4 workers to 16**. A colour
group finishes in the time of its slowest member, so a group of 9 tiles idles 8
workers on the straggler. This script replaces the groups with the pattern
``dvfopt/core/slp/cluster_lp_2tri.py`` already uses for clusters: one persistent
pool, a ready set, and greedy admission of any work item that conflicts with
nothing in flight — so a worker that frees picks up the next admissible tile
immediately instead of waiting on a barrier.

Three barriers the colour-group design pays and this one does not:

1. **Intra-group** — the straggler in a colour group. Gone: admission is per-slot.
2. **Inter-sweep (Schwarz)** — the giant tiler's sweeps were strictly sequential.
   Here a tile carries its own sweep counter ``k`` and is re-queued as soon as it
   completes still holding folds, capped at the engine's ``giant_max_sweeps``. The
   engine's plateau guard survives as a per-giant stall check at *virtual sweep*
   boundaries (every ``n_tiles`` completions in that giant).
3. **Round-loop vs giant** — the prior prototype ran all small boxes, then the
   giants. Here a round's small windows and every giant's tiles enter ONE stream,
   so a giant's tail overlaps the next small window instead of draining first.

What is reused verbatim (no math is re-derived): ``find_windows``,
``build_subproblem`` (through ``_solve_window``), ``_solve_window``,
``_solve_giant_schwarz``'s tile geometry (via ``_fit_tile``, so ``giant_tile_fit``
is honoured), ``min_field`` / ``pixel_fold_mask``, ``_InnerOpts``, ``SliceReport``.
Local to this file: the halo crop (as in the prior prototype), the conflict
predicate, the scheduler, and an incrementally-maintained fold map.

Grow-on-failure stays SERIAL (drained in the parent after the stream), as in the
prior prototype: it is a second attempt that depends on the first's result.

RESULT (docs/superpowers/notes/window-scheduler-findings.md): it does NOT break the
plateau. At equal geometry the continuous scheduler reaches 1.79-1.85 effective
concurrency against the colour groups' 1.84-1.89, and at today's ``giant_tile_fit``
defaults only 1.47 / 1.05x wall, flat from 4 workers to 12 — the raw B0039 z16 giant
tiles into 12 boxes whose conflict graph admits at most 4 concurrent solves. The
plateau is the dependency structure, not the batching. Recommendation: don't promote.

Usage::

    python -u benchmarks/parallel_scheduler_proto.py --selftest
    python -u benchmarks/parallel_scheduler_proto.py --cases z16_full --workers 4,8,12
"""

import os

# Pin every thread pool to 1 BEFORE numpy / osqp / clarabel are imported. Spawn
# workers re-execute this module top-level, so this is also what pins them.
# RAYON_NUM_THREADS is NOT optional here: the engine's default qp_backend is
# 'hybrid', whose interior-point leg is Clarabel (Rust/rayon). Without the pin
# every worker spawns its own rayon pool and an N-way process pool oversubscribes
# the box N-fold — the measured collapse that motivated this list.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from dvfopt.constraints import SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.windowed._common import (  # noqa: E402
    SliceReport,
    _fit_tile,
    _InnerOpts,
    _solve_window,
    find_windows,
)
from dvfopt.core.windowed._locality import (  # noqa: E402
    _locality_of,
    min_field,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# The volume / crops live in the main checkout (gitignored, not copied into worktrees).
MAIN = Path(r"C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing")
VOL = MAIN / "data" / "dvfs" / "b0039" / "b0039_laplacian_deformation_field.npy"


# ---------------------------------------------------------------------------
# worker (module level => picklable by reference, spawn-safe on Windows)
# ---------------------------------------------------------------------------


def _init_worker():  # pragma: no cover - runs in subprocess
    """Re-pin threads (belt and braces: the parent's env is inherited by spawn,
    but an inherited env is not a guarantee) and warm the isqp path so the first
    real task does not pay the osqp/clarabel import + CPR-colouring build.

    NOT ``dvfopt.core._pool``'s ``_warmup_worker``: that one JIT-compiles the 3D
    tet numba kernels, which a 2D windowed run never touches (~5-10 s/worker of
    pure waste). A promotion should make that initializer family-aware.
    """
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[v] = "1"
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.3, (2, 16, 16))
    c = SimplexConstraint2DBilinear(shape=(16, 16))
    _solve_window(
        phi, c, (4, 12, 4, 12), 0.01, NoneObjective(), 50, 1, SliceReport(), allow_grow=False
    )


def _solve_box_worker(args):
    """Solve ONE window inside a worker. ``crop`` is the halo crop, ``box`` its
    crop-local free box. Returns (solved free region, feasible flag, in-worker s)."""
    ctype, crop, box, threshold, objective, maxiter, ring, margin_delta, inner, opts = args
    t = time.perf_counter()
    c = ctype(shape=crop.shape[1:])
    rep = SliceReport()
    ok = _solve_window(
        crop,
        c,
        box,
        threshold,
        objective,
        maxiter,
        ring,
        rep,
        margin_delta=margin_delta,
        allow_grow=False,  # grow cascades stay in the parent (serial, engine recursion)
        inner=inner,
        opts=opts,
    )
    fy0, fy1, fx0, fx1 = box
    return crop[:, fy0:fy1, fx0:fx1].copy(), bool(ok), time.perf_counter() - t


def _halo_crop(phi, box, ring):
    """``(crop, crop_local_box)`` with a ``ring + 1`` halo on interior sides.

    ``build_subproblem`` infers its ``borders`` tuple from ``py0 == 0`` etc against
    the array it is handed, so a crop of exactly the patch would look like an
    all-image-border patch and enforce edge rows evaluated with the wrong one-sided
    difference. One spare row/col outside the patch on every *interior* side (and
    none where the box already reached the image border) reproduces the global
    flags exactly. ``--selftest`` asserts the resulting solve is bit-equal to the
    serial engine's. (A promotion would pass ``borders`` into ``build_subproblem``.)
    """
    H, W = phi.shape[1:]
    fy0, fy1, fx0, fx1 = box
    cy0, cy1 = max(0, fy0 - ring - 1), min(H, fy1 + ring + 1)
    cx0, cx1 = max(0, fx0 - ring - 1), min(W, fx1 + ring + 1)
    crop = np.array(phi[:, cy0:cy1, cx0:cx1], dtype=np.float64, copy=True)
    return crop, (fy0 - cy0, fy1 - cy0, fx0 - cx0, fx1 - cx0)


# ---------------------------------------------------------------------------
# conflict graph + incremental fold map
# ---------------------------------------------------------------------------


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _dilate(box, r):
    return (box[0] - r, box[1] + r, box[2] - r, box[3] + r)


def _conflict(a, b, ring):
    """True if ``a``/``b`` cannot run concurrently: one WRITES (its free box) where
    the other READS (its patch = free box +/- ring)."""
    return _overlap(a, _dilate(b, ring)) or _overlap(b, _dilate(a, ring))


def _refresh_mf(mf, constraint, phi, box, ring, halo=2):
    """Recompute the cached fold map ``mf`` only where pasting ``box`` changed it.

    A free pixel influences constraint rows up to ``ring`` away, so the stale
    region is ``box`` dilated by ``ring``. It is recomputed from a crop dilated by
    a further ``halo`` so the crop's own edge artefacts (one-sided differences on
    an interior cut, the ``+inf`` last cell row/col) fall OUTSIDE the region that
    is written back. Where the crop edge IS the image edge, the crop evaluates
    exactly as the global field does, so nothing is lost.

    Agrees with a full recompute to ~1e-16, not bit-for-bit: the same elementwise
    algebra over a strided crop rounds differently from the contiguous whole. It
    only ever drives threshold comparisons (does this tile still hold a fold), and
    the map is re-seeded from a full recompute at every pass boundary, so the noise
    cannot accumulate. ``--selftest`` asserts the FOLD MASK matches exactly.
    """
    H, W = phi.shape[1:]
    a0, a1, b0, b1 = box
    r0, r1 = max(0, a0 - ring), min(H, a1 + ring)
    s0, s1 = max(0, b0 - ring), min(W, b1 + ring)
    c0, c1 = max(0, r0 - halo), min(H, r1 + halo)
    d0, d1 = max(0, s0 - halo), min(W, s1 + halo)
    sub = min_field(constraint, phi[:, c0:c1, d0:d1])
    mf[r0:r1, s0:s1] = sub[r0 - c0 : r1 - c0, s0 - d0 : s1 - d0]


def _giant_tiles(box, ring, H, W, opts):
    """The giant tiler's tile list — same geometry as ``_solve_giant_schwarz``
    (inline there; a promotion should factor it out), ``giant_tile_fit`` included."""
    fy0, fy1, fx0, fx1 = box
    tile = opts.giant_tile
    if opts.giant_tile_fit:
        tile = _fit_tile(fy1 - fy0, fx1 - fx0, tile)
    it0 = fy0 + (ring if fy0 > 0 else 0)  # inset interior edges; keep image borders
    it1 = fy1 - (ring if fy1 < H else 0)
    ix0 = fx0 + (ring if fx0 > 0 else 0)
    ix1 = fx1 - (ring if fx1 < W else 0)
    step = max(1, tile - (2 * ring + 2))
    return [
        (ty, min(ty + tile, it1), tx, min(tx + tile, ix1))
        for ty in range(it0, it1, step)
        for tx in range(ix0, ix1, step)
        if min(ty + tile, it1) > ty and min(tx + tile, ix1) > tx
    ]


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One schedulable solve. ``kind`` is bookkeeping only — admission is by box."""

    box: tuple
    kind: str  # 'window' | 'mop' | 'tile'
    gid: int = -1  # giant id (tiles only)
    k: int = 0  # Schwarz sweeps already done by this tile


@dataclass
class Giant:
    """Per-giant Schwarz state: the engine's sweep cap and plateau guard, kept
    per-tile instead of per-barrier."""

    box: tuple
    n_tiles: int
    done: int = 0
    prev_nf: int | None = None
    alive: bool = True


@dataclass
class Stats:
    total: float = 0.0
    stream_wall: float = 0.0  # wall inside the scheduler loop (the parallel part)
    worker_solve_s: float = 0.0  # summed in-worker (or in-process) solve seconds
    foldmap: float = 0.0  # min_field / pixel_fold_mask (full + incremental)
    find: float = 0.0  # find_windows + tiling + conflict-degree ordering
    grow: float = 0.0  # SERIAL grow-on-failure drain
    accounting: float = 0.0
    pool_start_s: float = 0.0
    n_solves: int = 0
    n_windows: int = 0
    n_tiles: int = 0
    n_grows: int = 0
    rounds: int = 0
    mop_sweeps: int = 0
    giant_sweeps: int = 0  # summed virtual sweeps over all giants
    idle_slot_s: float = 0.0  # worker-slots * seconds with nothing admissible
    max_inflight: int = 0


@dataclass
class Ctx:
    constraint: object
    objective: object
    threshold: float
    maxiter: int
    ring: int
    margin_delta: float
    inner: str
    opts: object
    n_workers: int
    ex: object = None
    mf: object = None  # cached (H, W) fold map
    giants: dict = field(default_factory=dict)
    st: Stats = field(default_factory=Stats)


def _payload(phi, item, ctx):
    return (
        type(ctx.constraint),
        *_halo_crop(phi, item.box, ctx.ring),
        ctx.threshold,
        ctx.objective,
        ctx.maxiter,
        ctx.ring,
        ctx.margin_delta,
        ctx.inner,
        ctx.opts,
    )


def _ready(item, ctx):
    """Is this item admissible work right now?

    A one-shot window always is. A giant tile is on its first sweep (mirrors the
    engine, which sweeps every tile once regardless), or still holds a fold in its
    free box and has sweeps left and its giant has not plateaued.
    """
    if item.kind != "tile":
        return True
    g = ctx.giants[item.gid]
    if not g.alive or item.k >= ctx.opts.giant_max_sweeps:
        return False
    if item.k == 0:
        return True
    y0, y1, x0, x1 = item.box
    return bool((ctx.mf[y0:y1, x0:x1] < ctx.threshold).any())


def _order(items, ctx):
    """Largest-first within most-conflicted-first.

    Both halves are the standard list-scheduling defence against a long tail: the
    high-degree items block the most successors so they should not be left for the
    end, and the big ones are the long jobs (LPT). O(n^2) degree scan — a round
    never has more than a few hundred boxes.

    ponytail: no interval index. Add one if a round ever emits thousands of boxes.
    """
    deg = [sum(_conflict(a.box, b.box, ctx.ring) for b in items if b is not a) for a in items]
    area = [(a.box[1] - a.box[0]) * (a.box[3] - a.box[2]) for a in items]
    return [it for _, _, it in sorted(zip(deg, area, items), key=lambda t: (-t[0], -t[1]))]


def _run_stream(phi, items, ctx, allow_grow=True):
    """Drain ``items`` through the continuous scheduler, then grow failures serially.

    Invariant (the engine's no-damage guarantee, preserved): two items run
    concurrently only if neither's free box meets the other's patch, so no worker
    reads a region another in-flight worker writes. Crops are taken at ADMISSION
    time, so an item admitted after a neighbour completed sees the updated field in
    its frozen ring — exactly the Gauss-Seidel coupling the serial engine has.
    """
    t_find = time.perf_counter()
    pending = _order(items, ctx)
    ctx.st.find += time.perf_counter() - t_find
    failed = []
    t_stream = time.perf_counter()

    if ctx.ex is None:  # serial: same order, same accounting, no pool
        while pending:
            nxt = next((i for i, it in enumerate(pending) if _ready(it, ctx)), None)
            if nxt is None:
                break
            it = pending.pop(nxt)
            t = time.perf_counter()
            out, ok, _ = _solve_box_worker(_payload(phi, it, ctx))
            ctx.st.worker_solve_s += time.perf_counter() - t
            _complete(phi, it, out, ok, ctx, pending, failed, allow_grow)
        ctx.st.stream_wall += time.perf_counter() - t_stream
        _drain_grows(phi, failed, ctx)
        return

    inflight = {}
    while pending or inflight:
        admitted = True
        while len(inflight) < ctx.n_workers and admitted:
            admitted = False
            for i, it in enumerate(pending):
                if not _ready(it, ctx):
                    continue
                if any(_conflict(it.box, o.box, ctx.ring) for o in inflight.values()):
                    continue
                pending.pop(i)
                inflight[ctx.ex.submit(_solve_box_worker, _payload(phi, it, ctx))] = it
                admitted = True
                break
        if not inflight:
            break  # nothing admissible and nothing running
        ctx.st.max_inflight = max(ctx.st.max_inflight, len(inflight))
        t_w = time.perf_counter()
        done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        # worker-slots idled while blocked: the direct cost of the conflict graph
        ctx.st.idle_slot_s += (ctx.n_workers - len(inflight)) * (time.perf_counter() - t_w)
        for fut in done:
            out, ok, wt = fut.result()
            it = inflight.pop(fut)
            ctx.st.worker_solve_s += wt
            _complete(phi, it, out, ok, ctx, pending, failed, allow_grow)
    ctx.st.stream_wall += time.perf_counter() - t_stream
    _drain_grows(phi, failed, ctx)


def _complete(phi, it, out, ok, ctx, pending, failed, allow_grow):
    """Paste a finished item back, refresh the fold map locally, and re-queue the
    tile if its giant still has work (the engine's next Schwarz sweep, per tile)."""
    y0, y1, x0, x1 = it.box
    phi[:, y0:y1, x0:x1] = out
    t = time.perf_counter()
    _refresh_mf(ctx.mf, ctx.constraint, phi, it.box, ctx.ring)
    ctx.st.foldmap += time.perf_counter() - t
    ctx.st.n_solves += 1
    ctx.st.n_tiles += it.kind == "tile"
    ctx.st.n_windows += it.kind != "tile"
    if it.kind != "tile":
        if not ok and allow_grow:
            failed.append(it.box)
        return
    g = ctx.giants[it.gid]
    g.done += 1
    gy0, gy1, gx0, gx1 = g.box
    nf = int((ctx.mf[gy0:gy1, gx0:gx1] < ctx.threshold).sum())
    if nf == 0:
        g.alive = False  # region cleared
    elif g.done % g.n_tiles == 0:  # virtual sweep boundary: the engine's plateau guard
        ctx.st.giant_sweeps += 1
        if g.prev_nf is not None and nf >= g.prev_nf:
            g.alive = False  # no progress -> geometric floor, stop this giant
        g.prev_nf = nf
    it.k += 1
    if _ready(it, ctx):
        pending.append(it)  # re-sweep; already ordered ahead of nothing, tail is fine


def _drain_grows(phi, failed, ctx):
    """Grow-on-failure, SERIAL in the parent — the engine's own second attempt
    (``_solve_window(..., _grow=1)`` recursion), which depends on the first."""
    if not failed:
        return
    H, W = phi.shape[1:]
    t = time.perf_counter()
    for fy0, fy1, fx0, fx1 in failed:
        gb = (max(0, fy0 - 4), min(H, fy1 + 4), max(0, fx0 - 4), min(W, fx1 + 4))
        if gb == (fy0, fy1, fx0, fx1):
            continue
        _solve_window(
            phi,
            ctx.constraint,
            gb,
            ctx.threshold,
            ctx.objective,
            ctx.maxiter,
            ctx.ring,
            SliceReport(),
            _grow=1,
            margin_delta=ctx.margin_delta,
            inner=ctx.inner,
            opts=ctx.opts,
        )
        _refresh_mf(ctx.mf, ctx.constraint, phi, gb, ctx.ring)
        ctx.st.n_grows += 1
    ctx.st.grow += time.perf_counter() - t


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _items_for(boxes, ctx, phi, max_area, kind):
    """Turn a pass's boxes into stream items: small boxes stay whole, over-cap ones
    explode into their giant's tiles (all in the SAME stream, so a giant's tail
    overlaps the next window instead of draining first)."""
    H, W = phi.shape[1:]
    items = []
    for box in boxes:
        if (box[1] - box[0]) * (box[3] - box[2]) <= max_area:
            items.append(Item(box, kind))
            continue
        gid = len(ctx.giants)
        tiles = _giant_tiles(box, ctx.ring, H, W, ctx.opts)
        ctx.giants[gid] = Giant(box, max(1, len(tiles)))
        items.extend(Item(t, "tile", gid) for t in tiles)
    return items


def scheduled_windowed_correct(
    phi_in,
    *,
    constraint,
    objective=None,
    threshold,
    margin=3,
    maxiter=400,
    max_rounds=8,
    margin_delta=1e-3,
    max_window_area=3000,
    mop_margin=25,
    inner="isqp",
    n_workers=1,
    executor=None,
    **opt_kw,
):
    """``windowed_correct``'s structure with every pass driven by the continuous
    scheduler. Returns ``(phi_out, SliceReport, Stats)``."""
    loc = _locality_of(constraint)
    ring = loc.ring
    margin = max(margin, ring)
    ctx = Ctx(
        constraint,
        NoneObjective() if objective is None else objective,
        threshold,
        maxiter,
        ring,
        margin_delta,
        inner,
        _InnerOpts(**opt_kw),
        n_workers,
    )
    phi = np.array(phi_in, dtype=np.float64, copy=True)
    H, W = phi.shape[1:]
    rep = SliceReport()
    t0 = time.perf_counter()

    ctx.ex = executor
    own_pool = ctx.ex is None and n_workers > 1
    if own_pool:
        t = time.perf_counter()
        ctx.ex = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            mp_context=multiprocessing.get_context("spawn"),
        )
        list(ctx.ex.map(int, range(n_workers)))  # force spawn+warmup out of the timed region
        ctx.st.pool_start_s = time.perf_counter() - t
        t0 = time.perf_counter()

    try:
        t = time.perf_counter()
        ctx.mf = min_field(constraint, phi)
        ctx.st.foldmap += time.perf_counter() - t
        orig_fold = ctx.mf < threshold
        rep.folds_before = int(orig_fold.sum())
        rep.min_before = float(ctx.mf.min())
        touched = np.zeros((H, W), bool)

        prev_nfold = None
        for _rnd in range(max_rounds):
            t = time.perf_counter()
            ctx.mf = min_field(constraint, phi)  # re-seed: no incremental noise across passes
            ctx.st.foldmap += time.perf_counter() - t
            nfold = int((ctx.mf < threshold).sum())
            if nfold == 0 or (prev_nfold is not None and nfold >= prev_nfold):
                break
            prev_nfold = nfold
            rep.rounds = ctx.st.rounds = ctx.st.rounds + 1
            t = time.perf_counter()
            boxes = find_windows(ctx.mf < threshold, margin, ring)
            ctx.st.find += time.perf_counter() - t
            for fy0, fy1, fx0, fx1 in boxes:
                # touched = the ENFORCED footprint, exactly as the engine marks it,
                # so damage=0 stays by construction rather than by luck.
                touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
                if (fy1 - fy0) * (fx1 - fx0) > max_window_area:
                    rep.giant_regions += 1
                    rep.giant_boxes.append((fy0, fy1, fx0, fx1))
            _run_stream(phi, _items_for(boxes, ctx, phi, max_window_area, "window"), ctx)

        if mop_margin > 0:
            whole_cap = 4 * max_window_area
            for _sweep in range(3):  # _mop_pass's max_sweeps
                t = time.perf_counter()
                ctx.mf = min_field(constraint, phi)
                ctx.st.foldmap += time.perf_counter() - t
                n = int((ctx.mf < threshold).sum())
                if n == 0:
                    break
                ctx.st.mop_sweeps += 1
                t = time.perf_counter()
                lbl, _ = ndimage.label(ctx.mf < threshold)
                boxes = []
                for sy, sx in ndimage.find_objects(lbl):
                    fy0, fy1 = max(0, sy.start - mop_margin), min(H, sy.stop + mop_margin)
                    fx0, fx1 = max(0, sx.start - mop_margin), min(W, sx.stop + mop_margin)
                    touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
                    rep.mop_windows += 1
                    boxes.append((fy0, fy1, fx0, fx1))
                ctx.st.find += time.perf_counter() - t
                _run_stream(phi, _items_for(boxes, ctx, phi, whole_cap, "mop"), ctx)
                if int((min_field(constraint, phi) < threshold).sum()) >= n:
                    break  # no progress -> genuine local floor
            rep.mop_cleared = 0
    finally:
        if own_pool:
            ctx.ex.shutdown(wait=True)

    t = time.perf_counter()
    jf = min_field(constraint, phi)
    after = jf < threshold
    rep.folds_after = int(after.sum())
    rep.min_after = float(jf.min())
    rep.damage = int((after & ~orig_fold & ~touched).sum())
    rep.residual_in_window = int((after & touched).sum())
    rep.n_windows = ctx.st.n_solves
    ctx.st.accounting = time.perf_counter() - t
    rep.time_s = ctx.st.total = time.perf_counter() - t0
    return phi, rep, ctx.st


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def simplex_folds(phi, thr=0.01):
    """The task's fold metric: 2-triangle areas below ``thr``."""
    return int((np.minimum(*_triangle_areas_2d(phi[0], phi[1])) < thr).sum())


def load_case(name):
    if name == "z16_full":
        return np.ascontiguousarray(np.load(VOL, mmap_mode="r")[1:, 16], dtype=np.float64)
    for p in (REPO / "benchmarks/output/testcases", MAIN / "benchmarks/output/testcases"):
        if (p / f"{name}.npy").exists():
            return np.array(np.load(p / f"{name}.npy"), dtype=np.float64)
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="z16_full")
    ap.add_argument("--workers", default="4,8,12")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--reference", action="store_true", help="also time the shipped engine")
    ap.add_argument("--giant-tile", type=int, default=64, help="engine default 64")
    ap.add_argument("--giant-tile-fit", type=int, default=1, help="0 disables #77's geometry fit")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    _selftest()  # also the warm-up: pays import / osqp / colouring cost before timing
    rows = []
    for case in args.cases.split(","):
        phi = load_case(case)
        c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
        kw = dict(constraint=c, objective=NoneObjective(), threshold=args.threshold,
                  maxiter=args.maxiter, giant_tile=args.giant_tile,
                  giant_tile_fit=bool(args.giant_tile_fit))  # fmt: skip
        raw_folds = simplex_folds(phi)
        print(f"\n=== {case}  shape={phi.shape}  simplex folds before={raw_folds} ===", flush=True)

        if args.reference:
            from dvfopt.core.windowed import windowed_correct

            t = time.perf_counter()
            ref, rrep = windowed_correct(phi, **kw)
            dt = time.perf_counter() - t
            l2 = float(np.linalg.norm((ref - phi).ravel()))
            print(
                f"  engine serial     wall={dt:8.2f}s  folds={simplex_folds(ref):4d}"
                f"  rep_folds={rrep.folds_after:4d}  damage={rrep.damage}"
                f"  solves={rrep.n_windows:4d}  rounds={rrep.rounds}  L2={l2:.1f}",
                flush=True,
            )
            rows.append(dict(case=case, mode="engine", workers=0, wall=dt,
                             folds=simplex_folds(ref), rep_folds=rrep.folds_after,
                             damage=rrep.damage, solves=rrep.n_windows, rounds=rrep.rounds,
                             l2=l2))  # fmt: skip

        for w in [int(x) for x in args.workers.split(",")]:
            out, rep, st = scheduled_windowed_correct(phi, n_workers=w, **kw)
            l2 = float(np.linalg.norm((out - phi).ravel()))
            eff = st.worker_solve_s / st.stream_wall if st.stream_wall else 0.0
            print(
                f"  sched workers={w:<3d} wall={st.total:8.2f}s  folds={simplex_folds(out):4d}"
                f"  rep_folds={rep.folds_after:4d}  damage={rep.damage}"
                f"  solves={st.n_solves:4d}  rounds={st.rounds}  L2={l2:.1f}",
                flush=True,
            )
            print(
                f"      eff_conc={eff:.2f}  stream={st.stream_wall:.1f}s"
                f"  worker_solve={st.worker_solve_s:.1f}s  serial_rem="
                f"{st.total - st.stream_wall:.1f}s (grow={st.grow:.1f} foldmap={st.foldmap:.1f}"
                f" find={st.find:.1f} acct={st.accounting:.1f})  idle_slots={st.idle_slot_s:.1f}s"
                f"  max_inflight={st.max_inflight}  tiles={st.n_tiles} win={st.n_windows}"
                f"  giant_sweeps={st.giant_sweeps} mop_sweeps={st.mop_sweeps}"
                f"  grows={st.n_grows}  pool_start={st.pool_start_s:.2f}s",
                flush=True,
            )
            rows.append(dict(case=case, mode="sched", workers=w, wall=st.total,
                             folds=simplex_folds(out), rep_folds=rep.folds_after,
                             damage=rep.damage, solves=st.n_solves, rounds=st.rounds,
                             l2=l2, eff_conc=eff, stats=asdict(st)))  # fmt: skip

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")


def _selftest():
    """Runnable checks for the three pieces of local logic: the halo crop, the
    incremental fold map, and the conflict predicate / scheduler bookkeeping."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.35, (2, 24, 26))
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    ring = _locality_of(c).ring
    obj, opts = NoneObjective(), _InnerOpts()

    for box in [(4, 12, 5, 14), (0, 7, 0, 9), (16, 24, 15, 26)]:  # interior + both borders
        a = phi.copy()
        _solve_window(a, c, box, 0.01, obj, 200, ring, SliceReport(), allow_grow=False, opts=opts)
        crop, lb = _halo_crop(phi, box, ring)
        got, _ok, _t = _solve_box_worker((type(c), crop, lb, 0.01, obj, 200, ring, 1e-3, "isqp", opts))  # fmt: skip
        assert np.array_equal(got, a[:, box[0] : box[1], box[2] : box[3]]), f"halo {box}"
        # incremental fold map: same fold mask as a full recompute, values to 1e-12
        mf, full = min_field(c, phi), min_field(c, a)
        _refresh_mf(mf, c, a, box, ring)
        assert np.array_equal(mf < 0.01, full < 0.01), f"mf mask {box}"
        fin = np.isfinite(full)
        assert np.allclose(mf[fin], full[fin], rtol=0, atol=1e-12), f"mf values {box}"

    assert _conflict((0, 10, 0, 10), (10, 20, 0, 10), 1)  # touching => reads overlap writes
    assert not _conflict((0, 10, 0, 10), (12, 20, 0, 10), 1)
    # scheduler end-to-end (serial path) reaches the same feasibility as the engine
    from dvfopt.core.windowed import windowed_correct

    hard = rng.normal(0, 0.6, (2, 40, 40))
    kw = dict(constraint=SimplexConstraint2DBilinear(shape=(40, 40)), objective=obj,
              threshold=0.01, maxiter=200)  # fmt: skip
    e, erep = windowed_correct(hard, **kw)
    s, srep, _st = scheduled_windowed_correct(hard, n_workers=1, **kw)
    assert srep.damage == 0 and erep.damage == 0, (srep.damage, erep.damage)
    assert srep.folds_after <= erep.folds_after + 2, (srep.folds_after, erep.folds_after)
    print(f"selftest ok (engine folds={erep.folds_after}, sched folds={srep.folds_after})")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
