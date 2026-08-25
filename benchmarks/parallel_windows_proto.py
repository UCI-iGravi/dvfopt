"""Prototype: solve the windowed engine's per-round windows in PARALLEL processes.

Investigation only — nothing under ``dvfopt/`` is touched. This script re-creates
:func:`dvfopt.core.windowed._common.windowed_correct`'s round loop and hands each
round's *independent* windows to a ``ProcessPoolExecutor``, to measure how much
wall clock the ~90%-of-runtime OSQP solves can give back.

What is reused verbatim (no math is re-derived here): ``find_windows``,
``build_subproblem`` (through ``_solve_window``), ``_solve_window`` (inner solve +
no-TR fallback + paste-back + ``WindowRec``), ``min_field`` / ``pixel_fold_mask``,
``_InnerOpts``, ``SliceReport``. Only three things are local to this file:

1. :func:`_halo_crop` — the worker gets a small crop, not the whole field. A crop
   of exactly the patch would make ``build_subproblem`` believe every patch edge is
   an image border (it infers ``borders`` from ``py0 == 0`` etc. against the array
   it is handed), which would enforce rows evaluated with the wrong one-sided
   difference. Cropping with a ``ring + 1`` halo on interior sides reproduces the
   global border flags exactly, so the sub-problem the worker builds is identical
   to the one the serial engine builds. (A promotion would instead pass ``borders``
   into ``build_subproblem``.)
2. :func:`_conflict` / :func:`_groups` — window independence. A window READS its
   patch (free box +/- ring) and WRITES its free box, so two windows may run
   concurrently iff neither's free box meets the other's patch. Conflicting boxes
   are split into successive groups (greedy graph colouring).
3. Parallel copies of the giant tiler and the mop sweep: their tile/cluster
   geometry lives inline in ``_common.py``, so it is repeated here rather than
   refactored (see the findings note: promotion should factor ``_giant_tiles``).

Grow-on-failure stays SERIAL: a batch of first attempts runs in parallel, then any
window the inner declared infeasible is grown and retried in-process through the
engine's own recursion (``_solve_window(..., _grow=1)``), i.e. exactly the engine's
second attempt.

``--workers 1`` short-circuits the pool entirely and calls ``_solve_window`` in the
engine's own order (giants inline), which is byte-identical to ``windowed_correct``
— the property a promoted ``n_workers`` knob must keep.

Usage::

    python -u benchmarks/parallel_windows_proto.py --workers 1,4,8,16
    python -u benchmarks/parallel_windows_proto.py --cases z16_full --workers 1,8
"""

import os

# Pin BLAS/OMP to one thread per process BEFORE numpy is imported. Spawn workers
# re-execute this module top-level, so this is what silences their thread pools too
# (an 8-way pool of 24-thread OpenBLAS oversubscribes a 24-core box ~8x).
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from dvfopt.constraints import SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.windowed._common import (  # noqa: E402
    SliceReport,
    _InnerOpts,
    _solve_window,
    find_windows,
)
from dvfopt.core.windowed._locality import (  # noqa: E402
    _locality_of,
    min_field,
    pixel_fold_mask,
)
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CASE_DIR = REPO / "benchmarks" / "output" / "testcases"
# The crops + volume live in the main checkout (gitignored, not copied into worktrees).
MAIN = Path(r"C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing")
VOL = MAIN / "data" / "dvfs" / "b0039" / "b0039_laplacian_deformation_field.npy"


# ---------------------------------------------------------------------------
# worker (module level => picklable by reference, spawn-safe)
# ---------------------------------------------------------------------------


def _solve_box_worker(args):
    """Solve ONE window inside a worker process.

    ``crop`` is the halo crop (see :func:`_halo_crop`) and ``box`` its crop-local
    free box. Returns the solved free region, the inner's feasibility flag, the
    engine's own :class:`WindowRec`, and the in-worker solve time.
    """
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
    # _solve_window pastes back only free pixels, and the free mask IS the free box
    # clipped into the patch — so the free box slice is exactly what changed.
    return crop[:, fy0:fy1, fx0:fx1].copy(), bool(ok), rep.windows[0], time.perf_counter() - t


def _halo_crop(phi, box, ring):
    """``(crop, crop_local_box)`` with a ``ring + 1`` halo on interior sides.

    The halo is what makes ``build_subproblem`` in the worker infer the same
    ``borders`` tuple as the serial engine: an interior side keeps one spare row/col
    outside the patch (so ``py0 != 0``), while a side that reached the image border
    gets no halo (so ``py0 == 0``, as globally).
    """
    H, W = phi.shape[1:]
    fy0, fy1, fx0, fx1 = box
    cy0, cy1 = max(0, fy0 - ring - 1), min(H, fy1 + ring + 1)
    cx0, cx1 = max(0, fx0 - ring - 1), min(W, fx1 + ring + 1)
    crop = np.array(phi[:, cy0:cy1, cx0:cx1], dtype=np.float64, copy=True)
    return crop, (fy0 - cy0, fy1 - cy0, fx0 - cx0, fx1 - cx0)


# ---------------------------------------------------------------------------
# independence
# ---------------------------------------------------------------------------


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _dilate(box, ring):
    return (box[0] - ring, box[1] + ring, box[2] - ring, box[3] + ring)


def _conflict(a, b, ring):
    """True if boxes ``a``/``b`` cannot run concurrently: one writes (its free box)
    where the other reads (its patch = free box +/- ring)."""
    return _overlap(a, _dilate(b, ring)) or _overlap(b, _dilate(a, ring))


def _groups(boxes, ring):
    """Split boxes into successive conflict-free groups (greedy colouring).

    ponytail: O(n^2) pairwise scan — a grid/interval index only pays off past a few
    thousand boxes per round, which no measured slice comes close to.
    """
    groups = []
    for b in boxes:
        for g in groups:
            if not any(_conflict(b, o, ring) for o in g):
                g.append(b)
                break
        else:
            groups.append([b])
    return groups


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


@dataclass
class Phases:
    """Wall-clock split of one run (Amdahl input)."""

    foldmap: float = 0.0  # min_field / pixel_fold_mask recomputation
    find: float = 0.0  # find_windows (dilate + label + bboxes)
    grouping: float = 0.0  # conflict colouring
    solve_round: float = 0.0  # round-loop window solves (batch wall in parallel mode)
    solve_giant: float = 0.0  # giant-region tiler
    solve_mop: float = 0.0  # terminal mop pass
    grow: float = 0.0  # serial grow-on-failure retries
    accounting: float = 0.0  # final damage / residual bookkeeping
    total: float = 0.0
    # parallel-efficiency instrumentation
    worker_solve_s: float = 0.0  # summed in-worker solve time (the parallelisable work)
    batch_wall_s: float = 0.0  # summed batch wall time in the parent
    n_batches: int = 0
    n_dispatched: int = 0
    pickle_bytes: int = 0
    pool_start_s: float = 0.0


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
    rep: SliceReport = field(default_factory=SliceReport)
    ph: Phases = field(default_factory=Phases)


def _run_boxes(phi, boxes, ctx, ex, allow_grow=True):
    """Solve ``boxes`` (parallel when ``ex`` is not None), then grow failures serially."""
    if ex is None:  # serial: the engine's own call, grows inline => byte-identical
        for b in boxes:
            _solve_window(
                phi,
                ctx.constraint,
                b,
                ctx.threshold,
                ctx.objective,
                ctx.maxiter,
                ctx.ring,
                ctx.rep,
                margin_delta=ctx.margin_delta,
                allow_grow=allow_grow,
                inner=ctx.inner,
                opts=ctx.opts,
            )
        return

    H, W = phi.shape[1:]
    t_g = time.perf_counter()
    groups = _groups(boxes, ctx.ring)
    ctx.ph.grouping += time.perf_counter() - t_g
    failed = []
    for g in groups:
        payload = [
            (
                type(ctx.constraint),
                *_halo_crop(phi, b, ctx.ring),
                ctx.threshold,
                ctx.objective,
                ctx.maxiter,
                ctx.ring,
                ctx.margin_delta,
                ctx.inner,
                ctx.opts,
            )
            for b in g
        ]
        if ctx.ph.pickle_bytes == 0 and payload:
            ctx.ph.pickle_bytes = len(pickle.dumps(payload[0], protocol=pickle.HIGHEST_PROTOCOL))
        t_b = time.perf_counter()
        results = list(ex.map(_solve_box_worker, payload, chunksize=1))
        ctx.ph.batch_wall_s += time.perf_counter() - t_b
        ctx.ph.n_batches += 1
        ctx.ph.n_dispatched += len(g)
        for b, (free_out, ok, rec, wt) in zip(g, results):
            phi[:, b[0] : b[1], b[2] : b[3]] = free_out
            ctx.rep.windows.append(rec)
            ctx.ph.worker_solve_s += wt
            if not ok and allow_grow:
                failed.append(b)

    t_gr = time.perf_counter()
    for fy0, fy1, fx0, fx1 in failed:  # engine's second attempt, in-process
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
            ctx.rep,
            _grow=1,
            margin_delta=ctx.margin_delta,
            inner=ctx.inner,
            opts=ctx.opts,
        )
    ctx.ph.grow += time.perf_counter() - t_gr


def _giant_tiles(box, ring, H, W, tile=32):
    """Overlapping tiles of a giant region — copied from ``_solve_giant_schwarz``
    (the geometry is inline there; promotion should factor it out)."""
    fy0, fy1, fx0, fx1 = box
    it0 = fy0 + (ring if fy0 > 0 else 0)
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


def _giant(phi, box, ctx, ex, max_sweeps=8):
    H, W = phi.shape[1:]
    tiles = _giant_tiles(box, ctx.ring, H, W)
    fy0, fy1, fx0, fx1 = box
    prev = None
    for _sweep in range(max_sweeps):
        _run_boxes(phi, tiles, ctx, ex, allow_grow=False)
        t = time.perf_counter()
        nf = int((min_field(ctx.constraint, phi)[fy0:fy1, fx0:fx1] < ctx.threshold).sum())
        ctx.ph.foldmap += time.perf_counter() - t
        if nf == 0 or (prev is not None and nf >= prev):
            return
        prev = nf


def _run_mixed(phi, boxes, giants, ctx, ex):
    """Solve one pass's boxes, some of which are giants.

    Serial: the ENGINE's order — every box in ``find_objects`` order, giants inline.
    This matters: ``find_windows`` can emit boxes that touch (diagonally-offset
    clusters), so re-ordering them changes the trajectory. Measured on ``z16_full``,
    doing the small boxes before the giant cost **+64% wall** for the same result —
    reordering is not free, and any batching scheme pays it.

    Parallel: the small boxes as conflict-free batches first, then the giants.
    """
    gset = set(giants)
    if ex is None:
        for box in boxes:
            t = time.perf_counter()
            if box in gset:
                _giant(phi, box, ctx, ex)
                ctx.ph.solve_giant += time.perf_counter() - t
            else:
                _run_boxes(phi, [box], ctx, None)
                ctx.ph.solve_round += time.perf_counter() - t
        return
    t = time.perf_counter()
    _run_boxes(phi, [b for b in boxes if b not in gset], ctx, ex)
    ctx.ph.solve_round += time.perf_counter() - t
    t = time.perf_counter()
    for box in giants:
        _giant(phi, box, ctx, ex)
    ctx.ph.solve_giant += time.perf_counter() - t


def _mop(phi, ctx, ex, touched, mop_margin, max_window_area, max_sweeps=3):
    """Terminal mop pass — mirrors ``_common._mop_pass`` with a parallel batch."""
    H, W = phi.shape[1:]
    whole_cap = 4 * max_window_area
    for _sweep in range(max_sweeps):
        t = time.perf_counter()
        mask = pixel_fold_mask(ctx.constraint, phi, ctx.threshold)
        ctx.ph.foldmap += time.perf_counter() - t
        n = int(mask.sum())
        if n == 0:
            return
        lbl, _ = ndimage.label(mask)
        boxes, giants = [], []
        for sy, sx in ndimage.find_objects(lbl):
            fy0, fy1 = max(0, sy.start - mop_margin), min(H, sy.stop + mop_margin)
            fx0, fx1 = max(0, sx.start - mop_margin), min(W, sx.stop + mop_margin)
            box = (fy0, fy1, fx0, fx1)
            touched[
                max(0, fy0 - ctx.ring) : fy1 + ctx.ring, max(0, fx0 - ctx.ring) : fx1 + ctx.ring
            ] = True
            ctx.rep.mop_windows += 1
            boxes.append(box)
            if (fy1 - fy0) * (fx1 - fx0) > whole_cap:
                giants.append(box)
        _run_mixed(phi, boxes, giants, ctx, ex)
        t = time.perf_counter()
        left = int(pixel_fold_mask(ctx.constraint, phi, ctx.threshold).sum())
        ctx.ph.foldmap += time.perf_counter() - t
        if left >= n:
            return


def parallel_windowed_correct(
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
):
    """``windowed_correct``'s round loop with each round's windows batched to
    processes. ``n_workers <= 1`` runs the serial engine path (byte-identical)."""
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
        _InnerOpts(),
    )
    phi = np.array(phi_in, dtype=np.float64, copy=True)
    H, W = phi.shape[1:]
    t0 = time.perf_counter()

    ex = executor
    if ex is None and n_workers > 1:
        t = time.perf_counter()
        ex = ProcessPoolExecutor(max_workers=n_workers)
        list(ex.map(int, [0] * n_workers))  # force spawn now so it is not timed as solve
        ctx.ph.pool_start_s = time.perf_counter() - t
        t0 = time.perf_counter()

    try:
        t = time.perf_counter()
        j0 = min_field(constraint, phi)
        ctx.ph.foldmap += time.perf_counter() - t
        orig_fold = j0 < threshold
        ctx.rep.folds_before = int(orig_fold.sum())
        ctx.rep.min_before = float(j0.min())
        touched = np.zeros((H, W), bool)

        prev_nfold = None
        for _rnd in range(max_rounds):
            t = time.perf_counter()
            mask = pixel_fold_mask(constraint, phi, threshold)
            ctx.ph.foldmap += time.perf_counter() - t
            nfold = int(mask.sum())
            if nfold == 0 or (prev_nfold is not None and nfold >= prev_nfold):
                break
            prev_nfold = nfold
            ctx.rep.rounds += 1
            t = time.perf_counter()
            boxes = find_windows(mask, margin, ring)
            ctx.ph.find += time.perf_counter() - t
            giants = []
            for box in boxes:
                fy0, fy1, fx0, fx1 = box
                touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
                if (fy1 - fy0) * (fx1 - fx0) > max_window_area:
                    ctx.rep.giant_regions += 1
                    ctx.rep.giant_boxes.append(box)
                    giants.append(box)
            _run_mixed(phi, boxes, giants, ctx, ex)

        if mop_margin > 0:
            t = time.perf_counter()
            before = int(pixel_fold_mask(constraint, phi, threshold).sum())
            ctx.ph.foldmap += time.perf_counter() - t
            if before > 0:
                t = time.perf_counter()
                _mop(phi, ctx, ex, touched, mop_margin, max_window_area)
                ctx.ph.solve_mop += time.perf_counter() - t
                ctx.rep.mop_cleared = before - int(
                    pixel_fold_mask(constraint, phi, threshold).sum()
                )
    finally:
        if ex is not None and executor is None:
            ex.shutdown(wait=True)

    t = time.perf_counter()
    jf = min_field(constraint, phi)
    after = jf < threshold
    ctx.rep.folds_after = int(after.sum())
    ctx.rep.min_after = float(jf.min())
    ctx.rep.damage = int((after & ~orig_fold & ~touched).sum())
    ctx.rep.residual_in_window = int((after & touched).sum())
    ctx.rep.n_windows = len(ctx.rep.windows)
    ctx.ph.accounting = time.perf_counter() - t
    ctx.rep.time_s = ctx.ph.total = time.perf_counter() - t0
    return phi, ctx.rep, ctx.ph


# ---------------------------------------------------------------------------
# benchmark harness
# ---------------------------------------------------------------------------


def simplex_folds(phi, thr=0.01):
    """The task's fold metric: 2-triangle areas below ``thr``."""
    return int((np.minimum(*_triangle_areas_2d(phi[0], phi[1])) < thr).sum())


def load_case(name):
    if name == "z16_full":
        vol = np.load(VOL, mmap_mode="r")
        return np.ascontiguousarray(vol[1:, 16], dtype=np.float64)
    p = CASE_DIR / f"{name}.npy"
    if not p.exists():
        p = MAIN / "benchmarks" / "output" / "testcases" / f"{name}.npy"
    return np.array(np.load(p), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="z16_twist,z0_cluster,z0_sliver")
    ap.add_argument("--workers", default="1,4,8,16")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--reference", action="store_true", help="also time the shipped engine")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    _selftest()  # also the warm-up: pays import / OSQP / colouring-cache cost before timing

    rows = []
    for case in args.cases.split(","):
        phi = load_case(case)
        c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
        kw = dict(
            constraint=c,
            objective=NoneObjective(),
            threshold=args.threshold,
            maxiter=args.maxiter,
        )
        print(f"\n=== {case}  shape={phi.shape}  simplex folds before={simplex_folds(phi)} ===")
        ref = None
        if args.reference:
            from dvfopt.core.windowed import windowed_correct

            t = time.perf_counter()
            ref, rrep = windowed_correct(phi, **kw)
            dt = time.perf_counter() - t
            print(
                f"  engine serial       wall={dt:8.2f}s  folds_after(simplex)={simplex_folds(ref):5d}"
                f"  rep.folds_after={rrep.folds_after:5d}  damage={rrep.damage}"
                f"  windows={rrep.n_windows}"
            )
            rows.append(
                dict(
                    case=case,
                    mode="engine",
                    workers=0,
                    wall=dt,
                    folds=simplex_folds(ref),
                    rep_folds=rrep.folds_after,
                    damage=rrep.damage,
                    windows=rrep.n_windows,
                )
            )

        base = None
        for w in [int(x) for x in args.workers.split(",")]:
            out, rep, ph = parallel_windowed_correct(phi, n_workers=w, **kw)
            if base is None:
                base = out
            print(
                f"  proto workers={w:<3d}   wall={ph.total:8.2f}s  folds_after(simplex)="
                f"{simplex_folds(out):5d}  rep.folds_after={rep.folds_after:5d}"
                f"  damage={rep.damage}  windows={rep.n_windows}"
                f"  identical_to_w1={np.array_equal(out, base)}"
                + (f"  identical_to_engine={np.array_equal(out, ref)}" if ref is not None else "")
            )
            print(
                f"      phases: foldmap={ph.foldmap:.2f} find={ph.find:.2f} group={ph.grouping:.2f}"
                f" round={ph.solve_round:.2f} giant={ph.solve_giant:.2f} mop={ph.solve_mop:.2f}"
                f" grow={ph.grow:.2f} acct={ph.accounting:.2f} | pool_start={ph.pool_start_s:.2f}"
                f" dispatched={ph.n_dispatched} batches={ph.n_batches}"
                f" worker_solve={ph.worker_solve_s:.2f} batch_wall={ph.batch_wall_s:.2f}"
                f" pickle={ph.pickle_bytes}B"
            )
            rows.append(
                dict(
                    case=case,
                    mode="proto",
                    workers=w,
                    wall=ph.total,
                    folds=simplex_folds(out),
                    rep_folds=rep.folds_after,
                    damage=rep.damage,
                    windows=rep.n_windows,
                    identical_to_w1=bool(np.array_equal(out, base)),
                    identical_to_engine=(
                        bool(np.array_equal(out, ref)) if ref is not None else None
                    ),
                    max_abs_diff_vs_w1=float(np.abs(out - base).max()),
                    phases=asdict(ph),
                )
            )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")


def _selftest():
    """One runnable check: the halo crop reproduces the global sub-problem exactly,
    so a worker's window solve equals the serial engine's."""
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.35, (2, 24, 26))
    c = SimplexConstraint2DBilinear(shape=phi.shape[1:])
    ring = _locality_of(c).ring
    obj, opts = NoneObjective(), _InnerOpts()
    for box in [(4, 12, 5, 14), (0, 7, 0, 9), (16, 24, 15, 26)]:  # interior + both borders
        a = phi.copy()
        _solve_window(a, c, box, 0.01, obj, 200, ring, SliceReport(), allow_grow=False, opts=opts)
        crop, lb = _halo_crop(phi, box, ring)
        got, ok, rec, _ = _solve_box_worker(
            (type(c), crop, lb, 0.01, obj, 200, ring, 1e-3, "isqp", opts)
        )
        assert np.array_equal(got, a[:, box[0] : box[1], box[2] : box[3]]), f"halo mismatch {box}"
    assert _conflict((0, 10, 0, 10), (10, 20, 0, 10), 1)  # touching => reads overlap writes
    assert not _conflict((0, 10, 0, 10), (12, 20, 0, 10), 1)
    assert len(_groups([(0, 10, 0, 10), (10, 20, 0, 10), (30, 40, 0, 10)], 1)) == 2
    print("selftest ok")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
