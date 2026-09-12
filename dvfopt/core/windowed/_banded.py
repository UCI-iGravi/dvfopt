"""Banded full-volume driver for the windowed engine (phase 4 of the 3D port).

:func:`windowed_correct_banded` solves a large ``(3, D, H, W)`` volume by
splitting it into z-bands instead of handing the whole volume to
:func:`~dvfopt.core.windowed._common.windowed_correct` at once. Bands are the
z *cores* ``[k * band, min(D, (k + 1) * band))``; each band's worker receives
the *slab* ``[core_lo - overlap, core_hi + overlap)`` of ONE snapshot taken
before the sweep (every worker sees the same pre-sweep field — this is
restricted additive, Jacobi-style, Schwarz across bands, not a sequential
chain), runs the full engine on that slab alone (``giant_workers=0``,
``checkpoint_dir=None``, ``verbose=0``), and returns only its CORE planes plus
the core's own ``touched`` planes — the overlap exists solely so the worker's
engine sees enough context to enforce every constraint its core pixels
influence, and is discarded on return. The parent commits each core in place.

A cluster that straddles a core boundary is therefore solved independently
(and usually differently) by both neighbouring bands, each truncating its
half at the boundary; the mismatch this leaves is exactly the seam pass's
job. After every band is committed, ONE serial call to ``windowed_correct``
runs on the composed full field (``checkpoint_dir=<dir>/seam`` when
checkpointing, ``touched_out=touched``) and repairs whatever folds remain —
it only opens windows where folds are still present, so its cost is the
*cost of banding* (a boundary-free volume never triggers it), tracked as
``report.seam_windows``.

Memory: the parent process holds the full ``phi`` plus ``j0`` / ``orig_fold``
/ ``touched`` (all one array per voxel); each worker holds only its slab plus
that slab's own engine temporaries — so peak worker memory scales with
``band + 2 * overlap`` planes, not ``D``.

``overlap`` must be at least the constraint family's ``margin + ring`` (the
frozen context a window's free pixels need to evaluate correctly); this is
asserted at entry. The default of 8 is the phase-2 ring/margin sum (1 + 3 for
the simplex (3D) family) plus slack.

Requires a 3D constraint (``constraint.dim == 3``, i.e.
:class:`~dvfopt.constraints.SimplexConstraint3D`) — 2D volumes are handled by
:class:`~dvfopt.unified.DVFoptConfig`'s per-slice sweep, which already
parallelizes over independent z-slices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dvfopt._logging import log_info, logger

from ._common import SliceReport, windowed_correct
from ._locality import _locality_of, min_field, pixel_fold_mask

_FORBIDDEN_ENGINE_KW = ('checkpoint_dir', 'touched_out', 'verbose')


@dataclass
class BandedReport:
    bands: int = 0
    band: int = 0
    overlap: int = 0
    band_walls: list = field(default_factory=list)  # seconds per band, band order
    band_folds_after: list = field(default_factory=list)  # each band's own folds_after (slab-local)
    folds_before: int = 0
    folds_after: int = 0
    folds_after_zero: int = 0
    best_diag_floor_after: int = -1
    best_diag_floor_after_zero: int = -1
    min_before: float = 0.0
    min_after: float = 0.0
    damage: int = 0  # against the ORIGINAL input, touched = union over bands + seam pass
    n_windows: int = 0  # sum over bands + the seam pass
    seam_windows: int = 0  # the seam pass's windows (the cost of banding)
    seam_folds_before: int = 0  # folds on the composed field before the seam pass
    time_s: float = 0.0
    resumed_from: str = ''
    seam: SliceReport | None = None


def _band_task(args):
    """Pool worker: the engine on ONE z-slab; returns the core planes + the core's touched.

    Runs in a spawned worker (or, for a single leftover band / ``n_workers=1``,
    serially in the parent) — :func:`~dvfopt.core._pool.pin_worker_threads` is
    a no-op outside a pool worker, so calling it unconditionally is safe either
    way.
    """
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()
    slab, lo_in_slab, hi_in_slab, inner, ctype, threshold, objective, kw = args
    c = ctype(shape=slab.shape[1:])
    touched = np.zeros(slab.shape[1:], bool)
    t = time.perf_counter()
    out, rep = windowed_correct(
        slab,
        inner,
        constraint=c,
        objective=objective,
        threshold=threshold,
        touched_out=touched,
        checkpoint_dir=None,
        giant_workers=0,
        verbose=0,
        **kw,
    )
    return (
        out[:, lo_in_slab:hi_in_slab].copy(),
        touched[lo_in_slab:hi_in_slab].copy(),
        int(rep.n_windows),
        int(rep.folds_after),
        time.perf_counter() - t,
    )


def windowed_correct_banded(
    phi_in,
    inner='isqp',
    *,
    constraint,
    threshold,
    objective=None,
    band=24,
    overlap=8,
    n_workers=1,
    checkpoint_dir=None,
    verbose=1,
    **engine_kw,
):
    """Z-banded driver over :func:`windowed_correct` for a large 3D volume.

    See the module docstring for the design (band/core/slab geometry,
    restricted-additive Schwarz across bands, the terminal serial seam pass).
    ``engine_kw`` is forwarded to every ``windowed_correct`` call (band
    workers and the seam pass alike); it must not set ``checkpoint_dir`` /
    ``touched_out`` / ``verbose`` (this function owns those per-stage) and a
    ``giant_workers`` request is honoured only on the seam pass — a band
    worker cannot nest a process pool, so its own ``giant_workers`` is forced
    to 0 (logged at DEBUG if the caller asked for more).
    """
    if getattr(constraint, 'dim', 2) != 3:
        raise ValueError('windowed_correct_banded needs a 3D constraint (SimplexConstraint3D)')
    for bad in _FORBIDDEN_ENGINE_KW:
        if bad in engine_kw:
            raise TypeError(
                f'windowed_correct_banded: engine_kw must not contain {bad!r} '
                f'(it is set by this function, per stage)'
            )
    loc = _locality_of(constraint)
    need = int(engine_kw.get('margin', 3)) + loc.ring
    if overlap < need:
        raise ValueError(f'overlap {overlap} < margin + ring = {need}')

    from dvfopt.objectives import L2Objective

    objective = L2Objective() if objective is None else objective
    t0 = time.perf_counter()
    phi = np.array(phi_in, dtype=np.float64, copy=True)
    D = phi.shape[1]
    j0 = min_field(constraint, phi)
    orig_fold = j0 < threshold
    rep = BandedReport(
        band=int(band),
        overlap=int(overlap),
        folds_before=int(orig_fold.sum()),
        min_before=float(j0.min()),
    )
    touched = np.zeros(phi.shape[1:], bool)
    cores = [(z, min(D, z + band)) for z in range(0, D, band)]
    rep.bands = len(cores)

    ck = None
    if checkpoint_dir is not None:
        from dvfopt.checkpoint import RunCheckpoint, atomic_replace

        meta = dict(
            engine='windowed_banded',
            band=int(band),
            overlap=int(overlap),
            threshold=float(threshold),
            inner=str(inner),
            constraint=type(constraint).__name__,
            objective=type(objective).__name__,
        )
        ck = RunCheckpoint(
            checkpoint_dir,
            phi_in,
            meta,
            slab=lambda u: (slice(None), slice(*cores[int(str(u).rpartition(':')[2])])),
        ).open()
        tp = ck.dir / 'touched.npy'
        if ck.finished:
            rep.resumed_from = 'finished'
            phi[...] = ck.field
            touched[...] = np.load(tp)
        elif ck.done:
            ck.restore_into(phi)
            touched[...] = np.load(tp)
            rep.resumed_from = str(ck.done[-1])
            for u in ck.done:
                r = ck.rows.get(str(u), {})
                rep.band_walls.append(float(r.get('wall_s', 0.0)))
                rep.band_folds_after.append(int(r.get('folds_after', -1)))
                rep.n_windows += int(r.get('n_windows', 0))
        if verbose and rep.resumed_from:
            log_info(f'[banded resume] {rep.resumed_from} from {ck.dir}')

    def _save_touched():
        if ck is None:
            return
        tmp = ck.dir / 'touched.npy.tmp'
        with open(tmp, 'wb') as f:  # np.save(path) would append another .npy suffix
            np.save(f, touched)
        atomic_replace(tmp, ck.dir / 'touched.npy')

    if rep.resumed_from != 'finished':
        kw = dict(engine_kw)
        if 'giant_workers' in kw:
            logger.debug(
                'windowed_correct_banded: ignoring engine_kw giant_workers=%r for band '
                'workers (a worker cannot nest a process pool); the seam pass still '
                'honours it',
                kw.pop('giant_workers'),
            )
        todo = [k for k in range(len(cores)) if ck is None or not ck.is_done(f'band:{k}')]
        args = []
        for k in todo:
            lo, hi = cores[k]
            s0, s1 = max(0, lo - overlap), min(D, hi + overlap)
            args.append(
                (
                    phi[:, s0:s1].copy(),
                    lo - s0,
                    hi - s0,
                    inner,
                    type(constraint),
                    threshold,
                    objective,
                    kw,
                )
            )
        if n_workers > 1 and len(args) > 1:
            from dvfopt.core._pool import pool_map

            results = pool_map(_band_task, args, n_workers)
        else:
            results = [_band_task(a) for a in args]
        for k, (vals, tch, nw, fa, wall) in zip(todo, results):
            lo, hi = cores[k]
            phi[:, lo:hi] = vals
            touched[lo:hi] |= tch
            rep.band_walls.append(wall)
            rep.band_folds_after.append(fa)
            rep.n_windows += nw
            if ck is not None:
                _save_touched()
                ck.mark(f'band:{k}', vals, row=dict(wall_s=wall, folds_after=fa, n_windows=nw))
            if verbose:
                log_info(
                    f'[banded] band {k + 1}/{len(cores)} z[{lo},{hi}) windows {nw} '
                    f'folds_after {fa} {wall:.0f}s'
                )

        rep.seam_folds_before = int(pixel_fold_mask(constraint, phi, threshold).sum())
        seam_dir = None if ck is None else ck.dir / 'seam'
        out, srep = windowed_correct(
            phi,
            inner,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            checkpoint_dir=seam_dir,
            touched_out=touched,
            verbose=verbose,
            **engine_kw,
        )
        phi[...] = out
        rep.seam = srep
        rep.seam_windows = int(srep.n_windows)
        rep.n_windows += rep.seam_windows
        if ck is not None:
            _save_touched()
            ck.finish(phi)

    jf = min_field(constraint, phi)
    after = jf < threshold
    rep.folds_after = int(after.sum())
    rep.folds_after_zero = int((jf <= 0).sum())
    rep.min_after = float(jf.min())
    rep.damage = int((after & ~orig_fold & ~touched).sum())

    from dvfopt.jacobian.tetrahedron_sign import best_diagonal_min_volume

    best_min, _ = best_diagonal_min_volume(phi)
    rep.best_diag_floor_after = int((best_min <= threshold).sum())
    rep.best_diag_floor_after_zero = int((best_min <= 0.0).sum())
    rep.time_s = time.perf_counter() - t0
    return phi, rep


__all__ = ['BandedReport', 'windowed_correct_banded']
