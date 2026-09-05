"""End-to-end 3D fold-elimination orchestrator: ``correct_dvf_3d``.

Composes the validated, audited stages into ONE reproducible call that
drives a folded 3D displacement chunk toward strict simplex (3D) feasibility:

  Stage 0  Triage     — fixed-diagonal fold count + best-diagonal floor
                        (the "exists-a-positive-triangulation" predictor).
                        Early-out if already feasible.
  Stage 1  Bulk       — reduce the bulk of the folds. Routed by fold
                        sparsity: scattered folds -> active-band M10Tet
                        (per-cluster crops, ~70x); one big dense cluster ->
                        global M10Tet (CPU) or the GPU simplex (3D) barrier.
  Stage 2  Escape     — iterate the coupled k-ring SLSQP escape (+ local
                        M10Tet recovery) to break the shared-corner
                        local-minimum attractor that bulk solvers plateau
                        at. Stops when feasible or progress stalls.
  Stage 3  Verify     — strict re-check; annotate any irreducible residual
                        (incl. whether each residual cube is a genuine
                        "no positive triangulation under any diagonal" cell).

This is the packaged form of the hand-assembled chain that first reached
n_neg=0 on B0039 z=0..15 (REPORT Parts XIII–XVII). It reuses the
parallel kernels, active-band recovery, coupled-escape, and (optionally)
the GPU barrier — all deep-audited bit-exact / feasibility-safe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dvfopt._logging import log_info
from dvfopt.jacobian.tetrahedron_sign import (
    n_neg_best_diagonal,
    six_tet_min_volume_3d,
)
from dvfopt.metrics import fold_stats


@dataclass
class Correct3DReport:
    """Structured outcome of :func:`correct_dvf_3d`."""

    feasible: bool
    n_neg_in: int
    n_neg_out: int
    n_below_out: int
    min_T_out: float
    best_diag_floor_in: int  # cubes with NO positive triangulation (input)
    best_diag_floor_out: int  # ditto (output) — the true irreducible set
    l1_from_input: float
    wall_s: float
    stages: list = field(default_factory=list)  # per-stage dicts
    residual_cubes: list = field(default_factory=list)  # (cz,cy,cx, min_T, diag_fixable)


def _stats(phi, threshold):
    mv = six_tet_min_volume_3d(phi)
    st = fold_stats(mv, threshold)
    return mv, st.n_neg, st.n_below, st.min_val


def _fold_sparsity(mv, threshold):
    """Return (n_fold_cubes, max_axis_span_fraction) — used to route bulk."""
    fold = mv <= 0
    if not fold.any():
        return 0, 0.0
    cz, cy, cx = np.where(fold)
    Dc, Hc, Wc = mv.shape
    span = max(
        (cz.max() - cz.min() + 1) / max(1, Dc),
        (cy.max() - cy.min() + 1) / max(1, Hc),
        (cx.max() - cx.min() + 1) / max(1, Wc),
    )
    return int(fold.sum()), float(span)


def correct_dvf_3d(
    phi,
    *,
    threshold: float = 0.01,
    recover_threshold: Optional[float] = None,
    bulk: str = 'auto',  # 'auto'|'active_band'|'global'|'gpu'|'multiscale'
    max_escape_iters: int = 8,
    escape_k_ring: int = 2,
    escape_feasibility_thr: float = 1e-3,
    thorough: bool = True,
    n_workers: int = 1,
    sparse_span_cutoff: float = 0.6,
    verbose: int = 0,
    checkpoint_dir=None,
):
    """Drive a folded 3D DVF chunk toward strict simplex (3D) feasibility.

    Parameters
    ----------
    phi : ndarray (3, D, H, W), channels [dz, dy, dx]
    threshold : float, default 0.01
        Strict per-tet feasibility threshold (the success criterion is
        ``min tet volume >= threshold`` everywhere).
    recover_threshold : float | None
        Inner M10Tet target for bulk/recovery. None -> ``1.2 * threshold``
        (recover to a margin above the strict check).
    bulk : {'auto','active_band','global','gpu'}, default 'auto'
        Bulk-reduction route. ``'auto'`` picks active-band when folds are
        spatially localized (fast) and global M10Tet when they span the
        chunk. ``'gpu'`` uses the CUDA simplex (3D) barrier (for large dense
        chunks).
    max_escape_iters : int, default 8
        Max coupled-escape iterations to break the residual attractor.
    escape_k_ring, escape_feasibility_thr : coupled-escape knobs.
    n_workers : int, default 1
        Workers for active-band's per-cluster parallelism (default serial;
        see the spawn-tax note in the active-band docstring).
    sparse_span_cutoff : float, default 0.6
        In ``'auto'`` bulk mode, folds spanning more than this fraction of
        an axis are treated as "dense" -> global solve.
    verbose : int
    checkpoint_dir : str or path-like or None, default None
        Make the run resumable (``dvfopt.checkpoint.RunCheckpoint``): the
        whole field is mirrored to ``<dir>/field.npy`` after the bulk stage,
        after every escape iteration and after the multiscale re-seed, with
        the stage record (and the escape loop's state) in ``state.json``.
        Calling again with the same input / knobs / directory reloads the
        field at the last finished stage and skips those stages' work; a
        ``done`` checkpoint just reloads the result. The tighten stage is
        not a resume point on its own (the run finishes right after it).
        A mismatched checkpoint raises ``ValueError``.

    Returns
    -------
    phi_out : ndarray
    report : Correct3DReport
    """
    if phi.shape[0] != 3 or phi.ndim != 4:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')
    phi0 = np.asarray(phi, dtype=np.float64)
    # NaN-aware: the fused simplex (3D) min-volume kernel initialises at 1e300 and
    # uses `<` comparisons that NaN always fails, so a non-finite field would
    # otherwise sail through triage as "feasible" (silent success on
    # corrupted data).
    if not np.isfinite(phi0).all():
        raise ValueError(
            'phi contains non-finite values (NaN/Inf); correct_dvf_3d requires a finite field.'
        )
    rec_thr = recover_threshold if recover_threshold is not None else 1.2 * threshold

    # Lazy imports (avoid import cycle; keep top-level import cheap).
    from dvfopt import (
        CoupledKRing3DStrategy,
        L1Objective,
        SimplexConstraint3D,
        Solver,
    )
    from dvfopt.core.wallbreakers._coupled_kring_3d import (
        active_band_alm_recovery_3d,
    )

    constraint = SimplexConstraint3D(shape=phi0.shape[1:])
    objective = L1Objective(eps=1e-4)

    def _m10tet(p, thr, gpu=False):
        if gpu:
            from dvfopt import BarrierTet3DTorchStrategy

            strat = BarrierTet3DTorchStrategy(dtype='float64')
        else:
            from dvfopt import HarmonicALMBarrier3DStrategy

            strat = HarmonicALMBarrier3DStrategy()
        return (
            Solver(
                constraint=constraint,
                objective=objective,
                strategy=strat,
                threshold=thr,
            )
            .fit(p)
            .corrected
        )

    t0 = time.time()
    stages = []
    cur = phi0.copy()

    # ---- Stage 0: triage ----
    mv, n_neg_in, n_below_in, min_in = _stats(cur, threshold)
    bd_floor_in = n_neg_best_diagonal(cur, threshold=0.0)
    if verbose:
        log_info(
            f'[triage] n_neg={n_neg_in} n<thr={n_below_in} min_T={min_in:+.5f} '
            f'best-diag-floor={bd_floor_in}',
        )
    stages.append(
        dict(
            stage='triage',
            n_neg=n_neg_in,
            n_below=n_below_in,
            min_T=min_in,
            best_diag_floor=bd_floor_in,
        )
    )
    if n_neg_in == 0 and n_below_in == 0:
        # `cur` is unchanged since the triage measurement — reuse it.
        return cur, _finalize(
            cur,
            phi0,
            threshold,
            n_neg_in,
            bd_floor_in,
            stages,
            t0,
            stats=(mv, n_neg_in, n_below_in, min_in),
        )

    # ---- Resumable checkpoint (opt-in): whole-field units per stage ----
    ck = None
    if checkpoint_dir is not None:
        from dvfopt.checkpoint import RunCheckpoint

        meta = dict(
            threshold=threshold,
            recover_threshold=recover_threshold,
            bulk=bulk,
            max_escape_iters=max_escape_iters,
            escape_k_ring=escape_k_ring,
            escape_feasibility_thr=escape_feasibility_thr,
            thorough=thorough,
            sparse_span_cutoff=sparse_span_cutoff,
        )
        ck = RunCheckpoint(checkpoint_dir, phi0, meta, slab=lambda _unit: Ellipsis).open()
        if ck.finished:
            cur[...] = ck.field
            if verbose:
                log_info(f'[3d resume] finished run reloaded from {checkpoint_dir}')
            stages.append(dict(stage='resumed', n_neg=-1))
            return cur, _finalize(cur, phi0, threshold, n_neg_in, bd_floor_in, stages, t0)
        if ck.done:
            cur[...] = ck.field
            st = _stats(cur, threshold)
            if verbose:
                log_info(f'[3d resume] {ck.done} from {checkpoint_dir}')

    # ---- Stage 1: bulk reduction ----
    n_fold, span = _fold_sparsity(mv, threshold)
    n_cubes = int(mv.size)
    frac = n_fold / max(1, n_cubes)
    route = bulk
    if route == 'auto':
        # Active-band self-routes per cluster (oversized clusters are tiled
        # into bounded sub-boxes rather than solved globally), so it is the
        # right default for any localized fold set — crops are small
        # relative to the full extent. Use a plain global solve only when
        # folds saturate the chunk. Multi-scale is NOT
        # auto-selected (hard to predict when it helps); it is an explicit
        # route + the escape-stall fallback below, invoked only when the
        # cheap path actually plateaus.
        route = 'global' if frac > 0.5 else 'active_band'
    ts = time.time()
    if ck is not None and ck.is_done('bulk'):
        stages.append(ck.rows['bulk']['stage'])
    elif route == 'active_band':
        cur, info = active_band_alm_recovery_3d(
            cur,
            threshold=rec_thr,
            n_workers=n_workers,
            verbose=verbose,
        )
    elif route == 'multiscale':
        from dvfopt.core.wallbreakers._multiscale_3d import multiscale_seed_3d

        cur, _ = multiscale_seed_3d(cur, threshold=rec_thr, verbose=verbose)
    elif route == 'gpu':
        cur = _m10tet(cur, rec_thr, gpu=True)
    else:  # global
        cur = _m10tet(cur, rec_thr, gpu=False)
    if ck is None or not ck.is_done('bulk'):
        st = _stats(cur, threshold)  # threaded through the stages below
        n_neg_b, n_below_b, min_b = st[1:]
        if verbose:
            log_info(
                f'[bulk:{route}] n_neg={n_neg_b} min_T={min_b:+.5f} ({time.time() - ts:.1f}s)',
            )
        stages.append(
            dict(
                stage=f'bulk:{route}',
                n_neg=n_neg_b,
                n_below=n_below_b,
                min_T=min_b,
                wall_s=time.time() - ts,
            )
        )
        if ck is not None:
            ck.mark('bulk', cur, row=dict(stage=stages[-1]))
    mv, n_neg_b, n_below_b, min_b = st

    # ---- Stage 2: coupled-escape loop ----
    # Pathology guard (a-priori): if a large FRACTION of cubes have no
    # positive triangulation under ANY diagonal (best-diagonal floor), the
    # input is so deeply tangled that strict feasibility is unreachable —
    # the feasible set is ~empty. The coupled escape can neither fix those
    # cells nor stay local (cluster halos cover the small volume), so it
    # just grinds for hours. Take the bulk result and annotate the residual.
    floor_frac = bd_floor_in / max(1, mv.size)
    if n_neg_b > 0 and floor_frac > 0.2:
        if verbose:
            log_info(
                f'[escape] skipped — {bd_floor_in} cubes ({floor_frac:.0%}) have '
                f'no positive triangulation; feasible set ~empty, annotating',
            )
        stages.append(
            dict(stage='escape:skipped_pathological', n_neg=n_neg_b, best_diag_floor=bd_floor_in)
        )
        # `cur` is unchanged since the post-bulk measurement — reuse it.
        if ck is not None:
            ck.finish()
        return cur, _finalize(cur, phi0, threshold, n_neg_in, bd_floor_in, stages, t0, stats=st)

    # Iterate the coupled k-ring escape; when it stalls, ESCALATE the halo
    # (k=2 -> 3 -> 4). The research showed k=2 plateaus at the shared-corner
    # attractor while k=3 breaks it (Methods B/D, REPORT Part XIV) — a larger
    # halo contains the perturbation the residual cube needs.
    def _run_escape(cur, st, tag=''):
        # `st` is the caller's already-measured ``_stats(cur, threshold)``
        # 4-tuple; it is carried through the loop (each iteration measures
        # once, after the solve) and the final measurement is returned, so
        # an unchanged array is never re-measured.
        last_n = st[1]
        stall = 0
        k = escape_k_ring
        for it in range(max_escape_iters):
            name = f'escape{tag}{it + 1}'
            if ck is not None and ck.is_done(name):
                # Replay: the field is already restored; pick up the loop
                # state (halo, stall counter, reference count, whether the
                # loop ended here) the original iteration left behind.
                row = ck.rows[name]
                stages.append(row['stage'])
                k, stall, last_n, stop = row['loop']
                if stop:
                    break
                continue
            n_now = st[1]
            if n_now == 0:
                break
            ts = time.time()
            cur = (
                Solver(
                    constraint=constraint,
                    objective=objective,
                    strategy=CoupledKRing3DStrategy(
                        k_ring=k,
                        feasibility_thr=escape_feasibility_thr,
                        mode='cluster',
                        n_workers=n_workers,
                        recover=True,
                        recover_threshold=rec_thr,
                    ),
                    threshold=threshold,
                )
                .fit(cur)
                .corrected
            )
            st = _stats(cur, threshold)
            _, n_after, _, min_after = st
            if verbose:
                log_info(
                    f'[escape{tag} {it + 1} k={k}] n_neg {n_now}->{n_after} '
                    f'min_T={min_after:+.5f} ({time.time() - ts:.1f}s)',
                )
            stages.append(
                dict(
                    stage=f'escape{tag}{it + 1}',
                    k_ring=k,
                    n_neg=n_after,
                    min_T=min_after,
                    wall_s=time.time() - ts,
                )
            )
            stop = n_after == 0
            if not stop and n_after >= last_n:
                stall += 1
                chunk_min = min(cur.shape[1:])
                if k == escape_k_ring and chunk_min > 4 * (k + 1):
                    # escalate; `last_n` deliberately stays the pre-stall count
                    k += 1
                    stall = 0
                    if verbose:
                        log_info(f'[escape] stalled — escalating halo to k={k}')
                else:
                    if stall >= 2:
                        if verbose:
                            log_info('[escape] stalled — stopping')
                        stop = True
                    last_n = n_after
            elif not stop:
                stall = 0
                k = escape_k_ring
                last_n = n_after
            if ck is not None:
                ck.mark(name, cur, row=dict(stage=stages[-1], loop=[k, stall, last_n, stop]))
            if stop:
                break
        return cur, st

    cur, st = _run_escape(cur, st)

    # Multi-scale stall fallback: if the cheap path plateaued (still folded)
    # and the chunk is downsamplable and not pathological, re-seed via the
    # coarse-to-fine basin-hop (the ingredient that broke the thick dense
    # band's single-scale plateau in the research) and re-escape ONCE.
    n_esc = st[1]  # _run_escape returned the measurement of the final `cur`
    floor_frac = bd_floor_in / max(1, mv.size)
    if ck is not None and ck.is_done('multiscale_fallback'):
        stages.append(ck.rows['multiscale_fallback']['stage'])
        cur, st = _run_escape(cur, st, tag='2')
    elif (
        thorough
        and n_esc > 0
        and route != 'multiscale'
        and floor_frac <= 0.2
        and min(cur.shape[1:]) >= 8
    ):
        from dvfopt.core.wallbreakers._multiscale_3d import multiscale_seed_3d

        ts = time.time()
        if verbose:
            log_info(
                f'[multiscale-fallback] escape plateaued at {n_esc}; re-seeding coarse-to-fine',
            )
        seeded, _ = multiscale_seed_3d(cur, threshold=rec_thr, verbose=verbose)
        st_seed = _stats(seeded, threshold)
        n_seed = st_seed[1]
        # Accept the re-seed only if it didn't make things worse.
        if n_seed <= n_esc:
            cur = seeded
            st = st_seed
        stages.append(dict(stage='multiscale_fallback', n_neg=n_seed, wall_s=time.time() - ts))
        if ck is not None:
            ck.mark('multiscale_fallback', cur, row=dict(stage=stages[-1]))
        cur, st = _run_escape(cur, st, tag='2')

    # ---- Stage 3: final tighten (n<threshold but feasible) ----
    # `st` is the measurement of the current `cur` (from the escape loop, or
    # from the bulk/re-seed stats if the loop never ran an iteration).
    _, n_fin, n_below_fin, _ = st
    if n_fin == 0 and n_below_fin > 0:
        ts = time.time()
        # local recovery to lift near-threshold cells above the strict bar
        cur, _ = active_band_alm_recovery_3d(
            cur,
            threshold=rec_thr,
            n_workers=n_workers,
            verbose=verbose,
        )
        st = _stats(cur, threshold)  # cur changed — fresh measurement
        n_below_fin = st[2]
        stages.append(dict(stage='tighten', n_below=n_below_fin, wall_s=time.time() - ts))

    if ck is not None:
        ck.finish(cur)
    # `st` always holds the measurement of the final `cur` here — reuse it.
    return cur, _finalize(cur, phi0, threshold, n_neg_in, bd_floor_in, stages, t0, stats=st)


def _finalize(cur, phi0, threshold, n_neg_in, bd_floor_in, stages, t0, stats=None):
    # `stats` is an optional already-measured ``_stats(cur, threshold)``
    # 4-tuple ``(mv, n_neg, n_below, min_T)`` — pass it to skip a redundant
    # full-volume fused-kernel pass when `cur` is provably unchanged since
    # the measurement (mv is needed here for residual-cube enumeration).
    if stats is None:
        mv, n_neg, n_below, min_T = _stats(cur, threshold)
    else:
        mv, n_neg, n_below, min_T = stats
    bd_floor = n_neg_best_diagonal(cur, threshold=0.0)
    residual = []
    if n_neg > 0:
        from dvfopt.jacobian.tetrahedron_sign import best_diagonal_min_volume

        best_min, _ = best_diagonal_min_volume(cur)
        cz, cy, cx = np.where(mv <= 0)
        for z, y, x in zip(cz, cy, cx):
            residual.append(
                (
                    int(z),
                    int(y),
                    int(x),
                    float(mv[z, y, x]),
                    bool(best_min[z, y, x] > 0),  # fixable by re-triangulation?
                )
            )
    return Correct3DReport(
        feasible=(n_neg == 0 and n_below == 0),
        n_neg_in=n_neg_in,
        n_neg_out=n_neg,
        n_below_out=n_below,
        min_T_out=min_T,
        best_diag_floor_in=bd_floor_in,
        best_diag_floor_out=bd_floor,
        l1_from_input=float(np.abs(cur - phi0).sum()),
        wall_s=float(time.time() - t0),
        stages=stages,
        residual_cubes=residual,
    )


__all__ = ['Correct3DReport', 'correct_dvf_3d']
