"""End-to-end 2.5D marching fold-elimination orchestrator: ``correct_dvf_25d``.

Wires the two ported core modules —
:mod:`dvfopt.core.marching._marching_25d` (``layer_min_v`` / ``march_slice``)
and :mod:`dvfopt.core.marching._mop_interior_3d` (``mop_interior_3d``) — into
ONE reproducible call that drives a folded, in-plane-only 3D displacement
field toward strict simplex (3D) feasibility.

The 2.5D precondition
---------------------
The inter-layer simplex (3D) volume of a ``(lower, upper)`` layer pair depends only
on the two slices' in-plane displacement (``dy``/``dx``); the through-plane
channel ``dz`` must be identically zero. Run per-slice 2D correction first
(which yields ``dz == 0``) before calling this pipeline. ``dz`` (``phi[0]``)
is validated up front and is NEVER written by this pipeline.

The algorithm
-------------
Field layout ``(3, D, H, W)`` = ``[dz, dy, dx]``. A marching "slice" is
``phi[1:3, z]`` -> a ``(2, H, W)`` array ``[dy, dx]``.

1. Precondition check: the field must be finite and ``dz`` must be zero
   (raise otherwise).
2. Origin auto-selection: pick the inter-layer with the FEWEST folds as the
   frozen seed, so no slice is ever cold-started against raw data.
3. Bidirectional sweep outward from the origin: each slice is repaired
   against its already-repaired neighbour with ``march_slice``. The origin
   slice itself is repaired in the down sweep against its already-repaired
   upper neighbour, matching the validated research sweep (an origin at the
   very top of the volume has no upper neighbour and stays frozen).
4. Optional final mop (:func:`mop_interior_3d`): the sweep freezes a whole
   slice so it cannot fix folds needing BOTH slices of a pair to move; the
   mop frees both and cleans up the residual.

This is the packaged form of the validated "marching full volume"
experiment from ``research/strict_feasibility_3d``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dvfopt._logging import log_info
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d
from dvfopt.metrics import fold_stats


@dataclass
class Correct25DReport:
    """Structured outcome of :func:`correct_dvf_25d`."""

    feasible: bool
    n_neg_in: int
    n_neg_out: int
    n_below_out: int
    min_T_out: float
    l1_from_input: float
    wall_s: float
    origin: int
    stages: list = field(default_factory=list)  # per-stage dicts
    #: residual folded cells under the best-of-4-main-diagonals certificate —
    #: the honest floor (research/strict_feasibility_3d Part XXIII).
    n_neg_best_diag_out: int = -1


def _stats(phi, threshold):
    mv = six_tet_min_volume_3d(phi)
    st = fold_stats(mv, threshold)
    return mv, st.n_neg, st.n_below, st.min_val


def _finalize(cur, phi_in, threshold, n_neg_in, origin, stages, t0, stats=None):
    # `phi_in` is the caller's (never-mutated) array — used directly as the L1
    # reference rather than keeping a second full-volume copy around.
    # `stats` is an optional already-measured `(n_neg, n_below, min_T)` tuple
    # for the final state — pass it to skip a redundant full-volume measure.
    if stats is None:
        _, n_neg, n_below, min_T = _stats(cur, threshold)
    else:
        n_neg, n_below, min_T = stats
    from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal

    _best = int(n_neg_best_diagonal(cur, threshold))
    return Correct25DReport(
        n_neg_best_diag_out=_best,
        feasible=(n_neg == 0 and n_below == 0),
        n_neg_in=n_neg_in,
        n_neg_out=n_neg,
        n_below_out=n_below,
        min_T_out=min_T,
        l1_from_input=float(np.abs(cur - phi_in).sum()),
        wall_s=float(time.time() - t0),
        origin=int(origin),
        stages=stages,
    )


def _checkpoint_open(checkpoint_dir, phi, meta):
    """Open (or create) the resumable checkpoint under ``checkpoint_dir``:
    ``field.npy`` — a memmap mirror of the output, written slice by slice as
    the sweep repairs them — and ``state.json`` (``meta`` + ``n_done`` sweep
    slices + ``stage``). An existing checkpoint must match ``meta`` (shape,
    input hash, threshold, rows knob, origin) exactly; a mismatch raises
    rather than silently resuming the wrong run.
    """
    import hashlib
    import json
    from pathlib import Path

    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(
        meta,
        shape=list(phi.shape),
        input_sha256=hashlib.sha256(np.ascontiguousarray(phi)).hexdigest(),
    )
    sp, fp = d / 'state.json', d / 'field.npy'
    if sp.exists() and fp.exists():
        state = json.loads(sp.read_text(encoding='utf-8'))
        bad = {k: (state.get(k), v) for k, v in meta.items() if state.get(k) != v}
        if bad:
            raise ValueError(f'checkpoint {d} does not match this run (stored, this): {bad}')
        return np.lib.format.open_memmap(fp, mode='r+'), state
    state = dict(meta, n_done=0, stage='sweep')
    _checkpoint_save(d, state)
    return np.lib.format.open_memmap(fp, mode='w+', dtype=np.float64, shape=phi.shape), state


def _checkpoint_save(checkpoint_dir, state, **update):
    """Update ``state`` and rewrite ``state.json`` atomically (tmp + replace)."""
    import json
    import os
    from pathlib import Path

    state.update(update)
    sp = Path(checkpoint_dir) / 'state.json'
    tmp = sp.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state), encoding='utf-8')
    os.replace(tmp, sp)


def correct_dvf_25d(
    phi,
    *,
    threshold: float = 0.01,
    origin='auto',  # 'auto' | int inter-layer/slice seed index
    mop: bool = True,
    mop_max_folds: int = 20000,
    n_workers: int = 1,
    pad: int = 4,
    dil: int = 2,
    max_rounds: int = 6,
    max_box: int = 90,
    mu: float = 1000.0,
    dz_tol: float = 1e-12,
    verbose: int = 0,
    progress_callback=None,
    callback_copies: bool = True,
    orientation_delta=None,
    checkpoint_dir=None,
):
    """Drive a folded, in-plane-only 3D DVF toward strict simplex (3D) feasibility.

    Parameters
    ----------
    phi : ndarray (3, D, H, W), channels [dz, dy, dx]
        ``dz`` (``phi[0]``) MUST be identically zero — this is the 2.5D
        precondition (run per-slice 2D correction first). ``phi[0]`` is never
        written; the caller's array is never mutated.
    threshold : float, default 0.01
        Strict per-tet feasibility threshold. ``march_slice`` targets
        ``thr3 = threshold + 1e-4`` (strict with margin) and ``thr2 = threshold``.
    origin : {'auto'} or int, default 'auto'
        The seed slice. It anchors the up sweep frozen, then is itself
        repaired at the start of the down sweep against its already-repaired
        upper neighbour (matching the validated research sweep). An origin at
        the very top of the volume has no upper neighbour and stays frozen.
        ``'auto'`` picks the slice bordering the mildest (fewest-fold)
        inter-layer.
    mop : bool, default True
        Run the frozen-rim 3D-interior mop after the sweep to clean up folds
        that need BOTH slices of a pair to move.
    mop_max_folds : int, default 20000
        Only run the mop when ``0 < residual folds <= mop_max_folds``.
    n_workers : int, default 1
        Cluster parallelism inside ``march_slice`` (serial by default — the
        Windows process-spawn tax makes the pool unhelpful for small runs).
    pad, dil, max_rounds, max_box, mu : ``march_slice`` knobs.
    dz_tol : float, default 1e-12
        Tolerance for the ``dz == 0`` precondition check.
    verbose : int, default 0
        ``>= 1`` prints bracketed progress lines.
    progress_callback : callable or None
        When supplied, called after each sweep-slice repair and after the
        mop with ``{'phase': 'sweep'|'mop', 'index', 'total', 'n_neg',
        'phi'}``. Exceptions — notably ``KeyboardInterrupt`` — propagate,
        so a GUI can use it to stop between slices. ``None`` (default) is
        a no-op.
    orientation_delta : float or None, default None
        Opt-in axial edge-monotonicity rows (``dvfopt.core.marching._mono_rows``)
        appended to every sweep and mop LP; ``None`` = off (byte-identical
        to the pre-rows pipeline).
    checkpoint_dir : str or path-like or None, default None
        Make the run resumable: after every repaired sweep slice the output
        is mirrored to ``<dir>/field.npy`` (a memmap, so only that slice is
        written) and ``<dir>/state.json`` records the progress; the end of
        the run (after the mop) marks the checkpoint ``done``. Calling again
        with the same input, knobs and directory resumes from the last
        finished slice — or, on a ``done`` checkpoint, just reloads the
        result. A checkpoint from a different input / threshold / rows knob /
        origin raises ``ValueError``. ``None`` = no checkpointing.
    callback_copies : bool, default True
        When True, each progress event carries an independent snapshot
        (``'phi'`` is a copy of the output buffer at emit time), so
        events kept across calls do not alias each other. Set False to
        restore the zero-copy behaviour: ``'phi'`` is then the live
        mutable output buffer — cheaper (no per-event ``(3,D,H,W)``
        copy), but every retained event sees later mutations, so copy
        inside the callback if you keep it.

    Returns
    -------
    phi_out : ndarray
    report : Correct25DReport
    """
    phi = np.asarray(phi)
    if phi.ndim != 4 or phi.shape[0] != 3:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')
    D, H, W = phi.shape[1:]
    if D < 2 or H < 3 or W < 3:
        raise ValueError(f'phi must have D>=2, H>=3, W>=3, got {phi.shape}')

    # ---- 2.5D precondition: finite field, dz identically zero ----
    from dvfopt.core.marching._precondition import require_25d_input

    require_25d_input(phi, dz_tol)

    # Operate on a float64 copy — never mutate the caller's array; never write
    # dz. `phi` itself serves as the L1 reference, so no second copy is kept
    # (a full volume can be gigabytes).
    out = np.array(phi, dtype=np.float64, copy=True)

    # Lazy imports of solver deps (keep top-level import cheap).
    from dvfopt.core.marching._marching_25d import march_slice

    thr3 = threshold + 1e-4
    thr2 = threshold

    t0 = time.time()
    stages = []

    # ---- Stats + origin selection ----
    # `mv` is the per-cube min simplex (3D) volume of the whole volume; with dz == 0
    # its z-slab `mv[z]` equals the (z, z+1) inter-layer min-volume, so the
    # per-inter-layer fold counts come from this single measurement.
    mv, n_neg_in, n_below_in, min_T_in = _stats(out, threshold)

    if origin == 'auto':
        counts = (mv < 0).sum(axis=(1, 2))  # per-inter-layer folds (D-1,)
        origin_idx = int(np.argmin(counts))
    else:
        origin_idx = int(origin)
        if not (0 <= origin_idx < D):
            raise ValueError(f'origin {origin_idx} out of range [0, {D})')
        # counts are only needed for the verbose origin line here.
        counts = (mv < 0).sum(axis=(1, 2)) if verbose else None

    if verbose:
        adj = [int(counts[i]) for i in (origin_idx - 1, origin_idx) if 0 <= i < counts.shape[0]]
        log_info(f'[25d origin] z={origin_idx} (folds={max(adj)})')

    # Nothing to do only when the field already meets the sweep's own target
    # (min vol >= thr3): the sweep targets thr3 > threshold, so a fold-free
    # field with cubes in [threshold, thr3) still has work to do.
    if n_neg_in == 0 and n_below_in == 0 and min_T_in >= thr3 - 1e-9:
        if verbose:
            log_info('[25d] already strictly feasible; no-op')
        stages.append(dict(stage='noop', n_neg=0, min_T=min_T_in, wall_s=0.0))
        return out, _finalize(
            out,
            phi,
            threshold,
            n_neg_in,
            origin_idx,
            stages,
            t0,
            stats=(n_neg_in, n_below_in, min_T_in),
        )

    # Parallelism seam: inject the shared pool only when actually parallel.
    pool_map = None
    if n_workers > 1:
        from dvfopt.core._pool import pool_map as _pm

        pool_map = _pm

    march_kw = dict(
        thr3=thr3,
        thr2=thr2,
        orientation_delta=orientation_delta,
        mu=mu,
        pad=pad,
        dil=dil,
        max_rounds=max_rounds,
        max_box=max_box,
        n_workers=n_workers,
        pool_map=pool_map,
    )

    # ---- Bidirectional sweep outward from the origin ----
    ts = time.time()

    def _emit_progress(phase, index, total, n_neg):
        if progress_callback is not None:
            # callback_copies=True: snapshot so retained events don't alias
            # the live buffer; False: zero-copy (see docstring caveat).
            phi_evt = out.copy() if callback_copies else out
            progress_callback(
                {
                    'phase': phase,
                    'index': index,
                    'total': total,
                    'n_neg': int(n_neg),
                    'phi': phi_evt,
                }
            )

    # Up: repair slice z against frozen slice z-1 (cur is the upper layer),
    # then down: repair slice z against frozen slice z+1 (cur is the lower
    # layer). The down sweep starts AT the origin: the origin slice is
    # repaired against its already-repaired upper neighbour, matching the
    # validated research sweep. An origin at the top of the volume has no
    # upper neighbour and cannot be repaired — it stays frozen then.
    down_start = origin_idx if origin_idx + 1 < D else origin_idx - 1
    order = [(z, True) for z in range(origin_idx + 1, D)]
    order += [(z, False) for z in range(down_start, -1, -1)]

    # ---- Resumable checkpoint (opt-in) ----
    ck = state = None
    n_done = 0
    if checkpoint_dir is not None:
        meta = dict(threshold=threshold, orientation_delta=orientation_delta, origin=origin_idx)
        ck, state = _checkpoint_open(checkpoint_dir, phi, meta)
        if state['stage'] == 'done':
            out[1:3] = ck[1:3]
            if verbose:
                log_info(f'[25d resume] finished run reloaded from {checkpoint_dir}')
            stages.append(dict(stage='resumed', n_neg=-1, min_T=float('nan'), wall_s=0.0))
            return out, _finalize(out, phi, threshold, n_neg_in, origin_idx, stages, t0)
        n_done = int(state['n_done'])
        for z, _ in order[:n_done]:
            out[1:3, z] = ck[1:3, z]
        if verbose and n_done:
            log_info(f'[25d resume] {n_done}/{len(order)} sweep slices from {checkpoint_dir}')

    for k, (z, up) in enumerate(order):
        if k < n_done:
            continue
        frozen_sl = out[1:3, z - 1 if up else z + 1]
        cur, n_before, n_after = march_slice(frozen_sl, out[1:3, z], up, **march_kw)
        out[1:3, z] = cur
        if ck is not None:
            ck[1:3, z] = cur
            ck.flush()
            _checkpoint_save(checkpoint_dir, state, n_done=k + 1)
        _emit_progress('sweep', k + 1, D, n_after)
        if verbose:
            log_info(f'[25d {"up" if up else "dn"} z={z}] folds {n_before}->{n_after}')

    _, n_neg_sweep, n_below_sweep, min_sweep = _stats(out, threshold)
    stages.append(dict(stage='sweep', n_neg=n_neg_sweep, min_T=min_sweep, wall_s=time.time() - ts))
    if verbose:
        log_info(f'[25d sweep] n_neg={n_neg_sweep} min_T={min_sweep:+.5f}')

    # ---- Optional final mop ----
    # Gate on the below-threshold count (which strictly contains the
    # negatives): the mop now repairs sub-threshold cubes too, not just folds.
    stats = (n_neg_sweep, n_below_sweep, min_sweep)
    if mop and 0 < n_below_sweep <= mop_max_folds:
        from dvfopt.core.marching._mop_interior_3d import mop_interior_3d

        ts = time.time()
        # copy=False: the pipeline owns `out`; thr2/mu match the sweep's.
        out, info = mop_interior_3d(
            out,
            threshold=threshold,
            thr2=thr2,
            mu=mu,
            copy=False,
            verbose=verbose,
            orientation_delta=orientation_delta,
        )
        n_neg_mop = info['n_neg_after']
        stages.append(
            dict(stage='mop', n_neg=n_neg_mop, min_T=info['min_T_after'], wall_s=time.time() - ts)
        )
        _emit_progress('mop', 1, 1, n_neg_mop)
        if verbose:
            log_info(f'[25d mop] n_neg {n_neg_sweep}->{n_neg_mop}')
        # The mop already measured its final state; reuse it (the report's
        # n_below semantics — mv < threshold - 1e-5 — come from
        # `n_below_report_after`, measured exactly that way by the mop).
        stats = (n_neg_mop, info['n_below_report_after'], info['min_T_after'])
        if ck is not None:
            ck[1:3] = out[1:3]  # the mop moves voxels anywhere; mirror it whole

    if ck is not None:
        ck.flush()
        _checkpoint_save(checkpoint_dir, state, stage='done')

    return out, _finalize(out, phi, threshold, n_neg_in, origin_idx, stages, t0, stats=stats)


__all__ = ['Correct25DReport', 'correct_dvf_25d']
