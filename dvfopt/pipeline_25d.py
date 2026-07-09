"""End-to-end 2.5D marching fold-elimination orchestrator: ``correct_dvf_25d``.

Wires the two ported core modules —
:mod:`dvfopt.core.marching._marching_25d` (``layer_min_v`` / ``march_slice``)
and :mod:`dvfopt.core.marching._mop_interior_3d` (``mop_interior_3d``) — into
ONE reproducible call that drives a folded, in-plane-only 3D displacement
field toward strict 6-tet feasibility.

The 2.5D precondition
---------------------
The inter-layer 6-tet volume of a ``(lower, upper)`` layer pair depends only
on the two slices' in-plane displacement (``dy``/``dx``); the through-plane
channel ``dz`` must be identically zero. Run per-slice 2D correction first
(which yields ``dz == 0``) before calling this pipeline. ``dz`` (``phi[0]``)
is validated up front and is NEVER written by this pipeline.

The algorithm
-------------
Field layout ``(3, D, H, W)`` = ``[dz, dy, dx]``. A marching "slice" is
``phi[1:3, z]`` -> a ``(2, H, W)`` array ``[dy, dx]``.

1. Precondition check: ``dz`` must be zero (raise otherwise).
2. Origin auto-selection: pick the inter-layer with the FEWEST folds as the
   frozen seed, so no slice is ever cold-started against raw data.
3. Bidirectional sweep outward from the origin: each slice is repaired
   against its already-repaired neighbour with ``march_slice``.
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

from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d


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


def _stats(phi, threshold):
    mv = six_tet_min_volume_3d(phi)
    n_neg = int((mv <= 0).sum())
    n_below = int((mv < threshold - 1e-5).sum())
    return mv, n_neg, n_below, float(mv.min())


def _finalize(cur, phi_in, threshold, n_neg_in, origin, stages, t0):
    # `phi_in` is the caller's (never-mutated) array — used directly as the L1
    # reference rather than keeping a second full-volume copy around.
    _, n_neg, n_below, min_T = _stats(cur, threshold)
    return Correct25DReport(
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
):
    """Drive a folded, in-plane-only 3D DVF toward strict 6-tet feasibility.

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
        The frozen seed slice, never itself repaired. ``'auto'`` picks the
        slice bordering the mildest (fewest-fold) inter-layer.
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

    # ---- 2.5D precondition: dz must be identically zero ----
    if np.abs(phi[0]).max() > dz_tol:
        raise ValueError(
            'correct_dvf_25d requires the through-plane channel dz (phi[0]) '
            'to be identically zero: the 2.5D inter-layer 6-tet math depends '
            'only on adjacent slices\' in-plane displacement (dy/dx). '
            f'Found max|dz|={float(np.abs(phi[0]).max()):.3e} > dz_tol={dz_tol:.1e}. '
            'Run per-slice 2D correction first (which yields dz == 0) before '
            'calling this pipeline.'
        )

    # Operate on a float64 copy — never mutate the caller's array; never write
    # dz. `phi` itself serves as the L1 reference, so no second copy is kept
    # (a full volume can be gigabytes).
    out = np.array(phi, dtype=np.float64, copy=True)

    # Lazy imports of solver deps (keep top-level import cheap).
    from dvfopt.core.marching._marching_25d import layer_min_v, march_slice

    thr3 = threshold + 1e-4
    thr2 = threshold

    t0 = time.time()
    stages = []

    # ---- Stats + origin selection ----
    _, n_neg_in, n_below_in, min_T_in = _stats(out, threshold)

    # Per-inter-layer fold counts (D-1 of them).
    counts = [
        int((layer_min_v(out[1:3, z], out[1:3, z + 1]) < 0).sum())
        for z in range(D - 1)
    ]
    if origin == 'auto':
        origin_idx = int(np.argmin(counts)) if counts else 0
    else:
        origin_idx = int(origin)
        if not (0 <= origin_idx < D):
            raise ValueError(f'origin {origin_idx} out of range [0, {D})')

    if verbose:
        seed_folds = counts[min(origin_idx, len(counts) - 1)] if counts else 0
        print(f'[25d origin] z={origin_idx} (folds={seed_folds})', flush=True)

    # Nothing to do only when there are no folds AND nothing sits below the
    # threshold — the sweep targets thr3 > threshold, so a fold-free field with
    # sub-threshold cubes still has work to do (and `feasible` requires both).
    if n_neg_in == 0 and n_below_in == 0:
        if verbose:
            print('[25d] already strictly feasible; no-op', flush=True)
        stages.append(dict(stage='noop', n_neg=0, min_T=min_T_in, wall_s=0.0))
        return out, _finalize(out, phi, threshold, n_neg_in, origin_idx, stages, t0)

    # Parallelism seam: inject the shared pool only when actually parallel.
    pool_map = None
    if n_workers > 1:
        from dvfopt.core._pool import pool_map as _pm

        pool_map = _pm

    march_kw = dict(
        thr3=thr3,
        thr2=thr2,
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

    # Up: repair slice z against frozen slice z-1 (cur is the upper layer).
    for z in range(origin_idx + 1, D):
        frozen_sl = out[1:3, z - 1]
        cur_sl = out[1:3, z]
        cur, n_before, n_after = march_slice(frozen_sl, cur_sl, True, **march_kw)
        out[1:3, z] = cur
        if verbose:
            print(f'[25d up z={z}] folds {n_before}->{n_after}', flush=True)

    # Down: repair slice z against frozen slice z+1 (cur is the lower layer).
    for z in range(origin_idx - 1, -1, -1):
        frozen_sl = out[1:3, z + 1]
        cur_sl = out[1:3, z]
        cur, n_before, n_after = march_slice(frozen_sl, cur_sl, False, **march_kw)
        out[1:3, z] = cur
        if verbose:
            print(f'[25d dn z={z}] folds {n_before}->{n_after}', flush=True)

    _, n_neg_sweep, _, min_sweep = _stats(out, threshold)
    stages.append(
        dict(stage='sweep', n_neg=n_neg_sweep, min_T=min_sweep, wall_s=time.time() - ts)
    )
    if verbose:
        print(f'[25d sweep] n_neg={n_neg_sweep} min_T={min_sweep:+.5f}', flush=True)

    # ---- Optional final mop ----
    if mop and 0 < n_neg_sweep <= mop_max_folds:
        from dvfopt.core.marching._mop_interior_3d import mop_interior_3d

        ts = time.time()
        out, info = mop_interior_3d(out, threshold=threshold, verbose=verbose)
        _, n_neg_mop, _, min_mop = _stats(out, threshold)
        stages.append(
            dict(stage='mop', n_neg=n_neg_mop, min_T=min_mop, wall_s=time.time() - ts)
        )
        if verbose:
            print(f'[25d mop] n_neg {n_neg_sweep}->{n_neg_mop}', flush=True)

    return out, _finalize(out, phi, threshold, n_neg_in, origin_idx, stages, t0)


__all__ = ['Correct25DReport', 'correct_dvf_25d']
