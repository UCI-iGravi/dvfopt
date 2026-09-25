"""The pin chain: DVF-only preprocessing + 2D engine + 2.5D marching for a
Laplacian-interpolated volume — ``correct_dvf_pins``.

A field built by Laplacian interpolation of landmark correspondences folds
mostly because some landmarks CONTRADICT each other (two nearby pins whose
displacements differ by more than their separation must cross). The chain
removes that information from the field itself, then certifies the residual:

1. **pin read** — :func:`dvfopt.dvf.pins.detect_pins` on the raw field, at
   ``tau`` (``'auto'`` = :func:`dvfopt.dvf.pins.auto_tau`);
2. **pairwise drop** — pins ``p, q`` within ``radius`` with
   ``|d_p - d_q| > c |p - q|`` violate; a greedy vertex cover of the violation
   graph is dropped (:func:`dvfopt.dvf.pins.inconsistent_pins`);
3. **harmonic re-fill** — the in-plane channels re-solved from the KEPT pins
   (:func:`dvfopt.dvf.refill.harmonic_refill`; AMG with ``pyamg``);
4. **per-slice 2D windowed engine** — bilinear + ``isqp_windowed`` + L2 on
   every z (the :class:`dvfopt.DVFopt` facade; needs ``osqp``);
5. **2.5D marching** — :func:`dvfopt.correct_dvf_25d`.

The certificate of record is simplex (3D). The report also counts slices that
still carry a sub-pixel BILINEAR 2D fold — diagnostic only: a final bilinear
pass after the 2.5D stage was measured to push tets back under the 3D margin
(the two certificates are about different interpolants), so there is none.

Measured (2026-09-24, 7-brain cohort, ``laplacian_exterior`` fields
``(3, 528, 320, 456)``): 6 of 7 brains from 865k-988k folds to 0 folds / 0
best-diagonal floor / min +0.0101, landmark residual ~1 px median, 25-37 min per
volume on a contended box. B0304 does not certify — its sections carry per-slice
landmark offsets, a data defect the pairwise drop cannot fix (see
:mod:`dvfopt.dvf.pins`). The library never reads correspondence files; a
landmark-residual column is a benchmark-side diagnostic.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from dvfopt._logging import vlog


@dataclass
class PinChainReport:
    """Outcome of :func:`correct_dvf_pins`. Fold counts are per cube under the
    simplex (3D) min tet volume; ``feasible`` = no cube below ``threshold``."""

    feasible: bool
    tau: float
    n_pins: int
    n_violating_pairs: int
    n_dropped: int
    n_kept: int
    n_neg_in: int  # cubes < 0 on the input
    n_below_in: int  # cubes < threshold on the input
    min_T_in: float
    n_neg_out: int
    n_below_out: int
    min_T_out: float
    n_neg_best_diag_out: int  # cubes <= threshold under every main diagonal (the floor)
    bilinear_fold_slices: int  # diagnostic only (see the module docstring)
    l2_from_input: float
    moved_frac: float
    wall_s: float
    stages: list = field(default_factory=list)  # [(name, wall_s), ...]


def _census(phi, threshold):
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    v = six_tet_min_volume_3d(np.ascontiguousarray(phi, dtype=np.float64))
    return int((v < 0).sum()), int((v < threshold).sum()), float(v.min())


def _bilinear_fold_slices(phi, threshold):
    from dvfopt.jacobian.injectivity_radius import cell_min_jdet_2d

    # bilinear triangle area = corner Jdet / 2, the 'bilinear' constraint's scale
    return sum(
        bool((cell_min_jdet_2d(phi[1:, z]) / 2 < threshold).any()) for z in range(phi.shape[1])
    )


def correct_dvf_pins(
    phi,
    *,
    tau='auto',
    c: float = 1.0,
    radius: float = 60.0,
    threshold: float = 0.01,
    n_workers: int = 1,
    refill_rtol: float = 1e-4,
    run_2d: bool = True,
    run_25d: bool = True,
    checkpoint_dir=None,
    verbose: int = 0,
):
    """Run the pin chain on a Laplacian-interpolated ``(3, D, H, W)`` field with ``dz == 0``.

    Parameters
    ----------
    phi : ndarray ``(3, D, H, W)``, channels ``[dz, dy, dx]``; ``dz`` must be 0.
        Never mutated.
    tau : ``'auto'`` or float
        Pin threshold on ``source_strength``; ``'auto'`` = ``max(0.7, 10 * p98)``.
    c, radius : pairwise test ``|d_p - d_q| > c |p - q|`` over pins within ``radius``.
    threshold : simplex (3D) feasibility threshold, also the 2D engine's.
    n_workers : process pool size for the per-slice 2D stage and the 2.5D
        stage. Keep it small (2-4); on spawn platforms guard the calling
        script under ``if __name__ == '__main__':``.
    refill_rtol : CG tolerance of the re-fill.
    run_2d, run_25d : stop the chain early (the report still censuses the output).
    checkpoint_dir : resumable run. The re-fill is mirrored to
        ``<dir>/field.npy`` (with its pin counts in ``state.json``), the 2D
        stage checkpoints per slice under ``<dir>/2d`` and the 2.5D stage per
        sweep slice under ``<dir>/25d``; re-running with the same input, knobs
        and directory skips the finished work. A mismatch raises ``ValueError``.
    verbose : ``>= 1`` logs stage progress through the ``dvfopt`` logger.

    Returns
    -------
    phi_out : ndarray ``(3, D, H, W)`` float64
    report : PinChainReport
    """
    from dvfopt.core.marching._precondition import require_25d_input
    from dvfopt.dvf.pins import auto_tau, detect_pins, inconsistent_pins
    from dvfopt.dvf.refill import harmonic_refill

    phi = np.asarray(phi)
    if phi.ndim != 4 or phi.shape[0] != 3:
        raise ValueError(f'phi must have shape (3, D, H, W), got {phi.shape}')
    require_25d_input(phi)
    t0 = time.time()
    stages = []

    ck = None
    if checkpoint_dir is not None:
        from dvfopt.checkpoint import RunCheckpoint

        meta = dict(pipeline='pins', tau=tau, c=c, radius=radius, refill_rtol=refill_rtol)
        ck = RunCheckpoint(checkpoint_dir, phi, meta, slab=lambda _u: Ellipsis).open()

    t = time.time()
    if ck is not None and ck.is_done('refill'):
        out = np.array(ck.field)
        pin_row = ck.rows['refill']
        vlog(verbose, 1, f'[pins] re-fill reloaded from {checkpoint_dir}')
    else:
        tau_used = auto_tau(phi) if tau == 'auto' else float(tau)
        pins = detect_pins(phi, tau_used)
        if not pins.any():
            raise ValueError(
                f'no pins at tau={tau_used:.3g}: is this a Laplacian-interpolated field?'
            )
        coords, drop, bad = inconsistent_pins(phi, pins, c, radius)
        kept = np.zeros(pins.shape, bool)
        kept[tuple(coords[~drop].T)] = True
        pin_row = dict(
            tau=tau_used,
            n_pins=len(coords),
            n_violating_pairs=len(bad),
            n_dropped=int(drop.sum()),
            n_kept=int((~drop).sum()),
        )
        vlog(verbose, 1, f'[pins] {pin_row}')
        out = harmonic_refill(phi.astype(np.float64), kept, rtol=refill_rtol, verbose=verbose)
        if ck is not None:
            ck.mark('refill', out, row=pin_row)
    stages.append(('refill', time.time() - t))

    def _sub(name):
        return None if checkpoint_dir is None else os.path.join(checkpoint_dir, name)

    if run_2d:
        from dvfopt.unified import DVFopt

        t = time.time()
        out = (
            DVFopt(
                constraint='bilinear',
                solver='isqp_windowed',
                objective='l2',
                threshold=threshold,
                n_workers=n_workers,
                verbose=verbose,
                record_history=False,
                checkpoint_dir=_sub('2d'),
            )
            .fit(out)
            .corrected
        )
        stages.append(('2d', time.time() - t))
        vlog(verbose, 1, f'[pins] 2D stage {stages[-1][1]:.0f} s')

    if run_25d:
        from dvfopt.pipeline_25d import correct_dvf_25d

        t = time.time()
        out, _ = correct_dvf_25d(
            out,
            threshold=threshold,
            n_workers=n_workers,
            verbose=verbose,
            checkpoint_dir=_sub('25d'),
        )
        stages.append(('25d', time.time() - t))
        vlog(verbose, 1, f'[pins] 2.5D stage {stages[-1][1]:.0f} s')
    if ck is not None:
        ck.finish()

    from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal

    t = time.time()
    n_neg_in, n_below_in, min_in = _census(phi, threshold)
    n_neg_out, n_below_out, min_out = _census(out, threshold)
    diff = out - phi
    n_best = n_neg_best_diagonal(out, threshold)
    bil = _bilinear_fold_slices(out, threshold)
    l2 = float(np.linalg.norm(diff.ravel()))
    moved = float((np.abs(diff).max(axis=0) > 1e-6).mean())
    stages.append(('census', time.time() - t))
    report = PinChainReport(
        feasible=n_below_out == 0,
        **pin_row,
        n_neg_in=n_neg_in,
        n_below_in=n_below_in,
        min_T_in=min_in,
        n_neg_out=n_neg_out,
        n_below_out=n_below_out,
        min_T_out=min_out,
        n_neg_best_diag_out=n_best,
        bilinear_fold_slices=bil,
        l2_from_input=l2,
        moved_frac=moved,
        wall_s=time.time() - t0,
        stages=list(stages),
    )
    vlog(
        verbose,
        1,
        f'[pins] folds {n_below_in} -> {n_below_out} (< {threshold}), '
        f'min {min_in:+.4f} -> {min_out:+.4f}, floor {report.n_neg_best_diag_out}',
    )
    return out, report


__all__ = ['PinChainReport', 'correct_dvf_pins']
