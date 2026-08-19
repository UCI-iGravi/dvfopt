"""Cluster-localised LP/SLP for the 2-triangle constraint.

The direct LP at full B0039 slice scale (320x456 -> ~290k decision
variables) is too slow for practical use (>12 min wall on z=12). This
module decomposes the slice into connected fold clusters, runs SLP on
each cluster's padded crop with frozen-edge corners, and splices the
interior back into the slice.

The cluster decomposition pattern matches
``notebooks/manuscript/_run_2d_clusters.py::enumerate_clusters_2d``
but the inner solve is our :func:`slp_iter` instead of the analytic
SLSQP. Each cluster typically yields an LP of <2000 variables -- HiGHS
solves these in tens of milliseconds.

Triggers spec fallback row 5 (``cluster_lp``).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from scipy.ndimage import label as cc_label

from dvfopt._logging import log_info, log_warning
from dvfopt.core.slp.lp_direct_2tri import slp_iter
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

BBOX_PAD = 4  # cell-units of padding around each cluster's bbox
MERGE_DILATION_BASE = 2
MAX_OUTER_ITERS = 3
DEFAULT_PARALLEL_WORKERS = 1  # sequential by default; user can opt in to processes


def _solve_cluster_worker(args):
    """Top-level (picklable) worker for process-pool parallelism.

    Threads aren't safe (scipy linprog isn't thread-safe in our build);
    processes run on the package-shared pre-warmed pool
    (``dvfopt.core._pool``), so the Windows-spawn cost is paid once per
    session instead of once per slice.
    """
    phi_crop, inner_threshold, inner_trust_radius_0, inner_max_iter, inner_seed = args
    phi_corr, _ = slp_iter(
        phi_crop,
        threshold=inner_threshold,
        trust_radius_0=inner_trust_radius_0,
        max_iter=inner_max_iter,
        seed=inner_seed,
    )
    return phi_corr


def _partition_clusters_nonoverlapping(clusters):
    """Greedy graph-colour: pack clusters into rounds where no two in
    the same round have overlapping or adjacent bboxes (strict 1-cell
    gap). Mirrors the manuscript pipeline's pattern so concurrent
    splices into the slice are race-free.
    """
    rounds = []
    for c in clusters:
        y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
        placed = False
        for r in rounds:
            ok = True
            for c2 in r:
                if not (y1 < c2['y0'] or c2['y1'] < y0 or x1 < c2['x0'] or c2['x1'] < x0):
                    ok = False
                    break
            if ok:
                r.append(c)
                placed = True
                break
        if not placed:
            rounds.append([c])
    return rounds


def _boxes_conflict(a, b):
    """True if two cluster bboxes overlap or are adjacent (<1-cell gap).

    Same predicate ``_partition_clusters_nonoverlapping`` uses; factored
    out for the continuous (as-completed) scheduler's admission test.
    """
    return not (a['y1'] < b['y0'] or b['y1'] < a['y0'] or a['x1'] < b['x0'] or b['x1'] < a['x0'])


def _cell_min_T(phi_2hw: np.ndarray) -> np.ndarray:
    """Per-cell ``min(T1, T2)`` array for a ``(2, H, W)`` slice."""
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    return np.minimum(T1, T2)


def _fold_clusters(
    phi_2hw: np.ndarray,
    merge_dilation: int = MERGE_DILATION_BASE,
    target_threshold: float = 0.0,
    cell_min: np.ndarray | None = None,
):
    """Return list of cluster dicts for every connected component of
    cells where ``min(T1, T2) < target_threshold``.

    Default ``target_threshold=0`` finds geometric folds (cells with at
    least one flipped triangle). Setting it to the user threshold
    (e.g. 0.01) also catches barely-infeasible cells from splice noise
    in later outer rounds, so the cluster loop can sweep them up
    without falling back to the expensive global polish step.

    ``cell_min`` optionally provides a precomputed ``min(T1, T2)`` array
    for ``phi_2hw`` (from :func:`_cell_min_T`), skipping the full-slice
    triangle evaluation."""
    if cell_min is None:
        cell_min = _cell_min_T(phi_2hw)
    fold_mask = cell_min < target_threshold
    if not fold_mask.any():
        return []
    merged = (
        binary_dilation(fold_mask, iterations=merge_dilation) if merge_dilation > 0 else fold_mask
    )
    labels, n_comp = cc_label(merged)
    Hc, Wc = fold_mask.shape
    bboxes = find_objects(labels)
    out = []
    for idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        cy0, cy1 = bbox[0].start, bbox[0].stop
        cx0, cx1 = bbox[1].start, bbox[1].stop
        y0 = max(0, cy0 - BBOX_PAD)
        y1 = min(Hc, cy1 + BBOX_PAD)
        x0 = max(0, cx0 - BBOX_PAD)
        x1 = min(Wc, cx1 + BBOX_PAD)
        comp_cells = int(((labels[y0:y1, x0:x1] == (idx + 1)) & fold_mask[y0:y1, x0:x1]).sum())
        out.append(
            {
                'y0': y0,
                'y1': y1,
                'x0': x0,
                'x1': x1,
                'crop_cells_y': y1 - y0,
                'crop_cells_x': x1 - x0,
                'n_fold_cells': comp_cells,
            }
        )
    # Sort large clusters first.
    out.sort(key=lambda c: -c['n_fold_cells'])
    return out


def _splice_interior(phi_full: np.ndarray, c: dict, phi_crop_corrected: np.ndarray):
    """Splice only the interior corners of the cluster crop back into the
    full slice. The outer one-corner ring is left frozen so neighbouring
    clusters' boundary corners stay consistent."""
    y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
    crop_h = y1 - y0 + 1  # corner-grid height
    crop_w = x1 - x0 + 1
    if crop_h < 3 or crop_w < 3:
        # Too small for an interior — fall back to splicing the whole crop.
        phi_full[:, y0 : y0 + crop_h, x0 : x0 + crop_w] = phi_crop_corrected
        return
    # Interior corners only.
    phi_full[:, y0 + 1 : y0 + crop_h - 1, x0 + 1 : x0 + crop_w - 1] = phi_crop_corrected[
        :, 1:-1, 1:-1
    ]


def cluster_slp_iter(
    phi_in_2hw: np.ndarray,
    *,
    threshold: float = 0.01,
    inner_seed: str = 'm14_fast',
    inner_max_iter: int = 10,
    inner_trust_radius_0: float = 0.5,
    max_outer_iters: int = MAX_OUTER_ITERS,
    merge_dilation: int = MERGE_DILATION_BASE,
    final_global_polish: bool = True,
    n_workers: int = DEFAULT_PARALLEL_WORKERS,
    scheduler: str = 'subround',
    verbose: int = 0,
):
    """Per-cluster SLP with frozen-edge splice.

    Repeatedly: enumerate fold clusters, solve SLP on each cluster's
    padded crop (with the inner ``slp_iter``), splice interior back into
    the slice. Outer loop terminates when no folds remain OR no progress
    after one round.

    Parameters
    ----------
    phi_in_2hw : (2, H, W) float64
    threshold : float
    inner_seed : {'m10', 'm14', 'harmonic'}
        Seed kind passed to ``slp_iter`` per cluster.
    inner_max_iter : int
        SLP iteration cap per cluster.
    inner_trust_radius_0 : float
    max_outer_iters : int
    merge_dilation : int
        Cluster CCL merge-dilation in cell units.

    Returns
    -------
    phi_out_2hw : (2, H, W) float64
    info : dict with per-round + per-cluster bookkeeping.
    """
    t0 = time.time()
    # astype(copy=True) is the default: already a fresh, mutation-safe
    # array even when the input is float64 — no extra .copy() needed.
    phi_out = phi_in_2hw.astype(np.float64)
    info = {
        'rounds': [],
        'inner_seed': inner_seed,
        'total_cluster_solves': 0,
    }
    # Parallel rounds use the package-shared, pre-warmed process pool
    # (dvfopt.core._pool). A per-call ProcessPoolExecutor paid the
    # Windows-spawn cost (~1-2 s/worker) on EVERY slice — 10-25 min of
    # pure spawn overhead over a 528-slice volume. The shared pool is
    # created once per session and owned by the pool module (atexit
    # teardown); it must NEVER be shut down here.

    # Cell-min triangle-area array valid for the CURRENT phi_out, or None
    # if phi_out changed since it was computed. Threading it through the
    # loop avoids re-evaluating the full-slice triangle areas 2-3x per
    # round (cluster enumeration, pre_n_neg, polish trigger, final stats).
    cur_cell_min = None

    for outer_it in range(max_outer_iters):
        # First outer iter: target only actual folds (min_T <= 0). On
        # subsequent rounds: target full strict-feasibility (min_T <
        # threshold), so splice-noise cells get swept up by the cluster
        # loop instead of triggering the expensive global polish step.
        target = 0.0 if outer_it == 0 else (threshold - 1e-5)
        if cur_cell_min is None:
            cur_cell_min = _cell_min_T(phi_out)
        clusters = _fold_clusters(
            phi_out,
            merge_dilation=merge_dilation,
            target_threshold=target,
            cell_min=cur_cell_min,
        )
        if not clusters:
            info['rounds'].append({'outer': outer_it, 'n_clusters': 0, 'reason': 'feasible'})
            break

        pre_n_neg = int((cur_cell_min <= 0).sum())
        round_runs = []

        # Inner threshold has a slight margin (threshold + 1e-4) so
        # post-splice numerical noise doesn't drop min_T below the
        # user's target and force a polish. 1e-4 was the empirical
        # sweet spot on B0039 z=12 (L1 within 0.1% of unmargined LP
        # optimum). Larger margins (5e-3, 1e-2) sweep showed no help
        # on sparse slices and worse L1 on dense.
        inner_threshold = threshold + 1e-4

        if n_workers > 1 and scheduler == 'continuous':
            # Continuous (as-completed) scheduler: instead of barrier
            # sub-rounds (where a slow large cluster idles the workers that
            # finished small ones), keep the pool full — admit any pending
            # cluster that doesn't conflict (overlap/adjacent) with an
            # in-flight one, and splice each as it completes. Crop at
            # admission time so a cluster adjacent to a just-spliced
            # neighbour still sees the update in its frozen ring. Removes
            # the inter-sub-round straggler idle (the recoverable part of
            # the ~16% serial fraction measured on B0039 slices).
            from concurrent.futures import FIRST_COMPLETED
            from concurrent.futures import wait as _wait
            from concurrent.futures.process import BrokenProcessPool

            from dvfopt.core._pool import _shutdown_if_current, get_pool

            pool = get_pool(n_workers)  # shared pre-warmed pool (module-owned)
            pending = list(clusters)  # largest-first
            inflight = {}  # future -> cluster
            try:
                while pending or inflight:
                    # Greedily admit non-conflicting clusters.
                    while len(inflight) < n_workers:
                        pick = None
                        for i, c in enumerate(pending):
                            if not any(_boxes_conflict(c, c2) for c2 in inflight.values()):
                                pick = i
                                break
                        if pick is None:
                            break
                        c = pending.pop(pick)
                        y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
                        phi_crop = phi_out[:, y0 : y1 + 1, x0 : x1 + 1].copy()
                        try:
                            fut = pool.submit(
                                _solve_cluster_worker,
                                (
                                    phi_crop,
                                    inner_threshold,
                                    inner_trust_radius_0,
                                    inner_max_iter,
                                    inner_seed,
                                ),
                            )
                        except BrokenProcessPool:
                            # submit() can raise if the shared pool broke after
                            # admission started; push the popped cluster back so
                            # the outer handler's serial fallback re-solves it
                            # (otherwise it would be lost from both queues).
                            pending.append(c)
                            raise
                        inflight[fut] = c
                    if not inflight:
                        break  # nothing admittable and nothing running
                    done, _ = _wait(list(inflight), return_when=FIRST_COMPLETED)
                    for fut in done:
                        # Pop only AFTER fut.result() succeeds: if the pool
                        # broke, result() raises and the cluster must stay
                        # in `inflight` so the serial fallback re-solves it.
                        result = fut.result()
                        c = inflight.pop(fut)
                        _splice_interior(phi_out, c, result)
                        info['total_cluster_solves'] += 1
                        round_runs.append({**c})
            except BrokenProcessPool:
                # A worker died mid-round. Tear the broken shared pool down
                # (only if it is still the module's current pool, so the
                # next parallel call rebuilds a fresh one) and finish the
                # remaining clusters serially in-process — a dead worker
                # must never crash the caller.
                _shutdown_if_current(pool)
                remaining = list(inflight.values()) + pending
                inflight.clear()
                pending = []
                for c in remaining:
                    y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
                    phi_crop = phi_out[:, y0 : y1 + 1, x0 : x1 + 1].copy()
                    phi_corr = _solve_cluster_worker(
                        (
                            phi_crop,
                            inner_threshold,
                            inner_trust_radius_0,
                            inner_max_iter,
                            inner_seed,
                        )
                    )
                    _splice_interior(phi_out, c, phi_corr)
                    info['total_cluster_solves'] += 1
                    round_runs.append({**c})
        elif n_workers > 1:
            # Parallel via the shared pre-warmed pool. Threads were ruled
            # out (scipy linprog not thread-safe in this build, segfaulted
            # at >=2 workers). Partition into non-overlapping rounds so
            # concurrent splices don't race. pool_map falls back to a
            # serial in-process map if the pool breaks.
            from dvfopt.core._pool import pool_map

            sub_rounds = _partition_clusters_nonoverlapping(clusters)
            for sub_round in sub_rounds:
                arg_list = []
                for c in sub_round:
                    y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
                    phi_crop = phi_out[:, y0 : y1 + 1, x0 : x1 + 1].copy()
                    arg_list.append(
                        (
                            phi_crop,
                            inner_threshold,
                            inner_trust_radius_0,
                            inner_max_iter,
                            inner_seed,
                        )
                    )
                t_c = time.time()
                if len(sub_round) > 1:
                    results = pool_map(_solve_cluster_worker, arg_list, n_workers)
                else:
                    results = [_solve_cluster_worker(arg_list[0])]
                wall_round = time.time() - t_c
                for c, phi_corr in zip(sub_round, results):
                    _splice_interior(phi_out, c, phi_corr)
                    info['total_cluster_solves'] += 1
                    round_runs.append({**c, 'wall_round': wall_round})
        else:
            for c in clusters:
                y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
                phi_crop = phi_out[:, y0 : y1 + 1, x0 : x1 + 1].copy()
                t_c = time.time()
                try:
                    phi_corr, _ = slp_iter(
                        phi_crop,
                        threshold=inner_threshold,
                        trust_radius_0=inner_trust_radius_0,
                        max_iter=inner_max_iter,
                        seed=inner_seed,
                    )
                except Exception as exc:
                    log_warning(f'cluster solve FAILED: {type(exc).__name__}: {exc}')
                    round_runs.append({**c, 'error': f'{type(exc).__name__}: {exc}'})
                    continue
                _splice_interior(phi_out, c, phi_corr)
                info['total_cluster_solves'] += 1
                round_runs.append({**c, 'wall': time.time() - t_c})

        # phi_out changed (splices) — refresh the threaded cell-min array.
        cur_cell_min = _cell_min_T(phi_out)
        T_min = cur_cell_min
        post_n_neg = int((T_min <= 0).sum())
        post_n_below_threshold = int((T_min < threshold - 1e-5).sum())
        info['rounds'].append(
            {
                'outer': outer_it,
                'n_clusters': len(clusters),
                'pre_n_neg': pre_n_neg,
                'post_n_neg': post_n_neg,
                'post_n_below_threshold': post_n_below_threshold,
                'wall': time.time() - t0,
                'cluster_runs': round_runs if verbose else None,
            }
        )
        if verbose:
            log_info(
                f'[outer {outer_it}] {len(clusters)} clusters: '
                f'n_neg {pre_n_neg} -> {post_n_neg}  '
                f'n<threshold={post_n_below_threshold}  '
                f'({time.time() - t0:.1f}s)',
            )
        # Done iff no folds AND no margin violations.
        if post_n_neg == 0 and post_n_below_threshold == 0:
            break
        # No progress on n_neg (and no margin violations being targeted)
        # → expand merge_dilation and retry once.
        if outer_it == 0 and post_n_neg >= pre_n_neg:
            merge_dilation += 1
            if merge_dilation > MERGE_DILATION_BASE + 3:
                break

    # Final global polish: if cluster passes leave residual folds (the
    # splice boundary can introduce new infeasibility), run global M14
    # **anchored to the cluster output** (not the original input). M14
    # guarantees feasibility, and its L2-refine stays close to its
    # anchor — so the total L1 vs the original input stays near the
    # cluster_slp L1 (which is much lower than global M14 on hard
    # cases), plus a small fix-up term.
    # Reuse the threaded cell-min array when it is still valid for the
    # current phi_out (it always is on every loop-exit path; the None
    # check covers the max_outer_iters == 0 edge case).
    if cur_cell_min is None:
        cur_cell_min = _cell_min_T(phi_out)
    cluster_t_min = cur_cell_min
    cluster_min_T = float(cluster_t_min.min())
    cluster_n_neg = int((cluster_t_min <= 0).sum())
    # Trigger polish only when there are still folded cells OR the
    # min_T is meaningfully under the threshold (not just kissing it
    # within numerical slack). A bare 'min_T < threshold - safety_tol'
    # trigger fires the polish on cases where the cluster pass is
    # essentially done -- and the polish costs ~3× the cluster pass.
    polish_margin = 5 * 1e-5  # 5x safety_tol
    if final_global_polish and (cluster_n_neg > 0 or cluster_min_T < threshold - polish_margin):
        if verbose:
            log_info(
                f'[polish] cluster min_T={cluster_min_T:+.4f} < threshold; '
                f'running global M14 anchored to cluster output…',
            )
        t_p = time.time()
        from dvfopt import (
            HarmonicALMRefineRepairStrategy,
            L1Objective,
            Solver,
            TriConstraint2DFullCoverage,
        )

        H, W = phi_out.shape[1:]
        # Polish uses M14-fast (skip stage 4 barrier polish) since
        # the cluster output is already L1-good — we only need
        # feasibility-restoration here, not L1 minimisation.
        # stage1_mu_schedule=() also skips m10's internal stage-3
        # log-barrier polish, matching the M14_SEED_STAGE1_MU_SCHEDULE
        # convention of the lp_direct_2tri seeds. Benched 2026-07 on
        # B0039 polish-fire cases (z=0 + z=12 with default knobs,
        # z=360 with weakened knobs): outputs byte-identical to the
        # legacy m10 default schedule (same n_neg/n_below/min_T,
        # dL1 = 0.00%), wall within noise (+-4%) — m10 skips that
        # stage anyway unless its ALM seed is already strictly
        # feasible, and m14's stage-2 l2-refine re-anchors regardless.
        # () just makes the skip explicit and consistent with the
        # seed path.
        solver = Solver(
            constraint=TriConstraint2DFullCoverage(shape=(H, W)),
            objective=L1Objective(eps=1e-4),
            strategy=HarmonicALMRefineRepairStrategy(
                polish_mu=(),
                stage1_mu_schedule=(),
            ),
            threshold=threshold,
        )
        phi_out = solver.fit(phi_out).corrected
        info['polish_wall'] = time.time() - t_p
        info['polish_fired'] = True
        # Polish changed phi_out — the final min_T must be re-evaluated.
        final_t_min = _cell_min_T(phi_out)
    else:
        info['polish_fired'] = False
        final_t_min = cluster_t_min

    info['final_min_T_exact'] = float(final_t_min.min())
    info['L1_dev'] = float(np.abs(phi_out - phi_in_2hw).sum())
    info['wall_s'] = time.time() - t0
    return phi_out, info
