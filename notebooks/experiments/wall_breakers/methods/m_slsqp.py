"""Manuscript SLSQP baseline -- cluster-based windowed iterative SLSQP.

A clean, harness-compatible reimplementation of the manuscript run
pipeline (notebooks/manuscript/_run_2d_clusters.py), without the
disk-I/O / subprocess / checkpoint machinery:

  1. Find connected fold components (cells with min(T1, T2) <= 0).
  2. Dilate by ``merge_dilation`` cells and crop the bounding box.
  3. Build a frozen-edge interior mask: only corners adjacent to a
     folded cell become unknowns; every other corner of the crop
     stays fixed.
  4. Run scipy SLSQP on the windowed problem with the L2 anchor
     ``0.5 ||phi - phi_anchor||^2`` and the 2-tri constraint.
  5. Splice the solved interior back into the slice.
  6. Re-find clusters and repeat until ``n_neg == 0`` or budget hit.

This is the *same algorithm* the manuscript run uses on each slice
(without the global retry / extra_dilation escalation, which adds
diminishing returns). Gives us per-slice timing under the same
harness so SLSQP and m10 can be compared apples-to-apples.

OOM safety: every cluster has bounded size via ``MAX_CLUSTER_*``
caps; clusters above those caps are SKIPPED (and the slice is
reported infeasible) -- this matches the manuscript runner's
behaviour on the wall slices, which it could not crack.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.ndimage import (
    label as cc_label, binary_dilation, find_objects,
    generate_binary_structure)

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

NAME = 'slsqp_windowed'
DESCRIPTION = ('Manuscript SLSQP windowed/cluster pipeline '
               '(per-cluster SLSQP with frozen-edge interior, L2 anchor)')

MAX_CLUSTER_PER_AXIS = 100      # cells per axis cap on a cluster
MAX_CLUSTER_CELLS    = 5000     # total cells cap (avoids the OOM that
                                # bare iterative_serial hits on dense
                                # wall slices)
MERGE_DILATION       = 1
L2_PASS_MAX_ITER     = 100
L2_MAX_PASSES        = 3
OUTER_MAX            = 30


def _interior_pack_unpack(phi_win, interior_mask):
    """Returns ``(pack, unpack, n_int)`` -- packs only the interior
    corners into a flat optimisation vector.
    """
    int_idx = np.argwhere(interior_mask)
    n_int = len(int_idx)
    iy, ix = int_idx[:, 0], int_idx[:, 1]

    def pack(phi):
        return np.concatenate([phi[0][iy, ix], phi[1][iy, ix]])

    def unpack(z, base):
        out = base.copy()
        out[0][iy, ix] = z[:n_int]
        out[1][iy, ix] = z[n_int:]
        return out

    return pack, unpack, n_int


def _enumerate_clusters(phi_slice, threshold):
    """Find dilated bounding-box clusters of fold cells. Returns list of
    dicts with ``(y0, y1, x0, x1, interior_mask, comp_cells)`` and
    ``skipped_too_large``.
    """
    H, W = phi_slice.shape[1], phi_slice.shape[2]
    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    fold_cells = np.minimum(T1, T2) <= threshold
    if not fold_cells.any():
        return []
    grouped = binary_dilation(fold_cells, iterations=MERGE_DILATION)
    labels, n_comp = cc_label(grouped, generate_binary_structure(2, 2))
    clusters = []
    for cid in range(1, n_comp + 1):
        comp_mask = (labels == cid) & fold_cells
        if not comp_mask.any():
            continue
        # Bounding box of the (dilated) cluster cells.
        dilated = binary_dilation(comp_mask, iterations=MERGE_DILATION)
        ys, xs = np.where(dilated)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cell_w = y1 - y0
        cell_h = x1 - x0
        skipped = (cell_w > MAX_CLUSTER_PER_AXIS
                   or cell_h > MAX_CLUSTER_PER_AXIS
                   or cell_w * cell_h > MAX_CLUSTER_CELLS)
        # Corner-frame extents (cells y0..y1 -> corners y0..y1+1).
        cy0, cy1 = y0, min(H, y1 + 1)
        cx0, cx1 = x0, min(W, x1 + 1)
        # Interior mask in corner-frame: corners adjacent to any fold cell.
        crop_cells = dilated[y0:y1, x0:x1]
        interior_mask = np.zeros((cy1 - cy0, cx1 - cx0), dtype=bool)
        # corners of cells in crop_cells
        cy_local, cx_local = np.where(crop_cells)
        interior_mask[cy_local, cx_local] = True
        interior_mask[cy_local, cx_local + 1] = True
        interior_mask[cy_local + 1, cx_local] = True
        interior_mask[cy_local + 1, cx_local + 1] = True
        # Boundary corners (touching outside the crop) stay frozen.
        interior_mask[0, :] = False
        interior_mask[-1, :] = False
        interior_mask[:, 0] = False
        interior_mask[:, -1] = False
        clusters.append({
            'y0': int(y0), 'y1': int(y1),
            'x0': int(x0), 'x1': int(x1),
            'cy0': int(cy0), 'cy1': int(cy1),
            'cx0': int(cx0), 'cx1': int(cx1),
            'interior_mask': interior_mask,
            'comp_size': int(comp_mask.sum()),
            'cluster_size': int(cell_w * cell_h),
            'skipped_too_large': bool(skipped),
        })
    return clusters


class _DeadlineReached(Exception):
    """Sentinel: scipy SLSQP callback raises this when the per-cluster
    wall-time budget is exceeded. We catch it at the minimize() boundary
    and treat the current point as the best-effort result."""


def _solve_one_cluster(phi_slice, phi_anchor, c, threshold,
                       max_iter=L2_PASS_MAX_ITER, max_passes=L2_MAX_PASSES,
                       time_budget_s=120.0):
    """Run windowed SLSQP on one cluster. Mutates phi_slice on success.
    Returns (n_iter, wall_s, status).

    Bug #2 fix: SLSQP can spin inside a single iteration when the
    constraint Jacobian is rank-deficient (degenerate active set at
    the wall). We attach a callback that checks the wall clock between
    iterations and raises ``_DeadlineReached`` once the budget is up,
    so a single cluster cannot block the whole slice for hours.
    """
    if c['skipped_too_large']:
        return 0, 0.0, 'skipped'
    phi_crop = phi_slice[:, c['cy0']:c['cy1'], c['cx0']:c['cx1']].copy()
    anc_crop = phi_anchor[:, c['cy0']:c['cy1'], c['cx0']:c['cx1']].copy()
    pack, unpack, n_int = _interior_pack_unpack(phi_crop, c['interior_mask'])
    if n_int == 0:
        return 0, 0.0, 'empty'
    z_anchor = pack(anc_crop)

    def obj(z):
        d = z - z_anchor
        return 0.5 * float(d @ d), d

    def constr(z):
        ph = unpack(z, phi_crop)
        T1, T2 = _triangle_areas_2d(ph[0], ph[1])
        return np.concatenate([T1.ravel(), T2.ravel()])

    nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
    total_nit = 0
    t0 = time.time()
    deadline = t0 + time_budget_s

    def _deadline_cb(xk):
        if time.time() > deadline:
            raise _DeadlineReached()

    for p in range(max_passes):
        if time.time() > deadline:
            break
        z_init = pack(phi_crop)
        try:
            res = minimize(obj, z_init, jac=True, method='SLSQP',
                           constraints=[nl], callback=_deadline_cb,
                           options={'maxiter': max_iter, 'disp': False})
            phi_crop = unpack(res.x, phi_crop)
            total_nit += int(res.nit)
        except _DeadlineReached:
            # SLSQP got partway -- splice in whatever the callback last saw.
            phi_slice[:, c['cy0']:c['cy1'], c['cx0']:c['cx1']] = phi_crop
            return total_nit, time.time() - t0, 'timeout'
        except Exception:
            return total_nit, time.time() - t0, 'error'
        T1, T2 = _triangle_areas_2d(phi_crop[0], phi_crop[1])
        if int((T1 <= 0).sum() + (T2 <= 0).sum()) == 0:
            phi_slice[:, c['cy0']:c['cy1'], c['cx0']:c['cx1']] = phi_crop
            return total_nit, time.time() - t0, 'feasible'
    # Did not reach feasibility but accept best-effort splice (so re-
    # iteration sees progress).
    phi_slice[:, c['cy0']:c['cy1'], c['cx0']:c['cx1']] = phi_crop
    return total_nit, time.time() - t0, 'partial'


def solve(phi_in: np.ndarray, threshold: float = 0.01, *,
          time_budget_s: float = 600.0,
          verbose: int = 0) -> dict:
    """Manuscript-style cluster SLSQP loop: enumerate, solve, repeat."""
    H, W = phi_in.shape[1], phi_in.shape[2]
    phi_slice = phi_in.copy()
    phi_anchor = phi_in.copy()  # L2 anchor = original input

    t0 = time.time()
    log = []
    for outer in range(OUTER_MAX):
        if time.time() - t0 > time_budget_s:
            break
        clusters = _enumerate_clusters(phi_slice, threshold)
        if not clusters:
            break
        n_skipped = sum(1 for c in clusters if c['skipped_too_large'])
        n_processed = 0
        for c in clusters:
            if time.time() - t0 > time_budget_s:
                break
            nit, wall, status = _solve_one_cluster(
                phi_slice, phi_anchor, c, threshold,
                time_budget_s=max(30.0, time_budget_s * 0.2))
            if status not in ('skipped', 'empty'):
                n_processed += 1
        T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
        n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        log.append(dict(outer=outer, n_clusters=len(clusters),
                         n_skipped=n_skipped,
                         n_processed=n_processed,
                         n_neg_after=n_neg,
                         min_T_after=float(min(T1.min(), T2.min())),
                         wall=time.time() - t0))
        if verbose:
            print(f'  outer={outer:2d}  n_clusters={len(clusters):3d}  '
                  f'skipped={n_skipped:2d}  n_neg={n_neg:5d}  '
                  f'min_T={min(T1.min(), T2.min()):+.5f}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
        if n_neg == 0:
            break
        if log[-1]['n_neg_after'] == (log[-2]['n_neg_after']
                                       if len(log) > 1 else None):
            # No progress on this outer pass -- stop.
            break

    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    return {'phi_out': phi_slice,
            'info': {'final_min_T': float(min(T1.min(), T2.min())),
                     'final_n_neg': int((T1 <= 0).sum() + (T2 <= 0).sum()),
                     'outer_passes': len(log),
                     'log_last3': log[-3:]}}
