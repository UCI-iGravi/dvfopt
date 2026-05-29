"""Per-slice 2D benchmark using notebook-17/18-style full-grid SLSQP
per connected fold component.

For each z-slice:
  1. Find connected components of folded cells (any of T1/T2 <= threshold).
  2. For each component:
       - bbox + border crop
       - multi-pass full-grid L2 SLSQP on the crop (no frozen edges --
         every voxel-corner is a variable, matching notebook 17/18)
       - single L1 polish pass if L2 reached n_neg_tri = 0
  3. Splice each cluster's corrected phi back into the slice.

Records both per-slice rollups and per-cluster details so the manuscript
can characterise both. Resumable via per_slice.csv + checkpoint.npz.
"""

import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label, find_objects, binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
import _bench_worker

THRESHOLD = DEFAULT_PARAMS['threshold']
ERR_TOL = DEFAULT_PARAMS['err_tol']
EPS_L1 = 1e-4

# --- Cluster geometry --------------------------------------------------
# Cluster construction:
#   1. Dilate the fold mask by MERGE_DILATION cells, so fold components
#      that are within MERGE_DILATION of each other merge into one
#      cluster (they share corners via the gap cells, so they have to
#      be solved together).
#   2. Connected-component label on the dilated mask. Each dilated
#      component is one cluster.
#   3. Bbox = bbox(dilated_component) + BBOX_PAD cells (the +1 pad
#      includes the constraint neighbours whose corners share with
#      dilated cells).
#   4. interior_mask = voxel corners adjacent to any cell in the
#      dilated component. Slice-edge corners are naturally movable
#      (no outside cell to constrain).
# MERGE_DILATION scales DOWN with residual count: dense fold regions need
# joint solving (high merge), but few residuals just need tight individual
# bboxes (low merge). Floor at 1 -- a MERGE=0 cluster is just (fold cell +
# 1-pad) = 3x3 cell bbox = only 4 movable corners x 2 channels = 8 variables.
# Severe single-cell folds (min_tri ~ -1) need more degrees of freedom; one
# more layer of dilation gives 5x5 cell bbox = ~32 movable variables.
def _merge_for_n_neg(n_neg):
    """Return MERGE_DILATION as a function of current fold-triangle count.

    Always 1. The earlier escalation to 2 for >200 folds turned out to be
    the cause of the 7 stuck slices: it artificially fused dozens-to-
    hundreds of small fold clumps into one giant unsolvable component.
    The Schwarz-tiling experiment in notebooks/experiments/ confirmed
    those slices have many small components, not one large one, and they
    all converge with dilation=1 + the existing per-cell extra_dilation
    escalation for genuine cluster-solve failure.
    """
    return 1


BBOX_PAD = 1
MAX_CLUSTER_CELLS = 2000         # crops above this are skipped (over SLSQP cap)
MAX_CLUSTER_PER_AXIS = 60
MAX_CLUSTER_OUTER_ITERS = 20
MAX_SLICE_TIME_S = 144000        # 40 h cap; per-cluster timeout is now
                                  # 36000s so the slice budget must be a
                                  # multiple of that (~4 outer-iter
                                  # retries) on dense residuals

# --- Per-cluster solver budget (notebook 17/18 style) ---------------
L2_MAX_PASSES = 15               # max L2 SLSQP passes; loop exits when n_neg=0
L2_PASS_MAX_ITER = 80
L2_PASS_TIMEOUT_S = 36000
L1_POLISH_MAX_ITER = 120
L1_POLISH_TIMEOUT_S = 36000

CHECKPOINT_EVERY = 5             # checkpoint every N slices

# --- Parallel cluster execution ----------------------------------------
# Pool of long-lived workers means each cluster pays no fresh-spawn cost.
# Clusters are batched into "rounds" where no two clusters' bboxes overlap,
# so parallel solves don't write each other's boundary corners. After each
# round, only the *interior_mask* corners are spliced back into phi_slice
# so frozen-edge corners (which the solver assumed at snapshot values) stay
# at those snapshot values.
N_PARALLEL_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PER_CLUSTER_TIMEOUT_S = 36000     # passed to ``future.result(timeout=...)``
MAX_POOL_RETRIES = 3             # rebuild a crashed worker pool this many
                                  # times before skipping the slice

OUTPUT_DIR = os.path.join(_HERE, 'output', '2d_real_full')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SLICE_CSV_PATH = os.path.join(OUTPUT_DIR, 'per_slice.csv')
CLUSTER_CSV_PATH = os.path.join(OUTPUT_DIR, 'per_cluster.csv')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.npz')
LOG_PATH = os.path.join(OUTPUT_DIR, 'run.log')

# Per-slice artefacts: the corrected slice as its own .npy, and a
# before/after deformation-grid figure.
SLICES_DIR = os.path.join(OUTPUT_DIR, 'slices')
FIGS_DIR = os.path.join(OUTPUT_DIR, 'figs')
os.makedirs(SLICES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

DATA_PATH = os.path.join(_REPO_ROOT, 'data',
                         'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

SLICE_COLUMNS = [
    'z', 'H', 'W',
    'init_n_neg_tri', 'init_min_tri',
    'n_clusters', 'n_clusters_processed', 'n_clusters_skipped',
    'n_clusters_feasible', 'n_clusters_l2_timeout',
    'after_l2_n_neg_tri', 'after_l2_min_tri',
    'after_l1_n_neg_tri', 'after_l1_min_tri',
    'init_L1', 'after_l1_L1', 'after_l1_L2',
    'total_l2_t', 'total_l1_t', 'total_t',
    'feasible', 'l1_drop_pct',
]

CLUSTER_COLUMNS = [
    'z', 'cluster_outer_it', 'cluster_id', 'y0', 'y1', 'x0', 'x1',
    'crop_cells_y', 'crop_cells_x',
    'component_cells', 'init_n_neg_tri', 'init_min_tri',
    'after_l2_n_neg_tri', 'after_l2_min_tri',
    'l2_passes_run', 'l2_total_nit', 'l2_total_t', 'l2_any_timeout',
    'after_l1_n_neg_tri', 'after_l1_min_tri',
    'l1_nit', 'l1_t', 'l1_polished', 'l1_timed_out',
    'cluster_t', 'feasible', 'skipped_too_large',
]


def init_csvs():
    for path, cols in [(SLICE_CSV_PATH, SLICE_COLUMNS),
                       (CLUSTER_CSV_PATH, CLUSTER_COLUMNS)]:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(','.join(cols) + '\n')


def append_row(path, cols, row):
    parts = []
    for c in cols:
        v = row.get(c, '')
        if v is None:
            parts.append('')
        elif isinstance(v, float):
            parts.append(f'{v:.6g}')
        elif isinstance(v, bool):
            parts.append('True' if v else 'False')
        else:
            s = str(v).replace(',', ';').replace('\n', ' ')
            parts.append(s)
    with open(path, 'a') as f:
        f.write(','.join(parts) + '\n')


def load_done_zs():
    if not os.path.exists(SLICE_CSV_PATH):
        return set()
    try:
        df = pd.read_csv(SLICE_CSV_PATH)
        return set(int(z) for z in df['z'].values)
    except Exception:
        return set()


def save_checkpoint(arr):
    """Write the full corrected volume to checkpoint.npz.

    A checkpoint failure must never crash the run: the per-slice .npy
    files are the authoritative record, so on any failure we log and
    return. Windows can also transiently lock the destination file
    (antivirus, another reader) and make ``os.replace`` raise
    ``PermissionError`` -- that is retried with a short backoff.
    """
    tmp = CKPT_PATH + '.tmp.npz'
    try:
        np.savez_compressed(tmp, phi_corrected=arr)
    except Exception as exc:  # noqa: BLE001
        log_line(f'[WARN] checkpoint compress failed: '
                 f'{type(exc).__name__}: {exc} (per-slice npys intact)')
        return
    for _ in range(10):
        try:
            os.replace(tmp, CKPT_PATH)
            return
        except PermissionError:
            time.sleep(2.0)
        except Exception as exc:  # noqa: BLE001
            log_line(f'[WARN] checkpoint replace failed: '
                     f'{type(exc).__name__}: {exc} (per-slice npys intact)')
            break
    else:
        log_line('[WARN] checkpoint file stayed locked; skipping this '
                 'checkpoint -- per-slice npys are intact')
    try:
        os.remove(tmp)
    except Exception:  # noqa: BLE001
        pass


def load_checkpoint(default):
    if not os.path.exists(CKPT_PATH):
        return default.copy()
    try:
        with np.load(CKPT_PATH) as data:
            arr = data['phi_corrected']
            if arr.shape == default.shape:
                return arr.copy()
    except Exception:
        pass
    return default.copy()


def load_corrected_volume(default):
    """Reconstruct the corrected volume from the per-slice .npy files.

    The per-slice npys are the authoritative record -- each is written
    the moment its slice finishes, so this reconstruction is lag-free.
    checkpoint.npz, by contrast, is only refreshed every
    ``CHECKPOINT_EVERY`` slices and lags if the run is killed (or if a
    transient Windows file lock makes a checkpoint write fail). Falls
    back to checkpoint.npz, then the original field, when no per-slice
    npys exist yet.
    """
    files = []
    if os.path.isdir(SLICES_DIR):
        files = [f for f in os.listdir(SLICES_DIR)
                 if f.startswith('slice_z') and f.endswith('.npy')]
    if not files:
        return load_checkpoint(default)
    vol = default.copy()
    n = 0
    for f in files:
        try:
            z = int(f[len('slice_z'):-len('.npy')])
            vol[:, z] = np.load(os.path.join(SLICES_DIR, f))[:, 0]
            n += 1
        except Exception:  # noqa: BLE001
            pass
    log_line(f'loaded corrected volume from {n} per-slice npys')
    return vol


def log_line(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def _slice_npy_path(z):
    return os.path.join(SLICES_DIR, f'slice_z{int(z):03d}.npy')


def _grid_fig_path(z):
    return os.path.join(FIGS_DIR, f'grid_z{int(z):03d}.png')


def save_slice_npy(z, corrected_full):
    """Save the corrected slice z as its own ``(3, 1, H, W)`` array."""
    np.save(_slice_npy_path(z), corrected_full[:, int(z):int(z) + 1])


def save_grid_fig(z, phi_anchor_full, corrected_full):
    """Save a before/after deformation-grid figure for slice z.

    Full-resolution wireframe: every grid row and column line is drawn,
    warped by the displacement field. A fold shows up as grid lines
    crossing over each other. No per-cell colour fills (kept fast and
    light across all 528 slices).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    z = int(z)
    H, W = phi_anchor_full.shape[2], phi_anchor_full.shape[3]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, src, label in [
        (axes[0], phi_anchor_full, 'Initial'),
        (axes[1], corrected_full, 'Corrected'),
    ]:
        dy = src[1, z]
        dx = src[2, z]
        def_x = xx + dx
        def_y = yy + dy
        T1, T2 = _triangle_areas_2d(dy, dx)
        n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        # Row lines: one polyline per grid row; column lines: one per col.
        rows = [np.column_stack([def_x[i, :], def_y[i, :]]) for i in range(H)]
        cols = [np.column_stack([def_x[:, j], def_y[:, j]]) for j in range(W)]
        ax.add_collection(LineCollection(rows + cols, colors='k',
                                         linewidths=0.25))
        pad = max(W, H) * 0.03
        ax.set_xlim(def_x.min() - pad, def_x.max() + pad)
        ax.set_ylim(def_y.max() + pad, def_y.min() - pad)
        ax.set_aspect('equal')
        ax.set_title(f'{label}  z={z}  (folded triangles = {n_neg})',
                     fontsize=11)
    fig.suptitle(f'Deformation grid before/after correction  -  z={z}',
                 fontsize=13, fontweight='bold')
    fig.savefig(_grid_fig_path(z), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_slice_outputs(z, phi_anchor_full, corrected_full):
    """Save both per-slice artefacts (npy + grid figure) for slice z.

    Failures here must never abort the run, so each is guarded.
    """
    try:
        save_slice_npy(z, corrected_full)
    except Exception as exc:  # noqa: BLE001
        log_line(f'[WARN] z={z} npy save failed: {type(exc).__name__}: {exc}')
    try:
        save_grid_fig(z, phi_anchor_full, corrected_full)
    except Exception as exc:  # noqa: BLE001
        log_line(f'[WARN] z={z} grid fig failed: {type(exc).__name__}: {exc}')


def _component_corner_mask(component_mask, h, w):
    """Voxel-corner mask: a corner is movable iff at least one of its 4
    adjacent cells is in ``component_mask``. component_mask is the
    cell-grid mask shape (h, w); returned mask is voxel-corner grid
    shape (h+1, w+1).
    """
    corner_mask = np.zeros((h + 1, w + 1), dtype=bool)
    # Cell (i, j) has corners (i, j), (i+1, j), (i, j+1), (i+1, j+1).
    for di in (0, 1):
        for dj in (0, 1):
            corner_mask[di:di + h, dj:dj + w] |= component_mask
    return corner_mask


def enumerate_clusters_2d(phi_slice, max_axis, max_cells,
                           merge_dilation=None, extra_dilation=None):
    """phi_slice: (2, H, W). Returns list of cluster dicts.

    Fold cells (min(T1, T2) <= 0) are dilated by ``merge_dilation``,
    PLUS an extra per-cell dilation given by ``extra_dilation`` (shape
    (Hc, Wc), int). Fold cells that were in a cluster that failed to
    improve in the previous outer iter get +1 to their extra_dilation,
    causing their bbox to grow on the next attempt. Cells in succeeding
    clusters get reset to 0.
    """
    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    cell_min = np.minimum(T1, T2)
    cell_fold_mask = cell_min <= 0
    if not cell_fold_mask.any():
        return []
    n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    if merge_dilation is None:
        merge_dilation = _merge_for_n_neg(n_neg)
    merged_mask = (binary_dilation(cell_fold_mask, iterations=merge_dilation)
                   if merge_dilation > 0 else cell_fold_mask.copy())
    # Cells in clusters that failed to improve get progressively more
    # dilation on each retry. Other cells stay at the base.
    if extra_dilation is not None:
        max_e = int(extra_dilation.max())
        for e in range(1, max_e + 1):
            layer = (extra_dilation >= e) & cell_fold_mask
            if layer.any():
                merged_mask |= binary_dilation(layer, iterations=merge_dilation + e)
    labels, n_comp = cc_label(merged_mask)
    Hc, Wc = cell_fold_mask.shape
    bboxes = find_objects(labels)
    sizes = np.bincount(labels.ravel(), minlength=n_comp + 1)[1:]
    order = np.argsort(-sizes)
    clusters = []
    for idx in order:
        cid = int(idx) + 1
        bbox = bboxes[idx]
        if bbox is None:
            continue
        cy0, cy1 = bbox[0].start, bbox[0].stop
        cx0, cx1 = bbox[1].start, bbox[1].stop
        # Pad by 1 cell so the bbox includes the immediate-neighbor
        # cells (the cells whose corners share with cluster-cell corners
        # and whose constraints must stay satisfied).
        y0 = max(0, cy0 - BBOX_PAD); y1 = min(Hc, cy1 + BBOX_PAD)
        x0 = max(0, cx0 - BBOX_PAD); x1 = min(Wc, cx1 + BBOX_PAD)
        sy = y1 - y0; sx = x1 - x0
        # Rectangular interior_mask -- every corner inside the bbox is
        # movable, only the outer 1-corner ring is frozen. Empirically
        # this wins over the tighter component-shaped mask: SLSQP needs
        # the extra degrees of freedom to fix severe folds even though
        # the component-shaped mask is technically a subset of these.
        interior_mask = np.zeros((sy + 1, sx + 1), dtype=bool)
        if sy + 1 > 2 and sx + 1 > 2:
            interior_mask[1:-1, 1:-1] = True
        cluster_in_bbox = (labels[y0:y1, x0:x1] == cid)
        # Number of *fold* cells in this cluster (vs. dilated/merge
        # halo cells) -- this is what gets reported.
        comp_cells = int((cluster_in_bbox & cell_fold_mask[y0:y1, x0:x1]).sum())
        skipped = (sy > max_axis or sx > max_axis or sy * sx > max_cells)
        clusters.append({
            'cluster_id': cid,
            'y0': int(y0), 'y1': int(y1),
            'x0': int(x0), 'x1': int(x1),
            'crop_cells_y': sy, 'crop_cells_x': sx,
            'component_cells': comp_cells,
            'interior_mask': interior_mask,
            'n_movable_corners': int(interior_mask.sum()),
            'skipped_too_large': skipped,
        })
    return clusters


def _partition_clusters_nonoverlapping(clusters):
    """Greedy graph-colour: pack clusters into rounds where no two
    clusters in the same round have *adjacent or overlapping* bboxes.

    Two clusters are "conflicting" if their bboxes touch even at a
    boundary corner -- bbox A's bottom row of voxel corners is shared
    with bbox B's top row when their cell ranges are A:[y0,y1), B:[y1,y2).
    If A's interior_mask reaches that boundary corner and B's does too,
    parallel solves write the same corner with different values and one
    overwrites the other. Requiring a 1-cell gap (strict ``y1 < c2.y0``)
    guarantees disjoint voxel-corner grids.
    """
    rounds = []
    for c in clusters:
        y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
        placed = False
        for r in rounds:
            ok = True
            for c2 in r:
                # Disjoint (with 1-cell gap) iff one ends strictly before
                # the other starts in at least one axis.
                if not (y1 < c2['y0'] or c2['y1'] < y0
                        or x1 < c2['x0'] or c2['x1'] < x0):
                    ok = False
                    break
            if ok:
                r.append(c)
                placed = True
                break
        if not placed:
            rounds.append([c])
    return rounds


def _splice_interior(phi_slice, c, phi_l1):
    """Write back only the interior_mask corners (skip frozen edges).
    Safe even if another cluster's bbox shares cells with c's bbox.
    """
    y0, y1 = c['y0'], c['y1']
    x0, x1 = c['x0'], c['x1']
    mask = c['interior_mask']  # shape (y1-y0+1, x1-x0+1)
    iy, ix = np.where(mask)
    phi_slice[0, y0 + iy, x0 + ix] = phi_l1[0, iy, ix]
    phi_slice[1, y0 + iy, x0 + ix] = phi_l1[1, iy, ix]


def process_one_slice(z, phi_full, phi_anchor_full, executor=None):
    """Run all clusters in slice z. Returns (slice_row, cluster_rows, phi_slice_new).

    phi_full is mutable. If the slice has no folds, phi_slice_new is None.
    """
    H, W = phi_full.shape[2], phi_full.shape[3]
    phi_slice = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
    phi_anchor_slice = np.stack([phi_anchor_full[1, z], phi_anchor_full[2, z]])
    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    init_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    init_min_tri = float(min(T1.min(), T2.min()))
    init_L1 = float(np.abs(phi_slice - phi_anchor_slice).sum())
    t_slice = time.time()
    slice_row = {
        'z': int(z), 'H': int(H), 'W': int(W),
        'init_n_neg_tri': init_n_neg, 'init_min_tri': init_min_tri,
        'init_L1': init_L1,
    }
    if init_n_neg == 0:
        slice_row.update({
            'n_clusters': 0, 'n_clusters_processed': 0,
            'n_clusters_skipped': 0, 'n_clusters_feasible': 0,
            'n_clusters_l2_timeout': 0,
            'after_l2_n_neg_tri': 0, 'after_l2_min_tri': init_min_tri,
            'after_l1_n_neg_tri': 0, 'after_l1_min_tri': init_min_tri,
            'after_l1_L1': 0.0, 'after_l1_L2': 0.0,
            'total_l2_t': 0.0, 'total_l1_t': 0.0,
            'total_t': time.time() - t_slice,
            'feasible': True, 'l1_drop_pct': 0.0,
        })
        return slice_row, [], None
    cluster_rows = []
    n_processed = 0
    n_skipped = 0
    n_feasible = 0
    n_l2_timeout = 0
    total_l2_t = 0.0
    total_l1_t = 0.0
    total_clusters = 0
    # Outer loop: each full-grid cluster solve has no frozen edges so it
    # can introduce new folds at adjacent cells. Re-find clusters and
    # re-solve until n_neg = 0 or no improvement.
    t_slice_start = time.time()
    # Per-cell extra-dilation: counts how many times a fold cell has
    # been in a cluster that failed to improve. Fed back into
    # enumerate_clusters_2d so stuck clusters grow their bbox per retry,
    # while clusters that succeed reset to 0.
    cell_dims = (phi_slice.shape[1] - 1, phi_slice.shape[2] - 1)
    extra_dilation = np.zeros(cell_dims, dtype=np.int32)
    for outer_it in range(MAX_CLUSTER_OUTER_ITERS):
        if time.time() - t_slice_start > MAX_SLICE_TIME_S:
            break
        # merge_dilation: scales with n_neg (set inside the function).
        # extra_dilation: per-cell bump from failed clusters.
        clusters = enumerate_clusters_2d(phi_slice,
                                          MAX_CLUSTER_PER_AXIS,
                                          MAX_CLUSTER_CELLS,
                                          extra_dilation=extra_dilation)
        if not clusters:
            break
        total_clusters += len(clusters)
        any_processed_this_pass = False
        rounds = _partition_clusters_nonoverlapping(clusters)
        for r_idx, round_clusters in enumerate(rounds):
            if time.time() - t_slice_start > MAX_SLICE_TIME_S:
                break
            # Submit all clusters in this round to the pool (parallel).
            futures = []
            for c in round_clusters:
                y0, y1 = c['y0'], c['y1']
                x0, x1 = c['x0'], c['x1']
                vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
                phi_win = phi_slice[:, vy, vx].copy()
                phi_anchor_win = phi_anchor_slice[:, vy, vx].copy()
                c_with_z = dict(c)
                c_with_z['z'] = int(z)
                fut = executor.submit(
                    _bench_worker.solve_cluster_inline,
                    c_with_z, phi_win, phi_anchor_win,
                    THRESHOLD, EPS_L1,
                    L2_MAX_PASSES, L2_PASS_MAX_ITER, L1_POLISH_MAX_ITER,
                )
                futures.append((fut, c))
            # Collect this round's results -- splice each into phi_slice
            # via interior_mask so frozen edges aren't clobbered.
            for fut, c in futures:
                try:
                    crow, phi_l1 = fut.result(timeout=PER_CLUSTER_TIMEOUT_S)
                except FutTimeout:
                    crow = {
                        'z': int(z), 'cluster_id': c.get('cluster_id', -1),
                        'y0': c['y0'], 'y1': c['y1'],
                        'x0': c['x0'], 'x1': c['x1'],
                        'crop_cells_y': c['crop_cells_y'],
                        'crop_cells_x': c['crop_cells_x'],
                        'component_cells': c['component_cells'],
                        'skipped_too_large': False,
                        'l2_any_timeout': True,
                        'feasible': False, 'cluster_t': PER_CLUSTER_TIMEOUT_S,
                    }
                    phi_l1 = None
                crow['cluster_outer_it'] = outer_it
                cluster_rows.append(crow)
                if c['skipped_too_large']:
                    n_skipped += 1
                    continue
                n_processed += 1
                any_processed_this_pass = True
                if crow.get('l2_any_timeout'):
                    n_l2_timeout += 1
                if crow.get('feasible'):
                    n_feasible += 1
                total_l2_t += float(crow.get('l2_total_t', 0.0) or 0.0)
                total_l1_t += float(crow.get('l1_t', 0.0) or 0.0)
                if phi_l1 is not None:
                    _splice_interior(phi_slice, c, phi_l1)
                # Per-cell extra_dilation escalation is disabled: it was
                # the same over-merging problem at the per-cell level --
                # accumulated escalation merges nearby small failing
                # clusters into bigger blobs that SLSQP can't crack. The
                # Schwarz-tiling experiment confirmed constant dilation=1
                # is strictly better. extra_dilation stays at 0.
                # (Original behavior preserved as the +=1 branch below,
                # commented out for reference.)
                # y0, y1 = c['y0'], c['y1']
                # x0, x1 = c['x0'], c['x1']
                # if crow.get('feasible'):
                #     extra_dilation[y0:y1, x0:x1] = 0
                # else:
                #     extra_dilation[y0:y1, x0:x1] = np.minimum(
                #         extra_dilation[y0:y1, x0:x1] + 1, 4)
            # Heartbeat at the end of each round so the user sees progress.
            T1_now, T2_now = _triangle_areas_2d(phi_slice[0], phi_slice[1])
            n_now = int((T1_now <= 0).sum() + (T2_now <= 0).sum())
            log_line(f'    [z={z} outer={outer_it} round={r_idx+1}/{len(rounds)}]  '
                      f'+{len(round_clusters)} clusters | n_neg={n_now}  '
                      f'feas_so_far={n_feasible} skip={n_skipped} to={n_l2_timeout}  '
                      f'elapsed={time.time()-t_slice_start:.0f}s')
        # Continue iterating until n_neg=0 or MAX_CLUSTER_OUTER_ITERS.
        # Stubborn cells accumulate stuck_count and get extra local
        # dilation next iter -- the escalation is now per-cell, not
        # global.
        T1_now, T2_now = _triangle_areas_2d(phi_slice[0], phi_slice[1])
        n_neg_now = int((T1_now <= 0).sum() + (T2_now <= 0).sum())
        if n_neg_now == 0:
            break
        if not any_processed_this_pass:
            break
    T1_f, T2_f = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    after_l1_n_neg = int((T1_f <= 0).sum() + (T2_f <= 0).sum())
    after_l1_min_tri = float(min(T1_f.min(), T2_f.min()))
    after_l1_L1 = float(np.abs(phi_slice - phi_anchor_slice).sum())
    after_l1_L2 = float(np.linalg.norm(phi_slice - phi_anchor_slice))
    # We don't track after_l2 separately at the slice level here; the
    # cluster CSV has per-cluster details. Slice-level we report L1
    # numbers from the final state.
    slice_row.update({
        'n_clusters': total_clusters,
        'n_clusters_processed': n_processed,
        'n_clusters_skipped': n_skipped,
        'n_clusters_feasible': n_feasible,
        'n_clusters_l2_timeout': n_l2_timeout,
        'after_l2_n_neg_tri': '',  # not separately tracked here
        'after_l2_min_tri': '',
        'after_l1_n_neg_tri': after_l1_n_neg,
        'after_l1_min_tri': after_l1_min_tri,
        'after_l1_L1': after_l1_L1, 'after_l1_L2': after_l1_L2,
        'total_l2_t': total_l2_t, 'total_l1_t': total_l1_t,
        'total_t': time.time() - t_slice,
        'feasible': bool(after_l1_n_neg == 0),
        'l1_drop_pct': (100.0 * (init_L1 - after_l1_L1) / init_L1
                         if init_L1 > 0 else 0.0),
    })
    return slice_row, cluster_rows, phi_slice


def main():
    log_line(f'[start] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    phi_full = np.load(DATA_PATH)
    log_line(f'shape: {phi_full.shape}')
    D, H, W = phi_full.shape[1:]
    init_csvs()
    done = load_done_zs()
    corrected = load_corrected_volume(phi_full)
    phi_anchor = phi_full.copy()
    log_line(f'resuming with {len(done)}/{D} slices done')
    # Backfill per-slice artefacts for slices already done before this
    # feature existed: any done-slice missing its .npy / grid figure gets
    # one generated from the checkpoint volume. Only missing files are
    # written, so this is cheap on subsequent restarts.
    backfill = [z for z in sorted(done)
                if not (os.path.exists(_slice_npy_path(z))
                        and os.path.exists(_grid_fig_path(z)))]
    if backfill:
        log_line(f'backfilling per-slice outputs for {len(backfill)} slices')
        for z in backfill:
            save_slice_outputs(z, phi_anchor, corrected)
        log_line(f'backfill done ({len(backfill)} slices)')
    # Process slices in *ascending* order of initial fold count -- easy
    # slices first so we get fast wins, then attempt the harder ones.
    # Heavy slices (3000+ folds) can take an hour each; we don't want
    # those blocking the whole run.
    t_scan = time.time()
    init_folds = np.zeros(D, dtype=np.int64)
    for z in range(D):
        T1, T2 = _triangle_areas_2d(phi_anchor[1, z], phi_anchor[2, z])
        init_folds[z] = int((T1 <= 0).sum() + (T2 <= 0).sum())
    z_order = np.argsort(init_folds)  # ascending: easiest first
    log_line(f'pre-scan: {time.time()-t_scan:.1f}s  fold range [{init_folds.min()}..{init_folds.max()}]')
    t_run = time.time()
    log_line(f'parallel executor: {N_PARALLEL_WORKERS} workers')
    # One long-lived ProcessPoolExecutor for the whole run -- spawn cost
    # is paid once at startup instead of per cluster. If a worker dies
    # (OOM / native crash) the pool becomes permanently broken, so we
    # detect BrokenProcessPool, rebuild the pool, and retry the slice.
    ctx = mp.get_context('spawn')

    def _new_executor():
        return ProcessPoolExecutor(max_workers=N_PARALLEL_WORKERS,
                                   mp_context=ctx)

    executor = _new_executor()
    try:
        for z_int in z_order:
            z = int(z_int)
            if z in done:
                continue
            slice_row = cluster_rows = phi_slice_new = None
            pool_retries = 0
            while True:
                try:
                    slice_row, cluster_rows, phi_slice_new = process_one_slice(
                        z, corrected, phi_anchor, executor=executor)
                    break
                except BrokenProcessPool as exc:
                    pool_retries += 1
                    log_line(f'[POOL] z={z} worker pool broke ({exc}); '
                             f'rebuilding (retry {pool_retries}/'
                             f'{MAX_POOL_RETRIES})')
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    executor = _new_executor()
                    if pool_retries >= MAX_POOL_RETRIES:
                        log_line(f'[ERR] z={z} skipped: pool broke '
                                 f'{pool_retries}x in a row')
                        slice_row = None
                        break
                    # else: loop and retry the whole slice on the fresh pool
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc(limit=4).replace('\n', ' | ')
                    log_line(f'[ERR] z={z}  {type(exc).__name__}: {exc} '
                             f':: {tb}')
                    slice_row = None
                    break
            if slice_row is None:
                continue
            if phi_slice_new is not None:
                corrected[1, z] = phi_slice_new[0]
                corrected[2, z] = phi_slice_new[1]
            append_row(SLICE_CSV_PATH, SLICE_COLUMNS, slice_row)
            for crow in cluster_rows:
                append_row(CLUSTER_CSV_PATH, CLUSTER_COLUMNS, crow)
            done.add(z)
            # Per-slice artefacts: corrected slice .npy + before/after grid.
            save_slice_outputs(z, phi_anchor, corrected)
            log_line(
                f'[slice] z={z:>4d}  init={slice_row["init_n_neg_tri"]:>5d}  '
                f'final={slice_row["after_l1_n_neg_tri"]:>5d}  '
                f'min_tri={slice_row["after_l1_min_tri"]:+.4f}  '
                f'L1={slice_row["after_l1_L1"]:>8.3f} '
                f'(drop {slice_row["l1_drop_pct"]:>5.1f}%)  '
                f'clusters={slice_row["n_clusters"]}('
                f'feas={slice_row["n_clusters_feasible"]} '
                f'skip={slice_row["n_clusters_skipped"]} '
                f'to={slice_row["n_clusters_l2_timeout"]})  '
                f'L2_t={slice_row["total_l2_t"]:>5.1f}s '
                f'L1_t={slice_row["total_l1_t"]:>5.1f}s  '
                f'feas={slice_row["feasible"]}  '
                f'elapsed={time.time()-t_run:.0f}s')
            if len(done) % CHECKPOINT_EVERY == 0:
                save_checkpoint(corrected)
                log_line(f'[ckpt] saved, done={len(done)}/{D}')
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    save_checkpoint(corrected)
    log_line(f'[end] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  '
              f'done={len(done)}/{D}  elapsed={time.time()-t_run:.0f}s')


if __name__ == '__main__':
    main()
