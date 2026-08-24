"""Harmonic re-seed + isqp polish on a residual fold cluster (the "pass-through" closer).

The path test on B0039 z=16 showed the fold-free set is DISCONNECTED around twisted
residual cells: every local path to a valid configuration passes through deeply
inverted neighbours, so no monotone/protected/filtered solver can reach it. The
closer that works is to pass THROUGH the folds deliberately: replace a window's
interior by the harmonic (Laplace) extension of its boundary — a smooth field
that ignores the degenerate data — then run the no-damage windowed isqp polish
anchored (L2) to that seed. z=16: 0 folds, damage=0 (margin 20: 34 s, L2 152).

Seed-region size is the knob: too small and the boundary is still degenerate; too
large and the harmonic interpolant of a 150-px-displacement boundary tears
(z=16 margin 60: seed 832 folds, polish leaves 84). Two seeding modes:

- ``bbox``: one window = residual bounding box + margin.
- ``local``: seed only the neighbourhoods (radius ``--radius``) of the twisted /
  fully-inverted residual cells (the ones no local move can fix); thin cells are
  left to the polish.

Usage:
    python benchmarks/reseed_bench.py --input <residual.npy> --mode local --radius 6
"""

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy import ndimage  # noqa: E402

from dvfopt.constraints import TriConstraint2D  # noqa: E402
from dvfopt.core.windowed import min_field, windowed_correct  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import L2Objective  # noqa: E402


def fold_stats(phi, thr):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    m = np.minimum(T1, T2)
    return int((m < thr).sum()), int((m < 0).sum()), float(m.min())


def harmonic_reseed(phi, box):
    """Replace the box interior by the Laplace extension of its boundary (both channels)."""
    y0, y1, x0, x1 = box
    out = phi.copy()
    sub = out[:, y0:y1, x0:x1]
    h, w = sub.shape[1:]
    free = np.zeros((h, w), bool)
    free[1:-1, 1:-1] = True
    fp = np.argwhere(free)
    idx = -np.ones((h, w), int)
    idx[free] = np.arange(len(fp))
    n = len(fp)
    rows, cols, vals = [], [], []
    for k, (r, c) in enumerate(fp):
        rows.append(k)
        cols.append(k)
        vals.append(4.0)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if free[r + dr, c + dc]:
                rows.append(k)
                cols.append(idx[r + dr, c + dc])
                vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    for ch in range(2):
        b = np.zeros(n)
        for k, (r, c) in enumerate(fp):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if not free[r + dr, c + dc]:
                    b[k] += sub[ch, r + dr, c + dc]
        sub[ch][free] = spla.spsolve(A, b)
    return out


def hard_cells(phi, thr):
    """Cell mask of residuals no local move can fix: twisted (bow-tie) or fully inverted."""
    H, W = phi.shape[1:]
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    m = np.minimum(T1, T2)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X, Y = xx + phi[1], yy + phi[0]
    hard = np.zeros((H - 1, W - 1), bool)
    for i, j in zip(*np.where(m[: H - 1, : W - 1] < thr)):
        top = np.array([X[i, j + 1] - X[i, j], Y[i, j + 1] - Y[i, j]])
        bot = np.array([X[i + 1, j + 1] - X[i + 1, j], Y[i + 1, j + 1] - Y[i + 1, j]])
        left = np.array([X[i + 1, j] - X[i, j], Y[i + 1, j] - Y[i, j]])
        right = np.array([X[i + 1, j + 1] - X[i, j + 1], Y[i + 1, j + 1] - Y[i, j + 1]])
        twisted = (top @ bot < 0) or (left @ right < 0)
        hard[i, j] = twisted or (T1[i, j] < 0 and T2[i, j] < 0)
    return hard


def seed_boxes(phi, thr, mode, margin, radius):
    H, W = phi.shape[1:]
    if mode == "bbox":
        mask = min_field(TriConstraint2D(shape=(H, W)), phi) < thr
    else:
        mask = ndimage.binary_dilation(hard_cells(phi, thr), iterations=radius)
        mask = np.pad(mask, ((0, 1), (0, 1)))  # cell grid -> pixel grid
    lab, n = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(lab):
        y0, y1 = max(0, sl[0].start - margin), min(H, sl[0].stop + 1 + margin)
        x0, x1 = max(0, sl[1].start - margin), min(W, sl[1].stop + 1 + margin)
        boxes.append((int(y0), int(y1), int(x0), int(x1)))
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="save the polished field here (.npy)")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--mode", default="local", choices=["bbox", "local"])
    ap.add_argument(
        "--margin", type=int, default=6, help="pixels of margin around each seed region"
    )
    ap.add_argument(
        "--radius", type=int, default=6, help="local mode: dilation of the hard-cell mask"
    )
    ap.add_argument("--maxiter", type=int, default=800)
    a = ap.parse_args()

    thr = a.threshold
    phi0 = np.load(a.input).astype(np.float64)
    H, W = phi0.shape[1:]
    c = TriConstraint2D(shape=(H, W))
    print(f"input {a.input}: folds/neg/min = {fold_stats(phi0, thr)}", flush=True)
    boxes = seed_boxes(phi0, thr, a.mode, a.margin, a.radius)
    seed = phi0.copy()
    for b in boxes:
        seed = harmonic_reseed(seed, b)
    areas = [(b[1] - b[0]) * (b[3] - b[2]) for b in boxes]
    print(
        f"mode={a.mode} margin={a.margin} radius={a.radius}: {len(boxes)} seed boxes "
        f"(px {min(areas)}..{max(areas)}, total {sum(areas)}) -> seed folds/neg/min = {fold_stats(seed, thr)}",
        flush=True,
    )
    t = time.time()
    out, rep = windowed_correct(
        seed, "isqp", constraint=c, objective=L2Objective(), threshold=thr, maxiter=a.maxiter
    )
    fo = fold_stats(out, thr)
    print(
        f"polish: folds/neg/min = {fo} damage={rep.damage} giants={rep.giant_regions} "
        f"mop={rep.mop_windows} rounds={rep.rounds} | L2move(vs input)={np.linalg.norm(out - phi0):.1f} "
        f"L1={np.abs(out - phi0).sum():.0f} | {time.time() - t:.0f}s",
        flush=True,
    )
    if fo[0] == 0:
        print("0 FOLDS — harmonic re-seed + isqp polish cleared the residual", flush=True)
    if a.out:
        np.save(a.out, out)


if __name__ == "__main__":
    main()
