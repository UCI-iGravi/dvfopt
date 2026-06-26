"""Multi-experiment hyperparameter sweep on B0039 z=300.

Picks z=300 because it's the slowest slice in the 11-slice sweep,
so even small wins show clearly. Each row tests one hypothesis vs a
fixed baseline:

  H1: n_workers       -- intra-slice parallelism, more vs fewer cores
  H2: merge_dilation  -- cluster size (smaller -> more clusters)
  H3: inner_max_iter  -- outer SLP cap per cluster
  H4: trust_radius_0  -- LP trust region initial
  H5: bbox_pad        -- crop padding around each cluster

Reports wall_s, L1_dev, final min_T, feasibility for each.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from research.strict_feasibility_2d.algorithms import cluster_lp_2tri as cmod
from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
    cluster_slp_iter,
)
from research.strict_feasibility_2d.worst_cases._load import load_b0039_slice


def _stats(phi_2hw):
    T1, T2 = _triangle_areas_2d(phi_2hw[0], phi_2hw[1])
    Tmin = np.minimum(T1, T2)
    return int((Tmin <= 0).sum()), float(Tmin.min())


BASELINE = dict(
    threshold=0.01,
    n_workers=8,
    merge_dilation=2,
    inner_max_iter=10,
    inner_trust_radius_0=0.5,
)


def _run(phi_in, **overrides):
    kwargs = {**BASELINE, **overrides}
    t0 = time.time()
    phi_out, _ = cluster_slp_iter(phi_in, **kwargs)
    wall = time.time() - t0
    n_neg, mt = _stats(phi_out)
    L1 = float(np.abs(phi_out - phi_in).sum())
    return wall, L1, n_neg, mt


def main():
    _, phi_in, meta = load_b0039_slice(300)
    print(f'Slice: z=300  shape={meta["shape"]}  init_n_neg={meta["init_n_neg"]}')
    print(f'Baseline: {BASELINE}\n')

    print(f'{"experiment":<35s}  {"wall":>7s}  {"L1":>8s}  {"n_neg":>5s}  {"min_T":>9s}')
    print('-' * 75)

    # Establish baseline first (warmup + reference timing).
    w0, L1_0, n0, mt0 = _run(phi_in)
    print(f'{"baseline":<35s}  {w0:>7.1f}  {L1_0:>8.1f}  {n0:>5d}  {mt0:>+9.4f}')

    experiments = [
        # H1: n_workers
        ('n_workers=4', dict(n_workers=4)),
        ('n_workers=12', dict(n_workers=12)),
        ('n_workers=16', dict(n_workers=16)),
        # H2: merge_dilation
        ('merge_dilation=1', dict(merge_dilation=1)),
        ('merge_dilation=3', dict(merge_dilation=3)),
        ('merge_dilation=4', dict(merge_dilation=4)),
        # H3: inner_max_iter
        ('inner_max_iter=5', dict(inner_max_iter=5)),
        ('inner_max_iter=15', dict(inner_max_iter=15)),
        ('inner_max_iter=20', dict(inner_max_iter=20)),
        # H4: trust_radius_0
        ('trust_radius_0=0.25', dict(inner_trust_radius_0=0.25)),
        ('trust_radius_0=1.0', dict(inner_trust_radius_0=1.0)),
        ('trust_radius_0=2.0', dict(inner_trust_radius_0=2.0)),
    ]
    for label, overrides in experiments:
        try:
            w, L1, n, mt = _run(phi_in, **overrides)
            delta_wall = (w / w0 - 1) * 100
            delta_L1 = (L1 / L1_0 - 1) * 100
            tag = f'{label:<35s}  {w:>7.1f}  {L1:>8.1f}  {n:>5d}  {mt:>+9.4f}'
            tag += f'  [dW={delta_wall:+5.1f}%  dL1={delta_L1:+5.1f}%]'
            print(tag)
        except Exception as exc:
            print(f'{label:<35s}  ERROR: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
