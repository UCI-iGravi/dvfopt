"""Empirical verification: fp32 vs fp64 torch barrier solver.

Runs both dtypes on the benchmark cases and reports neg-Jdet,
min-Jdet, L2 distortion, and wall time.  Measures whether fp32 is
a safe default.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from test_cases import make_deformation
from dvfopt import jacobian_det2D, jacobian_det3D, scale_dvf_3d, generate_random_dvf_3d
from dvfopt.core.iterative2d_barrier import iterative_2d_barrier_torch
from dvfopt.core.iterative3d_barrier_torch import iterative_3d_barrier_torch


def time_call(fn, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0


def test_2d(key, mode):
    deformation, *_ = make_deformation(key)
    phi_init = np.stack([deformation[1, 0], deformation[2, 0]])

    results = {}
    for dtype in [torch.float32, torch.float64]:
        phi, t = time_call(iterative_2d_barrier_torch, deformation,
                           verbose=0, windowed=(mode == 'win'), dtype=dtype)
        j = jacobian_det2D(phi)[0]
        results[dtype] = {
            'neg': int((j <= 0).sum()),
            'min': float(j.min()),
            'l2': float(np.linalg.norm(phi - phi_init)),
            't': t,
        }
    return results


def test_3d(key, dvf, mode):
    results = {}
    for dtype in [torch.float32, torch.float64]:
        phi, t = time_call(iterative_3d_barrier_torch, dvf,
                           verbose=0, windowed=(mode == 'win'), dtype=dtype)
        j = jacobian_det3D(phi)
        results[dtype] = {
            'neg': int((j <= 0).sum()),
            'min': float(j.min()),
            'l2': float(np.linalg.norm(phi - dvf)),
            't': t,
        }
    return results


def print_row(name, mode, r):
    f32 = r[torch.float32]
    f64 = r[torch.float64]
    neg_ok = '=' if f32['neg'] == f64['neg'] else 'DIFF'
    min_delta = f32['min'] - f64['min']
    l2_rel = (f32['l2'] - f64['l2']) / max(f64['l2'], 1e-9)
    speedup = f64['t'] / max(f32['t'], 1e-9)
    print(f"  {name:<22s} {mode:>5s}  "
          f"f32: neg={f32['neg']:5d} min={f32['min']:+.5f} l2={f32['l2']:9.4f} t={f32['t']:7.2f}s  |  "
          f"f64: neg={f64['neg']:5d} min={f64['min']:+.5f} l2={f64['l2']:9.4f} t={f64['t']:7.2f}s  "
          f"|  neg:{neg_ok}  dmin={min_delta:+.2e}  dl2_rel={l2_rel:+.2e}  sp={speedup:.2f}x",
          flush=True)


def main():
    print(f"CUDA: {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    print("\n=== 2D cases ===", flush=True)
    for key in ['01a_10x10_crossing', '03c_20x20_opposite', '01c_20x40_edges']:
        for mode in ['win', 'full']:
            r = test_2d(key, mode)
            print_row(key, mode, r)

    print("\n=== 3D cases ===", flush=True)
    base = generate_random_dvf_3d((3, 3, 3, 3), 4.0, 42)
    synth = {
        '8x8x8':    scale_dvf_3d(base, (8, 8, 8)),
        '16x16x16': scale_dvf_3d(base, (16, 16, 16)),
        '32x32x32': scale_dvf_3d(base, (32, 32, 32)),
    }

    real_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data/corrected_correspondences_count_touching/registered_output/deformation3d.npy')
    print(f"real_path: {real_path} exists={os.path.exists(real_path)}")
    if os.path.exists(real_path):
        full = np.load(real_path)
        _, Df, Hf, Wf = full.shape
        print(f"  real full shape: (3, {Df}, {Hf}, {Wf})")
        synth['real_1_4'] = scale_dvf_3d(full, (Df // 4, Hf // 4, Wf // 4))
        synth['real_1_2'] = scale_dvf_3d(full, (Df // 2, Hf // 2, Wf // 2))
        del full

    for key, dvf in synth.items():
        for mode in ['win', 'full']:
            # Full-grid now uses active-set DOF reduction — radius=10 keeps the
            # LBFGS variable bounded by the size of the infeasible region, not
            # the full grid, so the legacy total-DOF cap no longer applies.
            r = test_3d(key, dvf, mode)
            print_row(key, mode, r)


if __name__ == '__main__':
    main()
