"""Microbench: tri_grad_T_v numpy vs numba.

Verifies numerical equivalence (atol=1e-10) then times both on
representative shapes (full B0039 slice + a few cluster crops).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from dvfopt.core.tri_primitives import (
    _tri_grad_T_v_numpy,
    tri_grad_T_v,
)


def _make_inputs(H, W, rng):
    phi = 0.05 * rng.standard_normal(2 * H * W)
    v = rng.standard_normal(2 * (H - 1) * (W - 1))
    return phi, v


def _bench_one(name, fn, phi, v, H, W, n_iter):
    # Warmup (numba JIT compile happens here on first call).
    fn(phi, H, W, v)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(phi, H, W, v)
    dt = time.perf_counter() - t0
    per_call_us = 1e6 * dt / n_iter
    print(f'  {name:<18s}  {n_iter:>5d} iter  {dt:>7.3f} s  ({per_call_us:>7.1f} us/call)')
    return dt


def main():
    rng = np.random.default_rng(0)
    shapes = [
        ('B0039 slice  ', 320, 456, 200),
        ('small cluster', 12, 16, 5000),
        ('med cluster  ', 30, 40, 2000),
        ('large cluster', 80, 100, 500),
    ]
    for label, H, W, n_iter in shapes:
        print(f'\n=== {label}  shape=({H}, {W})  iters={n_iter} ===')
        phi, v = _make_inputs(H, W, rng)
        g_ref = _tri_grad_T_v_numpy(phi, H, W, v)
        g_jit = tri_grad_T_v(phi, H, W, v)
        max_abs_err = float(np.max(np.abs(g_ref - g_jit)))
        rel_err = max_abs_err / (np.max(np.abs(g_ref)) + 1e-30)
        print(f'  equivalence  max_abs_err={max_abs_err:.2e}  rel={rel_err:.2e}')
        assert max_abs_err < 1e-9, f'JIT diverges from numpy by {max_abs_err}'
        t_np = _bench_one('numpy', _tri_grad_T_v_numpy, phi, v, H, W, n_iter)
        t_jit = _bench_one('numba', tri_grad_T_v, phi, v, H, W, n_iter)
        speedup = t_np / t_jit if t_jit > 0 else float('inf')
        print(f'  speedup      {speedup:.2f}x')


if __name__ == '__main__':
    main()
