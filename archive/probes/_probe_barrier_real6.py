"""Probe iterative_3d_barrier_torch on real_6 to see which stage stalls.

Benchmark output shows L-BFGS GPU-windowed, L-BFGS GPU-full, and L-BFGS
CPU-full all timed out at 300s on real_6 (shape=(88,53,76), 83 neg, min=-4.27)
while L-BFGS CPU-windowed succeeded in 52.9s. Something specific to the torch
path (both windowed and full) and the numpy full-grid path is stalling —
verbose=1 output should show which continuation stage burns all the time.

Runs each variant with a wall-cap watchdog that prints every 10s; kills the
process on timeout so torch state doesn't leak between variants.
"""

from __future__ import annotations

import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dvfopt import jacobian_det3D, scale_dvf_3d
from dvfopt.core.iterative3d_barrier import iterative_3d_barrier
from dvfopt.core.iterative3d_barrier_torch import iterative_3d_barrier_torch

FULL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching",
    "registered_output", "deformation3d.npy"))

WALL_CAP = 600.0


def watchdog(stop_evt: threading.Event, tag: str, start: float):
    while not stop_evt.wait(10.0):
        print(f"  [wd {tag}] +{time.time() - start:6.1f}s", flush=True)


def run(name: str, fn, dvf):
    print(f"\n>>> {name}", flush=True)
    start = time.time()
    stop_evt = threading.Event()
    result: dict = {}

    def worker():
        try:
            result["out"] = fn(dvf.copy())
        except Exception as e:
            result["err"] = e
        finally:
            stop_evt.set()

    wd_stop = threading.Event()
    wd = threading.Thread(target=watchdog, args=(wd_stop, name, start), daemon=True)
    t = threading.Thread(target=worker, daemon=True)
    wd.start()
    t.start()
    stop_evt.wait(WALL_CAP)
    wd_stop.set()
    elapsed = time.time() - start

    if not stop_evt.is_set():
        print(f"  [TIMEOUT at {elapsed:.1f}s] abandoning worker", flush=True)
        return
    if "err" in result:
        print(f"  [ERR] {type(result['err']).__name__}: {result['err']}", flush=True)
        return
    phi = result["out"]
    j1 = jacobian_det3D(phi)
    neg = int((j1 <= 0).sum())
    print(f"  done in {elapsed:.2f}s  final_neg={neg}  min={j1.min():+.5f}",
          flush=True)


def main():
    full = np.load(FULL)
    _, Df, Hf, Wf = full.shape
    factor = 1 / 6
    new_shape = (max(1, int(round(Df * factor))),
                 max(1, int(round(Hf * factor))),
                 max(1, int(round(Wf * factor))))
    dvf = scale_dvf_3d(full, new_shape)
    j0 = jacobian_det3D(dvf)
    init_neg = int((j0 <= 0).sum())
    print(f"=== real_6  shape={dvf.shape[1:]}  neg0={init_neg}  min0={j0.min():+.4f} ===",
          flush=True)

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        torch = None
        cuda_ok = False

    run("CPU windowed (baseline, succeeded in benchmark)",
        lambda d: iterative_3d_barrier(d, verbose=1, windowed=True),
        dvf)
    run("CPU full-grid (timed out in benchmark)",
        lambda d: iterative_3d_barrier(d, verbose=1, windowed=False),
        dvf)
    if cuda_ok:
        run("GPU windowed (timed out in benchmark)",
            lambda d: iterative_3d_barrier_torch(
                d, verbose=1, windowed=True, device="cuda"),
            dvf)
        run("GPU full-grid (timed out in benchmark)",
            lambda d: iterative_3d_barrier_torch(
                d, verbose=1, windowed=False, device="cuda"),
            dvf)
    else:
        print("\n[skip] no CUDA available — torch variants not run", flush=True)


if __name__ == "__main__":
    main()
