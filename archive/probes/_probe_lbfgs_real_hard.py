"""Verify the penalty-phase early-exit fix on the hard real cases.

Runs iterative_3d_barrier / iterative_3d_barrier_torch (windowed + full-grid,
CPU + GPU) on real_4 and real_3 — the cases that previously timed out in
benchmark-slsqp-vs-lbfgs.ipynb — and reports wall time + final neg/min.

Success criteria: final_neg == 0 AND wall time under the 1800s benchmark cap
for L-BFGS methods.
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

WALL_CAP = 1800.0  # matches the L-BFGS per-method timeout in the benchmark


def watchdog(stop_evt: threading.Event, tag: str, start: float):
    while not stop_evt.wait(15.0):
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
        return False
    if "err" in result:
        print(f"  [ERR] {type(result['err']).__name__}: {result['err']}", flush=True)
        return False
    phi = result["out"]
    j1 = jacobian_det3D(phi)
    neg = int((j1 <= 0).sum())
    ok = neg == 0
    print(f"  done in {elapsed:.2f}s  final_neg={neg}  min={j1.min():+.5f}  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    full = np.load(FULL)
    _, Df, Hf, Wf = full.shape

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    # Only the cases that previously timed out. Tested by factor:
    #   real_4 -> 1/4, real_3 -> 1/3
    cases = [
        ("real_4", 1/4),
        ("real_3", 1/3),
    ]
    all_pass = True
    for name, factor in cases:
        new_shape = (max(1, int(round(Df * factor))),
                     max(1, int(round(Hf * factor))),
                     max(1, int(round(Wf * factor))))
        dvf = scale_dvf_3d(full, new_shape)
        j0 = jacobian_det3D(dvf)
        init_neg = int((j0 <= 0).sum())
        print(f"\n=== {name}  shape={dvf.shape[1:]}  neg0={init_neg}  min0={j0.min():+.4f} ===",
              flush=True)

        # Windowed paths are typically fast; full-grid is the one we fixed.
        # Run full-grid first since that's where the behavior change is.
        all_pass &= run(f"{name} CPU full-grid",
                        lambda d: iterative_3d_barrier(d, verbose=1, windowed=False),
                        dvf)
        if cuda_ok:
            all_pass &= run(f"{name} GPU full-grid",
                            lambda d: iterative_3d_barrier_torch(
                                d, verbose=1, windowed=False, device="cuda"),
                            dvf)
            all_pass &= run(f"{name} GPU windowed",
                            lambda d: iterative_3d_barrier_torch(
                                d, verbose=1, windowed=True, device="cuda"),
                            dvf)

    print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} ===", flush=True)


if __name__ == "__main__":
    main()
