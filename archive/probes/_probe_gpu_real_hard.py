"""GPU-only probe on real_4 and real_3 — the main user path.

CPU full-grid on big volumes is intrinsically slow (~200s per lam stage on
132x80x114) and is not the primary convergence path anyone uses. This
script focuses on the GPU paths to verify the penalty-phase early-exit fix.
"""

from __future__ import annotations

import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dvfopt import jacobian_det3D, scale_dvf_3d
from dvfopt.core.iterative3d_barrier_torch import iterative_3d_barrier_torch

FULL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching",
    "registered_output", "deformation3d.npy"))

WALL_CAP = 1800.0


def watchdog(stop_evt, tag, start):
    while not stop_evt.wait(15.0):
        print(f"  [wd {tag}] +{time.time() - start:6.1f}s", flush=True)


def run(name, fn, dvf):
    print(f"\n>>> {name}", flush=True)
    start = time.time()
    stop_evt = threading.Event()
    result = {}

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
    wd.start(); t.start()
    stop_evt.wait(WALL_CAP)
    wd_stop.set()
    elapsed = time.time() - start

    if not stop_evt.is_set():
        print(f"  [TIMEOUT at {elapsed:.1f}s]", flush=True)
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

    import torch
    if not torch.cuda.is_available():
        print("[!] no CUDA available — skipping", flush=True)
        return

    cases = [("real_4", 1/4), ("real_3", 1/3)]
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
        all_pass &= run(
            f"{name} GPU windowed",
            lambda d: iterative_3d_barrier_torch(d, verbose=1, windowed=True, device="cuda"),
            dvf)
        all_pass &= run(
            f"{name} GPU full-grid",
            lambda d: iterative_3d_barrier_torch(d, verbose=1, windowed=False, device="cuda"),
            dvf)

    print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} ===", flush=True)


if __name__ == "__main__":
    main()
