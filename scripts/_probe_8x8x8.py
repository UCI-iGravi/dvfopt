"""Quick probe: iterative_3d on the benchmark's 8x8x8 case with 5 min cap.

Mirrors the benchmark setup (seed=42, MAX_MAG_3D=4.0, scale (8,8,8)) and runs
vanilla iterative_3d (no voxel cap). Watchdog prints every 10s.
"""
from __future__ import annotations

import os, sys, time, threading
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dvfopt import iterative_3d, jacobian_det3D, scale_dvf_3d, generate_random_dvf_3d

WALL_CAP = 300.0


def watchdog(stop_evt, start):
    while not stop_evt.wait(10.0):
        print(f"  [wd] +{time.time() - start:6.1f}s", flush=True)


def main():
    base = generate_random_dvf_3d((3, 3, 3, 3), 4.0, 42)
    dvf = scale_dvf_3d(base, (8, 8, 8))
    j0 = jacobian_det3D(dvf)
    print(f"=== rand_8x8x8 neg0={int((j0<=0).sum())} min0={j0.min():+.4f} ===", flush=True)

    start = time.time()
    stop_evt = threading.Event()
    result = {}

    def run():
        try:
            result["out"] = iterative_3d(dvf.copy(), verbose=1, max_iterations=500)
        except Exception as e:
            result["err"] = e
        finally:
            stop_evt.set()

    wd = threading.Thread(target=watchdog, args=(threading.Event(), start), daemon=True)
    t = threading.Thread(target=run, daemon=True)
    wd.start(); t.start()
    stop_evt.wait(WALL_CAP)
    elapsed = time.time() - start

    if not stop_evt.is_set():
        print(f"  [TIMEOUT at {elapsed:.1f}s]", flush=True)
        os._exit(2)

    if "err" in result:
        print(f"  [ERR] {type(result['err']).__name__}: {result['err']}", flush=True)
        return
    phi = result["out"]
    j1 = jacobian_det3D(phi)
    print(f"  done in {elapsed:.2f}s  final_neg={int((j1<=0).sum())}  min={j1.min():+.5f}", flush=True)


if __name__ == "__main__":
    main()
