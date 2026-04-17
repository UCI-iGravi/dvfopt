"""Run iterative_3d on real_6 with max_window_voxels=400 and verbose progress.

Wraps the call in a thread with a wall-time cap so a hang doesn't block the
terminal.  Emits a watchdog heartbeat every 10s so silence != progress.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dvfopt import iterative_3d, jacobian_det3D, scale_dvf_3d

FULL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching",
    "registered_output", "deformation3d.npy"))

WALL_CAP_SEC = 900.0  # 15 min


def watchdog(stop_evt, start):
    while not stop_evt.wait(10.0):
        print(f"  [wd] +{time.time() - start:6.1f}s", flush=True)


def run_with_cap(fn, *args, **kwargs):
    start = time.time()
    stop_evt = threading.Event()
    done = threading.Event()
    result = {}

    def run():
        try:
            result["out"] = fn(*args, **kwargs)
        except Exception as e:
            result["err"] = e
        finally:
            done.set()

    wd = threading.Thread(target=watchdog, args=(stop_evt, start), daemon=True)
    t = threading.Thread(target=run, daemon=True)
    wd.start(); t.start()
    done.wait(WALL_CAP_SEC)
    stop_evt.set(); wd.join(timeout=1.0)
    elapsed = time.time() - start
    if not done.is_set():
        print(f"  [TIMEOUT] {elapsed:.1f}s", flush=True)
        os._exit(2)
    if "err" in result:
        raise result["err"]
    return result["out"], elapsed


def main():
    full = np.load(FULL)
    _, Df, Hf, Wf = full.shape
    factor = 1 / 6
    sh = (int(round(Df * factor)), int(round(Hf * factor)), int(round(Wf * factor)))
    dvf = scale_dvf_3d(full, sh)
    j0 = jacobian_det3D(dvf)
    print(f"=== real_6  shape={sh}  neg0={int((j0<=0).sum())}  "
          f"min0={j0.min():+.4f} ===", flush=True)
    phi, wall = run_with_cap(iterative_3d, dvf.copy(),
                             verbose=1, max_iterations=500,
                             max_window_voxels=400,
                             max_window_voxels_ceiling=1200,
                             voxel_cap_stall_threshold=5)
    j1 = jacobian_det3D(phi)
    print(f"  wall={wall:.2f}s  final_neg={int((j1<=0).sum())}  "
          f"min={j1.min():+.5f}", flush=True)


if __name__ == "__main__":
    main()
