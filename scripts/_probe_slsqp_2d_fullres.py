"""Probe 2D windowed SLSQP on the 320x456 full-res slice.

The user recalls this working historically. We bypass the notebook's
pix>6000 skip and actually run it, with a watchdog and wall cap so we
can tell whether it:
  (a) completes normally (skip threshold is too conservative), OR
  (b) hangs inside a single scipy call (same failure mode as 3D), OR
  (c) livelocks across iters (progresses but neg plateaus).
"""

from __future__ import annotations

import os
import sys
import time
import threading
import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from test_cases._builders import load_slice
from dvfopt import iterative_serial, jacobian_det2D

MPOINTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching", "mpoints.npy"))
FPOINTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching", "fpoints.npy"))

THRESHOLD, ERR_TOL = 0.01, 1e-5
WALL_CAP_SEC = 300.0  # 5 min


def describe(deformation):
    H, W = deformation.shape[-2:]
    phi = np.stack([deformation[1, 0], deformation[2, 0]])
    j = jacobian_det2D(phi)[0]
    neg = j <= THRESHOLD - ERR_TOL
    total_neg = int(neg.sum())
    print(f"shape=(1,{H},{W})  pix={H*W}  neg={total_neg}  min={j.min():+.4f}",
          flush=True)
    structure = np.ones((3, 3))
    labeled, n_comp = label(neg, structure=structure)
    if n_comp:
        sizes = np.bincount(labeled.ravel())[1:]
        print(f"  components: n={n_comp}  max={sizes.max()}  "
              f"top5={sorted(sizes, reverse=True)[:5]}", flush=True)
        biggest = int(np.argmax(sizes)) + 1
        zs, ys = np.where(labeled == biggest)
        by = int(ys.max() - ys.min() + 1)
        bz = int(zs.max() - zs.min() + 1)
        print(f"  biggest component bbox: {bz}x{by}  "
              f"first window ~{bz+4}x{by+4}  "
              f"{2*(bz+4)*(by+4)} vars, {(bz+4)*(by+4)} cons",
              flush=True)


def watchdog(stop_evt, start):
    last = start
    while not stop_evt.wait(5.0):
        now = time.time()
        print(f"  [wd] +{now - start:6.1f}s (gap={now - last:.1f}s)",
              flush=True)
        last = now


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
    wd.start()
    t.start()
    done.wait(WALL_CAP_SEC)
    stop_evt.set()
    wd.join(timeout=1.0)
    elapsed = time.time() - start
    if not done.is_set():
        print(f"  [TIMEOUT] {elapsed:.1f}s — thread still running", flush=True)
        os._exit(2)
    if "err" in result:
        raise result["err"]
    return result["out"], elapsed


def main():
    for label_str, slice_idx, sf in [
        ("02a_64x91",   90, 0.2),
        ("02c_64x91",  350, 0.2),
        ("02a_320x456", 90, 1.0),
    ]:
        print(f"\n=== {label_str}  slice={slice_idx}  scale={sf} ===", flush=True)
        deformation, *_ = load_slice(slice_idx, sf,
                                     mpoints_path=MPOINTS, fpoints_path=FPOINTS)
        describe(deformation)
        try:
            phi, t = run_with_cap(iterative_serial,
                                  deformation.copy(), verbose=1,
                                  max_iterations=2000)
            j = jacobian_det2D(phi)[0]
            print(f"  [DONE] wall={t:.1f}s  final_neg={(j <= 0).sum()}  "
                  f"min={j.min():+.5f}", flush=True)
        except Exception as e:
            print(f"  [ERR] {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
