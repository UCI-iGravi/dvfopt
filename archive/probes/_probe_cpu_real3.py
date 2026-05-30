"""CPU full-grid probe on real_3."""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dvfopt import jacobian_det3D, scale_dvf_3d
from dvfopt.core.iterative3d_barrier import iterative_3d_barrier

FULL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching",
    "registered_output", "deformation3d.npy"))

def main():
    full = np.load(FULL)
    _, Df, Hf, Wf = full.shape
    factor = 1/3
    new_shape = (int(round(Df*factor)), int(round(Hf*factor)), int(round(Wf*factor)))
    dvf = scale_dvf_3d(full, new_shape)
    j0 = jacobian_det3D(dvf)
    print(f"=== real_3  shape={dvf.shape[1:]}  neg0={(j0<=0).sum()}  min0={j0.min():+.4f} ===",
          flush=True)
    t0 = time.time()
    phi = iterative_3d_barrier(dvf.copy(), verbose=1, windowed=False)
    elapsed = time.time() - t0
    j1 = jacobian_det3D(phi)
    neg = int((j1 <= 0).sum())
    print(f"\n  TOTAL: {elapsed:.2f}s  final_neg={neg}  min={j1.min():+.5f}  "
          f"{'PASS' if neg == 0 else 'FAIL'}", flush=True)

if __name__ == "__main__":
    main()
