"""Probe iterative_3d on the real_6 (88x53x76) downsample.

Measures the first bounding-window SLSQP would be asked to solve,
without actually calling SLSQP.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dvfopt import jacobian_det3D, scale_dvf_3d
from dvfopt.core.slsqp.spatial3d import (
    neg_jdet_bounding_window_3d, argmin_worst_voxel,
)

FULL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "data", "corrected_correspondences_count_touching",
    "registered_output", "deformation3d.npy"))
THRESHOLD, ERR_TOL = 0.01, 1e-5


def describe(factor: float, name: str):
    full = np.load(FULL)
    _, Df, Hf, Wf = full.shape
    new_shape = (max(1, int(round(Df * factor))),
                 max(1, int(round(Hf * factor))),
                 max(1, int(round(Wf * factor))))
    dvf = scale_dvf_3d(full, new_shape)
    j = jacobian_det3D(dvf)
    neg = j <= THRESHOLD - ERR_TOL
    total_neg = int(neg.sum())
    print(f"\n=== {name}  shape={dvf.shape[1:]}  neg={total_neg}  min={j.min():+.4f} ===")

    structure = np.ones((3, 3, 3))
    labeled, n_comp = label(neg, structure=structure)
    sizes = np.bincount(labeled.ravel())[1:]
    print(f"  connected components: {n_comp}")
    print(f"  component sizes: min={sizes.min()}  med={int(np.median(sizes))}  "
          f"max={sizes.max()}  top5={sorted(sizes, reverse=True)[:5]}")

    worst = argmin_worst_voxel(j)
    size, bbox_center = neg_jdet_bounding_window_3d(
        j, worst, THRESHOLD, ERR_TOL, labeled_array=labeled)
    sz, sy, sx = size
    nvars = 3 * sz * sy * sx
    ncons = sz * sy * sx
    print(f"  first-pick worst voxel: {worst}  jdet={j[worst]:+.4f}")
    print(f"  first SLSQP window: {sz}x{sy}x{sx} = {ncons} voxels, "
          f"{nvars} vars, {ncons} nonlinear constraints")
    # SLSQP inner LS-QP is roughly O(nvars^3) per QP step; with hundreds of
    # constraints and >1k vars it can easily cycle indefinitely.


if __name__ == "__main__":
    for factor, name in [(1/8, "real_8"), (1/6, "real_6"),
                         (1/4, "real_4"), (1/3, "real_3")]:
        describe(factor, name)
