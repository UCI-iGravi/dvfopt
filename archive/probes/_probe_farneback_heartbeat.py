"""Reproduce the Farneback-from-notebook case to verify the solver finishes
and the new [inner] heartbeat fires on slow sub-loops."""
import time
import numpy as np

import cv2

from dvfopt import jacobian_det2D
from dvfopt.core.slsqp.iterative import iterative_serial


def make_test_image(size, shapes):
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for cy, cx, ry, rx, intensity in shapes:
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        img[mask] = intensity
    return img


S = 64
fixed_np = make_test_image(S, [
    (S * 0.35, S * 0.35, S * 0.15, S * 0.12, 0.9),
    (S * 0.60, S * 0.55, S * 0.18, S * 0.14, 0.7),
    (S * 0.30, S * 0.65, S * 0.10, S * 0.10, 0.5),
    (S * 0.70, S * 0.30, S * 0.12, S * 0.08, 0.6),
])
moving_np = make_test_image(S, [
    (S * 0.30, S * 0.40, S * 0.18, S * 0.10, 0.9),
    (S * 0.65, S * 0.50, S * 0.15, S * 0.17, 0.7),
    (S * 0.25, S * 0.60, S * 0.12, S * 0.08, 0.5),
    (S * 0.72, S * 0.25, S * 0.10, S * 0.12, 0.6),
])

fixed_u8 = (fixed_np * 255).astype(np.uint8)
moving_u8 = (moving_np * 255).astype(np.uint8)

flow = cv2.calcOpticalFlowFarneback(
    fixed_u8, moving_u8, flow=None,
    pyr_scale=0.5, levels=5, winsize=15, iterations=10,
    poly_n=7, poly_sigma=1.5, flags=0,
)
dx = flow[..., 0].astype(np.float64)
dy = flow[..., 1].astype(np.float64)

deformation = np.zeros((3, 1, S, S), dtype=np.float64)
deformation[1, 0] = dy
deformation[2, 0] = dx

phi_init = np.stack([deformation[1, 0], deformation[2, 0]])
jac_init = jacobian_det2D(phi_init)
print(f"[probe] init neg={(jac_init <= 0).sum()}  min={jac_init.min():+.6f}")

t0 = time.perf_counter()
phi = iterative_serial(deformation.copy(), verbose=1, threshold=0.01)
elapsed = time.perf_counter() - t0

jac_final = jacobian_det2D(phi)
print(f"[probe] done in {elapsed:.2f}s  "
      f"final neg={(jac_final <= 0).sum()}  min={jac_final.min():+.6f}")
