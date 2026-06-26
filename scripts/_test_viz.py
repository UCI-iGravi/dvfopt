"""Minimal reproduction of the viz cell error."""
import os
import sys

sys.path.insert(0, ".")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dvfopt import jacobian_det2D
from dvfopt.jacobian import triangle_sign_areas2D
from test_cases import make_deformation


def _forward_jdet_2d(dy, dx):
    ddx_dx = dx[:-1, 1:] - dx[:-1, :-1]
    ddy_dy = dy[1:, :-1] - dy[:-1, :-1]
    ddx_dy = dx[1:, :-1] - dx[:-1, :-1]
    ddy_dx = dy[:-1, 1:] - dy[:-1, :-1]
    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx


def measure(phi):
    jd = np.squeeze(jacobian_det2D(phi))
    fd = _forward_jdet_2d(phi[0], phi[1])
    tri = triangle_sign_areas2D(phi)
    return dict(
        jd=jd, fd=fd, tri=tri,
        n_cd=int((jd <= 0).sum()), n_fd=int((fd <= 0).sum()),
        n_tr=int((tri <= 0).sum()),
        min_cd=float(jd.min()), min_fd=float(fd.min()), min_tr=float(tri.min()),
    )


deformation, *_ = make_deformation("01c_20x40_edges")
phi_init = np.stack([deformation[1, 0], deformation[2, 0]])
m0 = measure(phi_init)
phi_fd = phi_init * 0.9
m_fd = measure(phi_fd)
phi_an = phi_init * 0.7
m_an = measure(phi_an)
phi_both = phi_init * 0.1
m_both = measure(phi_both)


class R:
    def __init__(self, success):
        self.success = success


res_fd = R(False)
res_an = R(False)
res_both = R(True)
qi_init = qi_fd = qi_an = qi_both = []

variants = [
    ("initial",                     phi_init, m0,     len(qi_init), None),
    ("(A) finite-diff Jac",         phi_fd,   m_fd,   len(qi_fd),   res_fd),
    ("(B) analytical Jac",          phi_an,   m_an,   len(qi_an),   res_an),
    ("(C) analytical + warm-start", phi_both, m_both, len(qi_both), res_both),
]

print("about to compute vmax...")
print(f"len(variants) = {len(variants)}")
for i, v in enumerate(variants):
    print(f"  variants[{i}]: type(v)={type(v).__name__}, len(v)={len(v)}")
    print(f"    v[1] type={type(v[1]).__name__}")
    print(f"    v[1]['tri'] shape={v[1]['tri'].shape} dtype={v[1]['tri'].dtype}")
    print(f"    abs(v[1]['tri']).max() = {abs(v[1]['tri']).max()}")

vmax = max(abs(v[1]["tri"]).max() for v in variants)
print(f"vmax = {vmax}")
