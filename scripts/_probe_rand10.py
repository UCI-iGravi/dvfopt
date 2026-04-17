"""Trace iterative_serial on 2D rand_10x10 seed=42 to see where it stalls."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from dvfopt import iterative_serial, jacobian_det2D, scale_dvf, generate_random_dvf

base = generate_random_dvf((3, 1, 5, 5), 1.0, 42)
dvf = scale_dvf(base, (10, 10))
j0 = jacobian_det2D(dvf)[0]
print(f"rand_10x10 neg0={int((j0<=0).sum())} min0={j0.min():+.5f}")
neg_coords = np.argwhere(j0 <= 0)
print(f"neg voxel coords: {neg_coords.tolist()}")

phi = iterative_serial(dvf.copy(), verbose=2, max_iterations=50)
j1 = jacobian_det2D(phi)[0]
print(f"\nfinal: neg={int((j1<=0).sum())}  min={j1.min():+.5f}")
final_neg = np.argwhere(j1 <= 0)
print(f"remaining neg coords: {final_neg.tolist()}")
