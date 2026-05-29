import os, sys, time, contextlib, io, warnings
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label, binary_dilation, generate_binary_structure

from dvfopt import iterative_2d_tri_refine_repair
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d

from _m14_schwarz_proto import m14_schwarz, _stats, _fold_clusters

THRESHOLD = 0.01


def _silent(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _plant_fold(arr, cy, cx, amp=0.8):
    arr[cy, cx] += amp
    arr[cy+1, cx] -= amp
    arr[cy, cx+1] -= amp
    arr[cy+1, cx+1] += amp


def load_b0039(cy, cx, size):
    arr = np.load('../../data/dvfs/b0039/b0039_laplacian_deformation_field.npy')
    dy = arr[1, 12, cy:cy+size, cx:cx+size].astype(np.float64).copy()
    dx = arr[2, 12, cy:cy+size, cx:cx+size].astype(np.float64).copy()
    return np.stack([dy, dx])


def load_b0039_full():
    arr = np.load('../../data/dvfs/b0039/b0039_laplacian_deformation_field.npy')
    return np.stack([arr[1, 12].astype(np.float64).copy(),
                     arr[2, 12].astype(np.float64).copy()])
