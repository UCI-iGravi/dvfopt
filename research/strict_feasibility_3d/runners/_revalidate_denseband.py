import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import numpy as np

from dvfopt import correct_dvf_3d
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

OUT = Path(__file__).parent / 'output'
phi = np.load(OUT / 'b0039_FULL_stage3_z000_016.npy').astype(np.float64)
print(f'dense band {phi.shape} n_neg_in={int((six_tet_min_volume_3d(phi) <= 0).sum())}', flush=True)
t0 = time.time()
out, rep = correct_dvf_3d(phi, threshold=0.01, n_workers=1, thorough=True, verbose=1)
print(
    f'\nFINAL feasible={rep.feasible} {rep.n_neg_in}->{rep.n_neg_out} '
    f'n<0.01={rep.n_below_out} min_T={rep.min_T_out:+.6f} '
    f'floor_out={rep.best_diag_floor_out} L1={rep.l1_from_input:.1f} '
    f'wall={rep.wall_s:.1f}s',
    flush=True,
)
print('stages:', [s['stage'] for s in rep.stages], flush=True)
if rep.feasible:
    np.save(OUT / 'orch_v2_strict_denseband.npy', out)
