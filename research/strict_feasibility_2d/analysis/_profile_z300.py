"""Profile auto_slp on B0039 z=300 (slowest slice in 11-slice sweep).

Dumps a cProfile stats file to runners/output/profile_z300.prof and
prints the top time-consumers grouped by package.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from research.strict_feasibility_2d.algorithms.cluster_lp_2tri import (
    cluster_slp_iter,
)
from research.strict_feasibility_2d.worst_cases._load import load_b0039_slice

OUT = _HERE.parent / 'runners' / 'output' / 'profile_z300_seq.prof'


def main():
    case_id, phi_in, meta = load_b0039_slice(300)
    print(
        f'Profiling {case_id}  shape={meta["shape"]}  init_n_neg={meta["init_n_neg"]}  '
        f'(SEQUENTIAL, n_workers=1)',
        flush=True,
    )
    prof = cProfile.Profile()
    prof.enable()
    phi_out, info = cluster_slp_iter(phi_in, threshold=0.01, n_workers=1)
    prof.disable()
    import numpy as np

    # Build a minimal rec dict for the print line below
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
    from research.strict_feasibility_2d.runners._compare import run_method

    T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
    Tmin = np.minimum(T1, T2)
    rec = {
        'feasible': bool((Tmin > 0.01 - 1e-5).all() and (Tmin > 0).all()),
        'final_n_neg_2tri': int((Tmin <= 0).sum()),
        'L1_dev': float(np.abs(phi_out - phi_in).sum()),
        'wall_s': info['wall_s'],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prof.dump_stats(str(OUT))
    print(
        f'\nResult: feasible={rec["feasible"]}  n_neg={rec["final_n_neg_2tri"]}  '
        f'L1={rec["L1_dev"]:.1f}  wall={rec["wall_s"]:.1f}s'
    )
    print(f'Wrote {OUT}\n')

    stats = pstats.Stats(prof)
    print('=== Top 30 by cumtime ===')
    stats.sort_stats('cumulative').print_stats(30)
    print('=== Top 30 by tottime ===')
    stats.sort_stats('tottime').print_stats(30)


if __name__ == '__main__':
    main()
