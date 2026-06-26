"""Full-volume B0039 fold elimination — fast two-phase strategy.

The per-band-to-strict-0 loop (v1) was too slow: solving each band to
n_neg=0 with thorough=True spent the full escape + multiscale cost per
band (band 1 alone = 7.9 h) when seams reintroduce folds anyway. The
efficient factoring is:

  Phase A — band-parallel BULK reduction (``parallel_zband_solve``): all
    z-bands' active-band M10Tet run CONCURRENTLY across a small pool
    (memory-capped), interior planes pasted, one seam-cleanup pass. This
    is cheap per band (no per-band escape/multiscale) and the bands
    overlap in wall-clock, so the heavy bands amortise.

  Phase B — ONE global escape pass (``correct_dvf_3d(thorough=True)``) on
    the (now small) residual: bulk routes to active-band on the few
    scattered residual clusters, then the coupled escape + multiscale
    fallback clears them to strict 0.

Starts from the v1 checkpoint if present (band 1 already done), so that
work is reused. Checkpoints after Phase A and after Phase B.

GUARDED for Windows spawn (workers re-import this module).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))


def main():
    from dvfopt import correct_dvf_3d
    from dvfopt.core.wallbreakers._coupled_kring_3d import parallel_zband_solve
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    SRC = OUT / 'b0039_FULL_stage1.npy'
    V1_CKPT = OUT / 'b0039_FULL_corrected_ckpt.npy'   # reuse band-1 work
    CKPT_A = OUT / 'b0039_FULL_v2_phaseA.npy'
    FINAL = OUT / 'b0039_FULL_corrected.npy'

    THR = 0.01
    # Memory-capped band parallelism: whole-band M10Tet on dense bands is
    # RAM-heavy, so keep few concurrent (pool_map falls back to serial if a
    # worker still OOMs).
    n_workers_bulk = 3
    n_workers_escape = 24

    if CKPT_A.exists():
        cur = np.load(CKPT_A)
        print('RESUME: Phase A checkpoint found — skipping to Phase B',
              flush=True)
        phase_a_done = True
    elif V1_CKPT.exists():
        cur = np.load(V1_CKPT)
        print('START from v1 checkpoint (band 1 already corrected)', flush=True)
        phase_a_done = False
    else:
        cur = np.load(SRC).astype(np.float64)
        print('START from raw stage-1 field', flush=True)
        phase_a_done = False

    n0 = int((six_tet_min_volume_3d(cur) <= 0).sum())
    print(f'volume {cur.shape} n_neg={n0}', flush=True)

    # ---- Phase A: band-parallel bulk ----
    if not phase_a_done:
        t0 = time.time()
        cur, info = parallel_zband_solve(
            cur, threshold=THR, band_size=24, overlap=4, pad=4,
            n_workers=n_workers_bulk, seam_cleanup=True, verbose=1,
        )
        np.save(CKPT_A, cur)
        print(f'[Phase A] band-parallel bulk: {info["n_neg_before"]}->'
              f'{info["n_neg_after"]} bands={info["n_bands"]} '
              f'seam_cleanup={info["seam_cleanup_ran"]} '
              f'({(time.time()-t0)/3600:.2f}h)', flush=True)

    # ---- Phase B: global escape on residual ----
    gtot = int((six_tet_min_volume_3d(cur) <= 0).sum())
    print(f'[Phase B] residual before escape n_neg={gtot}', flush=True)
    if gtot > 0:
        t0 = time.time()
        cur, rep = correct_dvf_3d(
            cur, threshold=THR, n_workers=n_workers_escape,
            thorough=True, verbose=1,
        )
        print(f'[Phase B] global escape: ->{rep.n_neg_out} '
              f'feasible={rep.feasible} min_T={rep.min_T_out:+.6f} '
              f'({(time.time()-t0)/3600:.2f}h)', flush=True)

    mv = six_tet_min_volume_3d(cur)
    n_neg = int((mv <= 0).sum())
    n_below = int((mv < THR - 1e-5).sum())
    print(f'FINAL n_neg={n_neg} n<0.01={n_below} min_T={float(mv.min()):+.6f}',
          flush=True)
    # Gate the canonical save on strict feasibility (never silently deliver a
    # folded/sub-threshold field as the corrected output).
    if n_neg == 0 and n_below == 0:
        np.save(FINAL, cur)
        print(f'saved {FINAL} (strict-feasible)', flush=True)
    else:
        import sys
        partial = OUT / 'b0039_FULL_corrected_v2_PARTIAL.npy'
        np.save(partial, cur)
        print(f'NOT strict-feasible (n_neg={n_neg}, n<0.01={n_below}) — wrote '
              f'{partial}; NOT overwriting {FINAL.name}', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
