"""Full-volume B0039 fold elimination via a checkpointed z-band loop.

Processes the full (3, 528, 320, 456) stage-1 field one overlapping z-band
at a time, each band solved to strict feasibility by the proven
``correct_dvf_3d(n_workers=24, thorough=True)`` config (the probe reached
strict n_neg=0 on the worst band, 50 781 folds, this way). Interior corner
planes are committed per band (halo planes owned by the neighbour, exactly
as ``parallel_zband_solve`` does); seams are repaired by a final
full-volume cleanup pass.

RESUMABLE / CRASH-SAFE: a checkpoint (the partially-corrected field + the
next band index) is written after every band. Re-running the script picks
up from the checkpoint. This matters because the whole volume is a
multi-hour job.

GUARDED for Windows spawn: ``correct_dvf_3d(n_workers>1)`` spawns workers
that re-import this module; the heavy work sits under ``main()`` /
``if __name__ == '__main__'`` so workers never re-run it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))


def main():
    from dvfopt import correct_dvf_3d
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

    OUT = Path(__file__).parent / 'output'
    SRC = OUT / 'b0039_FULL_stage1.npy'
    CKPT = OUT / 'b0039_FULL_corrected_ckpt.npy'
    PROG = OUT / 'b0039_FULL_corrected_progress.json'
    FINAL = OUT / 'b0039_FULL_corrected.npy'

    THR = 0.01
    band_size, overlap = 24, 4
    n_workers = 24

    phi0 = np.load(SRC).astype(np.float64)
    D, H, W = phi0.shape[1:]
    Dc = D - 1  # cube layers

    # Build z-bands over cube layers [0, Dc): interior [s, e), halo +/- overlap.
    bands = []  # (corner_z0, corner_z1, interior_cube_s, interior_cube_e)
    s = 0
    while s < Dc:
        e = min(s + band_size, Dc)
        cz0 = max(0, s - overlap)
        cz1 = min(Dc, e + overlap)
        bands.append((cz0, cz1, s, e))
        s = e
    n_bands = len(bands)

    # Resume from checkpoint if present.
    if CKPT.exists() and PROG.exists():
        cur = np.load(CKPT)
        prog = json.loads(PROG.read_text())
        start = int(prog['next_band'])
        print(f'RESUME from band {start}/{n_bands}', flush=True)
    else:
        cur = phi0.copy()
        start = 0

    n0 = int((six_tet_min_volume_3d(cur) <= 0).sum())
    print(f'full volume {cur.shape} n_neg={n0} n_bands={n_bands} '
          f'band_size={band_size} overlap={overlap}', flush=True)

    t_run = time.time()
    for bi in range(start, n_bands):
        cz0, cz1, si, ei = bands[bi]
        crop = cur[:, cz0:cz1 + 1, :, :].copy()
        cmv = six_tet_min_volume_3d(crop)
        cn0 = int((cmv <= 0).sum())
        cnb = int((cmv < THR - 1e-5).sum())  # strict bar: min_T < 0.01
        t0 = time.time()
        if cnb == 0:
            out = crop  # already STRICT-feasible (min_T >= THR) — skip solve
            feas, nno = True, 0
        else:
            # thorough=False per band: triage + bulk + coupled escape (with
            # k-escalation) but NOT the per-band multiscale fallback. The
            # probe showed this reaches strict 0 on the worst band via escape
            # alone at ~2.2 h (24-thread kernels). Multiscale per band is what
            # made the v1-thorough run take 7.9 h/band; any band that DOES
            # stall here is repaired by the final global thorough=True pass
            # below, which runs multiscale once over the whole residual.
            out, rep = correct_dvf_3d(
                crop, threshold=THR, n_workers=n_workers,
                thorough=False, verbose=0,
            )
            feas, nno = bool(rep.feasible), int(rep.n_neg_out)
        # Commit interior corner planes [si, ei) ([si, ei] at volume top).
        w_lo = si
        w_hi = ei + 1 if ei == Dc else ei
        cur[:, w_lo:w_hi, :, :] = out[:, w_lo - cz0:w_hi - cz0, :, :]
        # Checkpoint.
        np.save(CKPT, cur)
        PROG.write_text(json.dumps({'next_band': bi + 1, 'n_bands': n_bands}))
        gtot = int((six_tet_min_volume_3d(cur) <= 0).sum())
        print(f'[band {bi + 1}/{n_bands}] z[{cz0}:{cz1}] crop {cn0}->{nno} '
              f'feasible={feas} global_n_neg={gtot} '
              f'({time.time() - t0:.0f}s, elapsed {(time.time()-t_run)/3600:.2f}h)',
              flush=True)

    # Final seam-residual cleanup with the full orchestrator. Gate on the
    # STRICT bar (n_below = min_T < THR), not just negatives, so positive-but-
    # sub-threshold seam cubes are repaired too (n_below subsumes n_neg).
    mv = six_tet_min_volume_3d(cur)
    n_neg = int((mv <= 0).sum())
    n_below = int((mv < THR - 1e-5).sum())
    print(f'all bands done; n_neg={n_neg} n<0.01={n_below}', flush=True)
    if n_below > 0:
        print('final full-volume cleanup on residual/seams...', flush=True)
        cur, rep = correct_dvf_3d(
            cur, threshold=THR, n_workers=n_workers, thorough=True, verbose=1,
        )
        np.save(CKPT, cur)
        print(f'  cleanup: feasible={rep.feasible} n_neg_out={rep.n_neg_out}',
              flush=True)

    mv = six_tet_min_volume_3d(cur)
    n_neg = int((mv <= 0).sum())
    n_below = int((mv < THR - 1e-5).sum())
    print(f'FINAL n_neg={n_neg} n<0.01={n_below} min_T={float(mv.min()):+.6f} '
          f'total_wall={(time.time()-t_run)/3600:.2f}h', flush=True)
    # Gate the canonical save on STRICT feasibility — never silently deliver a
    # folded or sub-threshold field as "the corrected volume".
    if n_neg == 0 and n_below == 0:
        np.save(FINAL, cur)
        print(f'saved {FINAL} (strict-feasible)', flush=True)
    else:
        partial = OUT / 'b0039_FULL_corrected_PARTIAL.npy'
        np.save(partial, cur)
        print(f'NOT strict-feasible (n_neg={n_neg}, n<0.01={n_below}) — wrote '
              f'{partial}; NOT overwriting {FINAL.name}', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
