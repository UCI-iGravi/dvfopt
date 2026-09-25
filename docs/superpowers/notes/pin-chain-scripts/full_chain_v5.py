"""THROWAWAY v5: the DVF-only pin chain, generic over the cohort.
  pins (tau) -> radius-pruned pairwise drop -> AMG harmonic re-fill -> per-slice 2D windowed engine
  -> 2.5D marching -> one census (+ the count of slices with a bilinear 2D fold, diagnostic).
python benchmarks/output/probe_source_space/full_chain_v5.py B0039 [laplacian_exterior]
"""
import os, sys, json, time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "benchmarks")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BRAIN = sys.argv[1] if len(sys.argv) > 1 else "B0039"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "laplacian_exterior"
_TAU = os.environ.get("PIN_TAU", "0.7")
THR, TAU, C_LIP, RADIUS, RTOL = 0.01, (None if _TAU == "auto" else float(_TAU)), float(os.environ.get("PIN_C", 1.0)), 60.0, 1e-4
DETREND = os.environ.get("PIN_DETREND", "0") == "1"
OUT = f"benchmarks/output/probe_source_space/cohort/{BRAIN}_{VARIANT}" + (f"_c{C_LIP}" if C_LIP != 1.0 else "") + (f"_tau{_TAU}" if _TAU != "0.7" else "") + ("_detrend" if DETREND else "")
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/chain.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def harmonic_refill(phi, dirichlet, rtol):
    import pyamg
    from scipy.sparse.linalg import cg
    from dvfopt.laplacian.utils import laplacianA3D, propagate_dirichlet_rhs

    shape = phi.shape[1:]
    bidx = np.flatnonzero(dirichlet.ravel())
    t = time.time()
    A = laplacianA3D(shape, bidx, log_fn=lambda m: None).tocsr()
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=2000, strength=("symmetric", {"theta": 0.0}))
    M = ml.aspreconditioner(cycle="V")
    log(f"  matrix + AMG setup {time.time() - t:.0f} s, {len(ml.levels)} levels")
    out = phi.copy()
    for ch in (1, 2):
        u = phi[ch].astype(np.float64)
        rhs = np.zeros(A.shape[0])
        rhs[bidx] = u.ravel()[bidx]
        propagate_dirichlet_rhs(shape, bidx, rhs)
        it = [0]
        t = time.time()
        x, info = cg(A, rhs, x0=u.ravel(), rtol=rtol, maxiter=200, M=M, callback=lambda _: it.__setitem__(0, it[0] + 1))
        x = x.reshape(shape)
        x[dirichlet] = u[dirichlet]
        out[ch] = x.astype(np.float32)
        log(f"  ch{ch}: cg info {info}, {it[0]} it, {time.time() - t:.0f} s")
    return out


def solve_slice(args):
    z, sl = args
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()
    from dvfopt import correct_dvf

    t = time.time()
    r = correct_dvf(sl, constraint="bilinear", strategy="isqp_windowed", objective="l2", threshold=THR)
    return z, np.asarray(r.corrected, dtype=np.float32), time.time() - t, int(r.final_n_neg)


def run_2d(vol, zs, tag):
    from dvfopt.core._pool import pinned_thread_env

    out = vol.copy()
    walls, left = [], 0
    jobs = ((z, np.ascontiguousarray(vol[:, z : z + 1].astype(np.float64))) for z in zs)
    with pinned_thread_env(), ProcessPoolExecutor(4) as ex:
        for z, sl, w, nn in ex.map(solve_slice, jobs):
            out[:, z] = sl[:, 0]
            walls.append(w)
            left += nn
    if walls:
        log(f"  {tag}: {len(zs)} slices, median {np.median(walls):.1f} s, max {max(walls):.0f} s, folds left {left}")
    return out


def bilinear_fold_slices(vol):
    from dvfopt.jacobian.injectivity_radius import cell_min_jdet_2d

    return [z for z in range(vol.shape[1]) if ((cell_min_jdet_2d(vol[1:, z].astype(np.float64)) / 2) < THR).any()]


def stage(name, fn):
    path = f"{OUT}/{name}.npy"
    if os.path.isfile(path):
        log(f"{name}: on disk, skipping")
        return np.load(path)
    log(f"{name}: start")
    t = time.time()
    arr = fn()
    np.save(path, arr)
    log(f"{name}: done in {time.time() - t:.0f} s")
    return arr


if __name__ == "__main__":
    from dvfopt.dvf.pins import detect_pins, violating_pairs_pruned, greedy_cover
    from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d, n_neg_best_diagonal
    from benchmark_utils import load_cohort_field, load_cohort_correspondences

    T0 = time.time()
    raw = np.asarray(load_cohort_field(BRAIN, VARIANT), dtype=np.float32)
    D, H, W = raw.shape[1:]
    rep = dict(brain=BRAIN, variant=VARIANT, shape=[D, H, W])
    log(f"{BRAIN}/{VARIANT} {raw.shape}, max|dz| {np.abs(raw[0]).max()}")

    def stage_c():
        t = time.time()
        global TAU
        if TAU is None:
            # self-calibrating: the non-pin |Δu| ceiling scales with |d| (B0304: 0.78 px vs B0039: 0.12);
            # p98 of all voxels sits below any plausible pin fraction. Measured: 0.7 on B0039/B0213, 2.35 on B0304.
            from dvfopt.dvf.pins import source_strength
            TAU = max(0.7, 10.0 * float(np.percentile(source_strength(raw), 98)))
            log(f"  auto tau = {TAU:.2f}")
        pins = detect_pins(raw, TAU)
        coords = np.argwhere(pins)
        d = raw[1:][:, pins].T
        log(f"  pins at tau {TAU}: {len(coords)} ({time.time() - t:.0f} s)")
        t = time.time()
        dt = d.astype(np.float64)
        if DETREND:
            # global affine trend d ~ A [z, y, x, 1], two-pass (refit without the 3-MAD outliers);
            # the pair test runs on the residual, the re-fill still uses the original values
            X = np.column_stack([coords.astype(np.float64), np.ones(len(coords))])
            keep_fit = np.ones(len(coords), bool)
            for _ in range(2):
                A_, *_ = np.linalg.lstsq(X[keep_fit], dt[keep_fit], rcond=None)
                resid = np.linalg.norm(dt - X @ A_, axis=1)
                mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
                keep_fit = resid <= np.median(resid) + 3 * 1.4826 * mad
            dt = dt - X @ A_
            log(f"  affine trend removed: |trend| median {np.median(np.linalg.norm(X @ A_, axis=1)):.1f} px, residual |d| median {np.median(np.linalg.norm(dt, axis=1)):.2f} p95 {np.percentile(np.linalg.norm(dt, axis=1), 95):.1f}")
        bad = violating_pairs_pruned(coords.astype(np.float64), dt, C_LIP, RADIUS)
        drop = greedy_cover(len(coords), bad)
        log(f"  {len(bad)} violating pairs, {int(drop.sum())} dropped, {int((~drop).sum())} kept ({time.time() - t:.0f} s)")
        rep.update(pins=len(coords), violating_pairs=int(len(bad)), dropped=int(drop.sum()))
        json.dump(rep, open(f"{OUT}/pins.json", "w"))
        kept = np.zeros(pins.shape, bool)
        kept[tuple(coords[~drop].T)] = True
        return harmonic_refill(raw, kept, RTOL)

    refilled = stage("C_refilled", stage_c)
    if os.path.isfile(f"{OUT}/pins.json"):
        rep.update(json.load(open(f"{OUT}/pins.json")))

    engine2d = stage("D_engine2d", lambda: run_2d(refilled, range(D), "2D"))

    def stage_e():
        from dvfopt.pipeline_25d import correct_dvf_25d

        out, r = correct_dvf_25d(engine2d.astype(np.float64), n_workers=4, verbose=0)
        rep["25d"] = dict(feasible=bool(r.feasible), n_neg_in=int(r.n_neg_in), n_neg_out=int(r.n_neg_out),
                          floor_out=int(r.n_neg_best_diag_out), wall_s=float(r.wall_s), stages=str(r.stages))
        return out.astype(np.float32)

    after25d = stage("E_25d", stage_e)

    # Stage F (a final per-slice bilinear pass) is DIAGNOSTIC ONLY: measured on B0039, fixing the
    # 298 slices' sub-pixel bilinear folds pushed 1,674 tets back under the 0.01 margin (floor 706).
    # The two certificates are about different interpolants; simplex-3D is the one of record here.
    final = after25d
    rep["wall_to_final_s"] = time.time() - T0

    v = six_tet_min_volume_3d(np.ascontiguousarray(final.astype(np.float64)))
    v0 = six_tet_min_volume_3d(np.ascontiguousarray(raw.astype(np.float64)))
    rep["raw"] = dict(folds_thr=int((v0 < THR).sum()), folds_zero=int((v0 < 0).sum()), min_v=float(v0.min()))
    rep["final"] = dict(folds_thr=int((v < THR).sum()), folds_zero=int((v < 0).sum()), min_v=float(v.min()),
                        floor=int(n_neg_best_diagonal(final.astype(np.float64), THR)),
                        bilinear_slices_left=len(bilinear_fold_slices(final)),
                        l2_move_from_raw=float(np.linalg.norm((final - raw).ravel())),
                        moved_frac=float((np.abs(final - raw).max(axis=0) > 1e-6).mean()))
    mp, fp = load_cohort_correspondences(BRAIN, VARIANT)  # [diagnostic] residual column only
    if mp is not None:
        pv = np.clip(np.round(fp).astype(int), 0, [D - 1, H - 1, W - 1])
        pd_ = (mp - fp)[:, 1:]
        res = np.linalg.norm(final[1:][:, pv[:, 0], pv[:, 1], pv[:, 2]].T - pd_, axis=1)
        rep["final"].update(corr_n=int(len(res)), pin_resid_median=float(np.median(res)), pin_within_10px=float((res <= 10).mean()))
    rep["total_wall_s"] = time.time() - T0
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
    log("REPORT " + json.dumps({k: rep[k] for k in ("brain", "raw", "final", "wall_to_final_s")}))
