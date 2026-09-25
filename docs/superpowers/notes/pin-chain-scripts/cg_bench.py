"""THROWAWAY: one-channel re-fill solve benchmark on the full B0039 kept-pin problem.
Arms: Jacobi-PCG (baseline), AMG-PCG (pyamg smoothed aggregation / Ruge-Stuben). Same rtol 1e-4.
python benchmarks/output/probe_source_space/cg_bench.py
"""
import os, sys, time, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "benchmarks")
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

OUT = "benchmarks/output/probe_source_space/full_v3"
TAU, C_LIP, RADIUS, RTOL = 0.7, 1.0, 60.0, 1e-4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    from benchmark_utils import load_cohort_field
    from dvfopt.dvf.pins import detect_pins, violating_pairs, greedy_cover
    from dvfopt.laplacian.utils import laplacianA3D, propagate_dirichlet_rhs

    raw = np.asarray(load_cohort_field("B0039"), dtype=np.float32)
    shape = raw.shape[1:]
    kp = f"{OUT}/kept.npy"
    if os.path.isfile(kp):
        kept = np.load(kp)
    else:
        pins = detect_pins(raw, TAU)
        coords = np.argwhere(pins)
        d = raw[1:][:, pins].T
        bad = violating_pairs(coords.astype(np.float64), d, C_LIP, RADIUS, n_workers=4)
        drop = greedy_cover(len(coords), bad)
        kept = np.zeros(pins.shape, bool)
        kept[tuple(coords[~drop].T)] = True
        np.save(kp, kept)
    log(f"kept pins {int(kept.sum())}")

    u = raw[1].astype(np.float64)
    bidx = np.flatnonzero(kept.ravel())
    t = time.time()
    A = laplacianA3D(shape, bidx, log_fn=lambda m: None)
    rhs = np.zeros(A.shape[0])
    rhs[bidx] = u.ravel()[bidx]
    propagate_dirichlet_rhs(shape, bidx, rhs)
    log(f"matrix {time.time() - t:.0f} s, nnz {A.nnz / 1e6:.0f}M")
    x0 = u.ravel().copy()
    results = {}

    def run(tag, M):
        it = [0]
        t = time.time()
        x, info = cg(A, rhs, x0=x0, rtol=RTOL, maxiter=5000, M=M, callback=lambda _: it.__setitem__(0, it[0] + 1))
        r = np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)
        results[tag] = dict(info=int(info), iters=it[0], solve_s=time.time() - t, rel_resid=float(r))
        log(f"{tag}: {results[tag]}")

    arm = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arm in ("all", "jacobi"):
        run("jacobi-pcg", diags(1.0 / A.diagonal(), format="csr"))
    if arm in ("all", "amg"):
        import pyamg

        t = time.time()
        ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=2000, strength=("symmetric", {"theta": 0.0}))
        log(f"SA-AMG setup {time.time() - t:.0f} s, {len(ml.levels)} levels, complexity {ml.operator_complexity():.2f}")
        results["amg_setup_s"] = time.time() - t
        run("sa-amg-pcg", ml.aspreconditioner(cycle="V"))
    json.dump(results, open(f"{OUT}/cg_bench.json", "w"), indent=1)
