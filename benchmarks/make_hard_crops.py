"""Build fast hard-case crops from B0039 raw slices + validate their difficulty.

Each case is a small crop of the RAW field around a mapped pathology, so it keeps
the property that ordinary solvers fail to reach simplex 0 folds while running in
seconds-to-minutes instead of an hour+:

- z16_twist:  the bow-tie cell (collapsed edges, 65-160 px displacement ring)
              that defeated every 2-tri-row method, all ladder variants, and M14.
- z0_cluster: the ~3x-compressed dense cluster (area transport + twists + a
              three-corners-coincident cell) — hours for the full staged pipeline.
- z0_sliver:  cells pinned at ~-4e-4 below threshold — the TR-acceptance
              regression case (ratio tests freeze; line-search clears).

Validation per case: the DISCRIMINATOR (standard 2tri windowed isqp, objective
none) should FAIL to clear it; the RECIPE (bilinear rows, TR pass + no-TR retry)
should clear it to simplex 0 folds, fast. Crops are written to
benchmarks/output/testcases/ (gitignored; this script regenerates them).
"""

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import dvfopt.core.primitives.isqp as _isqp_mod  # noqa: E402
from dvfopt.constraints import SimplexConstraint2D, SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.windowed import min_field, windowed_correct  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402

OUT = "benchmarks/output/testcases"
THR = 0.01


def simplex_stats(phi):
    m = np.minimum(*_triangle_areas_2d(phi[0], phi[1]))
    return int((m < THR).sum()), float(m.min())


def crop_around(field, mask, margin):
    ys, xs = np.where(mask)
    H, W = field.shape[1:]
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 2 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 2 + margin)
    return np.ascontiguousarray(field[:, y0:y1, x0:x1]), (int(y0), int(y1), int(x0), int(x1))


def build():
    os.makedirs(OUT, exist_ok=True)
    vol = np.load("data/dvfs/b0039/b0039_laplacian_deformation_field.npy", mmap_mode="r")
    cases = {}

    # z16 twist: locate the residual fold of the 2tri-converged state, crop RAW there.
    res16 = np.load("benchmarks/output/ladder/inputs/z16_2tri_out.npy")
    m16 = np.minimum(*_triangle_areas_2d(res16[0], res16[1]))
    raw16 = np.asarray(vol[1:, 16], dtype=np.float64)
    cases["z16_twist"], box = crop_around(raw16, m16 < THR, margin=24)
    print(f"z16_twist crop {box}")

    # z0 cluster: bbox of the 158-fold pure-feasibility residual, crop RAW z0.
    res0 = np.load("benchmarks/output/ladder/inputs/z0_feasnone_out.npy")
    m0 = np.minimum(*_triangle_areas_2d(res0[0], res0[1]))
    raw0 = np.asarray(vol[1:, 0], dtype=np.float64)
    lab0, n0 = ndimage.label(m0 < THR)
    big0 = np.argmax(np.bincount(lab0.ravel())[1:]) + 1  # largest hard cluster only
    cases["z0_cluster"], box = crop_around(raw0, lab0 == big0, margin=12)
    print(f"z0_cluster crop {box}")

    # z0 sliver: largest cluster of near-threshold bilinear residuals after pass 1.
    sl = np.load("benchmarks/output/isqp_campaign/v5_bilinear_z0.npy")
    bf = min_field(SimplexConstraint2DBilinear(shape=sl.shape[1:]), sl) < THR
    lab, n = ndimage.label(bf)
    if n:
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        cases["z0_sliver"], box = crop_around(sl, lab == big, margin=24)
        print(f"z0_sliver crop {box} ({n} residual clusters, largest kept)")

    for name, phi in cases.items():
        np.save(f"{OUT}/{name}.npy", phi)
        f, mn = simplex_stats(phi)
        bl = int((min_field(SimplexConstraint2DBilinear(shape=phi.shape[1:]), phi) < THR).sum())
        print(
            f"{name}: shape {phi.shape[1:]}, simplex folds={f} min={mn:+.3f}, bilinear folds={bl}"
        )
    return list(cases)


def _run(phi, constraint, maxiter, no_tr=False):
    orig = _isqp_mod.isqp_solve
    if no_tr:

        def patched(*a, **k):
            k["trust_region"] = False
            return orig(*a, **k)

        _isqp_mod.isqp_solve = patched
        import dvfopt.core.windowed._inners as inners

        if hasattr(inners, "isqp_solve"):
            inners.isqp_solve = patched
    try:
        out, _rep = windowed_correct(
            phi,
            "isqp",
            constraint=constraint,
            objective=NoneObjective(),
            threshold=THR,
            maxiter=maxiter,
        )
    finally:
        _isqp_mod.isqp_solve = orig
        import dvfopt.core.windowed._inners as inners

        if hasattr(inners, "isqp_solve"):
            inners.isqp_solve = orig
    return out


def validate(names):
    for name in names:
        phi = np.load(f"{OUT}/{name}.npy")
        H, W = phi.shape[1:]
        if name == "z0_sliver":
            # acceptance-rule regression case (simplex-clean input): bilinear TR-only
            # must FREEZE on the slivers; the no-TR pass must clear them.
            c = SimplexConstraint2DBilinear(shape=(H, W))
            t = time.time()
            frozen = int((min_field(c, _run(phi, c, maxiter=600)) < THR).sum())
            t_tr = time.time() - t
            t = time.time()
            left = int((min_field(c, _run(phi, c, maxiter=800, no_tr=True)) < THR).sum())
            disc = "HARD (acceptance-rule regression)" if frozen else "not discriminating"
            print(
                f"{name}: TR-only leaves {frozen} bilinear slivers in {t_tr:.0f}s -> {disc} | "
                f"no-TR clears to {left} in {time.time() - t:.0f}s",
                flush=True,
            )
            continue
        t = time.time()
        out = _run(phi, SimplexConstraint2D(shape=(H, W)), maxiter=200)
        f_std, mn_std = simplex_stats(out)
        t_std = time.time() - t
        t = time.time()
        out = _run(phi, SimplexConstraint2DBilinear(shape=(H, W)), maxiter=600)
        for _retry in range(2):  # no-TR retries close TR-vetoed slivers (2-tri gauge)
            if simplex_stats(out)[0] == 0:
                break
            out = _run(out, SimplexConstraint2DBilinear(shape=(H, W)), maxiter=800, no_tr=True)
        f_rec, mn_rec = simplex_stats(out)
        t_rec = time.time() - t
        disc = "HARD (discriminates)" if f_std > 0 else "not discriminating"
        rec = "cleared" if f_rec == 0 else f"NOT cleared ({f_rec} folds)"
        print(
            f"{name}: 2tri-standard leaves {f_std} folds (min {mn_std:+.4f}) in {t_std:.0f}s -> {disc} | "
            f"recipe {rec} (min {mn_rec:+.4f}) in {t_rec:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    names = build()
    if not a.build_only:
        validate(names)
