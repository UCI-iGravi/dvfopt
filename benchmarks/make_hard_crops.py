"""Build fast hard-case crops from B0039 raw slices + validate their difficulty.

Each case is a small crop of the RAW field around a mapped pathology, so it keeps
the property that ordinary solvers fail to reach simplex 0 folds while running in
seconds-to-minutes instead of an hour+:

- z16_twist:  the bow-tie cell (collapsed edges, 65-160 px displacement ring)
              that defeated every 2-tri-row method, all ladder variants, and M14.
- z0_cluster: the ~3x-compressed dense cluster (area transport + twists + a
              three-corners-coincident cell) — hours for the full staged pipeline.
- z0_sliver:  cells pinned at ~-4e-4 below threshold — simplex-clean on input,
              so the 2tri rows are BLIND to it; only the bilinear gauge sees it.

Validation per case, both runs `windowed_correct` once on engine defaults: the
DISCRIMINATOR (standard 2tri rows, objective none) leaves bilinear folds behind —
either because it cannot clear them (z16_twist, z0_cluster) or because its rows
never saw them (z0_sliver) — and the RECIPE (bilinear rows) clears the case to
simplex 0 folds, fast (the bilinear residual is printed alongside). Crops are
written to data/dvfs/crops/ (gitignored; this script regenerates
them).
"""

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from dvfopt.constraints import SimplexConstraint2D, SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.windowed import min_field, windowed_correct  # noqa: E402
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402

OUT = "data/dvfs/crops"
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


def _run(phi, constraint, maxiter):
    """One `windowed_correct` call on engine defaults.

    The per-window no-TR fallback (PR #73) and backend fallback (PR #78) are
    the engine's job — no monkeypatching, no retry loop out here.
    """
    out, _rep = windowed_correct(
        phi,
        "isqp",
        constraint=constraint,
        objective=NoneObjective(),
        threshold=THR,
        maxiter=maxiter,
    )
    return out


def bilinear_folds(phi):
    return int((min_field(SimplexConstraint2DBilinear(shape=phi.shape[1:]), phi) < THR).sum())


def validate(names):
    for name in names:
        phi = np.load(f"{OUT}/{name}.npy")
        H, W = phi.shape[1:]
        # DISCRIMINATOR: the standard 2tri rows leave bilinear folds behind —
        # either unclearable (z16_twist, z0_cluster) or unseen (z0_sliver, whose
        # input is already simplex-clean).
        t = time.time()
        std = _run(phi, SimplexConstraint2D(shape=(H, W)), maxiter=200)
        t_std = time.time() - t
        f_std, _ = simplex_stats(std)
        bl_std = bilinear_folds(std)
        # RECIPE: bilinear rows, one call, defaults.
        t = time.time()
        out = _run(phi, SimplexConstraint2DBilinear(shape=(H, W)), maxiter=600)
        t_rec = time.time() - t
        f_rec, mn_rec = simplex_stats(out)
        bl_rec = bilinear_folds(out)
        disc = "HARD (discriminates)" if bl_std > 0 else "not discriminating"
        # Pass/fail is the simplex gauge, as before; the bilinear residual is
        # reported because it is what makes these crops hard in the first place.
        rec = "cleared" if f_rec == 0 else f"NOT cleared ({f_rec} folds)"
        print(
            f"{name}: 2tri-standard leaves {f_std} simplex / {bl_std} bilinear folds "
            f"in {t_std:.0f}s -> {disc} | recipe {rec} "
            f"(simplex min {mn_rec:+.4f}, bilinear residual {bl_rec}) in {t_rec:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    names = build()
    if not a.build_only:
        validate(names)
