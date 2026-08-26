"""Local area-budget map of a residual fold cluster - the transport picture.

For every ``block x block`` sub-region inside the residual's window, compare the
enclosed cell area with what "every triangle >= t" demands (2*t per cell). A
sub-region whose ratio need/have exceeds 1 is locally INFEASIBLE with its own
boundary frozen: it can only be fixed by importing area from outside, and the
distance to the nearest surplus block is the transport distance the solver must
achieve. Prints the deficit map, the surplus reservoir, and the transport distance
histogram - the quantities that decide whether an inflation ladder can converge.

Usage:
    python benchmarks/budget_map.py --input benchmarks/output/ladder/inputs/z0_feasnone_out.npy
"""

import argparse

import numpy as np
from scipy import ndimage

from dvfopt.constraints import TriConstraint2D
from dvfopt.core.windowed import min_field
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--margin", type=int, default=6)
    a = ap.parse_args()

    phi = np.load(a.input).astype(np.float64)
    H, W = phi.shape[1:]
    thr, B = a.threshold, a.block
    mask = min_field(TriConstraint2D(shape=(H, W)), phi) < thr
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - a.margin), min(H, ys.max() + 2 + a.margin)
    x0, x1 = max(0, xs.min() - a.margin), min(W, xs.max() + 2 + a.margin)
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    area = (T1 + T2)[y0 : y1 - 1, x0 : x1 - 1]
    fold = mask[y0 : y1 - 1, x0 : x1 - 1]
    h, w = area.shape
    print(f"window y[{y0},{y1}) x[{x0},{x1}) -> {h}x{w} cells, {int(fold.sum())} residual folds")
    print(
        f"global: area={area.sum():.1f}  need={2 * thr * area.size:.1f}  "
        f"utilization={2 * thr * area.size / area.sum():.3f}"
    )

    nb, mb = h // B, w // B
    ratio = np.zeros((nb, mb))
    folds_b = np.zeros((nb, mb), int)
    for i in range(nb):
        for j in range(mb):
            blk = area[i * B : (i + 1) * B, j * B : (j + 1) * B]
            ratio[i, j] = 2 * thr * blk.size / max(blk.sum(), 1e-12)
            folds_b[i, j] = int(fold[i * B : (i + 1) * B, j * B : (j + 1) * B].sum())
    deficit = ratio > 1.0
    tight = (ratio > 0.5) & ~deficit
    print(
        f"\n{B}x{B} blocks: {nb}x{mb} | DEFICIT (need>have, locally infeasible w/ frozen ring): "
        f"{int(deficit.sum())} | tight (>50% used): {int(tight.sum())} | "
        f"surplus (<50%): {int((ratio <= 0.5).sum())}"
    )
    print(
        "block utilization map (need/have; '#'>1 deficit, '+'>0.5 tight, '.' surplus, 'F' = folds in block):"
    )
    for i in range(nb):
        row = ""
        for j in range(mb):
            ch = "#" if deficit[i, j] else ("+" if tight[i, j] else ".")
            if folds_b[i, j] and ch == ".":
                ch = "F"
            row += ch
        print("   " + row)
    # transport distance: from each deficit/fold block to nearest surplus block
    surplus = ratio <= 0.5
    if surplus.any() and (deficit | (folds_b > 0)).any():
        dist = ndimage.distance_transform_edt(~surplus) * B
        src = deficit | (folds_b > 0)
        d = dist[src]
        print(
            f"\ntransport distance (px) from deficit/fold blocks to nearest surplus block: "
            f"min={d.min():.0f} median={np.median(d):.0f} max={d.max():.0f}"
        )
        print(
            "  -> a monotone ladder must move area across ~median px per cell; each isqp "
            "step is local, so expect O(distance/step) rungs unless the Hessian is non-local."
        )
    print(
        f"\nfold-block utilization: {[round(float(ratio[i, j]), 2) for i, j in zip(*np.where(folds_b > 0))][:12]}"
    )


if __name__ == "__main__":
    main()
