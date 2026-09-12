"""Phase-4 full-volume driver for the 3D windowed engine: banded or serial runs of
:func:`windowed_correct_banded` / :func:`windowed_correct` on a whole ``(3, D, H, W)``
volume, plus the two artefact builders the measurement ladder needs.

    python benchmarks/windowed_3d_full.py --build-ds2 B0039                  # cohort_ds2/B0039_laplacian_exterior_ds2.npy
    python benchmarks/windowed_3d_full.py --cut-slab 0 64                    # crops_3d/slab_0_64.npy
    python benchmarks/windowed_3d_full.py --run PATH --tag TAG [--band 24 --overlap 8 --n-workers 4 --checkpoint DIR]
    python benchmarks/windowed_3d_full.py --run PATH --tag TAG --serial [--checkpoint DIR]
    python benchmarks/windowed_3d_full.py --table                            # full.md

``--build-ds2`` / ``--cut-slab`` default to the repo-relative cohort field / raw B0039
field; ``--src PATH`` overrides that (needed when running from a detached snapshot
worktree whose ``data/dvfs/cohort`` is empty). ``--run`` accepts ``.npy`` and cohort
``.npz`` (key ``arr``) inputs, loaded via ``mmap_mode='r'`` and materialised as float64
only for the run; the corrected field and its record are always written, even under
``--serial``. Objective is fixed at L2 (the h2h convention).
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from benchmark_utils import load_cohort_field

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.wallbreakers._multiscale_3d import _downsample_2x
from dvfopt.core.windowed import windowed_correct, windowed_correct_banded
from dvfopt.io.fields import save_dvf
from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal, six_tet_min_volume_3d
from dvfopt.objectives import L2Objective

OUT = os.path.join("benchmarks", "output", "windowed_3d")
DS2 = os.path.join("data", "dvfs", "cohort_ds2")
CROPS = os.path.join("data", "dvfs", "crops_3d")
RESULTS = os.path.join("data", "dvfs", "results", "windowed_3d_full")
RAW = os.path.join("data", "dvfs", "b0039", "b0039_laplacian_deformation_field.npy")
THR = 0.01  # same value as windowed_3d_gate.THR; not imported from there to avoid its
# import-time instrumentation (QP-solve spying/printing, unbounded EXITS/QP_LOG growth) --
# unwanted on a real full-volume run with no use for that instrumentation.


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _commit():
    out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_repo_root())
    return out.decode().strip()


def _fold_report(phi):
    """(fold_mask, n_folds, n_folds_zero, floor, min) — one ``six_tet_min_volume_3d`` pass."""
    mv = six_tet_min_volume_3d(phi)
    fold_mask = mv < THR
    n_folds_zero = int((mv <= 0).sum())
    floor = int(n_neg_best_diagonal(phi, THR))
    return fold_mask, int(fold_mask.sum()), n_folds_zero, floor, float(mv.min())


def build_ds2(brain, variant="laplacian_exterior", src=None):
    if src is None:
        vol = np.asarray(load_cohort_field(brain, variant), dtype=np.float64)
    else:
        vol = _load(src)
    ds = _downsample_2x(vol)
    assert ds is not None, f"{brain}/{variant}: volume too small to downsample 2x"
    os.makedirs(DS2, exist_ok=True)
    path = os.path.join(DS2, f"{brain}_{variant}_ds2.npy")
    np.save(path, ds)
    _mask, folds, folds_zero, floor, min_v = _fold_report(ds)
    print(
        f"{path}: shape {ds.shape} folds {folds} folds_zero {folds_zero} floor {floor} "
        f"min {min_v:.4g}",
        flush=True,
    )
    return path


def cut_slab(z0, z1, src=None):
    path = src if src is not None else RAW
    vol = np.load(path, mmap_mode="r")
    slab = np.asarray(vol[:, z0:z1], dtype=np.float64)
    os.makedirs(CROPS, exist_ok=True)
    out_path = os.path.join(CROPS, f"slab_{z0}_{z1}.npy")
    np.save(out_path, slab)
    print(f"{out_path}: shape {slab.shape}", flush=True)
    return out_path


def _load(path):
    lower = path.lower()
    if lower.endswith(".npz"):
        with np.load(path) as z:
            arr = z["arr"]
            return np.asarray(arr, dtype=np.float64)
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr, dtype=np.float64)


def run(path, tag, band, overlap, n_workers, checkpoint_dir, serial):
    phi = _load(path)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    fold_in_mask, folds_in, folds_in_zero, floor_in, min_in = _fold_report(phi)
    t = time.perf_counter()
    if serial:
        out, rep = windowed_correct(
            phi.copy(),
            "isqp",
            constraint=c,
            objective=L2Objective(),
            threshold=THR,
            checkpoint_dir=checkpoint_dir,
            verbose=1,
        )
        mode = "serial"
        bands, band_walls, seam_windows, seam_folds_before = 0, [], -1, -1
    else:
        out, rep = windowed_correct_banded(
            phi.copy(),
            "isqp",
            constraint=c,
            objective=L2Objective(),
            threshold=THR,
            band=band,
            overlap=overlap,
            n_workers=n_workers,
            checkpoint_dir=checkpoint_dir,
            verbose=1,
        )
        mode = "banded"
        bands = rep.bands
        band_walls = rep.band_walls
        seam_windows = rep.seam_windows
        seam_folds_before = rep.seam_folds_before
    wall = time.perf_counter() - t
    fold_out_mask, folds_out, folds_out_zero, floor_out, min_out = _fold_report(out)
    move = out - phi
    rec = dict(
        case=os.path.basename(path),
        shape=list(map(int, phi.shape[1:])),
        n_voxels=int(np.prod(phi.shape[1:])),
        mode=mode,
        band=int(band) if not serial else 0,
        overlap=int(overlap) if not serial else 0,
        n_workers=int(n_workers) if not serial else 1,
        folds_in=folds_in,
        folds_in_zero=folds_in_zero,
        floor_in=floor_in,
        min_in=min_in,
        folds_out=folds_out,
        folds_out_zero=folds_out_zero,
        floor_out=floor_out,
        min_out=min_out,
        new_folds=int((fold_out_mask & ~fold_in_mask).sum()),
        damage=int(rep.damage),
        moved_frac=float((np.abs(move).max(axis=0) > 1e-9).mean()),
        l1_move=float(np.abs(move).sum()),
        l2_move=float(np.linalg.norm(move.ravel())),
        n_windows=int(rep.n_windows),
        seam_windows=int(seam_windows),
        seam_folds_before=int(seam_folds_before),
        bands=int(bands),
        band_walls=list(map(float, band_walls)),
        wall_s=wall,
        resumed_from=str(rep.resumed_from),
        commit=_commit(),
    )
    os.makedirs(RESULTS, exist_ok=True)
    save_dvf(os.path.join(RESULTS, f"{tag}.npy"), out)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"full_{tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec), flush=True)
    return rec


COLS = [
    "case",
    "mode",
    "band",
    "overlap",
    "n_workers",
    "folds_in",
    "folds_out",
    "floor_out",
    "new_folds",
    "damage",
    "moved_frac",
    "n_windows",
    "seam_windows",
    "wall_s",
    "l2_move",
]


def table():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("full_") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                rows.append(json.load(fh))
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        cells = []
        for k in COLS:
            v = r.get(k, "")
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "full.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-ds2", metavar="BRAIN")
    ap.add_argument("--variant", default="laplacian_exterior")
    ap.add_argument("--cut-slab", type=int, nargs=2, metavar=("Z0", "Z1"))
    ap.add_argument("--src", help="override the default source for --build-ds2 / --cut-slab")
    ap.add_argument("--run", metavar="PATH")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--band", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--checkpoint", metavar="DIR")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.build_ds2:
        build_ds2(a.build_ds2, a.variant, a.src)
    if a.cut_slab:
        cut_slab(a.cut_slab[0], a.cut_slab[1], a.src)
    if a.run:
        run(a.run, a.tag, a.band, a.overlap, a.n_workers, a.checkpoint, a.serial)
    if a.table:
        table()


if __name__ == "__main__":
    main()
