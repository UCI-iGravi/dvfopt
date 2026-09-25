"""Generator for ``source_space_pin_probe.ipynb``.

The notebook source is the percent-format string below (``# %%`` code cells,
``# %% [markdown]`` markdown cells). ``python _build_source_space_probe_nb.py``
syntax-checks every code cell and writes the notebook WITHOUT executing it.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SOURCE = r'''
# %% [markdown]
# # Source-space probe: is the z[0,24) wall sparse in `f = Δu`?
#
# **Claim under test.** A Laplacian-interpolated DVF is harmonic everywhere except at its
# Dirichlet pins, so the discrete Laplacian of the DVF — read from the DVF alone — should be
# ~0 except at the pin voxels. If so, the wall of B0039 z[0,24) (73k simplex-3D folds) is a
# few thousand bad entries in source space, and the 2026-09-21 data probe's pairwise
# Lipschitz filter (`|Δdisp| > c·|Δx|` → 1,635 folds) can be reproduced **without reading the
# correspondences**, as a DVF-only harmonic re-fill in front of the windowed engine.
#
# **Three tests, each with a kill criterion:**
#
# | # | Test | Dead if |
# |---|------|---------|
# | 1 | `‖Δu‖` is bimodal; its support is the pin set | histogram smeared / precision-recall vs the true pins poor |
# | 2 | the pairwise test on detected sources flags the pins the probe flagged | low overlap with the probe's dropped set |
# | 3 | harmonic re-fill of the dropped sources collapses the fold count | global re-fill far above the probe's 1,635; local radii never get close |
#
# **Scope.** Every *edit* reads only the DVF. The correspondence files are read in cells marked
# **[diagnostic]** purely to score the detection and to report the landmark residual beside the
# fold count — set `USE_GROUND_TRUTH = False` to skip them all.
#
# **Cost.** One CG Laplace solve on the 40-slice slab is ~100 s; the notebook does
# `1 + len(RADII)` of them. No fold solver runs here.

# %%
import os, sys, json, time, collections
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_utils import load_cohort_field, load_cohort_correspondences
from dvfopt.laplacian.solver import solveLaplacianFromCorrespondences
from dvfopt.jacobian.tetrahedron_sign import six_tet_min_volume_3d

# %% [markdown]
# ## Config

# %%
BRAIN = "B0039"
ZS, ZE = 40, 24            # slab z[0, ZS) is edited (plane ZS-1 stays pinned); folds are counted in z < ZE
THR = 0.01                 # simplex-3D fold threshold
FIELD_KEY = "B_all"        # which loaded field tests 2-3 run on: "raw" or "B_all" (the calibration re-solve).
                           # Measured 2026-09-22: raw fails test 1 on precision (tau 0.1: 101,610 detected for
                           # 8,381 pins, 8 % precision, 99.9 % recall — the production CG residual, rtol 1e-2,
                           # is not below the pin scale); B_all passes it exactly (8,379 / 8,381, precision 1.0)
TAU = 0.1                  # source-strength threshold (px); None = Otsu on log10 strength. Measured 2026-09-22:
                           # Otsu fails (the zero spike / the CG-residual mode dominate); 0.1 sits in the gap on
                           # B_all (pins >= 0.3, harmonic <= 3e-3) and gives ~97 % recall / ~95 % precision on raw
C_LIP, PAIR_RADIUS = 1.0, 60.0   # the probe's arm E: drop until no pair within 60 px has |Δd| > c·|Δx|
RADII = (2, 4, 8)          # local re-fill radii (6-connected dilation steps around each dropped source)
SHOW_Z = (2, 8, 16, 22)
MAX_SOURCES = 60_000       # the slab has ~8.4k true pin voxels; far more detections = test 1 failed
USE_GROUND_TRUTH = True    # [diagnostic] reads of the correspondence files

PROBE = ROOT / "benchmarks/output/probe_z0_24"          # the 2026-09-21 data probe's artefacts
OUT = ROOT / "benchmarks/output/probe_source_space"
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load
#
# `raw` is the production field. `B_all` (if the data probe's artefact is on disk) is the same
# slab re-solved by `dvfopt.laplacian` from all pins at `rtol=1e-4` — a **calibration field**
# whose stencil, tolerance and pin set are known exactly. If test 1 passes on `B_all` and fails
# on `raw`, the production field differs in stencil / spacing / tolerance / post-processing,
# not the idea.

# %%
fields = {"raw": np.asarray(load_cohort_field(BRAIN))[:, :ZS].astype(np.float64)}
if (PROBE / "B_all.npy").is_file():
    fields["B_all"] = np.load(PROBE / "B_all.npy").astype(np.float64)
_, D, H, W = fields["raw"].shape
print({k: v.shape for k, v in fields.items()}, "| max |dz| raw:", np.abs(fields["raw"][0]).max())

pin_mask = pin_disp = None
if USE_GROUND_TRUTH:                                    # [diagnostic]
    mp, fp = load_cohort_correspondences(BRAIN)
    keep = np.round(fp[:, 0]).astype(int) < ZS - 1      # same selection as the data probe
    mp, fp = mp[keep], fp[keep]
    pin_vox = np.clip(np.round(fp).astype(int), 0, [D - 1, H - 1, W - 1])
    pin_mask = np.zeros((D, H, W), bool)
    pin_mask[tuple(pin_vox.T)] = True
    pin_disp = (mp - fp)[:, 1:]                         # prescribed (dy, dx)
    print(f"true pins in slab: {len(fp)} correspondences -> {int(pin_mask.sum())} voxels")

# %% [markdown]
# ## Helpers
#
# `source_map` / the pairwise test / the cover / the re-fill now live in the library
# (`dvfopt.dvf.pins`, `dvfopt.dvf.refill`); `source_strength` below is the slab-specific wrapper.

# %%
from dvfopt.dvf.pins import greedy_cover, source_map, violating_pairs_pruned
from dvfopt.dvf.refill import harmonic_refill


def source_strength(phi):
    s = np.hypot(source_map(phi[1]), source_map(phi[2]))
    s[ZS - 1:] = 0.0            # the cut plane: its z+1 neighbour is outside the slab
    return s


def otsu_log(x, bins=256):
    # ponytail: Otsu is biased when one class is ~0.1 % of voxels; TAU overrides, and the
    # precision/recall sweep below shows how much the choice matters.
    h, e = np.histogram(np.log10(x[x > 0]), bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); m0 = s0 / np.maximum(w0, 1); m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return 10 ** c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def fold_metrics(phi, tag):
    v = six_tet_min_volume_3d(np.ascontiguousarray(phi[:, :ZE + 1]))[:ZE]
    return dict(arm=tag, folds_thr=int((v < THR).sum()), folds_zero=int((v < 0).sum()),
                min_v=float(v.min()), per_z=[int((v[z] < THR).sum()) for z in range(ZE)])


def consistent_subset(pts, d, c=C_LIP, radius=PAIR_RADIUS):
    """Greedy max-degree cover of the violation graph (the data probe's arm E), via the library.
    Returns (alive, n_violating_pairs). The recorded run used the old dict-based cover; ties may
    break differently here, so drop counts can differ by a handful from the logged figures."""
    bad = violating_pairs_pruned(pts, d, c, radius)
    return ~greedy_cover(len(pts), bad), len(bad)


def refill(phi, dirichlet, tag, rtol=1e-4):
    """Harmonic re-fill of every voxel outside `dirichlet`, Dirichlet values read FROM `phi`."""
    t = time.time()
    out = harmonic_refill(phi, dirichlet, rtol=rtol)
    print(f"{tag}: {int((~dirichlet).sum())} free voxels, {time.time() - t:.0f} s", flush=True)
    return out

# %% [markdown]
# ## Test 1 — is the field sparse in source space?
#
# Histogram of `log10 ‖Δu‖` over the slab. A pass looks like two well-separated modes: a huge
# one at the CG-residual level (harmonic voxels) and a small one orders of magnitude higher
# (pins). **[diagnostic]** With the true pins loaded, the histogram is split by class and the
# threshold is scored by precision / recall.

# %%
strength = {k: source_strength(v) for k, v in fields.items()}
taus = {k: (TAU if TAU is not None else otsu_log(s)) for k, s in strength.items()}
test1 = {}

fig, axes = plt.subplots(1, len(fields), figsize=(7 * len(fields), 4), squeeze=False)
for ax, (k, s) in zip(axes[0], strength.items()):
    body = s[:ZS - 1]
    logs = np.log10(np.maximum(body, 1e-16))
    bins = np.linspace(logs.min(), logs.max(), 200)
    if pin_mask is not None:
        pm = pin_mask[:ZS - 1]
        ax.hist(logs[~pm], bins=bins, alpha=0.6, label="non-pin voxels")
        ax.hist(logs[pm], bins=bins, alpha=0.6, label="true pin voxels [diagnostic]")
        det = body > taus[k]
        tp = int((det & pm).sum())
        test1[k] = dict(tau=float(taus[k]), detected=int(det.sum()), true_pins=int(pm.sum()),
                        precision=tp / max(int(det.sum()), 1), recall=tp / max(int(pm.sum()), 1))
    else:
        ax.hist(logs, bins=bins)
        test1[k] = dict(tau=float(taus[k]), detected=int((body > taus[k]).sum()))
    ax.axvline(np.log10(taus[k]), color="k", ls="--", label=f"tau = {taus[k]:.2e}")
    ax.set_yscale("log"); ax.set_xlabel("log10 ||Δu||"); ax.set_title(k); ax.legend()
plt.tight_layout(); plt.show()
pd.DataFrame(test1).T

# %%
# [diagnostic] how sensitive is the detection to tau?
if pin_mask is not None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, s in strength.items():
        body, pm = s[:ZS - 1], pin_mask[:ZS - 1]
        sweep = np.logspace(np.log10(max(body[body > 0].min(), 1e-12)), np.log10(body.max()), 60)
        P = [(body[pm] > t).sum() / max((body > t).sum(), 1) for t in sweep]
        R = [(body[pm] > t).mean() for t in sweep]
        ax.plot(R, P, marker=".", label=k)
    ax.set_xlabel("recall (true pins detected)"); ax.set_ylabel("precision"); ax.legend()
    ax.set_title("source detection vs the true pin set"); plt.show()

# %% [markdown]
# **Gate.** Tests 2-3 run on `fields[FIELD_KEY]`. If `raw` failed test 1 but `B_all` passed,
# set `FIELD_KEY = "B_all"` to validate the rest of the chain on the calibration field — and
# treat "why is the production field not harmonic off-pin" as the finding.

# %%
phi = fields[FIELD_KEY]
src_mask = strength[FIELD_KEY] > taus[FIELD_KEY]
src_mask[ZS - 1:] = False
n_src = int(src_mask.sum())
print(f"{FIELD_KEY}: {n_src} detected source voxels")
assert n_src <= MAX_SOURCES, "source map is not sparse (test 1 failed) — raise TAU or stop here"

# %% [markdown]
# ## Test 2 — the pairwise Lipschitz test, from the DVF alone
#
# Points = detected source voxels, displacement = the DVF's own `(dy, dx)` there. Same rule and
# same greedy cover as the data probe's arm E. **[diagnostic]** The probe's dropped set is
# recomputed from the true pins for comparison (it removed 5,934 of 8,381).

# %%
src_vox = np.argwhere(src_mask)
src_d = phi[1:][:, src_mask].T                               # (n, 2) = (dy, dx) at each source
alive, n_bad = consistent_subset(src_vox.astype(np.float64), src_d)
dropped_mask = np.zeros_like(src_mask); dropped_mask[tuple(src_vox[~alive].T)] = True
kept_mask = src_mask & ~dropped_mask
test2 = dict(sources=n_src, violating_pairs=int(n_bad), dropped=int((~alive).sum()), kept=int(alive.sum()))

if pin_mask is not None:                                     # [diagnostic]
    alive_gt, n_bad_gt = consistent_subset(fp, pin_disp)
    gt_dropped = np.zeros_like(src_mask); gt_dropped[tuple(pin_vox[~alive_gt].T)] = True
    inter, union = int((gt_dropped & dropped_mask).sum()), int((gt_dropped | dropped_mask).sum())
    test2.update(probe_violating_pairs=int(n_bad_gt), probe_dropped_vox=int(gt_dropped.sum()),
                 dropped_jaccard=inter / max(union, 1),
                 probe_dropped_recovered=inter / max(int(gt_dropped.sum()), 1))
pd.Series(test2)

# %%
# dropped sources per z (the probe removed ALL pins at z < 11)
fig, ax = plt.subplots(figsize=(8, 3))
z = np.arange(ZS - 1)
ax.bar(z, src_mask[:ZS - 1].sum(axis=(1, 2)), label="detected sources")
ax.bar(z, dropped_mask[:ZS - 1].sum(axis=(1, 2)), label="dropped")
ax.axvline(ZE - 0.5, color="k", ls=":"); ax.set_xlabel("z"); ax.set_yscale("log"); ax.legend(); plt.show()

# %% [markdown]
# ## Test 3 — the edit: harmonic re-fill of the dropped sources
#
# * **global** — Dirichlet set = kept sources + the far plane, everything else re-solved. This is
#   the DVF-only reproduction of the probe's filtered rebuild (target: ~1,635 folds). It moves
#   every voxel a little, so it is *not* a no-damage edit.
# * **local r** — only voxels within `r` dilation steps of a dropped source are re-filled (kept
#   sources inside stay pinned); everything else is bit-identical to the input. The 3D Green's
#   function decays like 1/r, so expect a gap to the global result — the question is how big.

# %%
far = np.zeros_like(src_mask); far[ZS - 1] = True
results = [fold_metrics(phi, f"A input ({FIELD_KEY})")]
edited = {}

edited["global"] = refill(phi, kept_mask | far, "global")
for r in RADII:
    free = ndimage.binary_dilation(dropped_mask, iterations=r) & ~kept_mask & ~far
    # ponytail: rtol is relative to a RHS dominated by the ~5.7M Dirichlet values; tighten if the
    # local fills look under-converged (folds_thr not monotone in r is the symptom).
    edited[f"local r={r}"] = refill(phi, ~free, f"local r={r}", rtol=1e-6)

for tag, out in edited.items():
    m = fold_metrics(out, tag)
    delta = np.linalg.norm(out[1:, :ZE] - phi[1:, :ZE], axis=0)
    m.update(moved_frac=float((delta > 1e-3).mean()), move_mean=float(delta.mean()), move_max=float(delta.max()))
    if pin_mask is not None:                                 # [diagnostic] landmark residual beside the fold count
        res = np.linalg.norm(out[1:][:, pin_vox[:, 0], pin_vox[:, 1], pin_vox[:, 2]].T - pin_disp, axis=1)
        m.update(pin_resid_median=float(np.median(res)), pin_within_10px=float((res <= 10).mean()))
    results.append(m)
    np.save(OUT / f"{FIELD_KEY}_{tag.replace(' ', '_').replace('=', '')}.npy", out.astype(np.float32))

if (PROBE / "E_pairwise_1.0.npy").is_file():                 # the probe's correspondence-side result, same metric
    results.append(fold_metrics(np.load(PROBE / "E_pairwise_1.0.npy").astype(np.float64), "E probe (filtered rebuild)"))

table = pd.DataFrame(results).drop(columns="per_z").set_index("arm")
json.dump(dict(config=dict(brain=BRAIN, field=FIELD_KEY, tau=float(taus[FIELD_KEY]), c=C_LIP, radius=PAIR_RADIUS),
               test1=test1, test2=test2, test3=results), open(OUT / f"results_{FIELD_KEY}.json", "w"), indent=1)
table

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
by = {r["arm"]: r for r in results}
a.plot(RADII, [by[f"local r={r}"]["folds_thr"] for r in RADII], "o-", label="local re-fill")
for arm, style in ((results[0]["arm"], "r--"), ("global", "g--"), ("E probe (filtered rebuild)", "k:")):
    if arm in by:
        a.axhline(by[arm]["folds_thr"], color=style[0], ls=style[1:], label=arm)
a.set_yscale("log"); a.set_xlabel("re-fill radius (dilation steps)"); a.set_ylabel(f"simplex-3D folds < {THR}, z < {ZE}")
a.set_title("how local can the edit be?"); a.legend()
for r in results:
    b.plot(range(ZE), r["per_z"], marker=".", label=r["arm"])
b.set_yscale("symlog"); b.set_xlabel("z"); b.set_ylabel("folds per slice"); b.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Slices
#
# Per z: displacement magnitude, the source map with kept (cyan) / dropped (red) sources, and the
# fold map before and after the global re-fill (a cell is drawn folded if its min tet volume < THR).

# %%
v_in = six_tet_min_volume_3d(np.ascontiguousarray(phi[:, :ZE + 1]))[:ZE]
v_out = six_tet_min_volume_3d(np.ascontiguousarray(edited["global"][:, :ZE + 1]))[:ZE]
fig, axes = plt.subplots(len(SHOW_Z), 4, figsize=(18, 4.2 * len(SHOW_Z)), squeeze=False)
for row, zc in zip(axes, SHOW_Z):
    row[0].imshow(np.hypot(phi[1, zc], phi[2, zc]), cmap="viridis"); row[0].set_title(f"z={zc}  |u| (px)")
    row[1].imshow(np.log10(np.maximum(strength[FIELD_KEY][zc], 1e-12)), cmap="magma"); row[1].set_title("log10 ||Δu||")
    for mask, color in ((kept_mask, "cyan"), (dropped_mask, "red")):
        yy, xx = np.nonzero(mask[zc]); row[1].scatter(xx, yy, s=3, c=color, linewidths=0)
    row[2].imshow(v_in[zc] < THR, cmap="gray_r"); row[2].set_title(f"folds in: {int((v_in[zc] < THR).sum())}")
    row[3].imshow(v_out[zc] < THR, cmap="gray_r"); row[3].set_title(f"folds after global re-fill: {int((v_out[zc] < THR).sum())}")
    for ax in row:
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Verdict (run 2026-09-22, `benchmarks/output/probe_source_space/`)
#
# * **Test 1 — sparse?** `B_all` YES, exactly: at `tau = 0.1` 8,379 of 8,381 pins detected,
#   precision 1.0, a 1.5-decade gap between the harmonic residual (≤ 3e-3) and the pins (≥ 0.3).
#   `raw` NO on precision: 99.9 % recall but 101,610 detections (8 %) — the production field's CG
#   residual (`rtol` 1e-2) is not below the pin scale, so the DVF-only read needs a field solved
#   to ~1e-4 (or a pin-scale threshold ~0.5-1 px at ~90 % precision; not pursued).
# * **Test 2 — the pairwise test from the DVF alone:** YES. 870,618 violating pairs (probe:
#   850,778); 6,816 sources dropped, recovering 96.4 % of the probe's 5,934 (Jaccard 0.81) — the
#   extra drops are the non-unique greedy cover plus rounded-voxel vs sub-voxel displacements.
# * **Test 3 — the edit:** global re-fill **73,012 → 1,263 folds** (min −300 → −1.22), BETTER
#   than the probe's correspondence-side 1,635, at a landmark residual median 3.4 px / 86 % within
#   10 px (the probe: 3.2 / 86 %). It moves 93 % of the voxels (mean 2.9 px). The local re-fill
#   FAILS: r = 2 / 4 / 8 leave 64k / 58k / 47k folds — the pulled pins' far field is the fold
#   source, and 1/r decay means no small bubble removes it. So the edit is global or nothing.
# * **Next:** the windowed engine on `B_all_global.npy` (1,263 sparse folds — its regime), with
#   the residual reported beside the certificate; then the same read on the production field
#   after a tight re-solve, and a volume-wide census of the source-space inconsistency.
'''


def main():
    nb = nbf.v4.new_notebook()
    for chunk in SOURCE.split('\n# %%')[1:]:
        head, _, body = chunk.partition('\n')
        if head.strip() == '[markdown]':
            text = '\n'.join(ln[2:] if ln.startswith('# ') else ln.lstrip('#') for ln in body.strip('\n').split('\n'))
            nb.cells.append(nbf.v4.new_markdown_cell(text))
        else:
            code = body.strip('\n')
            compile(code, f'<cell {len(nb.cells)}>', 'exec')   # syntax check only, nothing runs
            nb.cells.append(nbf.v4.new_code_cell(code))
    out = HERE / 'source_space_pin_probe.ipynb'
    nbf.write(nb, out)
    print(f'Wrote {out} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
