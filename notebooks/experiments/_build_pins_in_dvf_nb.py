"""Generator for ``pins_in_dvf.ipynb`` (percent-format source below; cells syntax-checked, not run)."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SOURCE = r'''
# %% [markdown]
# # Reading a DVF's pins out of the DVF
#
# A Laplacian-interpolated field is discrete-harmonic except at its Dirichlet pins, so the
# graph Laplacian of the field itself, `‖Δu‖`, is ~0 at every free voxel and spikes at the pins.
# `dvfopt.dvf.pins` turns that into three things, none of which read a correspondence file:
#
# 1. `detect_pins` — where the pins are;
# 2. the field's own displacement at each pin — what each pin prescribes;
# 3. `inconsistent_pins` — which pins contradict their neighbours (`|Δd| > c·|Δx|`, i.e. two
#    pins that would have to cross) and a small set whose removal leaves no contradiction.
#
# Requirement: the field must be solved tightly enough that the CG residual sits below the pin
# scale (~`rtol` 1e-4). The production cohort fields (`rtol` 1e-2) are not readable this way —
# see the `source_space_pin_probe` notebook — so this notebook takes a field path and, for a
# cohort field, offers a tight re-solve first.

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "benchmarks"))
from dvfopt.dvf.pins import source_strength, detect_pins, inconsistent_pins

# %% [markdown]
# ## Input
#
# Either a `.npy` field `(3, D, H, W)` / `(2, H, W)`, or a cohort brain (re-solved tightly on a
# z-slab). Set one of the two.

# %%
FIELD_PATH = ROOT / "benchmarks/output/probe_z0_24/B_all.npy"   # any tightly solved field; None to use the cohort path below
COHORT = dict(brain="B0039", z0=0, z1=40, rtol=1e-4)            # used only when FIELD_PATH is None
TAU = 0.1
C_LIP, PAIR_RADIUS = 1.0, 60.0
SHOW_Z = (10, 16, 22)

if FIELD_PATH is not None and Path(FIELD_PATH).is_file():
    phi = np.load(FIELD_PATH).astype(np.float64)
else:
    # tight re-solve of a cohort slab from ITS OWN pins is not possible without the pins, so
    # this branch reads the correspondences ONCE to build a readable field — calibration only.
    from benchmark_utils import load_cohort_field, load_cohort_correspondences
    from dvfopt.laplacian.solver import solveLaplacianFromCorrespondences
    z0, z1 = COHORT["z0"], COHORT["z1"]
    raw = np.asarray(load_cohort_field(COHORT["brain"]))[:, z0:z1].astype(np.float64)
    mp, fp = load_cohort_correspondences(COHORT["brain"])
    keep = (np.round(fp[:, 0]) >= z0) & (np.round(fp[:, 0]) < z1 - 1)
    mp, fp = mp[keep] - [z0, 0, 0], fp[keep] - [z0, 0, 0]
    H, W = raw.shape[2:]
    yy, xx = np.mgrid[0:H, 0:W]
    pf = np.column_stack([np.full(H * W, z1 - z0 - 1.0), yy.ravel(), xx.ravel()])
    pm = pf.copy(); pm[:, 1] += raw[1, -1].ravel(); pm[:, 2] += raw[2, -1].ravel()
    phi = solveLaplacianFromCorrespondences(raw.shape[1:], np.vstack([mp, pm]), np.vstack([fp, pf]),
                                            rtol=COHORT["rtol"], maxiter=4000)
print("field", phi.shape)

# %% [markdown]
# ## 1. Where are the pins?

# %%
s = source_strength(phi)
if phi.ndim == 4:
    s[-1] = 0.0            # a slab's cut plane has a missing neighbour; ignore it
pins = s > TAU
print(f"{int(pins.sum())} pins at tau = {TAU}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(np.log10(np.maximum(s[s > 0], 1e-16)), bins=200)
ax.axvline(np.log10(TAU), color="k", ls="--", label=f"tau = {TAU}")
ax.set_yscale("log"); ax.set_xlabel("log10 ||Δu||"); ax.set_ylabel("voxels"); ax.legend()
ax.set_title("two modes = readable; a smear = solved too loosely"); plt.show()

# %% [markdown]
# ## 2. What does each pin prescribe?
#
# The field's own displacement at each pin voxel. Arrows are drawn from the pin.

# %%
coords, drop, bad = inconsistent_pins(phi, pins, c=C_LIP, radius=PAIR_RADIUS)
d = phi[(slice(None), *coords.T)].T                      # (n, C)
mag = np.linalg.norm(d[:, -2:], axis=1)                  # in-plane |(dy, dx)|
print(f"{len(bad)} violating pairs; {int(drop.sum())} of {len(coords)} pins in the greedy cover")
print(f"prescribed |d| median {np.median(mag):.2f} px, p95 {np.percentile(mag, 95):.1f}, max {mag.max():.1f}")

# %%
def show_slice(z=None):
    sel = np.ones(len(coords), bool) if z is None else coords[:, 0] == z
    img = np.hypot(*phi[-2:]) if z is None else np.hypot(phi[-2, z], phi[-1, z])
    c = coords[sel][:, -2:]; dd = d[sel][:, -2:]; dr = drop[sel]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img, cmap="gray"); ax.set_title(f"z={z}  |u| (px); arrows = prescribed displacement; red = in the cover")
    ax.quiver(c[~dr, 1], c[~dr, 0], dd[~dr, 1], dd[~dr, 0], color="cyan", angles="xy", scale_units="xy", scale=1, width=0.002)
    ax.quiver(c[dr, 1], c[dr, 0], dd[dr, 1], dd[dr, 0], color="red", angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.set_xticks([]); ax.set_yticks([]); plt.show()

for zc in (SHOW_Z if phi.ndim == 4 else (None,)):
    show_slice(zc)

# %% [markdown]
# ## 3. Which pins contradict each other?
#
# Violation graph on the pins: an edge joins two pins closer than `PAIR_RADIUS` whose
# displacements differ by more than `C_LIP` × their separation. Pins are coloured by degree;
# the cover (red) is the small set whose removal leaves no edge.

# %%
deg = np.bincount(bad.ravel(), minlength=len(coords)) if len(bad) else np.zeros(len(coords), int)
if phi.ndim == 4:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    zs = np.arange(phi.shape[1])
    axes[0].bar(zs, np.bincount(coords[:, 0], minlength=len(zs)), label="pins")
    axes[0].bar(zs, np.bincount(coords[drop, 0], minlength=len(zs)), label="in the cover")
    axes[0].set_xlabel("z"); axes[0].legend(); axes[0].set_title("pins per slice")
    axes[1].hist(deg[deg > 0], bins=50); axes[1].set_yscale("log")
    axes[1].set_xlabel("violating pairs per pin"); axes[1].set_title("degree in the violation graph")
    plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(11, 8))
proj = coords[:, -2:]
sc = ax.scatter(proj[:, 1], proj[:, 0], c=np.log10(deg + 1), s=6, cmap="viridis")
ax.scatter(proj[drop, 1], proj[drop, 0], s=10, facecolors="none", edgecolors="red", linewidths=0.5, label="cover")
ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend(); plt.colorbar(sc, label="log10(1 + degree)")
ax.set_title("all pins, projected in-plane, coloured by how many neighbours they contradict"); plt.show()

# %% [markdown]
# ## 4. A contradiction, up close
#
# The highest-degree pin and its violating partners: the arrows show why no smooth field can
# satisfy both — the prescribed displacements differ by more than the distance between the pins.

# %%
if len(bad):
    worst = int(np.argmax(deg))
    partners = np.unique(bad[(bad == worst).any(axis=1)].ravel())
    z = coords[worst, 0] if phi.ndim == 4 else None
    c0 = coords[worst]; r = 40
    y0, y1 = max(c0[-2] - r, 0), c0[-2] + r; x0, x1 = max(c0[-1] - r, 0), c0[-1] + r
    img = (np.hypot(phi[-2, z], phi[-1, z]) if z is not None else np.hypot(*phi[-2:]))[y0:y1, x0:x1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", extent=(x0, x1, y1, y0))
    near = partners if z is None else partners[coords[partners, 0] == z]
    ax.quiver(coords[near, -1], coords[near, -2], d[near, -1], d[near, -2], color="orange", angles="xy", scale_units="xy", scale=1)
    ax.quiver(c0[-1], c0[-2], d[worst, -1], d[worst, -2], color="red", angles="xy", scale_units="xy", scale=1, width=0.01)
    ax.set_title(f"pin {tuple(map(int, c0))}: |d| = {mag[worst]:.1f} px, contradicts {deg[worst]} pins "
                 f"({len(near)} shown on this slice)"); plt.show()
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
            compile(code, f'<cell {len(nb.cells)}>', 'exec')
            nb.cells.append(nbf.v4.new_code_cell(code))
    out = HERE / 'pins_in_dvf.ipynb'
    nbf.write(nb, out)
    print(f'Wrote {out} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
