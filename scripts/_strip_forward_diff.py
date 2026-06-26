"""Strip all forward-/finite-difference content from the shoelace notebook.

Edits:
- TOC: drop the "Central-difference vs forward-difference Jdet" entry and
  promote the manual proof from h3 to a top-level section.
- Remove cells 2c268783 (section markdown) and 6b8af035 (forward-diff code+plot).
- In the manual-proof markdown (11fe0c39): promote to h2.
- In the manual-proof code (3eb6023f): strip the forward-diff contrast addendum.
- In the re-quantify code (79f62291): strip all jdet_fwd computation/prints.
- In the triangulated re-quantify (tri-requantify-code): same.
- In the summary (summary-md): drop the forward-diff row.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "shoelace-artifact-example.ipynb"


NEW_TOC = r"""<a id="sec-toc"></a>
## Table of contents

1. [Construction](#sec-construction)
2. [Visualisation on the DVF quad grid](#sec-viz)
3. [Manual proof — central-diff Jdet is positive at the four shoelace vertices](#sec-manual-proof)
4. [Verification 1 — Injectivity metric (monotonicity)](#sec-injectivity)
5. [Verification 2 — Neighborhood measure](#sec-neighborhood)
    - [The IFT radius formula](#sec-ift-formula)
6. [Correction — optimise for positive Jdet and positive shoelace](#sec-correction)
    - [Re-quantify the corrected field](#sec-requantify)
    - [Before / after — quad grid and diagnostic heatmaps](#sec-beforeafter)
7. [Strengthening — triangulated shoelace closes the bowtie loophole](#sec-triangulated)
    - [Diagonal-split intuition](#sec-tri-intuition)
    - [Re-quantify the triangulated correction](#sec-tri-requantify)
    - [Before / after — triangulated](#sec-tri-beforeafter)
8. [Summary](#sec-summary)"""


NEW_MANUAL_PROOF_MD = r"""<a id="sec-manual-proof"></a>
## Manual proof — central-diff Jdet is positive at the four shoelace vertices

Computing `jacobian_det2D` by hand at each of the four corners of the bowtie cell (TL = (3, 3), TR = (3, 4), BR = (4, 4), BL = (4, 3)):

```
Jdet(i, j) = (1 + ∂dx/∂x)(1 + ∂dy/∂y) − (∂dx/∂y)(∂dy/∂x)
```

with interior central differences

```
∂f/∂x at (i, j) = (f[i, j+1] − f[i, j-1]) / 2
∂f/∂y at (i, j) = (f[i+1, j] − f[i-1, j]) / 2
```

The central stencil samples a pixel's neighbours one step away on each side, so a perturbation at a single pixel is averaged across 2Δ. At pixel (3, 3), for example, `∂dx/∂x = (dx[3, 4] − dx[3, 2]) / 2 = (−1.2 − 0) / 2 = −0.6`, giving `Jdet = 1 + (−0.6) = 0.4 > 0` — the central stencil never even sees `dx[3, 3] = +1.2`, so the bowtie is invisible to it.

The cell below walks through the arithmetic for all four vertices and cross-checks the result against `numpy.gradient`.

[↑ Back to TOC](#sec-toc)"""


NEW_MANUAL_PROOF_CODE = r"""def cdiff_jdet(label, i, j):
    dxdx = (dx[i, j + 1] - dx[i, j - 1]) / 2
    dxdy = (dx[i + 1, j] - dx[i - 1, j]) / 2
    dydx = (dy[i, j + 1] - dy[i, j - 1]) / 2
    dydy = (dy[i + 1, j] - dy[i - 1, j]) / 2
    det = (1 + dxdx) * (1 + dydy) - dxdy * dydx
    print(f"  {label}  pixel ({i}, {j})")
    print(f"    dx/dx = (dx[{i},{j+1}] - dx[{i},{j-1}]) / 2 "
          f"= ({dx[i, j+1]:+.1f} - {dx[i, j-1]:+.1f}) / 2 = {dxdx:+.4f}")
    print(f"    dx/dy = (dx[{i+1},{j}] - dx[{i-1},{j}]) / 2 "
          f"= ({dx[i+1, j]:+.1f} - {dx[i-1, j]:+.1f}) / 2 = {dxdy:+.4f}")
    print(f"    dy/dx = (dy[{i},{j+1}] - dy[{i},{j-1}]) / 2 "
          f"= ({dy[i, j+1]:+.1f} - {dy[i, j-1]:+.1f}) / 2 = {dydx:+.4f}")
    print(f"    dy/dy = (dy[{i+1},{j}] - dy[{i-1},{j}]) / 2 "
          f"= ({dy[i+1, j]:+.1f} - {dy[i-1, j]:+.1f}) / 2 = {dydy:+.4f}")
    print(f"    Jdet  = ({1+dxdx:+.4f}) * ({1+dydy:+.4f}) "
          f"- ({dxdy:+.4f}) * ({dydx:+.4f})  =  {det:+.4f}")
    print()
    return det


print("Central-difference Jdet at the four vertices of the folded quad cell:\n")
j_tl = cdiff_jdet("TL", 3, 3)
j_tr = cdiff_jdet("TR", 3, 4)
j_br = cdiff_jdet("BR", 4, 4)
j_bl = cdiff_jdet("BL", 4, 3)

vals = (j_tl, j_tr, j_br, j_bl)
print(f"All four vertices strictly positive : {all(v > 0 for v in vals)}  "
      f"(values: {[round(v, 4) for v in vals]})")
print("Match np.gradient output            :",
      np.isclose(j_tl, jdet[3, 3]) and np.isclose(j_tr, jdet[3, 4])
      and np.isclose(j_br, jdet[4, 4]) and np.isclose(j_bl, jdet[4, 3]))

assert all(v > 0 for v in vals), "central-diff Jdet should be positive at every shoelace vertex"
"""


NEW_REQUANTIFY = r"""# --- Pixel Jdet + shoelace on the corrected field -----------------------
jdet_corr = np.squeeze(jacobian_det2D(phi_corr))
shoe_corr = np.squeeze(shoelace_det2D(phi_corr))

# --- Monotonicity diffs --------------------------------------------------
h_mono_c, v_mono_c = _monotonicity_diffs_2d(dy_corr, dx_corr)
d1_c, d2_c = _diagonal_monotonicity_diffs_2d(dy_corr, dx_corr)

# --- Cell-min Jdet + IFT radius -----------------------------------------
cell_min_corr = cell_min_jdet_2d(phi_corr)
ift_r_corr    = ift_radius_2d(phi_corr)


def _row(name, orig, corr, fmt="{:+.4f}"):
    return (f"  {name:<24s}  "
            f"{fmt.format(orig):>10s}   ->   {fmt.format(corr):>10s}")


print("Before / after comparison (min values):\n")
print(_row("pixel Jdet (central)",  jdet.min(),        jdet_corr.min()))
print(_row("shoelace area",         shoe.min(),        shoe_corr.min()))
print(_row("h_mono",                h_mono.min(),      h_mono_c.min()))
print(_row("v_mono",                v_mono.min(),      v_mono_c.min()))
print(_row("d1 (diag)",             d1.min(),          d1_c.min()))
print(_row("d2 (diag)",             d2.min(),          d2_c.min()))
print(_row("cell_min_jdet",         cell_min.min(),    cell_min_corr.min()))
print(_row("ift_radius",            ift_r.min(),       ift_r_corr.min(),
           fmt="{:.4f}"))

print("\nCounts of non-positive entries (before -> after):")
print(_row("pixel Jdet neg",        (jdet <= 0).sum(),
                                    (jdet_corr <= 0).sum(), fmt="{:d}"))
print(_row("shoelace neg",          (shoe <= 0).sum(),
                                    (shoe_corr <= 0).sum(), fmt="{:d}"))
print(_row("cell_min_jdet neg",     (cell_min <= 0).sum(),
                                    (cell_min_corr <= 0).sum(), fmt="{:d}"))
print(_row("h_mono neg",            (h_mono <= 0).sum(),
                                    (h_mono_c <= 0).sum(), fmt="{:d}"))

# --- What the enforced constraints actually guarantee -------------------
# The solver was asked for positive pixel Jdet + positive shoelace; those
# are what we can assert. The stronger edge-wise conditions (cell_min_jdet,
# h_mono) require `enforce_injectivity=True` — see discussion below.
assert (jdet_corr > 0).all(), "central-diff pixel Jdet must be positive"
assert (shoe_corr > 0).all(), "shoelace areas must be positive"

if (cell_min_corr > 0).all():
    print("\nBonus: cell_min_jdet_2d is also fully positive — the shoelace "
          "constraint happened to certify sub-pixel injectivity here.")
else:
    n_bad = int((cell_min_corr <= 0).sum())
    print(f"\nNote: {n_bad} cell(s) still fail cell_min_jdet_2d > 0. "
          "Shoelace positivity (signed area) is strictly weaker than "
          "edge-wise monotonicity / bilinear positivity — a quad can "
          "have positive net area while one edge still crosses. To close "
          "this gap, pass `enforce_injectivity=True` to the solver.")"""


NEW_TRI_REQ = r"""from dvfopt.jacobian import triangulated_shoelace_det2D
from dvfopt.jacobian.intersection import has_quad_self_intersections

# --- Pixel Jdet + shoelace families on the triangulated-corrected field -
jdet_tri  = np.squeeze(jacobian_det2D(phi_tri))
shoe_tri  = np.squeeze(shoelace_det2D(phi_tri))
T1T2_tri  = triangulated_shoelace_det2D(phi_tri)           # (2, H-1, W-1)
T1_tri, T2_tri = T1T2_tri[0], T1T2_tri[1]

# Monotonicity + cell-min + IFT radius.
h_mono_t, v_mono_t = _monotonicity_diffs_2d(dy_tri, dx_tri)
d1_t, d2_t         = _diagonal_monotonicity_diffs_2d(dy_tri, dx_tri)
cell_min_tri       = cell_min_jdet_2d(phi_tri)
ift_r_tri          = ift_radius_2d(phi_tri)


def _row3(name, orig, shoe_only, tri, fmt="{:+.4f}"):
    def f(v):
        if isinstance(v, float) and np.isnan(v):
            return "        --"
        return fmt.format(v)
    return f"  {name:<22s}  {f(orig):>10s}   {f(shoe_only):>10s}   {f(tri):>10s}"


print(f"  {'measure':<22s}  {'bowtie':>10s}   {'shoelace':>10s}   {'triangulated':>10s}")
print("  " + "-" * 70)
print(_row3("pixel Jdet (central)", jdet.min(),      jdet_corr.min(),     jdet_tri.min()))
print(_row3("shoelace area",        shoe.min(),      shoe_corr.min(),     shoe_tri.min()))
print(_row3("T1 (upper triangle)",  float("nan"),    float("nan"),        float(T1_tri.min())))
print(_row3("T2 (lower triangle)",  float("nan"),    float("nan"),        float(T2_tri.min())))
print(_row3("h_mono",               h_mono.min(),    h_mono_c.min(),      h_mono_t.min()))
print(_row3("v_mono",               v_mono.min(),    v_mono_c.min(),      v_mono_t.min()))
print(_row3("d1 (diag)",            d1.min(),        d1_c.min(),          d1_t.min()))
print(_row3("d2 (diag)",            d2.min(),        d2_c.min(),          d2_t.min()))
print(_row3("cell_min_jdet",        cell_min.min(), cell_min_corr.min(), cell_min_tri.min()))
print(_row3("ift_radius",           ift_r.min(),     ift_r_corr.min(),    ift_r_tri.min(),
            fmt="{:.4f}"))

print("\nCounts of non-positive entries (bowtie -> shoelace -> triangulated):")
print(_row3("pixel Jdet neg",       int((jdet <= 0).sum()),
                                    int((jdet_corr <= 0).sum()),
                                    int((jdet_tri <= 0).sum()), fmt="{:d}"))
print(_row3("shoelace neg",         int((shoe <= 0).sum()),
                                    int((shoe_corr <= 0).sum()),
                                    int((shoe_tri <= 0).sum()), fmt="{:d}"))
print(_row3("cell_min_jdet neg",    int((cell_min <= 0).sum()),
                                    int((cell_min_corr <= 0).sum()),
                                    int((cell_min_tri <= 0).sum()), fmt="{:d}"))
print(_row3("T1 neg",               float("nan"),   float("nan"),
                                    int((T1_tri <= 0).sum()), fmt="{:d}"))
print(_row3("T2 neg",               float("nan"),   float("nan"),
                                    int((T2_tri <= 0).sum()), fmt="{:d}"))

# Hard checks on what enforce_shoelace_triangulated guarantees:
assert (jdet_tri > 0).all(),                   "central-diff pixel Jdet must be positive"
assert (T1_tri  > 0).all(),                    "every T1 triangle must be positive"
assert (T2_tri  > 0).all(),                    "every T2 triangle must be positive"
assert (shoe_tri > 0).all(),                   "shoelace = T1 + T2 > 0 follows from tri > 0"
assert not has_quad_self_intersections(phi_tri), "deformed quad grid must be intersection-free"
print("\nAll triangulated assertions pass - the bowtie is fully unfolded.")"""


NEW_SUMMARY = r"""<a id="sec-summary"></a>
## Summary

### Diagnostic behaviour on the pre-correction bowtie

| Measure                       | Stencil                         | Flags the fold? |
|-------------------------------|---------------------------------|-----------------|
| `jacobian_det2D`              | central differences (pixel)     | **no**          |
| `shoelace_det2D`              | 4-corner signed area (per quad) | yes             |
| Monotonicity diffs (`h/v/d1/d2`) | forward differences          | yes             |
| `cell_min_jdet_2d`            | bilinear corner minimum         | yes             |
| `ift_radius_2d`               | σ_min / Hessian norm            | collapses locally |
| `triangulated_shoelace_det2D` | per-triangle signed areas (TL–BR split) | yes |

A solver that constrains only the central-difference pixel Jdet can accept a folded field.

### Three levels of geometric correction

| Constraint | What it guarantees | Blocks asymmetric bowties? |
|---|---|---|
| `jacobian_det2D > 0`                 | central-difference Jdet positive everywhere | **no** |
| `enforce_shoelace=True`              | per-quad **net** signed area > 0            | **no** (this notebook's headline artefact) |
| `enforce_shoelace_triangulated=True` | both triangle signed areas > 0              | **yes** |
| `enforce_injectivity=True`           | edge-wise monotonicity (h, v, d1, d2) > 0   | yes (strictly stronger: also certifies `cell_min_jdet > 0`) |

**Why triangulated shoelace works.** Shoelace positivity is the *signed net area* of a quad, so an asymmetric bowtie with lobes of unequal area survives it. Splitting each quad along its TL–BR diagonal into triangles $T_1$ and $T_2$ removes the cancellation — each triangle's signed area is a direct orientation test on its three corners. Requiring both positive means every triangle preserves orientation, so no self-intersecting quad can slip through.

The triangulated variant is the minimal strengthening that closes the bowtie loophole while still being polynomial and (unlike full injectivity / monotonicity) purely *cell-local*.

### Takeaway

Pick the constraint by what you need:

* **Only worried about central-diff Jdet sign?** Default solver.
* **Need geometric fold-free quads, fine with rare bowtie survivors?** `enforce_shoelace=True`.
* **Need truly non-self-intersecting quads without the asymmetric-bowtie gap?** `enforce_shoelace_triangulated=True`.
* **Need full sub-pixel bilinear injectivity?** `enforce_injectivity=True`.

[↑ Back to TOC](#sec-toc)"""


REPLACE = {
    "f9637b1b": ("markdown", NEW_TOC),
    "11fe0c39": ("markdown", NEW_MANUAL_PROOF_MD),
    "3eb6023f": ("code", NEW_MANUAL_PROOF_CODE),
    "79f62291": ("code", NEW_REQUANTIFY),
    "tri-requantify-code": ("code", NEW_TRI_REQ),
    "summary-md": ("markdown", NEW_SUMMARY),
}

REMOVE_IDS = {"2c268783", "6b8af035"}


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # Drop the central-vs-forward section header + plot cell.
    nb["cells"] = [c for c in nb["cells"] if c.get("id") not in REMOVE_IDS]

    # Replace cells by id.
    for c in nb["cells"]:
        cid = c.get("id")
        if cid in REPLACE:
            kind, src = REPLACE[cid]
            if c["cell_type"] != kind:
                raise SystemExit(f"cell {cid}: expected {kind}, got {c['cell_type']}")
            c["source"] = src.splitlines(keepends=True)
            if kind == "code":
                c["outputs"] = []
                c["execution_count"] = None

    missing = set(REPLACE) - {c.get("id") for c in nb["cells"]}
    if missing:
        raise SystemExit(f"missing cells: {missing}")

    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NB}  - {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
