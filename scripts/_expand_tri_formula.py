"""Expand the triangulated-shoelace markdown with explicit formulas + prose.

Kept KaTeX-safe: no `\boxed{\begin{aligned}...}` (KaTeX refuses that combo),
no `{+}` spacing tricks, no `\\Bigl`/`\\Bigr` pairs — plain `aligned` blocks only.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "shoelace-artifact-example.ipynb"


TRI_HEADER_MD = r"""<a id="sec-triangulated"></a>
## Strengthening — triangulated shoelace closes the bowtie loophole

The plain shoelace constraint failed to unfold the bowtie because the shoelace area is the **signed net area** of a quad: if one lobe of a bowtie is larger than the other, the net area is still positive even though the quad self-intersects.

<a id="sec-tri-intuition"></a>
### Enhanced shoelace constraint — setup and formula

**Setup.** For each grid cell $(r, c)$, the four deformed corners are

$$
\begin{aligned}
\mathrm{TL} &= (\,c + dx[r, c],\ \ r + dy[r, c]\,) \\
\mathrm{TR} &= (\,c + 1 + dx[r, c+1],\ \ r + dy[r, c+1]\,) \\
\mathrm{BR} &= (\,c + 1 + dx[r+1, c+1],\ \ r + 1 + dy[r+1, c+1]\,) \\
\mathrm{BL} &= (\,c + dx[r+1, c],\ \ r + 1 + dy[r+1, c]\,).
\end{aligned}
$$

Write $(x_k, y_k)$ for $k \in \{\mathrm{TL}, \mathrm{TR}, \mathrm{BR}, \mathrm{BL}\}$ below.

**Plain shoelace (what failed).** The signed area of the deformed quad is

$$
A_{\mathrm{quad}} \;=\; \tfrac{1}{2}\,\bigl[\,(x_0 y_1 - x_1 y_0) + (x_1 y_2 - x_2 y_1) + (x_2 y_3 - x_3 y_2) + (x_3 y_0 - x_0 y_3)\,\bigr],
$$

with the corners ordered TL → TR → BR → BL. Requiring $A_{\mathrm{quad}} > 0$ catches most folds, but a self-intersecting bowtie has two lobes of opposite orientation that cancel in this sum — so an asymmetric bowtie can satisfy $A_{\mathrm{quad}} > 0$ while still crossing itself.

**Triangulated formulation (what we use).** Split each quad along its TL → BR diagonal into two triangles, and require *each* triangle to have positive signed area separately. Using the 2-D cross-product form (twice the signed triangle area):

$$
\begin{aligned}
T_1(r, c) &= \tfrac{1}{2}\,\bigl[\,(x_{\mathrm{TR}} - x_{\mathrm{TL}})(y_{\mathrm{BR}} - y_{\mathrm{TL}}) \;-\; (x_{\mathrm{BR}} - x_{\mathrm{TL}})(y_{\mathrm{TR}} - y_{\mathrm{TL}})\,\bigr], \\[2pt]
T_2(r, c) &= \tfrac{1}{2}\,\bigl[\,(x_{\mathrm{BR}} - x_{\mathrm{TL}})(y_{\mathrm{BL}} - y_{\mathrm{TL}}) \;-\; (x_{\mathrm{BL}} - x_{\mathrm{TL}})(y_{\mathrm{BR}} - y_{\mathrm{TL}})\,\bigr].
\end{aligned}
$$

By construction $A_{\mathrm{quad}} = T_1 + T_2$, so the triangulated constraint is strictly stronger than the plain shoelace.

**The SLSQP constraint.** For every cell $(r, c)$ in the sub-window and a threshold $\tau$ (default $10^{-2}$),

$$
T_1(r, c) \;\geq\; \tau \qquad \text{and} \qquad T_2(r, c) \;\geq\; \tau.
$$

These two inequality families are added to the optimiser alongside the usual $\mathrm{Jdet} \geq \tau$, with an analytic sparse Jacobian (six non-zeros per row — three $dx$'s and three $dy$'s, since each triangle touches only three of the four corners).

**Why this works in plain English.**

* Each triangle's signed area is a 2-D cross product of two edges. Its sign is the orientation of the triangle (counter-clockwise = positive). There is no second lobe to cancel against, so a negative value is an unambiguous certificate that the three corners have been twisted out of order.
* A quad with both $T_1 > 0$ and $T_2 > 0$ has **every** triangle (and therefore every edge of those triangles) in its original orientation, so nothing has crossed. The asymmetric-bowtie loophole — positive net area but one lobe flipped — is closed because the flipped lobe is now one of the triangles, and it must be positive on its own.
* The diagonal split is arbitrary (we pick TL → BR); any fixed diagonal choice works because a planar quad self-intersects iff at least one of its two diagonals produces a flipped triangle, and the TL → BR split flips iff the quad fails.

This is exposed as `enforce_shoelace_triangulated=True` on `iterative_serial`, backed by `triangulated_shoelace_det2D` and `triangulated_shoelace_constraint_jacobian_2d`.

[↑ Back to TOC](#sec-toc)"""


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "tri-header-md":
            if c["cell_type"] != "markdown":
                raise SystemExit("tri-header-md is not markdown")
            c["source"] = TRI_HEADER_MD.splitlines(keepends=True)
            break
    else:
        raise SystemExit("tri-header-md not found")
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NB}")


if __name__ == "__main__":
    main()
