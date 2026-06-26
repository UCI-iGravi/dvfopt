"""Insert close-up + signed-area + neighborhood cells into 08_global-invertibility-gap.ipynb."""

import json

path = "notebooks/two-triangle-check/08_global-invertibility-gap.ipynb"
nb = json.load(open(path, encoding="utf-8"))


def md(text, cid):
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text, cid):
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# Insert after 'zoom-code' (or, if the prior run already inserted, after the last of that group).
anchor = "zoom-code"
for candidate in ("signed-area-code", "closeup-code"):
    for c in nb["cells"]:
        if c.get("id") == candidate:
            anchor = candidate
            break

insert_after = None
for idx, c in enumerate(nb["cells"]):
    if c.get("id") == anchor:
        insert_after = idx

# Remove any previous closeup/signed-area cells so we don't stack duplicates on repeated runs.
nb["cells"] = [
    c
    for c in nb["cells"]
    if c.get("id") not in {"closeup-md", "closeup-code", "signed-area-md", "signed-area-code"}
]

# Recompute insert point (after the final remaining anchor, typically zoom-code).
for idx, c in enumerate(nb["cells"]):
    if c.get("id") == "zoom-code":
        insert_after = idx

closeup_md = md(
    "### Close-up with neighborhood: is the local neighborhood locally invertible?\n"
    "\n"
    "For each intersecting pair `(A, B)` we zoom to a bounding box covering both cells *plus a radius-2 ring of neighboring cells around each*. The two focus cells (A in red, B in blue) are filled to show their overlap. Neighborhood cells are drawn as outlines only: **dark blue if the 2-triangle check still flags them**, **faint gray if they're locally valid**. After the best 2-triangle SLSQP run, every neighborhood cell should be gray - confirming the local neighborhood is clean everywhere around the intersection point.\n"
    "\n"
    "Each quad is split along its TR-BL diagonal (dashed) and the two triangles in A and B are annotated with their signed areas. All four must be `>= threshold` by construction (SLSQP convergence). The overlap area is computed via a Sutherland-Hodgman convex-polygon clip and printed in the panel title.",
    "closeup-md",
)

closeup_code = code(
    "def _sutherland_hodgman(subject, clip):\n"
    "    def inside(p, a, b):\n"
    "        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])\n"
    "    def intersect(p1, p2, a, b):\n"
    "        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]\n"
    "        dx2, dy2 = b[0]  - a[0],  b[1]  - a[1]\n"
    "        denom = dx1 * dy2 - dy1 * dx2\n"
    "        if denom == 0:\n"
    "            return p2\n"
    "        t = ((a[0] - p1[0]) * dy2 - (a[1] - p1[1]) * dx2) / denom\n"
    "        return (p1[0] + t * dx1, p1[1] + t * dy1)\n"
    "    def orientation(poly):\n"
    "        s = 0.0\n"
    "        for i in range(len(poly)):\n"
    "            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]\n"
    "            s += (x2 - x1) * (y2 + y1)\n"
    "        return s\n"
    "    if orientation(clip) > 0:     clip = list(reversed(clip))\n"
    "    if orientation(subject) > 0:  subject = list(reversed(subject))\n"
    "    output = list(subject)\n"
    "    for i in range(len(clip)):\n"
    "        if not output:\n"
    "            break\n"
    "        A, B = clip[i], clip[(i + 1) % len(clip)]\n"
    "        inp = output; output = []\n"
    "        for j in range(len(inp)):\n"
    "            P = inp[j]; Q = inp[(j + 1) % len(inp)]\n"
    "            p_in = inside(P, A, B) >= 0; q_in = inside(Q, A, B) >= 0\n"
    "            if p_in:\n"
    "                output.append(P)\n"
    "                if not q_in:\n"
    "                    output.append(intersect(P, Q, A, B))\n"
    "            elif q_in:\n"
    "                output.append(intersect(P, Q, A, B))\n"
    "    return output\n"
    "\n"
    "def _polygon_area(poly):\n"
    "    n = len(poly)\n"
    "    if n < 3:\n"
    "        return 0.0\n"
    "    s = 0.0\n"
    "    for i in range(n):\n"
    "        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]\n"
    "        s += x1 * y2 - x2 * y1\n"
    "    return 0.5 * abs(s)\n"
    "\n"
    "\n"
    "def close_up_pair(phi, pair, tri_signed, ax, neighborhood_radius=2):\n"
    "    (ra, ca), (rb, cb) = pair\n"
    "    Hh, Ww = phi[0].shape\n"
    "    nr, nc = Hh - 1, Ww - 1\n"
    "    gx = phi[1] + np.mgrid[:Hh, :Ww][1]\n"
    "    gy = phi[0] + np.mgrid[:Hh, :Ww][0]\n"
    "    local_fold_mask = (tri_signed.min(axis=0) <= 0)\n"
    "\n"
    "    def quad(r, c):\n"
    "        return [(gx[r, c],      gy[r, c]),\n"
    "                (gx[r, c+1],    gy[r, c+1]),\n"
    "                (gx[r+1, c+1],  gy[r+1, c+1]),\n"
    "                (gx[r+1, c],    gy[r+1, c])]\n"
    "\n"
    "    # Neighborhood = radius-N ring around both A and B.\n"
    "    nbhd_cells = set()\n"
    "    for (r, c) in [(ra, ca), (rb, cb)]:\n"
    "        for dr in range(-neighborhood_radius, neighborhood_radius + 1):\n"
    "            for dc in range(-neighborhood_radius, neighborhood_radius + 1):\n"
    "                rr, cc = r + dr, c + dc\n"
    "                if 0 <= rr < nr and 0 <= cc < nc:\n"
    "                    nbhd_cells.add((rr, cc))\n"
    "    nbhd_cells.discard((ra, ca)); nbhd_cells.discard((rb, cb))\n"
    "\n"
    "    # Gather all vertex coords to compute close-up bounding box.\n"
    "    all_pts = []\n"
    "    for r, c in list(nbhd_cells) + [(ra, ca), (rb, cb)]:\n"
    "        all_pts.extend(quad(r, c))\n"
    "    pad = 0.25\n"
    "    xlim = (min(p[0] for p in all_pts) - pad, max(p[0] for p in all_pts) + pad)\n"
    "    ylim = (min(p[1] for p in all_pts) - pad, max(p[1] for p in all_pts) + pad)\n"
    "\n"
    "    # Draw neighborhood cell outlines (gray for valid, dark blue for any local fold).\n"
    "    n_nbhd_bad = 0\n"
    "    for (r, c) in nbhd_cells:\n"
    "        q = quad(r, c)\n"
    "        loop = q + [q[0]]\n"
    "        is_bad = bool(local_fold_mask[r, c])\n"
    "        col = '#1565c0' if is_bad else '#c0c0c0'\n"
    "        lw = 1.4 if is_bad else 0.7\n"
    "        ax.plot([p[0] for p in loop], [p[1] for p in loop],\n"
    "                color=col, lw=lw, zorder=2)\n"
    "        if is_bad:\n"
    "            n_nbhd_bad += 1\n"
    "\n"
    "    # Focus pair: filled quads.\n"
    "    qA = quad(ra, ca); qB = quad(rb, cb)\n"
    "    ax.fill([p[0] for p in qA], [p[1] for p in qA], color='#ef5350', alpha=0.45, zorder=3)\n"
    "    ax.fill([p[0] for p in qB], [p[1] for p in qB], color='#1565c0', alpha=0.45, zorder=3)\n"
    "    for (poly, col) in [(qA, '#c62828'), (qB, '#0d47a1')]:\n"
    "        loop = poly + [poly[0]]\n"
    "        ax.plot([p[0] for p in loop], [p[1] for p in loop], color=col, lw=2.0, zorder=4)\n"
    "        ax.plot([poly[1][0], poly[3][0]], [poly[1][1], poly[3][1]],\n"
    "                color=col, lw=0.9, ls='--', zorder=4)\n"
    "\n"
    "    # Triangle signed-area annotations (T1 = indices 1,3,2; T2 = indices 0,3,1).\n"
    "    def centroid(quad_vs, idx):\n"
    "        pts = [quad_vs[i] for i in idx]\n"
    "        return (sum(p[0] for p in pts)/3, sum(p[1] for p in pts)/3)\n"
    "    t1A, t2A = tri_signed[0, ra, ca], tri_signed[1, ra, ca]\n"
    "    t1B, t2B = tri_signed[0, rb, cb], tri_signed[1, rb, cb]\n"
    "    for (q, col, t1, t2) in [(qA, '#b71c1c', t1A, t2A), (qB, '#0d47a1', t1B, t2B)]:\n"
    "        cT1 = centroid(q, [1, 3, 2]); cT2 = centroid(q, [0, 3, 1])\n"
    "        ax.text(cT1[0], cT1[1], 'T1' + chr(10) + f'{t1:+.3f}',\n"
    "                 ha='center', va='center', color=col, fontsize=8, fontweight='bold', zorder=5)\n"
    "        ax.text(cT2[0], cT2[1], 'T2' + chr(10) + f'{t2:+.3f}',\n"
    "                 ha='center', va='center', color=col, fontsize=8, fontweight='bold', zorder=5)\n"
    "\n"
    "    # Overlap area via Sutherland-Hodgman\n"
    "    overlap = _sutherland_hodgman(qA, qB)\n"
    "    ov_area = _polygon_area(overlap)\n"
    "\n"
    "    ax.set_xlim(xlim); ax.set_ylim(ylim)\n"
    "    ax.invert_yaxis(); ax.set_aspect('equal')\n"
    "    title = (f'A=({ra},{ca})  B=({rb},{cb})' + chr(10) +\n"
    "              f'overlap={ov_area:.3f}   neighborhood folded cells: {n_nbhd_bad}')\n"
    "    ax.set_title(title, fontsize=9)\n"
    "    ax.set_xticks([]); ax.set_yticks([])\n"
    "    return ov_area, n_nbhd_bad\n"
    "\n"
    "\n"
    "# Pick three representative pairs across the fold\n"
    "if len(qi_out) >= 3:\n"
    "    picks = [qi_out[0], qi_out[len(qi_out) // 2], qi_out[-1]]\n"
    "else:\n"
    "    picks = qi_out[:]\n"
    "\n"
    "tri_after = r['tri']\n"
    "fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.2), layout='constrained')\n"
    "if len(picks) == 1:\n"
    "    axes = [axes]\n"
    "for ax, pair in zip(axes, picks):\n"
    "    close_up_pair(phi_out, pair, tri_after, ax, neighborhood_radius=2)\n"
    "plt.suptitle(f'{CASE_KEY} post-SLSQP: close-up of intersecting pairs + radius-2 neighborhood' + chr(10) +\n"
    "              'gray = locally valid, dark-blue = locally folded (should see no dark-blue rings)',\n"
    "              fontsize=10)\n"
    "plt.show()",
    "closeup-code",
)

sat_md = md(
    "### Signed-area table across all intersecting pairs + neighborhood check\n"
    "\n"
    "For every one of the intersecting pairs: the four signed triangle areas in the two cells, the computed overlap area, and the count of *neighborhood* cells (radius 2 around each of the two cells, i.e., up to ~48 surrounding cells) that are locally folded. The last column should read `0` on every row - confirming that each intersection happens in a region that is otherwise entirely locally invertible. The failure is strictly non-local.",
    "signed-area-md",
)

sat_code = code(
    "threshold = THRESHOLD\n"
    "tri_after = r['tri']\n"
    "local_fold_mask = tri_after.min(axis=0) <= 0\n"
    "nr_total, nc_total = local_fold_mask.shape\n"
    "\n"
    "def _radius_count(r, c, radius=2):\n"
    "    cells = set()\n"
    "    for dr in range(-radius, radius + 1):\n"
    "        for dc in range(-radius, radius + 1):\n"
    "            rr, cc = r + dr, c + dc\n"
    "            if 0 <= rr < nr_total and 0 <= cc < nc_total:\n"
    "                cells.add((rr, cc))\n"
    "    return cells\n"
    "\n"
    "gx_full = phi_out[1] + np.mgrid[:phi_out[0].shape[0], :phi_out[0].shape[1]][1]\n"
    "gy_full = phi_out[0] + np.mgrid[:phi_out[0].shape[0], :phi_out[0].shape[1]][0]\n"
    "\n"
    "def _quad_pts(r_, c_):\n"
    "    return [(gx_full[r_, c_],      gy_full[r_, c_]),\n"
    "            (gx_full[r_, c_+1],    gy_full[r_, c_+1]),\n"
    "            (gx_full[r_+1, c_+1],  gy_full[r_+1, c_+1]),\n"
    "            (gx_full[r_+1, c_],    gy_full[r_+1, c_])]\n"
    "\n"
    "print(f'threshold = {threshold}   (all four triangle areas below should be >= this)')\n"
    "print()\n"
    "hdr = (f\"  {'pair':<24s}  {'T1_A':>7s}  {'T2_A':>7s}  {'T1_B':>7s}  {'T2_B':>7s}  \"\n"
    "        f\"{'overlap':>8s}  {'nbhd_folds':>10s}\")\n"
    "print(hdr)\n"
    "print('-' * len(hdr))\n"
    "\n"
    "total_overlap = 0.0\n"
    "min_tri_across = +np.inf\n"
    "total_nbhd_bad = 0\n"
    "for (a, b) in qi_out:\n"
    "    t1a, t2a = tri_after[0, a[0], a[1]], tri_after[1, a[0], a[1]]\n"
    "    t1b, t2b = tri_after[0, b[0], b[1]], tri_after[1, b[0], b[1]]\n"
    "    qA = _quad_pts(*a); qB = _quad_pts(*b)\n"
    "    oa = _polygon_area(_sutherland_hodgman(qA, qB))\n"
    "    nbhd = _radius_count(a[0], a[1]) | _radius_count(b[0], b[1])\n"
    "    nbhd.discard(a); nbhd.discard(b)\n"
    "    n_bad = sum(1 for (rr, cc) in nbhd if local_fold_mask[rr, cc])\n"
    "    total_overlap += oa\n"
    "    total_nbhd_bad += n_bad\n"
    "    min_tri_across = min(min_tri_across, t1a, t2a, t1b, t2b)\n"
    "    print(f'  {str(a):>10s} <-> {str(b):>10s}  '\n"
    "          f'{t1a:>+7.3f}  {t2a:>+7.3f}  {t1b:>+7.3f}  {t2b:>+7.3f}  '\n"
    "          f'{oa:>8.3f}  {n_bad:>10d}')\n"
    "\n"
    "print()\n"
    "print(f'  min triangle signed area across all 4*{len(qi_out)} = {4*len(qi_out)} triangles in flagged pairs: {min_tri_across:+.4f}')\n"
    "print(f'  -> all >= threshold ({threshold}): {min_tri_across >= threshold}')\n"
    "print(f'  total neighborhood folded cells (summed across all pairs): {total_nbhd_bad}')\n"
    "print(f'  total overlap area across {len(qi_out)} pairs: {total_overlap:.3f}')",
    "signed-area-code",
)

nb["cells"] = (
    nb["cells"][: insert_after + 1]
    + [closeup_md, closeup_code, sat_md, sat_code]
    + nb["cells"][insert_after + 1 :]
)

# Compile check
for c in (closeup_code, sat_code):
    try:
        compile("".join(c["source"]), c["id"], "exec")
        print(f'{c["id"]}: compile OK')
    except SyntaxError as e:
        print(f'{c["id"]}: SYNTAX {e}')

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f'cells now {len(nb["cells"])}')
