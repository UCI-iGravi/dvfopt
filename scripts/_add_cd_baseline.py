"""Add CD-constraint SLSQP baseline + fix misleading [OK] tag on initial."""

import json

path = "notebooks/two-triangle-check/triangle-sign-solver-engineering.ipynb"
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


# ------------------------------------------------------------------
# 1. Insert CD-SLSQP baseline cell pair after se-preopt-code.
# ------------------------------------------------------------------
for idx, c in enumerate(nb["cells"]):
    if c.get("id") == "se-preopt-code":
        insert_idx = idx
        break

cd_md = md(
    "### Baseline SLSQP with central-diff constraint\n"
    "\n"
    "The three variants in Parts 4-5 below all use the same **2-triangle constraint**; they differ only in the Jacobian method (finite-diff vs. analytical) and the warm-start. For a different kind of comparison — a *different constraint formulation* altogether — we also run classic central-difference SLSQP: constraint is `jacobian_det2D(phi) ≥ threshold` per pixel.\n"
    "\n"
    "On the 7×7 bowtie (see `../shoelace-artifact-example.ipynb`) CD-SLSQP sees 0 folds and does nothing. On `01c_20x40_edges` the fold magnitudes are large enough that CD *does* see 32 of the 68 geometric folds, so SLSQP has something to correct. The question: how far can CD push it, and what does the 2-triangle check say about the result?",
    "se-cd-baseline-md",
)

cd_code_src = (
    "# Central-difference SLSQP — same data term and same solver, different constraint.\n"
    "from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d\n"
    "\n"
    "def _cd_flat(dy_, dx_):\n"
    "    return _numpy_jdet_2d(dy_, dx_).flatten()\n"
    "\n"
    "t0 = time.time()\n"
    "res_cd = minimize(\n"
    "    lambda z: L2Objective()(z - z0_init),\n"
    "    z0_init.copy(),\n"
    "    jac=True,\n"
    "    method='SLSQP',\n"
    "    constraints=[NonlinearConstraint(\n"
    "        lambda z: _cd_flat(*unpack(z)), THRESHOLD, np.inf\n"
    "    )],\n"
    "    options={'maxiter': 500, 'disp': False},\n"
    ")\n"
    "t_cd = time.time() - t0\n"
    "dy_cd, dx_cd = unpack(res_cd.x)\n"
    "phi_cd = np.stack([dy_cd, dx_cd])\n"
    "m_cd = measure(phi_cd)\n"
    "qi_cd = list_intersecting_quads(phi_cd)\n"
    "\n"
    "print('CD-constraint SLSQP:')\n"
    "print(f'  nit={res_cd.nit}  time={t_cd:.2f}s  success={res_cd.success}  status={res_cd.status}')\n"
    "print(f'  message: {res_cd.message}')\n"
    "print(f'  neg_CD={m_cd[\"n_cd\"]}   neg_FD={m_cd[\"n_fd\"]}   neg_TR={m_cd[\"n_tr\"]}   QI={len(qi_cd)}')\n"
    "print(f'  min_CD={m_cd[\"min_cd\"]:+.4f}   min_TR={m_cd[\"min_tr\"]:+.4f}')"
)
cd_code = code(cd_code_src, "se-cd-baseline-code")

nb["cells"] = nb["cells"][: insert_idx + 1] + [cd_md, cd_code] + nb["cells"][insert_idx + 1 :]

# ------------------------------------------------------------------
# 2. Update variant-viz-code: add CD column + fix [OK] tag on initial.
# ------------------------------------------------------------------
viz_src = (
    "def plot_warped_grid(ax, phi, title, highlight_folds=True):\n"
    "    dy = phi[0]; dx = phi[1]\n"
    "    Hh, Ww = dy.shape\n"
    "    yy, xx = np.mgrid[:Hh, :Ww]\n"
    "    gx = xx + dx; gy = yy + dy\n"
    "    for i in range(Hh):\n"
    "        ax.plot(xx[i], yy[i], color='#f0f0f0', lw=0.4)\n"
    "    for j in range(Ww):\n"
    "        ax.plot(xx[:, j], yy[:, j], color='#f0f0f0', lw=0.4)\n"
    "    for i in range(Hh):\n"
    "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.9)\n"
    "    for j in range(Ww):\n"
    "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.9)\n"
    "    if highlight_folds:\n"
    "        tri = triangle_sign_areas2D(phi)\n"
    "        bad = np.argwhere(tri.min(axis=0) <= 0)\n"
    "        for (cy, cx) in bad:\n"
    "            poly_x = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
    "            poly_y = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
    "            ax.plot(poly_x, poly_y, color='#1565c0', lw=1.6)\n"
    "    ax.set_aspect('equal')\n"
    "    ax.invert_yaxis()\n"
    "    ax.set_title(title, fontsize=9)\n"
    "    ax.set_xticks([]); ax.set_yticks([])\n"
    "\n"
    "\n"
    "variants = [\n"
    "    ('initial',                          phi_init, m0,     len(qi_init), None),\n"
    "    ('(A) CD-SLSQP',                     phi_cd,   m_cd,   len(qi_cd),   res_cd),\n"
    "    ('(B) 2tri finite-diff Jac',         phi_fd,   m_fd,   len(qi_fd),   res_fd),\n"
    "    ('(C) 2tri analytical Jac',          phi_an,   m_an,   len(qi_an),   res_an),\n"
    "    ('(D) 2tri analytical + warm-start', phi_both, m_both, len(qi_both), res_both),\n"
    "]\n"
    "\n"
    "vmax = max(abs(m['tri']).max() for (_, _, m, _, _) in variants)\n"
    "NL = chr(10)\n"
    "\n"
    "n_var = len(variants)\n"
    "fig, axes = plt.subplots(2, n_var, figsize=(4 * n_var, 7.5), layout='constrained')\n"
    "for k, (label, phi, m, qi_n, res) in enumerate(variants):\n"
    "    if res is None:\n"
    "        line1 = label\n"
    "    else:\n"
    "        tag = 'OK' if res.success else 'FAIL'\n"
    "        line1 = f'{label}  [{tag}]'\n"
    "    line2 = f\"TR={m['n_tr']}  QI={qi_n}  min_TR={m['min_tr']:+.3f}\"\n"
    "    plot_warped_grid(axes[0, k], phi, line1 + NL + line2)\n"
    "\n"
    "    tri_min = m['tri'].min(axis=0)\n"
    "    im = axes[1, k].imshow(tri_min, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')\n"
    "    axes[1, k].set_title(f'min(T1, T2)   {label}', fontsize=9)\n"
    "    axes[1, k].set_xticks([]); axes[1, k].set_yticks([])\n"
    "\n"
    "cbar = fig.colorbar(im, ax=axes[1, :], orientation='horizontal',\n"
    "                    fraction=0.035, pad=0.04, shrink=0.55)\n"
    "cbar.set_label('signed triangle area (blue = fold, red = valid)')\n"
    "\n"
    "plt.suptitle(f'{CASE_KEY} - grid deformation across SLSQP variants', fontsize=11)\n"
    "plt.show()\n"
)

for c in nb["cells"]:
    if c.get("id") == "variant-viz-code":
        c["source"] = viz_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None
        break

# ------------------------------------------------------------------
# 3. Update se-per-pixel-code: include CD in labels + wider figure.
# ------------------------------------------------------------------
per_pixel_src = (
    "from dvfopt.viz.triangle_debug import find_problematic_pixels\n"
    "from dvfopt.viz import plot_triangle_debug\n"
    "\n"
    "labels = [\n"
    "    ('initial',                  phi_init),\n"
    "    ('(A) CD-SLSQP',             phi_cd),\n"
    "    ('(B) 2tri finite-diff',     phi_fd),\n"
    "    ('(C) 2tri analytical',      phi_an),\n"
    "    ('(D) 2tri + warm-start',    phi_both),\n"
    "]\n"
    "\n"
    "K = 6\n"
    "tri0 = triangle_sign_areas2D(phi_init)\n"
    "tri0_min = tri0.min(axis=0)\n"
    "worst_cells = np.argsort(tri0_min.ravel())[:K]\n"
    "nr, nc = tri0_min.shape\n"
    "worst_pixels = []\n"
    "for flat in worst_cells:\n"
    "    cy, cx = divmod(int(flat), nc)\n"
    "    worst_pixels.append((cx if cx >= 1 else cx + 1, cy))\n"
    "seen = set(); uniq = []\n"
    "for p in worst_pixels:\n"
    "    if p not in seen:\n"
    "        seen.add(p); uniq.append(p)\n"
    "worst_pixels = uniq[:K]\n"
    "print(f'top {len(worst_pixels)} worst pixels (by initial min_TR):', worst_pixels)\n"
    "\n"
    "fig, axes = plt.subplots(len(worst_pixels), len(labels),\n"
    "                          figsize=(3.0 * len(labels), 3.0 * len(worst_pixels)),\n"
    "                          layout='constrained', squeeze=False)\n"
    "for row, (x, y) in enumerate(worst_pixels):\n"
    "    for col, (label, phi) in enumerate(labels):\n"
    "        plot_triangle_debug(phi, x=x, y=y, ax=axes[row, col], show_formula=False)\n"
    "        axes[row, col].set_title(f'{label}  pixel (x={x}, y={y})', fontsize=9)\n"
    "plt.suptitle('Worst initial pixels (rows) evolved across variants (cols)', fontsize=11)\n"
    "plt.show()\n"
)

for c in nb["cells"]:
    if c.get("id") == "se-per-pixel-code":
        c["source"] = per_pixel_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None
        break

# Compile check
for cid in ("se-cd-baseline-code", "variant-viz-code", "se-per-pixel-code"):
    for c in nb["cells"]:
        if c.get("id") == cid:
            try:
                compile("".join(c["source"]), cid, "exec")
                print(f"{cid}: compile OK")
            except SyntaxError as e:
                print(f"{cid}: SYNTAX {e}")
            break

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f"cells now {len(nb['cells'])}")
