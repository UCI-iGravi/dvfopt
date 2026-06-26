"""Simplify triangle-sign-optimization: only 2-tri constraint, vary jacobian method."""

import json

path = "notebooks/two-triangle-check/triangle-sign-optimization.ipynb"
nb = json.load(open(path, encoding="utf-8"))


def find(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_md(cid, text):
    c = find(cid)
    c["source"] = text.splitlines(keepends=True)


def set_code(cid, text):
    c = find(cid)
    c["source"] = text.splitlines(keepends=True)
    c["outputs"] = []
    c["execution_count"] = None


set_md(
    "intro",
    (
        "# 2-Triangle Constraint Optimization - Jacobian Method Comparison\n"
        "\n"
        "Companion to [../shoelace-artifact-example.ipynb](../shoelace-artifact-example.ipynb). Uses the same 7x7 bowtie field (`dx[3,3]=+1.2, dx[3,4]=-1.2`). Every run here uses the **same 2-triangle constraint**; what varies is how the solver's inner QP gets the constraint Jacobian:\n"
        "\n"
        "1. **(A) 2tri + finite-diff Jac** - scipy's default numerical Jacobian via 2-point finite differences on the constraint function.\n"
        "2. **(B) 2tri + analytical Jac** - closed-form `triangle_sign_constraint_jac_2d` plugged into `NonlinearConstraint(..., jac=...)`.\n"
        "3. **(C) 2tri + analytical + warm-start** - same as (B) plus a small Gaussian perturbation + resume, the fix from `triangle-sign-solver-engineering.ipynb` that unsticks SLSQP when it hits status 8.\n"
        "\n"
        "For a cross-constraint comparison (CD vs FD vs 2-tri), see [triangle-sign-constraint-comparison.ipynb](triangle-sign-constraint-comparison.ipynb).\n"
        "\n"
        "Color convention: red = positive Jdet / valid triangle, blue = negative / fold (`RdBu_r`)."
    ),
)

set_code(
    "imports",
    (
        "import os, sys, time\n"
        "sys.path.insert(0, os.path.abspath('../..'))\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.optimize import minimize, NonlinearConstraint\n"
        "from scipy.sparse import csr_matrix\n"
        "\n"
        "from dvfopt import DEFAULT_PARAMS, jacobian_det2D\n"
        "from dvfopt.jacobian import triangle_sign_det2D, triangle_sign_areas2D\n"
        "from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d\n"
        "from dvfopt.jacobian.triangle_sign import _triangle_areas_2d\n"
        "from dvfopt.core.objective import objective_euc\n"
        "from dvfopt.viz import plot_problematic_triangles, plot_triangle_debug\n"
        "from dvfopt.viz.triangle_debug import find_problematic_pixels\n"
        "\n"
        "THRESHOLD = DEFAULT_PARAMS['threshold']  # 0.01\n"
        "print(f'threshold = {THRESHOLD}')\n"
        "\n"
        "\n"
        "def _forward_jdet_2d(dy, dx):\n"
        "    ddx_dx = dx[:-1, 1:]  - dx[:-1, :-1]\n"
        "    ddy_dy = dy[1:,  :-1] - dy[:-1, :-1]\n"
        "    ddx_dy = dx[1:,  :-1] - dx[:-1, :-1]\n"
        "    ddy_dx = dy[:-1, 1:]  - dy[:-1, :-1]\n"
        "    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx"
    ),
)

set_code(
    "case",
    (
        "H = W = 7\n"
        "dy0 = np.zeros((H, W))\n"
        "dx0 = np.zeros((H, W))\n"
        "dx0[3, 3] = +1.2\n"
        "dx0[3, 4] = -1.2\n"
        "phi0 = np.stack([dy0, dx0])\n"
        "\n"
        "def report(phi, label):\n"
        "    jd = np.squeeze(jacobian_det2D(phi))\n"
        "    fd = _forward_jdet_2d(phi[0], phi[1])\n"
        "    tri = triangle_sign_areas2D(phi)\n"
        "    return {\n"
        "        'label': label,\n"
        "        'jd': jd, 'fd': fd, 'tri': tri,\n"
        "        'n_cd': int((jd <= 0).sum()),\n"
        "        'n_fd': int((fd <= 0).sum()),\n"
        "        'n_tr': int((tri <= 0).sum()),\n"
        "        'min_cd': float(jd.min()), 'min_fd': float(fd.min()), 'min_tr': float(tri.min()),\n"
        "        'l2': float(np.linalg.norm(phi - phi0)),\n"
        "    }\n"
        "\n"
        "r0 = report(phi0, 'initial (no correction)')\n"
        "print(f\"{'measure':<22s}  {'n_neg':>6s}  {'min':>8s}\")\n"
        "print('-' * 42)\n"
        "print(f\"{'central-diff Jdet':<22s}  {r0['n_cd']:>6d}  {r0['min_cd']:+8.3f}\")\n"
        "print(f\"{'forward-diff Jdet':<22s}  {r0['n_fd']:>6d}  {r0['min_fd']:+8.3f}\")\n"
        "print(f\"{'2-triangle areas':<22s}  {r0['n_tr']:>6d}  {r0['min_tr']:+8.3f}\")"
    ),
)

# pre-opt-md and pre-opt-code stay as-is

set_md(
    "solver-md",
    (
        "## Solver\n"
        "\n"
        "Full-grid SLSQP: `2*H*W` variables packed as `z = [dx_flat | dy_flat]`, quadratic data term `1/2 ||phi - phi_init||^2`, single nonlinear constraint vector = per-cell 2-triangle signed areas. The only thing that differs between the three runs is the `jac` argument passed to `NonlinearConstraint`."
    ),
)

set_code(
    "solver",
    (
        "def _triangle_flat(dy_, dx_):\n"
        "    T1, T2 = _triangle_areas_2d(dy_, dx_)\n"
        "    return np.concatenate([T1.flatten(), T2.flatten()])\n"
        "\n"
        "\n"
        "def triangle_sign_constraint_jac_2d(dy, dx):\n"
        "    '''Analytical sparse Jacobian ∂g/∂z of the 2-triangle constraint.'''\n"
        "    H_, W_ = dy.shape\n"
        "    nr, nc = H_ - 1, W_ - 1\n"
        "    N = H_ * W_\n"
        "    ref_y, ref_x = np.mgrid[:H_, :W_]\n"
        "    X = ref_x + dx; Y = ref_y + dy\n"
        "    TLx, TLy = X[:-1, :-1], Y[:-1, :-1]\n"
        "    TRx, TRy = X[:-1, 1:],  Y[:-1, 1:]\n"
        "    BLx, BLy = X[1:,  :-1], Y[1:,  :-1]\n"
        "    BRx, BRy = X[1:,  1:],  Y[1:,  1:]\n"
        "    cy_grid, cx_grid = np.mgrid[:nr, :nc]\n"
        "    cy = cy_grid.ravel(); cx = cx_grid.ravel()\n"
        "    def dx_idx(py, px): return py * W_ + px\n"
        "    def dy_idx(py, px): return N + py * W_ + px\n"
        "    n_cells = nr * nc\n"
        "    rows_T1 = np.arange(n_cells)\n"
        "    T1_rows = np.tile(rows_T1, 6)\n"
        "    T1_cols = np.concatenate([\n"
        "        dx_idx(cy, cx + 1), dy_idx(cy, cx + 1),\n"
        "        dx_idx(cy + 1, cx), dy_idx(cy + 1, cx),\n"
        "        dx_idx(cy + 1, cx + 1), dy_idx(cy + 1, cx + 1),\n"
        "    ])\n"
        "    T1_vals = np.concatenate([\n"
        "        0.5 * (BRy - BLy).ravel(), 0.5 * (BLx - BRx).ravel(),\n"
        "        0.5 * (TRy - BRy).ravel(), 0.5 * (BRx - TRx).ravel(),\n"
        "        0.5 * (BLy - TRy).ravel(), 0.5 * (TRx - BLx).ravel(),\n"
        "    ])\n"
        "    rows_T2 = np.arange(n_cells) + n_cells\n"
        "    T2_rows = np.tile(rows_T2, 6)\n"
        "    T2_cols = np.concatenate([\n"
        "        dx_idx(cy, cx),          dy_idx(cy, cx),\n"
        "        dx_idx(cy + 1, cx),      dy_idx(cy + 1, cx),\n"
        "        dx_idx(cy, cx + 1),      dy_idx(cy, cx + 1),\n"
        "    ])\n"
        "    T2_vals = np.concatenate([\n"
        "        0.5 * (TRy - BLy).ravel(), 0.5 * (BLx - TRx).ravel(),\n"
        "        0.5 * (TLy - TRy).ravel(), 0.5 * (TRx - TLx).ravel(),\n"
        "        0.5 * (BLy - TLy).ravel(), 0.5 * (TLx - BLx).ravel(),\n"
        "    ])\n"
        "    rows = np.concatenate([T1_rows, T2_rows])\n"
        "    cols = np.concatenate([T1_cols, T2_cols])\n"
        "    vals = np.concatenate([T1_vals, T2_vals])\n"
        "    return csr_matrix((vals, (rows, cols)), shape=(2 * n_cells, 2 * N))\n"
        "\n"
        "\n"
        "def run_2tri_slsqp(phi_init, use_analytical_jac=False, warm_start=False,\n"
        "                    threshold=THRESHOLD, max_iter=500):\n"
        "    _, H_, W_ = phi_init.shape\n"
        "    pixels = H_ * W_\n"
        "    dy, dx = phi_init[0], phi_init[1]\n"
        "    z0 = np.concatenate([dx.flatten(), dy.flatten()])\n"
        "    z0_init = z0.copy()\n"
        "\n"
        "    def unpack(z):\n"
        "        dx_ = z[:pixels].reshape(H_, W_)\n"
        "        dy_ = z[pixels:].reshape(H_, W_)\n"
        "        return dy_, dx_\n"
        "\n"
        "    def constr(z):\n"
        "        return _triangle_flat(*unpack(z))\n"
        "\n"
        "    def constr_jac(z):\n"
        "        dy_, dx_ = unpack(z)\n"
        "        return triangle_sign_constraint_jac_2d(dy_, dx_)\n"
        "\n"
        "    nl_kwargs = dict(lb=threshold, ub=np.inf)\n"
        "    if use_analytical_jac:\n"
        "        nl_kwargs['jac'] = constr_jac\n"
        "\n"
        "    t0 = time.time()\n"
        "    res = minimize(\n"
        "        lambda z: objective_euc(z, z0_init),\n"
        "        z0, jac=True, method='SLSQP',\n"
        "        constraints=[NonlinearConstraint(constr, **nl_kwargs)],\n"
        "        options={'maxiter': max_iter, 'disp': False},\n"
        "    )\n"
        "    total_nit = res.nit\n"
        "    total_time = time.time() - t0\n"
        "\n"
        "    # Optional warm-start retry if SLSQP stalled with status 8 (positive\n"
        "    # directional derivative for linesearch).\n"
        "    if warm_start and not res.success and res.status == 8:\n"
        "        rng = np.random.default_rng(123)\n"
        "        z_warm = res.x + rng.normal(scale=0.01, size=res.x.shape)\n"
        "        t1 = time.time()\n"
        "        res = minimize(\n"
        "            lambda z: objective_euc(z, z0_init),\n"
        "            z_warm, jac=True, method='SLSQP',\n"
        "            constraints=[NonlinearConstraint(constr, **nl_kwargs)],\n"
        "            options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False},\n"
        "        )\n"
        "        total_nit += res.nit\n"
        "        total_time += time.time() - t1\n"
        "\n"
        "    dy_out, dx_out = unpack(res.x)\n"
        "    phi_out = np.stack([dy_out, dx_out])\n"
        "    r = report(phi_out, 'SLSQP[2-tri]')\n"
        "    r.update(t=total_time, nit=total_nit, success=bool(res.success),\n"
        "             status=int(res.status), message=str(res.message), phi=phi_out)\n"
        "    return r"
    ),
)

set_code(
    "run-all",
    (
        "runs = {\n"
        "    '(A) finite-diff Jac':         run_2tri_slsqp(phi0, use_analytical_jac=False, warm_start=False),\n"
        "    '(B) analytical Jac':          run_2tri_slsqp(phi0, use_analytical_jac=True,  warm_start=False),\n"
        "    '(C) analytical + warm-start': run_2tri_slsqp(phi0, use_analytical_jac=True,  warm_start=True),\n"
        "}\n"
        "\n"
        "hdr = f\"{'variant':<32s}  {'nit':>4s}  {'time':>6s}  {'neg_TR':>6s}  {'min_TR':>8s}  {'L2':>6s}  success\"\n"
        "print(hdr)\n"
        "print('-' * len(hdr))\n"
        "print(f\"{'initial':<32s}  {'-':>4s}  {'-':>6s}  {r0['n_tr']:>6d}  {r0['min_tr']:+8.3f}  {r0['l2']:>6.3f}\")\n"
        "for key, r in runs.items():\n"
        "    print(\n"
        "        f\"{key:<32s}  {r['nit']:>4d}  {r['t']:>6.3f}  \"\n"
        "        f\"{r['n_tr']:>6d}  {r['min_tr']:+8.3f}  {r['l2']:>6.3f}  {r['success']}\"\n"
        "    )"
    ),
)

set_md(
    "plots-md",
    (
        "## Visual comparison\n"
        "\n"
        "Four rows (initial + three variants), two columns: warped quad grid with folded cells outlined, and the per-cell `min(T1, T2)` signed-area heatmap. All three variants use the 2-triangle constraint, so the only differences in the corrected fields come from numerical noise in the Jacobian + whether the warm-start fired."
    ),
)

set_code(
    "plots",
    (
        "all_rows = [('initial', r0)] + list(runs.items())\n"
        "\n"
        "vmax_tri = max(abs(r['tri']).max() for _, r in all_rows)\n"
        "NL = chr(10)\n"
        "\n"
        "def plot_warped_grid(ax, phi, title, highlight_folds=True):\n"
        "    dy = phi[0]; dx = phi[1]\n"
        "    Hh, Ww = dy.shape\n"
        "    yy, xx = np.mgrid[:Hh, :Ww]\n"
        "    gx = xx + dx; gy = yy + dy\n"
        "    for i in range(Hh):\n"
        "        ax.plot(xx[i], yy[i], color='#f0f0f0', lw=0.5)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(xx[:, j], yy[:, j], color='#f0f0f0', lw=0.5)\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=1.0)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=1.0)\n"
        "    if highlight_folds:\n"
        "        tri = triangle_sign_areas2D(phi)\n"
        "        bad = np.argwhere(tri.min(axis=0) <= 0)\n"
        "        for (cy, cx) in bad:\n"
        "            px = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "            py = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "            ax.plot(px, py, color='#1565c0', lw=1.8)\n"
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(title, fontsize=9)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "\n"
        "fig, axes = plt.subplots(len(all_rows), 2, figsize=(10, 3.3 * len(all_rows)), layout='constrained')\n"
        "for i, (label, r) in enumerate(all_rows):\n"
        "    phi = phi0 if label == 'initial' else r['phi']\n"
        "    if label == 'initial':\n"
        "        line1 = label\n"
        "    else:\n"
        "        tag = 'OK' if r['success'] else 'FAIL'\n"
        "        line1 = f\"{label}  [{tag}]\"\n"
        "    line2 = f\"TR={r['n_tr']}  min_TR={r['min_tr']:+.3f}  L2={r['l2']:.3f}\"\n"
        "    plot_warped_grid(axes[i, 0], phi, line1 + NL + line2)\n"
        "\n"
        "    tri_min = r['tri'].min(axis=0)\n"
        "    im = axes[i, 1].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri, aspect='auto')\n"
        "    axes[i, 1].set_title(f'min(T1, T2)   {label}', fontsize=9)\n"
        "    axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])\n"
        "\n"
        "plt.suptitle('7x7 bowtie - 2-triangle SLSQP across three Jacobian variants', fontsize=11)\n"
        "plt.show()"
    ),
)

# Delete grid-md, grid cells (redundant with warped grid in plots above),
# and tri-overlay-md, tri-overlay-code (replaced by combined plots cell).
to_delete = {"grid-md", "grid", "tri-overlay-md", "tri-overlay-code"}
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in to_delete]

# Update debug cell — now iterate runs dict keys dynamically.
set_md(
    "debug-md",
    (
        "## Per-pixel debug inspection\n"
        "\n"
        "For every pixel that's problematic in *any* variant, show its T1/T2 across (initial, (A), (B), (C)). On the bowtie the two problematic pixels are `(3, 3)` and `(4, 2)` in the initial state; all three 2-tri variants should clear them."
    ),
)

set_code(
    "debug",
    (
        "labels = [('initial', phi0)] + [(k, v['phi']) for k, v in runs.items()]\n"
        "\n"
        "all_bad = set()\n"
        "for _, phi in labels:\n"
        "    all_bad.update(find_problematic_pixels(phi))\n"
        "all_bad = sorted(all_bad)\n"
        "print(f'{len(all_bad)} pixel(s) problematic in at least one variant: {all_bad}')\n"
        "\n"
        "if all_bad:\n"
        "    fig, axes = plt.subplots(len(all_bad), len(labels),\n"
        "                              figsize=(3.2 * len(labels), 3.0 * len(all_bad)),\n"
        "                              layout='constrained', squeeze=False)\n"
        "    for row, (x, y) in enumerate(all_bad):\n"
        "        for col, (label, phi) in enumerate(labels):\n"
        "            plot_triangle_debug(phi, x=x, y=y, ax=axes[row, col], show_formula=False)\n"
        "            axes[row, col].set_title(f'{label}  pixel (x={x}, y={y})', fontsize=9)\n"
        "    plt.suptitle('Per-pixel 2-triangle debug  (rows: pixel, cols: variant)', fontsize=11)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No problematic pixels across any variant.')"
    ),
)

set_md(
    "summary",
    (
        "## Summary\n"
        "\n"
        "- All three variants hold the same 2-triangle constraint; only the Jacobian source (and the warm-start) changes.\n"
        "- On the 7x7 bowtie, all three converge to `neg_TR=0` in well under a second. Differences are purely in iteration count / wall time.\n"
        "- Warm-start doesn't fire here (SLSQP succeeds on the first attempt), so (B) and (C) produce identical outputs; the branch is retained in case future runs include harder cases where SLSQP stalls.\n"
        "- For a direct comparison against central-diff and forward-diff *constraint* formulations (not just different Jacobians of the 2-triangle constraint), open [triangle-sign-constraint-comparison.ipynb](triangle-sign-constraint-comparison.ipynb)."
    ),
)

# Verify all remaining code cells compile.
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            compile(src, c.get("id", "?"), "exec")
        except SyntaxError as e:
            print(f'SYNTAX in {c.get("id")}: {e}')
            break
    c.setdefault("metadata", {})

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f"simplified: {len(nb['cells'])} cells now")
