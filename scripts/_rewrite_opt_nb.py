"""Rewrite triangle-sign-optimization.ipynb: swap shoelace for forward-diff."""

import json

path = "notebooks/two-triangle-check/triangle-sign-optimization.ipynb"
nb = json.load(open(path, encoding="utf-8"))


def find(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs")
    if cell["cell_type"] == "code":
        cell["execution_count"] = None


# --- intro (markdown) ---
set_src(
    find("intro"),
    (
        "# 2-Triangle Constraint Optimization vs. Central-Diff & Forward-Diff\n"
        "\n"
        "Companion to [../shoelace-artifact-example.ipynb](../shoelace-artifact-example.ipynb). Uses the same 7×7 bowtie field (`dx[3,3]=+1.2, dx[3,4]=−1.2`) but here we run full-grid SLSQP with three different constraint formulations and compare the corrected fields:\n"
        "\n"
        "1. **Central-diff Jdet** (`jacobian_det2D` ≥ threshold) — per-pixel Jacobian via `np.gradient`'s symmetric 2Δ stencil. This field already passes the check (fold is invisible to central differences), so the solver reports *no work to do* and leaves it folded.\n"
        "2. **Forward-diff Jdet** (per-cell Jacobian via one-sided forward differences ≥ threshold) — catches the collapsed cell at the bowtie because `∂dx/∂x` flips sign locally.\n"
        "3. **Two-triangle signed areas** (`triangle_sign_areas2D` ≥ threshold) — the TR-BL-split per-pixel check, twice as many constraints per cell as FD.\n"
        "\n"
        "For each corrected field we report `(neg_CD, neg_FD, neg_TR)`, min value under each measure, L2 distortion `‖φ − φ₀‖`, and wall time.\n"
        "\n"
        "Color convention matches the rest of the workspace: red = positive Jdet, blue = negative (`RdBu_r`)."
    ),
)

# --- imports (code) ---
set_src(
    find("imports"),
    (
        "import os, sys, time\n"
        "sys.path.insert(0, os.path.abspath('../..'))\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.optimize import minimize, NonlinearConstraint\n"
        "\n"
        "from dvfopt import DEFAULT_PARAMS, jacobian_det2D\n"
        "from dvfopt.jacobian import triangle_sign_det2D, triangle_sign_areas2D\n"
        "from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d\n"
        "from dvfopt.jacobian.triangle_sign import _triangle_areas_2d\n"
        "from dvfopt.objectives import L2Objective\n"
        "from dvfopt.viz import plot_problematic_triangles\n"
        "\n"
        "THRESHOLD = DEFAULT_PARAMS['threshold']  # 0.01\n"
        "print(f'threshold = {THRESHOLD}')\n"
        "\n"
        "\n"
        "def _forward_jdet_2d(dy, dx):\n"
        "    \"\"\"Per-cell Jacobian determinant via forward finite differences.\"\"\"\n"
        "    ddx_dx = dx[:-1, 1:]  - dx[:-1, :-1]\n"
        "    ddy_dy = dy[1:,  :-1] - dy[:-1, :-1]\n"
        "    ddx_dy = dx[1:,  :-1] - dx[:-1, :-1]\n"
        "    ddy_dx = dy[:-1, 1:]  - dy[:-1, :-1]\n"
        "    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx"
    ),
)

# --- case-md (markdown) — no change needed (references the bowtie) ---

# --- case (code): replace report() to use forward-diff instead of shoelace ---
set_src(
    find("case"),
    (
        "H = W = 7\n"
        "dy0 = np.zeros((H, W))\n"
        "dx0 = np.zeros((H, W))\n"
        "dx0[3, 3] = +1.2\n"
        "dx0[3, 4] = -1.2\n"
        "phi0 = np.stack([dy0, dx0])  # (2, H, W) channels [dy, dx]\n"
        "\n"
        "def report(phi, label):\n"
        "    jd = np.squeeze(jacobian_det2D(phi))\n"
        "    fd = _forward_jdet_2d(phi[0], phi[1])\n"
        "    tri = triangle_sign_areas2D(phi)\n"
        "    n_cd = int((jd <= 0).sum())\n"
        "    n_fd = int((fd <= 0).sum())\n"
        "    n_tr = int((tri <= 0).sum())\n"
        "    return {\n"
        "        'label': label,\n"
        "        'jd': jd, 'fd': fd, 'tri': tri,\n"
        "        'n_cd': n_cd, 'n_fd': n_fd, 'n_tr': n_tr,\n"
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

# --- solver (code): replace _shoelace_flat with _forward_flat ---
set_src(
    find("solver"),
    (
        "def run_slsqp(phi_init, constraint_fn, label, threshold=THRESHOLD, max_iter=200):\n"
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
        "    t0 = time.time()\n"
        "    res = minimize(\n"
        "        lambda z: L2Objective()(z - z0_init),\n"
        "        z0,\n"
        "        jac=True,\n"
        "        method='SLSQP',\n"
        "        constraints=[NonlinearConstraint(\n"
        "            lambda z: constraint_fn(*unpack(z)),\n"
        "            threshold, np.inf,\n"
        "        )],\n"
        "        options={'maxiter': max_iter, 'disp': False},\n"
        "    )\n"
        "    t = time.time() - t0\n"
        "\n"
        "    dy_out, dx_out = unpack(res.x)\n"
        "    phi_out = np.stack([dy_out, dx_out])\n"
        "    r = report(phi_out, label)\n"
        "    r['t'] = t\n"
        "    r['nit'] = res.nit\n"
        "    r['success'] = bool(res.success)\n"
        "    r['phi'] = phi_out\n"
        "    return r\n"
        "\n"
        "\n"
        "def _central_diff_flat(dy_, dx_):\n"
        "    return _numpy_jdet_2d(dy_, dx_).flatten()\n"
        "\n"
        "def _forward_flat(dy_, dx_):\n"
        "    return _forward_jdet_2d(dy_, dx_).flatten()\n"
        "\n"
        "def _triangle_flat(dy_, dx_):\n"
        "    T1, T2 = _triangle_areas_2d(dy_, dx_)\n"
        "    return np.concatenate([T1.flatten(), T2.flatten()])"
    ),
)

# --- run-all (code): swap 'shoelace' → 'forward-diff' ---
set_src(
    find("run-all"),
    (
        "runs = {\n"
        "    'central-diff': run_slsqp(phi0, _central_diff_flat, 'SLSQP[central-diff]'),\n"
        "    'forward-diff': run_slsqp(phi0, _forward_flat,      'SLSQP[forward-diff]'),\n"
        "    '2-triangle':   run_slsqp(phi0, _triangle_flat,     'SLSQP[2-triangle]'),\n"
        "}\n"
        "\n"
        "print(f\"{'run':<14s}  {'nit':>4s}  {'time':>6s}  {'neg_CD':>6s}  {'neg_FD':>6s}  {'neg_TR':>6s}  {'min_FD':>8s}  {'min_TR':>8s}  {'L2':>7s}  success\")\n"
        "print('-' * 92)\n"
        "print(f\"{'initial':<14s}  {'-':>4s}  {'-':>6s}  {r0['n_cd']:>6d}  {r0['n_fd']:>6d}  {r0['n_tr']:>6d}  {r0['min_fd']:+8.3f}  {r0['min_tr']:+8.3f}  {r0['l2']:>7.3f}\")\n"
        "for key, r in runs.items():\n"
        "    print(\n"
        "        f\"{key:<14s}  {r['nit']:>4d}  {r['t']:>6.2f}  \"\n"
        "        f\"{r['n_cd']:>6d}  {r['n_fd']:>6d}  {r['n_tr']:>6d}  \"\n"
        "        f\"{r['min_fd']:+8.3f}  {r['min_tr']:+8.3f}  {r['l2']:>7.3f}  {r['success']}\"\n"
        "    )"
    ),
)

# --- plots-md ---
set_src(
    find("plots-md"),
    (
        "## Visual comparison\n"
        "\n"
        "Four rows: initial, SLSQP with each constraint. Three columns: central-diff Jdet (`H×W`), forward-diff Jdet (`(H−1)×(W−1)`), min of the two triangle signed areas per cell (`(H−1)×(W−1)`). All three columns use the same `RdBu_r` colormap centered on 0 so blue = fold, red = valid."
    ),
)

# --- plots (code): replace shoelace column with forward-diff ---
set_src(
    find("plots"),
    (
        "all_rows = [('initial', r0)] + [(k, runs[k]) for k in ('central-diff', 'forward-diff', '2-triangle')]\n"
        "\n"
        "# Shared limits so the colormaps are directly comparable.\n"
        "vmax = max(abs(r['jd']).max() for _, r in all_rows)\n"
        "vmax_fd = max(abs(r['fd']).max() for _, r in all_rows)\n"
        "vmax_tri = max(abs(r['tri']).max() for _, r in all_rows)\n"
        "\n"
        "fig, axes = plt.subplots(len(all_rows), 3, figsize=(11, 2.8 * len(all_rows)))\n"
        "for i, (label, r) in enumerate(all_rows):\n"
        "    tri_min = r['tri'].min(axis=0)\n"
        "\n"
        "    axes[i, 0].imshow(r['jd'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)\n"
        "    axes[i, 0].set_title(f\"{label}\\ncentral-diff  neg={r['n_cd']}  min={r['min_cd']:+.3f}\")\n"
        "\n"
        "    axes[i, 1].imshow(r['fd'], cmap='RdBu_r', vmin=-vmax_fd, vmax=vmax_fd)\n"
        "    axes[i, 1].set_title(f\"forward-diff  neg={r['n_fd']}  min={r['min_fd']:+.3f}\")\n"
        "\n"
        "    axes[i, 2].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri)\n"
        "    axes[i, 2].set_title(f\"min(T1, T2)  neg={r['n_tr']}  min={r['min_tr']:+.3f}\")\n"
        "\n"
        "    for ax in axes[i]:\n"
        "        ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
)

# --- grid-md unchanged ---

# --- grid (code): update titles to reference central-diff, forward-diff, 2-triangle ---
set_src(
    find("grid"),
    (
        "def plot_warped(ax, phi, title):\n"
        "    dy = phi[0]; dx = phi[1]\n"
        "    Hh, Ww = dy.shape\n"
        "    yy, xx = np.mgrid[:Hh, :Ww]\n"
        "    gx = xx + dx; gy = yy + dy\n"
        "    # Reference grid\n"
        "    for i in range(Hh):\n"
        "        ax.plot(xx[i], yy[i], color='#dddddd', lw=0.4)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(xx[:, j], yy[:, j], color='#dddddd', lw=0.4)\n"
        "    # Warped grid\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=1.0)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=1.0)\n"
        "    ax.set_aspect('equal')\n"
        "    ax.invert_yaxis()\n"
        "    ax.set_title(title)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))\n"
        "plot_warped(axes[0], phi0, 'initial (bowtie)')\n"
        "for k, key in enumerate(['central-diff', 'forward-diff', '2-triangle']):\n"
        "    plot_warped(axes[k + 1], runs[key]['phi'],\n"
        "                f\"SLSQP [{key}]\\nL2={runs[key]['l2']:.3f}\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
)

# --- debug-md and debug (code): iterate over central-diff/forward-diff/2-triangle ---
set_src(
    find("debug"),
    (
        "print('--- initial ---')\n"
        "_ = plot_problematic_triangles(phi0, title='initial')\n"
        "\n"
        "for key in ('central-diff', 'forward-diff', '2-triangle'):\n"
        "    print(f'--- SLSQP [{key}] ---')\n"
        "    _ = plot_problematic_triangles(runs[key]['phi'], title=f'SLSQP [{key}]')"
    ),
)

# --- summary (md) ---
set_src(
    find("summary"),
    (
        "## Summary\n"
        "\n"
        "- **SLSQP with central-diff constraint** leaves the fold in place — the field already satisfies its constraint at the initial point (all pixel Jdets ≥ threshold), so the solver makes near-zero updates and L2 distortion ≈ 0. Forward-diff and 2-triangle measures still report the fold.\n"
        "- **SLSQP with forward-diff constraint** unfolds the bowtie under its own measure (`neg_FD → 0`). Because forward-diff is one per cell while the 2-triangle check is two per cell under TR-BL split, the 2-triangle measure can still flag a residual flipped triangle even after FD clears.\n"
        "- **SLSQP with 2-triangle constraint** also unfolds, with a stricter constraint structure: 2·(H−1)·(W−1) = 72 constraints vs. forward-diff's (H−1)·(W−1) = 36. The TR-BL triangulation is asymmetric to TL-BR; under this check, `n_neg` goes to 0 under all three measures. Typical result: slightly larger L2 distortion than forward-diff for the same case, because the per-triangle threshold is more restrictive than the per-cell forward-diff Jacobian threshold.\n"
        "- Debug figures for the 2-triangle-corrected run should be empty (no problematic triangles); the initial run's debug figures visualize the folded T2 triangles at pixels `(3, 3)` and `(4, 2)`."
    ),
)

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print("notebook rewritten")
