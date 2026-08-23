"""Build 08_global-invertibility-gap.ipynb - deep dive on 03c_20x20_opposite."""

import json

path = "notebooks/two-triangle-check/08_global-invertibility-gap.ipynb"


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


cells = []

cells.append(
    md(
        "# Local Invertibility is Not Global Invertibility\n"
        "\n"
        "Deep dive on `03c_20x20_opposite` - the canonical case where the 2-triangle constraint SLSQP succeeds with `neg_TR = 0` but the warped field is still **not globally invertible**. Previously surfaced as one row in [03_demos.ipynb](03_demos.ipynb) Part 3; this notebook gives it the same treatment 05 and 07 give their hard cases.\n"
        "\n"
        "The story:\n"
        "\n"
        "1. **Run best-2tri SLSQP** (analytical Jacobian + warm-start) on `03c_20x20_opposite`. Converges with `success=True`, `neg_TR = 0`. Every cell is individually non-folded.\n"
        "2. **Global quad self-intersection check** reports `QI > 0`. Two non-adjacent quads' edges still cross somewhere in the warped grid.\n"
        "3. **Visualize where and why.** The 'opposite-motion' fold pattern pushes two regions of the grid against each other. Each cell at the collision front stays locally convex, but the *regions* overlap.\n"
        "4. **Damping experiment.** Linearly interpolate between the identity and `phi_out` and scan `alpha`: how much of the correction can we apply before the global check starts failing?\n"
        "5. **Next steps** discussion.\n"
        "\n"
        "This is the counterexample that motivates an outer-loop global repulsion (option 1 from [03_demos.ipynb](03_demos.ipynb)'s 'next steps for global invertibility').",
        "intro",
    )
)

cells.append(
    code(
        "import os, sys, time\n"
        "sys.path.insert(0, os.path.abspath('../..'))\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.optimize import minimize, NonlinearConstraint\n"
        "from scipy.sparse import csr_matrix\n"
        "\n"
        "from dvfopt import DEFAULT_PARAMS, jacobian_det2D\n"
        "from dvfopt.jacobian import triangle_sign_areas2D\n"
        "from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d\n"
        "from dvfopt.jacobian.triangle_sign import _triangle_areas_2d\n"
        "from dvfopt.jacobian.intersection import has_quad_self_intersections, _quads_intersect\n"
        "from dvfopt.objectives import L2Objective\n"
        "\n"
        "from test_cases import make_deformation\n"
        "\n"
        "THRESHOLD = DEFAULT_PARAMS['threshold']\n"
        "print(f'threshold = {THRESHOLD}')",
        "imports",
    )
)

cells.append(
    code(
        "def _forward_jdet_2d(dy, dx):\n"
        "    ddx_dx = dx[:-1, 1:]  - dx[:-1, :-1]\n"
        "    ddy_dy = dy[1:,  :-1] - dy[:-1, :-1]\n"
        "    ddx_dy = dx[1:,  :-1] - dx[:-1, :-1]\n"
        "    ddy_dx = dy[:-1, 1:]  - dy[:-1, :-1]\n"
        "    return (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx\n"
        "\n"
        "def measure(phi):\n"
        "    jd = np.squeeze(jacobian_det2D(phi))\n"
        "    fd = _forward_jdet_2d(phi[0], phi[1])\n"
        "    tri = triangle_sign_areas2D(phi)\n"
        "    return dict(jd=jd, fd=fd, tri=tri,\n"
        "                n_cd=int((jd <= 0).sum()), n_fd=int((fd <= 0).sum()),\n"
        "                n_tr=int((tri <= 0).sum()),\n"
        "                min_cd=float(jd.min()), min_fd=float(fd.min()), min_tr=float(tri.min()))\n"
        "\n"
        "def list_intersecting_quads(phi):\n"
        "    dy, dx = phi[0], phi[1]\n"
        "    H, W = dy.shape\n"
        "    nr, nc = H - 1, W - 1\n"
        "    if nr <= 0 or nc <= 0:\n"
        "        return []\n"
        "    rows = np.arange(H, dtype=float)[:, None]\n"
        "    cols = np.arange(W, dtype=float)[None, :]\n"
        "    Y = rows + dy; X = cols + dx\n"
        "    y_tl, x_tl = Y[:-1, :-1], X[:-1, :-1]\n"
        "    y_tr, x_tr = Y[:-1, 1:],  X[:-1, 1:]\n"
        "    y_br, x_br = Y[1:,  1:],  X[1:,  1:]\n"
        "    y_bl, x_bl = Y[1:,  :-1], X[1:,  :-1]\n"
        "    ymin = np.minimum(np.minimum(y_tl, y_tr), np.minimum(y_bl, y_br)).ravel()\n"
        "    ymax = np.maximum(np.maximum(y_tl, y_tr), np.maximum(y_bl, y_br)).ravel()\n"
        "    xmin = np.minimum(np.minimum(x_tl, x_tr), np.minimum(x_bl, x_br)).ravel()\n"
        "    xmax = np.maximum(np.maximum(x_tl, x_tr), np.maximum(x_bl, x_br)).ravel()\n"
        "    corners = np.stack([\n"
        "        np.stack([y_tl.ravel(), x_tl.ravel()], axis=1),\n"
        "        np.stack([y_tr.ravel(), x_tr.ravel()], axis=1),\n"
        "        np.stack([y_br.ravel(), x_br.ravel()], axis=1),\n"
        "        np.stack([y_bl.ravel(), x_bl.ravel()], axis=1),\n"
        "    ], axis=1)\n"
        "    n_quads = nr * nc\n"
        "    pairs = []\n"
        "    for i in range(n_quads):\n"
        "        ri, ci = divmod(i, nc)\n"
        "        for j in range(i + 1, n_quads):\n"
        "            rj, cj = divmod(j, nc)\n"
        "            if abs(ri - rj) <= 1 and abs(ci - cj) <= 1: continue\n"
        "            if (ymin[i] > ymax[j] or ymax[i] < ymin[j]\n"
        "                    or xmin[i] > xmax[j] or xmax[i] < xmin[j]): continue\n"
        "            if _quads_intersect(corners[i], corners[j]):\n"
        "                pairs.append(((ri, ci), (rj, cj)))\n"
        "    return pairs",
        "measure-helpers",
    )
)

cells.append(
    code(
        "def triangle_sign_constraint_jac_2d(dy, dx):\n"
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
        "    T1_rows = np.tile(np.arange(n_cells), 6)\n"
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
        "    T2_rows = np.tile(np.arange(n_cells) + n_cells, 6)\n"
        "    T2_cols = np.concatenate([\n"
        "        dx_idx(cy, cx),     dy_idx(cy, cx),\n"
        "        dx_idx(cy + 1, cx), dy_idx(cy + 1, cx),\n"
        "        dx_idx(cy, cx + 1), dy_idx(cy, cx + 1),\n"
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
        "def run_2tri_best(phi_init, threshold=THRESHOLD, max_iter=500):\n"
        "    _, H_, W_ = phi_init.shape\n"
        "    pixels = H_ * W_\n"
        "    def unpack(z):\n"
        "        return z[pixels:].reshape(H_, W_), z[:pixels].reshape(H_, W_)\n"
        "    def tri_flat(dy, dx):\n"
        "        T1, T2 = _triangle_areas_2d(dy, dx)\n"
        "        return np.concatenate([T1.flatten(), T2.flatten()])\n"
        "    def tri_jac(z):\n"
        "        dy, dx = unpack(z)\n"
        "        return triangle_sign_constraint_jac_2d(dy, dx)\n"
        "    z0 = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])\n"
        "    z0_init = z0.copy()\n"
        "    t0 = time.time()\n"
        "    res = minimize(lambda z: L2Objective()(z - z0_init), z0,\n"
        "                   jac=True, method='SLSQP',\n"
        "                   constraints=[NonlinearConstraint(\n"
        "                       lambda z: tri_flat(*unpack(z)), threshold, np.inf, jac=tri_jac)],\n"
        "                   options={'maxiter': max_iter, 'disp': False})\n"
        "    total_nit = res.nit; total_time = time.time() - t0\n"
        "    if not res.success and res.status == 8:\n"
        "        rng = np.random.default_rng(123)\n"
        "        z_warm = res.x + rng.normal(scale=0.01, size=res.x.shape)\n"
        "        t1 = time.time()\n"
        "        res = minimize(lambda z: L2Objective()(z - z0_init), z_warm,\n"
        "                       jac=True, method='SLSQP',\n"
        "                       constraints=[NonlinearConstraint(\n"
        "                           lambda z: tri_flat(*unpack(z)), threshold, np.inf, jac=tri_jac)],\n"
        "                       options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})\n"
        "        total_nit += res.nit; total_time += time.time() - t1\n"
        "    dy_o, dx_o = unpack(res.x)\n"
        "    phi_out = np.stack([dy_o, dx_o])\n"
        "    m = measure(phi_out)\n"
        "    m.update(phi=phi_out, nit=total_nit, time=total_time,\n"
        "             success=bool(res.success), status=int(res.status),\n"
        "             message=str(res.message),\n"
        "             l2=float(np.linalg.norm(phi_out - phi_init)))\n"
        "    return m",
        "anal-jac-runner",
    )
)

cells.append(
    md(
        "## Setup - load `03c_20x20_opposite`\n"
        "\n"
        "Opposite-motion fold on a 20x20 grid. Two regions push against each other producing a characteristic collision front.",
        "setup-md",
    )
)

cells.append(
    code(
        "CASE_KEY = '03c_20x20_opposite'\n"
        "deformation, *_ = make_deformation(CASE_KEY)\n"
        "phi_init = np.stack([deformation[1, 0], deformation[2, 0]])\n"
        "m0 = measure(phi_init)\n"
        "qi0 = list_intersecting_quads(phi_init)\n"
        "H_, W_ = phi_init[0].shape\n"
        "print(f'{CASE_KEY}  shape=({H_}, {W_})  pixels={H_*W_}  vars={2*H_*W_}')\n"
        "print(f'init  neg_CD={m0[\"n_cd\"]:>3d}  neg_FD={m0[\"n_fd\"]:>3d}  neg_TR={m0[\"n_tr\"]:>3d}  QI={len(qi0):>3d}')\n"
        "print(f'init  min_CD={m0[\"min_cd\"]:+.3f}  min_FD={m0[\"min_fd\"]:+.3f}  min_TR={m0[\"min_tr\"]:+.3f}')",
        "setup-code",
    )
)

cells.append(
    md(
        "## Pre-optimization fold structure",
        "preopt-md",
    )
)

cells.append(
    code(
        "def plot_warped_grid(ax, phi, title, highlight_mask=None, fold_color='#1565c0'):\n"
        "    dy = phi[0]; dx = phi[1]\n"
        "    Hh, Ww = dy.shape\n"
        "    yy, xx = np.mgrid[:Hh, :Ww]\n"
        "    gx = xx + dx; gy = yy + dy\n"
        "    for i in range(Hh):\n"
        "        ax.plot(xx[i], yy[i], color='#f4f4f4', lw=0.3)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(xx[:, j], yy[:, j], color='#f4f4f4', lw=0.3)\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.8)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.8)\n"
        "    if highlight_mask is not None:\n"
        "        for (cy, cx) in np.argwhere(highlight_mask):\n"
        "            px = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "            py = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "            ax.plot(px, py, color=fold_color, lw=1.6)\n"
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "tri_init = triangle_sign_areas2D(phi_init)\n"
        "local_fold_mask_init = tri_init.min(axis=0) <= 0\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')\n"
        "plot_warped_grid(axes[0], phi_init,\n"
        "                  f'warped grid   {int(local_fold_mask_init.sum())} folded cell(s) outlined',\n"
        "                  highlight_mask=local_fold_mask_init)\n"
        "vmax_cd = float(max(abs(m0['jd']).max(), 1.0))\n"
        "im1 = axes[1].imshow(m0['jd'], cmap='RdBu_r', vmin=-vmax_cd, vmax=vmax_cd, aspect='auto')\n"
        "axes[1].set_title(f\"central-diff Jdet  neg={m0['n_cd']}  min={m0['min_cd']:+.3f}\")\n"
        "axes[1].set_xticks([]); axes[1].set_yticks([])\n"
        "fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)\n"
        "tri_min0 = tri_init.min(axis=0)\n"
        "vmax_tr = float(max(abs(tri_min0).max(), 1.0))\n"
        "im2 = axes[2].imshow(tri_min0, cmap='RdBu_r', vmin=-vmax_tr, vmax=vmax_tr, aspect='auto')\n"
        "axes[2].set_title(f\"min(T1, T2)  neg={m0['n_tr']}  min={m0['min_tr']:+.3f}\")\n"
        "axes[2].set_xticks([]); axes[2].set_yticks([])\n"
        "fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)\n"
        "plt.suptitle(f'{CASE_KEY} - pre-optimization  (init QI = {len(qi0)})', fontsize=11)\n"
        "plt.show()",
        "preopt-code",
    )
)

cells.append(
    md(
        "## Run best 2-triangle SLSQP\n"
        "\n"
        "Analytical Jacobian + perturbation warm-start. Converges locally. The question is what happens globally.",
        "run-md",
    )
)

cells.append(
    code(
        "r = run_2tri_best(phi_init)\n"
        "phi_out = r['phi']\n"
        "qi_out = list_intersecting_quads(phi_out)\n"
        "print(f\"SLSQP:  nit={r['nit']}  time={r['time']:.2f}s  success={r['success']}  status={r['status']}\")\n"
        "print(f\"        message: {r['message']}\")\n"
        "print()\n"
        "print(f\"  BEFORE   neg_TR={m0['n_tr']:>3d}   QI={len(qi0):>3d}\")\n"
        "print(f\"  AFTER    neg_TR={r['n_tr']:>3d}   QI={len(qi_out):>3d}\")\n"
        "print()\n"
        "print(f\"  L2 from phi_init = {r['l2']:.3f}\")\n"
        "if r['n_tr'] == 0 and len(qi_out) > 0:\n"
        "    print()\n"
        "    print(f\"  >>> LOCAL PASS, GLOBAL FAIL.  {len(qi_out)} non-adjacent quad pair(s) still cross.\")",
        "run-code",
    )
)

cells.append(
    md(
        "## Visualization: initial vs. after best 2-tri SLSQP\n"
        "\n"
        "Row: warped grid + `min(T1, T2)` heatmap. Left column is initial; right column is the post-SLSQP result. The top row's warped grid outlines local 2-triangle folds in dark blue.",
        "viz-md",
    )
)

cells.append(
    code(
        "tri_after = r['tri']\n"
        "local_fold_mask_after = tri_after.min(axis=0) <= 0\n"
        "\n"
        "vmax_tri = max(abs(tri_init).max(), abs(tri_after).max())\n"
        "NL = chr(10)\n"
        "\n"
        "fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')\n"
        "\n"
        "plot_warped_grid(axes[0, 0], phi_init,\n"
        "                  'initial' + NL + f\"TR={m0['n_tr']}  QI={len(qi0)}  min_TR={m0['min_tr']:+.3f}\",\n"
        "                  highlight_mask=local_fold_mask_init)\n"
        "plot_warped_grid(axes[0, 1], phi_out,\n"
        "                  'after best 2-tri SLSQP' + NL + f\"TR={r['n_tr']}  QI={len(qi_out)}  min_TR={r['min_tr']:+.3f}\",\n"
        "                  highlight_mask=local_fold_mask_after)\n"
        "\n"
        "tri0_min = tri_init.min(axis=0)\n"
        "im1 = axes[1, 0].imshow(tri0_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri, aspect='auto')\n"
        "axes[1, 0].set_title(f\"min(T1, T2)  initial\", fontsize=9)\n"
        "axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])\n"
        "tri1_min = tri_after.min(axis=0)\n"
        "im2 = axes[1, 1].imshow(tri1_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri, aspect='auto')\n"
        "axes[1, 1].set_title(f\"min(T1, T2)  after\", fontsize=9)\n"
        "axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])\n"
        "cbar = fig.colorbar(im2, ax=axes[1, :], orientation='horizontal',\n"
        "                    fraction=0.035, pad=0.04, shrink=0.55)\n"
        "cbar.set_label('signed triangle area (blue = fold, red = valid)')\n"
        "plt.suptitle(f'{CASE_KEY} - local fold structure before/after best 2-tri SLSQP', fontsize=11)\n"
        "plt.show()",
        "viz-code",
    )
)

cells.append(
    md(
        "## Global quad-intersection analysis\n"
        "\n"
        "Even though `neg_TR = 0` after the SLSQP run, `QI > 0`. Every cell is locally convex, yet somewhere in the field two non-adjacent quads' edges cross. The sections below pinpoint where, show the pairs, and discuss why the 2-triangle constraint cannot prevent it.",
        "invertibility-md",
    )
)

cells.append(
    code(
        "# List the intersecting pairs and the cells involved.\n"
        "print(f'{len(qi_out)} intersecting pair(s) after best 2-tri SLSQP:')\n"
        "flagged_cells = set()\n"
        "for (a, b) in qi_out:\n"
        "    flagged_cells.add(a); flagged_cells.add(b)\n"
        "print(f'  -> {len(flagged_cells)} distinct cell(s) involved')\n"
        "print()\n"
        "for i, (a, b) in enumerate(qi_out[:20]):\n"
        "    print(f\"  pair {i:>2d}:  cell {a}  <->  cell {b}\")\n"
        "if len(qi_out) > 20:\n"
        "    print(f\"  ... (+{len(qi_out) - 20} more)\")",
        "pairs-list-code",
    )
)

cells.append(
    code(
        "# Overlay the flagged quads on the warped grid.\n"
        "phi = phi_out\n"
        "Hh, Ww = phi[0].shape\n"
        "yy, xx = np.mgrid[:Hh, :Ww]\n"
        "gx = xx + phi[1]; gy = yy + phi[0]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')\n"
        "for i in range(Hh):\n"
        "    ax.plot(xx[i], yy[i], color='#f4f4f4', lw=0.3)\n"
        "for j in range(Ww):\n"
        "    ax.plot(xx[:, j], yy[:, j], color='#f4f4f4', lw=0.3)\n"
        "for i in range(Hh):\n"
        "    ax.plot(gx[i], gy[i], color='#dddddd', lw=0.6)\n"
        "for j in range(Ww):\n"
        "    ax.plot(gx[:, j], gy[:, j], color='#dddddd', lw=0.6)\n"
        "for (cy, cx) in flagged_cells:\n"
        "    poly_x = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "    poly_y = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "    ax.plot(poly_x, poly_y, color='#d32f2f', lw=1.8)\n"
        "ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "ax.set_title(f'{CASE_KEY} post-SLSQP: quads involved in {len(qi_out)} intersecting pair(s)' + NL + \\\n"
        "              f'{len(flagged_cells)} distinct cells outlined in red', fontsize=11)\n"
        "ax.set_xticks([]); ax.set_yticks([])\n"
        "plt.show()",
        "flagged-cells-code",
    )
)

cells.append(
    md(
        "### Zoom: one representative intersecting pair\n"
        "\n"
        "Pick the first pair from the list; draw both quads overlaid with distinct colors on the warped grid background. Makes the crossing geometry concrete.",
        "zoom-md",
    )
)

cells.append(
    code(
        "def plot_pair(phi, pair, title):\n"
        "    (ra, ca), (rb, cb) = pair\n"
        "    Hh, Ww = phi[0].shape\n"
        "    yy, xx = np.mgrid[:Hh, :Ww]\n"
        "    gx = xx + phi[1]; gy = yy + phi[0]\n"
        "    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#ededed', lw=0.5)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#ededed', lw=0.5)\n"
        "    for (cy, cx), col, tag in [((ra, ca), '#ef5350', f'A = cell ({ra}, {ca})'),\n"
        "                                ((rb, cb), '#1565c0', f'B = cell ({rb}, {cb})')]:\n"
        "        poly_x = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "        poly_y = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "        ax.plot(poly_x, poly_y, color=col, lw=2.2, label=tag)\n"
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.legend(loc='best', fontsize=9)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "    plt.show()\n"
        "\n"
        "if qi_out:\n"
        "    plot_pair(phi_out, qi_out[0],\n"
        "              f'{CASE_KEY}  representative intersecting pair' + NL + 'both quads locally valid, globally crossing')",
        "zoom-code",
    )
)

cells.append(
    md(
        "## Damping experiment: how much of the correction can we apply?\n"
        "\n"
        "Interpolate linearly between `phi_init` and `phi_out`:\n"
        "\n"
        "    phi(alpha) = phi_init + alpha * (phi_out - phi_init)\n"
        "\n"
        "At `alpha = 0` we have the uncorrected initial field (`neg_TR` large, `QI` large). At `alpha = 1` we have the SLSQP output (`neg_TR = 0`, `QI > 0`). Scan `alpha` across the interval to see the trajectory.",
        "damping-md",
    )
)

cells.append(
    code(
        "alphas = np.linspace(0.0, 1.0, 21)\n"
        "tr_counts = []; qi_counts = []\n"
        "for a in alphas:\n"
        "    phi_a = phi_init + a * (phi_out - phi_init)\n"
        "    m_a = measure(phi_a)\n"
        "    qi_a = list_intersecting_quads(phi_a)\n"
        "    tr_counts.append(m_a['n_tr'])\n"
        "    qi_counts.append(len(qi_a))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')\n"
        "ax.plot(alphas, tr_counts, 'o-', color='#1565c0', label='neg_TR  (local folds)')\n"
        "ax.plot(alphas, qi_counts, 's-', color='#d32f2f', label='QI  (global intersections)')\n"
        "ax.set_xlabel('alpha  (0 = initial, 1 = best 2-tri SLSQP output)')\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title(f'{CASE_KEY}: damping interpolation between phi_init and phi_out', fontsize=11)\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.legend(fontsize=10)\n"
        "plt.show()\n"
        "\n"
        "# Print the trade-off table\n"
        "print()\n"
        "print(f\"{'alpha':>6s}  {'neg_TR':>7s}  {'QI':>5s}\")\n"
        "print('-' * 24)\n"
        "for a, tr, qi in zip(alphas, tr_counts, qi_counts):\n"
        "    print(f'{a:>6.2f}  {tr:>7d}  {qi:>5d}')",
        "damping-code",
    )
)

cells.append(
    md(
        "## Summary and next steps\n"
        "\n"
        "**What happened.**\n"
        "`03c_20x20_opposite` has two regions with large displacements pointing *toward each other* (the 'opposite motion' pattern). Best 2-triangle SLSQP drives every cell's two triangles to at least `threshold`, so every cell is locally convex and `neg_TR = 0`. But the displacement pushed the two regions into each other, so quads from region A land on top of quads from region B in image space. Each cell is fine *in isolation*; the *map* is not injective.\n"
        "\n"
        "**Why the 2-triangle constraint can't fix this.**\n"
        "The constraint is per-cell. It sees three vertices at a time (per triangle) and evaluates one signed area. It has no way of knowing that two cells ten columns apart have their interiors overlapping. To catch that you need a non-local term - something that evaluates pairs of quads at a time.\n"
        "\n"
        "**Damping trade-off** (from the scan above): reducing `alpha` below 1 re-introduces local folds faster than it removes global ones on this case, because the initial field is *also* heavily folded (large `neg_TR` at `alpha = 0`). So naive damping is not a good fix here - the two failure modes trade off linearly, not in our favor.\n"
        "\n"
        "**Actual next steps** (from [03_demos.ipynb](03_demos.ipynb) 'next steps for global invertibility'):\n"
        "\n"
        "1. **Outer-loop global constraint.** After 2-triangle SLSQP converges, call `list_intersecting_quads`. For each pair in the list, add a repulsion term to the objective (pair-specific signed-distance penalty) and re-solve. Each inner solve stays smooth; only the active pair set changes between outer iterations.\n"
        "2. **Displacement damping toward identity** (not toward phi_init as tested above). Scale `phi_out` toward zero displacement until global intersections vanish. Guaranteed to succeed at `alpha = 0` but gives up all registration quality.\n"
        "3. **Log-barrier on quad-pair signed distance** inside the L-BFGS barrier solver (`iterative_2d_barrier`). Expensive per-iteration but composes with the existing penalty pipeline.\n"
        "\n"
        "Option 1 is the natural next implementation: cheap when pairs are empty, only pays the `O(n_quads^2)` cost when a genuine long-range crossing exists, and the pair list is already in hand after the inner SLSQP completes.",
        "summary",
    )
)

# Compile check
for c in cells:
    if c["cell_type"] == "code":
        try:
            compile("".join(c["source"]), c["id"], "exec")
        except SyntaxError as e:
            print(f'SYNTAX in {c["id"]}: {e}')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f"wrote {path} with {len(cells)} cells")
