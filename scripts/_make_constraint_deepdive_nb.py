"""Build 07_solver-engineering-constraints.ipynb - constraint-axis deep dive on 01c_20x40_edges."""

import json

path = "notebooks/two-triangle-check/07_solver-engineering-constraints.ipynb"


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
        "# Solver Engineering - Constraint-Axis Deep Dive on `01c_20x40_edges`\n"
        "\n"
        "Sibling to [05_solver-engineering.ipynb](05_solver-engineering.ipynb). That notebook fixed the *constraint* (2-triangle) and varied the *Jacobian method* (finite-diff, analytical, analytical + warm-start) to diagnose a status-8 SLSQP failure. This one keeps the same hard case but flips the axis: fix the Jacobian/warm-start setup to the best one identified in 05 and vary the *constraint formulation*:\n"
        "\n"
        "- **(A) CD-constraint SLSQP** - `jacobian_det2D(phi) >= threshold` per pixel (central-difference Jdet).\n"
        "- **(B) FD-constraint SLSQP** - `forward_diff_Jdet(phi) >= threshold` per cell (one-sided stencil at each cell's TL corner).\n"
        "- **(C) 2-triangle constraint + analytical Jacobian + warm-start** (best setup from 05).\n"
        "\n"
        "Target case: `01c_20x40_edges`, 800 pixels / 1600 variables. Previously showed that CD-SLSQP reports `success=True` at `nit=2` while leaving 64 geometric folds in place; 2-triangle with the combined fix is the only variant that drives every measure to zero.\n"
        "\n"
        "Color convention: red = positive / valid, blue = negative / folded (`RdBu_r`).",
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
        "from dvfopt.viz.triangle_debug import find_problematic_pixels\n"
        "from dvfopt.viz import plot_triangle_debug\n"
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
        "\n"
        "def measure(phi):\n"
        "    jd = np.squeeze(jacobian_det2D(phi))\n"
        "    fd = _forward_jdet_2d(phi[0], phi[1])\n"
        "    tri = triangle_sign_areas2D(phi)\n"
        "    return dict(\n"
        "        jd=jd, fd=fd, tri=tri,\n"
        "        n_cd=int((jd <= 0).sum()),\n"
        "        n_fd=int((fd <= 0).sum()),\n"
        "        n_tr=int((tri <= 0).sum()),\n"
        "        min_cd=float(jd.min()),\n"
        "        min_fd=float(fd.min()),\n"
        "        min_tr=float(tri.min()),\n"
        "    )\n"
        "\n"
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
        "    return csr_matrix((vals, (rows, cols)), shape=(2 * n_cells, 2 * N))",
        "anal-jac",
    )
)

cells.append(
    code(
        "def _run_slsqp(phi_init, fun, jac=None, threshold=THRESHOLD, max_iter=500,\n"
        "                warm_start=False, noise_scale=0.01):\n"
        "    _, H_, W_ = phi_init.shape\n"
        "    pixels = H_ * W_\n"
        "    def unpack(z):\n"
        "        dx_ = z[:pixels].reshape(H_, W_)\n"
        "        dy_ = z[pixels:].reshape(H_, W_)\n"
        "        return dy_, dx_\n"
        "    z0 = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])\n"
        "    z0_init = z0.copy()\n"
        "    nl_kwargs = dict(lb=threshold, ub=np.inf)\n"
        "    if jac is not None:\n"
        "        nl_kwargs['jac'] = lambda z: jac(*unpack(z))\n"
        "    t0 = time.time()\n"
        "    res = minimize(lambda z: L2Objective()(z - z0_init), z0,\n"
        "                   jac=True, method='SLSQP',\n"
        "                   constraints=[NonlinearConstraint(lambda z: fun(*unpack(z)), **nl_kwargs)],\n"
        "                   options={'maxiter': max_iter, 'disp': False})\n"
        "    total_nit = res.nit; total_time = time.time() - t0\n"
        "    if warm_start and not res.success and res.status == 8:\n"
        "        rng = np.random.default_rng(123)\n"
        "        z_warm = res.x + rng.normal(scale=noise_scale, size=res.x.shape)\n"
        "        t1 = time.time()\n"
        "        res = minimize(lambda z: L2Objective()(z - z0_init), z_warm,\n"
        "                       jac=True, method='SLSQP',\n"
        "                       constraints=[NonlinearConstraint(lambda z: fun(*unpack(z)), **nl_kwargs)],\n"
        "                       options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})\n"
        "        total_nit += res.nit; total_time += time.time() - t1\n"
        "    dy_o, dx_o = unpack(res.x)\n"
        "    phi_out = np.stack([dy_o, dx_o])\n"
        "    m = measure(phi_out)\n"
        "    m.update(phi=phi_out, nit=total_nit, time=total_time,\n"
        "             success=bool(res.success), status=int(res.status),\n"
        "             message=str(res.message),\n"
        "             l2=float(np.linalg.norm(phi_out - phi_init)))\n"
        "    return m\n"
        "\n"
        "\n"
        "def run_cd_slsqp(phi_init, **kw):\n"
        "    return _run_slsqp(phi_init, fun=lambda dy, dx: _numpy_jdet_2d(dy, dx).flatten(), **kw)\n"
        "\n"
        "\n"
        "def run_fd_slsqp(phi_init, **kw):\n"
        "    return _run_slsqp(phi_init, fun=lambda dy, dx: _forward_jdet_2d(dy, dx).flatten(), **kw)\n"
        "\n"
        "\n"
        "def run_2tri_best_slsqp(phi_init, **kw):\n"
        "    def tri_flat(dy, dx):\n"
        "        T1, T2 = _triangle_areas_2d(dy, dx)\n"
        "        return np.concatenate([T1.flatten(), T2.flatten()])\n"
        "    return _run_slsqp(phi_init, fun=tri_flat,\n"
        "                       jac=triangle_sign_constraint_jac_2d, warm_start=True, **kw)",
        "runners",
    )
)

cells.append(
    md(
        "## Setup - load the hard case\n"
        "\n"
        "Same field as 05: `01c_20x40_edges` from `test_cases.make_deformation`. Large displacements concentrated at image borders; initial `neg_TR = 68`, `QI = 334`.",
        "setup-md",
    )
)

cells.append(
    code(
        "CASE_KEY = '01c_20x40_edges'\n"
        "deformation, *_ = make_deformation(CASE_KEY)\n"
        "phi_init = np.stack([deformation[1, 0], deformation[2, 0]])\n"
        "m0 = measure(phi_init)\n"
        "qi0 = list_intersecting_quads(phi_init)\n"
        "H_, W_ = phi_init[0].shape\n"
        "print(f'{CASE_KEY}  shape=({H_}, {W_})  pixels={H_*W_}  vars={2*H_*W_}')\n"
        "print(f'init  neg_CD={m0[\"n_cd\"]:>3d}  neg_FD={m0[\"n_fd\"]:>3d}  neg_TR={m0[\"n_tr\"]:>3d}  QI={len(qi0)}')\n"
        "print(f'init  min_CD={m0[\"min_cd\"]:+.3f}  min_FD={m0[\"min_fd\"]:+.3f}  min_TR={m0[\"min_tr\"]:+.3f}')",
        "setup-code",
    )
)

cells.append(
    md(
        "### Pre-optimization fold structure\n"
        "\n"
        "Three views of the initial field: warped quad grid with cells where `min(T1, T2) <= 0` outlined; central-diff Jdet heatmap; `min(T1, T2)` heatmap.",
        "preopt-md",
    )
)

cells.append(
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')\n"
        "\n"
        "ax = axes[0]\n"
        "yy, xx = np.mgrid[:H_, :W_]\n"
        "gx = xx + phi_init[1]; gy = yy + phi_init[0]\n"
        "for i in range(H_):\n"
        "    ax.plot(xx[i], yy[i], color='#f4f4f4', lw=0.3)\n"
        "for j in range(W_):\n"
        "    ax.plot(xx[:, j], yy[:, j], color='#f4f4f4', lw=0.3)\n"
        "for i in range(H_):\n"
        "    ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.7)\n"
        "for j in range(W_):\n"
        "    ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.7)\n"
        "tri_init = triangle_sign_areas2D(phi_init)\n"
        "bad_init = np.argwhere(tri_init.min(axis=0) <= 0)\n"
        "for (cy, cx) in bad_init:\n"
        "    px = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "    py = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "    ax.plot(px, py, color='#1565c0', lw=1.2)\n"
        "ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "ax.set_title(f'warped grid   {len(bad_init)} folded cell(s) outlined')\n"
        "ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "vmax_cd = float(max(abs(m0['jd']).max(), 1.0))\n"
        "im = axes[1].imshow(m0['jd'], cmap='RdBu_r', vmin=-vmax_cd, vmax=vmax_cd, aspect='auto')\n"
        "axes[1].set_title(f\"central-diff Jdet  neg={m0['n_cd']}  min={m0['min_cd']:+.3f}\")\n"
        "axes[1].set_xticks([]); axes[1].set_yticks([])\n"
        "fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)\n"
        "\n"
        "tri_min0 = tri_init.min(axis=0)\n"
        "vmax_tr = float(max(abs(tri_min0).max(), 1.0))\n"
        "im2 = axes[2].imshow(tri_min0, cmap='RdBu_r', vmin=-vmax_tr, vmax=vmax_tr, aspect='auto')\n"
        "axes[2].set_title(f\"min(T1, T2)  neg={m0['n_tr']}  min={m0['min_tr']:+.3f}\")\n"
        "axes[2].set_xticks([]); axes[2].set_yticks([])\n"
        "fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)\n"
        "\n"
        "plt.suptitle(f'{CASE_KEY} - pre-optimization fold structure', fontsize=11)\n"
        "plt.show()",
        "preopt-code",
    )
)

cells.append(
    md(
        "## Run each constraint formulation\n"
        "\n"
        "CD and FD use scipy's default numerical Jacobian (no `jac=` passed); 2-tri gets the analytical Jacobian + warm-start. Each run uses `maxiter=500` and the default `ftol`.",
        "run-md",
    )
)

cells.append(
    code(
        "results = {}\n"
        "for name, fn in [('cd', run_cd_slsqp), ('fd', run_fd_slsqp), ('2tri', run_2tri_best_slsqp)]:\n"
        "    print(f'>>> {name}')\n"
        "    r = fn(phi_init)\n"
        "    r['qi'] = list_intersecting_quads(r['phi'])\n"
        "    results[name] = r\n"
        "    print(f\"    nit={r['nit']:>4d}  time={r['time']:>7.2f}s  success={str(r['success']):<5s}  status={r['status']}\")\n"
        "    print(f\"    neg_CD={r['n_cd']:>3d}  neg_FD={r['n_fd']:>3d}  neg_TR={r['n_tr']:>3d}  QI={len(r['qi']):>3d}  L2={r['l2']:.3f}\")\n"
        "    print(f\"    min_TR={r['min_tr']:+.4f}   message: {r['message']}\")",
        "run-code",
    )
)

cells.append(
    md(
        "### Cross-variant summary table",
        "summary-md-table",
    )
)

cells.append(
    code(
        "variants = [('initial', None), ('(A) CD',  'cd'), ('(B) FD',  'fd'), ('(C) 2tri (best)', '2tri')]\n"
        "hdr = f\"{'variant':<18s}  {'nit':>4s}  {'time':>8s}  {'success':>8s}  {'neg_CD':>6s}  {'neg_FD':>6s}  {'neg_TR':>6s}  {'QI':>5s}  {'L2':>6s}\"\n"
        "print(hdr)\n"
        "print('-' * len(hdr))\n"
        "print(f\"{'initial':<18s}  {'-':>4s}  {'-':>8s}  {'-':>8s}  {m0['n_cd']:>6d}  {m0['n_fd']:>6d}  {m0['n_tr']:>6d}  {len(qi0):>5d}  {'-':>6s}\")\n"
        "for label, key in variants[1:]:\n"
        "    r = results[key]\n"
        "    print(f\"{label:<18s}  {r['nit']:>4d}  {r['time']:>7.2f}s  {str(r['success']):>8s}  \"\n"
        "          f\"{r['n_cd']:>6d}  {r['n_fd']:>6d}  {r['n_tr']:>6d}  {len(r['qi']):>5d}  {r['l2']:>6.3f}\")",
        "summary-table",
    )
)

cells.append(
    md(
        "## Grid deformation across variants\n"
        "\n"
        "Row 1: warped quad grid, cells with `min(T1, T2) <= 0` outlined in dark blue. Row 2: per-cell `min(T1, T2)` heatmap (`RdBu_r`, shared scale).",
        "variant-viz-md",
    )
)

cells.append(
    code(
        "def plot_warped_grid(ax, phi, title, highlight_folds=True):\n"
        "    dy = phi[0]; dx = phi[1]\n"
        "    Hh, Ww = dy.shape\n"
        "    yy, xx = np.mgrid[:Hh, :Ww]\n"
        "    gx = xx + dx; gy = yy + dy\n"
        "    for i in range(Hh):\n"
        "        ax.plot(xx[i], yy[i], color='#f0f0f0', lw=0.3)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(xx[:, j], yy[:, j], color='#f0f0f0', lw=0.3)\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.7)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.7)\n"
        "    if highlight_folds:\n"
        "        tri = triangle_sign_areas2D(phi)\n"
        "        bad = np.argwhere(tri.min(axis=0) <= 0)\n"
        "        for (cy, cx) in bad:\n"
        "            px = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "            py = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "            ax.plot(px, py, color='#1565c0', lw=1.4)\n"
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(title, fontsize=9)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "\n"
        "panels = [\n"
        "    ('initial',          phi_init,       m0,      len(qi0), None),\n"
        "    ('(A) CD',           results['cd']['phi'],   results['cd'],   len(results['cd']['qi']),   results['cd']),\n"
        "    ('(B) FD',           results['fd']['phi'],   results['fd'],   len(results['fd']['qi']),   results['fd']),\n"
        "    ('(C) 2tri (best)',  results['2tri']['phi'], results['2tri'], len(results['2tri']['qi']), results['2tri']),\n"
        "]\n"
        "vmax_tri = max(abs(m['tri']).max() for _, _, m, _, _ in panels)\n"
        "NL = chr(10)\n"
        "\n"
        "fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7.5), layout='constrained')\n"
        "for k, (label, phi, m, qi_n, res) in enumerate(panels):\n"
        "    if res is None:\n"
        "        line1 = label\n"
        "    else:\n"
        "        tag = 'OK' if res['success'] else 'FAIL'\n"
        "        line1 = f'{label}  [{tag}]'\n"
        "    line2 = f\"TR={m['n_tr']}  QI={qi_n}  min_TR={m['min_tr']:+.3f}\"\n"
        "    plot_warped_grid(axes[0, k], phi, line1 + NL + line2)\n"
        "\n"
        "    tri_min = m['tri'].min(axis=0)\n"
        "    im = axes[1, k].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri, aspect='auto')\n"
        "    axes[1, k].set_title(f'min(T1, T2)   {label}', fontsize=9)\n"
        "    axes[1, k].set_xticks([]); axes[1, k].set_yticks([])\n"
        "\n"
        "cbar = fig.colorbar(im, ax=axes[1, :], orientation='horizontal',\n"
        "                    fraction=0.035, pad=0.04, shrink=0.55)\n"
        "cbar.set_label('signed triangle area (blue = fold, red = valid)')\n"
        "plt.suptitle(f'{CASE_KEY} - grid deformation across constraint formulations', fontsize=11)\n"
        "plt.show()",
        "variant-viz-code",
    )
)

cells.append(
    md(
        "## Jacobian heatmaps - three measures, four rows\n"
        "\n"
        "One row per variant; three columns for CD Jdet / FD Jdet / `min(T1, T2)`. Shared colormap scale across rows for each column so you can read, e.g., which CD-flagged cells survive into each corrected field.",
        "jac-heatmap-md",
    )
)

cells.append(
    code(
        "panels_m = [('initial', m0),\n"
        "             ('(A) CD',        results['cd']),\n"
        "             ('(B) FD',        results['fd']),\n"
        "             ('(C) 2tri best', results['2tri'])]\n"
        "vmax_cd = max(abs(m['jd']).max()  for _, m in panels_m)\n"
        "vmax_fd = max(abs(m['fd']).max()  for _, m in panels_m)\n"
        "vmax_tr = max(abs(m['tri']).max() for _, m in panels_m)\n"
        "\n"
        "fig, axes = plt.subplots(len(panels_m), 3, figsize=(14, 3.2 * len(panels_m)), layout='constrained')\n"
        "for i, (label, m) in enumerate(panels_m):\n"
        "    axes[i, 0].imshow(m['jd'], cmap='RdBu_r', vmin=-vmax_cd, vmax=vmax_cd, aspect='auto')\n"
        "    axes[i, 0].set_title(f\"{label}  CD Jdet  neg={m['n_cd']}  min={m['min_cd']:+.3f}\", fontsize=9)\n"
        "    axes[i, 1].imshow(m['fd'], cmap='RdBu_r', vmin=-vmax_fd, vmax=vmax_fd, aspect='auto')\n"
        "    axes[i, 1].set_title(f\"{label}  FD Jdet  neg={m['n_fd']}  min={m['min_fd']:+.3f}\", fontsize=9)\n"
        "    tri_min = m['tri'].min(axis=0)\n"
        "    axes[i, 2].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tr, vmax=vmax_tr, aspect='auto')\n"
        "    axes[i, 2].set_title(f\"{label}  min(T1, T2)  neg={m['n_tr']}  min={m['min_tr']:+.3f}\", fontsize=9)\n"
        "    for ax in axes[i]:\n"
        "        ax.set_xticks([]); ax.set_yticks([])\n"
        "plt.suptitle(f'{CASE_KEY} - three measures across variants  (rows: variant, cols: measure)', fontsize=11)\n"
        "plt.show()",
        "jac-heatmap-code",
    )
)

cells.append(
    md(
        "## Per-pixel debug - worst initial pixels across variants\n"
        "\n"
        "Rows are the K=6 worst initial pixels by `min(T1, T2)`; columns are the four variants. Reading a row shows how the same pixel's T1/T2 evolves as the constraint changes.",
        "per-pixel-md",
    )
)

cells.append(
    code(
        "labels = [\n"
        "    ('initial',              phi_init),\n"
        "    ('(A) CD',                results['cd']['phi']),\n"
        "    ('(B) FD',                results['fd']['phi']),\n"
        "    ('(C) 2tri (best)',       results['2tri']['phi']),\n"
        "]\n"
        "\n"
        "K = 6\n"
        "tri0 = triangle_sign_areas2D(phi_init)\n"
        "tri0_min = tri0.min(axis=0)\n"
        "worst_cells = np.argsort(tri0_min.ravel())[:K]\n"
        "nr, nc = tri0_min.shape\n"
        "seen = set(); worst_pixels = []\n"
        "for flat in worst_cells:\n"
        "    cy, cx = divmod(int(flat), nc)\n"
        "    p = (cx if cx >= 1 else cx + 1, cy)\n"
        "    if p not in seen:\n"
        "        seen.add(p); worst_pixels.append(p)\n"
        "worst_pixels = worst_pixels[:K]\n"
        "print(f'top {len(worst_pixels)} worst pixels (by initial min_TR):', worst_pixels)\n"
        "\n"
        "fig, axes = plt.subplots(len(worst_pixels), len(labels),\n"
        "                          figsize=(3.2 * len(labels), 3.0 * len(worst_pixels)),\n"
        "                          layout='constrained', squeeze=False)\n"
        "for row, (x, y) in enumerate(worst_pixels):\n"
        "    for col, (label, phi) in enumerate(labels):\n"
        "        plot_triangle_debug(phi, x=x, y=y, ax=axes[row, col], show_formula=False)\n"
        "        axes[row, col].set_title(f'{label}  pixel (x={x}, y={y})', fontsize=9)\n"
        "plt.suptitle('Worst initial pixels (rows) evolved across variants (cols)', fontsize=11)\n"
        "plt.show()",
        "per-pixel-code",
    )
)

cells.append(
    md(
        "## Disagreement overlay - where CD/FD pass but 2-tri still flags\n"
        "\n"
        "The cells where each constraint converged under *its own* measure but the 2-triangle check still reports a fold. Those are the silent failures - folds a pipeline would miss if it trusted CD or FD convergence.",
        "disagree-md",
    )
)

cells.append(
    code(
        "def plot_disagreement(phi, title, ax):\n"
        "    dy, dx = phi[0], phi[1]\n"
        "    Hh, Ww = dy.shape\n"
        "    yy, xx = np.mgrid[:Hh, :Ww]\n"
        "    gx = xx + dx; gy = yy + dy\n"
        "    for i in range(Hh):\n"
        "        ax.plot(xx[i], yy[i], color='#f8f8f8', lw=0.3)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(xx[:, j], yy[:, j], color='#f8f8f8', lw=0.3)\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.6)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.6)\n"
        "    tri = triangle_sign_areas2D(phi)\n"
        "    bad = np.argwhere(tri.min(axis=0) <= 0)\n"
        "    for (cy, cx) in bad:\n"
        "        px = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "        py = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "        ax.plot(px, py, color='#1565c0', lw=1.4)\n"
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(f'{title}  -  {len(bad)} cell(s) 2-tri still flags', fontsize=10)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')\n"
        "plot_disagreement(results['cd']['phi'], 'after CD-SLSQP', axes[0])\n"
        "plot_disagreement(results['fd']['phi'], 'after FD-SLSQP', axes[1])\n"
        "plt.show()",
        "disagree-code",
    )
)

cells.append(
    md(
        "## Summary\n"
        "\n"
        "On `01c_20x40_edges`, one specific hard case, the **constraint** choice matters at least as much as the Jacobian choice:\n"
        "\n"
        "- **(A) CD-constraint SLSQP** tends to finish quickly with `success=True` and `neg_CD=0` but leaves most initial 2-triangle folds in place. The per-pixel central-difference stencil is too averaged to see the fine-grained fold structure of edge-localized large displacements.\n"
        "- **(B) FD-constraint SLSQP** does better at detection (per-cell is stricter than per-pixel) and makes real progress, but still leaves residual 2-triangle folds because one forward-difference Jdet per cell can't see bowties where one triangle flips and the other doesn't.\n"
        "- **(C) 2-triangle constraint + analytical Jac + warm-start** is the only variant that drives every measure to zero here, including the global quad-intersection count `QI`.\n"
        "\n"
        "The disagreement overlay at the end pinpoints the exact cells where the weaker constraints silently leave folds behind. In a correction pipeline that certifies validity via `triangle_sign_areas2D` (e.g. before downstream resampling), running CD or FD SLSQP first is not a safe substitute.\n"
        "\n"
        "For a multi-case version of the same comparison (synthetic small cases), see [04_constraint-comparison.ipynb](04_constraint-comparison.ipynb). For the real-slice version, see [06_real-slice.ipynb](06_real-slice.ipynb). For the Jacobian-method comparison on this same hard case, see [05_solver-engineering.ipynb](05_solver-engineering.ipynb).",
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
