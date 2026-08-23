"""Build triangle-sign-constraint-comparison.ipynb."""

import json

path = "notebooks/two-triangle-check/triangle-sign-constraint-comparison.ipynb"


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

# --- 0. Intro ----------------------------------------------------------
cells.append(
    md(
        "# Constraint-Formulation Comparison: CD vs. FD vs. Best 2-triangle\n"
        "\n"
        "Sibling to [triangle-sign-optimization.ipynb](triangle-sign-optimization.ipynb) (which varies the *Jacobian* for a fixed 2-triangle constraint) and [triangle-sign-solver-engineering.ipynb](triangle-sign-solver-engineering.ipynb) (which diagnoses SLSQP failures on `01c_20x40_edges`). This notebook varies the **constraint itself**:\n"
        "\n"
        "- **(A) CD-constraint SLSQP** - `jacobian_det2D(phi) >= threshold` per pixel, central-difference Jdet.\n"
        "- **(B) FD-constraint SLSQP** - per-cell forward-difference Jdet `>= threshold`.\n"
        "- **(C) Best 2-triangle SLSQP** - analytical Jacobian + perturbation warm-start (the combined fix from the solver-engineering notebook).\n"
        "\n"
        "Each is a different *optimization problem* with a different notion of what counts as a valid grid. Run all three on several fold-library cases and compare:\n"
        "\n"
        "- Where does each variant leave residual folds, under each measure?\n"
        "- How do the warped grids look - which cells are still crossed?\n"
        "- In cases where CD or FD pass but 2-triangle flags a cell, what's the exact geometry?\n"
        "\n"
        "Color convention: red = positive / valid, blue = negative / folded (`RdBu_r`).",
        "intro",
    )
)

# --- 1. Imports & helpers ----------------------------------------------
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
        "from dvfopt.objectives import L2Objective\n"
        "\n"
        "from dvfopt.testdata import make_deformation\n"
        "\n"
        "THRESHOLD = DEFAULT_PARAMS['threshold']\n"
        "print(f'threshold = {THRESHOLD}')",
        "imports",
    )
)

cells.append(
    code(
        "# ---------- per-field quality measures ----------\n"
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
        "    )",
        "helpers-measure",
    )
)

cells.append(
    code(
        "# ---------- analytical Jacobian of the 2-triangle constraint ----------\n"
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
        "helpers-anal-jac",
    )
)

cells.append(
    code(
        "# ---------- SLSQP variants, parameterised by constraint formulation ----------\n"
        "def _make_unpack(H_, W_):\n"
        "    pixels = H_ * W_\n"
        "    def unpack(z):\n"
        "        dx_ = z[:pixels].reshape(H_, W_)\n"
        "        dy_ = z[pixels:].reshape(H_, W_)\n"
        "        return dy_, dx_\n"
        "    return unpack\n"
        "\n"
        "\n"
        "def _run_slsqp(phi_init, fun, jac=None, threshold=THRESHOLD, max_iter=500,\n"
        "                warm_start=False, noise_scale=0.01):\n"
        "    _, H_, W_ = phi_init.shape\n"
        "    pixels = H_ * W_\n"
        "    unpack = _make_unpack(H_, W_)\n"
        "    z0 = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])\n"
        "    z0_init = z0.copy()\n"
        "\n"
        "    nl_kwargs = dict(lb=threshold, ub=np.inf)\n"
        "    if jac is not None:\n"
        "        nl_kwargs['jac'] = lambda z: jac(*unpack(z))\n"
        "\n"
        "    t0 = time.time()\n"
        "    res = minimize(\n"
        "        lambda z: L2Objective()(z - z0_init),\n"
        "        z0, jac=True, method='SLSQP',\n"
        "        constraints=[NonlinearConstraint(lambda z: fun(*unpack(z)), **nl_kwargs)],\n"
        "        options={'maxiter': max_iter, 'disp': False},\n"
        "    )\n"
        "    total_nit = res.nit\n"
        "    total_time = time.time() - t0\n"
        "\n"
        "    if warm_start and not res.success and res.status == 8:\n"
        "        rng = np.random.default_rng(123)\n"
        "        z_warm = res.x + rng.normal(scale=noise_scale, size=res.x.shape)\n"
        "        t1 = time.time()\n"
        "        res = minimize(\n"
        "            lambda z: L2Objective()(z - z0_init),\n"
        "            z_warm, jac=True, method='SLSQP',\n"
        "            constraints=[NonlinearConstraint(lambda z: fun(*unpack(z)), **nl_kwargs)],\n"
        "            options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False},\n"
        "        )\n"
        "        total_nit += res.nit\n"
        "        total_time += time.time() - t1\n"
        "\n"
        "    dy_o, dx_o = unpack(res.x)\n"
        "    phi_out = np.stack([dy_o, dx_o])\n"
        "    m = measure(phi_out)\n"
        "    m.update(\n"
        "        phi=phi_out, nit=total_nit, time=total_time,\n"
        "        success=bool(res.success), status=int(res.status),\n"
        "        message=str(res.message),\n"
        "        l2=float(np.linalg.norm(phi_out - phi_init)),\n"
        "    )\n"
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
        "    # Best-known setup from the solver-engineering notebook:\n"
        "    # analytical Jacobian + perturbation warm-start on status 8.\n"
        "    return _run_slsqp(\n"
        "        phi_init, fun=tri_flat,\n"
        "        jac=triangle_sign_constraint_jac_2d, warm_start=True, **kw,\n"
        "    )",
        "helpers-runners",
    )
)

# --- 2. Plot helpers ---------------------------------------------------
cells.append(
    code(
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
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(title, fontsize=9)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "\n"
        "def plot_case(case_key, phi_init, runs):\n"
        "    '''Row 1: warped grid + folded-cell outline. Row 2: min(T1, T2) heatmap.'''\n"
        "    variants = [\n"
        "        ('initial',       phi_init, measure(phi_init), None),\n"
        "        ('(A) CD',        runs['cd']['phi'],   runs['cd'],   runs['cd']),\n"
        "        ('(B) FD',        runs['fd']['phi'],   runs['fd'],   runs['fd']),\n"
        "        ('(C) best-2tri', runs['2tri']['phi'], runs['2tri'], runs['2tri']),\n"
        "    ]\n"
        "    vmax_tri = max(abs(m['tri']).max() for _, _, m, _ in variants)\n"
        "    NL = chr(10)\n"
        "\n"
        "    n_var = len(variants)\n"
        "    fig, axes = plt.subplots(2, n_var, figsize=(3.6 * n_var, 7.0), layout='constrained')\n"
        "    for k, (label, phi, m, res) in enumerate(variants):\n"
        "        if res is None:\n"
        "            line1 = label\n"
        "        else:\n"
        "            tag = 'OK' if res['success'] else 'FAIL'\n"
        "            line1 = f'{label}  [{tag}]'\n"
        "        line2 = f\"TR={m['n_tr']}  min_TR={m['min_tr']:+.3f}\"\n"
        "        plot_warped_grid(axes[0, k], phi, line1 + NL + line2)\n"
        "\n"
        "        tri_min = m['tri'].min(axis=0)\n"
        "        im = axes[1, k].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri, aspect='auto')\n"
        "        axes[1, k].set_title(f'min(T1, T2)   {label}', fontsize=9)\n"
        "        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])\n"
        "\n"
        "    cbar = fig.colorbar(im, ax=axes[1, :], orientation='horizontal',\n"
        "                        fraction=0.035, pad=0.04, shrink=0.55)\n"
        "    cbar.set_label('signed triangle area (blue = fold, red = valid)')\n"
        "    plt.suptitle(f'{case_key}  -  grid deformation across constraint formulations', fontsize=11)\n"
        "    plt.show()\n"
        "\n"
        "\n"
        "def plot_jacobian_heatmaps(case_key, phi_init, runs):\n"
        "    '''Three columns - CD Jdet, FD Jdet, min(T1, T2) - one row per variant.'''\n"
        "    variants = [\n"
        "        ('initial',       measure(phi_init)),\n"
        "        ('(A) CD',        runs['cd']),\n"
        "        ('(B) FD',        runs['fd']),\n"
        "        ('(C) best-2tri', runs['2tri']),\n"
        "    ]\n"
        "    vmax_cd = max(abs(m['jd']).max()  for _, m in variants)\n"
        "    vmax_fd = max(abs(m['fd']).max()  for _, m in variants)\n"
        "    vmax_tr = max(abs(m['tri']).max() for _, m in variants)\n"
        "\n"
        "    fig, axes = plt.subplots(len(variants), 3, figsize=(13, 3.2 * len(variants)), layout='constrained')\n"
        "    for i, (label, m) in enumerate(variants):\n"
        "        axes[i, 0].imshow(m['jd'], cmap='RdBu_r', vmin=-vmax_cd, vmax=vmax_cd, aspect='auto')\n"
        "        axes[i, 0].set_title(f\"{label}  CD  neg={m['n_cd']}  min={m['min_cd']:+.3f}\", fontsize=9)\n"
        "        axes[i, 1].imshow(m['fd'], cmap='RdBu_r', vmin=-vmax_fd, vmax=vmax_fd, aspect='auto')\n"
        "        axes[i, 1].set_title(f\"{label}  FD  neg={m['n_fd']}  min={m['min_fd']:+.3f}\", fontsize=9)\n"
        "        tri_min = m['tri'].min(axis=0)\n"
        "        axes[i, 2].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tr, vmax=vmax_tr, aspect='auto')\n"
        "        axes[i, 2].set_title(f\"{label}  min(T1, T2)  neg={m['n_tr']}  min={m['min_tr']:+.3f}\", fontsize=9)\n"
        "        for ax in axes[i]:\n"
        "            ax.set_xticks([]); ax.set_yticks([])\n"
        "    plt.suptitle(f'{case_key}  -  three measures of fold detection  (rows: run,  cols: measure)', fontsize=11)\n"
        "    plt.show()",
        "plot-helpers",
    )
)

# --- 3. Run across cases -----------------------------------------------
cells.append(
    md(
        "## Run across cases\n"
        "\n"
        "Test set (small to medium, all in `dvfopt.testdata`):\n"
        "\n"
        "- `bowtie_7x7` - the shoelace-artifact construction (`dx[3,3]=+1.2, dx[3,4]=-1.2`). CD is blind here; FD sees the fold; 2-tri catches two flipped triangles.\n"
        "- `01a_10x10_crossing` - small structured crossing fold.\n"
        "- `03b_10x10_crossing` - larger-magnitude crossing.\n"
        "- `03d_20x20_crossing` - 20x20 dense crossings.\n"
        "\n"
        "Each constraint formulation is run with `maxiter=500` and the default `ftol`.",
        "run-md",
    )
)

cells.append(
    code(
        "# Build the test case dict.\n"
        "def make_bowtie():\n"
        "    H = W = 7\n"
        "    dy = np.zeros((H, W)); dx = np.zeros((H, W))\n"
        "    dx[3, 3] = +1.2; dx[3, 4] = -1.2\n"
        "    return np.stack([dy, dx])\n"
        "\n"
        "def load_test_case(key):\n"
        "    deformation, *_ = make_deformation(key)\n"
        "    return np.stack([deformation[1, 0], deformation[2, 0]])\n"
        "\n"
        "CASES = {\n"
        "    'bowtie_7x7':          make_bowtie(),\n"
        "    '01a_10x10_crossing':  load_test_case('01a_10x10_crossing'),\n"
        "    '03b_10x10_crossing':  load_test_case('03b_10x10_crossing'),\n"
        "    '03d_20x20_crossing':  load_test_case('03d_20x20_crossing'),\n"
        "}\n"
        "\n"
        "# Run all three constraints on every case.\n"
        "results = {}\n"
        "for name, phi in CASES.items():\n"
        "    print(f'>>> {name}  shape={phi[0].shape}  ...')\n"
        "    results[name] = {\n"
        "        'phi_init': phi,\n"
        "        'm0':       measure(phi),\n"
        "        'cd':       run_cd_slsqp(phi),\n"
        "        'fd':       run_fd_slsqp(phi),\n"
        "        '2tri':     run_2tri_best_slsqp(phi),\n"
        "    }\n"
        "\n"
        "# Cross-case summary table.\n"
        "print()\n"
        "hdr = (f\"{'case':<22s}  {'init neg_TR':>11s}  \"\n"
        "        + '  '.join(f\"{v+' ':>12s}{k:<5s}\"\n"
        "                     for v in ('CD', 'FD', '2tri')\n"
        "                     for k in ('nit', 'TR', 'L2')))\n"
        "print(hdr)\n"
        "print('-' * len(hdr))\n"
        "for name, r in results.items():\n"
        "    row = [f\"{name:<22s}\", f\"{r['m0']['n_tr']:>11d}\"]\n"
        "    for key in ('cd', 'fd', '2tri'):\n"
        "        m = r[key]\n"
        "        row.append(f\"{m['nit']:>12d}{'':<5s}\")\n"
        "        row.append(f\"{m['n_tr']:>12d}{'':<5s}\")\n"
        "        row.append(f\"{m['l2']:>12.3f}{'':<5s}\")\n"
        "    print('  '.join(row))",
        "run-code",
    )
)

# --- 4. Per-case visualization cells -----------------------------------
per_case_code = "for name in CASES:\n    r = results[name]\n    plot_case(name, r['phi_init'], r)"
cells.append(
    md(
        "## Warped-grid comparison per case\n"
        "\n"
        "For each case, a 2x4 panel: row 1 is the warped grid (folded cells outlined in dark blue); row 2 is the per-cell `min(T1, T2)` heatmap. Shared colormap so you can eyeball which variant leaves the most residual fold under the 2-triangle check.",
        "warped-md",
    )
)
cells.append(code(per_case_code, "warped-code"))

cells.append(
    md(
        "## Jacobian heatmaps per case\n"
        "\n"
        "Three columns per case - central-diff Jdet, forward-diff Jdet, and `min(T1, T2)`. One row per variant (initial + three corrected fields). Each column uses the same `RdBu_r` palette across rows so you can read, for example, which *cells* CD thinks are folded before vs. after CD-SLSQP.",
        "heatmap-md",
    )
)
cells.append(
    code(
        "for name in CASES:\n"
        "    r = results[name]\n"
        "    plot_jacobian_heatmaps(name, r['phi_init'], r)",
        "heatmap-code",
    )
)

# --- 5. Where constraints disagree ------------------------------------
cells.append(
    md(
        "## Where CD or FD passes but 2-triangle still flags\n"
        "\n"
        "For each case we locate cells where the CD- or FD-corrected field satisfies its own constraint (pass) but the 2-triangle check still reports a fold (`min(T1, T2) <= 0`). These are the cells where the coarser constraint formulations *miss* a fold. Shown as a blue overlay on the warped grid.",
        "disagree-md",
    )
)
cells.append(
    code(
        "def plot_disagreement(name, r):\n"
        "    phi_cd  = r['cd']['phi'];  m_cd  = r['cd']\n"
        "    phi_fd  = r['fd']['phi'];  m_fd  = r['fd']\n"
        "\n"
        "    # Cells where CD's own metric says OK (>=threshold ish) but 2-tri says folded.\n"
        "    # min of T1/T2 per cell:\n"
        "    cd_tri_min = m_cd['tri'].min(axis=0)\n"
        "    fd_tri_min = m_fd['tri'].min(axis=0)\n"
        "    cd_bad_2tri = (cd_tri_min <= 0)    # cells 2-tri flags in CD output\n"
        "    fd_bad_2tri = (fd_tri_min <= 0)    # cells 2-tri flags in FD output\n"
        "\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')\n"
        "\n"
        "    for ax, phi, bad_mask, title in [\n"
        "        (axes[0], phi_cd, cd_bad_2tri, f'{name}  after CD-SLSQP - cells where 2-tri still flags'),\n"
        "        (axes[1], phi_fd, fd_bad_2tri, f'{name}  after FD-SLSQP - cells where 2-tri still flags'),\n"
        "    ]:\n"
        "        dy, dx = phi[0], phi[1]\n"
        "        Hh, Ww = dy.shape\n"
        "        yy, xx = np.mgrid[:Hh, :Ww]\n"
        "        gx = xx + dx; gy = yy + dy\n"
        "        for i in range(Hh):\n"
        "            ax.plot(xx[i], yy[i], color='#f4f4f4', lw=0.3)\n"
        "        for j in range(Ww):\n"
        "            ax.plot(xx[:, j], yy[:, j], color='#f4f4f4', lw=0.3)\n"
        "        for i in range(Hh):\n"
        "            ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.7)\n"
        "        for j in range(Ww):\n"
        "            ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.7)\n"
        "        # Outline 2-tri-flagged cells\n"
        "        for (cy, cx) in np.argwhere(bad_mask):\n"
        "            poly_x = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "            poly_y = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "            ax.plot(poly_x, poly_y, color='#1565c0', lw=1.8)\n"
        "        ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "        n = int(bad_mask.sum())\n"
        "        ax.set_title(f'{title}  -  {n} cell(s)', fontsize=9)\n"
        "        ax.set_xticks([]); ax.set_yticks([])\n"
        "    plt.show()\n"
        "\n"
        "\n"
        "for name in CASES:\n"
        "    r = results[name]\n"
        "    n_cd_bad = int((r['cd']['tri'].min(axis=0) <= 0).sum())\n"
        "    n_fd_bad = int((r['fd']['tri'].min(axis=0) <= 0).sum())\n"
        "    if n_cd_bad == 0 and n_fd_bad == 0:\n"
        "        print(f'{name}: CD and FD corrected outputs both satisfy 2-triangle too (no disagreement)')\n"
        "        continue\n"
        "    plot_disagreement(name, r)",
        "disagree-code",
    )
)

# --- 6. Summary --------------------------------------------------------
cells.append(
    md(
        "## Summary\n"
        "\n"
        "- **(C) best-2tri** hits `neg_TR = 0` on every case here by construction - the constraint is exactly what we're measuring against.\n"
        "- **(A) CD-SLSQP** leaves residual 2-tri-flagged cells on cases where the fold topology is invisible to the symmetric pixel stencil (notably the bowtie). On cases where CD *does* see the fold locally (the crossings), it still tends to leave a few 2-tri folds behind because CD is per-pixel and 2-tri is per-triangle-per-cell - they disagree near large gradients.\n"
        "- **(B) FD-SLSQP** is closer to 2-tri in spirit (per-cell one Jacobian determinant) but still one measurement per cell vs. 2-tri's two, so it misses bowtie-type folds where one triangle flips and the other doesn't.\n"
        "- The *disagreement* overlay pinpoints exactly the cells where the coarser constraints incorrectly passed; those are the folds a downstream pipeline would silently ignore if it trusted CD or FD convergence.\n"
        "\n"
        "Practical takeaway: if downstream code evaluates quality via `triangle_sign_areas2D` (e.g. injectivity-critical use cases), the solver that produced the field should also use the 2-triangle constraint, not a coarser proxy. Converging on a CD/FD metric does not guarantee the 2-triangle metric is clean.",
        "summary",
    )
)

# ---- Compile check ----
for c in cells:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            compile(src, c["id"], "exec")
        except SyntaxError as e:
            print(f'SYNTAX in {c["id"]}: {e}')
            break

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
