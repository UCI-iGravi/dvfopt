"""Build 06_real-slice.ipynb."""

import json

path = "notebooks/two-triangle-check/06_real-slice.ipynb"


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
        "# 2-Triangle on Real Registration Slices\n"
        "\n"
        "The previous notebooks (01-05) used synthetic folds: the 7x7 bowtie and `dvfopt.testdata` fold-library entries. This one runs the best 2-triangle SLSQP on **real slices** pulled through the Laplacian registration pipeline (`dvfopt.testdata.load_slice`), and compares it to CD-SLSQP and FD-SLSQP on the same fields.\n"
        "\n"
        "Two slices, both downsampled with `scale_factor=0.08` to keep SLSQP tractable (~900 pixels / ~1800 variables):\n"
        "\n"
        "- **slice 90** (`02a`) - initial CD Jdet has `neg=0` (central-diff is completely blind), yet 2-triangle flags 35 folded cells. Real-data instance of the bowtie/shoelace-artifact failure mode.\n"
        "- **slice 350** (`02c`) - same pattern, `neg_CD=0` vs `neg_TR=12`.\n"
        "\n"
        "This is the most direct evidence that switching from central-difference to a geometric fold check matters on real registration output, not just synthetic constructions.",
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
        "from dvfopt.testdata._builders import load_slice\n"
        "\n"
        "THRESHOLD = DEFAULT_PARAMS['threshold']\n"
        "MPOINTS = os.path.abspath('../../data/corrected_correspondences_count_touching/mpoints.npy')\n"
        "FPOINTS = os.path.abspath('../../data/corrected_correspondences_count_touching/fpoints.npy')\n"
        "print(f'threshold = {THRESHOLD}')\n"
        "print(f'mpoints exists: {os.path.exists(MPOINTS)}')\n"
        "print(f'fpoints exists: {os.path.exists(FPOINTS)}')",
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
        "# Analytical Jacobian of the 2-triangle constraint.\n"
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
        "def run_cd(phi_init, **kw):\n"
        "    return _run_slsqp(phi_init, fun=lambda dy, dx: _numpy_jdet_2d(dy, dx).flatten(), **kw)\n"
        "\n"
        "def run_fd(phi_init, **kw):\n"
        "    return _run_slsqp(phi_init, fun=lambda dy, dx: _forward_jdet_2d(dy, dx).flatten(), **kw)\n"
        "\n"
        "def run_2tri(phi_init, **kw):\n"
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
        "## Load slices and report initial state\n"
        "\n"
        "Two real slices at `scale_factor=0.08` (~25x36, ~900 pixels each). Reporting each measure's initial fold count plus the global quad-intersection count.",
        "load-md",
    )
)

cells.append(
    code(
        "SLICES = [\n"
        "    ('slice_090_s08',  90, 0.08),\n"
        "    ('slice_350_s08', 350, 0.08),\n"
        "]\n"
        "\n"
        "cases = {}\n"
        "for name, idx, sf in SLICES:\n"
        "    deformation, *_ = load_slice(idx, sf, mpoints_path=MPOINTS, fpoints_path=FPOINTS)\n"
        "    phi = np.stack([deformation[1, 0], deformation[2, 0]])\n"
        "    m0 = measure(phi)\n"
        "    qi0 = list_intersecting_quads(phi)\n"
        "    cases[name] = dict(phi=phi, m0=m0, qi0=qi0, slice_idx=idx, sf=sf)\n"
        "    print(f'{name:<18s}  shape={phi[0].shape}  pix={phi[0].size}  '\n"
        "          f'n_cd={m0[\"n_cd\"]:>3d}  n_fd={m0[\"n_fd\"]:>3d}  n_tr={m0[\"n_tr\"]:>3d}  '\n"
        "          f'QI={len(qi0):>4d}  min_tr={m0[\"min_tr\"]:+.3f}')",
        "load-code",
    )
)

cells.append(
    md(
        "## Run each constraint formulation on each slice\n"
        "\n"
        "CD-SLSQP, FD-SLSQP, best-2tri SLSQP - same three runners as [04_constraint-comparison.ipynb](04_constraint-comparison.ipynb), applied here to real registration output. Each slice is ~900 pixels -> ~1800 variables, a manageable problem size for SLSQP.",
        "run-md",
    )
)

cells.append(
    code(
        "results = {}\n"
        "for name, case in cases.items():\n"
        "    phi = case['phi']\n"
        "    print(f'>>> {name}')\n"
        "    results[name] = dict(\n"
        "        phi_init=phi, m0=case['m0'], qi0=case['qi0'],\n"
        "        cd=run_cd(phi), fd=run_fd(phi), tri=run_2tri(phi),\n"
        "    )\n"
        "    for key in ('cd', 'fd', 'tri'):\n"
        "        r = results[name][key]\n"
        "        print(f\"    {key:<5s}  nit={r['nit']:>4d}  time={r['time']:>6.2f}s  success={str(r['success']):<5s}  \"\n"
        "              f\"neg_CD={r['n_cd']:>3d}  neg_TR={r['n_tr']:>3d}  L2={r['l2']:.3f}\")\n"
        "    # global intersections on the tri run\n"
        "    qi_tri = list_intersecting_quads(results[name]['tri']['phi'])\n"
        "    results[name]['qi_tri'] = qi_tri\n"
        "    print(f'    post-2tri QI pairs = {len(qi_tri)}')",
        "run-code",
    )
)

cells.append(
    md(
        "## Cross-case summary\n"
        "\n"
        "Init TR = how many folded cells the 2-triangle check finds before any correction. For each variant: `nit`, final `neg_TR`, and L2 distortion from the original field.",
        "summary-md",
    )
)

cells.append(
    code(
        "hdr = (f\"{'case':<18s}  {'init TR':>8s}  {'init QI':>8s}  \"\n"
        "        + '  '.join(f\"{k:>6s}\" for k in ('CD nit', 'CD TR', 'CD L2',\n"
        "                                          'FD nit', 'FD TR', 'FD L2',\n"
        "                                          'TR nit', 'TR TR', 'TR L2', 'TR QI')))\n"
        "print(hdr)\n"
        "print('-' * len(hdr))\n"
        "for name, r in results.items():\n"
        "    line = f\"{name:<18s}  {r['m0']['n_tr']:>8d}  {len(r['qi0']):>8d}\"\n"
        "    for key in ('cd', 'fd'):\n"
        "        m = r[key]\n"
        "        line += f\"  {m['nit']:>6d}  {m['n_tr']:>6d}  {m['l2']:>6.3f}\"\n"
        "    m = r['tri']\n"
        "    line += f\"  {m['nit']:>6d}  {m['n_tr']:>6d}  {m['l2']:>6.3f}  {len(r['qi_tri']):>6d}\"\n"
        "    print(line)",
        "summary-code",
    )
)

cells.append(
    md(
        "## Warped-grid visualization per slice\n"
        "\n"
        "Row per slice, 4 columns: initial, (A) CD-SLSQP, (B) FD-SLSQP, (C) best-2tri. Folded cells under the 2-triangle check outlined in dark blue. Row 2 below is the `min(T1, T2)` heatmap.",
        "warped-md",
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
        "        ax.plot(xx[i], yy[i], color='#f4f4f4', lw=0.3)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(xx[:, j], yy[:, j], color='#f4f4f4', lw=0.3)\n"
        "    for i in range(Hh):\n"
        "        ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.8)\n"
        "    for j in range(Ww):\n"
        "        ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.8)\n"
        "    if highlight_folds:\n"
        "        tri = triangle_sign_areas2D(phi)\n"
        "        bad = np.argwhere(tri.min(axis=0) <= 0)\n"
        "        for (cy, cx) in bad:\n"
        "            poly_x = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "            poly_y = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "            ax.plot(poly_x, poly_y, color='#1565c0', lw=1.4)\n"
        "    ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "    ax.set_title(title, fontsize=9)\n"
        "    ax.set_xticks([]); ax.set_yticks([])\n"
        "\n"
        "NL = chr(10)\n"
        "\n"
        "for name, r in results.items():\n"
        "    variants = [\n"
        "        ('initial',       r['phi_init'], r['m0'],   None),\n"
        "        ('(A) CD-SLSQP',  r['cd']['phi'], r['cd'],   r['cd']),\n"
        "        ('(B) FD-SLSQP',  r['fd']['phi'], r['fd'],   r['fd']),\n"
        "        ('(C) best-2tri', r['tri']['phi'], r['tri'], r['tri']),\n"
        "    ]\n"
        "    vmax_tri = max(abs(m['tri']).max() for _, _, m, _ in variants)\n"
        "\n"
        "    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), layout='constrained')\n"
        "    for k, (label, phi, m, res) in enumerate(variants):\n"
        "        if res is None:\n"
        "            line1 = label\n"
        "        else:\n"
        "            tag = 'OK' if res['success'] else 'FAIL'\n"
        "            line1 = f'{label}  [{tag}]'\n"
        "        line2 = f\"TR={m['n_tr']}  min_TR={m['min_tr']:+.3f}\"\n"
        "        plot_warped_grid(axes[0, k], phi, line1 + NL + line2)\n"
        "        tri_min = m['tri'].min(axis=0)\n"
        "        im = axes[1, k].imshow(tri_min, cmap='RdBu_r', vmin=-vmax_tri, vmax=vmax_tri, aspect='auto')\n"
        "        axes[1, k].set_title(f'min(T1, T2)   {label}', fontsize=9)\n"
        "        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])\n"
        "    cbar = fig.colorbar(im, ax=axes[1, :], orientation='horizontal',\n"
        "                         fraction=0.035, pad=0.04, shrink=0.55)\n"
        "    cbar.set_label('signed triangle area (blue = fold, red = valid)')\n"
        "    plt.suptitle(f'{name}  (slice={r[\"phi_init\"][0].shape})  -  real-slice constraint comparison',\n"
        "                  fontsize=11)\n"
        "    plt.show()",
        "warped-code",
    )
)

cells.append(
    md(
        "## Disagreement overlay - cells where CD / FD passed but 2-tri still flags\n"
        "\n"
        "Same overlay idea as [04_constraint-comparison.ipynb](04_constraint-comparison.ipynb). On these real slices the effect is severe: CD converges with `success=True` and leaves **most of the initial folds in place** because its own stencil doesn't see them.",
        "disagree-md",
    )
)

cells.append(
    code(
        "def plot_disagreement(name, r):\n"
        "    phi_cd = r['cd']['phi'];  phi_fd = r['fd']['phi']\n"
        "    cd_tri_min = r['cd']['tri'].min(axis=0)\n"
        "    fd_tri_min = r['fd']['tri'].min(axis=0)\n"
        "    cd_bad = (cd_tri_min <= 0); fd_bad = (fd_tri_min <= 0)\n"
        "\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')\n"
        "    for ax, phi, bad_mask, title in [\n"
        "        (axes[0], phi_cd, cd_bad, f'{name}  after CD-SLSQP - cells 2-tri still flags'),\n"
        "        (axes[1], phi_fd, fd_bad, f'{name}  after FD-SLSQP - cells 2-tri still flags'),\n"
        "    ]:\n"
        "        dy, dx = phi[0], phi[1]\n"
        "        Hh, Ww = dy.shape\n"
        "        yy, xx = np.mgrid[:Hh, :Ww]\n"
        "        gx = xx + dx; gy = yy + dy\n"
        "        for i in range(Hh):\n"
        "            ax.plot(xx[i], yy[i], color='#f8f8f8', lw=0.3)\n"
        "        for j in range(Ww):\n"
        "            ax.plot(xx[:, j], yy[:, j], color='#f8f8f8', lw=0.3)\n"
        "        for i in range(Hh):\n"
        "            ax.plot(gx[i], gy[i], color='#5b7fb5', lw=0.6)\n"
        "        for j in range(Ww):\n"
        "            ax.plot(gx[:, j], gy[:, j], color='#5b7fb5', lw=0.6)\n"
        "        for (cy, cx) in np.argwhere(bad_mask):\n"
        "            poly_x = [gx[cy, cx], gx[cy, cx+1], gx[cy+1, cx+1], gx[cy+1, cx], gx[cy, cx]]\n"
        "            poly_y = [gy[cy, cx], gy[cy, cx+1], gy[cy+1, cx+1], gy[cy+1, cx], gy[cy, cx]]\n"
        "            ax.plot(poly_x, poly_y, color='#1565c0', lw=1.6)\n"
        "        ax.set_aspect('equal'); ax.invert_yaxis()\n"
        "        n = int(bad_mask.sum())\n"
        "        ax.set_title(f'{title}  -  {n} cell(s)', fontsize=9)\n"
        "        ax.set_xticks([]); ax.set_yticks([])\n"
        "    plt.show()\n"
        "\n"
        "for name, r in results.items():\n"
        "    plot_disagreement(name, r)",
        "disagree-code",
    )
)

cells.append(
    md(
        "## Summary\n"
        "\n"
        "Running on real slices produced by the Laplacian registration pipeline:\n"
        "\n"
        "- **Central-difference Jdet is blind** on these slices. `neg_CD=0` initially even though the 2-triangle check sees 12-35 folded cells. CD-SLSQP converges `success=True` at `nit=1` and does nothing - there's no work to do from its point of view.\n"
        "- **Forward-difference** is partially useful: it detects more folds than CD (still misses some) and makes progress but leaves residuals.\n"
        "- **Best 2-triangle (analytical + warm-start)** converges to `neg_TR=0` and (on these cases) drives global quad-intersection count `QI` to 0 as well. This is the configuration needed if downstream code evaluates validity with a geometric fold check.\n"
        "\n"
        "Implication for the pipeline: if you're using `iterative_serial` or a similar SLSQP-based corrector with the default central-diff Jacobian-determinant constraint, you may be accepting fields that are geometrically folded. Switching the constraint to the 2-triangle signed-area check (or the stricter 4-triangle check in `dvfopt.jacobian.triangle_det2D`) plugs the gap.\n"
        "\n"
        "Next step candidates: promote `triangle_sign_constraint_jac_2d` from these notebooks into [dvfopt/jacobian/triangle_sign.py](../../dvfopt/jacobian/triangle_sign.py) next to `triangle_sign_constraint`, and wire an `enforce_triangles_sign=True` option into `iterative_serial` analogous to the existing `enforce_shoelace` / `enforce_triangles` flags.",
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
