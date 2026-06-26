"""Split triangle-sign-demos.ipynb into demos (Parts 1-3) + solver-engineering (Parts 4-5)."""

import json
import os

src_path = 'notebooks/two-triangle-check/triangle-sign-demos.ipynb'
dst_path = 'notebooks/two-triangle-check/triangle-sign-solver-engineering.ipynb'

nb = json.load(open(src_path, encoding='utf-8'))

part4_5_ids = {
    '0dab09b8',  # Part 4 markdown
    '40e209be',  # Part 4 code (SLSQP exit status)
    'd0d7f2fb',  # Part 4 code (heatmap + warped grid)
    'd80ada1c',  # Part 4 code (maxiter/ftol retry)
    '1eb0d3a2',  # Part 4 code (warm-start)
    'ffea9260',  # Part 5 markdown (derivation)
    '99586d1d',  # Part 5 code (implementation)
    'd737a896',  # Part 5 markdown (validation)
    '9766d2fc',  # Part 5 code (validation)
    '873815db',  # Part 5 markdown (head-to-head)
    '1238931a',  # Part 5 code (head-to-head)
    'anal-warm-md',  # combined-fix markdown
    'anal-warm-code',  # combined-fix code
}

part4_5_cells = [c for c in nb['cells'] if c.get('id') in part4_5_ids]


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


intro = md(
    "# 2-Triangle SLSQP — Solver Engineering\n"
    "\n"
    "Companion to [triangle-sign-demos.ipynb](triangle-sign-demos.ipynb). That notebook showed that 2-triangle SLSQP stalls on the `01c_20x40_edges` case with `success=False`, status 8 (*\"Positive directional derivative for linesearch\"*) at `nit=42`, leaving 14 local folds and 109 global quad intersections. This notebook is a focused deep-dive on **why** it stalls and **how** to fix it.\n"
    "\n"
    "Two independent fixes are tested, then combined:\n"
    "\n"
    "- **Part 4**: diagnose the exit status and try perturbation warm-start.\n"
    "- **Part 5**: derive, validate, and test the analytical constraint Jacobian — eliminates finite-difference noise in the QP sub-problem.\n"
    "\n"
    "The combined fix (analytical Jacobian + warm-start perturbation) fully clears the case.",
    "se-intro",
)

imports_src = (
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
    "from dvfopt.jacobian.triangle_sign import _triangle_areas_2d\n"
    "from dvfopt.jacobian.intersection import has_quad_self_intersections, _quads_intersect\n"
    "from dvfopt.core.objective import objective_euc\n"
    "\n"
    "from test_cases import make_deformation\n"
    "\n"
    "THRESHOLD = DEFAULT_PARAMS['threshold']  # 0.01\n"
    "print(f'threshold = {THRESHOLD}')"
)
imports = code(imports_src, "se-imports")

helpers_src = (
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
    "        min_cd=float(jd.min()), min_fd=float(fd.min()), min_tr=float(tri.min()),\n"
    "    )\n"
    "\n"
    "\n"
    "def _triangle_flat(dy_, dx_):\n"
    "    T1, T2 = _triangle_areas_2d(dy_, dx_)\n"
    "    return np.concatenate([T1.flatten(), T2.flatten()])\n"
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
    "    aabb_ymin = np.minimum(np.minimum(y_tl, y_tr), np.minimum(y_bl, y_br))\n"
    "    aabb_ymax = np.maximum(np.maximum(y_tl, y_tr), np.maximum(y_bl, y_br))\n"
    "    aabb_xmin = np.minimum(np.minimum(x_tl, x_tr), np.minimum(x_bl, x_br))\n"
    "    aabb_xmax = np.maximum(np.maximum(x_tl, x_tr), np.maximum(x_bl, x_br))\n"
    "    corners = np.stack([\n"
    "        np.stack([y_tl.ravel(), x_tl.ravel()], axis=1),\n"
    "        np.stack([y_tr.ravel(), x_tr.ravel()], axis=1),\n"
    "        np.stack([y_br.ravel(), x_br.ravel()], axis=1),\n"
    "        np.stack([y_bl.ravel(), x_bl.ravel()], axis=1),\n"
    "    ], axis=1)\n"
    "    ymin_flat = aabb_ymin.ravel(); ymax_flat = aabb_ymax.ravel()\n"
    "    xmin_flat = aabb_xmin.ravel(); xmax_flat = aabb_xmax.ravel()\n"
    "    n_quads = nr * nc\n"
    "    pairs = []\n"
    "    for i in range(n_quads):\n"
    "        ri, ci = divmod(i, nc)\n"
    "        for j in range(i + 1, n_quads):\n"
    "            rj, cj = divmod(j, nc)\n"
    "            if abs(ri - rj) <= 1 and abs(ci - cj) <= 1: continue\n"
    "            if (ymin_flat[i] > ymax_flat[j] or ymax_flat[i] < ymin_flat[j]\n"
    "                    or xmin_flat[i] > xmax_flat[j] or xmax_flat[i] < xmin_flat[j]): continue\n"
    "            if _quads_intersect(corners[i], corners[j]):\n"
    "                pairs.append(((ri, ci), (rj, cj)))\n"
    "    return pairs"
)
helpers = code(helpers_src, "se-helpers")

setup_md_src = (
    "## Setup — reproduce the stuck `01c_20x40_edges` iterate\n"
    "\n"
    "Load the field, run the baseline SLSQP (finite-difference Jacobian, default `maxiter=500`). This matches the baseline from `triangle-sign-demos.ipynb` Part 2 so the rest of this notebook has the same starting point."
)
setup_md_cell = md(setup_md_src, "se-setup-md")

setup_code_src = (
    "CASE_KEY = '01c_20x40_edges'\n"
    "deformation, *_ = make_deformation(CASE_KEY)\n"
    "phi_init = np.stack([deformation[1, 0], deformation[2, 0]])\n"
    "m0 = measure(phi_init)\n"
    "\n"
    "_, H_, W_ = phi_init.shape\n"
    "pixels = H_ * W_\n"
    "\n"
    "def unpack(z):\n"
    "    return z[pixels:].reshape(H_, W_), z[:pixels].reshape(H_, W_)\n"
    "\n"
    "def constr(z):\n"
    "    dy_, dx_ = unpack(z)\n"
    "    return _triangle_flat(dy_, dx_)\n"
    "\n"
    "z0_init = np.concatenate([phi_init[1].flatten(), phi_init[0].flatten()])\n"
    "\n"
    "t0 = time.time()\n"
    "res_base = minimize(\n"
    "    lambda z: objective_euc(z, z0_init),\n"
    "    z0_init.copy(),\n"
    "    jac=True,\n"
    "    method='SLSQP',\n"
    "    constraints=[NonlinearConstraint(constr, THRESHOLD, np.inf)],\n"
    "    options={'maxiter': 500, 'disp': False},\n"
    ")\n"
    "t_base = time.time() - t0\n"
    "dy_b, dx_b = unpack(res_base.x)\n"
    "phi_out = np.stack([dy_b, dx_b])\n"
    "m1 = measure(phi_out)\n"
    "qi_init = list_intersecting_quads(phi_init)\n"
    "qi_out = list_intersecting_quads(phi_out)\n"
    "\n"
    "runs = {\n"
    "    CASE_KEY: dict(\n"
    "        phi_init=phi_init, phi_out=phi_out,\n"
    "        m0=m0, m1=m1, qi_init=qi_init, qi_out=qi_out,\n"
    "        info=dict(nit=res_base.nit, time=t_base, success=bool(res_base.success),\n"
    "                  message=str(res_base.message), status=int(res_base.status)),\n"
    "        l2=float(np.linalg.norm(phi_out - phi_init)),\n"
    "    )\n"
    "}\n"
    "\n"
    "r = runs[CASE_KEY]; info = r['info']\n"
    "print(f'shape             : {phi_init[0].shape}   pixels: {pixels}')\n"
    "print(f'initial neg_TR    : {m0[\"n_tr\"]}    initial QI pairs: {len(qi_init)}')\n"
    "print(f'SLSQP nit         : {info[\"nit\"]}    time: {info[\"time\"]:.2f}s    status: {info[\"status\"]}')\n"
    "print(f'SLSQP success     : {info[\"success\"]}   message: {info[\"message\"]}')\n"
    "print(f'post neg_TR       : {m1[\"n_tr\"]}    post QI pairs: {len(qi_out)}')\n"
    "print(f'post min_TR       : {m1[\"min_tr\"]:+.5f}   L2 from phi_init: {r[\"l2\"]:.3f}')"
)
setup_code_cell = code(setup_code_src, "se-setup-code")

se_summary_src = (
    "## Summary\n"
    "\n"
    "- **Part 4 diagnosis**: SLSQP stalls at `nit=42` with status 8 (*\"Positive directional derivative for linesearch\"*). Raising `maxiter` and tightening `ftol` does nothing — not a budget issue. The QP sub-problem's line search can't find a descent direction.\n"
    "- **Part 4 fix**: warm-start from the stuck iterate with σ=0.01 Gaussian perturbation. Breaks the degenerate direction — SLSQP re-converges to `neg_TR=0`, `QI=0`, `success=True`.\n"
    "- **Part 5 derivation**: the 2-triangle signed area is a polynomial so `∂T/∂z` is closed-form. Six nonzeros per constraint row. Validated to FD step-floor precision (rel err ~3e-7).\n"
    "- **Part 5 head-to-head** on `01c_20x40_edges`:\n"
    "\n"
    "  | variant | nit | time | neg_TR | QI | success |\n"
    "  |---|---:|---:|---:|---:|:---:|\n"
    "  | finite-diff Jac (baseline) | 42 | ~104s | 14 | 109 | False |\n"
    "  | analytical Jac | 29 | ~78s | 9 | 64 | False |\n"
    "  | **analytical Jac + warm-start** | 80 | ~169s | **0** | **0** | **True** |\n"
    "\n"
    "- **Conclusions:**\n"
    "  1. Analytical Jacobian alone is strictly better but not sufficient here — it reduces the failure (more progress per iter, faster) but SLSQP still hits the line-search degeneracy.\n"
    "  2. Perturbation warm-start alone is what unsticks the solver; analytical Jacobian composes with it.\n"
    "  3. For a robust SLSQP corrector, **use both**: analytical Jacobian from the start, with a status-8 catch that re-runs with `phi_out + noise`.\n"
    "\n"
    "- **Next step**: promote `triangle_sign_constraint_jac_2d` from this notebook into [dvfopt/jacobian/triangle_sign.py](../../dvfopt/jacobian/triangle_sign.py) alongside `triangle_sign_constraint`, matching the pattern used by `shoelace_constraint_jacobian_2d`."
)
se_summary_cell = md(se_summary_src, "se-summary")

new_cells = (
    [intro, imports, helpers, setup_md_cell, setup_code_cell] + part4_5_cells + [se_summary_cell]
)

new_nb = {
    "cells": new_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

json.dump(new_nb, open(dst_path, 'w', encoding='utf-8'), indent=1)
print(f'created {dst_path} with {len(new_cells)} cells')

# Trim demos notebook: remove Parts 4-5, rewrite the summary.
demos_new_cells = [c for c in nb['cells'] if c.get('id') not in part4_5_ids]

new_demos_summary = (
    "## Summary\n"
    "\n"
    "- **Part 1** uses `plot_triangle_debug` on identity, translation, scale, rotation, and small random deformations. In every case both T1 and T2 are positive and fill red — useful sanity check for the formula and the red-for-valid color convention.\n"
    "- **Part 2** runs 2-triangle SLSQP on six synthetic folds from the `test_cases` library:\n"
    "  - **CD** (central-diff, per-pixel), **FD** (forward-diff, per-cell), **TR** (2-triangle, per-triangle), **QI** (global quad self-intersections).\n"
    "  - The **initial** ordering `neg_CD ≤ neg_FD ≤ neg_TR` reflects increasing locality.\n"
    "  - After correction, **FD = TR = 0 on every converged case**; **CD can remain > 0** — those residuals are central-diff stencil artifacts, not geometric folds.\n"
    "- **Part 3** compares local vs. global invertibility:\n"
    "\n"
    "  | case | init QI pairs | out QI pairs | local TR out | `success` |\n"
    "  |---|---:|---:|---:|:---:|\n"
    "  | `01a_10x10_crossing` | 75 | **0** | 0 | True |\n"
    "  | `01b_10x10_opposite` | 10 | **0** | 0 | True |\n"
    "  | `03b_10x10_crossing` | 34 | **0** | 0 | True |\n"
    "  | `03c_20x20_opposite` | 57 | **20** | 0 | **True** |\n"
    "  | `03d_20x20_crossing` | 168 | **0** | 0 | True |\n"
    "  | `01c_20x40_edges` | 334 | **109** | 14 | **False** |\n"
    "\n"
    "- Two distinct failure modes:\n"
    "  1. **`03c_20x20_opposite` — genuine local-vs-global gap.** SLSQP fully converged (`success=True`, TR=0) yet 20 non-adjacent quad pairs still intersect globally. Fundamental limitation of the local constraint formulation.\n"
    "  2. **`01c_20x40_edges` — SLSQP solver failure.** `success=False`, status 8 (line-search degeneracy). Diagnosed and resolved in the companion [triangle-sign-solver-engineering.ipynb](triangle-sign-solver-engineering.ipynb).\n"
    "\n"
    "- **Next-step options for `03c`-type local-vs-global gaps:**\n"
    "  1. Outer-loop penalty on `list_intersecting_quads(phi_out)` pairs; re-solve.\n"
    "  2. Displacement damping: scale `φ_out` toward `φ_init` until `list_intersecting_quads == []`.\n"
    "  3. Log-barrier on quad-pair signed distance in the L-BFGS barrier solver.\n"
    "  4. Strict 4-triangle local check — still local, but closes the asymmetric-split loophole.\n"
    "\n"
    "  Option 1 is the most surgical; `list_intersecting_quads(phi_out)` already returns the active pair set."
)

for c in demos_new_cells:
    if c.get('id') == 'summary':
        c['source'] = new_demos_summary.splitlines(keepends=True)
        break

nb['cells'] = demos_new_cells
json.dump(nb, open(src_path, 'w', encoding='utf-8'), indent=1)
print(f'trimmed {src_path}: {len(demos_new_cells)} cells remain')
