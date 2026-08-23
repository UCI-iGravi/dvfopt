"""Add the triangulated-shoelace constraint as a new mode to the SLSQP
constraint-mode benchmark notebook.

Touches four cells:
* ``a1`` intro markdown — new row in the mode table.
* ``a2`` imports — pull ``triangulated_shoelace_det2D`` from ``dvfopt.jacobian``.
* ``a4`` ``MODES`` dict — add ``"Jac + Triangulated"``.
* ``a6`` ``_injectivity_stats`` + per-mode print — add ``n_neg_tri`` tracking.
* ``a11`` summary table — print a ``(Tri viol)`` row.
"""

import json
from pathlib import Path

NB = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "solvers"
    / "slsqp"
    / "benchmark-constraint-modes.ipynb"
)


A1 = """# Benchmark: Constraint Mode Comparison

Compares the five constraint configurations on the same deformation fields:

| Mode | Flags |
|------|-------|
| **Jacobian only** | default |
| **Jacobian + Shoelace** | `enforce_shoelace=True` |
| **Jacobian + Triangulated** | `enforce_shoelace_triangulated=True` (strictly stronger than shoelace) |
| **Jacobian + Injectivity** | `enforce_injectivity=True` |
| **All constraints** | all three flags `True` |

Metrics: runtime, L2 error, final min Jdet, SLSQP iterations (outer), and
whether all negative Jacobians were eliminated."""


A2 = """import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

import time
import numpy as np
import matplotlib.pyplot as plt

from dvfopt import (
    iterative_serial,
    iterative_parallel,
    jacobian_det2D,
    generate_random_dvf,
    scale_dvf,
    shoelace_det2D,
)
from dvfopt.jacobian import (
    _monotonicity_diffs_2d,
    _diagonal_monotonicity_diffs_2d,
    triangulated_shoelace_det2D,
)
from dvfopt.jacobian.intersection import has_quad_self_intersections
from dvfopt.testdata import SYNTHETIC_CASES, make_deformation, make_random_dvf
from dvfopt.viz import plot_deformations, plot_grid_before_after
from benchmark_utils import (
    get_output_dir, save_figure, save_results_csv, save_summary_json, log_run_header, log_run_footer, results_to_rows, show_and_save, reset_figure_counter,
)"""


A4 = """MODES = {
    "Jacobian only":       {"enforce_shoelace": False, "enforce_injectivity": False},
    "Jac + Shoelace":      {"enforce_shoelace": True,  "enforce_injectivity": False},
    "Jac + Triangulated":  {"enforce_shoelace_triangulated": True},
    "Jac + Injectivity":   {"enforce_shoelace": False, "enforce_injectivity": True,
                            "injectivity_threshold": INJECTIVITY_THRESHOLD},
    "All constraints":     {"enforce_shoelace": True,  "enforce_injectivity": True,
                            "enforce_shoelace_triangulated": True,
                            "injectivity_threshold": INJECTIVITY_THRESHOLD},
}
"""


A6 = """from matplotlib.colors import TwoSlopeNorm


def _injectivity_stats(phi):
    \"\"\"Compute shoelace, triangulated-shoelace, h/v/d1/d2 monotonicity
    violation counts and global intersection.\"\"\"
    shoe = np.squeeze(shoelace_det2D(phi))
    T1, T2 = triangulated_shoelace_det2D(phi)
    tri_min = np.minimum(np.squeeze(T1), np.squeeze(T2))
    h_m, v_m = _monotonicity_diffs_2d(phi[0], phi[1])
    d1, d2   = _diagonal_monotonicity_diffs_2d(phi[0], phi[1])
    # Global check: any non-adjacent quad cells intersect geometrically?
    # O(n^2) \u2014 slow on large grids but definitive.
    global_intersect = has_quad_self_intersections(phi)
    return dict(
        n_neg_shoe       = int((shoe[1:-1, 1:-1] <= 0).sum()),
        n_neg_tri        = int((tri_min[1:-1, 1:-1] <= 0).sum()),
        n_h_viol         = int((h_m[1:-1, 1:-1] <= 0).sum()),
        n_v_viol         = int((v_m[1:-1, 1:-1] <= 0).sum()),
        n_d1_viol        = int((d1[1:-1, 1:-1] <= 0).sum()),
        n_d2_viol        = int((d2[1:-1, 1:-1] <= 0).sum()),
        global_intersect = global_intersect,
    )


def _plot_jdet_before_after(jac_before, jac_after, title, jdet_vmax=None):
    \"\"\"Side-by-side Jdet heatmaps sharing a diverging colour scale centred at 0.\"\"\"
    jb = np.squeeze(jac_before)
    ja = np.squeeze(jac_after)
    j_all = np.concatenate([jb.ravel(), ja.ravel()])
    vmin = min(float(j_all.min()), -0.5)
    vmax = max(float(j_all.max()),  1.5)
    if jdet_vmax is not None:
        vmin = max(vmin, -jdet_vmax)
        vmax = min(vmax,  jdet_vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, j, sub in [(axes[0], jb, "Initial"), (axes[1], ja, "Corrected")]:
        im = ax.imshow(j, cmap="bwr", norm=norm, interpolation="nearest")
        neg = int((j <= 0).sum())
        ax.set_title(f"{sub} \u2014 neg Jdet = {neg}, min = {float(j.min()):+.4f}",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, label="Jacobian determinant", shrink=0.85)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.show()


def run_case(deformation_i, label, n_runs=1):
    \"\"\"Run all constraint modes on a single deformation field.\"\"\"
    phi_init = np.stack([deformation_i[-2, 0], deformation_i[-1, 0]])
    jac_init = jacobian_det2D(phi_init)
    n_neg_init = int((jac_init <= 0).sum())
    H, W = deformation_i.shape[-2:]

    print(f\"\\n{'='*80}\")
    print(f\"  {label}  |  {H}x{W}  |  Initial neg-Jdet: {n_neg_init}\")
    print(f\"{'='*80}\")

    results = {}
    for mode_name, flags in MODES.items():
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            phi = SOLVER(
                deformation_i.copy(), verbose=0, **flags
            )
            times.append(time.perf_counter() - t0)

        jac_final = jacobian_det2D(phi)
        final_neg = int((jac_final <= 0).sum())
        final_min = float(jac_final.min())
        l2_err = float(np.sqrt(np.sum((phi - phi_init) ** 2)))
        avg_t = np.mean(times)
        inj = _injectivity_stats(phi)

        results[mode_name] = {
            \"time\": avg_t,
            \"final_neg\": final_neg,
            \"final_min\": final_min,
            \"l2_err\": l2_err,
            \"phi\": phi,
            **inj,
        }

        jac_ok  = \"OK\" if final_neg == 0                  else f\"FAIL({final_neg})\"
        shoe_ok = \"OK\" if inj[\"n_neg_shoe\"] == 0          else f\"FAIL({inj['n_neg_shoe']})\"
        tri_ok  = \"OK\" if inj[\"n_neg_tri\"] == 0           else f\"FAIL({inj['n_neg_tri']})\"
        h_ok    = \"OK\" if inj[\"n_h_viol\"] == 0            else f\"FAIL({inj['n_h_viol']})\"
        v_ok    = \"OK\" if inj[\"n_v_viol\"] == 0            else f\"FAIL({inj['n_v_viol']})\"
        d1_ok   = \"OK\" if inj[\"n_d1_viol\"] == 0           else f\"FAIL({inj['n_d1_viol']})\"
        d2_ok   = \"OK\" if inj[\"n_d2_viol\"] == 0           else f\"FAIL({inj['n_d2_viol']})\"
        g_ok    = \"OK\" if not inj[\"global_intersect\"]     else \"INTERSECT\"
        print(f\"  {mode_name:<22s}  {avg_t:8.2f}s  \"
              f\"neg={final_neg:3d}  min_jdet={final_min:+.6f}  L2={l2_err:.4f}  \"
              f\"[Jac:{jac_ok}  Shoe:{shoe_ok}  Tri:{tri_ok}  h:{h_ok}  v:{v_ok}  \"
              f\"d1:{d1_ok}  d2:{d2_ok}  glob:{g_ok}]\")

        # Grid visualisation immediately after each mode
        plot_grid_before_after(deformation_i, phi, title=f\"{label} \u2014 {mode_name}\",
                               jdet_vmax=JDET_VMAX)

        # Jdet heatmaps (before vs after) alongside the grid plot
        _plot_jdet_before_after(jac_init, jac_final,
                                title=f\"{label} \u2014 {mode_name}: Jdet\",
                                jdet_vmax=JDET_VMAX)

    return results"""


A11 = """mode_names = list(MODES.keys())

print(f\"{'Test Case':<28s}\", end=\"\")
for m in mode_names:
    print(f\"  {m:>20s}\", end=\"\")
print()
print(\"-\" * (28 + 22 * len(mode_names)))

for label, results in all_results.items():
    # Time row
    print(f\"{label:<28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        print(f\"  {r['time']:>17.2f}s  \", end=\"\")
    print(\"  (time)\")

    # L2 row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        print(f\"  {r['l2_err']:>18.4f} \", end=\"\")
    print(\"  (L2)\")

    # Min Jdet row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        print(f\"  {r['final_min']:>+18.6f}\", end=\"\")
    print(\"  (min Jdet)\")

    # Shoelace row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        v = r['n_neg_shoe']
        s = 'OK' if v == 0 else str(v)
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (Shoelace viol)\")

    # Triangulated row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        v = r['n_neg_tri']
        s = 'OK' if v == 0 else str(v)
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (Tri viol)\")

    # h-mono row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        v = r['n_h_viol']
        s = 'OK' if v == 0 else str(v)
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (h-mono viol)\")

    # v-mono row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        v = r['n_v_viol']
        s = 'OK' if v == 0 else str(v)
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (v-mono viol)\")

    # d1-mono row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        v = r['n_d1_viol']
        s = 'OK' if v == 0 else str(v)
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (d1-mono viol)\")

    # d2-mono row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        v = r['n_d2_viol']
        s = 'OK' if v == 0 else str(v)
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (d2-mono viol)\")

    # Global intersection row
    print(f\"{'':28s}\", end=\"\")
    for m in mode_names:
        r = results[m]
        s = 'INTERSECT' if r['global_intersect'] else 'OK'
        print(f\"  {s:>18s} \", end=\"\")
    print(\"  (Glob intersect)\")

    print()"""


CELL_SOURCES = {"a1": A1, "a2": A2, "a4": A4, "a6": A6, "a11": A11}


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    touched = set()
    for c in nb["cells"]:
        cid = c.get("id")
        if cid in CELL_SOURCES:
            c["source"] = CELL_SOURCES[cid].splitlines(keepends=True)
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
            touched.add(cid)
    missing = set(CELL_SOURCES) - touched
    if missing:
        raise SystemExit(f"cells not found: {missing}")
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Patched {NB} ({len(touched)} cells)")


if __name__ == "__main__":
    main()
