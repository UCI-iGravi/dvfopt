"""Add a multi-constraint-mode comparison to the post-correction section.

After the existing jdet-only summary table, inserts:
  A) MODES dict + driver that runs Jac+Shoelace, Jac+Injectivity, and
     All-constraints on every case that had negative pixel Jdet.
  B) Aggregate summary table: for each mode, report total pixel-neg /
     cell-neg / sub-pixel counts before vs after.
"""
import io
import json
import sys
from pathlib import Path

NB = Path("benchmarks/scaling/benchmark-neighborhood-injectivity.ipynb")


EXTRA_MODES_MD = r"""## Constraint-mode comparison

The default-mode correction above enforces only the pixel Jacobian constraint. The 2D solver also accepts `enforce_shoelace=True` (positive quad area) and `enforce_injectivity=True` (coordinate monotonicity + diagonal convexity) — these are *geometric* conditions that act on quads rather than pixel samples, so in principle they should close the sub-pixel gap that pixel-Jdet alone cannot. Below we re-run the corrector in all four modes and compare the three diagnostic counts in aggregate.
"""


EXTRA_DRIVER_CODE = r"""INJECTIVITY_THRESHOLD = 0.3

MODES = {
    "Jac only":        dict(enforce_shoelace=False, enforce_injectivity=False),
    "Jac + Shoelace":  dict(enforce_shoelace=True,  enforce_injectivity=False),
    "Jac + Injectivity": dict(enforce_shoelace=False, enforce_injectivity=True,
                              injectivity_threshold=INJECTIVITY_THRESHOLD),
    "All constraints": dict(enforce_shoelace=True,  enforce_injectivity=True,
                            injectivity_threshold=INJECTIVITY_THRESHOLD),
}


def _run_mode(dvf, **kwargs):
    phi_corr = np.asarray(iterative_serial(dvf.copy(), verbose=0, **kwargs))
    if phi_corr.ndim == 4:
        phi_corr = np.stack([phi_corr[1, 0], phi_corr[2, 0]])
    jac_c  = np.squeeze(jacobian_det2D(phi_corr))
    ift_c  = ift_radius_2d(phi_corr)
    cell_c = cell_min_jdet_2d(phi_corr)
    return {
        "n_pix_neg":   int((jac_c <= 0).sum()),
        "n_cell_neg":  int((cell_c <= 0).sum()),
        "n_sub_pixel": int((ift_c < 1.0).sum()),
    }


mode_stats = {name: {"pix": 0, "cell": 0, "sub": 0, "cases_pix_clean": 0,
                     "cases_cell_clean": 0, "cases": 0}
              for name in MODES}
mode_stats["initial"] = {"pix": 0, "cell": 0, "sub": 0}

for rec in all_records:
    if int((rec["jac"] <= 0).sum()) == 0:
        continue
    label = rec["label"]
    mode_stats["initial"]["pix"]  += int((rec["jac"] <= 0).sum())
    mode_stats["initial"]["cell"] += int((rec["cell_min"] <= 0).sum())
    mode_stats["initial"]["sub"]  += int((rec["ift_r"] < 1.0).sum())
    print(f"\n{label}")
    for mode_name, kwargs in MODES.items():
        try:
            r = _run_mode(rec["dvf"], **kwargs)
        except Exception as e:
            print(f"  {mode_name:<20s}  FAILED: {type(e).__name__}: {e}")
            continue
        mode_stats[mode_name]["pix"]  += r["n_pix_neg"]
        mode_stats[mode_name]["cell"] += r["n_cell_neg"]
        mode_stats[mode_name]["sub"]  += r["n_sub_pixel"]
        mode_stats[mode_name]["cases"] += 1
        if r["n_pix_neg"] == 0:
            mode_stats[mode_name]["cases_pix_clean"] += 1
        if r["n_cell_neg"] == 0:
            mode_stats[mode_name]["cases_cell_clean"] += 1
        print(f"  {mode_name:<20s}  pix={r['n_pix_neg']:>4d}  "
              f"cell={r['n_cell_neg']:>4d}  sub-pixel={r['n_sub_pixel']:>4d}")
"""


EXTRA_SUMMARY_CODE = r"""init = mode_stats["initial"]
total_pix_area  = sum(rec["jac"].size     for rec in all_records if (rec["jac"] <= 0).any())
total_cell_area = sum(rec["cell_min"].size for rec in all_records if (rec["jac"] <= 0).any())

print(f"Aggregate over {sum(1 for r in all_records if (r['jac'] <= 0).any())} corrected cases "
      f"(pixel area {total_pix_area}, cell area {total_cell_area})\n")

print(f"{'Mode':<20s} | {'pixel neg':<12s} | {'cell neg':<12s} | {'sub-pixel (r<1)':<16s} | {'cases pix\u2713':<10s} | {'cases cell\u2713':<11s}")
print("-" * 100)
print(f"{'(initial)':<20s} | {init['pix']:>12d} | {init['cell']:>12d} | {init['sub']:>16d} | {'-':>10s} | {'-':>11s}")
for name in MODES:
    s = mode_stats[name]
    print(f"{name:<20s} | {s['pix']:>12d} | {s['cell']:>12d} | {s['sub']:>16d} | "
          f"{s['cases_pix_clean']:>3d}/{s['cases']:<3d}    | {s['cases_cell_clean']:>3d}/{s['cases']:<3d}")
print()
print("cases pix\u2713  = cases where every pixel Jdet > 0 after correction")
print("cases cell\u2713 = cases where every bilinear cell-min Jdet > 0 after correction")
print("              (sub-pixel injectivity certificate — the stronger diagnostic)")
"""


def as_source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": as_source_lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": as_source_lines(text),
        "outputs": [],
        "execution_count": None,
    }


def main() -> None:
    with io.open(NB, encoding="utf-8") as f:
        nb = json.load(f)

    # Idempotent removal of previous multi-mode insertion.
    kept = []
    skip_until_sep = False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if skip_until_sep:
            if c["cell_type"] == "markdown" and src.strip() == "---":
                skip_until_sep = False
            continue
        if c["cell_type"] == "markdown" and "## Constraint-mode comparison" in src:
            skip_until_sep = True
            continue
        if c["cell_type"] == "code" and "INJECTIVITY_THRESHOLD = 0.3" in src and "MODES = {" in src:
            continue
        if c["cell_type"] == "code" and 'mode_stats["initial"]' in src:
            continue
        kept.append(c)
    nb["cells"] = kept

    # Find _fmt_cell summary cell to insert right after it.
    anchor_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "_fmt_cell" in src:
            anchor_idx = i
            break

    if anchor_idx is None:
        print("could not locate anchor summary cell")
        sys.exit(1)

    # Insert new cells after the next '---' separator that follows the anchor.
    insert_at = anchor_idx + 1
    while insert_at < len(nb["cells"]):
        c = nb["cells"][insert_at]
        if c["cell_type"] == "markdown" and "".join(c["source"]).strip() == "---":
            insert_at += 1
            break
        insert_at += 1

    new_cells = [
        md_cell(EXTRA_MODES_MD),
        code_cell(EXTRA_DRIVER_CODE),
        md_cell("---\n"),
        code_cell(EXTRA_SUMMARY_CODE),
        md_cell("---\n"),
    ]
    nb["cells"][insert_at:insert_at] = new_cells
    print(f"inserted {len(new_cells)} cells @ {insert_at} (after anchor @ {anchor_idx})")

    with io.open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
