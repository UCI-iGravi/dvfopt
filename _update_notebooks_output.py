"""Add standardized output (PNG saves, CSV, JSON summary) to all benchmark notebooks.

For each notebook:
1. Ensure benchmark_utils output functions are imported
2. Insert an OUTPUT_DIR setup cell after the first code cell
3. Replace plt.show() with show_and_save(OUTPUT_DIR)
4. Add a final cell that saves CSV + JSON summary (if 'results' dict exists)
"""
import json
import re
from pathlib import Path

# Notebook -> (method, name) mapping
NB_MAP = {
    # solvers/slsqp
    "solvers/slsqp/benchmark-3d-correction": ("slsqp", "3d-correction"),
    "solvers/slsqp/benchmark-constraint-modes": ("slsqp", "constraint-modes"),
    "solvers/slsqp/benchmark-serial-vs-parallel": ("slsqp", "serial-vs-parallel"),
    "solvers/slsqp/benchmark-windowed-vs-fullgrid": ("slsqp", "windowed-vs-fullgrid"),
    # solvers/barrier
    "solvers/barrier/benchmark-3d-barrier": ("barrier", "3d-correction"),
    "solvers/barrier/benchmark-barrier-cpu-vs-gpu": ("barrier", "cpu-vs-gpu"),
    # scaling
    "scaling/benchmark-scalability": ("slsqp", "scalability"),
    "scaling/benchmark-scalability-barrier": ("barrier", "scalability"),
    "scaling/benchmark-folding-severity": ("slsqp", "folding-severity"),
    "scaling/benchmark-folding-severity-barrier": ("barrier", "folding-severity"),
    "scaling/benchmark-l2-jdet-correlation": ("slsqp", "l2-jdet-correlation"),
    "scaling/benchmark-l2-jdet-correlation-barrier": ("barrier", "l2-jdet-correlation"),
    # registration
    "registration/benchmark-registration-methods": ("slsqp", "registration-methods"),
    "registration/benchmark-registration-methods-barrier": ("barrier", "registration-methods"),
    "registration/correct-real-ants-warp": ("slsqp", "ants-warp"),
    "registration/elastix-registration": ("slsqp", "elastix"),
    "registration/opencv-optflow-registration": ("slsqp", "opencv-optflow"),
    "registration/transmorph-registration": ("slsqp", "transmorph"),
    "registration/voxelmorph-registration": ("slsqp", "voxelmorph"),
    # pipelines
    "pipelines/correct-3d-slices": ("slsqp", "3d-slices"),
    "pipelines/correct-3d-slices-serial": ("slsqp", "3d-slices-serial"),
    "pipelines/correct-3d-slices-parallel": ("slsqp", "3d-slices-parallel"),
    "pipelines/correct-3d-slices-multi-constraint": ("slsqp", "3d-slices-multi-constraint"),
}

# Output helper imports to inject
OUTPUT_IMPORTS = [
    "get_output_dir",
    "save_figure",
    "save_results_csv",
    "save_summary_json",
    "log_run_header",
    "log_run_footer",
    "results_to_rows",
    "show_and_save",
    "reset_figure_counter",
]


def make_setup_cell(method, name, depth):
    """Create the OUTPUT_DIR setup code cell."""
    # depth = how many levels from benchmarks/ (e.g., scaling/ = 1, solvers/slsqp/ = 2)
    parent_dots = "/".join([".."] * depth)
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [
            f'METHOD = "{method}"\n',
            f'NOTEBOOK_NAME = "{name}"\n',
            f'OUTPUT_DIR = get_output_dir(METHOD, NOTEBOOK_NAME, base="{parent_dots}/output")\n',
            "reset_figure_counter()\n",
            "summary = log_run_header(METHOD, NOTEBOOK_NAME, OUTPUT_DIR)\n",
        ],
        "outputs": [],
        "execution_count": None,
    }


def make_save_cell():
    """Create the final save cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# --- Save results ---\n",
            "if 'results' in dir() and isinstance(results, dict) and results:\n",
            "    rows, cols = results_to_rows(results)\n",
            "    save_results_csv(rows, cols, OUTPUT_DIR)\n",
            "    summary = log_run_footer(summary, results)\n",
            "    save_summary_json(summary, OUTPUT_DIR)\n",
            "elif 'mag_results' in dir():\n",
            "    rows, cols = results_to_rows(mag_results)\n",
            "    save_results_csv(rows, cols, OUTPUT_DIR, name='results_magnitude')\n",
            "    if 'density_results' in dir():\n",
            "        rows2, cols2 = results_to_rows(density_results)\n",
            "        save_results_csv(rows2, cols2, OUTPUT_DIR, name='results_density')\n",
            "    combined = {**mag_results, **density_results} if 'density_results' in dir() else mag_results\n",
            "    summary = log_run_footer(summary, combined)\n",
            "    save_summary_json(summary, OUTPUT_DIR)\n",
            "else:\n",
            "    save_summary_json(summary, OUTPUT_DIR)\n",
            "    print('No results dict found; saved summary only.')\n",
        ],
        "outputs": [],
        "execution_count": None,
    }


def ensure_imports(cell_source, needed):
    """Ensure benchmark_utils imports include the needed functions."""
    joined = "".join(cell_source)
    if "from benchmark_utils" not in joined:
        return cell_source  # no benchmark_utils import at all — skip

    # Find existing imported names
    existing = set()
    for m in re.finditer(r"from\s+benchmark_utils\s+import\s*\(([^)]+)\)", joined, re.DOTALL):
        for name in re.findall(r"\b(\w+)\b", m.group(1)):
            existing.add(name)
    for m in re.finditer(r"from\s+benchmark_utils\s+import\s+(.+?)$", joined, re.MULTILINE):
        for name in re.findall(r"\b(\w+)\b", m.group(1)):
            existing.add(name)

    missing = [n for n in needed if n not in existing]
    if not missing:
        return cell_source

    # Add missing imports: append a new import line
    extra_line = f"from benchmark_utils import {', '.join(missing)}\n"
    return cell_source + [extra_line]


def add_benchmark_utils_import(cell_source, needed):
    """If cell has dvfopt imports but no benchmark_utils, add one."""
    joined = "".join(cell_source)
    if "benchmark_utils" in joined:
        return ensure_imports(cell_source, needed)
    if "from dvfopt" in joined or "import dvfopt" in joined:
        extra = f"from benchmark_utils import (\n    {', '.join(needed)},\n)\n"
        return cell_source + [extra]
    return cell_source


def replace_plt_show(cell_source):
    """Replace plt.show() with show_and_save(OUTPUT_DIR), including inline uses."""
    new = []
    for line in cell_source:
        if "plt.show()" in line and "show_and_save" not in line:
            line = line.replace("plt.show()", "show_and_save(OUTPUT_DIR)")
        new.append(line)
    return new


def process_notebook(nb_path, method, name):
    """Transform a single notebook."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Determine depth from benchmarks/
    rel = str(nb_path.parent.relative_to(Path("benchmarks")))
    depth = len(Path(rel).parts) if rel != "." else 0

    # 1. Find first code cell and ensure imports
    first_code_idx = None
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            src = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(keepends=True)
            joined = "".join(src)
            if "from dvfopt" in joined or "from benchmark_utils" in joined or "import dvfopt" in joined:
                cell["source"] = add_benchmark_utils_import(src, OUTPUT_IMPORTS)
                first_code_idx = i
                break
            if first_code_idx is None:
                first_code_idx = i

    if first_code_idx is None:
        print(f"  SKIP {nb_path} (no code cells)")
        return

    # 2. Insert setup cell after the import cell (if not already present)
    has_setup = any(
        "get_output_dir" in "".join(c.get("source", []))
        for c in cells
        if c["cell_type"] == "code" and "METHOD" in "".join(c.get("source", []))
    )
    if not has_setup:
        cells.insert(first_code_idx + 1, make_setup_cell(method, name, depth))

    # 3. Replace plt.show() in all code cells
    for cell in cells:
        if cell["cell_type"] == "code":
            src = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(keepends=True)
            cell["source"] = replace_plt_show(src)

    # 4. Add final save cell if not present
    last_src = "".join(cells[-1].get("source", []))
    if "save_results_csv" not in last_src and "save_summary_json" not in last_src:
        cells.append(make_save_cell())

    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  OK   {nb_path}")


def main():
    benchmarks = Path("benchmarks")
    for rel_key, (method, name) in NB_MAP.items():
        nb_path = benchmarks / f"{rel_key}.ipynb"
        if not nb_path.exists():
            print(f"  MISS {nb_path}")
            continue
        process_notebook(nb_path, method, name)
    print("\nDone!")


if __name__ == "__main__":
    main()
