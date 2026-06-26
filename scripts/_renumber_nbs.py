"""Rename the two-triangle-check notebooks into reading order and update links."""

import json
import os
import shutil

FOLDER = "notebooks/two-triangle-check"

RENAMES = [
    ("triangle-sign-vs-central-diff.ipynb", "01_vs-central-diff.ipynb"),
    ("triangle-sign-optimization.ipynb", "02_optimization.ipynb"),
    ("triangle-sign-demos.ipynb", "03_demos.ipynb"),
    ("triangle-sign-constraint-comparison.ipynb", "04_constraint-comparison.ipynb"),
    ("triangle-sign-solver-engineering.ipynb", "05_solver-engineering.ipynb"),
]

# Map old filename → new filename for link rewriting.
LINK_MAP = {old: new for old, new in RENAMES}

# --- 1. Rename files ---
for old, new in RENAMES:
    old_path = os.path.join(FOLDER, old)
    new_path = os.path.join(FOLDER, new)
    if os.path.exists(new_path):
        print(f"skip (already exists): {new_path}")
        continue
    if not os.path.exists(old_path):
        print(f"missing source: {old_path}")
        continue
    shutil.move(old_path, new_path)
    print(f"renamed {old} -> {new}")

# --- 2. Rewrite cross-references in every notebook's markdown cells ---
for _, new in RENAMES:
    path = os.path.join(FOLDER, new)
    nb = json.load(open(path, encoding="utf-8"))
    dirty = False
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        src_list = c["source"]
        new_src = []
        for line in src_list:
            new_line = line
            for old_name, new_name in LINK_MAP.items():
                if old_name in new_line:
                    new_line = new_line.replace(old_name, new_name)
            new_src.append(new_line)
        if new_src != src_list:
            c["source"] = new_src
            dirty = True
    if dirty:
        json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
        print(f"updated links in {new}")
    else:
        print(f"no link changes needed in {new}")

# --- 3. Print the final ordering for the user ---
print("\nFinal notebook ordering:")
for _, new in RENAMES:
    size_kb = os.path.getsize(os.path.join(FOLDER, new)) // 1024
    print(f"  {new:<40s} {size_kb:>6d} KB")
