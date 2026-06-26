"""Generator for ``slack_reform_slsqp.ipynb`` — keeps the prototype
source in plain text so it can be regenerated / diff-reviewed.
Run: ``python _build_slack_reform_nb.py`` from this directory.

We store each cell as a (kind, source_path) pair pointing to a .py /
.md sibling file. This avoids the triple-quote-inside-triple-quote
parsing hazards of inlining notebook source as Python string literals.
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SRCDIR = HERE / '_slack_reform_cells'

CELLS = [
    ('md', 'intro.md'),
    ('code', 'imports.py'),
    ('md', 'sec1.md'),
    ('code', 'slack_solver.py'),
    ('md', 'sec2.md'),
    ('code', 'fd_check.py'),
    ('md', 'sec3.md'),
    ('code', 'baseline_ref.py'),
    ('md', 'sec4.md'),
    ('code', 'cases_helpers.py'),
    ('code', 'run_synth_20x20.py'),
    ('code', 'run_synth_30x30.py'),
    ('code', 'run_b0039_z200.py'),
    ('code', 'run_b0039_z12.py'),
    ('md', 'sec5.md'),
    ('code', 'summary_df.py'),
    ('md', 'sec6.md'),
    ('code', 'viz.py'),
    ('md', 'sec7.md'),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, name in CELLS:
        src = (SRCDIR / name).read_text(encoding='utf-8')
        if kind == 'md':
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    out = HERE / 'slack_reform_slsqp.ipynb'
    nbf.write(nb, out)
    print(f'Wrote {out} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
