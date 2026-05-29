"""Generator for ``m14_schwarz_prototype.ipynb``."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SRCDIR = HERE / '_slack_reform_cells'

CELLS = [
    ('md', 'm14_schwarz_intro.md'),
    ('code', 'm14_schwarz_imports.py'),
    ('md', 'm14_schwarz_sec1.md'),
    ('code', 'm14_schwarz_cases.py'),
    ('md', 'm14_schwarz_sec2.md'),
    ('code', 'm14_schwarz_runner.py'),
    ('code', 'm14_schwarz_run_all.py'),
    ('md', 'm14_schwarz_sec3.md'),
    ('code', 'm14_schwarz_summary.py'),
    ('md', 'm14_schwarz_sec4.md'),
    ('code', 'm14_schwarz_viz.py'),
    ('md', 'm14_schwarz_findings.md'),
]


def main():
    nb = nbf.v4.new_notebook()
    for kind, name in CELLS:
        src = (SRCDIR / name).read_text(encoding='utf-8')
        if kind == 'md':
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    out = HERE / 'm14_schwarz_prototype.ipynb'
    nbf.write(nb, out)
    print(f'Wrote {out} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
