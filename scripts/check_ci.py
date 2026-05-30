"""Mirror what the GitHub Actions Tests workflow runs, locally.

Why this exists
---------------

PR #10 failed CI even though local ``ruff check`` was green. The cause:
local was running ``ruff check dvfopt tests`` while CI runs
``ruff check dvfopt tests benchmarks``. The discrepancy (RUF100
unused-noqa) only surfaced in CI.

Running this script before pushing keeps local in lockstep with what CI
will check. Mirrors the steps in ``.github/workflows/test.yml`` exactly:

1. ``ruff check`` on the same three trees as CI
2. ``ruff format --check`` on the same trees
3. py_compile every ``benchmarks/_run_*.py`` script (the benchmark-smoke job)
4. ``pytest tests/``

Usage::

    python scripts/check_ci.py
    python scripts/check_ci.py --skip-tests   # lint/format/benchmark only
"""

from __future__ import annotations

import argparse
import glob
import py_compile
import subprocess
import sys

REPO_ROOT_RELATIVE_TARGETS = ['dvfopt', 'tests', 'benchmarks']


def _run(label: str, cmd: list[str]) -> bool:
    print(f'\n>>> {label}')
    print(f'    $ {" ".join(cmd)}')
    proc = subprocess.run(cmd)
    ok = proc.returncode == 0
    print(f'    {"PASS" if ok else "FAIL"}')
    return ok


def _benchmark_smoke() -> bool:
    print('\n>>> benchmark-smoke (py_compile)')
    failed = []
    for f in sorted(glob.glob('benchmarks/_run_*.py')):
        try:
            py_compile.compile(f, doraise=True)
            print(f'    OK  {f}')
        except py_compile.PyCompileError as e:
            failed.append((f, str(e)))
            print(f'    ERR {f}: {e}')
    print(f'    {"PASS" if not failed else f"FAIL ({len(failed)})"}')
    return not failed


_NO_TORCH_SIM_PY = '''
import sys
class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked (simulating CI no-torch install)")
        return None
sys.meta_path.insert(0, _BlockTorch())
'''


def _import_smoke_no_torch(py: str) -> bool:
    """Confirm dvfopt can be imported even when torch isn't installed.

    Simulates the CI ``[dev]`` install (which doesn't include torch) by
    blocking the import. Catches the regression class where a module is
    unconditionally ``import torch``-ing at top level — would have caught
    the PR #11 follow-up failure.
    """
    print('\n>>> no-torch import smoke')
    code = (
        _NO_TORCH_SIM_PY
        + '\nimport dvfopt\n'
        + 'from dvfopt.constraints import JdetConstraint2D\n'
        + 'from dvfopt.core import iterative2d_barrier  # was the regression site\n'
        + 'print("OK")\n'
    )
    proc = subprocess.run([py, '-c', code], capture_output=True, text=True)
    if proc.returncode != 0:
        print('    FAIL — stderr:')
        for line in (proc.stderr or '').splitlines()[-6:]:
            print(f'      {line}')
        return False
    print('    PASS')
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip the (slow) pytest step. Use for fast pre-push lint check.',
    )
    args = parser.parse_args()

    # Make the script robust to being invoked from a subdirectory.
    import os
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    # Use the same interpreter that's running this script — otherwise a
    # literal "python" shells out to whichever python comes first on PATH,
    # which is rarely the venv where ruff/pytest are installed.
    py = sys.executable

    results = []
    results.append(_run('Lint', [py, '-m', 'ruff', 'check', *REPO_ROOT_RELATIVE_TARGETS]))
    results.append(
        _run(
            'Format check',
            [py, '-m', 'ruff', 'format', '--check', *REPO_ROOT_RELATIVE_TARGETS],
        )
    )
    results.append(_benchmark_smoke())
    results.append(_import_smoke_no_torch(py))
    if not args.skip_tests:
        results.append(_run('Tests', [py, '-m', 'pytest', 'tests/', '-q']))
    else:
        print('\n>>> Tests SKIPPED (--skip-tests)')

    print()
    print('=' * 50)
    print(f'Summary: {sum(results)} / {len(results)} jobs passed')
    print('=' * 50)
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
