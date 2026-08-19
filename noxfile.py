"""Task runner — ``nox -s <session>``.

One source of truth for how tests/lint/typecheck run, so contributors and CI
invoke them identically. In particular ``tests`` is scoped to ``tests/`` and
never over-collects the ``notebooks/experiments/_run_*_test.py`` scratch
scripts (which load gitignored data at import and error at collection).

    nox                    # default: lint + format_check + tests
    nox -s tests           # pytest tests/  (append `-- -n auto` for parallel)
    nox -s cov             # pytest with coverage
    nox -s lint            # ruff check
    nox -s format_check    # ruff format --check
    nox -s typecheck       # mypy (scoped; see pyproject [tool.mypy])

Sessions run on the interpreter that launched nox (no version pinning), so
they work on any dev box; CI uses its own Python matrix.
"""

import nox

nox.options.sessions = ["lint", "format_check", "tests"]

_LINT_PATHS = ("dvfopt", "dvfopt_gui", "tests", "benchmarks")
_RUFF = "ruff==0.15.21"  # keep in lockstep with pyproject [dev] + .pre-commit-config.yaml


@nox.session
def tests(session):
    session.install("-e", ".[dev,fast]")
    session.run("pytest", "tests/", "-q", *session.posargs)


@nox.session
def cov(session):
    session.install("-e", ".[dev,fast]")
    session.run("pytest", "tests/", "--cov=dvfopt", "--cov-report=term-missing", *session.posargs)


@nox.session
def lint(session):
    session.install(_RUFF)
    session.run("ruff", "check", *_LINT_PATHS)


@nox.session
def format_check(session):
    session.install(_RUFF)
    session.run("ruff", "format", "--check", *_LINT_PATHS)


@nox.session
def typecheck(session):
    session.install("-e", ".[dev]")
    session.run("mypy")
