"""Shared fixtures for dvfopt unit tests."""

import logging

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate_dvfopt_logger():
    """Snapshot/restore the ``dvfopt`` logger around every test.

    The logger carries process-global state (handlers, level, propagate) that
    solver-verbosity emits through (``dvfopt._logging``). Without isolation a
    test that installs a handler or raises the level leaks into later tests —
    e.g. the ``verbose=`` capsys tests then see no ``[init]`` line. Rolling the
    logger back after each test keeps ordering-independent.
    """
    lg = logging.getLogger("dvfopt")
    saved = (lg.handlers[:], lg.level, lg.propagate, lg.disabled)
    try:
        yield
    finally:
        lg.handlers[:] = saved[0]
        lg.setLevel(saved[1])
        lg.propagate = saved[2]
        lg.disabled = saved[3]


@pytest.fixture
def identity_phi_2d():
    """A 10x10 zero-displacement field (identity deformation)."""
    return np.zeros((2, 10, 10), dtype=np.float64)


@pytest.fixture
def identity_deformation_2d():
    """A (3,1,10,10) zero-displacement deformation field."""
    return np.zeros((3, 1, 10, 10), dtype=np.float64)


@pytest.fixture
def small_deformation_2d():
    """A (3,1,8,8) deformation with mild smooth displacement (no negative Jdet)."""
    rng = np.random.default_rng(42)
    d = np.zeros((3, 1, 8, 8), dtype=np.float64)
    # Small smooth displacements that won't cause negative Jacobians
    d[1, 0] = rng.uniform(-0.1, 0.1, (8, 8))  # dy
    d[2, 0] = rng.uniform(-0.1, 0.1, (8, 8))  # dx
    return d


@pytest.fixture
def identity_phi_3d():
    """A 6x6x6 zero-displacement 3D field."""
    return np.zeros((3, 6, 6, 6), dtype=np.float64)


@pytest.fixture
def identity_deformation_3d():
    """A (3,6,6,6) zero-displacement 3D deformation field."""
    return np.zeros((3, 6, 6, 6), dtype=np.float64)


# ---------------------------------------------------------------------------
# Shared helper for "planted fold" 2D fields used across many test files.
# Previously duplicated 5x in test_tri_*.py, test_wallbreakers.py, test_unified.py
# with diverging defaults (scale=0.3 vs 0.4). Now lives here.
# ---------------------------------------------------------------------------


def planted_fold(H: int = 10, W: int = 10, *, seed: int = 0, scale: float = 0.4) -> np.ndarray:
    """Return a ``(2, H, W)`` field with channels ``[dy, dx]`` and a
    planted fold (probabilistically). ``scale=0.4`` typically produces
    a few-to-dozen negative triangles on a 10×10 grid.
    """
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, scale, (H, W)), rng.normal(0, scale, (H, W))])


def planted_fold_3d(
    D: int = 4, H: int = 6, W: int = 6, *, seed: int = 0, depth: float = 1.4
) -> np.ndarray:
    """Return a ``(3, D, H, W)`` ``[dz, dy, dx]`` volume with a punched
    central fold. 3D sibling of :func:`planted_fold` — consolidates the
    folded-volume builders previously duplicated across the 3D test files.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.02, (3, D, H, W))
    v[1, 1 : D - 1, 2:4, 2:4] -= depth
    v[2, 1 : D - 1, 2:4, 2:4] -= depth
    return v


# ---------------------------------------------------------------------------
# Shared offscreen QApplication for the GUI test files (PyQt5-gated).
# Session-scoped: one QApplication per test session (Qt allows only one).
# ---------------------------------------------------------------------------


@pytest.fixture(scope='session')
def qapp():
    PyQt5 = pytest.importorskip('PyQt5', reason='dvfopt_gui requires the [gui] extra (PyQt5)')
    del PyQt5
    import os

    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
