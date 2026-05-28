"""Smoke tests for ``iterative_2d_tri_barrier``.

Previously untested module — these guard the public contract:
- The return shape switches between ``phi`` and ``(phi, history)`` based on
  ``record_history``. The unified API relied on tuple-vs-ndarray detection,
  so silent contract drift would re-introduce the unified.py bug.
- ``anchor='l1'`` and ``anchor='none'`` paths must work; only ``'l2'`` was
  exercised indirectly before.
"""

import numpy as np
import pytest

from dvfopt.core.iterative2d_tri_barrier import (iterative_2d_tri_barrier,
                                                 _tri_areas_flat)


def _planted_fold(H=12, W=12, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, 0.3, (H, W)),
                     rng.normal(0, 0.3, (H, W))])


class TestReturnShape:
    def test_record_history_false_returns_phi_only(self):
        phi = _planted_fold()
        out = iterative_2d_tri_barrier(phi, verbose=0, record_history=False)
        # Critical: must be an ndarray, not a tuple. The unified API
        # relies on this exact distinction.
        assert isinstance(out, np.ndarray)
        assert out.shape == phi.shape

    def test_record_history_true_returns_phi_and_history(self):
        phi = _planted_fold()
        out = iterative_2d_tri_barrier(phi, verbose=0, record_history=True)
        assert isinstance(out, tuple)
        assert len(out) == 2
        phi_corr, history = out
        assert phi_corr.shape == phi.shape
        assert isinstance(history, list)
        # Each entry is a per-step dict with at least these keys.
        for h in history:
            assert "phase" in h
            assert "n_neg" in h
            assert "min_tri" in h


class TestFeasibility:
    def test_reduces_neg_triangle_count(self):
        phi = _planted_fold(H=14, W=14, seed=3)
        H, W = phi.shape[1], phi.shape[2]
        phi_flat_init = np.concatenate([phi[0].ravel(), phi[1].ravel()])
        T0 = _tri_areas_flat(phi_flat_init, H, W)
        n_neg_init = int((T0 <= 0).sum())
        assert n_neg_init > 0, "test setup needs an initial fold"

        phi_corr = iterative_2d_tri_barrier(phi, verbose=0, record_history=False)
        phi_flat_corr = np.concatenate([phi_corr[0].ravel(),
                                        phi_corr[1].ravel()])
        T1 = _tri_areas_flat(phi_flat_corr, H, W)
        n_neg_final = int((T1 <= 0).sum())
        assert n_neg_final <= n_neg_init


class TestAnchorModes:
    @pytest.mark.parametrize("anchor", ["l2", "l1", "none"])
    def test_anchor_does_not_crash(self, anchor):
        phi = _planted_fold(H=10, W=10, seed=1)
        out = iterative_2d_tri_barrier(phi, anchor=anchor, verbose=0,
                                       record_history=False)
        assert out.shape == phi.shape
        assert np.all(np.isfinite(out))


class TestAcceptsBothShapes:
    def test_2hw_input(self):
        phi = _planted_fold(H=8, W=8)
        out = iterative_2d_tri_barrier(phi, verbose=0, record_history=False)
        assert out.shape == (2, 8, 8)

    def test_31hw_input_coerced(self):
        phi2 = _planted_fold(H=8, W=8)
        phi = np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])[:, None]
        # (3, 1, 8, 8) shape — the entry point should coerce it.
        out = iterative_2d_tri_barrier(phi, verbose=0, record_history=False)
        assert out.shape == (2, 8, 8)
