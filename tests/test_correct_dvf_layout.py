"""``correct_dvf`` / ``dvfopt correct`` on the canonical ``(3, 1, H, W)`` layout.

Regression: the one-call API inferred the constraint shape as ``phi.shape[1:]``,
which is ``(1, H, W)`` for the canonical layout every loader produces, and every
2D constraint then rejected its own input — the CLI failed the same way. The
suite only ever passed ``(2, H, W)`` (``planted_fold``), so nothing caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt import correct_dvf
from dvfopt.cli import main as cli_main
from dvfopt.metrics import constraint_fold_stats
from tests.conftest import planted_fold


def _canonical(phi2):  # (2, H, W) -> (3, 1, H, W) [dz, dy, dx], dz = 0
    return np.stack([np.zeros_like(phi2[0]), phi2[0], phi2[1]])[:, None]


@pytest.mark.parametrize(
    'constraint,strategy,objective',
    [('simplex', 'auto', 'l1'), ('bilinear', 'barrier', 'l2'), ('jdet', 'barrier', 'l2')],
)
def test_correct_dvf_accepts_canonical_layout(constraint, strategy, objective):
    phi = _canonical(planted_fold(12, 14, seed=1, scale=0.4))
    assert constraint_fold_stats(phi, constraint)[1].n_neg > 0
    res = correct_dvf(phi, constraint=constraint, strategy=strategy, objective=objective)
    out = np.asarray(res.corrected)
    assert out.shape[-2:] == (12, 14)
    assert (
        constraint_fold_stats(out, constraint)[1].n_neg
        < constraint_fold_stats(phi, constraint)[1].n_neg
    )


def test_correct_dvf_explicit_shape_and_bare_layout_unchanged():
    phi2 = planted_fold(12, 14, seed=1, scale=0.4)
    a = correct_dvf(phi2, constraint='simplex', strategy='auto', objective='l1')
    b = correct_dvf(_canonical(phi2), constraint='simplex', strategy='auto', objective='l1')
    np.testing.assert_allclose(
        np.asarray(a.corrected).reshape(2, 12, 14), np.asarray(b.corrected)[1:].reshape(2, 12, 14)
    )


def test_cli_correct_accepts_canonical_layout(tmp_path):
    src, dst = tmp_path / 'in.npy', tmp_path / 'out.npy'
    np.save(src, _canonical(planted_fold(12, 14, seed=1, scale=0.4)))
    code = cli_main(
        [
            'correct',
            str(src),
            str(dst),
            '--constraint',
            'bilinear',
            '--strategy',
            'barrier',
            '--objective',
            'l2',
        ]
    )
    assert code in (0, 1)  # 0 feasible / 1 folds remain — never a usage/shape error (2)
    assert dst.is_file() and np.load(dst).shape[-2:] == (12, 14)
