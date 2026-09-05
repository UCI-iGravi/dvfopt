"""``dvfopt.testdata.make_patch_folded_dvf`` — the GUI's synthetic-fold source."""

import numpy as np

from dvfopt.metrics import constraint_fold_stats
from dvfopt.testdata import make_patch_folded_dvf


def test_layout_and_folds_with_defaults():
    phi = make_patch_folded_dvf()
    assert phi.shape == (3, 1, 128, 128)
    assert phi.dtype == np.float64
    assert np.all(phi[0] == 0)
    _, st = constraint_fold_stats(phi, 'simplex_standard')
    assert st.n_neg > 0
    assert st.min_val < 0


def test_seeded_and_seed_sensitive():
    a = make_patch_folded_dvf((32, 40), seed=3)
    b = make_patch_folded_dvf((32, 40), seed=3)
    c = make_patch_folded_dvf((32, 40), seed=4)
    assert a.shape == (3, 1, 32, 40)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
