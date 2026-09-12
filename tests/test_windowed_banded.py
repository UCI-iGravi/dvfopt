import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.windowed import min_field, windowed_correct_banded
from dvfopt.objectives import L2Objective
from tests.conftest import planted_fold_3d

pytest.importorskip('osqp')


def _two_clusters(D=30, H=12, W=12):
    """Two planted fold clusters: one inside band 0 (core ``[0, 15)``), one
    straddling the ``z=15`` core boundary (band 1's core is ``[15, 30)``).

    ``planted_fold_3d(8, H, W, depth=1.4)`` punches its fold through the
    WHOLE depth of an 8-slab, at ``z in [1, 7)`` (see ``tests/conftest.py``).
    Pasting that 8-slab at z-offset 1 puts cluster A's fold at ``z in [2, 8)``
    (inside ``[0, 15)``); pasting it again at z-offset 11 puts cluster B's
    fold at ``z in [12, 18)`` (straddling ``z=15``).
    """
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, D, H, W))
    a = planted_fold_3d(8, H, W, depth=1.4)  # (3, 8, H, W); fold at z in [1, 7)
    phi[:, 1:9] = a  # cluster A: z in [2, 8) -- inside band 0
    phi[:, 11:19] = a  # cluster B: z in [12, 18) -- crosses the z=15 seam
    return phi


def _fold_mask(phi, thr=0.01):
    return min_field(SimplexConstraint3D(shape=phi.shape[1:]), phi) < thr


def _folds(phi, thr=0.01):
    return int(_fold_mask(phi, thr).sum())


def test_two_clusters_fixture_straddles_the_band_boundary():
    """R4 sanity check: don't trust the paste-offset arithmetic, verify it."""
    phi = _two_clusters()
    mask = _fold_mask(phi)
    assert mask[:15].any()  # folds before the z=15 seam (cluster A and/or B's low half)
    assert mask[15:].any()  # cluster B's high half: folds past the seam too
    assert mask[:10].any()  # cluster A alone, well clear of the seam


@pytest.mark.parametrize('n_workers', [1, 2])
def test_banded_certifies_and_moves_locally(n_workers):
    phi = _two_clusters()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert _folds(phi) > 0
    out, rep = windowed_correct_banded(
        phi,
        'isqp',
        constraint=c,
        threshold=0.01,
        objective=L2Objective(),
        band=15,
        overlap=4,
        n_workers=n_workers,
        verbose=0,
    )
    assert rep.bands == 2 and len(rep.band_walls) == 2
    assert rep.folds_after == 0 and rep.folds_after_zero == 0
    assert rep.best_diag_floor_after == 0
    assert rep.damage == 0
    moved = np.any(out != phi, axis=0)
    assert 0 < moved.mean() < 0.5  # local: the two clusters' neighbourhoods, not the volume
    assert rep.n_windows >= rep.seam_windows >= 0


def test_banded_rejects_2d_and_thin_overlap():
    from dvfopt.constraints import SimplexConstraint2DBilinear
    from tests.conftest import planted_fold

    phi2 = planted_fold(20, 20).astype(np.float64)
    with pytest.raises(ValueError, match='3D'):
        windowed_correct_banded(
            phi2, 'isqp', constraint=SimplexConstraint2DBilinear(shape=(20, 20)), threshold=0.01
        )
    phi = _two_clusters()
    with pytest.raises(ValueError, match='overlap'):
        windowed_correct_banded(
            phi,
            'isqp',
            constraint=SimplexConstraint3D(shape=phi.shape[1:]),
            threshold=0.01,
            band=15,
            overlap=1,
        )


def test_banded_checkpoint_reloads_a_finished_run(tmp_path):
    phi = _two_clusters()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out1, rep1 = windowed_correct_banded(
        phi,
        'isqp',
        constraint=c,
        threshold=0.01,
        band=15,
        overlap=4,
        checkpoint_dir=tmp_path,
        verbose=0,
    )
    assert (tmp_path / 'state.json').exists() and (tmp_path / 'seam' / 'state.json').exists()
    out2, rep2 = windowed_correct_banded(
        phi,
        'isqp',
        constraint=c,
        threshold=0.01,
        band=15,
        overlap=4,
        checkpoint_dir=tmp_path,
        verbose=0,
    )
    np.testing.assert_array_equal(out1, out2)
    assert rep2.resumed_from == 'finished'
    assert rep2.damage == 0 and rep2.folds_after == 0
