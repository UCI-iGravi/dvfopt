import json

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

    ``planted_fold_3d(6, H, W, depth=3.0)`` punches its fold through the
    WHOLE depth of a 6-slab, at ``z in [1, 5)`` (see ``tests/conftest.py``).
    Pasting that 6-slab at z-offset 1 puts cluster A's fold at ``z in [2, 6)``
    (inside ``[0, 15)``, well clear of the seam); pasting it again at
    z-offset 12 puts cluster B's fold at ``z in [13, 17)`` -- 2 planes on
    each side of ``z=15``.

    ``depth=1.4`` (the original choice) leaves ``seam_folds_before == 0``:
    each band's slab (``overlap=4``) sees cluster B in FULL, so the two
    independent per-band solves converge to values close enough at the
    seam that the composed cube never dips below threshold -- i.e. that
    milder fixture never actually exercised the seam pass's repair job.
    ``depth=3.0`` (measured, see the module test) makes the two
    independent fixes disagree enough at the ``z=14``/``z=15`` boundary to
    leave 2 genuine folds there, which the seam pass then clears to 0.
    """
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.02, (3, D, H, W))
    a = planted_fold_3d(6, H, W, depth=3.0)  # (3, 6, H, W); fold at z in [1, 5)
    phi[:, 1:7] = a  # cluster A: z in [2, 6) -- inside band 0, clear of the seam
    phi[:, 12:18] = a  # cluster B: z in [13, 17) -- straddles the z=15 seam
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
    # The seam pass must have real work to do: cluster B's two independent per-band
    # fixes disagree at the z=14/z=15 boundary (measured: 2 folds for both n_workers)
    # -- this is what pins that banding, not just re-running the whole-volume engine,
    # is actually being exercised here.
    assert rep.seam_folds_before > 0
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
    # The reload must carry the real counters, not zeros: the band rows plus the
    # run-level 'seam' row (the driver writes exactly these into its record).
    assert len(rep2.band_walls) == 2 and rep2.band_folds_after == rep1.band_folds_after
    assert rep2.seam_windows == rep1.seam_windows
    assert rep2.seam_folds_before == rep1.seam_folds_before
    assert rep2.n_windows == rep1.n_windows


@pytest.mark.parametrize('n_workers', [1, 2])
def test_banded_checkpoint_write_failure_propagates(tmp_path, monkeypatch, n_workers):
    """R15: a failed checkpoint write is not a broken pool.

    It must propagate out of the sweep instead of being swallowed by the pool-break
    guard and retried — a retry would re-run and re-commit that band, double-appending
    its ``band_walls`` / ``band_folds_after`` row, double-counting ``n_windows`` and
    re-marking ``band:k``. The checkpoint left behind must still be consistent: only
    the bands whose mark completed, each exactly once, so an unpatched re-run resumes
    and finishes.
    """
    import dvfopt.checkpoint as checkpoint

    phi = _two_clusters()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    kw = dict(constraint=c, threshold=0.01, band=15, overlap=4, n_workers=n_workers, verbose=0)
    real = checkpoint.atomic_replace
    writes = []

    def flaky(tmp, dst, **rest):
        if dst.name == 'touched.npy':
            writes.append(dst)
            if len(writes) == 2:  # band 1's commit — band 0 is committed and marked
                raise OSError('touched.npy write failed')
        return real(tmp, dst, **rest)

    monkeypatch.setattr(checkpoint, 'atomic_replace', flaky)
    with pytest.raises(OSError, match=r'touched\.npy write failed'):
        windowed_correct_banded(phi, 'isqp', checkpoint_dir=tmp_path, **kw)
    done = [str(u) for u in json.loads((tmp_path / 'state.json').read_text())['done']]
    assert done == ['band:0']  # band 1 never marked, and no band marked twice
    assert len(done) == len(set(done))

    monkeypatch.undo()
    out, rep = windowed_correct_banded(phi, 'isqp', checkpoint_dir=tmp_path, **kw)
    assert rep.resumed_from == 'band:0' and len(rep.band_walls) == 2
    assert rep.folds_after == 0 and rep.damage == 0
    assert _folds(out) == 0


def test_banded_checkpoint_resumes_mid_sweep(tmp_path):
    """The capstone's primary resume path: band 0 committed, band 1 not.

    Simulated by rewriting ``state.json`` back to ``stage='run'`` with only
    ``band:0`` done (and only its row), the way a kill between the two band marks
    leaves it. The seam sub-checkpoint is removed with them: a real mid-sweep
    interruption cannot have one (the seam pass runs after the whole sweep), and
    band 1 re-runs against band 0's RESTORED neighbour planes, so the composed
    field it produces is legitimately not the one run 1 handed the seam pass --
    a resumed run is documented as equivalent, not byte-identical (keeping the
    stale seam checkpoint here makes its own input hash refuse the run, which is
    the correct refusal for a checkpoint of a different composed field).
    """
    import shutil

    phi = _two_clusters()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    kw = dict(constraint=c, threshold=0.01, band=15, overlap=4, n_workers=1, verbose=0)
    windowed_correct_banded(phi, 'isqp', checkpoint_dir=tmp_path, **kw)

    sp = tmp_path / 'state.json'
    state = json.loads(sp.read_text())
    assert state['done'] == ['band:0', 'band:1'] and 'seam' in state['rows']
    state['stage'] = 'run'
    state['done'] = ['band:0']
    state['rows'] = {'band:0': state['rows']['band:0']}
    sp.write_text(json.dumps(state))
    shutil.rmtree(tmp_path / 'seam')

    out, rep = windowed_correct_banded(phi, 'isqp', checkpoint_dir=tmp_path, **kw)
    assert rep.resumed_from == 'band:0'
    assert rep.bands == 2 and len(rep.band_walls) == 2  # one restored, one fresh
    assert rep.seam_windows >= 1  # the seam pass ran again
    assert rep.folds_after == 0 and rep.folds_after_zero == 0
    assert rep.best_diag_floor_after == 0
    assert rep.damage == 0
    assert _folds(out) == 0  # the returned field certifies, independently of the report
