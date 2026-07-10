"""Headless tests for the ``dvfopt_gui`` non-widget logic.

Covers the pure, event-loop-free pieces: the worker's snapshot
conversion + bounded history deque, the read-only :class:`ReplayHistory`
stand-in, and the save/load persistence round-trip. These are exactly
the off-by-one-prone bits that break silently in the GUI, so they get
coverage without ever constructing a ``QApplication``.

Skipped wholesale if the GUI extras (PyQt5) aren't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('PyQt5', reason='dvfopt_gui requires the [gui] extra (PyQt5)')

from dvfopt_gui import persistence
from dvfopt_gui.worker import (
    FEASIBILITY_THRESHOLD,
    ReplayHistory,
    SolverWorker,
    StateSnapshot,
    _infeasible_count,
    _metric_counts,
    _state_to_snapshot,
)


def _bowtie_2hw(H=7, W=7):
    """7×7 shoelace bowtie: 0 neg-Jdet pixels but 2 folded 2-tri cells."""
    phi = np.zeros((2, H, W))
    phi[1, 3, 3] = 1.2  # dx
    phi[1, 3, 4] = -1.2
    return phi


def _bowtie_deformation(H=7, W=7):
    out = np.zeros((3, 1, H, W))
    out[2, 0, 3, 3] = 1.2
    out[2, 0, 3, 4] = -1.2
    return out


def _make_state(phi, *, cy=2, cx=3, sy=3, sx=3, osy=3, osx=3, **over):
    state = {
        'phi': phi,
        'window_center': (cy, cx),
        'window_size': (sy, sx),
        'opt_size': (osy, osx),
        'is_padded': False,
        'neg_index': (1, 1),
        'per_index_iter': 0,
        'outer_iter': 0,
        'n_neg': 0,
        'min_T': 0.0,
    }
    state.update(over)
    return state


def _snap(phi, **over):
    base = dict(
        phi=phi,
        window_y0=0,
        window_y1=0,
        window_x0=0,
        window_x1=0,
        opt_y0=0,
        opt_y1=0,
        opt_x0=0,
        opt_x1=0,
        is_padded=False,
        neg_y=0,
        neg_x=0,
        per_index_iter=0,
        outer_iter=0,
        n_neg=0,
        min_T=0.0,
    )
    base.update(over)
    return StateSnapshot(**base)


# ---------------------------------------------------------------------------
# snapshot conversion
# ---------------------------------------------------------------------------


def test_state_to_snapshot_copies_phi():
    phi = np.zeros((2, 6, 6))
    snap = _state_to_snapshot(_make_state(phi))
    phi[0, 0, 0] = 99.0  # mutate the live buffer after snapshotting
    assert snap.phi[0, 0, 0] == 0.0  # snapshot kept its own copy


def test_state_to_snapshot_window_bounds_clamped():
    # window centred at (0,0) with size 3 must clamp to grid bounds.
    phi = np.zeros((2, 5, 5))
    snap = _state_to_snapshot(_make_state(phi, cy=0, cx=0, sy=3, sx=3))
    assert snap.window_y0 == 0 and snap.window_x0 == 0
    assert 0 < snap.window_y1 <= 5 and 0 < snap.window_x1 <= 5


def test_state_to_snapshot_outer_iter_none_becomes_zero():
    snap = _state_to_snapshot(_make_state(np.zeros((2, 4, 4)), outer_iter=None))
    assert snap.outer_iter == 0


# ---------------------------------------------------------------------------
# per-run metric consistency
# ---------------------------------------------------------------------------


def test_metric_counts_2tri_catches_subpixel_fold():
    # Bowtie: central-diff Jdet sees nothing, 2-tri sees two folded cells.
    phi = _bowtie_2hw()
    n_neg_tri, min_T_tri = _metric_counts(phi, '2tri')
    n_neg_jdet, min_jdet = _metric_counts(phi, 'jdet')
    assert n_neg_tri == 2
    assert min_T_tri < 0
    assert n_neg_jdet == 0
    assert min_jdet >= 0


def test_metric_counts_folds_use_le_zero_for_both_metrics():
    # Bowtie still reads 2 folds; the convention is now <= 0 for jdet too
    # (matching the live SLSQP callback's (jac <= 0) count), not < 0.
    phi = _bowtie_2hw()
    assert _metric_counts(phi, '2tri')[0] == 2
    assert _metric_counts(phi, 'jdet')[0] == 0  # bowtie min Jdet ~0.4 > 0


def test_infeasible_count_flags_below_threshold_without_folds():
    # A near-singular expansion: Jdet ~0.005 everywhere — positive (no
    # folds) but inside the solver's 0.01 feasibility margin.
    H = W = 6
    _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    phi = np.zeros((2, H, W))
    phi[1] = -0.995 * xx  # dx ramp → d(dx)/dx ≈ -0.995
    n_neg, min_T = _metric_counts(phi, 'jdet')
    assert n_neg == 0  # nothing inverted
    assert 0 < min_T < FEASIBILITY_THRESHOLD  # positive yet sub-threshold
    # Every cell is below the solver threshold → all flagged infeasible.
    assert _infeasible_count(phi, 'jdet') == H * W


def test_infeasible_count_zero_for_clearly_feasible_field():
    # Identity field: Jdet == 1 >> threshold, triangle areas == 0.5.
    phi = np.zeros((2, 6, 6))
    assert _infeasible_count(phi, 'jdet') == 0
    assert _infeasible_count(phi, '2tri') == 0


@pytest.mark.parametrize(
    'method_id,expected',
    [
        ('slsqp_windowed_2tri', 'jdet'),  # SLSQP reports Jdet even with 2-tri flag
        ('slsqp_windowed_jdet', 'jdet'),
        ('m14_2tri', '2tri'),
        ('m14_schwarz_2tri', '2tri'),
        ('barrier_jdet', 'jdet'),
        ('nmvf_jdet', 'jdet'),
    ],
)
def test_trajectory_metric_kind(method_id, expected):
    w = SolverWorker(deformation_i=np.zeros((3, 1, 5, 5)), method_id=method_id)
    assert w._trajectory_metric_kind() == expected


def test_initial_snapshot_uses_run_metric():
    # A 2-tri run must record the 2-tri fold count at step 0 (not Jdet 0),
    # so the convergence trajectory starts consistent with its tail.
    w = SolverWorker(deformation_i=_bowtie_deformation(), method_id='m14_2tri')
    w._emit_initial_snapshot('2tri')
    assert w.history_get(0).n_neg == 2
    # The same input under a Jdet run records 0 (CD stencil misses it).
    w2 = SolverWorker(deformation_i=_bowtie_deformation(), method_id='barrier_jdet')
    w2._emit_initial_snapshot('jdet')
    assert w2.history_get(0).n_neg == 0


# ---------------------------------------------------------------------------
# SolverWorker history deque
# ---------------------------------------------------------------------------


def test_history_records_and_total():
    w = SolverWorker(deformation_i=np.zeros((3, 1, 4, 4)), history_max_size=10)
    for i in range(3):
        w._record(_snap(np.full((2, 4, 4), float(i))))
    assert w.history_len() == 3
    assert w.history_total == 3
    assert w.history_get(0).phi[0, 0, 0] == 0.0
    assert w.history_get(2).phi[0, 0, 0] == 2.0
    assert w.history_get(3) is None
    assert w.history_get(-1) is None


def test_history_cap_ages_out_oldest():
    # max_size floors at 2; total keeps counting past the cap.
    w = SolverWorker(deformation_i=np.zeros((3, 1, 4, 4)), history_max_size=2)
    for i in range(5):
        w._record(_snap(np.full((2, 4, 4), float(i))))
    assert w.history_len() == 2
    assert w.history_total == 5
    # Oldest dropped — buffer holds the last two (3, 4).
    assert w.history_get(0).phi[0, 0, 0] == 3.0
    assert w.history_get(1).phi[0, 0, 0] == 4.0


def test_take_latest_drains_once():
    w = SolverWorker(deformation_i=np.zeros((3, 1, 4, 4)))
    w._record(_snap(np.zeros((2, 4, 4))))
    assert w.take_latest() is not None
    assert w.take_latest() is None  # nothing new since last poll


# ---------------------------------------------------------------------------
# ReplayHistory
# ---------------------------------------------------------------------------


def test_replay_history_read_surface():
    snaps = [_snap(np.full((2, 3, 3), float(i))) for i in range(4)]
    rh = ReplayHistory(snaps, history_total=10)
    assert rh.history_len() == 4
    assert rh.history_total == 10
    assert rh.history_get(2).phi[0, 0, 0] == 2.0
    assert rh.history_get(99) is None
    assert rh.take_latest() is None
    assert rh.isRunning() is False
    assert rh.callback_count == 0


def test_replay_history_total_defaults_to_len():
    rh = ReplayHistory([_snap(np.zeros((2, 3, 3)))])
    assert rh.history_total == 1


# ---------------------------------------------------------------------------
# persistence: normalise_to_volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'shape',
    [(2, 5, 6), (3, 5, 6), (3, 1, 5, 6), (3, 4, 5, 6)],
)
def test_normalise_to_volume_accepts_valid_shapes(shape):
    vol = persistence.normalise_to_volume(np.zeros(shape))
    assert vol.ndim == 4 and vol.shape[0] == 3
    assert vol.dtype == np.float64


def test_normalise_to_volume_2hw_maps_channels_to_dy_dx():
    arr = np.stack([np.full((4, 4), 1.0), np.full((4, 4), 2.0)])  # [dy, dx]
    vol = persistence.normalise_to_volume(arr)
    assert np.all(vol[0] == 0.0)  # dz padded
    assert np.all(vol[1, 0] == 1.0)  # dy
    assert np.all(vol[2, 0] == 2.0)  # dx


@pytest.mark.parametrize('shape', [(4, 4), (4, 5, 6), (5, 1, 4, 4), (3, 2, 4, 5, 6)])
def test_normalise_to_volume_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        persistence.normalise_to_volume(np.zeros(shape))


# ---------------------------------------------------------------------------
# persistence: save/load round-trip
# ---------------------------------------------------------------------------


def test_save_payload_without_history():
    vol = np.zeros((3, 1, 5, 5))
    payload = persistence.build_save_payload(
        phi_active=vol[1:, 0],
        full_volume=vol,
        z=0,
        constraint='2tri',
        method='m14',
        objective='l1',
        time_budget_s=60.0,
        max_iterations=200,
        history_max_size=500,
        history_snaps=[],
        history_total=0,
    )
    assert int(payload['n_history_steps']) == 0
    assert 'history_phi' not in payload
    assert str(payload['method']) == 'm14'


def test_save_then_load_roundtrip_with_history(tmp_path):
    H, W = 5, 6
    vol = np.zeros((3, 2, H, W))
    vol[1, 0] = 0.25  # mark active slice so we can check it survives
    snaps = [
        _snap(np.full((2, H, W), float(i)), n_neg=i, min_T=float(-i), outer_iter=i)
        for i in range(4)
    ]
    payload = persistence.build_save_payload(
        phi_active=vol[1:, 0],
        full_volume=vol,
        z=0,
        constraint='2tri',
        method='m14',
        objective='l2',
        time_budget_s=30.0,
        max_iterations=100,
        history_max_size=10,
        history_snaps=snaps,
        history_total=7,  # some aged out before save
    )
    assert int(payload['n_history_steps']) == 4
    assert payload['history_phi'].shape == (4, 2, H, W)

    path = tmp_path / 'run.npz'
    np.savez_compressed(path, **payload)

    loaded = np.load(path, allow_pickle=False)
    run = persistence.parse_loaded(loaded)
    loaded.close()

    assert run.volume.shape == (3, 2, H, W)
    assert run.constraint == '2tri'
    assert run.method == 'm14'
    assert run.objective == 'l2'
    assert run.history_total == 7
    assert len(run.snapshots) == 4
    # Snapshot fields survived the round-trip.
    assert run.snapshots[3].phi[0, 0, 0] == 3.0
    assert run.snapshots[2].n_neg == 2
    assert run.snapshots[1].min_T == -1.0
    # Active slice preserved inside the full volume.
    assert np.allclose(run.volume[1, 0], 0.25)


def test_input_volume_roundtrips_and_is_distinct_from_corrected(tmp_path):
    # Save with a separate pre-correction input volume; load must restore
    # it distinct from the (corrected) phi_full_volume.
    H, W = 5, 5
    inp = np.zeros((3, 1, H, W))
    inp[1, 0, 2, 2] = 0.7  # original
    cur = np.zeros((3, 1, H, W))
    cur[1, 0, 2, 2] = 0.1  # "corrected"
    payload = persistence.build_save_payload(
        phi_active=cur[1:, 0],
        full_volume=cur,
        z=0,
        constraint='2tri',
        method='m14',
        objective='l1',
        time_budget_s=60.0,
        max_iterations=200,
        history_max_size=500,
        history_snaps=[],
        history_total=0,
        input_volume=inp,
    )
    assert 'phi_input_volume' in payload
    path = tmp_path / 'r.npz'
    np.savez_compressed(path, **payload)
    loaded = np.load(path, allow_pickle=False)
    run = persistence.parse_loaded(loaded)
    loaded.close()
    assert run.input_volume is not None
    assert run.input_volume[1, 0, 2, 2] == pytest.approx(0.7)
    assert run.volume[1, 0, 2, 2] == pytest.approx(0.1)


def test_input_volume_omitted_when_not_provided():
    payload = persistence.build_save_payload(
        phi_active=np.zeros((2, 4, 4)),
        full_volume=np.zeros((3, 1, 4, 4)),
        z=0,
        constraint='',
        method='',
        objective='',
        time_budget_s=1.0,
        max_iterations=1,
        history_max_size=2,
        history_snaps=[],
        history_total=0,
    )
    assert 'phi_input_volume' not in payload


def test_parse_loaded_bare_npy(tmp_path):
    arr = np.zeros((2, 4, 4))
    path = tmp_path / 'bare.npy'
    np.save(path, arr)
    loaded = np.load(path, allow_pickle=False)
    run = persistence.parse_loaded(loaded)
    assert run.volume.shape == (3, 1, 4, 4)
    assert run.snapshots == []
    assert run.history_total == 0


def test_parse_loaded_npz_phi_only(tmp_path):
    # An NPZ that carries just ``phi`` (e.g. the canonical data/dvfs schema)
    # — no saved-run history.
    path = tmp_path / 'phi.npz'
    np.savez_compressed(path, phi=np.zeros((2, 4, 4)))
    loaded = np.load(path, allow_pickle=False)
    run = persistence.parse_loaded(loaded)
    loaded.close()
    assert run.volume.shape == (3, 1, 4, 4)
    assert run.snapshots == []


def test_build_strategy_adds_2d_fullgrid_and_schwarz():
    from dvfopt import SchwarzStrategy, SLSQPFullGridStrategy

    w1 = SolverWorker(deformation_i=np.zeros((3, 1, 6, 6)), method_id='slsqp_fullgrid_2tri')
    assert isinstance(w1._build_strategy(), SLSQPFullGridStrategy)
    w2 = SolverWorker(deformation_i=np.zeros((3, 1, 6, 6)), method_id='schwarz_2tri')
    assert isinstance(w2._build_strategy(), SchwarzStrategy)


# ---------------------------------------------------------------------------
# 3D fold-metric helpers
# ---------------------------------------------------------------------------


def _folded_volume_3d(D=4, H=8, W=8):
    # A z-direction shear large enough to invert tets.
    _, _, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    phi = np.zeros((3, D, H, W))
    phi[2] = -1.8 * xx  # dx ramp -> strong compression, inverts cells
    return phi


def test_metric_counts_3d_tet_and_jdet():
    from dvfopt_gui.worker import _infeasible_count_3d, _metric_counts_3d

    phi = _folded_volume_3d()
    n_tet, min_tet = _metric_counts_3d(phi, 'tet3d')
    n_jdet, min_jdet = _metric_counts_3d(phi, 'jdet3d')
    assert n_tet > 0 and min_tet < 0
    assert n_jdet > 0 and min_jdet < 0
    # Identity volume: no folds, nothing infeasible.
    ident = np.zeros((3, 4, 8, 8))
    assert _metric_counts_3d(ident, 'tet3d') == (0, pytest.approx(1 / 6))
    assert _metric_counts_3d(ident, 'jdet3d') == (0, pytest.approx(1.0))
    assert _infeasible_count_3d(ident, 'tet3d') == 0
    assert _infeasible_count_3d(ident, 'jdet3d') == 0


# ---------------------------------------------------------------------------
# 3D worker dispatch: metric kind + strategy + end-to-end solve
# ---------------------------------------------------------------------------


def test_worker_3d_trajectory_metric_and_strategy():
    from dvfopt import (
        BarrierStrategy,
        HarmonicALMRefineRepair3DStrategy,
        SLSQPFullGrid3DStrategy,
    )

    vol = np.zeros((3, 4, 8, 8))
    assert (
        SolverWorker(deformation_i=vol, method_id='m14_tet3d')._trajectory_metric_kind() == 'tet3d'
    )
    assert (
        SolverWorker(deformation_i=vol, method_id='barrier_jdet3d')._trajectory_metric_kind()
        == 'jdet3d'
    )
    assert isinstance(
        SolverWorker(deformation_i=vol, method_id='m14_tet3d')._build_strategy(),
        HarmonicALMRefineRepair3DStrategy,
    )
    assert isinstance(
        SolverWorker(deformation_i=vol, method_id='slsqp_fullgrid_tet3d')._build_strategy(),
        SLSQPFullGrid3DStrategy,
    )
    assert isinstance(
        SolverWorker(deformation_i=vol, method_id='barrier_jdet3d')._build_strategy(),
        BarrierStrategy,
    )

    from dvfopt import SLSQPWindowedStrategy

    w = SolverWorker(deformation_i=vol, method_id='slsqp_windowed_jdet3d')
    assert w._trajectory_metric_kind() == 'jdet3d'
    assert isinstance(w._build_strategy(), SLSQPWindowedStrategy)
    assert (
        SolverWorker(deformation_i=vol, method_id='m10_tet3d')._trajectory_metric_kind() == 'tet3d'
    )
    assert (
        SolverWorker(deformation_i=vol, method_id='m14_schwarz_tet3d')._trajectory_metric_kind()
        == 'tet3d'
    )


def test_worker_3d_experimental_strategies():
    # The research 3D strategies are dispatchable + route to the 6-tet metric.
    from dvfopt import ActiveBandALM3DStrategy, CoupledKRing3DStrategy

    vol = np.zeros((3, 4, 8, 8))
    wa = SolverWorker(deformation_i=vol, method_id='active_band_tet3d')
    assert isinstance(wa._build_strategy(), ActiveBandALM3DStrategy)
    assert wa._trajectory_metric_kind() == 'tet3d'
    wc = SolverWorker(deformation_i=vol, method_id='coupled_kring_tet3d')
    assert isinstance(wc._build_strategy(), CoupledKRing3DStrategy)
    assert wc._trajectory_metric_kind() == 'tet3d'


def test_worker_3d_solve_reaches_feasibility():
    # Small folded volume; M14Tet should clear folds end-to-end.
    _, yy, xx = np.meshgrid(np.arange(4), np.arange(10), np.arange(10), indexing='ij')
    vol = np.zeros((3, 4, 10, 10))
    vol[2, :, 4:6, 4:6] = 1.5
    from dvfopt_gui.worker import _metric_counts_3d

    n_before, _ = _metric_counts_3d(vol[:, :], 'tet3d')
    assert n_before > 0
    w = SolverWorker(deformation_i=vol, method_id='m14_tet3d', params={'time_budget_s': 60.0})
    phi_out = w._run_via_solver_3d(w._build_strategy(), 'tet3d', metric_kind='tet3d')
    assert phi_out.shape == (3, 4, 10, 10)
    n_after, _ = _metric_counts_3d(phi_out, 'tet3d')
    assert n_after <= n_before
    # history has an input snapshot (ndim 4) + at least the final.
    assert w.history_len() >= 2
    assert w.history_get(0).phi.ndim == 4


def test_worker_3d_memory_guard_keeps_init_and_final(monkeypatch):
    import dvfopt_gui.worker as W

    monkeypatch.setattr(W, 'MAX_3D_HISTORY_BYTES', 1)  # force the guard
    _, yy, xx = np.meshgrid(np.arange(4), np.arange(10), np.arange(10), indexing='ij')
    vol = np.zeros((3, 4, 10, 10))
    vol[2, :, 4:6, 4:6] = 1.5
    w = SolverWorker(deformation_i=vol, method_id='m14_tet3d', params={'time_budget_s': 60.0})
    w._run_via_solver_3d(w._build_strategy(), 'tet3d', metric_kind='tet3d')
    # Guard tripped: only the input + final snapshots, no mid stages.
    assert w.history_len() == 2


# ---------------------------------------------------------------------------
# persistence: 3D history round-trip
# ---------------------------------------------------------------------------


def test_persistence_3d_history_roundtrip(tmp_path):
    from dvfopt_gui.worker import _volume_snapshot

    D, H, W = 3, 5, 5
    vol = np.zeros((3, D, H, W))
    snaps = [
        _volume_snapshot(np.full((3, D, H, W), float(i)), n_neg=i, min_T=float(-i), outer_iter=i)
        for i in range(3)
    ]
    payload = persistence.build_save_payload(
        phi_active=vol[1:, 0],
        full_volume=vol,
        z=0,
        constraint='tet3d',
        method='m14_tet3d',
        objective='l2',
        time_budget_s=60.0,
        max_iterations=200,
        history_max_size=8,
        history_snaps=snaps,
        history_total=3,
        input_volume=vol,
        dim=3,
    )
    assert int(payload['dim']) == 3
    assert payload['history_phi'].shape == (3, 3, D, H, W)
    path = tmp_path / 'run3d.npz'
    np.savez_compressed(path, **payload)
    loaded = np.load(path, allow_pickle=False)
    run = persistence.parse_loaded(loaded)
    loaded.close()
    assert len(run.snapshots) == 3
    assert run.snapshots[2].phi.shape == (3, D, H, W)
    assert run.snapshots[1].n_neg == 1


# ---------------------------------------------------------------------------
# SLP default + Auto strategy picker dispatch
# ---------------------------------------------------------------------------


def test_slp_and_auto_dispatch():
    from dvfopt import SLPStrategy

    vol2d = np.zeros((3, 1, 8, 8))
    w = SolverWorker(deformation_i=vol2d, method_id='slp_2tri')
    assert isinstance(w._build_strategy(), SLPStrategy)
    assert w._trajectory_metric_kind() == '2tri'

    # Auto on a mildly folded field resolves to a registry label and
    # records it on the worker.
    phi = np.zeros((3, 1, 8, 8))
    phi[2, 0, 3, 3] = 1.2
    phi[2, 0, 3, 4] = -1.2
    wa = SolverWorker(deformation_i=phi, method_id='auto_2tri', params={'objective_id': 'l1'})
    strat = wa._build_strategy()
    assert strat is not None
    # 2-tri + L1 now auto-routes to the SLP champion at every fold tier.
    assert wa.resolved_strategy_label == 'slp'
    # Non-l1 objectives keep the legacy tiered routing.
    wl2 = SolverWorker(deformation_i=phi, method_id='auto_2tri', params={'objective_id': 'l2'})
    wl2._build_strategy()
    assert wl2.resolved_strategy_label in ('m10', 'barrier', 'slsqp')
    wj = SolverWorker(deformation_i=phi, method_id='auto_jdet', params={'objective_id': 'l1'})
    wj._build_strategy()
    assert wj.resolved_strategy_label in ('barrier', 'slsqp_windowed')


# ---------------------------------------------------------------------------
# user-editable feasibility threshold
# ---------------------------------------------------------------------------


def test_threshold_param_reaches_solver(monkeypatch):
    import dvfopt

    captured = {}

    class _FakeSolver:
        def __init__(self, *, constraint, objective, strategy, threshold=None):
            captured['threshold'] = threshold

        def fit(self, phi, **kw):
            class R:
                corrected = np.zeros((2, 6, 6))

            return R()

    monkeypatch.setattr(dvfopt, 'Solver', _FakeSolver)
    w = SolverWorker(
        deformation_i=np.zeros((3, 1, 6, 6)),
        method_id='m14_2tri',
        params={'threshold': 0.02, 'objective_id': 'l1'},
    )
    w._run_via_solver(w._build_strategy(), '2tri', metric_kind='2tri')
    assert captured['threshold'] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 2.5D marching + full-3D pipeline runners, torch barrier
# ---------------------------------------------------------------------------


def test_marching_and_pipeline3d_dispatch(monkeypatch):
    vol = np.zeros((3, 4, 8, 8))
    called = {}
    monkeypatch.setattr(
        SolverWorker, '_run_marching_25d', lambda self: called.setdefault('m', True) or vol
    )
    w = SolverWorker(deformation_i=vol, method_id='marching25d_tet3d')
    w.run()
    assert called.get('m'), 'marching runner not dispatched'
    assert w._trajectory_metric_kind() == 'tet3d'

    monkeypatch.setattr(
        SolverWorker, '_run_pipeline_3d', lambda self: called.setdefault('p', True) or vol
    )
    w2 = SolverWorker(deformation_i=vol, method_id='pipeline3d_tet3d')
    w2.run()
    assert called.get('p'), 'pipeline3d runner not dispatched'


def test_marching_25d_end_to_end():
    # Per-slice-feasible, inter-layer-folded fixture (same as the library test).
    vol = np.zeros((3, 4, 8, 8))
    for k in range(4):
        vol[2, k, :, 4] = 1.5 if k % 2 == 0 else -1.5
    from dvfopt_gui.worker import _metric_counts_3d

    n0, _ = _metric_counts_3d(vol, 'tet3d')
    assert n0 > 0
    w = SolverWorker(deformation_i=vol, method_id='marching25d_tet3d', params={'threshold': 0.01})
    out = w._run_marching_25d()
    n1, _ = _metric_counts_3d(out, 'tet3d')
    assert n1 < n0
    assert w.pipeline_report is not None and w.pipeline_report.n_neg_out == n1
    assert w.history_len() >= 2 and w.history_get(0).phi.ndim == 4


def test_pipeline3d_dispatch_uses_stub(monkeypatch):
    import dvfopt

    vol = np.zeros((3, 4, 8, 8))

    class _R:
        n_neg_in, n_neg_out, feasible, wall_s = 3, 0, True, 0.1

    monkeypatch.setattr(dvfopt, 'correct_dvf_3d', lambda v, **kw: (v.copy(), _R()))
    w = SolverWorker(deformation_i=vol, method_id='pipeline3d_tet3d', params={'threshold': 0.02})
    out = w._run_pipeline_3d()
    assert out.shape == vol.shape
    assert w.pipeline_report is not None


def test_barrier_torch_dispatch():
    pytest.importorskip('torch', reason='torch barrier needs torch')
    from dvfopt import BarrierTet3DTorchStrategy

    w = SolverWorker(deformation_i=np.zeros((3, 4, 8, 8)), method_id='barrier_torch_tet3d')
    assert isinstance(w._build_strategy(), BarrierTet3DTorchStrategy)


# ---------------------------------------------------------------------------
# auto-generated per-strategy parameter panel
# ---------------------------------------------------------------------------


def test_strategy_params_introspection():
    from dvfopt import SLPStrategy
    from dvfopt_gui.strategy_params import editable_fields, strategy_class_for

    assert strategy_class_for('slp') is SLPStrategy
    assert strategy_class_for('auto') is None
    fields = {name: (kind, default) for name, kind, default in editable_fields(SLPStrategy)}
    assert fields['accuracy'][0] == 'choice'
    assert 'time_budget_s' not in fields


def test_strategy_params_excludes_supports_3d_and_2d_windowed():
    from dvfopt import SLPStrategy
    from dvfopt_gui.strategy_params import editable_fields, strategy_class_for

    names = {name for name, _k, _d in editable_fields(SLPStrategy)}
    assert 'supports_3d' not in names
    assert strategy_class_for('slsqp_windowed') is None  # 2D: ignored -> honest
    assert strategy_class_for('slsqp_windowed@jdet3d') is not None
    assert strategy_class_for('barrier@jdet3d') is not None


def test_worker_applies_strategy_overrides():
    from dvfopt import SLPStrategy

    w = SolverWorker(
        deformation_i=np.zeros((3, 1, 6, 6)),
        method_id='slp_2tri',
        params={'strategy_overrides': {'cluster_pixel_threshold': 123}},
    )
    strat = w._build_strategy()
    assert isinstance(strat, SLPStrategy)
    assert strat.cluster_pixel_threshold == 123


def test_worker_bad_override_raises():
    w = SolverWorker(
        deformation_i=np.zeros((3, 1, 6, 6)),
        method_id='slp_2tri',
        params={'strategy_overrides': {'no_such_field': 1}},
    )
    with pytest.raises(ValueError):
        w._build_strategy()
