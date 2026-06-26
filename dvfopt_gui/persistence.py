"""Pure, headless-testable save/load logic for GUI solver runs.

The GUI's Save writes a compressed ``.npz`` capturing the current DVF
plus the full per-step optimization history; Load can read those same
archives back so a saved run can be re-scrubbed in the history slider.

Keeping the (de)serialisation here — free of any Qt / widget state —
makes it unit-testable without an event loop and keeps
:class:`dvfopt_gui.app.LiveSolverWindow` focused on UI wiring.

NPZ schema (all keys present unless noted)
------------------------------------------
* ``phi`` — ``(2, H, W)`` float64, the current (possibly corrected)
  field for the active z-slice.
* ``phi_full_volume`` — ``(3, D, H, W)`` float64, the full volume
  (with the ``dz`` channel). Provenance for multi-slice datasets.
* ``phi_input_volume`` — ``(3, D, H, W)`` float64, the *original* loaded
  volume before any correction (optional; absent for archives written by
  older versions). Restored as the window's pristine baseline so Revert
  and a fresh Run after loading operate on the true input, not the
  already-corrected ``phi_full_volume``.
* ``z`` — 0-d int, the active slice index.
* ``constraint``, ``method``, ``objective`` — 0-d strings (the dropdown
  selections at save time).
* ``time_budget_s``, ``max_iterations`` — 0-d float/int.
* ``history_max_size`` — 0-d int, the cap that bounded the run's buffer.
* ``final_min_jdet``, ``final_n_neg_jdet`` — 0-d, fold stats of ``phi``.
* ``n_history_steps`` — 0-d int, number of retained snapshots.

When at least one snapshot was retained, also:

* ``history_phi`` — ``(N, 2, H, W)`` float64, every snapshot's phi.
* ``history_n_neg``, ``history_min_T`` — ``(N,)`` fold-count / worst-area.
* ``history_outer_iter``, ``history_per_index_iter`` — ``(N,)`` int
  solver bookkeeping (mostly meaningful for SLSQP-windowed).
* ``history_total`` — 0-d int, total snapshots ever emitted (may exceed
  ``n_history_steps`` if older entries aged out of the cap).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dvfopt_gui.worker import StateSnapshot


def build_save_payload(
    *,
    phi_active: np.ndarray,
    full_volume: np.ndarray,
    z: int,
    constraint: str,
    method: str,
    objective: str,
    time_budget_s: float,
    max_iterations: int,
    history_max_size: int,
    history_snaps: list[StateSnapshot],
    history_total: int,
    input_volume: np.ndarray | None = None,
) -> dict:
    """Assemble the NPZ payload dict from plain values + a snapshot list.

    ``phi_active`` is the ``(2, H, W)`` field for the active slice;
    ``history_snaps`` is the worker's retained snapshots (oldest first,
    possibly empty). ``input_volume`` is the original ``(3, D, H, W)``
    field before correction — saved as ``phi_input_volume`` so a reloaded
    run restores its true baseline; pass ``None`` to omit it. All Qt
    access happens in the caller — this function sees only numpy +
    builtins so it can be tested headlessly.
    """
    phi_active = np.asarray(phi_active, dtype=np.float64)
    payload: dict = {
        'phi': phi_active,
        'phi_full_volume': np.asarray(full_volume, dtype=np.float64),
        'z': np.int64(z),
        'constraint': np.asarray(constraint or ''),
        'method': np.asarray(method or ''),
        'objective': np.asarray(objective or ''),
        'time_budget_s': np.float64(time_budget_s),
        'max_iterations': np.int64(max_iterations),
        'history_max_size': np.int64(history_max_size),
    }
    if input_volume is not None:
        payload['phi_input_volume'] = np.asarray(input_volume, dtype=np.float64)

    # Per-slice fold stats, computed fresh on save.
    from dvfopt.jacobian.numpy_jdet import jacobian_det2D

    jac = jacobian_det2D(phi_active)[0]
    payload['final_min_jdet'] = np.float64(jac.min())
    payload['final_n_neg_jdet'] = np.int64((jac < 0).sum())

    n = len(history_snaps)
    if n > 0:
        H, W = phi_active.shape[1:]
        phi_hist = np.empty((n, 2, H, W), dtype=np.float64)
        n_neg_arr = np.empty(n, dtype=np.int64)
        min_T_arr = np.empty(n, dtype=np.float64)
        outer_arr = np.empty(n, dtype=np.int64)
        sub_arr = np.empty(n, dtype=np.int64)
        for i, snap in enumerate(history_snaps):
            phi_hist[i] = snap.phi
            n_neg_arr[i] = snap.n_neg
            min_T_arr[i] = snap.min_T
            outer_arr[i] = snap.outer_iter
            sub_arr[i] = snap.per_index_iter
        payload['n_history_steps'] = np.int64(n)
        payload['history_phi'] = phi_hist
        payload['history_n_neg'] = n_neg_arr
        payload['history_min_T'] = min_T_arr
        payload['history_outer_iter'] = outer_arr
        payload['history_per_index_iter'] = sub_arr
        payload['history_total'] = np.int64(history_total)
    else:
        payload['n_history_steps'] = np.int64(0)

    return payload


@dataclass
class LoadedRun:
    """Result of parsing a loaded NPZ/NPY archive.

    ``volume`` is always present (normalised ``(3, D, H, W)``). The
    history fields are populated only when the archive carried a saved
    run (``snapshots`` non-empty); a bare ``.npy`` DVF yields an empty
    history.
    """

    volume: np.ndarray
    z: int = 0
    snapshots: list[StateSnapshot] = field(default_factory=list)
    history_total: int = 0
    history_max_size: int | None = None
    constraint: str | None = None
    method: str | None = None
    objective: str | None = None
    # Original pre-correction volume, when the archive carried one
    # (``phi_input_volume``). None for bare DVFs / older archives.
    input_volume: np.ndarray | None = None


def normalise_to_volume(arr: np.ndarray) -> np.ndarray:
    """Accept any of ``(2, H, W)``, ``(3, H, W)``, ``(3, 1, H, W)``,
    ``(3, D, H, W)`` and return a ``(3, D, H, W)`` float64 volume.

    Raises ``ValueError`` on any other shape.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 2:
        H, W = arr.shape[1:]
        vol = np.zeros((3, 1, H, W), dtype=np.float64)
        vol[1, 0] = arr[0]
        vol[2, 0] = arr[1]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        vol = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[0] == 3:
        vol = arr
    else:
        raise ValueError(
            f'expected (2,H,W), (3,H,W), (3,1,H,W), or (3,D,H,W); got {arr.shape}'
        )
    return vol.astype(np.float64)


def _as_str(val) -> str | None:
    """Decode a 0-d numpy string/bytes scalar to a Python str (or None)."""
    if val is None:
        return None
    s = val.item() if isinstance(val, np.ndarray) else val
    if isinstance(s, bytes):
        s = s.decode('utf-8', 'replace')
    s = str(s)
    return s or None


def parse_loaded(mapping) -> LoadedRun:
    """Parse an ``np.load`` result (an ``NpzFile`` or a bare ndarray)
    into a :class:`LoadedRun`.

    For NPZ archives the field is taken from the ``phi_full_volume`` key
    (a full saved run) if present, else ``phi``, else the first array.
    When ``history_phi`` is present the per-step snapshots are
    reconstructed so the GUI can re-scrub a saved run.
    """
    is_npz = isinstance(mapping, np.lib.npyio.NpzFile)
    if not is_npz:
        return LoadedRun(volume=normalise_to_volume(mapping))

    files = set(mapping.files)
    if 'phi_full_volume' in files:
        vol_src = mapping['phi_full_volume']
    elif 'phi' in files:
        vol_src = mapping['phi']
    else:
        vol_src = mapping[mapping.files[0]]
    volume = normalise_to_volume(vol_src)

    input_volume = (
        normalise_to_volume(mapping['phi_input_volume'])
        if 'phi_input_volume' in files
        else None
    )

    z = int(mapping['z']) if 'z' in files else 0
    z = max(0, min(volume.shape[1] - 1, z))

    snapshots: list[StateSnapshot] = []
    history_total = 0
    if 'history_phi' in files:
        phi_hist = np.asarray(mapping['history_phi'], dtype=np.float64)
        n = phi_hist.shape[0]

        def _col(key, default, dtype):
            if key in files:
                return np.asarray(mapping[key], dtype=dtype)
            return np.full(n, default, dtype=dtype)

        n_neg = _col('history_n_neg', 0, np.int64)
        min_T = _col('history_min_T', 0.0, np.float64)
        outer = _col('history_outer_iter', 0, np.int64)
        sub = _col('history_per_index_iter', 0, np.int64)
        for i in range(n):
            snapshots.append(
                StateSnapshot(
                    phi=phi_hist[i].copy(),
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
                    per_index_iter=int(sub[i]),
                    outer_iter=int(outer[i]),
                    n_neg=int(n_neg[i]),
                    min_T=float(min_T[i]),
                )
            )
        history_total = int(mapping['history_total']) if 'history_total' in files else n

    return LoadedRun(
        volume=volume,
        z=z,
        input_volume=input_volume,
        snapshots=snapshots,
        history_total=history_total,
        history_max_size=(int(mapping['history_max_size']) if 'history_max_size' in files else None),
        constraint=_as_str(mapping['constraint']) if 'constraint' in files else None,
        method=_as_str(mapping['method']) if 'method' in files else None,
        objective=_as_str(mapping['objective']) if 'objective' in files else None,
    )
