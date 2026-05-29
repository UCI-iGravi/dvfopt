"""Common test harness for the wall-breaker experiments.

Goal: a uniform contract that every candidate method satisfies, so the
benchmark suite can compare them on the same inputs with the same metrics.

Method contract
---------------
A method module exposes::

    NAME : str                    # short, used as table column
    DESCRIPTION : str             # one-line summary
    def solve(phi_in: np.ndarray, threshold: float,
              **kwargs) -> dict:
        # phi_in: (2, H, W) array; channels [dy, dx]; reference frame is
        # pixel ids (so warped grid is ref + phi).
        # Returns: {'phi_out': (2,H,W) ndarray, 'info': {...}}.
        # The info dict is free-form and is preserved verbatim in the
        # CSV alongside the standard metrics this harness computes.

Test cases
----------
Two flavours, both drawn from the real volume the manuscript run failed on:

* ``slice_full``  -- the full (320 x 456) slice. Whatever the method
  does, it must reduce to a (2, H, W) corrected slice. Frozen edge is
  NOT required: a method that changes the whole field is fair game (we
  measure feasibility AND L2 distance from input separately).
* ``worst_crop``  -- the bounding box of the slice's largest fold
  component plus a pad of 4 cells. This is the same crop the SLSQP /
  barrier / continuation probes operated on (~15 x 24 cells, 1472
  triangles), useful for cheap iteration before a full-slice run.

The benchmark suite reports for each (method, fixture):

* ``tri_neg``, ``tri_min``      -- 2-triangle (manuscript) metric
* ``sho_neg``, ``sho_min``      -- shoelace forward-diff metric
* ``jdet_neg``, ``jdet_min``    -- central-diff Jdet metric
* ``l2_delta``                  -- ``||phi_out - phi_in||_2`` (size of correction)
* ``feasible_2tri``             -- ``tri_neg == 0 and tri_min >= threshold - 1e-4``
* ``wall_s``                    -- wall-clock seconds
* ``error``                     -- exception message if the method blew up
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from dvfopt.jacobian.shoelace import _shoelace_areas_2d
from dvfopt.jacobian.numpy_jdet import jacobian_det2D

THRESHOLD = 0.01                       # manuscript threshold
WORST_SLICES = (11, 12, 13, 14, 16, 17)  # the 6 hard slices from the main run
DATA_PATH = os.path.join(
    _REPO, 'data', 'corrected_correspondences_count_touching',
    'registered_output', 'deformation3d.npy')


# ---------------------------------------------------------------- fixtures

def load_volume(path: str | None = None, mmap: bool = False) -> np.ndarray:
    """Returns the (3, D, H, W) raw deformation volume from the given
    path. Supports both ``.npy`` and ``.npz`` (uses key 'arr' or the
    first array in the file). Always casts to float64.

    Use ``mmap=True`` to avoid materialising the whole volume in memory.
    For .npy this returns a memmap; for .npz this returns a lazy-loaded
    NpzFile-backed array whose slices are read on demand.
    """
    import warnings as _warnings
    p = path or DATA_PATH
    if p.endswith('.npz'):
        if mmap:
            _warnings.warn(
                f'load_volume(mmap=True) is ignored for .npz files '
                f'({p!r}); NPZ is zlib-compressed and decompresses fully '
                f'into memory. Convert to .npy for true mmap support.',
                RuntimeWarning, stacklevel=2)
        z = np.load(p)
        key = 'arr' if 'arr' in z.files else z.files[0]
        v = z[key]
    else:
        v = np.load(p, mmap_mode=('r' if mmap else None))
    if v.dtype != np.float64:
        v = v.astype(np.float64)
    return v


def load_slice_2d(path: str, z: int) -> np.ndarray:
    """Memory-frugal: returns the ``(2, H, W)`` ``[dy, dx]`` slice ``z``
    without materialising the full (3, D, H, W) volume. Designed for
    parallel workers so each only pays slice-sized memory.
    """
    if path.endswith('.npz'):
        with np.load(path) as nz:
            key = 'arr' if 'arr' in nz.files else nz.files[0]
            full = nz[key]
            phi = np.stack([full[1, z].astype(np.float64),
                            full[2, z].astype(np.float64)])
    else:
        full = np.load(path, mmap_mode='r')
        phi = np.stack([np.asarray(full[1, z]).astype(np.float64),
                        np.asarray(full[2, z]).astype(np.float64)])
    return phi


def get_slice(phi_full: np.ndarray, z: int) -> np.ndarray:
    """``(2, H, W)`` slice ordered ``[dy, dx]`` (channels 1, 2 of the volume)."""
    return np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])


def get_worst_component_crop(phi2: np.ndarray, pad: int = 4) -> tuple[np.ndarray, tuple]:
    """Bounding-box crop of the largest folded connected component.

    Returns ``(phi_crop, (y0, y1, x0, x1))`` -- the slicing window so a
    method can re-embed if needed.
    """
    from scipy.ndimage import label as cc_label, binary_dilation, find_objects
    H, W = phi2.shape[1], phi2.shape[2]
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    fold = np.minimum(T1, T2) <= 0
    labels, _ = cc_label(binary_dilation(fold, iterations=1))
    comps = []
    for sl in find_objects(labels):
        if sl is not None:
            comps.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    if not comps:
        return phi2.copy(), (0, H, 0, W)
    comps.sort(key=lambda c: (c[1] - c[0]) * (c[3] - c[2]), reverse=True)
    cy0, cy1, cx0, cx1 = comps[0]
    y0 = max(0, cy0 - pad); y1 = min(H, cy1 + pad)
    x0 = max(0, cx0 - pad); x1 = min(W, cx1 + pad)
    return phi2[:, y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


# ---------------------------------------------------------------- metrics

def _as_2hw(phi: np.ndarray) -> np.ndarray:
    """Normalise input to ``(2, H, W)`` ``[dy, dx]``.

    Accepts:
      * ``(2, H, W)`` -- passthrough
      * ``(3, 1, H, W)`` -- manuscript ``[dz, dy, dx]`` 3D-with-singleton-z
      * ``(3, H, W)``    -- ``[dz, dy, dx]`` with no singleton z
    Raises ``ValueError`` for any other shape.
    """
    if phi.ndim == 3 and phi.shape[0] == 2:
        return phi
    if phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] == 1:
        return np.stack([phi[1, 0], phi[2, 0]])
    if phi.ndim == 3 and phi.shape[0] == 3:
        return np.stack([phi[1], phi[2]])
    raise ValueError(
        f'expected shape (2,H,W), (3,1,H,W), or (3,H,W); got {phi.shape}')


def metrics(phi2: np.ndarray) -> dict:
    """All three feasibility metrics + per-cell scalars.

    Accepts any DVF shape supported by ``_as_2hw`` (see that function).
    """
    phi2 = _as_2hw(phi2)
    T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
    tri = np.minimum(T1, T2)
    sho = _shoelace_areas_2d(phi2[0], phi2[1])
    jdet = np.squeeze(jacobian_det2D(phi2))
    return dict(
        tri_neg=int((tri <= 0).sum()),
        tri_min=float(tri.min()),
        tri_p1=float(np.percentile(tri, 1.0)),
        sho_neg=int((sho <= 0).sum()),
        sho_min=float(sho.min()),
        jdet_neg=int((jdet <= 0).sum()),
        jdet_min=float(jdet.min()),
    )


def is_2tri_feasible(m: dict, threshold: float = THRESHOLD,
                     tol: float = 1e-4) -> bool:
    """Strict-feasibility check.

    Returns True iff *both*:
      * ``tri_neg == 0``       (no folded cells), AND
      * ``tri_min >= threshold - tol``

    The ``tol`` slack matches the manuscript's numerical safety margin --
    ``tri_min = 0.00990`` still counts as feasible because solvers binding
    exactly at threshold can land 1e-5 below it due to scipy convergence
    tolerances. Pass ``tol=0`` for strict ``tri_min >= threshold``.
    """
    return m['tri_neg'] == 0 and m['tri_min'] >= threshold - tol


def is_fold_free(m: dict) -> bool:
    """``tri_neg == 0`` -- no folded cells, regardless of margin above threshold."""
    return m['tri_neg'] == 0


def l2_delta(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    return float(np.linalg.norm((phi_a - phi_b).ravel()))


def l1_delta(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    return float(np.abs(phi_a - phi_b).sum())


# ---------------------------------------------------------------- runner

@dataclass
class MethodResult:
    method: str
    fixture: str
    z: int
    shape: tuple
    threshold: float
    wall_s: float
    init: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)
    l2_delta: float = 0.0
    feasible_2tri: bool = False
    error: Optional[str] = None
    info: dict = field(default_factory=dict)
    phi_out: Optional[np.ndarray] = None    # transient; not serialised to JSON
    l1_delta: float = 0.0

    def to_row(self) -> dict:
        H, W = self.shape[1], self.shape[2]
        n_pixels = H * W                  # corners (= dy,dx pairs)
        n_entries = 2 * H * W             # scalars (dy AND dx)
        l2_per_entry = self.l2_delta / np.sqrt(n_entries) if n_entries else None
        l2_per_pixel = self.l2_delta / np.sqrt(n_pixels) if n_pixels else None
        l1_per_entry = self.l1_delta / n_entries if n_entries else None
        l1_per_pixel = self.l1_delta / n_pixels if n_pixels else None
        row = {
            'method': self.method, 'fixture': self.fixture, 'z': self.z,
            'H': self.shape[1], 'W': self.shape[2],
            'wall_s': round(self.wall_s, 2),
            'init_tri_neg': self.init.get('tri_neg'),
            'init_tri_min': self.init.get('tri_min'),
            'final_tri_neg': self.final.get('tri_neg'),
            'final_tri_min': self.final.get('tri_min'),
            'final_sho_neg': self.final.get('sho_neg'),
            'final_sho_min': self.final.get('sho_min'),
            'final_jdet_neg': self.final.get('jdet_neg'),
            'final_jdet_min': self.final.get('jdet_min'),
            'l2_delta': self.l2_delta,
            'l1_delta': self.l1_delta,
            'l2_per_entry': l2_per_entry,
            'l2_per_pixel': l2_per_pixel,
            'l1_per_entry': l1_per_entry,
            'l1_per_pixel': l1_per_pixel,
            'feasible_2tri': self.feasible_2tri,
            'error': self.error,
        }
        # Flatten select info fields if simple-typed.
        for k, v in self.info.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f'info_{k}'] = v
        return row


def run_method(method_mod, phi_in: np.ndarray, *, fixture: str, z: int,
               threshold: float = THRESHOLD, **kwargs) -> MethodResult:
    """Run one method on one fixture, capturing timing / errors / metrics."""
    init = metrics(phi_in)
    res = MethodResult(
        method=getattr(method_mod, 'NAME', method_mod.__name__),
        fixture=fixture, z=z, shape=tuple(phi_in.shape),
        threshold=threshold, wall_s=0.0, init=init)

    t0 = time.time()
    try:
        out = method_mod.solve(phi_in, threshold=threshold, **kwargs)
        phi_out = np.asarray(out['phi_out'])
        info = out.get('info', {}) or {}
    except Exception as exc:
        res.wall_s = time.time() - t0
        res.error = f'{type(exc).__name__}: {exc}'
        res.info = {'traceback': traceback.format_exc()[-800:]}
        return res

    res.wall_s = time.time() - t0
    if phi_out.shape != phi_in.shape:
        res.error = f'shape mismatch: in {phi_in.shape}, out {phi_out.shape}'
        res.info = info
        return res

    # Bug #3 guard: methods that overflow / NaN-trap (m05, m06, m11 on some
    # inputs) used to silently return non-finite phi_out which then poisoned
    # every aggregation downstream. Detect and surface as an error.
    n_bad = int(np.sum(~np.isfinite(phi_out)))
    if n_bad > 0:
        res.error = f'non-finite values in phi_out ({n_bad} entries)'
        res.info = info
        return res

    res.final = metrics(phi_out)
    res.l2_delta = l2_delta(phi_out, phi_in)
    res.l1_delta = l1_delta(phi_out, phi_in)
    res.feasible_2tri = is_2tri_feasible(res.final, threshold)
    res.info = info
    res.phi_out = phi_out
    return res


# ---------------------------------------------------------------- I/O

def save_result(result: MethodResult, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir,
                      f'{result.method}__{result.fixture}__z{result.z:03d}.json')
    with open(fn, 'w') as f:
        json.dump({
            'method': result.method, 'fixture': result.fixture, 'z': result.z,
            'shape': list(result.shape), 'threshold': result.threshold,
            'wall_s': result.wall_s, 'init': result.init, 'final': result.final,
            'l2_delta': result.l2_delta,
            'feasible_2tri': result.feasible_2tri,
            'error': result.error, 'info': result.info,
        }, f, indent=2, default=_json_default)
    return fn


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def fmt_row(result: MethodResult) -> str:
    """One-line summary suitable for console / log."""
    feas = 'FEAS' if result.feasible_2tri else 'fail'
    if result.error:
        return (f'{result.method:>22s}  {result.fixture:>10s}  z={result.z:3d}  '
                f'ERROR ({result.wall_s:6.1f}s): {result.error[:60]}')
    f = result.final
    return (f'{result.method:>22s}  {result.fixture:>10s}  z={result.z:3d}  '
            f'tri n_neg={f["tri_neg"]:5d}  min={f["tri_min"]:+.5f}  '
            f'jdet min={f["jdet_min"]:+.4f}  '
            f'L2={result.l2_delta:7.2f}  '
            f'({result.wall_s:6.1f}s)  {feas}')
