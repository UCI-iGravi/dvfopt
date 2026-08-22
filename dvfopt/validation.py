"""Input validation for DVFopt.

All package entry points (``Solver.fit``, ``DVFopt.fit``,
``correct_dvf``, ``Constraint.coerce``) route their input through
:func:`validate_dvf`. The goal is to surface bad inputs *at the
boundary* with actionable messages, instead of letting them propagate
into the solver and produce cryptic numpy errors 5 frames deep.

The validator handles:

1. **Array-like coercion** — lists, tuples, masked arrays, anything
   exposing ``__array__`` are converted via :func:`numpy.asarray`. A
   helpful :class:`TypeError` is raised otherwise.
2. **dtype** — silently up-promotes to ``float64`` (numerical solvers
   need it). Lower-precision inputs go through unchanged in value.
3. **Shape** — enforces the channel layout for 2D vs 3D constraints
   and surfaces what's expected when the user passes something else.
4. **Finite values** — rejects NaN/Inf with a count, so the user knows
   their data has holes.
5. **Minimum spatial size** — H, W (and D for 3D) must be ≥ 3 so the
   ``(H-1)·(W-1)`` cell grid has at least one constraint to enforce.

The validator is intentionally strict by default; flags exist for
the small number of legitimate edge cases (3-channel 2D input from
the laplacian module, singleton-D 2D layouts).
"""

from __future__ import annotations

import numpy as np

from dvfopt.exceptions import SolverConfigError


def coerce_to_ndarray(
    phi,
    *,
    dtype=np.float64,
    name: str = 'phi',
) -> np.ndarray:
    """Convert ``phi`` to a writeable :class:`numpy.ndarray` of ``dtype``.

    Wraps :func:`numpy.asarray` with a clearer error message and a
    guarantee that the returned array is writeable (the solver mutates
    in place during inner loops).
    """
    try:
        arr = np.asarray(phi, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f'{name} must be convertible to a numpy array '
            f'(got {type(phi).__name__}); underlying error: {exc}'
        ) from exc
    # ``asarray`` may return a read-only view; copy if so.
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def validate_finite(phi: np.ndarray, *, name: str = 'phi') -> None:
    """Raise :class:`ValueError` if ``phi`` contains NaN or Inf.

    Reports the count of each so the caller knows whether they have
    a few stray pixels or systematic data corruption.
    """
    finite_mask = np.isfinite(phi)
    if not finite_mask.all():
        n_nan = int(np.isnan(phi).sum())
        n_inf = int((~np.isfinite(phi) & ~np.isnan(phi)).sum())
        raise ValueError(
            f'{name} contains non-finite values: {n_nan} NaN, {n_inf} Inf. '
            f'Numerical solvers cannot start from a non-finite iterate — '
            f'mask or replace these values before calling fit().'
        )


def validate_spatial_min_size(
    spatial_shape: tuple[int, ...],
    *,
    min_size: int = 3,
    name: str = 'phi',
) -> None:
    """Raise :class:`SolverConfigError` if any spatial axis is < ``min_size``."""
    axes = ['D', 'H', 'W'][-len(spatial_shape) :]
    for axis, n in zip(axes, spatial_shape):
        if n < min_size:
            raise SolverConfigError(
                f'{name} spatial dimension {axis}={n} is below the minimum '
                f'of {min_size}. The 2-triangle / Jdet constraint operates '
                f'on the (H-1)(W-1) cell grid, which is empty for {axis}<2; '
                f'use {axis}>={min_size} for a meaningful problem.'
            )


def validate_dvf(
    phi,
    *,
    dim: int,
    accept_3channel_2d: bool = True,
    accept_singleton_d_2d: bool = True,
    min_spatial_size: int = 3,
    require_finite: bool = True,
    name: str = 'phi',
) -> np.ndarray:
    """Validate + coerce a DVF input to canonical ``(C, *spatial)`` form.

    Parameters
    ----------
    phi
        The input. Anything :func:`numpy.asarray` can handle: ndarray,
        list-of-lists, masked array, PyTorch tensor (via ``__array__``),
        ...
    dim
        Spatial dimensionality (2 for ``H×W`` slices, 3 for ``D×H×W``).
        For 2D, returns ``(2, H, W)`` ``float64``. For 3D, returns
        ``(3, D, H, W)`` ``float64``.
    accept_3channel_2d
        For 2D, also accept ``(3, H, W)`` and ``(3, 1, H, W)`` layouts
        (the canonical output of the laplacian module includes a dz
        channel). The dz channel is dropped.
    accept_singleton_d_2d
        For 2D, accept ``(C, 1, H, W)`` (3D-like with unit D dimension).
        Drops the unit axis.
    min_spatial_size
        Minimum allowed value on each spatial axis. Default 3.
    require_finite
        If ``True`` (default), reject inputs containing NaN/Inf.
    name
        Variable name used in error messages.

    Returns
    -------
    ndarray
        Canonical ``(C, *spatial)`` ``float64`` array. Always a fresh
        writeable copy; mutating the return value does not alias the
        caller's input.

    Raises
    ------
    TypeError
        If ``phi`` isn't array-like.
    SolverConfigError
        If the shape/channel count doesn't match what was expected.
    ValueError
        If ``require_finite`` and the values contain NaN/Inf.
    """
    if dim not in (2, 3):
        raise SolverConfigError(f'dim must be 2 or 3, got {dim}')

    arr = coerce_to_ndarray(phi, dtype=np.float64, name=name)

    if dim == 2:
        arr = _coerce_2d(
            arr,
            accept_3channel=accept_3channel_2d,
            accept_singleton_d=accept_singleton_d_2d,
            name=name,
        )
    else:
        arr = _coerce_3d(arr, name=name)

    if require_finite:
        validate_finite(arr, name=name)
    validate_spatial_min_size(arr.shape[1:], min_size=min_spatial_size, name=name)
    return arr


# ---------------------------------------------------------------------------
# 2D / 3D shape coercion
# ---------------------------------------------------------------------------


_VALID_2D_SHAPES = (
    '(2, H, W)',
    '(3, H, W) — drops dz',
    '(2, 1, H, W)',
    '(3, 1, H, W) — drops dz',
)
_VALID_3D_SHAPES = ('(3, D, H, W)',)


def _coerce_2d(
    arr: np.ndarray,
    *,
    accept_3channel: bool,
    accept_singleton_d: bool,
    name: str,
) -> np.ndarray:
    """Massage a 2D-family input into canonical ``(2, H, W)``."""
    if arr.ndim == 3:
        if arr.shape[0] == 2:
            return arr
        if arr.shape[0] == 3 and accept_3channel:
            # (dz, dy, dx) -> drop dz, keep (dy, dx).
            return np.stack([arr[1], arr[2]])
    if arr.ndim == 4 and accept_singleton_d:
        # (C, 1, H, W) singleton-D form.
        if arr.shape[1] != 1:
            raise SolverConfigError(
                f'{name} has 4 dims but axis-1 size is {arr.shape[1]} '
                f'(must be 1 for a 2D singleton layout). Accepted shapes: '
                f'{", ".join(_VALID_2D_SHAPES)}'
            )
        if arr.shape[0] == 2:
            return arr[:, 0]
        if arr.shape[0] == 3 and accept_3channel:
            return np.stack([arr[1, 0], arr[2, 0]])
    raise SolverConfigError(
        f'{name} shape {arr.shape} is not a valid 2D DVF layout. '
        f'Accepted: {", ".join(_VALID_2D_SHAPES)}'
    )


def _coerce_3d(arr: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a 3D input is canonical ``(3, D, H, W)``."""
    if arr.ndim == 4 and arr.shape[0] == 3:
        return arr
    raise SolverConfigError(
        f'{name} shape {arr.shape} is not a valid 3D DVF layout. '
        f'Accepted: {", ".join(_VALID_3D_SHAPES)}'
    )


__all__ = [
    'coerce_to_ndarray',
    'validate_dvf',
    'validate_finite',
    'validate_spatial_min_size',
]
