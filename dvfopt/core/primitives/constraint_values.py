"""Shared helper for evaluating constraint values on a 2D phi array.

This consolidates what used to live in (similar but not identical)
``_compute_constraint_2d`` copies in :mod:`dvfopt._plots` and
:mod:`dvfopt.unified`. The two call sites have slightly different
needs:

* **Plotting code** wants the result reshaped onto a ``(H-1, W-1)`` cell
  grid, so it needs ``T1 + T2`` only — *no* corner patches (they would
  break the reshape).
* **Stats / reporting code** wants the constraint vector that the
  solver actually sees, including corner patches under the default
  ``'2tri'`` (full-coverage) variant.

Hence one helper with an ``include_patches`` flag, rather than two
near-duplicates that drift apart over time.
"""

from __future__ import annotations

import numpy as np

from dvfopt.jacobian.numpy_jdet import jacobian_det2D
from dvfopt.jacobian.triangle_sign import (
    _corner_patch_areas_2d,
    _triangle_areas_2d,
)


def compute_constraint_values_2d(
    phi2: np.ndarray,
    kind: str,
    *,
    include_patches: bool = True,
) -> np.ndarray:
    """Return the constraint values as a flat ``(n_constraints,)`` ndarray.

    Parameters
    ----------
    phi2 : ndarray, shape ``(2, H, W)``, channels ``[dy, dx]``.
    kind : str
        One of ``'2tri'``, ``'2tri_standard'``, ``'jdet'``, ``'jdet_2d'``.
    include_patches : bool
        Only meaningful for ``kind == '2tri'``. When ``True`` (default),
        appends the two corner-patch triangle areas to match what the
        solver enforces under the full-coverage default. Set to
        ``False`` for plotting code that reshapes the output to a
        ``(H-1, W-1)`` grid (the patches would break the reshape).

    Returns
    -------
    ndarray, shape ``(n_constraints,)``.

    Raises
    ------
    ValueError
        If ``kind`` is not recognized.
    """
    if kind == '2tri':
        T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
        if include_patches:
            patches = _corner_patch_areas_2d(phi2[0], phi2[1])
            return np.concatenate([T1.ravel(), T2.ravel(), patches])
        return np.concatenate([T1.ravel(), T2.ravel()])
    if kind == '2tri_standard':
        T1, T2 = _triangle_areas_2d(phi2[0], phi2[1])
        return np.concatenate([T1.ravel(), T2.ravel()])
    if kind in ('jdet', 'jdet_2d'):
        return np.squeeze(jacobian_det2D(phi2)).ravel()
    raise ValueError(f'unknown constraint kind: {kind}')


__all__ = ['compute_constraint_values_2d']
