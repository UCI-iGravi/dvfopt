"""Shared default parameters and constants for 2D and 3D optimisation."""

DEFAULT_PARAMS = {
    "threshold": 0.01,
    "err_tol": 1e-5,
    "max_iterations": 10000,
    "max_per_index_iter": 200,
    "max_minimize_iter": 1000,
}


def _log(verbose, level, msg):
    """Print *msg* if *verbose* >= *level*."""
    if verbose >= level:
        print(msg)


def _resolve_params(**overrides):
    """Merge *overrides* into ``DEFAULT_PARAMS``, returning resolved dict."""
    p = dict(DEFAULT_PARAMS)
    for name, val in overrides.items():
        if val is not None:
            p[name] = val
    return p


def _unpack_size(submatrix_size):
    """Normalize *submatrix_size* to a ``(sy, sx)`` tuple.

    Accepts a single int (square) or a 2-tuple/list (rectangular).
    """
    if isinstance(submatrix_size, (tuple, list)):
        return int(submatrix_size[0]), int(submatrix_size[1])
    return int(submatrix_size), int(submatrix_size)


def _unpack_size_3d(subvolume_size):
    """Normalize *subvolume_size* to a ``(sz, sy, sx)`` tuple.

    Accepts a single int (cubic) or a 3-tuple/list (rectangular).
    """
    if isinstance(subvolume_size, (tuple, list)):
        if len(subvolume_size) == 3:
            return int(subvolume_size[0]), int(subvolume_size[1]), int(subvolume_size[2])
        raise ValueError(
            f"_unpack_size_3d expects int or 3-tuple, got length-{len(subvolume_size)} sequence"
        )
    return int(subvolume_size), int(subvolume_size), int(subvolume_size)
