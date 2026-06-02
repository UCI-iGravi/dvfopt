"""Runnable demo — load a sample DVF and launch the live-viz window.

Usage::

    python -m dvfopt_gui.demo                  # auto-pick a small case
    python -m dvfopt_gui.demo --b0039 12       # B0039 z=12 (dense extreme)
    python -m dvfopt_gui.demo --canonical 03d  # canonical synthetic
    python -m dvfopt_gui.demo --b0039 100 --max-iter 20

The demo wires the windowed-SLSQP path (``iterative_serial``) into the
live-viz window. Other solvers don't currently expose the
``step_callback`` hook — see :mod:`dvfopt_gui.worker` docstring for
the contract to add.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_B0039_PATH = _REPO_ROOT / 'data' / 'dvfs' / 'b0039' / 'b0039_laplacian_deformation_field.npy'


def _b0039_slice(z: int) -> np.ndarray:
    """Load slice ``z`` of the B0039 Laplacian DVF as ``(3, 1, H, W)``."""
    if not _B0039_PATH.exists():
        raise FileNotFoundError(
            f'B0039 DVF not found at {_B0039_PATH}; either install it or use --canonical instead.'
        )
    phi_volume = np.load(_B0039_PATH).astype(np.float64)
    if not (0 <= z < phi_volume.shape[1]):
        raise IndexError(f'z={z} out of range [0, {phi_volume.shape[1]})')
    phi_2d = phi_volume[1:, z]  # (2, H, W)
    out = np.zeros((3, 1, *phi_2d.shape[1:]), dtype=np.float64)
    out[1, 0] = phi_2d[0]
    out[2, 0] = phi_2d[1]
    return out


def _canonical_case(key: str) -> np.ndarray:
    from test_cases import canonical_2tri_2d

    for name, phi, _meta in canonical_2tri_2d():
        if key in name:
            phi2 = phi.astype(np.float64).copy()
            out = np.zeros((3, 1, *phi2.shape[1:]), dtype=np.float64)
            out[1, 0] = phi2[0]
            out[2, 0] = phi2[1]
            return out
    raise KeyError(
        f'no canonical case matches {key!r}; available keys include 01a, 01b, 03a, 03b, 03c, 03d.'
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '--b0039', type=int, metavar='Z', help='Load slice z of the B0039 Laplacian DVF.'
    )
    g.add_argument(
        '--canonical',
        type=str,
        default=None,
        metavar='KEY',
        help='Load a canonical_2tri_2d case (e.g. 03d).',
    )
    p.add_argument(
        '--max-iter',
        type=int,
        default=None,
        help='max_iterations for the windowed SLSQP path (default: solver default).',
    )
    p.add_argument(
        '--max-per-index-iter',
        type=int,
        default=None,
        help='max sub-iterations per pixel (default: solver default).',
    )
    p.add_argument(
        '--method', type=str, default='SLSQP', help='scipy.optimize method name (default: SLSQP).'
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.b0039 is not None:
        print(f'Loading B0039 z={args.b0039}…', flush=True)
        deformation_i = _b0039_slice(args.b0039)
    elif args.canonical is not None:
        print(f'Loading canonical {args.canonical}…', flush=True)
        deformation_i = _canonical_case(args.canonical)
    else:
        # The canonical "bowtie" crossing — 10x10 grid with two
        # opposing correspondence points that swap diagonally. Small
        # enough that every view mode renders instantly, but the
        # crossing produces ~16 cells with at least one flipped
        # triangle, so the deformation-grid view shows a clear red
        # bowtie of folded cells out of the gate.
        print('Loading default canonical 01a_10x10_crossing (bowtie)…', flush=True)
        deformation_i = _canonical_case('01a')

    print(f'  shape: {deformation_i.shape}', flush=True)

    solver_kwargs = {'method_name': args.method}
    if args.max_iter is not None:
        solver_kwargs['max_iterations'] = args.max_iter
    if args.max_per_index_iter is not None:
        solver_kwargs['max_per_index_iter'] = args.max_per_index_iter

    try:
        from dvfopt_gui.app import launch
    except ImportError as exc:
        print(
            f'Failed to import GUI: {exc}\nInstall the GUI extras: `pip install -e \'.[gui]\'`',
            file=sys.stderr,
        )
        return 2
    return launch(deformation_i, solver_kwargs=solver_kwargs)


if __name__ == '__main__':
    sys.exit(main())
