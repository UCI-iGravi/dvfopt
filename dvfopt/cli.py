"""Command-line interface — ``dvfopt {info, correct, gui}``.

A thin argparse layer over the library: field I/O via :mod:`dvfopt.io`
(``.npy``/``.npz`` + NIfTI/MetaImage/NRRD), metrics via
:mod:`dvfopt.metrics`, correction via :func:`dvfopt.correct_dvf` and the
2.5D / 3D pipelines. Solver progress streams through the ``dvfopt``
logger (``-v``/``-vv``; ``--log-file`` tees records to a file).

Exit codes: 0 success (``correct``: strictly feasible output; ``info``:
feasible when ``--check``), 1 folds remain, 2 usage / data errors.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


def _parse_params(pairs) -> dict:
    """``['k=v', ...]`` -> dict with literal-eval'd values ('3'->3, 'x'->'x')."""
    out = {}
    for pair in pairs or []:
        key, sep, val = pair.partition('=')
        if not sep or not key:
            raise SystemExit(f'--param expects KEY=VALUE, got {pair!r}')
        try:
            out[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            out[key] = val
    return out


def _json_default(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _setup_logging(verbose: int, log_file) -> None:
    from dvfopt._logging import enable_default_handler, logger

    level = logging.DEBUG if verbose >= 2 else logging.INFO
    if verbose:
        enable_default_handler(level=level)
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)


def _cmd_info(args) -> int:
    from dvfopt._defaults import DEFAULT_PARAMS
    from dvfopt.io import load_dvf
    from dvfopt.metrics import constraint_fold_stats

    phi = load_dvf(args.input)
    name, st = constraint_fold_stats(phi, constraint=args.constraint, threshold=args.threshold)
    thr = DEFAULT_PARAMS['threshold'] if args.threshold is None else args.threshold
    report = {
        'input': str(args.input),
        'shape': list(phi.shape),
        'constraint': name,
        'threshold': thr,
        **asdict(st),
        'feasible': st.feasible,
    }
    bilinear_folded = False
    if args.ift:
        from dvfopt.metrics import injectivity_stats

        inj = injectivity_stats(phi)
        report['injectivity'] = asdict(inj)
        bilinear_folded = bool(inj.n_cells_nonpos)  # None (3D) and 0 are both clean
    print(json.dumps(report, indent=2, default=_json_default))
    return 1 if (args.check and (not st.feasible or bilinear_folded)) else 0


def _correct_slice(job):
    """Correct one 2D slice. Module-level, with picklable args and plain-value
    results, so it doubles as a ``ProcessPoolExecutor`` worker under spawn."""
    from dvfopt.core._pool import pin_worker_threads

    pin_worker_threads()  # no-op in the parent, so the serial path is untouched
    phi2, kwargs = job
    from dvfopt import correct_dvf

    res = correct_dvf(phi2, **kwargs)
    return res.corrected, res.feasible, res.init_n_neg, res.final_n_neg, res.wall_time


def _map_slices(jobs, n_workers):
    """Run the per-slice jobs in a process pool; results stay in slice order."""
    from concurrent.futures import ProcessPoolExecutor

    from dvfopt.core._pool import pinned_thread_env

    with pinned_thread_env(), ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_correct_slice, jobs))


def _correct_slices(phi, args, params):
    """Per-slice 2D solver sweep over a (3, D, H, W) volume (pre-2.5D step).

    Serial by default; ``--n-workers N`` (N > 1, more than one slice) solves the
    slices in a process pool — each slice is an independent solve. Inner solves
    stay serial there: :func:`dvfopt.core._pool.get_pool` refuses to nest pools,
    and :func:`dvfopt.core._pool.pin_worker_threads` gives each worker exactly
    one compute thread (so N workers use N cores, not N x every core).
    """
    # ponytail: plain loop; the DVFopt facade adds per-slice dataframes/plots
    # (and a torch import) — reach for it in Python when you want those.
    import numpy as np

    if phi.ndim != 4:
        raise ValueError(f'--pipeline slices needs a (3, D, H, W) volume, got {phi.shape}')
    h, w = phi.shape[-2:]
    kwargs = dict(
        constraint=args.constraint,
        objective=args.objective,
        strategy=args.strategy,
        shape=(h, w),
        verbose=args.verbose,
        **({} if args.threshold is None else {'threshold': args.threshold}),
        **params,
    )
    jobs = [(phi[:, z : z + 1], kwargs) for z in range(phi.shape[1])]
    n_workers = args.n_workers or 1
    results = (
        _map_slices(jobs, n_workers)
        if n_workers > 1 and len(jobs) > 1
        else [_correct_slice(job) for job in jobs]
    )

    outs, rows, all_ok = [], [], True
    for z, (corrected, feasible, init_n_neg, final_n_neg, wall_time) in enumerate(results):
        outs.append(corrected)
        all_ok &= feasible
        rows.append(
            {
                'z': z,
                'feasible': feasible,
                'init_n_neg': init_n_neg,
                'final_n_neg': final_n_neg,
                'wall_time_s': wall_time,
            }
        )
    out = np.concatenate(outs, axis=1)
    summary = {
        'pipeline': 'slices',
        'constraint': args.constraint,
        'objective': args.objective,
        'strategy': args.strategy,
        'feasible': all_ok,
        'n_slices': phi.shape[1],
        'final_n_neg': int(sum(r['final_n_neg'] for r in rows)),
        'per_slice': rows,
    }
    return out, all_ok, summary, None


def _cmd_correct(args) -> int:
    from dvfopt.io import load_dvf, save_dvf

    phi = load_dvf(args.input)
    params = _parse_params(args.param)
    common = {} if args.threshold is None else {'threshold': args.threshold}
    report_dir = Path(args.report_dir) if args.report_dir else None

    if args.pipeline == 'solver':
        from dvfopt import correct_dvf

        res = correct_dvf(
            phi,
            constraint=args.constraint,
            objective=args.objective,
            strategy=args.strategy,
            verbose=args.verbose,
            record_history=report_dir is not None,
            **common,
            **params,
        )
        out, feasible, solve_info = res.corrected, res.feasible, res.info
        summary = {
            'pipeline': 'solver',
            'constraint': args.constraint,
            'objective': args.objective,
            'strategy': args.strategy,
            'feasible': res.feasible,
            'init_n_neg': res.init_n_neg,
            'init_min_T': res.init_min_T,
            'final_n_neg': res.final_n_neg,
            'final_min_T': res.final_min_T,
            'wall_time_s': res.wall_time,
        }
    elif args.pipeline == 'slices':
        out, feasible, summary, solve_info = _correct_slices(phi, args, params)
    elif args.pipeline == '25d':
        from dvfopt import correct_dvf_25d

        out, rep = correct_dvf_25d(phi, verbose=args.verbose, **common, **params)
        feasible, solve_info = rep.feasible, None
        summary = {'pipeline': '25d', **asdict(rep)}
    else:  # '3d'
        from dvfopt import correct_dvf_3d

        out, rep = correct_dvf_3d(phi, verbose=args.verbose, **common, **params)
        feasible, solve_info = rep.feasible, None
        summary = {'pipeline': '3d', **asdict(rep)}

    save_dvf(args.output, out)
    summary.update(input=str(args.input), output=str(args.output))

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding='utf-8'
        )
        if solve_info is not None and getattr(solve_info, 'phases', None):
            import matplotlib

            matplotlib.use('Agg')
            from dvfopt.viz.solveinfo import plot_solve_info

            plot_solve_info(
                solve_info,
                threshold=args.threshold,
                save_path=str(report_dir / 'convergence.png'),
            )

    remaining = summary.get('final_n_neg', summary.get('n_neg_out', '?'))
    print(
        f"{'feasible' if feasible else 'NOT feasible'} ({remaining} folds remain) -> {args.output}"
    )
    return 0 if feasible else 1


def _cmd_gui(args) -> int:
    try:
        from dvfopt_gui.demo import main as gui_main
    except ImportError as exc:
        print(
            f'GUI extras not installed ({exc}). Install with: pip install -e ".[gui]"',
            file=sys.stderr,
        )
        return 2
    return gui_main(args.rest)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='dvfopt', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='command', required=True)

    def _common(sp):
        sp.add_argument(
            '-v', '--verbose', action='count', default=0, help='-v solver progress, -vv debug'
        )
        sp.add_argument('--log-file', default=None, help='also write dvfopt log records here')
        sp.add_argument(
            '--threshold', type=float, default=None, help='feasibility threshold (default 0.01)'
        )

    pi = sub.add_parser('info', help='fold metrics of a field (no correction)')
    pi.add_argument('input', help='.npy/.npz/.nii/.nii.gz/.mha/.mhd/.nrrd displacement field')
    pi.add_argument(
        '--constraint',
        default='auto',
        help="'auto' (simplex for 2D, simplex_3d for 3D) | simplex | simplex_standard | "
        'bilinear | finite | jdet | jdet_3d | simplex_3d (legacy labels 2tri, '
        '2tri_standard, 6tet, 6tet_3d still accepted)',
    )
    pi.add_argument('--check', action='store_true', help='exit 1 when not strictly feasible')
    pi.add_argument(
        '--ift',
        action='store_true',
        help='add sub-pixel injectivity diagnostics: IFT radius estimate '
        '(orientation-blind; saturates at its window cap) plus, in 2D, the exact '
        'bilinear cell min-Jdet certificate — with --check, nonpositive bilinear '
        'cells also exit 1. Memory-heavy on large 3D volumes (full-volume batched SVD).',
    )
    _common(pi)

    pc = sub.add_parser('correct', help='correct a field and save the result')
    pc.add_argument('input')
    pc.add_argument('output')
    pc.add_argument(
        '--pipeline',
        choices=('solver', 'slices', '25d', '3d'),
        default='solver',
        help='solver: one Solver run (2D slice, or a 3D constraint on a volume); '
        'slices: per-slice 2D sweep over a (3,D,H,W) volume; '
        '25d: marching fold prevention (needs dz==0); 3d: full 3D fold repair',
    )
    pc.add_argument(
        '--constraint',
        default='simplex',
        help="simplex | simplex_standard | bilinear | finite | jdet | jdet_3d | "
        'simplex_3d (default: simplex)',
    )
    pc.add_argument(
        '--objective',
        default='l1',
        help='l1 | l2 | none | auto (default: l1; auto = l2 on trap-heavy fields, none + polish elsewhere)',
    )
    pc.add_argument(
        '--strategy',
        default='auto',
        help="a strategy label, or 'auto' (default). auto routing, 2D: bilinear -> "
        "isqp_windowed at any objective; simplex_standard + 'none' -> isqp_windowed; "
        "simplex* + 'l1' -> slp; simplex* + 'l2' -> density-tiered "
        '(slsqp/barrier/m10); jdet and finite -> barrier when dense, isqp_windowed '
        'when mild. The isqp_windowed routes need osqp installed and fall back to '
        'the tier heuristic without it; 3D routing is unchanged. The measured robust '
        "0-fold 2D recipe is --constraint bilinear --strategy isqp_windowed "
        '--objective none (docs/recipe-2d-zero-folds.md)',
    )
    pc.add_argument(
        '--param',
        action='append',
        metavar='KEY=VALUE',
        default=[],
        help="extra strategy/pipeline kwarg (repeatable; values are literal-eval'd)",
    )
    pc.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='--pipeline slices: solve this many z-slices at once in worker '
        'processes (default: serial; inner solves stay serial — no nested pools). '
        'Keep it SMALL (2-4): the solves are memory-bandwidth bound, so throughput '
        'peaks well below the core count',
    )
    pc.add_argument(
        '--report-dir',
        default=None,
        help='write summary.json (+ convergence.png for --pipeline solver) here',
    )
    _common(pc)

    pg = sub.add_parser('gui', help='launch the live-solver GUI (needs the [gui] extras)')
    pg.add_argument('rest', nargs=argparse.REMAINDER, help='arguments forwarded to dvfopt_gui.demo')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != 'gui':
        _setup_logging(args.verbose, args.log_file)
    handlers = {'info': _cmd_info, 'correct': _cmd_correct, 'gui': _cmd_gui}
    try:
        return handlers[args.command](args)
    except (OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
