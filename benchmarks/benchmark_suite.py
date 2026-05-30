"""Generic benchmark harness.

Replace the hand-written ``_run_*.py`` scripts under ``benchmarks/``
with declarative :class:`BenchmarkSuite` constructions:

    suite = BenchmarkSuite(
        cases={
            '20x20_synth': synth_field(20, 20, scale=0.3),
            'b0039_z12':   load_b0039_z12_slice(),
            ...
        },
        solvers={
            'barrier_l1':     Solver.from_spec(constraint='2tri', objective='l1',
                                                strategy='barrier', shape=...),
            'm14_l1':         Solver.from_spec(... 'm14'),
            ...
        },
        out_csv='benchmarks/results/my_benchmark.csv',
    )
    df = suite.run()
    print(suite.summary(df))

The suite:

* Skips already-feasible cases (configurable).
* Captures errors per row rather than aborting.
* Writes CSV row-by-row (line-buffered) so partial progress is visible
  in long runs.
* Returns a pandas DataFrame for downstream analysis.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dvfopt import Solver
from dvfopt._defaults import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


@dataclass
class BenchmarkSuite:
    """Run every (case, solver) pair, report a flat result frame.

    Parameters
    ----------
    cases : dict[str, ndarray]
        Map case label -> input field. The field's leading-channel
        layout must match what each ``Solver``'s constraint expects
        (``(2, H, W)`` for 2D 2-tri; ``(3, D, H, W)`` for 3D Jdet).
    solvers : dict[str, Solver]
        Map solver label -> configured Solver instance.
    threshold : float, optional
        Feasibility lower bound used for the headline columns
        (``init_n_neg``, ``feasible``). Defaults to
        ``DEFAULT_PARAMS['threshold']`` and is independent of any
        per-Solver threshold (the suite reports against this for
        cross-solver comparability).
    skip_feasible : bool
        Skip (case, solver) pairs where the case is already feasible
        at ``threshold``. Default True.
    out_csv : str, optional
        If set, write CSV rows as they complete.
    verbose : bool
    """

    cases: dict
    solvers: dict
    threshold: Optional[float] = None
    skip_feasible: bool = True
    out_csv: Optional[str] = None
    verbose: bool = True
    extra_columns: dict = field(default_factory=dict)  # static columns per row

    def __post_init__(self):
        if self.threshold is None:
            self.threshold = DEFAULT_PARAMS['threshold']

    # ----------------------------- stats helpers -----------------------
    @staticmethod
    def _stats_2tri(phi):
        """Per-cell min(T1, T2) — used for the headline init/final cols."""
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        return int((np.minimum(T1, T2) <= 0).sum()), float(min(T1.min(), T2.min()))

    # ----------------------------- run ---------------------------------
    def _solve_one(self, solver: Solver, phi_in: np.ndarray):
        """Run one Solver on one case, capturing exceptions + wall."""
        buf = io.StringIO()
        t0 = time.perf_counter()
        err = ''
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                res = solver.fit(phi_in.copy(), verbose=0)
            except Exception as exc:
                wall = time.perf_counter() - t0
                err = f'{type(exc).__name__}: {exc}'
                return None, wall, err
        wall = time.perf_counter() - t0
        return res, wall, err

    def run(self):
        """Run all (case, solver) pairs and return a pandas DataFrame."""
        import pandas as pd

        rows = []
        if self.out_csv:
            os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        # We collect every column key we see and write the union as the
        # CSV header, so users can add free-form ``extra_columns`` without
        # the header drifting per row.
        column_order = [
            'case',
            'solver',
            'wall_s',
            'init_n_neg',
            'init_min_T',
            'final_n_neg',
            'final_min_T',
            'L1',
            'L2',
            'feasible',
            'error',
        ]
        column_order.extend(self.extra_columns)

        f = open(self.out_csv, 'a', buffering=1) if self.out_csv else None  # noqa: SIM115 — closed in `finally`
        try:
            if f and os.path.getsize(self.out_csv) == 0:
                f.write(','.join(column_order) + '\n')

            for case_label, phi_in in self.cases.items():
                init_n_neg, init_min = self._stats_2tri(phi_in)
                if self.skip_feasible and init_n_neg == 0:
                    if self.verbose:
                        print(f'[skip] {case_label}: already feasible (init n_neg=0)', flush=True)
                    continue
                if self.verbose:
                    print(
                        f'\n=== {case_label}  (init n_neg={init_n_neg}  min={init_min:+.4f}) ===',
                        flush=True,
                    )

                for solver_label, solver in self.solvers.items():
                    res, wall, err = self._solve_one(solver, phi_in)
                    if res is None:
                        row = dict(
                            case=case_label,
                            solver=solver_label,
                            wall_s=wall,
                            init_n_neg=init_n_neg,
                            init_min_T=init_min,
                            final_n_neg=-1,
                            final_min_T=float('nan'),
                            L1=float('nan'),
                            L2=float('nan'),
                            feasible=False,
                            error=err,
                        )
                    else:
                        diff = (res.corrected - phi_in).ravel()
                        row = dict(
                            case=case_label,
                            solver=solver_label,
                            wall_s=wall,
                            init_n_neg=init_n_neg,
                            init_min_T=init_min,
                            final_n_neg=res.final_n_neg,
                            final_min_T=res.final_min_T,
                            L1=float(np.abs(diff).sum()),
                            L2=float(np.sqrt(np.dot(diff, diff))),
                            feasible=res.feasible,
                            error='',
                        )
                    row.update(self.extra_columns)
                    rows.append(row)
                    if f:
                        vals = [str(row.get(c, '')) for c in column_order]
                        f.write(','.join(vals) + '\n')
                    if self.verbose:
                        tag = ' OK ' if row['feasible'] else ('ERR' if row['error'] else 'FAIL')
                        extra = f'  err={row["error"]}' if row["error"] else ''
                        print(
                            f'  [{tag:>4}] {solver_label:<18}  '
                            f'wall={row["wall_s"]:>7.2f}s  '
                            f'n_neg={row["final_n_neg"]:>5}  '
                            f'min_T={row["final_min_T"]:+.4f}  '
                            f'L1={row["L1"]:>9.2f}  '
                            f'L2={row["L2"]:>8.3f}' + extra,
                            flush=True,
                        )
        finally:
            if f:
                f.close()
        return pd.DataFrame(rows)

    # ----------------------------- summary -----------------------------
    @staticmethod
    def summary(df) -> str:
        """Per-solver aggregate (feasibility rate, avg wall, avg L1/L2)."""
        lines = [
            f'\n{"solver":<22} {"feas_rate":>10} {"avg_wall":>10} {"avg_L1":>10} {"avg_L2":>10}'
        ]
        lines.append('-' * 70)
        for solver, sub in df.groupby('solver'):
            total = len(sub)
            ok = sub['feasible'].sum()
            avg_wall = sub['wall_s'].mean()
            ok_sub = sub[sub['feasible']]
            avg_l1 = ok_sub['L1'].mean() if len(ok_sub) else float('nan')
            avg_l2 = ok_sub['L2'].mean() if len(ok_sub) else float('nan')
            lines.append(
                f'{solver:<22} {ok}/{total:<8} {avg_wall:>9.2f}s {avg_l1:>10.2f} {avg_l2:>10.3f}'
            )
        return '\n'.join(lines)


__all__ = ['BenchmarkSuite']
