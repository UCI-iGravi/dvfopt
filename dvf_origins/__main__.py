"""CLI: ``python -m dvf_origins {list, generate, sweep}`` (run from the repo root)."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from dvf_origins import CASES, MECHANISMS, ROOT, build
from dvf_origins.morphology import COLUMNS, morphology


def cmd_list(_a):
    for name, (mech, fn, kw) in CASES.items():
        print(f'{name:28s} m{mech}  {fn.__module__.split(".")[-1]:10s} {fn.__name__:20s} {kw}')
    print()
    for k, v in MECHANISMS.items():
        print(f'  m{k}: {v}')


def cmd_generate(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    unknown = [c for c in a.case or [] if c not in CASES]
    unknown += [f'm{m}' for m in a.mechanism or [] if m not in MECHANISMS]
    if unknown:
        print(f'unknown case / mechanism: {", ".join(unknown)} (see `python -m dvf_origins list`)')
        raise SystemExit(2)
    names = [
        n
        for n, (mech, _, _) in CASES.items()
        if (a.mechanism is None or mech in a.mechanism) and (a.case is None or n in a.case)
    ]
    built, skipped, failed = [], [], []
    for n in names:
        t0 = time.perf_counter()
        try:
            phi, meta = build(n)
        except (FileNotFoundError, ModuleNotFoundError) as e:  # data / optional dep absent
            skipped.append(n)
            print(f'skip  {n}: {e}')
            continue
        except Exception as e:  # keep going: the contract is "build what you can, say why not"
            failed.append(n)
            print(f'FAIL  {n}: {type(e).__name__}: {e}')
            continue
        meta['build_s'] = round(time.perf_counter() - t0, 2)
        np.save(out / f'{n}.npy', phi)
        (out / f'{n}.json').write_text(json.dumps(meta, indent=1, default=str))
        print(f'built {n}: {tuple(phi.shape[-2:])} in {meta["build_s"]:.1f}s')
        built.append(n)
    print(f'\n{len(built)} built, {len(skipped)} skipped, {len(failed)} failed -> {out}')
    if failed:
        raise SystemExit(1)


def cmd_sweep(a):
    src = Path(a.inp)
    files = {n: src / f'{n}.npy' for n in CASES if (src / f'{n}.npy').is_file()}
    stale = sorted(f.stem for f in src.glob('*.npy') if f.stem not in CASES)
    if stale:
        print(f'ignoring {len(stale)} field(s) with no CASES row: {", ".join(stale)}')
    if not files:
        raise SystemExit(f'no case fields under {src} — run `python -m dvf_origins generate` first')
    rows = []
    for name, f in files.items():
        mp = f.with_suffix('.json')
        meta = json.loads(mp.read_text()) if mp.is_file() else {}
        row = {
            'case': name,
            'mechanism': CASES[name][0],  # the registry, not the JSON, is the authority
            'source': meta.get('source', ''),
            'tool': meta.get('tool', ''),
            'dz_max_dropped': meta.get('dz_max_dropped', ''),
            'off_image_frac': meta.get('off_image_frac', ''),  # learned rows: 1.0 = collapsed
            **morphology(np.load(f), a.threshold),
        }
        rows.append(row)
        print(
            f'{row["case"]:28s} m{row["mechanism"]} {row["H"]}x{row["W"]:<4d} '
            f'jdet<=0 {row["jdet_neg_px"]:6d}  simplex cells {row["simplex_neg_cells"]:6d}  '
            f'bilinear-only {row["bilinear_only_cells"]:5d}  clusters {row["n_clusters"]:5d} '
            f'(med {row["cluster_area_med"]:.0f} / max {row["cluster_area_max"]})  '
            f'min {row["simplex_min"]:.3g}'
        )
    out = Path(a.out) / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    cols = ['case', 'mechanism', 'source', 'tool', 'dz_max_dropped', 'off_image_frac', *COLUMNS]
    with open(out / 'results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'\n{len(rows)} rows -> {out / "results.csv"}')


def main(argv=None):
    p = argparse.ArgumentParser(prog='dvf_origins', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='print the case registry').set_defaults(fn=cmd_list)
    g = sub.add_parser('generate', help='build cases -> <out>/<case>.npy + .json')
    g.add_argument('--out', default=str(ROOT / 'data' / 'origins'))
    g.add_argument('--mechanism', type=int, nargs='+', help='only these mechanisms (1-4)')
    g.add_argument('--case', nargs='+', help='only these case names')
    g.set_defaults(fn=cmd_generate)
    s = sub.add_parser('sweep', help='fold-morphology table over generated fields')
    s.add_argument('--in', dest='inp', default=str(ROOT / 'data' / 'origins'))
    s.add_argument('--out', default=str(ROOT / 'output' / 'origins'))
    s.add_argument('--threshold', type=float, default=0.01)
    s.set_defaults(fn=cmd_sweep)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
