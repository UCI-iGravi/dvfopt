"""CLI: ``python -m dvf_origins {list, generate, sweep}`` (run from the repo root).

``generate`` -> ``<out>/<mechanism dir>/<case>.npy + .json`` and rebuilds
``<out>/manifest.json`` from the tree; ``sweep`` -> ``<out>/<timestamp>/results.csv``
and, for a sweep of the default input root (or ``--latest``), ``<out>/results_latest.csv``.
Layout and naming: ``dvf_origins/README.md``.
"""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from dvf_origins import CASES, MECHANISMS, ORIGINS, RENAMED, ROOT, build, case_dir
from dvf_origins.morphology import COLUMNS, morphology
from dvfopt.io.fields import load_dvf, save_dvf

OUT_ROOT = ROOT / 'output' / 'origins'
_W = max(map(len, CASES))  # case-name column width


def _write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def _case_files(root):
    """``{case: path}`` for every registered case whose field exists under ``root``."""
    return {n: f for n in CASES if (f := case_dir(n, root) / f'{n}.npy').is_file()}


def migrate(root):
    """Move fields saved under an old name (``RENAMED``; flat pre-layout root or a
    mechanism directory) to their current case path, recording ``renamed_from``."""
    moved = []
    for old, new in RENAMED.items():
        dst = case_dir(new, root) / f'{new}.npy'
        for src in [
            root / f'{old}.npy',
            *(root / d / f'{old}.npy' for d, _ in MECHANISMS.values()),
        ]:
            if not src.is_file() or dst.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            meta_src = src.with_suffix('.json')
            meta = json.loads(meta_src.read_text()) if meta_src.is_file() else {}
            meta.update(case=new, renamed_from=old)
            _write_atomic(dst.with_suffix('.json'), json.dumps(meta, indent=1, default=str))
            if meta_src.is_file():
                meta_src.unlink()
            moved.append(f'{old} -> {dst.parent.name}/{new}')
    if moved:
        print(f'migrated {len(moved)} field(s): {", ".join(moved)}')
    return moved


def write_manifest(root):
    """Rebuild ``<root>/manifest.json`` from the tree: case -> file, mechanism (the
    registry's), tool, source, shape, build time. A pure function of what is on disk,
    so concurrent or interrupted ``generate`` runs cannot leave it inconsistent."""
    manifest = {}
    for n, f in _case_files(root).items():
        mp = f.with_suffix('.json')
        meta = json.loads(mp.read_text()) if mp.is_file() else {}
        manifest[n] = dict(
            file=f'{f.parent.name}/{f.name}',
            mechanism=CASES[n][0],
            tool=meta.get('tool', ''),
            source=meta.get('source', ''),
            shape=list(np.load(f, mmap_mode='r').shape),
            build_s=meta.get('build_s', ''),
            built=meta.get(
                'built', time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(f.stat().st_mtime))
            ),
        )
    _write_atomic(root / 'manifest.json', json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def cmd_list(_a):
    for name, (mech, fn, kw) in CASES.items():
        print(f'{name:{_W}s} m{mech}  {fn.__module__.split(".")[-1]:10s} {fn.__name__:20s} {kw}')
    print()
    for k, (d, desc) in MECHANISMS.items():
        print(f'  m{k}: {desc}  ({d}/)')


def cmd_generate(a):
    out = Path(a.out)
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
    out.mkdir(parents=True, exist_ok=True)
    migrate(out)
    built, skipped, failed = [], [], []
    for n in names:
        d = case_dir(n, out)
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
        meta['built'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        d.mkdir(parents=True, exist_ok=True)
        save_dvf(d / f'{n}.npy', phi)
        _write_atomic(d / f'{n}.json', json.dumps(meta, indent=1, default=str))
        print(f'built {n}: {tuple(phi.shape[-2:])} in {meta["build_s"]:.1f}s -> {d.name}/')
        built.append(n)
    n_man = len(write_manifest(out))
    print(f'manifest: {n_man} cases on disk -> {out / "manifest.json"}')
    print(f'\n{len(built)} built, {len(skipped)} skipped, {len(failed)} failed -> {out}')
    if failed:
        raise SystemExit(1)


def cmd_sweep(a):
    src = Path(a.inp)
    if src.is_dir():
        migrate(src)
    files = _case_files(src)
    expected = {case_dir(n, src) / f'{n}.npy' for n in CASES}
    candidates = list(src.glob('*.npy'))  # the pre-layout flat files ...
    for d, _ in MECHANISMS.values():  # ... and anything in a mechanism directory
        candidates += (src / d).glob('*.npy')
    stale = sorted(f'{f.parent.name}/{f.name}' for f in candidates if f not in expected)
    if stale:
        print(f'ignoring {len(stale)} field(s) not at a registered case path: {", ".join(stale)}')
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
            **morphology(load_dvf(f), a.threshold),
        }
        rows.append(row)
        print(
            f'{row["case"]:{_W}s} m{row["mechanism"]} {row["H"]}x{row["W"]:<4d} '
            f'jdet<=0 {row["jdet_neg_px"]:6d}  simplex cells {row["simplex_neg_cells"]:6d}  '
            f'bilinear-only {row["bilinear_only_cells"]:5d}  clusters {row["n_clusters"]:5d} '
            f'(med {row["cluster_area_med"]:.0f} / max {row["cluster_area_max"]})  '
            f'min {row["simplex_min"]:.3g}'
        )
    write_manifest(src)
    out_root = Path(a.out)
    out = out_root / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    cols = ['case', 'mechanism', 'source', 'tool', 'dz_max_dropped', 'off_image_frac', *COLUMNS]
    with open(out / 'results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'\n{len(rows)} rows -> {out / "results.csv"}')
    # the stable path tracks full sweeps of the canonical tree only, so a scratch or
    # partial sweep cannot silently replace the table a paper build points at
    if a.latest or (a.latest is None and src.resolve() == ORIGINS.resolve()):
        latest = out_root / 'results_latest.csv'
        try:
            shutil.copyfile(out / 'results.csv', latest)
            print(f'{latest} <- {out.name}/results.csv')
        except OSError as e:  # e.g. open in Excel on Windows
            print(f'WARNING: could not update {latest} ({e}); it still holds the previous sweep')


def main(argv=None):
    p = argparse.ArgumentParser(prog='dvf_origins', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='print the case registry').set_defaults(fn=cmd_list)
    g = sub.add_parser('generate', help='build cases -> <out>/<mechanism>/<case>.npy + .json')
    g.add_argument('--out', default=str(ORIGINS))
    g.add_argument('--mechanism', type=int, nargs='+', help='only these mechanisms (1-4)')
    g.add_argument('--case', nargs='+', help='only these case names')
    g.set_defaults(fn=cmd_generate)
    s = sub.add_parser('sweep', help='fold-morphology table over generated fields')
    s.add_argument('--in', dest='inp', default=str(ORIGINS))
    s.add_argument('--out', default=str(OUT_ROOT))
    s.add_argument('--threshold', type=float, default=0.01)
    s.add_argument(
        '--latest',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='also write <out>/results_latest.csv (default: only when --in is the default root)',
    )
    s.set_defaults(fn=cmd_sweep)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
