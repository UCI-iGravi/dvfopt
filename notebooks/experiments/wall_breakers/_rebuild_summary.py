"""Rebuild summary.csv / summary.md from per-result JSONs in results/."""
import glob, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_all import write_summary

rows = []
for fn in sorted(glob.glob(os.path.join(_HERE, 'results', '*.json'))):
    with open(fn) as f:
        d = json.load(f)
    row = {
        'method': d.get('method'), 'fixture': d.get('fixture'),
        'z': d.get('z'),
        'H': d.get('shape', [0, 0, 0])[1], 'W': d.get('shape', [0, 0, 0])[2],
        'wall_s': round(d.get('wall_s') or 0, 2),
        'init_tri_neg': d.get('init', {}).get('tri_neg'),
        'init_tri_min': d.get('init', {}).get('tri_min'),
        'final_tri_neg': d.get('final', {}).get('tri_neg'),
        'final_tri_min': d.get('final', {}).get('tri_min'),
        'final_sho_neg': d.get('final', {}).get('sho_neg'),
        'final_sho_min': d.get('final', {}).get('sho_min'),
        'final_jdet_neg': d.get('final', {}).get('jdet_neg'),
        'final_jdet_min': d.get('final', {}).get('jdet_min'),
        'l2_delta': d.get('l2_delta', 0),
        'feasible_2tri': d.get('feasible_2tri', False),
        'error': d.get('error'),
    }
    rows.append(row)

csv_path, md_path = write_summary(rows, os.path.join(_HERE, 'results'))
print(f'rebuilt:\n  {csv_path}\n  {md_path}\n  ({len(rows)} rows)')
