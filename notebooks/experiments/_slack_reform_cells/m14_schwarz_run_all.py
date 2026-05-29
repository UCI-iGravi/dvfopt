rows = []
for label, phi, kw in CASES:
    print(f'\n=== {label} ===')
    for method_label, fn in [
        ('m14_global',  iterative_2d_tri_refine_repair),
        ('m14_schwarz', m14_schwarz),
    ]:
        r = run_one(method_label, fn, phi, **kw)
        r['case'] = label
        rows.append(r)
        extras = ''
        if 'fallback' in r:
            extras = f"  fallback={r['fallback']}  clusters={r['n_clusters']}"
        tag = 'OK' if r['feasible'] else ('ERR' if r['error'] else 'FAIL')
        print(f'  [{tag:>4}] {method_label:<14}  wall={r["wall_s"]:>7.2f}s  '
              f'n_neg={r["n_neg"]:>5}  min_T={r["min_T"]:+.4f}  '
              f'L1={r["L1"]:>9.1f}  L2={r["L2"]:>8.2f}'
              + extras)
