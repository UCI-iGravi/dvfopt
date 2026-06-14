import pandas as pd
df = pd.DataFrame(rows)
df_summary = df.pivot_table(index='case', columns='method',
                              values=['wall_s', 'L1', 'L2', 'feasible'],
                              aggfunc='first')
print('=== Speedup + L1 ratio ===')
for label, _, _ in CASES:
    g = next((r for r in rows if r['case'] == label
              and r['method'] == 'm14_global'), None)
    s = next((r for r in rows if r['case'] == label
              and r['method'] == 'm14_schwarz'), None)
    if g is None or s is None:
        continue
    sp = g['wall_s'] / s['wall_s'] if s['wall_s'] > 0 else float('inf')
    l1_ratio = s['L1'] / g['L1'] if g['L1'] > 0 else float('nan')
    fb = '(fallback)' if s.get('fallback') else ''
    print(f'  {label:<22}  speedup={sp:>5.2f}x  L1 ratio={l1_ratio:>5.2f}x  {fb}')
df
