import pandas as pd
flat = []
for case, rows in all_rows:
    for r in rows:
        flat.append({'case': case, **r})
df = pd.DataFrame(flat)
df
