# B0039 z=12 crops (the canonical hard case): a small (slack-reform
# runs) and a larger (skipped). The slack-reform's behavior on the
# small dense crop is the strongest signal we get on whether the
# reformulation helps on real data.
for sz in [12, 20]:
    phi = crop_b0039(z=12, cy=140, cx=200, sz=sz)
    all_rows.append(
        (f'B0039 z=12 crop {sz}x{sz}',
         report_case(f'B0039 z=12 crop {sz}x{sz}', phi)))
