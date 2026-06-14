# B0039 z=200 crops: a small (slack-reform runs) and a larger (skipped).
for sz in [12, 20]:
    phi = crop_b0039(z=200, cy=140, cx=200, sz=sz)
    all_rows.append(
        (f'B0039 z=200 crop {sz}x{sz}',
         report_case(f'B0039 z=200 crop {sz}x{sz}', phi)))
