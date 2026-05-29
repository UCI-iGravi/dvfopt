# Larger synthetic dense — slack-reform skipped (above pixel cap); shows
# whether the relative L1/L2 ranking from the small grids extends to
# sizes where baseline starts to struggle.
for size, scale in [((16, 16), 0.35), ((20, 20), 0.3)]:
    H, W = size
    phi = make_synth(H, W, scale=scale, seed=11)
    all_rows.append(
        (f'synth {H}x{W} dense (scale={scale})',
         report_case(f'synth {H}x{W} dense', phi)))
