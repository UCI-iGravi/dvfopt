# Small synthetic cases (slack-reform within its 12x12 envelope).
for size in [(8, 8), (10, 10), (12, 12)]:
    for scale, tag in [(0.15, 'sparse'), (0.3, 'moderate'), (0.5, 'dense')]:
        H, W = size
        phi = make_synth(H, W, scale=scale, seed=7)
        all_rows.append(
            (f'synth {H}x{W} {tag} (scale={scale})',
             report_case(f'synth {H}x{W} {tag}', phi)))
