def synth_sparse(seed=0):
    np.random.seed(seed)
    H, W = 30, 30
    dy = np.random.normal(0, 0.05, (H, W))
    dx = np.random.normal(0, 0.05, (H, W))
    _plant_fold(dx, 5, 5)
    _plant_fold(dx, 5, 20)
    _plant_fold(dy, 22, 12)
    return np.stack([dy, dx])


CASES = [
    ('synth_30x30_sparse_8', synth_sparse(), dict()),
    ('z12_30x30_379',        load_b0039(120, 180, 30), dict()),
    ('z12_30x30_1484',       load_b0039(180, 180, 30), dict()),
    ('z12_60x60',            load_b0039(140, 160, 60), dict()),
    ('z12_full_320x456',     load_b0039_full(), dict(time_budget_s=900.0)),
]
for label, phi, _ in CASES:
    H, W = phi.shape[1], phi.shape[2]
    n_neg, min_T = _stats(phi)
    print(f'{label:<22} {H}x{W:<8}  init n_neg={n_neg:>5}  init min_T={min_T:>+8.3f}')
