def make_synth(H, W, scale, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(0, scale, (H, W)),
                     rng.normal(0, scale, (H, W))])

def crop_b0039(z, cy, cx, sz=40):
    arr = np.load('../../data/dvfs/b0039/b0039_laplacian_deformation_field.npy')
    dy = arr[1, z, cy:cy+sz, cx:cx+sz].astype(np.float64).copy()
    dx = arr[2, z, cy:cy+sz, cx:cx+sz].astype(np.float64).copy()
    return np.stack([dy, dx])

# Slack-reform scales O(QP^3) in (n_phi + n_T) and gets very slow above
# ~16x16 (an earlier execution of this notebook timed out at 30 min on a
# 20x20 case). Run it only when the problem is small enough to complete
# in tens of seconds, and skip it with a clear note otherwise.
SLACK_REFORM_MAX_PIXELS = 12 * 12   # = 144

def report_case(name, phi_in):
    T1, T2 = _triangle_areas_2d(phi_in[0], phi_in[1])
    n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    min_T = float(min(T1.min(), T2.min()))
    H, W = phi_in.shape[1], phi_in.shape[2]
    pixels = H * W
    print(f'\n=== {name}  ({H}x{W}, init n_neg={n_neg}, init min={min_T:+.3f}) ===')
    rows = []
    runners = [
        ('baseline-slsqp', lambda p: baseline_slsqp(
            p.copy(), anchor='l1', max_iter=200, warm_max_iter=800)),
        ('slack-reform',   lambda p: slack_reform_slsqp(
            p.copy(), anchor='l1', max_iter=80, warm_max_iter=240, verbose=0)),
        ('m10 (ref)',      lambda p: reference_m10(p.copy(), anchor='l1')),
    ]
    for label, fn in runners:
        if label == 'slack-reform' and pixels > SLACK_REFORM_MAX_PIXELS:
            print(f'  {label:<16}  SKIPPED ({pixels} pixels > {SLACK_REFORM_MAX_PIXELS} cap)')
            rows.append(dict(method=label, wall_s=float('nan'),
                              n_neg=float('nan'), min_T=float('nan'),
                              L1=float('nan'), L2=float('nan'),
                              skipped=True))
            continue
        try:
            phi_out, info = fn(phi_in)
            costs = l_cost(phi_out, phi_in)
            row = dict(method=label,
                       wall_s=info.get('total_t', float('nan')),
                       n_neg=info.get('final_n_neg', float('nan')),
                       min_T=info.get('final_min', float('nan')),
                       L1=costs['L1'], L2=costs['L2'])
            rows.append(row)
            print(f'  {label:<16}  '
                  f'wall={row["wall_s"]:6.2f}s  '
                  f'n_neg={row["n_neg"]:5d}  '
                  f'min_T={row["min_T"]:+.4f}  '
                  f'L1={row["L1"]:8.2f}  L2={row["L2"]:7.3f}')
        except Exception as e:
            print(f'  {label:<16}  FAILED: {type(e).__name__}: {e}')
            rows.append(dict(method=label, wall_s=float('nan'),
                              n_neg=float('nan'), min_T=float('nan'),
                              L1=float('nan'), L2=float('nan'),
                              error=str(e)))
    return rows

all_rows = []
