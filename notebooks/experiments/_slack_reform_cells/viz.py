def _min_T_map(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return np.minimum(T1, T2)

# Use the dense 12x12 case — small enough that slack-reform completes
# while still being the regime where baseline struggles.
phi = make_synth(12, 12, scale=0.5, seed=7)
phi_base, info_base   = baseline_slsqp(phi.copy(), anchor='l1', max_iter=200,
                                       warm_max_iter=800)
phi_slack, info_slack = slack_reform_slsqp(phi.copy(), anchor='l1',
                                            max_iter=80, warm_max_iter=240,
                                            verbose=0)
phi_m10, info_m10     = reference_m10(phi.copy(), anchor='l1')

maps = [_min_T_map(phi), _min_T_map(phi_base), _min_T_map(phi_slack), _min_T_map(phi_m10)]
titles = [
    f'init  (n_neg={int((maps[0] <= 0).sum())}, min={maps[0].min():+.3f})',
    f'baseline-slsqp  ({info_base["total_t"]:.2f}s, '
    f'n_neg={info_base["final_n_neg"]}, min={info_base["final_min"]:+.4f})',
    f'slack-reform  ({info_slack["total_t"]:.2f}s, '
    f'n_neg={info_slack["final_n_neg"]}, min={info_slack["final_min"]:+.4f})',
    f'm10  ({info_m10["total_t"]:.2f}s, '
    f'n_neg={info_m10["final_n_neg"]}, min={info_m10["final_min"]:+.4f})',
]
vmax = max(0.05, abs(maps[0].min()))
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
for ax, m, t in zip(axes, maps, titles):
    im = ax.imshow(m, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85)
plt.suptitle('min(T1, T2) per cell  —  blue = fold')
plt.show()
