from matplotlib.patches import Rectangle

phi_viz = load_b0039(140, 160, 60)
T1, T2 = _triangle_areas_2d(phi_viz[0], phi_viz[1])
min_T = np.minimum(T1, T2)

bboxes, fold_mask = _fold_clusters(phi_viz, merge_dilation=2)
n_clusters = len(bboxes)

# Run m14-schwarz with viz pad=4 for the boxes.
phi_after, hist = m14_schwarz(phi_viz.copy(), threshold=THRESHOLD,
                                anchor='l1', pad=4, merge_dilation=2,
                                verbose=0, record_history=True)
T1a, T2a = _triangle_areas_2d(phi_after[0], phi_after[1])
min_T_after = np.minimum(T1a, T2a)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

vm = max(abs(min_T.min()), 0.05)
im0 = axes[0].imshow(min_T, cmap='RdBu_r', vmin=-vm, vmax=vm)
axes[0].set_title(f'before: n_folds={int(fold_mask.sum())}', fontsize=10)
axes[0].set_xticks([]); axes[0].set_yticks([])
fig.colorbar(im0, ax=axes[0], shrink=0.85)

# Cluster labels.
grouped = binary_dilation(fold_mask, iterations=2)
labels_arr, _ = cc_label(grouped, structure=generate_binary_structure(2, 2))
axes[1].imshow(labels_arr, cmap='tab20')
for b in bboxes:
    H, W = phi_viz.shape[1], phi_viz.shape[2]
    pad = 4
    y0 = max(0, b['cy0'] - pad)
    y1 = min(H, b['cy1'] + pad + 2)
    x0 = max(0, b['cx0'] - pad)
    x1 = min(W, b['cx1'] + pad + 2)
    rect = Rectangle((x0 - 0.5, y0 - 0.5), (x1 - x0), (y1 - y0),
                     fill=False, edgecolor='red', linewidth=1.5)
    axes[1].add_patch(rect)
axes[1].set_title(f'{n_clusters} clusters + bounding boxes', fontsize=10)
axes[1].set_xticks([]); axes[1].set_yticks([])

vm_a = max(abs(min_T_after.min()), 0.05)
im2 = axes[2].imshow(min_T_after, cmap='RdBu_r', vmin=-vm_a, vmax=vm_a)
axes[2].set_title(f'after m14-Schwarz: n_folds={int((min_T_after<=0).sum())}',
                  fontsize=10)
axes[2].set_xticks([]); axes[2].set_yticks([])
fig.colorbar(im2, ax=axes[2], shrink=0.85)

plt.suptitle('B0039 z=12 60x60 crop: cluster decomposition')
plt.show()
