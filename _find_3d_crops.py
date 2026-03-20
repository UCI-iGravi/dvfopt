"""Find crops from the real 3D deformation that have negative Jdet."""
import numpy as np
from modules.dvfopt3d import jacobian_det3D

d = np.load('data/corrected_correspondences_count_touching/registered_output/deformation3d.npy')
print(f'Full volume: {d.shape}')

for label, z0, z1 in [('slice350', 348, 353), ('slice200', 198, 203), ('slice090', 88, 93)]:
    sub = d[:, z0:z1, :, :]
    jdet = jacobian_det3D(sub)
    neg_indices = np.argwhere(jdet <= 0)
    if len(neg_indices) == 0:
        print(f'{label}: no neg Jdet')
        continue
    
    worst_idx = np.argmin(jdet)
    wz, wy, wx = np.unravel_index(worst_idx, jdet.shape)
    print(f'\n{label}: neg={len(neg_indices)}, worst=({wz},{wy},{wx}), min={jdet[wz,wy,wx]:.4f}')
    
    # Try different crop sizes
    for pad in [5, 8, 10]:
        y0 = max(wy - pad, 0)
        y1 = min(wy + pad, sub.shape[2])
        x0 = max(wx - pad, 0)
        x1 = min(wx + pad, sub.shape[3])
        crop = sub[:, :, y0:y1, x0:x1]
        jc = jacobian_det3D(crop)
        neg_c = int((jc <= 0).sum())
        print(f'  pad={pad}: crop {crop.shape[1:]}, neg={neg_c}/{jc.size}, min={jc.min():.4f}')
