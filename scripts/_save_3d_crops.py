"""Extract real-data 3D crops and save for test notebook."""
import os
import numpy as np
from dvfopt import jacobian_det3D

d = np.load('data/corrected_correspondences_count_touching/registered_output/deformation3d.npy')

out_dir = 'data/test_cases_3d'
os.makedirs(out_dir, exist_ok=True)

crops = {
    'slice350_5x10x10': {'z': (348, 353), 'worst_yx': (181, 207), 'pad': 5},
    'slice200_5x10x10': {'z': (198, 203), 'worst_yx': (106, 180), 'pad': 5},
    'slice090_5x10x10': {'z': (88, 93),   'worst_yx': (181, 273), 'pad': 5},
}

for name, cfg in crops.items():
    z0, z1 = cfg['z']
    wy, wx = cfg['worst_yx']
    pad = cfg['pad']
    sub = d[:, z0:z1, :, :]
    y0 = max(wy - pad, 0)
    y1 = min(wy + pad, sub.shape[2])
    x0 = max(wx - pad, 0)
    x1 = min(wx + pad, sub.shape[3])
    crop = sub[:, :, y0:y1, x0:x1]
    
    jdet = jacobian_det3D(crop)
    neg = int((jdet <= 0).sum())
    
    path = os.path.join(out_dir, f'{name}.npy')
    np.save(path, crop)
    print(f'Saved {path}: shape={crop.shape}, neg={neg}, min={jdet.min():.4f}')

print('\nDone.')
