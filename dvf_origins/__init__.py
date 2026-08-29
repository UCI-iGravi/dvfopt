"""Standalone harness: sample DVFs grouped by fold-origin mechanism.

Generates the paper's §4 cases — one field per (mechanism, tool, severity) —
so the same fold-morphology table can be computed over all of them. NOT part
of the ``dvfopt`` package: it imports ``dvfopt`` only as a library (Laplacian
solve, fold metrics). Run from the repo root::

    python -m dvf_origins list
    python -m dvf_origins generate            # -> data/origins/<case>.npy + .json
    python -m dvf_origins sweep               # -> output/origins/<ts>/results.csv
    pytest dvf_origins                        # self-check

Every builder returns ``(phi, meta)`` with ``phi`` a ``(3, 1, H, W)`` float64
array, channels ``[dz, dy, dx]``, ``dz == 0``, pull-back convention, voxel
units — i.e. exactly what ``dvfopt`` consumes.
"""

from dvf_origins import real, registered, synthetic

MECHANISMS = {
    1: 'interpolation of sparse correspondences',
    2: 'dense weakly-regularized optimization',
    3: 'learned displacement field',
    4: 'discretized diffeomorphic warp',
}

# name -> (mechanism, builder, kwargs). Add a row to add a case.
# Builders raise FileNotFoundError / ImportError when their data or optional
# dependency is absent; ``generate`` skips those and says why.
CASES = {
    # -- 1: Laplacian interpolation of sparse correspondences ----------------
    'm1_interp_clean': (1, synthetic.interp_sparse, dict(seed=0)),
    'm1_interp_outliers': (1, synthetic.interp_sparse, dict(seed=0, outlier_frac=0.08)),
    'm1_interp_collapse': (1, synthetic.interp_sparse, dict(seed=0, n_collapse=4)),
    'm1_interp_mixed': (
        1,
        synthetic.interp_sparse,
        dict(seed=0, outlier_frac=0.05, n_collapse=2, jitter=1.0),
    ),
    'm1_laplacian_B0039_z264': (1, real.laplacian_slice, dict(brain='B0039', z=264)),
    # -- 2: dense optimization, regularization as the dial -------------------
    'm2_tvl1_weak': (2, synthetic.dense_weak_reg, dict(seed=0, attachment=200)),
    'm2_tvl1_strong': (2, synthetic.dense_weak_reg, dict(seed=0, attachment=15)),
    'm2_ilk_small_radius': (2, synthetic.dense_weak_reg, dict(seed=0, method='ilk', radius=3)),
    'm2_demons_brain': (2, registered.demons, dict(sigma=0.5)),
    'm2_demons_brain_smooth': (2, registered.demons, dict(sigma=2.0)),
    'm2_ffd_brain_aggressive': (2, registered.bspline_ffd, dict(mesh=24)),
    'm2_ffd_brain_coarse': (2, registered.bspline_ffd, dict(mesh=6)),
    'm2_tvl1_brain': (2, registered.tvl1, dict(attachment=60)),
    # -- 3: learned displacement fields --------------------------------------
    'm3_learned_proxy': (3, synthetic.learned_proxy, dict(seed=0, noise_amp=1.0)),
    'm3_learned_proxy_mild': (3, synthetic.learned_proxy, dict(seed=0, noise_amp=0.5)),
    # drop VoxelMorph / TransMorph notebook outputs here (needs torch to make)
    'm3_voxelmorph_saved': (3, real.saved_field, dict(path='data/origins/external/voxelmorph.npy')),
    'm3_transmorph_saved': (3, real.saved_field, dict(path='data/origins/external/transmorph.npy')),
    # -- 4: diffeomorphic in the continuum, folds only from discretization ---
    'm4_svf_decimated': (4, synthetic.diffeo_discretized, dict(seed=0, n_steps=6, decimate=2)),
    'm4_svf_subpixel': (
        4,
        synthetic.diffeo_discretized,
        dict(seed=0, svf_sigma=10, svf_max=60, n_steps=6, decimate=1),
    ),
    'm4_svf_coarse_steps': (4, synthetic.diffeo_discretized, dict(seed=0, n_steps=1, decimate=1)),
    'm4_ants_B0039_z264': (4, real.ants_slice, dict(brain='B0039', z=264)),
}


def build(name):
    """Build one registered case: ``(phi (3,1,H,W), meta)``."""
    mech, fn, kw = CASES[name]
    phi, meta = fn(**kw)
    return phi, {'case': name, 'mechanism': mech, 'mechanism_name': MECHANISMS[mech], **meta}
