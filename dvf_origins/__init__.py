"""Standalone harness: sample DVFs grouped by fold-origin mechanism.

Generates the paper's §4 cases — one field per (mechanism, tool, data, variant)
— so the same fold-morphology table can be computed over all of them. NOT part
of the ``dvfopt`` package (not installed; run from the repo root): it imports
``dvfopt`` only as a library (Laplacian solve, field I/O, fold metrics)::

    python -m dvf_origins list
    python -m dvf_origins generate   # -> data/origins/<mechanism dir>/<case>.npy + .json, manifest.json
    python -m dvf_origins sweep      # -> output/origins/<ts>/results.csv + results_latest.csv
    pytest dvf_origins               # self-check

Every builder returns ``(phi, meta)`` with ``phi`` a ``(3, 1, H, W)`` float64
array, channels ``[dz, dy, dx]``, ``dz == 0``, pull-back convention, voxel
units — i.e. exactly what ``dvfopt`` consumes.

Case names are ``m<k>_<tool>_<data>_<variant>``: mechanism, the tool that made
the field, the data it was made from (``synthetic`` = generated images or
pins, ``ellipses`` = the notebooks' toy images, ``brainpair`` = the in-repo
B0039/template slice pair, ``cohort`` = the 7-brain RegTools cohort) and the
variant, so a file is unambiguous when copied out of its directory.
"""

from dvf_origins import learned, real, registered, synthetic
from dvf_origins._common import ROOT, pack2d

__all__ = ['CASES', 'MECHANISMS', 'MECH_DIR', 'ROOT', 'build', 'case_dir', 'pack2d']

MECHANISMS = {
    1: 'interpolation of sparse correspondences',
    2: 'dense weakly-regularized optimization',
    3: 'learned displacement field',
    4: 'discretized diffeomorphic warp',
}
# one directory per mechanism under the generate/sweep root
MECH_DIR = {
    1: 'm1_interpolation',
    2: 'm2_dense_optimization',
    3: 'm3_learned',
    4: 'm4_diffeomorphic',
}

# name -> (mechanism, builder, kwargs). Add a row to add a case.
# Builders raise FileNotFoundError / ModuleNotFoundError when their data or
# optional dependency is absent; ``generate`` skips those and says why.
CASES = {
    # -- 1: Laplacian interpolation of sparse correspondences ----------------
    'm1_laplacian_synthetic_clean': (1, synthetic.interp_sparse, dict(seed=0)),
    'm1_laplacian_synthetic_outliers': (
        1,
        synthetic.interp_sparse,
        dict(seed=0, outlier_frac=0.08),
    ),
    'm1_laplacian_synthetic_collapse': (1, synthetic.interp_sparse, dict(seed=0, n_collapse=4)),
    'm1_laplacian_synthetic_mixed': (
        1,
        synthetic.interp_sparse,
        dict(seed=0, outlier_frac=0.05, n_collapse=2, jitter=1.0),
    ),
    'm1_laplacian_cohort_B0039_z264': (1, real.laplacian_slice, dict(brain='B0039', z=264)),
    # -- 2: dense optimization, regularization as the dial -------------------
    'm2_tvl1_synthetic_weak': (2, synthetic.dense_weak_reg, dict(seed=0, attachment=200)),
    'm2_tvl1_synthetic_strong': (2, synthetic.dense_weak_reg, dict(seed=0, attachment=15)),
    'm2_ilk_synthetic_r3': (2, synthetic.dense_weak_reg, dict(seed=0, method='ilk', radius=3)),
    'm2_demons_brainpair_weak': (2, registered.demons, dict(sigma=0.5)),
    'm2_demons_brainpair_smooth': (2, registered.demons, dict(sigma=2.0)),
    'm2_ffd_brainpair_fine': (2, registered.bspline_ffd, dict(mesh=24)),
    'm2_ffd_brainpair_coarse': (2, registered.bspline_ffd, dict(mesh=6)),
    'm2_tvl1_brainpair': (2, registered.tvl1, dict(attachment=60)),
    # -- 3: learned displacement fields --------------------------------------
    'm3_proxy_synthetic': (3, synthetic.learned_proxy, dict(seed=0, noise_amp=1.0)),
    'm3_proxy_synthetic_mild': (3, synthetic.learned_proxy, dict(seed=0, noise_amp=0.5)),
    # trained here on the notebooks' toy images (needs the torch venv — see learned.py);
    # direct = no diffeo layer
    'm3_voxelmorph_ellipses_direct': (3, learned.voxelmorph, dict(seed=0, integration_steps=0)),
    'm3_voxelmorph_ellipses_diffeo': (3, learned.voxelmorph, dict(seed=0, integration_steps=7)),
    'm3_transmorph_ellipses_direct': (3, learned.transmorph, dict(seed=0, integration_steps=0)),
    'm3_transmorph_ellipses_diffeo': (3, learned.transmorph, dict(seed=0, integration_steps=7)),
    # the same networks trained on REAL data: cohort brains affinely aligned onto the
    # template (RegTools outputs, external); test pair = B0039 at z=264, the plane of the
    # m1/m4 cohort rows — but on a x3-downsampled 96x128 grid, so compare fold FRACTIONS
    'm3_voxelmorph_cohort_direct': (
        3,
        learned.voxelmorph,
        dict(seed=0, integration_steps=0, data=learned.cohort_data),
    ),
    'm3_voxelmorph_cohort_diffeo': (
        3,
        learned.voxelmorph,
        dict(seed=0, integration_steps=7, data=learned.cohort_data),
    ),
    'm3_transmorph_cohort_direct': (
        3,
        learned.transmorph,
        dict(seed=0, integration_steps=0, data=learned.cohort_data),
    ),
    'm3_transmorph_cohort_diffeo': (
        3,
        learned.transmorph,
        dict(seed=0, integration_steps=7, data=learned.cohort_data),
    ),
    # or drop any saved learned field here (`real.saved_field`, any dvfopt-readable format)
    'm3_external_saved': (3, real.saved_field, dict(path='data/origins/external/learned.npy')),
    # -- 4: diffeomorphic in the continuum, folds only from discretization ---
    'm4_svf_synthetic_decimated': (
        4,
        synthetic.diffeo_discretized,
        dict(seed=0, n_steps=6, decimate=2),
    ),
    'm4_svf_synthetic_subpixel': (
        4,
        synthetic.diffeo_discretized,
        dict(seed=0, svf_sigma=10, svf_max=60, n_steps=6, decimate=1),
    ),
    'm4_svf_synthetic_coarse_steps': (
        4,
        synthetic.diffeo_discretized,
        dict(seed=0, n_steps=1, decimate=1),
    ),
    'm4_ants_cohort_B0039_z264': (4, real.ants_slice, dict(brain='B0039', z=264)),
}


def case_dir(name, root):
    """Directory a case's files live in: ``<root>/<MECH_DIR[mechanism]>``."""
    return root / MECH_DIR[CASES[name][0]]


def build(name):
    """Build one registered case: ``(phi (3,1,H,W), meta)``."""
    mech, fn, kw = CASES[name]
    phi, meta = fn(**kw)
    return phi, {'case': name, 'mechanism': mech, 'mechanism_name': MECHANISMS[mech], **meta}
