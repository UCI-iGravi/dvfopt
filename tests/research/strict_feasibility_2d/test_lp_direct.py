import numpy as np

from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from research.strict_feasibility_2d.algorithms.lp_direct_2tri import (
    lp_oneshot,
    slp_iter,
)


def _bowtie_phi():
    phi = np.zeros((2, 7, 7), dtype=np.float64)
    phi[1, 3, 3] = +1.2
    phi[1, 3, 4] = -1.2
    return phi


def test_lp_oneshot_returns_phi_and_info():
    phi_in = _bowtie_phi()
    phi_out, info = lp_oneshot(phi_in, threshold=0.01)
    assert phi_out.shape == phi_in.shape
    assert phi_out.dtype == np.float64
    for k in ('seed', 'lp_status', 'L1_dev', 'final_min_T_exact', 'wall_s'):
        assert k in info, f'missing info key: {k}'


def test_lp_oneshot_strictly_improves_min_T_vs_input():
    phi_in = _bowtie_phi()
    T1_in, T2_in = _triangle_areas_2d(phi_in[0], phi_in[1])
    min_T_in = float(np.minimum(T1_in, T2_in).min())
    phi_out, info = lp_oneshot(phi_in, threshold=0.01)
    assert info['final_min_T_exact'] > min_T_in


def test_lp_oneshot_L1_is_smaller_than_harmonic_only():
    """LP should pull the seed back toward phi_in -- L1 must drop."""
    from dvfopt.core.wallbreakers import harmonic_extension_2d

    phi_in = _bowtie_phi()
    seed = harmonic_extension_2d(phi_in, threshold=0.01)
    seed_L1 = float(np.abs(seed - phi_in).sum())
    phi_out, info = lp_oneshot(phi_in, threshold=0.01)
    out_L1 = float(np.abs(phi_out - phi_in).sum())
    assert out_L1 <= seed_L1 + 1e-9, f'LP L1 {out_L1} > seed L1 {seed_L1}'


def test_slp_iter_returns_phi_and_info():
    phi_in = _bowtie_phi()
    phi_out, info = slp_iter(phi_in, threshold=0.01)
    assert phi_out.shape == phi_in.shape
    for k in ('iters', 'L1_dev', 'final_min_T_exact', 'converged', 'wall_s', 'trust_radius_final'):
        assert k in info, f'missing info key: {k}'


def test_slp_iter_strictly_feasible_on_bowtie():
    """The whole point: at termination, min(T1, T2) >= threshold - safety_tol."""
    phi_in = _bowtie_phi()
    phi_out, info = slp_iter(phi_in, threshold=0.01)
    assert info['final_min_T_exact'] >= 0.01 - 1e-5, info['final_min_T_exact']


def test_slp_iter_L1_le_lp_oneshot_L1():
    """Iteration should not increase L1 vs the one-shot baseline by much."""
    phi_in = _bowtie_phi()
    _, info_one = lp_oneshot(phi_in, threshold=0.01)
    _, info_slp = slp_iter(phi_in, threshold=0.01)
    # Tolerate a 5% gap for numerical reasons.
    assert info_slp['L1_dev'] <= info_one['L1_dev'] * 1.05 + 1e-6


def test_slp_iter_terminates_within_max_iter():
    phi_in = _bowtie_phi()
    _, info = slp_iter(phi_in, threshold=0.01, max_iter=20)
    assert info['iters'] <= 20
    assert info['converged'] or info['iters'] == 20


def test_slp_iter_pinned_reference_no_drift():
    """Determinism guard: slp_iter on a fixed seeded planted crop must
    reach strict feasibility with an L1 that matches a pinned reference.

    The reference L1 was computed once with the canonical implementation
    (seed='m14', threshold=0.01, all other knobs default) and inlined.
    Every stage of the pipeline is deterministic (no RNG in the solver),
    so any change to this value signals silent numerical drift in the
    linearisation / LP-step / seed path — investigate before repinning.
    """
    from dvfopt.core.primitives.tri import tri_areas_flat

    rng = np.random.default_rng(42)
    H, W = 12, 12
    phi_in = rng.normal(scale=0.05, size=(2, H, W))
    phi_in[0, 5, 5] = 1.3
    phi_in[0, 5, 6] = -1.3

    phi_out, info = slp_iter(phi_in, threshold=0.01, seed='m14')

    areas = tri_areas_flat(np.concatenate([phi_out[0].ravel(), phi_out[1].ravel()]), H, W)
    assert int((areas <= 0).sum()) == 0
    assert info['final_min_T_exact'] >= 0.01 - 1e-5

    L1_REF = 0.60869235592814497  # pinned 2026-07-09
    assert abs(info['L1_dev'] - L1_REF) <= 1e-9, (
        f"L1_dev {info['L1_dev']!r} drifted from pinned reference {L1_REF!r}"
    )
