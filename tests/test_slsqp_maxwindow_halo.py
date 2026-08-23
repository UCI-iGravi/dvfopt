"""Tests for the max-window patch constraint builder (halo no-damage rows).

The builder makes the max-window sub-problem's feasible set equal the
solver's paste-back accept criterion: Jdet is constrained over the window
AND the 1-voxel halo ring around it, evaluated on a context patch with
the same stencils the full-field accept check uses.
"""

import numpy as np
import pytest

from dvfopt.core.slsqp_windowed.constraints3d import _build_constraints_3d_maxwindow
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d, jacobian_det3D

RNG = np.random.default_rng(42)
THR = 0.01


def _pack(phi):
    """(3, D, H, W) [dz, dy, dx] -> flat [dx, dy, dz] (the solver packing)."""
    return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])


def _region_masks(patch_size, win_start, win_size):
    """Reference (window, region) boolean masks over the patch, C-order."""
    pz, py, px = patch_size
    oz, oy, ox = win_start
    sz, sy, sx = win_size
    window = np.zeros((pz, py, px), dtype=bool)
    window[oz : oz + sz, oy : oy + sy, ox : ox + sx] = True
    region = np.zeros((pz, py, px), dtype=bool)
    region[
        max(oz - 1, 0) : min(oz + sz + 1, pz),
        max(oy - 1, 0) : min(oy + sy + 1, py),
        max(ox - 1, 0) : min(ox + sx + 1, px),
    ] = True
    return window, region


def _smooth_random_phi(shape, scale=0.3):
    """Small random smooth-ish displacement field (3, D, H, W)."""
    phi = scale * (RNG.random((3, *shape)) - 0.5)
    return phi


class TestMaxWindowBuilder:
    def test_single_constraint_row_count_and_lb_healthy(self):
        # Zero field: Jdet == 1 everywhere -> every halo row is healthy,
        # so lb == THR on ALL rows (window and halo alike).
        patch_size, win_start, win_size = (7, 7, 7), (2, 2, 2), (3, 3, 3)
        phi = np.zeros((3, *patch_size))
        cons = _build_constraints_3d_maxwindow(_pack(phi), patch_size, win_start, win_size, THR)
        assert len(cons) == 1
        nlc = cons[0]
        _, region = _region_masks(patch_size, win_start, win_size)
        assert nlc.fun(np.zeros(3 * 27)).size == int(region.sum())  # 5^3 = 125
        np.testing.assert_allclose(np.asarray(nlc.lb), THR)

    def test_lb_bad_halo_keeps_current_value(self):
        # Fold sheet through the halo (outside the window): dx jump at x=1.
        patch_size, win_start, win_size = (7, 7, 7), (2, 2, 2), (3, 3, 3)
        phi = np.zeros((3, *patch_size))
        phi[2, :, :, 1] = 3.0  # halo plane x=1 folds; window spans x in [2, 5)
        x0 = _pack(phi)
        cons = _build_constraints_3d_maxwindow(x0, patch_size, win_start, win_size, THR)
        nlc = cons[0]
        window, region = _region_masks(patch_size, win_start, win_size)
        rows = np.flatnonzero(region.ravel())
        window_rows = window.ravel()[rows]
        jdet0 = _numpy_jdet_3d(phi[0], phi[1], phi[2]).ravel()[rows]
        lb = np.asarray(nlc.lb)
        # Window rows demand full threshold even where currently folded.
        np.testing.assert_allclose(lb[window_rows], THR)
        # Halo rows: healthy keep THR, folded keep their current value.
        halo = ~window_rows
        np.testing.assert_allclose(lb[halo], np.minimum(THR, jdet0[halo]))
        assert (jdet0[halo] < THR).any(), 'fixture must plant a bad halo row'

    def test_x0_feasible_on_halo_rows(self):
        patch_size, win_start, win_size = (7, 6, 8), (2, 2, 2), (3, 2, 4)
        phi = _smooth_random_phi(patch_size)
        x0_win = _pack(phi[:, 2:5, 2:4, 2:6])
        cons = _build_constraints_3d_maxwindow(_pack(phi), patch_size, win_start, win_size, THR)
        nlc = cons[0]
        window, region = _region_masks(patch_size, win_start, win_size)
        window_rows = window.ravel()[np.flatnonzero(region.ravel())]
        vals = nlc.fun(x0_win)
        lb = np.asarray(nlc.lb)
        halo = ~window_rows
        assert np.all(vals[halo] >= lb[halo] - 1e-12)

    def test_fun_equals_full_field_jdet(self):
        # KEY exactness property: constraint rows == what the accept check
        # measures on the full field, for interior windows AND clamped ones.
        vol_shape = (12, 11, 13)
        phi = _smooth_random_phi(vol_shape)
        jdet_full = jacobian_det3D(phi)
        for win_lo in [(4, 4, 4), (0, 0, 0)]:  # interior; volume-corner clamp
            sz = sy = sx = 3
            lo_z, lo_y, lo_x = win_lo
            pz0, py0, px0 = max(lo_z - 2, 0), max(lo_y - 2, 0), max(lo_x - 2, 0)
            pz1 = min(lo_z + sz + 2, vol_shape[0])
            py1 = min(lo_y + sy + 2, vol_shape[1])
            px1 = min(lo_x + sx + 2, vol_shape[2])
            patch = phi[:, pz0:pz1, py0:py1, px0:px1]
            patch_size = (pz1 - pz0, py1 - py0, px1 - px0)
            win_start = (lo_z - pz0, lo_y - py0, lo_x - px0)
            x0_win = _pack(phi[:, lo_z : lo_z + sz, lo_y : lo_y + sy, lo_x : lo_x + sx])
            cons = _build_constraints_3d_maxwindow(
                _pack(patch), patch_size, win_start, (sz, sy, sx), THR
            )
            _, region = _region_masks(patch_size, win_start, (sz, sy, sx))
            vals = cons[0].fun(x0_win)
            # Full-field oracle: region voxels in volume coordinates.
            reg_idx = np.argwhere(region) + np.array([pz0, py0, px0])
            oracle = jdet_full[reg_idx[:, 0], reg_idx[:, 1], reg_idx[:, 2]]
            np.testing.assert_allclose(
                vals,
                oracle,
                atol=1e-12,
                err_msg='patch rows must equal full-field Jdet (same stencils)',
            )

    def test_jac_matches_finite_difference(self):
        patch_size, win_start, win_size = (5, 6, 7), (1, 2, 2), (3, 3, 3)
        phi = _smooth_random_phi(patch_size, scale=0.2)
        cons = _build_constraints_3d_maxwindow(_pack(phi), patch_size, win_start, win_size, THR)
        nlc = cons[0]
        n_win = 3 * 27
        x = _pack(phi[:, 1:4, 2:5, 2:5]) + 0.01 * (RNG.random(n_win) - 0.5)
        J = nlc.jac(x)
        J = J.toarray() if hasattr(J, 'toarray') else np.asarray(J)
        eps = 1e-6
        fd = np.empty_like(J)
        for j in range(n_win):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fd[:, j] = (nlc.fun(xp) - nlc.fun(xm)) / (2 * eps)
        np.testing.assert_allclose(J, fd, atol=1e-6)

    def test_fun_does_not_mutate_patch_baseline(self):
        # embed() must not corrupt the captured patch: two calls with
        # different x from the same builder give independent results.
        patch_size, win_start, win_size = (7, 7, 7), (2, 2, 2), (3, 3, 3)
        phi = _smooth_random_phi(patch_size)
        cons = _build_constraints_3d_maxwindow(_pack(phi), patch_size, win_start, win_size, THR)
        nlc = cons[0]
        x_a = np.zeros(3 * 27)
        first = nlc.fun(x_a).copy()
        nlc.fun(RNG.random(3 * 27))  # perturbed evaluation in between
        np.testing.assert_array_equal(nlc.fun(x_a), first)


class TestMaxWindowSolveIntegration:
    def _fold_sheet_volume(self):
        """The tripwire fixture: fold component far larger than a 3^3 window."""
        D = H = W = 8
        d = np.zeros((3, D, H, W), dtype=np.float64)
        d[2, :, :, 3] = 3.0
        return d

    def test_no_new_negatives_anywhere(self):
        # Border no-damage, globally: an accepted max-window solve fixes
        # window voxels and cannot create fresh negatives in the halo;
        # a rejected one is rolled back. So the set of negative voxels
        # never grows.
        from dvfopt.core.slsqp_windowed.iterative3d import iterative_3d

        d = self._fold_sheet_volume()
        neg_before = jacobian_det3D(d) <= THR - 1e-5
        phi = iterative_3d(d, verbose=0, max_window=(3, 3, 3), max_iterations=5)
        neg_after = jacobian_det3D(phi) <= THR - 1e-5
        assert not (neg_after & ~neg_before).any(), 'solver created new negative voxels'
        assert neg_after.sum() < neg_before.sum(), 'solver made no progress'

    def test_maxwindow_requires_patch_ctx(self):
        from dvfopt.core.slsqp_windowed.coordinator3d import _optimize_single_window_3d

        x0 = np.zeros(3 * 27)
        with pytest.raises(AssertionError):
            _optimize_single_window_3d(
                x0,
                x0,
                (3, 3, 3),
                np.zeros((3, 3, 3), bool),
                THR,
                50,
                'SLSQP',
                window_reached_max=True,
            )
