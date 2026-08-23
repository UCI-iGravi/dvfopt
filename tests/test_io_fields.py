"""dvfopt.io.fields — .npy/.npz + SimpleITK DVF I/O, plus GUI LoadWorker dispatch.

Library I/O is testable without the ``[gui]`` extra; only the LoadWorker
case needs PySide6 (guarded locally). SimpleITK is a core dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip('SimpleITK', reason='sitk interop tests need SimpleITK')

from dvfopt.io import fields as io_formats


def _vol(D=3, H=4, W=5):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.2, (3, D, H, W)).astype(np.float64)


def test_roundtrip_nii(tmp_path):
    vol = _vol()
    p = tmp_path / 'field.nii.gz'
    io_formats.save_dvf_sitk(p, vol)
    back = io_formats.load_dvf_sitk(p)
    assert back.shape == vol.shape
    np.testing.assert_allclose(back, vol, atol=1e-6)


def test_channel_convention_matches_sitk_jdet(tmp_path):
    # Component order must follow dvfopt/jacobian/sitk_jdet.py: sitk stores
    # [dx, dy, dz]; our numpy layout is [dz, dy, dx].
    vol = np.zeros((3, 2, 3, 3))
    vol[0] = 1.0  # dz
    vol[2] = 3.0  # dx
    p = tmp_path / 'conv.mha'
    io_formats.save_dvf_sitk(p, vol)
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)  # (D,H,W,3) components [dx,dy,dz]
    assert arr[..., 0].max() == pytest.approx(3.0)  # dx component
    assert arr[..., 2].max() == pytest.approx(1.0)  # dz component


def test_load_rejects_scalar_image(tmp_path):
    img = sitk.GetImageFromArray(np.zeros((3, 4, 5)))
    p = tmp_path / 'scalar.nii'
    sitk.WriteImage(img, str(p))
    with pytest.raises(ValueError):
        io_formats.load_dvf_sitk(p)


def test_2d_vector_image_maps_to_single_slice(tmp_path):
    arr = np.zeros((4, 5, 2))  # (H,W,2) components [dx,dy]
    arr[..., 0] = 2.0  # dx
    img = sitk.GetImageFromArray(arr, isVector=True)
    p = tmp_path / 'twod.mha'
    sitk.WriteImage(img, str(p))
    vol = io_formats.load_dvf_sitk(p)
    assert vol.shape == (3, 1, 4, 5)
    assert vol[2].max() == pytest.approx(2.0)  # dx channel
    assert vol[0].max() == 0.0  # dz zero


def test_is_sitk_path():
    assert io_formats.is_sitk_path('x.nii.gz') and io_formats.is_sitk_path('X.MHA')
    assert not io_formats.is_sitk_path('x.npy')


def test_loadworker_npy_and_sitk(tmp_path, qapp_placeholder=None):
    pytest.importorskip('PySide6', reason='dvfopt_gui requires the [gui] extra')
    from dvfopt_gui.worker import LoadWorker

    npy = tmp_path / 'f.npy'
    np.save(npy, np.zeros((3, 2, 4, 4)))
    results = []
    w = LoadWorker(str(npy))
    w.loadedRun.connect(lambda r: results.append(r))
    w.run()  # synchronous: exercise the body without a thread
    assert results and results[0].volume.shape == (3, 2, 4, 4)

    sp = tmp_path / 'f.nii.gz'
    io_formats.save_dvf_sitk(sp, _vol())
    w2 = LoadWorker(str(sp))
    w2.loadedRun.connect(lambda r: results.append(r))
    w2.run()
    assert results[-1].volume.shape == (3, 3, 4, 5)


# --- format-dispatching load_dvf / save_dvf (no GUI, no SimpleITK needed) ---


def test_load_save_dvf_npy_roundtrip(tmp_path):
    from dvfopt.io import load_dvf, save_dvf

    vol = np.zeros((3, 2, 4, 5))
    vol[1] = 1.5
    p = tmp_path / 'f.npy'
    save_dvf(p, vol)
    back = load_dvf(p)
    np.testing.assert_array_equal(back, vol)
    assert back.dtype == np.float64


def test_load_dvf_npz_single_array(tmp_path):
    from dvfopt.io import load_dvf

    vol = np.ones((2, 4, 5))
    p = tmp_path / 'f.npz'
    np.savez(p, field=vol)
    np.testing.assert_array_equal(load_dvf(p), vol)


def test_load_dvf_npz_multi_array_rejected(tmp_path):
    from dvfopt.io import load_dvf

    p = tmp_path / 'f.npz'
    np.savez(p, a=np.ones(3), b=np.ones(3))
    with pytest.raises(ValueError, match='2 arrays'):
        load_dvf(p)


def test_load_save_dvf_unsupported_extension(tmp_path):
    from dvfopt.io import load_dvf, save_dvf

    with pytest.raises(ValueError, match='unsupported'):
        load_dvf(tmp_path / 'f.txt')
    with pytest.raises(ValueError, match='unsupported'):
        save_dvf(tmp_path / 'f.txt', np.zeros((2, 4, 5)))
