"""Analyze the Laplacian boundary-condition correspondences on a 2D slice.

For each moving->fixed pair on the slice, compute the prescribed in-plane
displacement and how well the deformation field honors it (residual), before and
after correction. Flags outliers that are weird / very distorted:

- **large-disp**   — prescribed |moving-fixed| far above the robust spread.
- **high-residual** — the corrected field no longer maps fixed back to moving
  (the fold-correction perturbed this registration constraint).
- **incoherent**   — the correspondence's displacement disagrees sharply with
  its spatial neighbours' (locally inconsistent match — a likely bad pairing).

Convention (verified empirically): field[:, fixed] == moving - fixed, so the
in-plane residual is ``|(dy,dx)@fixed - (moving-fixed)@(y,x)|``.
"""

import numpy as np


def _mad(v):
    """Median and a robust (MAD-based) sigma estimate."""
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)) * 1.4826 + 1e-9)


def analyze_slice(sec_init, sec_out, mp_slice, fp_slice, top=25, k_neighbors=8):
    """Correspondence diagnostics for one slice. Returns a dict, or None if empty.

    *sec_init* / *sec_out* are ``(3, 1, H, W)`` before/after fields; *mp_slice* /
    *fp_slice* are the ``(M, 3)`` ``[z, y, x]`` correspondences on this slice.
    """
    m = len(fp_slice)
    if m == 0:
        return None
    fy, fx = fp_slice[:, 1].astype(int), fp_slice[:, 2].astype(int)
    my, mx = mp_slice[:, 1].astype(int), mp_slice[:, 2].astype(int)
    pdy = (mp_slice[:, 1] - fp_slice[:, 1]).astype(np.float64)  # prescribed dy = moving_y-fixed_y
    pdx = (mp_slice[:, 2] - fp_slice[:, 2]).astype(np.float64)
    disp_mag = np.hypot(pdy, pdx)

    def resid(sec):
        return np.hypot(sec[1, 0, fy, fx] - pdy, sec[2, 0, fy, fx] - pdx)

    r_before, r_after = resid(sec_init.astype(np.float64)), resid(sec_out.astype(np.float64))

    med_m, mad_m = _mad(disp_mag)
    mag_out = disp_mag > med_m + 3 * mad_m
    med_r, mad_r = _mad(r_after)
    # BCs are honored to ~0.03 px before correction; flag only real breaks.
    resid_out = r_after > max(1.0, med_r + 5 * mad_r)

    incoh = np.zeros(m, bool)
    dev = np.zeros(m)
    if m > k_neighbors + 1:
        from scipy.spatial import cKDTree

        pts = np.stack([fy, fx], axis=1).astype(np.float64)
        _, idx = cKDTree(pts).query(pts, k=min(k_neighbors + 1, m))
        vec = np.stack([pdy, pdx], axis=1)
        nb_mean = vec[idx[:, 1:]].mean(axis=1)  # exclude self (col 0)
        dev = np.linalg.norm(vec - nb_mean, axis=1)
        med_d, mad_d = _mad(dev)
        incoh = dev > med_d + 4 * mad_d

    is_out = mag_out | resid_out | incoh

    def _z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    sev = np.maximum.reduce([_z(disp_mag), _z(r_after), _z(dev)])
    order = np.where(is_out)[0]
    order = order[np.argsort(-sev[order])][:top]
    outliers = []
    for rank, i in enumerate(order, 1):
        types = [
            t
            for t, on in (
                ("large-disp", mag_out[i]),
                ("high-residual", resid_out[i]),
                ("incoherent", incoh[i]),
            )
            if on
        ]
        outliers.append(
            {
                "rank": rank,
                "y": int(fy[i]),
                "x": int(fx[i]),
                "my": int(my[i]),
                "mx": int(mx[i]),
                "disp": float(disp_mag[i]),
                "resid_before": float(r_before[i]),
                "resid_after": float(r_after[i]),
                "types": ", ".join(types),
            }
        )

    stats = {
        "n": int(m),
        "mean_disp": float(disp_mag.mean()),
        "max_disp": float(disp_mag.max()),
        "mean_resid_before": float(r_before.mean()),
        "mean_resid_after": float(r_after.mean()),
        "n_outliers": int(is_out.sum()),
        "n_large": int(mag_out.sum()),
        "n_high_resid": int(resid_out.sum()),
        "n_incoherent": int(incoh.sum()),
    }
    return {
        "fy": fy.astype(np.float32),
        "fx": fx.astype(np.float32),
        "my": my.astype(np.float32),
        "mx": mx.astype(np.float32),
        "outlier_idx": np.where(is_out)[0].astype(int).tolist(),
        "outliers": outliers,
        "stats": stats,
    }


def slice_correspondences(mp, fp, z):
    """Return the ``(mp_slice, fp_slice)`` correspondences whose fixed z == *z*."""
    if mp is None or fp is None:
        return None, None
    mask = fp[:, 0].astype(int) == z
    return mp[mask], fp[mask]
