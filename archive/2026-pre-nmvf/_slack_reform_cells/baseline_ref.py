def baseline_slsqp(phi_in, *, threshold=0.01, anchor='l1', max_iter=200,
                   warm_max_iter=800, verbose=0):
    """Wrap iterative_2d_tri_slsqp to return the same (phi, info)
    shape as slack_reform_slsqp."""
    t0 = time.time()
    phi_out, hist = iterative_2d_tri_slsqp(
        phi_in, threshold=threshold,
        max_iter=max_iter, warm_max_iter=warm_max_iter,
        anchor=anchor, verbose=verbose,
        record_history=True,
    )
    wall = time.time() - t0
    T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
    info = dict(
        total_t=wall,
        cold_nit=hist[0]['nit'],
        cold_status=hist[0]['status'],
        cold_success=hist[0]['success'],
        warm_fired=(len(hist) > 1),
        warm_t=(hist[1]['wall_s'] if len(hist) > 1 else 0.0),
        final_min=float(min(T1.min(), T2.min())),
        final_n_neg=int((T1 <= 0).sum() + (T2 <= 0).sum()),
    )
    return phi_out, info


def reference_m10(phi_in, *, threshold=0.01, anchor='l1', verbose=0):
    """Always-feasibility reference (m10)."""
    t0 = time.time()
    phi_out = iterative_2d_tri_harmonic_polished(
        phi_in, threshold=threshold, anchor=anchor, verbose=verbose,
    )
    wall = time.time() - t0
    T1, T2 = _triangle_areas_2d(phi_out[0], phi_out[1])
    info = dict(
        total_t=wall,
        final_min=float(min(T1.min(), T2.min())),
        final_n_neg=int((T1 <= 0).sum() + (T2 <= 0).sum()),
    )
    return phi_out, info


def l_cost(phi_out, phi_anchor):
    diff = (phi_out - phi_anchor).ravel()
    return {
        'L1': float(np.abs(diff).sum()),
        'L2': float(np.sqrt(np.dot(diff, diff))),
    }
