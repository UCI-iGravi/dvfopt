def slack_reform_slsqp(
    phi_in, *,
    threshold=None,
    anchor='l1',
    eps_l1=1e-4,
    max_iter=200,
    warm_max_iter=800,
    warm_seed=123,
    warm_sigma=0.01,
    verbose=1,
    return_res=False,
):
    """Equality-form SLSQP with slack variables for the 2-tri constraint.

    Variables: w = [z, s] of length 2*H*W + 2*(H-1)*(W-1).
    Constraints: T(z) - s = threshold (equality), s >= 0 (bound).
    Objective: anchor(z - z_anchor); does not depend on s.
    """
    if threshold is None:
        threshold = DEFAULT_PARAMS['threshold']
    H, W = phi_in.shape[1], phi_in.shape[2]
    n_phi = 2 * H * W
    n_T   = 2 * (H - 1) * (W - 1)
    n_w   = n_phi + n_T

    z_anchor = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])

    def obj(w):
        z = w[:n_phi]
        diff = z - z_anchor
        val, grad_z = anchor_term(diff, anchor, eps_l1)
        grad = np.zeros(n_w)
        grad[:n_phi] = grad_z
        return val, grad

    def constr(w):
        z = w[:n_phi]; s = w[n_phi:]
        return tri_areas_flat(z, H, W) - s - threshold

    jac_T = _build_full_grid_tri_jac(H, W, full_coverage=False)
    neg_I = -sp.eye(n_T, format='csr')

    def jac(w):
        z = w[:n_phi]
        return sp.hstack([jac_T(z), neg_I], format='csr')

    # Initial slack: feasible if T(z_anchor) >= threshold; otherwise
    # SLSQP will pull s down toward 0 and z toward feasibility.
    T0 = tri_areas_flat(z_anchor, H, W) - threshold
    s_init = np.maximum(T0, 0.0)
    w_init = np.concatenate([z_anchor, s_init])

    bounds = [(None, None)] * n_phi + [(0.0, None)] * n_T
    nlc = NonlinearConstraint(constr, lb=0.0, ub=0.0, jac=jac)

    t0 = time.time()
    res = minimize(
        obj, w_init, jac=True, method='SLSQP', bounds=bounds,
        constraints=[nlc],
        options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': verbose >= 3},
    )
    cold_t = time.time() - t0
    cold_nit   = int(res.nit)
    cold_status = int(res.status)
    cold_success = bool(res.success)

    warm_t = 0.0
    warm_fired = False
    if not res.success:
        warm_fired = True
        rng = np.random.default_rng(warm_seed)
        w_warm = res.x.copy()
        if res.status == 8:
            w_warm[:n_phi] += rng.normal(scale=warm_sigma, size=n_phi)
        t1 = time.time()
        res = minimize(
            obj, w_warm, jac=True, method='SLSQP', bounds=bounds,
            constraints=[nlc],
            options={'maxiter': warm_max_iter, 'ftol': 1e-10,
                     'disp': verbose >= 3},
        )
        warm_t = time.time() - t1

    z_out = res.x[:n_phi]
    dy = z_out[:H * W].reshape(H, W)
    dx = z_out[H * W:].reshape(H, W)
    phi_out = np.stack([dy, dx])

    T = tri_areas_flat(z_out, H, W)
    info = dict(
        cold_nit=cold_nit, cold_status=cold_status, cold_success=cold_success,
        cold_t=cold_t, warm_fired=warm_fired, warm_t=warm_t,
        total_t=cold_t + warm_t,
        final_status=int(res.status), final_success=bool(res.success),
        final_nit=int(res.nit),
        final_min=float(T.min()),
        final_n_neg=int((T <= 0).sum()),
    )
    if verbose >= 1:
        print(f'[slack-reform] {H}x{W} '
              f'cold_nit={cold_nit} status={cold_status} success={cold_success} '
              f'warm_fired={warm_fired} '
              f'final_neg={info["final_n_neg"]} '
              f'min_T={info["final_min"]:+.5f} '
              f't={info["total_t"]:.2f}s')
    if return_res:
        return phi_out, info, res
    return phi_out, info
