def _fd_check(H=4, W=5, seed=0, tol=1e-7):
    rng = np.random.default_rng(seed)
    phi_in = np.stack([rng.normal(scale=0.1, size=(H, W)),
                       rng.normal(scale=0.1, size=(H, W))])
    n_phi = 2 * H * W
    n_T = 2 * (H - 1) * (W - 1)

    threshold = 0.01
    z_anchor = np.concatenate([phi_in[0].ravel(), phi_in[1].ravel()])
    T0 = tri_areas_flat(z_anchor, H, W) - threshold
    w0 = np.concatenate([z_anchor, np.maximum(T0, 0.0)])
    jac_T = _build_full_grid_tri_jac(H, W, full_coverage=False)
    neg_I = -sp.eye(n_T, format='csr')

    def constr(w):
        z = w[:n_phi]; s = w[n_phi:]
        return tri_areas_flat(z, H, W) - s - threshold

    def jac(w):
        z = w[:n_phi]
        return sp.hstack([jac_T(z), neg_I], format='csr')

    n_w = n_phi + n_T
    J_ana = jac(w0).toarray()
    J_num = np.zeros((n_T, n_w))
    eps = 1e-6
    for i in range(n_w):
        wp = w0.copy(); wp[i] += eps
        wm = w0.copy(); wm[i] -= eps
        J_num[:, i] = (constr(wp) - constr(wm)) / (2 * eps)
    err = np.abs(J_num - J_ana).max()
    rank = int(np.linalg.matrix_rank(J_ana))
    print(f'H={H} W={W}: max |J_num - J_ana| = {err:.2e}  '
          f'rank = {rank} / {n_T}  (full row rank: {rank == n_T})')
    assert err < tol, f'Jacobian FD mismatch: {err:.2e}'

_fd_check(4, 5, seed=0)
_fd_check(5, 7, seed=1)
