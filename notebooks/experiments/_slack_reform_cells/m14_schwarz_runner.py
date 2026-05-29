def run_one(method_label, fn, phi_in, **kwargs):
    t0 = time.time()
    extra = {}
    try:
        if method_label == 'm14_schwarz':
            phi_out, hist = fn(phi_in.copy(), threshold=THRESHOLD,
                                anchor='l1', verbose=0, record_history=True,
                                **kwargs)
            extra = dict(
                fallback=hist.get('fallback_to_global', False),
                n_clusters=len(hist.get('cluster_runs', [])),
                outer_rounds=len(hist.get('outer_rounds', [])),
            )
        else:
            phi_out = _silent(fn, phi_in.copy(),
                              threshold=THRESHOLD, anchor='l1', verbose=0,
                              **kwargs)
    except Exception as exc:
        return dict(method=method_label, wall_s=time.time()-t0,
                    error=f'{type(exc).__name__}: {exc}', feasible=False,
                    n_neg=-1, min_T=float('nan'), L1=float('nan'),
                    L2=float('nan'), **extra)
    wall = time.time() - t0
    n_neg, min_T = _stats(phi_out)
    diff = (phi_out - phi_in).ravel()
    return dict(
        method=method_label, wall_s=wall,
        n_neg=n_neg, min_T=min_T,
        L1=float(np.abs(diff).sum()),
        L2=float(np.sqrt(np.dot(diff, diff))),
        feasible=(n_neg == 0 and min_T >= THRESHOLD - 1e-5),
        error='', **extra,
    )
