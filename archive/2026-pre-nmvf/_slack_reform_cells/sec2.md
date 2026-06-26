## 2. FD check on the augmented Jacobian

Quick sanity: build a tiny case, verify `jac(w)` matches the numerical
Jacobian of `constr(w)` to ~1e-9. This is the full-row-rank sparse block
that the whole prototype hinges on.
