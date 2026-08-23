"""Byte-identity + trace contract for the vendored traced C-SLSQP driver."""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


def _problem():
    rng = np.random.default_rng(7)
    n = 30
    x0 = rng.normal(0, 1, n)
    tgt = rng.normal(0, 1, n)
    A = rng.normal(0, 1, (8, n))
    b = A @ tgt + np.abs(rng.normal(0, 1, 8))  # active constraints at optimum

    def f(x):
        d = x - tgt
        return float(d @ d), 2.0 * d

    cons = [{"type": "ineq", "fun": lambda x: A @ x - b, "jac": lambda x: A}]
    return f, x0, cons


class TestByteIdentity:
    def test_identical_to_scipy_slsqp(self):
        from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

        f, x0, cons = _problem()
        ref = minimize(
            f,
            x0,
            jac=True,
            method="SLSQP",
            constraints=cons,
            options={"maxiter": 100, "ftol": 1e-8},
        )
        r = minimize_slsqp_traced(
            lambda x: f(x)[0],
            x0,
            jac=lambda x: f(x)[1],
            constraints=cons,
            maxiter=100,
            ftol=1e-8,
            trace=None,
        )
        assert (r.status, r.nit) == (ref.status, ref.nit)
        assert np.array_equal(r.x, ref.x)

    def test_trace_records_majors(self):
        from dvfopt.core.primitives.slsqp import minimize_slsqp_traced

        f, x0, cons = _problem()
        tr: dict = {}
        r = minimize_slsqp_traced(
            lambda x: f(x)[0],
            x0,
            jac=lambda x: f(x)[1],
            constraints=cons,
            maxiter=100,
            ftol=1e-8,
            trace=tr,
        )
        assert tr["iters"] and tr["nit"] == r.nit
        last = tr["iters"][-1]
        assert last["max_viol"] < 1e-8
        assert {"obj", "opt", "alpha", "nfev"} <= set(last)


class TestIneqDict:
    def test_lb_shift_and_sparse_densify(self):
        from dvfopt.core.primitives.slsqp import ineq_dict

        def fun(x):
            return np.array([x[0], x[1] * 2.0])

        def jac(x):
            return sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))

        d = ineq_dict(fun, jac, lb=0.5)
        assert d["type"] == "ineq"
        np.testing.assert_allclose(d["fun"](np.array([1.0, 1.0])), [0.5, 1.5])
        J = d["jac"](np.array([1.0, 1.0]))
        assert isinstance(J, np.ndarray)
        np.testing.assert_allclose(J, [[1.0, 0.0], [0.0, 2.0]])
