import pytest

from dvfopt import L1Objective, L2Objective, SimplexConstraint2D, Solver
from dvfopt.exceptions import IncompatibleObjectiveError
from dvfopt.strategies import SLPStrategy


def test_slp_rejects_l2_at_construction():
    with pytest.raises(IncompatibleObjectiveError, match="SLPStrategy"):
        Solver(
            constraint=SimplexConstraint2D(shape=(8, 8)),
            objective=L2Objective(),
            strategy=SLPStrategy(),
        )


def test_slp_accepts_l1():
    Solver(
        constraint=SimplexConstraint2D(shape=(8, 8)),
        objective=L1Objective(),
        strategy=SLPStrategy(),
    )  # must not raise
