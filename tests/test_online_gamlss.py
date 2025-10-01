import numpy as np
import pytest
from sklearn.datasets import make_regression

from ondil.distributions import JSU
from ondil.estimators import OnlineDistributionalRegression

FIT_INTERCEPT = [True, False]
N_FEATURES = np.round(np.geomspace(11, 100, 5)).astype(int)


@pytest.mark.parametrize("n_features", N_FEATURES, ids=lambda x: f"n_features_{x}")
@pytest.mark.parametrize(
    "fit_intercept", FIT_INTERCEPT, ids=lambda x: f"fit_intercept_{x}"
)
def test_get_J_from_equation(n_features, fit_intercept):
    equation = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, 4),
        3: np.array([True] * 10 + [False] * (n_features - 10)).astype(bool),
    }

    EXPECTED = {
        0: {True: n_features + 1, False: n_features},
        1: {True: 1, False: 1},
        2: {True: 5, False: 4},
        3: {True: 11, False: 10},
    }

    X, _ = make_regression(n_samples=100, n_features=n_features)
    distribution = JSU()

    estimator = OnlineDistributionalRegression(
        distribution=distribution,
        equation=equation,
        fit_intercept=fit_intercept,
    )
    estimator._prepare_estimator()
    J = estimator.get_J_from_equation(X)
    for param, expected_dict in EXPECTED.items():
        assert J[param] == expected_dict[fit_intercept], f"Wrong J for param == {param}"


def test_get_J_from_equation_warnings():
    n_features = 10
    fit_intercept = True

    equation_fail_2 = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, 20),
        3: np.array([True] * n_features).astype(bool),
    }

    X, _ = make_regression(n_samples=100, n_features=n_features)
    distribution = JSU()

    estimator = OnlineDistributionalRegression(
        distribution=distribution,
        equation=equation_fail_2,
        fit_intercept=fit_intercept,
    )
    estimator._prepare_estimator()
    with pytest.raises(ValueError, match="Shape does not match for param 2."):
        _ = estimator.get_J_from_equation(X)

    # Test for parameter three
    equation_fail_3 = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, n_features),
        3: np.array([True, False] * 10).astype(bool),
    }

    X, _ = make_regression(n_samples=100, n_features=10)
    distribution = JSU()
    estimator = OnlineDistributionalRegression(
        distribution=distribution,
        equation=equation_fail_3,
        fit_intercept=fit_intercept,
    )
    estimator._prepare_estimator()

    with pytest.raises(
        ValueError,
        match="Shape does not match for param 3.",
    ):
        _ = estimator.get_J_from_equation(X)
