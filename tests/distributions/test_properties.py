from itertools import product

import numpy as np
import pytest

import ondil

from .utils import get_univariate_distributions

DISTRIBUTIONS = [
    getattr(ondil.distributions, name)() for name in ondil.distributions.__all__
]


@pytest.mark.parametrize(
    "distribution",
    get_univariate_distributions(),
    ids=lambda dist: dist.__class__.__name__,
)
def test_initial_values(distribution):
    n_params = distribution.n_params
    lower = np.clip(distribution.distribution_support[0], -1e3, 1e3)
    upper = np.clip(distribution.distribution_support[1], -1e3, 1e3)
    y = np.random.uniform(low=lower, high=upper, size=1000)
    theta = distribution.initial_values(y)
    assert theta.shape == (y.shape[0], n_params), "Initial values shape mismatch"
    assert np.all(
        distribution.parameter_support[param][0]
        <= theta[:, param]
        <= distribution.parameter_support[param][1]
        for param in range(n_params)
    ), "Initial values out of parameter support"


@pytest.mark.parametrize(
    "distribution",
    get_univariate_distributions(),
    ids=lambda dist: dist.__class__.__name__,
)
def test_raise_error_cross_derivative(distribution):
    n_params = distribution.n_params
    # We just want some values here on a reasonable space to ensure that
    # We can test the raise of the derivative
    lower = np.clip(distribution.distribution_support[0], -1e3, 1e3)
    upper = np.clip(distribution.distribution_support[1], -1e3, 1e3)
    y = np.random.uniform(low=lower, high=upper, size=1000)
    theta = distribution.initial_values(y)

    for a, b in product(range(n_params), range(n_params)):
        if a == b:
            with pytest.raises(
                ValueError, match="Cross derivatives must use different parameters."
            ):
                distribution.dl2_dpp(y, theta, (a, b))
        else:
            deriv = distribution.dl2_dpp(y, theta, (a, b))
            assert y.shape == deriv.shape, "Derivative shape should match y.shape"
