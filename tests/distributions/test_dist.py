import numpy as np
import pytest
from ondil import HAS_SCORINGRULES
import rpy2.robjects as robjects

from .utils import get_distributions_with_gamlss

N = 100
CLIP_BOUNDS = (-1e5, 1e5)


# Distributions that need higher tolerance
SPECIAL_TOLERANCE_DISTRIBUTIONS = {
    "InverseGaussian": 1e-3,
    "SkewT": 1e-5,
}

SPECIAL_BOUNDS_DISTRIBUTIONS = {
    "PowerExponential": (-1e4, 1e4),
    "Weibull": (1e-3, 100),
}


@pytest.mark.parametrize(
    "distribution",
    get_distributions_with_gamlss(),
    ids=lambda dist: dist.__class__.__name__,
)
def test_distribution_functions(distribution):
    """Test that Python distribution functions match R GAMLSS functions."""

    # Set seed for reproducibility
    np.random.seed(42)

    clip_bounds = SPECIAL_BOUNDS_DISTRIBUTIONS.get(
        distribution.__class__.__name__,
        CLIP_BOUNDS,  # default np.allclose tolerance
    )

    # Generate random data within distribution support
    x = np.random.uniform(
        np.clip(distribution.distribution_support[0], *clip_bounds),
        np.clip(distribution.distribution_support[1], *clip_bounds),
        N,
    )
    robjects.globalenv["x"] = robjects.FloatVector(x)
    robjects.globalenv["q"] = robjects.FloatVector(x)
    prob_grid = np.linspace(0.01, 1 - 0.01, N)
    robjects.globalenv["p"] = robjects.FloatVector(prob_grid)

    # Generate random parameters within support bounds
    theta = np.array([
        np.random.uniform(
            np.clip(distribution.parameter_support[i][0], *clip_bounds),
            np.clip(distribution.parameter_support[i][1], *clip_bounds),
            N,
        )
        for i in range(distribution.n_params)
    ]).T

    # Assign R variables programmatically
    for i, param_name in distribution.parameter_names.items():
        robjects.globalenv[param_name] = robjects.FloatVector(theta[:, i])

    code = f"""
        available_vars <- c("x", "q", "p", "mu", "sigma", "nu", "tau")
        list(
          "cdf" = do.call(gamlss.dist::p{distribution.corresponding_gamlss}, mget(intersect(available_vars, names(formals(gamlss.dist::p{distribution.corresponding_gamlss}))), envir = .GlobalEnv)),
          "pdf" = do.call(gamlss.dist::d{distribution.corresponding_gamlss}, mget(intersect(available_vars, names(formals(gamlss.dist::d{distribution.corresponding_gamlss}))), envir = .GlobalEnv)),
          "ppf" = do.call(gamlss.dist::q{distribution.corresponding_gamlss}, mget(intersect(available_vars, names(formals(gamlss.dist::q{distribution.corresponding_gamlss}))), envir = .GlobalEnv))
        )
        """

    # Execute the R code to obtain the results
    R_list = robjects.r(code)

    if distribution.is_discrete:
        function_mapping = {
            "cdf": lambda: distribution.cdf(x, theta=theta),
            "pdf": lambda: distribution.pmf(x, theta=theta),
            "ppf": lambda: distribution.ppf(prob_grid, theta=theta),
        }
    else:
        function_mapping = {
            "cdf": lambda: distribution.cdf(x, theta=theta),
            "pdf": lambda: distribution.pdf(x, theta=theta),
            "ppf": lambda: distribution.ppf(prob_grid, theta=theta),
        }

    # Compare R and Python functions - only for available functions
    available_functions = R_list.names

    # Get tolerance for this distribution
    atol = SPECIAL_TOLERANCE_DISTRIBUTIONS.get(
        distribution.__class__.__name__,
        1e-8,  # default np.allclose tolerance
    )

    for key in available_functions:
        if key in function_mapping:
            print(f"Testing {key} for {distribution.__class__.__name__}")
            assert np.allclose(function_mapping[key](), R_list.rx2(key), atol=atol), (
                f"Function {key} doesn't match for {distribution.__class__.__name__}\n"
                f"Results: {np.round(R_list.rx2(key)[:10], 3)} (R)\n"
                f"Results: {np.round(function_mapping[key]()[:10], 3)} (Python)"
            )

    # Test CRPS if scoringrules is installed

    if HAS_SCORINGRULES:
        crps_values = distribution.crps(y=x, theta=theta)
        assert crps_values.shape == x.shape, "Shape mismatch in CRPS output"
