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
    # Generate random parameters within support bounds
    theta = np.array([
        np.random.uniform(
            np.clip(distribution.parameter_support[i][0], *clip_bounds),
            np.clip(distribution.parameter_support[i][1], *clip_bounds),
            N,
        )
        for i in range(distribution.n_params)
    ]).T

    # Test CRPS if scoringrules is installed

    if HAS_SCORINGRULES:
        crps_values = distribution.crps(y=x, theta=theta)
        assert crps_values.shape == x.shape, "Shape mismatch in CRPS output"
