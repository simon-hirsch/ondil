"""Tests for CRPS (Continuous Ranked Probability Score) functionality."""
import numpy as np
import pytest

import ondil
from ondil.distributions import (
    Beta,
    Exponential,
    Gamma,
    Logistic,
    LogNormal,
    Normal,
    NormalMeanVariance,
    Poisson,
    StudentT,
    MultivariateNormalInverseCholesky,
    Weibull,
)


def test_has_scoringrules_flag():
    """Test that HAS_SCORINGRULES flag is defined."""
    assert hasattr(ondil, 'HAS_SCORINGRULES')
    assert isinstance(ondil.HAS_SCORINGRULES, bool)


@pytest.mark.skipif(not ondil.HAS_SCORINGRULES, reason="scoringrules not installed")
def test_crps_raises_for_multivariate():
    """Test that CRPS raises ValueError for multivariate distributions."""
    dist = MultivariateNormalInverseCholesky()
    y = np.random.randn(10, 2)
    theta = {0: np.random.randn(10, 2), 1: np.random.randn(10, 3)}
    
    with pytest.raises(ValueError, match="not supported for multivariate"):
        dist.crps(y[:, 0], theta)


@pytest.mark.skipif(ondil.HAS_SCORINGRULES, reason="Test only when scoringrules not installed")
def test_crps_raises_without_scoringrules():
    """Test that CRPS raises ImportError when scoringrules is not installed."""
    dist = Normal()
    y = np.array([1.0, 2.0, 3.0])
    theta = np.array([[1.5, 0.5], [2.5, 0.5], [3.5, 0.5]])
    
    with pytest.raises(ImportError, match="scoringrules package is required"):
        dist.crps(y, theta)


# Test distributions with CRPS support
DISTRIBUTIONS_WITH_CRPS = [
    Normal(),
    NormalMeanVariance(),
    StudentT(),
    Gamma(),
    Beta(),
    Exponential(),
    Logistic(),
    LogNormal(),
    Poisson(),
]


@pytest.mark.skipif(not ondil.HAS_SCORINGRULES, reason="scoringrules not installed")
@pytest.mark.parametrize(
    "distribution",
    DISTRIBUTIONS_WITH_CRPS,
    ids=lambda dist: dist.__class__.__name__,
)
def test_crps_returns_array(distribution):
    """Test that CRPS method always returns a numpy array."""
    np.random.seed(42)
    n = 10
    
    # Generate valid theta values within parameter support (use conservative ranges)
    theta = np.array([
        np.random.uniform(
            max(distribution.parameter_support[i][0], -10),
            min(distribution.parameter_support[i][1], 10),
            n,
        )
        for i in range(distribution.n_params)
    ]).T
    
    # Ensure positive values for scale/shape parameters with conservative bounds
    if distribution.n_params >= 2:
        theta[:, 1] = np.random.uniform(0.5, 2.0, n)  # More conservative scale
    if distribution.n_params >= 3:
        theta[:, 2] = np.random.uniform(3.0, 10.0, n)  # More conservative df for StudentT
    
    # Generate observations within distribution support (use conservative bounds)
    if distribution.is_discrete:
        y = np.random.poisson(5, n).astype(float)
    else:
        lower = max(distribution.distribution_support[0], 0.1)
        upper = min(distribution.distribution_support[1], 10)
        if lower == upper:
            y = np.ones(n) * lower
        elif np.isinf(lower) and np.isinf(upper):
            y = np.random.randn(n)
        elif np.isinf(lower):
            y = np.random.uniform(-10, upper, n)
        elif np.isinf(upper):
            y = np.random.uniform(lower, 10, n)
        else:
            y = np.random.uniform(lower, upper, n)
    
    result = distribution.crps(y, theta)
    
    # Must return a numpy array
    assert isinstance(result, np.ndarray)
    assert result.shape == (n,)
    
    # Most values should be finite and non-negative
    finite_vals = result[np.isfinite(result)]
    assert len(finite_vals) > 0, "At least some CRPS values should be finite"
    assert np.all(finite_vals >= 0), "Finite CRPS values should be non-negative"


@pytest.mark.skipif(not ondil.HAS_SCORINGRULES, reason="scoringrules not installed")
def test_normal_crps_basic():
    """Test Normal distribution CRPS with known values."""
    dist = Normal()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    theta = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    
    result = dist.crps(y, theta)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # CRPS should be small when observation matches mean
    assert np.all(result < 1.0)


@pytest.mark.skipif(not ondil.HAS_SCORINGRULES, reason="scoringrules not installed")
def test_crps_quantile_approximation():
    """Test that CRPS works with quantile approximation for distributions without closed form."""
    # Weibull doesn't have a crps_function defined, so it should use quantile approximation
    dist = Weibull()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    theta = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    
    result = dist.crps(y, theta)
    
    # Should still return an array using quantile approximation
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)
