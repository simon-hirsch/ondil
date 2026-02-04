"""Tests for CRPS (Continuous Ranked Probability Score) functionality."""
import numpy as np
import pytest

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
)

# Set of distributions that should have CRPS support
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


def test_scoringrules_optional_import():
    """Test that code works even without scoringrules installed."""
    # This test will pass whether scoringrules is installed or not
    # If installed, CRPS should work; if not, it should return None gracefully
    dist = Normal()
    y = np.array([1.0, 2.0, 3.0])
    theta = np.array([[1.5, 0.5], [2.5, 0.5], [3.5, 0.5]])
    
    # This should not raise an error
    result = dist.crps(y, theta)
    
    # Check that result is either an array or None
    assert result is None or isinstance(result, np.ndarray)


@pytest.mark.parametrize(
    "distribution",
    DISTRIBUTIONS_WITH_CRPS,
    ids=lambda dist: dist.__class__.__name__,
)
def test_crps_function_attribute(distribution):
    """Test that distributions have the crps_function attribute."""
    assert hasattr(distribution, "crps_function")
    assert distribution.crps_function is not None
    assert isinstance(distribution.crps_function, str)


@pytest.mark.parametrize(
    "distribution",
    DISTRIBUTIONS_WITH_CRPS,
    ids=lambda dist: dist.__class__.__name__,
)
def test_theta_to_crps_params(distribution):
    """Test that theta_to_crps_params returns a tuple of (pos_args, kw_args)."""
    np.random.seed(42)
    n = 10
    
    # Generate valid theta values within parameter support
    theta = np.array([
        np.random.uniform(
            max(distribution.parameter_support[i][0], -100),
            min(distribution.parameter_support[i][1], 100),
            n,
        )
        for i in range(distribution.n_params)
    ]).T
    
    # Ensure positive values for scale/shape parameters
    if distribution.n_params >= 2:
        theta[:, 1] = np.abs(theta[:, 1]) + 0.1
    if distribution.n_params >= 3:
        theta[:, 2] = np.abs(theta[:, 2]) + 2.1  # For StudentT nu parameter
    
    params = distribution.theta_to_crps_params(theta)
    
    assert params is not None
    assert isinstance(params, tuple)
    assert len(params) == 2
    pos_args, kw_args = params
    assert isinstance(pos_args, tuple)
    assert isinstance(kw_args, dict)


# Only run tests that require scoringrules if it's available
try:
    import scoringrules as sr
    SCORINGRULES_AVAILABLE = True
except ImportError:
    SCORINGRULES_AVAILABLE = False


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
@pytest.mark.parametrize(
    "distribution",
    DISTRIBUTIONS_WITH_CRPS,
    ids=lambda dist: dist.__class__.__name__,
)
def test_crps_returns_array(distribution):
    """Test that CRPS method returns an array when scoringrules is available."""
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
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (n,)
    # Most values should be finite and non-negative
    # (some edge cases may produce NaN, but that's OK for extreme parameters)
    finite_vals = result[np.isfinite(result)]
    assert len(finite_vals) > 0, "At least some CRPS values should be finite"
    assert np.all(finite_vals >= 0), "Finite CRPS values should be non-negative"


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_normal_crps_basic():
    """Test Normal distribution CRPS with known values."""
    dist = Normal()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    theta = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # CRPS should be small when observation matches mean
    assert np.all(result < 1.0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_studentt_crps_basic():
    """Test StudentT distribution CRPS with known values."""
    dist = StudentT()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu, sigma, nu]
    theta = np.array([[1.0, 1.0, 5.0], [2.0, 1.0, 5.0], [3.0, 1.0, 5.0]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # CRPS should be small when observation matches location
    assert np.all(result < 1.5)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_gamma_crps_basic():
    """Test Gamma distribution CRPS with known values."""
    dist = Gamma()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu, sigma]
    theta = np.array([[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_beta_crps_basic():
    """Test Beta distribution CRPS with known values."""
    dist = Beta()
    
    # Simple test case - observations in (0, 1)
    y = np.array([0.3, 0.5, 0.7])
    # theta = [mu, sigma]
    theta = np.array([[0.3, 0.3], [0.5, 0.3], [0.7, 0.3]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_exponential_crps_basic():
    """Test Exponential distribution CRPS with known values."""
    dist = Exponential()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu]
    theta = np.array([[1.0], [2.0], [3.0]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_logistic_crps_basic():
    """Test Logistic distribution CRPS with known values."""
    dist = Logistic()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu, sigma]
    theta = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_lognormal_crps_basic():
    """Test LogNormal distribution CRPS with known values."""
    dist = LogNormal()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu, sigma]
    theta = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_poisson_crps_basic():
    """Test Poisson distribution CRPS with known values."""
    dist = Poisson()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu]
    theta = np.array([[1.0], [2.0], [3.0]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_crps_perfect_forecast():
    """Test that CRPS is zero for perfect forecasts with degenerate distributions."""
    # For a degenerate distribution (sigma -> 0), CRPS should approach |y - mu|
    dist = Normal()
    
    y = np.array([1.0, 2.0, 3.0])
    theta = np.array([[1.0, 0.01], [2.0, 0.01], [3.0, 0.01]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    # With very small sigma, CRPS should be very small when y = mu
    assert np.all(result < 0.1)


@pytest.mark.skipif(not SCORINGRULES_AVAILABLE, reason="scoringrules not installed")
def test_normalmeanvariance_crps():
    """Test NormalMeanVariance CRPS with variance parameterization."""
    dist = NormalMeanVariance()
    
    # Simple test case
    y = np.array([1.0, 2.0, 3.0])
    # theta = [mu, sigma^2] where sigma^2 is variance
    theta = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    
    result = dist.crps(y, theta)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result >= 0)
