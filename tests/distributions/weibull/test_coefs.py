import numpy as np
import pytest
from scipy.stats import weibull_min

from ondil.distributions import Weibull
from ondil.estimators import OnlineDistributionalRegression


def test_weibull_scipy_mapping():
    """Test that the Weibull distribution maps correctly to scipy."""
    wei = Weibull()
    
    # Test theta_to_scipy_params
    theta = np.array([[1.5, 2.0]])
    scipy_params = wei.theta_to_scipy_params(theta)
    
    assert "scale" in scipy_params
    assert "c" in scipy_params
    assert np.allclose(scipy_params["scale"], 1.5)
    assert np.allclose(scipy_params["c"], 2.0)


def test_weibull_pdf():
    """Test that PDF matches scipy."""
    wei = Weibull()
    
    np.random.seed(42)
    y = weibull_min.rvs(c=2.0, scale=1.5, size=100)
    theta = np.tile([[1.5, 2.0]], (len(y), 1))
    
    pdf_ondil = wei.pdf(y, theta)
    pdf_scipy = weibull_min.pdf(y, c=2.0, scale=1.5)
    
    assert np.allclose(pdf_ondil, pdf_scipy)


def test_weibull_cdf():
    """Test that CDF matches scipy."""
    wei = Weibull()
    
    np.random.seed(42)
    y = weibull_min.rvs(c=2.0, scale=1.5, size=100)
    theta = np.tile([[1.5, 2.0]], (len(y), 1))
    
    cdf_ondil = wei.cdf(y, theta)
    cdf_scipy = weibull_min.cdf(y, c=2.0, scale=1.5)
    
    assert np.allclose(cdf_ondil, cdf_scipy)


def test_weibull_ppf():
    """Test that PPF matches scipy."""
    wei = Weibull()
    
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    theta = np.tile([[1.5, 2.0]], (len(q), 1))
    
    ppf_ondil = wei.ppf(q, theta)
    ppf_scipy = weibull_min.ppf(q, c=2.0, scale=1.5)
    
    assert np.allclose(ppf_ondil, ppf_scipy)


def test_weibull_rvs():
    """Test that RVS generates values with correct shape."""
    wei = Weibull()
    
    theta = np.array([[1.5, 2.0], [2.0, 1.5]])
    size = 50
    
    samples = wei.rvs(size, theta)
    
    assert samples.shape == (2, 50)
    assert np.all(samples > 0)  # Weibull is positive


def test_weibull_derivatives_mu():
    """Test first and second derivatives w.r.t. mu match R GAMLSS formulas."""
    wei = Weibull()
    
    y = np.array([1.0, 2.0, 3.0])
    mu = 1.5
    sigma = 2.0
    theta = np.array([[mu, sigma], [mu, sigma], [mu, sigma]])
    
    # First derivative: dldm = ((y/mu)^sigma - 1) * (sigma/mu)
    dl1_mu = wei.dl1_dp1(y, theta, param=0)
    expected_dl1_mu = ((y / mu) ** sigma - 1) * (sigma / mu)
    assert np.allclose(dl1_mu, expected_dl1_mu)
    
    # Second derivative: d2ldm2 = -sigma^2 / mu^2
    dl2_mu = wei.dl2_dp2(y, theta, param=0)
    expected_dl2_mu = -(sigma**2) / (mu**2)
    assert np.allclose(dl2_mu, expected_dl2_mu)


def test_weibull_derivatives_sigma():
    """Test first and second derivatives w.r.t. sigma match R GAMLSS formulas."""
    wei = Weibull()
    
    y = np.array([1.0, 2.0, 3.0])
    mu = 1.5
    sigma = 2.0
    theta = np.array([[mu, sigma], [mu, sigma], [mu, sigma]])
    
    # First derivative: dldd = 1/sigma - log(y/mu) * ((y/mu)^sigma - 1)
    dl1_sigma = wei.dl1_dp1(y, theta, param=1)
    expected_dl1_sigma = (1 / sigma) - np.log(y / mu) * ((y / mu) ** sigma - 1)
    assert np.allclose(dl1_sigma, expected_dl1_sigma)
    
    # Second derivative: d2ldd2 = -1.82368 / sigma^2
    dl2_sigma = wei.dl2_dp2(y, theta, param=1)
    expected_dl2_sigma = -1.82368 / (sigma**2)
    assert np.allclose(dl2_sigma, expected_dl2_sigma)


def test_weibull_cross_derivative():
    """Test cross derivative matches R GAMLSS formula."""
    wei = Weibull()
    
    y = np.array([1.0, 2.0, 3.0])
    mu = 1.5
    sigma = 2.0
    theta = np.array([[mu, sigma], [mu, sigma], [mu, sigma]])
    
    # Cross derivative: d2ldmdd = 0.422784 / mu
    dl2_cross = wei.dl2_dpp(y, theta, params=(0, 1))
    expected_dl2_cross = 0.422784 / mu
    assert np.allclose(dl2_cross, expected_dl2_cross)


def test_weibull_initial_values():
    """Test that initial values are computed correctly."""
    wei = Weibull()
    
    np.random.seed(42)
    y = weibull_min.rvs(c=2.0, scale=1.5, size=100)
    
    initial = wei.initial_values(y)
    
    assert initial.shape == (100, 2)
    assert np.all(initial[:, 0] > 0)  # mu should be positive
    assert np.all(initial[:, 1] > 0)  # sigma should be positive


def test_weibull_gamlss_name():
    """Test that the GAMLSS name is correct."""
    wei = Weibull()
    assert wei.corresponding_gamlss == "WEI"


def test_weibull_parameter_names():
    """Test that parameter names are correct."""
    wei = Weibull()
    assert wei.parameter_names == {0: "mu", 1: "sigma"}


def test_weibull_parameter_support():
    """Test that parameter support is correct."""
    wei = Weibull()
    
    # Both parameters should be positive
    assert wei.parameter_support[0][0] > 0
    assert wei.parameter_support[0][1] == np.inf
    assert wei.parameter_support[1][0] > 0
    assert wei.parameter_support[1][1] == np.inf


def test_weibull_distribution_support():
    """Test that distribution support is correct."""
    wei = Weibull()
    
    # Weibull is defined for positive values
    assert wei.distribution_support[0] > 0
    assert wei.distribution_support[1] == np.inf


def test_weibull_estimation():
    """Test that Weibull can be used in estimation."""
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    
    # Generate y from Weibull with varying parameters
    y = weibull_min.rvs(c=2.0, scale=1.5, size=n)
    
    # Create and fit estimator
    wei = Weibull()
    estimator = OnlineDistributionalRegression(
        distribution=wei,
        equation={0: np.array([0, 1])},  # Only mu parameter with intercept and first covariate
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )
    
    estimator.fit(X=X, y=y)
    
    # Check that coefficients were estimated
    assert estimator.beta[0].shape[0] == 2  # Intercept + 1 covariate
    assert not np.any(np.isnan(estimator.beta[0]))
