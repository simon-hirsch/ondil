import numpy as np

from ondil.distributions import Poisson


def test_poisson_distribution():
    """Test basic functionality of the Poisson distribution."""
    dist = Poisson()

    # Test basic properties
    assert dist.n_params == 1
    assert dist.corresponding_gamlss == "PO"
    assert dist.parameter_names == {0: "mu"}

    # Test with sample data
    y = np.array([0, 1, 2, 3, 4, 5])
    theta = np.array([[5.0]] * len(y))

    # Test PMF
    pmf_vals = dist.pmf(y, theta)
    assert pmf_vals.shape == y.shape
    assert np.all(pmf_vals >= 0)
    assert np.all(pmf_vals <= 1)

    # Test CDF
    cdf_vals = dist.cdf(y, theta)
    assert cdf_vals.shape == y.shape
    assert np.all(cdf_vals >= 0)
    assert np.all(cdf_vals <= 1)
    # CDF should be monotonically increasing
    assert np.all(np.diff(cdf_vals) >= 0)

    # Test PPF
    q = np.array([0.1, 0.5, 0.9])
    theta_ppf = np.array([[5.0]] * len(q))
    ppf_vals = dist.ppf(q, theta_ppf)
    assert ppf_vals.shape == q.shape
    # PPF should be monotonically increasing
    assert np.all(np.diff(ppf_vals) >= 0)

    # Test derivatives
    dl1_dp1 = dist.dl1_dp1(y, theta, param=0)
    assert dl1_dp1.shape == y.shape
    expected_dl1 = (y - 5.0) / 5.0
    assert np.allclose(dl1_dp1, expected_dl1)

    dl2_dp2 = dist.dl2_dp2(y, theta, param=0)
    assert dl2_dp2.shape == y.shape
    expected_dl2 = -1.0 / 5.0
    assert np.allclose(dl2_dp2, expected_dl2)

    # Test initial values
    initial = dist.initial_values(y)
    assert initial.shape == (len(y), 1)
    assert np.allclose(initial[0, 0], np.mean(y))


def test_poisson_scipy_consistency():
    """Test that Poisson matches scipy.stats.poisson."""
    import scipy.stats as st

    dist = Poisson()

    # Test with various mu values
    for mu in [1.0, 2.5, 5.0, 10.0]:
        y = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20])
        theta = np.array([[mu]] * len(y))

        # Test PMF
        pmf_ondil = dist.pmf(y, theta)
        pmf_scipy = st.poisson(mu=mu).pmf(y)
        assert np.allclose(pmf_ondil, pmf_scipy), f"PMF mismatch for mu={mu}"

        # Test CDF
        cdf_ondil = dist.cdf(y, theta)
        cdf_scipy = st.poisson(mu=mu).cdf(y)
        assert np.allclose(cdf_ondil, cdf_scipy), f"CDF mismatch for mu={mu}"

        # Test PPF
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        theta_ppf = np.array([[mu]] * len(q))
        ppf_ondil = dist.ppf(q, theta_ppf)
        ppf_scipy = st.poisson(mu=mu).ppf(q)
        assert np.allclose(ppf_ondil, ppf_scipy), f"PPF mismatch for mu={mu}"


def test_poisson_log_functions():
    """Test log PMF and log CDF functions."""
    dist = Poisson()

    y = np.array([0, 1, 2, 3, 4, 5])
    theta = np.array([[5.0]] * len(y))

    # Test logpmf
    logpmf_vals = dist.logpmf(y, theta)
    pmf_vals = dist.pmf(y, theta)
    assert np.allclose(logpmf_vals, np.log(pmf_vals))

    # Test logcdf
    logcdf_vals = dist.logcdf(y, theta)
    cdf_vals = dist.cdf(y, theta)
    assert np.allclose(logcdf_vals, np.log(cdf_vals))

    # Test that logpdf delegates to logpmf for discrete distributions
    logpdf_vals = dist.logpdf(y, theta)
    assert np.allclose(logpdf_vals, logpmf_vals)

    # Test that pdf delegates to pmf for discrete distributions
    pdf_vals = dist.pdf(y, theta)
    assert np.allclose(pdf_vals, pmf_vals)


def test_poisson_rvs():
    """Test random variate generation."""
    dist = Poisson()

    mu = 5.0
    theta = np.array([[mu]] * 1)
    size = 1000

    samples = dist.rvs(size=size, theta=theta)
    assert samples.shape == (1, size)

    # Check that mean is approximately mu
    sample_mean = np.mean(samples)
    # Allow for some variance due to randomness
    assert np.abs(sample_mean - mu) < 1.0, (
        f"Sample mean {sample_mean} too far from {mu}"
    )


def test_poisson_derivatives_shape():
    """Test that derivatives have correct shapes."""
    dist = Poisson()

    y = np.array([0, 1, 2, 3, 4, 5])
    theta = np.array([[5.0]] * len(y))

    # Test first derivative
    dl1_dp1 = dist.dl1_dp1(y, theta, param=0)
    assert dl1_dp1.shape == y.shape

    # Test second derivative
    dl2_dp2 = dist.dl2_dp2(y, theta, param=0)
    assert dl2_dp2.shape == y.shape

    # Test cross derivative (should be zero for single parameter)
    dl2_dpp = dist.dl2_dpp(y, theta, params=(0, 0))
    assert dl2_dpp.shape == y.shape
    # For Poisson with single parameter, this should raise an error or return zeros
    # The validation should catch that params must be different


def test_poisson_with_estimator():
    """Test Poisson distribution with OnlineDistributionalRegression."""
    from ondil.estimators import OnlineDistributionalRegression

    # Generate some Poisson data
    np.random.seed(42)
    n = 100

    # Create predictors
    X = np.random.randn(n, 2)

    # True coefficients
    true_beta = np.array([1.5, 0.3, -0.2])

    # Generate mu
    mu = np.exp(true_beta[0] + X[:, 0] * true_beta[1] + X[:, 1] * true_beta[2])

    # Generate Poisson samples
    y = np.random.poisson(lam=mu)

    # Fit the model
    estimator = OnlineDistributionalRegression(
        distribution=Poisson(),
        equation={0: np.array([0, 1])},  # Use both predictors for mu
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X, y=y)

    # Check that coefficients are reasonably close to true values
    coef_error = np.abs(estimator.beta[0] - true_beta).mean()
    assert coef_error < 0.5, f"Coefficient error too large: {coef_error}"

    # Make predictions
    fv = estimator.predict(X=X)
    assert fv.shape == (n,)

    # Check that predictions are reasonable
    pred_error = np.abs(fv - mu).mean()
    assert pred_error < 2.0, f"Prediction error too large: {pred_error}"
