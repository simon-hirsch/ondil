import numpy as np

from ondil.distributions import Weibull


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
