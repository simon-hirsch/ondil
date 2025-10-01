import numpy as np
import rpy2.robjects as robjects
from ondil.distributions import DistributionPowerExponential


def test_powerexponential_distribution():
    dist = DistributionPowerExponential()

    code = f"""
    set.seed(1)
    y <- rep(0.3, 3)
    mu <- rnorm(3)
    sigma <- rexp(3)
    nu <- rep(1.8, 3)
    dist <- gamlss.dist::{dist.corresponding_gamlss}()
    list(
      "y" = y,
      "mu" = mu,
      "sigma" = sigma,
      "nu" = nu,
      "dldm" = dist$dldm(y, mu, sigma, nu),
      "dldd" = dist$dldd(y, mu, sigma, nu),
      "d2ldm2" = dist$d2ldm2(y, mu, sigma, nu),
      "d2ldd2" = dist$d2ldd2(sigma, nu),
      "d2ldmdd" = dist$d2ldmdd(y)
    )
    """

    # Obtain data and derivatives from R
    R_list = robjects.r(code)

    # We take the data from R
    y = np.array(R_list.rx2("y"))
    theta = np.array(
        [
            R_list.rx2("mu"),
            R_list.rx2("sigma"),
            R_list.rx2("nu"),
        ]
    )
    theta = theta.T

    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta, param=0)
    dl1_dp1_1 = dist.dl1_dp1(y, theta=theta, param=1)
    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta, param=0)
    dl2_dp2_1 = dist.dl2_dp2(y, theta=theta, param=1)
    dl2_dpp = dist.dl2_dpp(y, theta=theta, params=(0, 1))

    assert np.allclose(dl1_dp1_0, R_list.rx2("dldm")), (
        "First derivative wrt mu doesn't match"
    )
    assert np.allclose(dl1_dp1_1, R_list.rx2("dldd")), (
        "First derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dp2_0, R_list.rx2("d2ldm2")), (
        "Second derivative wrt mu doesn't match"
    )
    assert np.allclose(dl2_dp2_1, R_list.rx2("d2ldd2")), (
        "Second derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dpp, R_list.rx2("d2ldmdd")), (
        "Second derivative wrt mu and sigma doesn't match"
    )
