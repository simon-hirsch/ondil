import numpy as np
import rpy2.robjects as robjects
from ondil.distributions import DistributionPowerExponential


def test_powerexponential_distribution():
    dist = DistributionPowerExponential()

    code = f"""
    library(gamlss.dist)
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
      "cdf" = pPE(y, mu, sigma, nu),
      "pdf" = dPE(y, mu, sigma, nu),
      "ppf" = qPE(pPE(y, mu, sigma, nu), mu, sigma, nu)
    )
    """

    # Obtain data and derivatives from R
    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array(
        [
            R_list.rx2("mu"),  # mu
            R_list.rx2("sigma"),  # sigma
            R_list.rx2("nu"),  # nu
        ]
    )

    # Compare Python vs R
    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(cdf, theta=theta.T)

    assert np.allclose(cdf, R_list.rx2("cdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
    assert np.allclose(pdf, R_list.rx2("pdf")), (
        "Densities don't match, inspect the pdf function"
    )
    assert np.allclose(ppf, R_list.rx2("ppf")), (
        "Quantiles don't match, inspect the ppf function"
    )
