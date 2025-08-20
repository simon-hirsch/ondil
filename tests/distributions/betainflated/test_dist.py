import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import BetaInflated


def test_betainflated_distribution():
    dist = BetaInflated()

    code = f"""
    library(gamlss.dist)
    set.seed(12)
    y <- rep(0.3, 3)
    mu <- runif(3)
    sigma <- runif(3)
    nu <- rexp(3)
    tau <- rexp(3)
    dist <- gamlss.dist::{dist.corresponding_gamlss}()

    list(
      "y" = y,
      "mu" = mu,
      "sigma" = sigma,
      "nu" = nu,
      "tau" = tau,
      "cdf" = pBEINF(y, mu, sigma, nu, tau),
      "pdf" = dBEINF(y, mu, sigma, nu, tau),
      "ppf" = qBEINF(y, mu, sigma, nu, tau),
      "logcdf" = pBEINF(y, mu, sigma, nu, tau, log.p = TRUE),
      "logpdf" = dBEINF(y, mu, sigma, nu, tau, log = TRUE)
    )
    """

    # obtain info from R

    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array(
        [
            R_list.rx2("mu"),  # mu
            R_list.rx2("sigma"),  # sigma
            R_list.rx2("nu"),  # nu
            R_list.rx2("tau"),  # tau
        ]
    )

    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(y, theta=theta.T)
    logcdf = dist.logcdf(y, theta=theta.T)
    logpdf = dist.logpdf(y, theta=theta.T)

    assert np.allclose(cdf, R_list.rx2("cdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
    assert np.allclose(pdf, R_list.rx2("pdf")), (
        "Densities don't match, inspect the pdf function"
    )
    assert np.allclose(ppf, R_list.rx2("ppf")), (
        "Quantiles don't match, inspect the ppf function"
    )
    assert np.allclose(logcdf, R_list.rx2("logcdf")), (
        "Quantiles don't match, inspect the ppf function"
    )
    assert np.allclose(logpdf, R_list.rx2("logpdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
