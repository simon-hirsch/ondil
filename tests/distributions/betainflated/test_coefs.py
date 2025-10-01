import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import BetaInflated
from ondil.estimators import OnlineDistributionalRegression


def test_beta_distribution():
    dist = BetaInflated()

    code = """
    library(gamlss)
    set.seed(1)
    n <- 2000
    mu <- runif(1)
    sigma <- runif(1)
    nu <- runif(1)
    tau <- runif(1)
    y <- gamlss.dist::rBEINF(n, mu, sigma, nu, tau)
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    model <- gamlss(
    y ~ x1 + x2,
    sigma.formula = ~x1 + x2,
    nu.formula = ~x1 + x2,
    tau.formula = ~x1 + x2,
    family = BEINF()
    )
    list(
    "y" = y,
    "x1" = x1, 
    "x2" = x2,
    "mu" = mu,
    "sigma" = sigma,
    "nu" = nu,
    "tau" = tau,
    "coef_R_mu" = coef(model, "mu"),
    "coef_R_sg" = coef(model, "sigma"),
    "coef_R_nu" = coef(model, "nu"),
    "coef_R_tau" = coef(model, "tau")    
    )
    """

    # obtain info from R

    R_list = robjects.r(code)
    y = np.array(R_list.rx2("y"))
    x1 = np.array(R_list.rx2("x1"))
    x2 = np.array(R_list.rx2("x2"))
    X = np.column_stack((x1, x2))

    estimator = OnlineDistributionalRegression(
        distribution=dist,
        equation={0: "all", 1: "all", 2: "all", 3: "all"},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X, y=y)
    print("Difference in estimates: ", estimator.beta[0] - R_list.rx2("coef_R_mu"))
    assert np.allclose(estimator.beta[0], R_list.rx2("coef_R_mu")), (
        "Location coefficients don't match"
    )
    print("Difference in estimates: ", estimator.beta[1] - R_list.rx2("coef_R_sg"))
    assert np.allclose(estimator.beta[1], R_list.rx2("coef_R_sg")), (
        "Scale coefficients don't match"
    )
    print("Difference in estimates: ", estimator.beta[2] - R_list.rx2("coef_R_nu"))
    assert np.allclose(estimator.beta[2], R_list.rx2("coef_R_nu"), atol=1e-3), (
        "Skew coefficients don't match"
    )
    print("Difference in estimates: ", estimator.beta[3] - R_list.rx2("coef_R_tau"))
    assert np.allclose(estimator.beta[3], R_list.rx2("coef_R_tau"), atol=1e-3), (
        "Kurtosis coefficients don't match"
    )
