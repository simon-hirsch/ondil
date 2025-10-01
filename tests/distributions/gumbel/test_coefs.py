import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import Gumbel
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_gumbel_distribution():
    dist = Gumbel()
    code = f"""
    library(gamlss)
    data(mtcars)
    model = gamlss(
        mpg ~ cyl + hp,
        sigma.formula = ~cyl + hp,
        family=gamlss.dist::{dist.corresponding_gamlss}(),
        data=as.data.frame(mtcars)
    )
    list(
        "mu" = coef(model, "mu"),
        "sigma" = coef(model, "sigma")
    )
    """
    R_list = robjects.r(code)
    coef_R_mu = np.array(R_list.rx2("mu"))
    coef_R_sg = np.array(R_list.rx2("sigma"))

    estimator = OnlineDistributionalRegression(
        distribution=dist,
        equation={0: np.array([0, 2]), 1: np.array([0, 2])},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )
    estimator.fit(X=X, y=y)
    assert np.allclose(estimator.beta[0], coef_R_mu, atol=0.01)
    assert np.allclose(estimator.beta[1], coef_R_sg, atol=0.01)
