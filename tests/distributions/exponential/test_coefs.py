# %%
import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import Exponential
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_exponential_distribution():
    dist = Exponential()

    code = f"""
    library("gamlss")
    data(mtcars)

    model = gamlss(
        mpg ~ cyl + hp,
        family=gamlss.dist::{dist.corresponding_gamlss}(),
        data=as.data.frame(mtcars)
    )

    list(
        "mu" = coef(model, "mu")
    )
    """

    # Obtain coefficients from R
    R_list = robjects.r(code)

    coef_R_mu = R_list.rx2("mu")

    estimator = OnlineDistributionalRegression(
        distribution=dist,
        equation={0: np.array([0, 2])},  # Only mu
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X, y=y)

    assert np.allclose(estimator.beta[0], coef_R_mu, atol=0.0001), (
        "Location coefficients don't match"
    )
