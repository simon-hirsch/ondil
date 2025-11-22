import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import Poisson
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 1]
X = mtcars[:, [0] + list(range(2, mtcars.shape[1]))]


def test_possion_distribution():
    dist = Poisson()
    code = f"""
    library(gamlss)
    data(mtcars)
    model = gamlss(
        cyl ~ mpg + hp,
        family=gamlss.dist::{dist.corresponding_gamlss}(),
        data=as.data.frame(mtcars),
    )
    list(
        "mu" = coef(model, "mu")
    )
    """
    R_list = robjects.r(code)
    coef_R_mu = np.array(R_list.rx2("mu"))

    estimator = OnlineDistributionalRegression(
        distribution=dist,
        equation={0: np.array([0, 2])},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )
    estimator.fit(X=X, y=y)
    assert np.allclose(estimator.beta[0], coef_R_mu, atol=0.01), (
        "Mu coefficients do not match R GAMLSS results. Expected: {}, Got: {}".format(
            coef_R_mu.round(3), estimator.beta[0].round(3)
        )
    )
