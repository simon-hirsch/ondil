import numpy as np

from ondil.distributions import Pareto
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_pareto_distribution():
    # Run R code to get coefficients
    # library("gamlss")
    # library("gamlss.dist")
    # data(mtcars)

    # model = gamlss(
    #     mpg ~ cyl + hp,
    #     sigma.formula = ~cyl + hp,
    #     family=PARETO2(),
    #     data=as.data.frame(mtcars)
    # )

    # coef(model, "mu")
    # coef(model, "sigma")

    # To get these coefficients (to be updated after running R)
    coef_R_mu = np.array([3.47299, -0.04899, -0.00343])
    coef_R_sg = np.array([2.18723, -0.18287, 0.00131])

    estimator = OnlineDistributionalRegression(
        distribution=Pareto(),
        equation={0: np.array([0, 2]), 1: np.array([0, 2])},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X, y=y)

    assert np.allclose(estimator.beta[0], coef_R_mu, atol=0.01), (
        "Location coefficients don't match"
    )
    assert np.allclose(estimator.beta[1], coef_R_sg, atol=0.01), (
        "Scale coefficients don't match"
    )
