import numpy as np

from ondil.distributions import Normal
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_normal_distribution():
    # Run R code to get coefficients
    # library("gamlss")
    # data(mtcars)

    # model = gamlss(
    #     mpg ~ cyl + hp,
    #     sigma.formula = ~cyl + hp,
    #     family=NO(),
    #     data=as.data.frame(mtcars)
    # )

    # coef(model, "mu")
    # coef(model, "sigma")

    # To get these coefficients
    coef_R_mu = np.array([36.51776626, -2.32470221, -0.01421071])
    coef_R_sg = np.array([1.8782995906, -0.1262290913, -0.0003943062])

    estimator = OnlineDistributionalRegression(
        distribution=Normal(),
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
