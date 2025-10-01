import numpy as np

from ondil.distributions import Beta
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

eps = 1e-10
y = mtcars[:, 0] / (np.max(mtcars[:, 0]) + eps)
X = mtcars[:, 1:]


def test_beta_distribution():
    # Run R code to get coefficients
    # library("gamlss")
    # data(mtcars)
    # eps <- 1e-10
    # mtcars$mpg <- mtcars$mpg/(max(mtcars$mpg) +  eps)
    #
    # model = gamlss(
    #     mpg ~ cyl + hp,
    #     sigma.formula = ~cyl + hp,
    #     family=BE(),
    #     data=as.data.frame(mtcars)
    # )
    #
    # coef(model, "mu")
    # coef(model, "sigma")

    # To get these coefficients
    coef_R_mu = np.array([3.0143212798, -0.4316644910, 0.0006211862])
    coef_R_sg = np.array([1.38959112, 0.10110228, -0.01992618])

    estimator = OnlineDistributionalRegression(
        distribution=Beta(),
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
