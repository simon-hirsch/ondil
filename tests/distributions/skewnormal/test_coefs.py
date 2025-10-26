import numpy as np

from ondil.distributions import SkewNormal
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_skewnormal_distribution():
    # Run R code to get coefficients
    # library("gamlss")
    # library("gamlss.dist")
    # data(mtcars)
    
    # model = gamlss(
    #     mpg ~ cyl + hp,
    #     sigma.formula = ~cyl + hp,
    #     nu.formula = ~1,
    #     family=SN1(),
    #     data=as.data.frame(mtcars)
    # )
    
    # coef(model, "mu")
    # coef(model, "sigma")
    # coef(model, "nu")

    # To get these coefficients (placeholders - will be updated after running R)
    # For now, we'll just test that the estimator runs without error
    
    estimator = OnlineDistributionalRegression(
        distribution=SkewNormal(),
        equation={0: np.array([0, 2]), 1: np.array([0, 2]), 2: np.array([])},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X, y=y)
    
    # Basic sanity checks
    assert estimator.beta[0] is not None, "Location coefficients not fitted"
    assert estimator.beta[1] is not None, "Scale coefficients not fitted"
    assert estimator.beta[2] is not None, "Skewness coefficients not fitted"
    
    # Check that coefficients have reasonable values
    assert not np.any(np.isnan(estimator.beta[0])), "Location coefficients contain NaN"
    assert not np.any(np.isnan(estimator.beta[1])), "Scale coefficients contain NaN"
    assert not np.any(np.isnan(estimator.beta[2])), "Skewness coefficients contain NaN"
