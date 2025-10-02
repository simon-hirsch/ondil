# %%
import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import DistributionPowerExponential
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]

X_design = np.column_stack((np.ones(X.shape[0]), X))


def test_powerexponential_distribution():
    dist = DistributionPowerExponential()

    code = f"""
    library(gamlss)
    data(mtcars)

    model = gamlss(
        mpg ~ cyl + hp,
        sigma.formula = ~cyl + hp,
        nu.formula = ~1,
        family=gamlss.dist::{dist.corresponding_gamlss}(),
        data=mtcars
    )

    list(
        "mu" = coef(model, "mu"),
        "sigma" = coef(model, "sigma"),
        "nu" = coef(model, "nu")
    )
    """

    R_list = robjects.r(code)
    coef_R_mu = np.array(R_list.rx2("mu"))
    coef_R_sg = np.array(R_list.rx2("sigma"))
    coef_R_nu = np.array(R_list.rx2("nu"))

    estimator = OnlineDistributionalRegression(
        distribution=dist,
        equation={
            0: np.array([1, 3]),  # mu ~ cyl + hp
            1: np.array([1, 3]),  # sigma ~ cyl + hp
            2: "intercept",  # nu ~ intercept
        },
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X_design, y=y)

    assert np.allclose(
        estimator.beta[0], coef_R_mu, atol=0.012
    ), "Location coefficients don't match"
    assert np.allclose(
        estimator.beta[1], coef_R_sg, atol=0.01
    ), "Scale coefficients don't match"
    assert np.allclose(
        estimator.beta[2], coef_R_nu, atol=0.01
    ), "Shape (nu) coefficients don't match"
