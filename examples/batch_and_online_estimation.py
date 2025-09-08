from pprint import pprint

import numpy as np
from sklearn.datasets import load_diabetes

import ondil
from ondil.distributions import Normal, StudentT
from ondil.estimators import OnlineDistributionalRegression

np.set_printoptions(precision=3, suppress=True)
print(ondil.__version__)

# This example illustrates how to use `ondil` for both
# batch and online estimation of a distributional regression
# model. We use the diabetes data set from `sklearn` as an example.
# We first fit a batch model using OLS and then a LASSO model
# using the StudentT distribution. We then move to the online
# setting and fit a LASSO model to the first N-1 observations
# and update it with the last observation.


## Diabetes data set
X, y = load_diabetes(return_X_y=True)

## Simple linear regression with OLS
# We use all variables to model both mu and keep sigma as constant (intercept)
# Can also use: "intercept" or pass a numpy array with indices / boolean
equation = {
    0: "all",
    1: "intercept",
}

batch_model_ols = OnlineDistributionalRegression(
    distribution=Normal(),
    method="ols",
    equation=equation,
    fit_intercept=True,
)

batch_model_ols.fit(X, y)

print("OLS Coefficients \n")
pprint(batch_model_ols.beta)

## Use LASSO and the StudentT distribution
## We use all variables to model mu, sigma and nu
equation = {
    0: "all",
    1: "all",
    2: "all",
}

# Alternatively, we could also consider something like this:
# equation = {
#     0: "all",
#     1: np.array([0, 2, 3], dtype=int),  # only use columns 0, 2, and 3 to model sigma
#     2: "intercept",  # only use intercept to model nu
# }
# but then you need to be careful with stacking all betas (as we do below)
# as beta is a dictionary with arrays of different lengths

batch_model_lasso = OnlineDistributionalRegression(
    distribution=StudentT(),
    method="lasso",
    equation=equation,
    fit_intercept=True,
    ic="bic",
)
batch_model_lasso.fit(X, y)

print("LASSO Coefficients \n")
print(np.vstack([*batch_model_lasso.beta.values()]).T)

## Now we move to the online setting
## We create a new model and fit it to the first N-1 observations
online_model_lasso = OnlineDistributionalRegression(
    distribution=StudentT(),
    method="lasso",
    equation=equation,
    ic="bic",
)
online_model_lasso.fit(X=X[:-1, :], y=y[:-1])

print("Coefficients for the first N-1 observations \n")
print(np.vstack([*online_model_lasso.beta.values()]).T)

## Now we update the model with the last observation
online_model_lasso.update(X[[-1], :], y[[-1]])

print("\nCoefficients after update call \n")
print(np.vstack([*online_model_lasso.beta.values()]).T)
