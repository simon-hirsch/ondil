import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

import ondil
from ondil.estimators.online_linear_model import OnlineLinearModel
from ondil.methods import LassoPath

np.set_printoptions(precision=3, suppress=True)
print(ondil.__version__)


## Estimators and EstimationMethods

# `ondil` allows for the separation of estimation method
# (LASSO, OLS, ...) and the model (a simple linear model,
# a distributional regression) and therefore provides ample
# flexibility for the modeller. For example, we might want to
# estimate a simple linear model using the LASSO on a
# regularization path that is only 10 regularization steps(lambdas)
# long (instead of the default 100, as in e.g. `R`s `glmnet`)
# and box-constrain our coefficients to be positive.

# This can be achieved simply by the following few lines.


## Diabetes data set
X, y = load_diabetes(return_X_y=True)

fit_intercept = False
scale_inputs = True

print("############################ OLS ##########################")
model = OnlineLinearModel(
    method="ols",
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
print(model.beta)

model.update(X[-10:, :], y[-10:])
print(model.beta)


print("############################ LASSO ##########################")
model = OnlineLinearModel(
    method="lasso",  # default parameters
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)


model.update(X[-10:, :], y[-10:])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)

print("############################ LASSO CONSTRAINED ##########################")

model = OnlineLinearModel(
    method=LassoPath(
        lambda_n=10,  # Only fit 10 lambdas
        beta_lower_bound=np.zeros(
            X.shape[1] + fit_intercept
        ),  # all positive parameters
    ),
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)

model.update(X[-10:, :], y[-10:])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)
