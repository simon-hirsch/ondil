# The `Estimator()` and `EstimationMethod()` classes

## Overview

Our package separates `Estimator` classes and `EstimationMethod` classes in the design. An `Estimator` is a python object that provides the user interface to set-up, fit, update and predict models. `EstimationMethod` classes are concerned with the estimation of the model coefficients (or weights). This page briefly explains the separation and options provided by it using the `OnlineLinearModel()` class.

Estimators are your bread and butter partner for modelling. They provide the methods:

- `Estimator().fit(X, y)`
- `Estimator().update(X, y)`
- `Estimator().predict_distribution_parameters(X, y)`
- `Estimator().predict(X)`

which one commonly uses for modelling.

Each estimator is initialized by choosing an estimation method passed to the `method` parameter, if the method is not explicitly called in the name of the estimator (like in the `OnlineLasso()`). The `method` accepts either a `string`, or an `EstimationMethod()` instance.

## Example

Let's return to the aforementioned example: We want to fit a simple linear model. We can estimate the parameters either using ordinary least squares (OLS) or using coordinate descent, minimizing the LASSO penalised loss.

### Ordinary Least Squares

First, we start with OLS:

```python
# Set up packages and
from ondil.estimators.online_linear_model import OnlineLinearModel
from ondil.methods import LassoPath, OrdinaryLeastSquares
from sklearn.datasets import load_diabetes

import matplotlib.pyplot as plt
import numpy as np

# Get data
X, y = load_diabetes(return_X_y=True)

fit_intercept = False
scale_inputs = True

# This is the Estimator Class
model = OnlineLinearModel(
    method="ols", 
    fit_intercept=fit_intercept, 
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
model.update(X[-10:, :], y[-10:])

# This is equivalent
model = OnlineLinearModel(
    method=OrdinaryLeastSquares(), 
    fit_intercept=fit_intercept, 
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
model.update(X[-10:, :], y[-10:])
```

Since ordinary least squares is a pretty simple method, it does not have a lot of parameters. However, if we look at LASSO, things change, because now we can actually play with the parameters.

### LASSO and the `LassoPath()`

The `LassoPath()` estimates the coefficients using coordinate descent along a path of decreasing regularization strength. In this example, we will change some of the parameters of the estimation.

The `LassoPath()` has for example the following parameters

- `lambda_n` which defines the length of the regularization path.
- `beta_lower_bounds` which provides the option to place a lower bound on the coefficients/weights.

Let's have a look at a basic LASSO-estimated model:

```python

model = OnlineLinearModel(
    method="lasso",
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)

# Equivalent, we can do:

model = OnlineLinearModel(
    method=LassoPath(),
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)

```

Now we want to change the parameters:

```python

estimation_method = LassoPath(
    lambda_n=10,  # Only fit ten lambdas
    beta_lower_bound=np.zeros(
        X.shape[1] + fit_intercept
    ),  # all positive parameters
)

model = OnlineLinearModel(
    method=estimation_method,
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
model.fit(X[:-10, :], y[:-10])
plt.plot(model.beta_path)
plt.show(block=False)
print(model.beta)
```

And we see that the coefficient path is both shorter and non-negative.
