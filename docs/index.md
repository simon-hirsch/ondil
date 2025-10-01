# Welcome to `ondil` - Online Distributional Learning

`ondil` is a `Python` package for online distributional learning. We provide an online implementation of the well-known GAMLSS model using online coordinate descent (OCD).

!!! note
    `ondil` is currently in the first alpha phase. Please expect changes to happen frequently.

## Introduction

This package provides an online estimation of models for distributional regression, respectively, models for conditional heteroskedastic data. The main contribution is an online/incremental implementation of the generalized additive models for location, shape and scale (GAMLSS, see [Rigby & Stasinopoulos, 2005](https://academic.oup.com/jrsssc/article-abstract/54/3/507/7113027)) developed in [Hirsch, Berrisch & Ziel, 2024](https://arxiv.org/abs/2407.08750).

Please have a look at the [documentation](https://simon-hirsch.github.io/ondil/) or the [example notebook](https://github.com/simon-hirsch/ondil/blob/main/example.ipynb).

We're actively working on the package and welcome contributions from the community. Have a look at the [Release Notes](https://github.com/simon-hirsch/ondil/releases) and the [Issue Tracker](https://github.com/simon-hirsch/ondil/issues).

## Distributional Regression

The main idea of distributional regression (or regression beyond the mean, multiparameter regression) is that the response variable $Y$ is distributed according to a specified distribution $\mathcal{F}(\theta)$, where $\theta$ is the parameter vector for the distribution. In the Gaussian case, we have $\theta = (\theta_1, \theta_2) = (\mu, \sigma)$. We then specify an individual regression model for all parameters of the distribution of the form 

$$g_k(\theta_k) = \eta_k = X_k\beta_k$$

where $g_k(\cdot)$ is a link function, which ensures that the predicted distribution parameters are in a sensible range (we don't want, e.g. negative standard deviations), and $\eta_k$ is the predictor. For the Gaussian case, this would imply that we have two regression equations, one for the mean (location) and one for the standard deviation (scale) parameters. Distributions other than the normal distribution are possible, and we have already implemented them, e.g., Student's $ t$ distribution and Johnson's $S_U$ distribution. If you are interested in another distribution, please open an Issue.

This allows us to specify very flexible models that consider the conditional behaviour of the variable's volatility, skewness and tail behaviour. A simple example for electricity markets is wind forecasts, which are skewed depending on the production level - intuitively, there is a higher risk of having lower production if the production level is already high since it cannot go much higher than "full load" and if, the turbines might cut-off. Modelling these conditional probabilistic behaviours is the key strength of distributional regression models.

## Installation

`ondil` is available on the [Python Package Index](https://pypi.org/project/ondil/) and can be installed via `pip`:

```shell
pip install ondil
```
## I was looking for `rolch` but I found `ondil`?

`rolch` (Regularized Online Learning for Conditional Heteroskedasticity) was the original name of this package, but we decided to rename it to `ondil` (Online Distributional Learning) to better reflect its purpose and functionality, since conditional heteroskedasticity (=non constant variance) is just one of the many applications for distributional regression models that can be estimated with this package.

## Example

The following few lines give an introduction. We use the `diabetes` data set and model the response variable \(Y\) as Student-\(t\) distributed, where all distribution parameters (location, scale and tail) are modelled conditional on the explanatory variables in \(X\). We use LASSO to estimate the coefficients and the Bayesian information criterion to select the best model along a grid of regularization strengths.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from ondil.estimators import OnlineDistributionalRegression
from ondil.distributions import StudentT

X, y = load_diabetes(return_X_y=True)

# Model coefficients
equation = {
    0: "all",  # Can also use "intercept" or np.ndarray of integers / booleans
    1: "all",
    2: "all",
}

# Create the estimator
online_gamlss_lasso = OnlineDistributionalRegression(
    distribution=StudentT(),
    method="lasso",
    equation=equation,
    fit_intercept=True,
    ic="bic",
)

# Initial Fit
online_gamlss_lasso.fit(
    X=X[:-11, :],
    y=y[:-11],
)
print("Coefficients for the first N-11 observations \n")
print(online_gamlss_lasso.beta)

# Update call
online_gamlss_lasso.update(X=X[[-11], :], y=y[[-11]])
print("\nCoefficients after update call \n")
print(online_gamlss_lasso.beta)

# Prediction for the last 10 observations
prediction = online_gamlss_lasso.predict_distribution_parameters(X=X[-10:, :])

print("\n Predictions for the last 10 observations")
# Location, scale and shape (degrees of freedom)
print(prediction)
```
