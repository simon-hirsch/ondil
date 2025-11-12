# `ondil`: Online Distributional Learning

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![License](https://img.shields.io/github/license/simon-hirsch/rolch)](https://opensource.org/license/gpl-3-0)
![GitHub Release](https://img.shields.io/github/v/release/simon-hirsch/ondil?display_name=release&label=Release)
[![Downloads](https://static.pepy.tech/badge/ondil)](https://pepy.tech/project/ondil)
[![Tests](https://github.com/simon-hirsch/ondil/actions/workflows/ci_run_tests.yml/badge.svg?branch=main)](https://github.com/simon-hirsch/ondil/actions/workflows/ci_run_tests.yml)
[![Docs](https://github.com/simon-hirsch/ondil/actions/workflows/ci_build_docs.yml/badge.svg?branch=main)](https://github.com/simon-hirsch/ondil/actions/workflows/ci_build_docs.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/simon-hirsch/ondil/badge)](https://www.codefactor.io/repository/github/simon-hirsch/ondil)

## Introduction

This package provides an online estimation of distributional regression and linear regression models in Python. We provide:

- Online linear regression models including regularization (Lasso, Ridge, Elastic Net).
- An online implementation of the generalized additive models for location, shape and scale (GAMLSS, see [Rigby & Stasinopoulos, 2005](https://academic.oup.com/jrsssc/article-abstract/54/3/507/7113027)) developed in [Hirsch, Berrisch & Ziel, 2024](https://arxiv.org/abs/2407.08750).
- The multivariate extension for online distributional regression models developed in [Hirsch, 2025](https://arxiv.org/abs/2504.02518).

All models are implemented in a way that they are fully compatible with `scikit-learn` estimators and transformers. The main advantage of the online approach is that the model can be updated incrementally using `model.update(X, y)` without the need to refit the whole model from scratch. This is especially useful for large datasets or streaming data. 

Please have a look at the [documentation](https://simon-hirsch.github.io/ondil/) or the [example files](https://github.com/simon-hirsch/ondil/tree/main/examples). We're actively working on the package and welcome contributions from the community. Have a look at the [Release Notes](https://github.com/simon-hirsch/ondil/releases) and the [Issue Tracker](https://github.com/simon-hirsch/ondil/issues).

## Distributional Regression

The main idea of distributional regression (or regression beyond the mean, multiparameter regression) is that the response variable $Y$ is distributed according to a specified distribution $\mathcal{F}(\theta)$, where $\theta$ is the parameter vector for the distribution. In the Gaussian case, we have $\theta = (\theta_1, \theta_2) = (\mu, \sigma)$. We then specify an individual regression model for all parameters of the distribution of the form

$$g_k(\theta_k) = \eta_k = X_k\beta_k$$

where $g_k(\cdot)$ is a link function, which ensures that the predicted distribution parameters are in a sensible range (we don't want, e.g. negative standard deviations), and $\eta_k$ is the predictor. For the Gaussian case, this would imply that we have two regression equations, one for the mean (location) and one for the standard deviation (scale) parameters. Distributions other than the normal distribution are possible, and we have already implemented them, e.g., Student's $t$-distribution and Johnson's $S_U$ distribution. If you are interested in another distribution, please open an Issue.

This allows us to specify very flexible models that consider the conditional behaviour of the variable's volatility, skewness and tail behaviour. A simple example for electricity markets is wind forecasts, which are skewed depending on the production level - intuitively, there is a higher risk of having lower production if the production level is already high since it cannot go much higher than "full load" and if, the turbines might cut-off. Modelling these conditional probabilistic behaviours is the key strength of distributional regression models.

## Features

- 🚀 First native `Python` implementation of generalized additive models for location, shape and scale (GAMLSS).
- 🚀 Online-first approach, which allows for incremental updates of the model using `model.update(X, y)`.
- 🚀 Support for various distributions, including Gaussian, Student's $t$, Johnson's $S_U$, Gamma, Log-normal, Exponential, Beta, Gumbel, Inverse Gaussian and more. Implementing new distributions is straight-forward.
- 🚀 Flexible link functions for each distribution, allowing for custom transformations of the parameters.
- 🚀 Support for regularization methods like Lasso, Ridge and Elastic Net.
- 🚀 Fast and efficient implementation using [`numba`](https://numba.pydata.org/) for just-in-time compilation.
- 🚀 Full compatibility with [`scikit-learn`](https://scikit-learn.org/stable/) estimators and transformers.

## Example

Basic estimation and updating procedure:

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

## Installation & Dependencies

The package is available from [pypi](https://pypi.org/project/ondil/) - do `pip install ondil` and enjoy.

`ondil` is designed to have minimal dependencies. We rely on `python>=3.10`, `numpy`, `numba`, `scipy` and `scikit-learn` in a reasonably up-to-date versions.

## I was looking for `rolch` but I found `ondil`?

`rolch` (Regularized Online Learning for Conditional Heteroskedasticity) was the original name of this package, but we decided to rename it to `ondil` (Online Distributional Learning) to better reflect its purpose and functionality, since conditional heteroskedasticity (=non constant variance) is just one of the many applications for distributional regression models that can be estimated with this package.

## Contributing

We welcome every contribution from the community. Feel free to open an issue if you find bugs or want to propose changes. We're still in an early phase and welcome feedback, especially on the usability and "look and feel" of the package. Secondly, we're working to port distributions from the `R`-GAMLSS package and welcome according PRs.

## Contributors

`ondil` was developed by Simon Hirsch, Jonathan Berrisch and Florian Ziel. 
We're grateful for contributions below (sorted alphabetically by GitHub username).

| Contribution | GitHub Users |
|-------------------|--------------|
| Code | [@BerriJ](https://github.com/BerriJ), [@Jaiminoza229984](https://github.com/Jaiminoza229984), [@flziel](https://github.com/flziel), [@joza26](https://github.com/joza26), [@murthy-econometrics-5819](https://github.com/murthy-econometrics-5819), [@simon-hirsch](https://github.com/simon-hirsch) |
| Reported (closed) Issues | [@BerriJ](https://github.com/BerriJ), [@Jaiminoza229984](https://github.com/Jaiminoza229984), [@fkiraly](https://github.com/fkiraly), [@joshdunnlime](https://github.com/joshdunnlime), [@murthy-econometrics-5819](https://github.com/murthy-econometrics-5819), [@simon-hirsch](https://github.com/simon-hirsch) |
| Merged PRs | [@BerriJ](https://github.com/BerriJ), [@Jaiminoza229984](https://github.com/Jaiminoza229984), [@flziel](https://github.com/flziel), [@murthy-econometrics-5819](https://github.com/murthy-econometrics-5819), [@simon-hirsch](https://github.com/simon-hirsch) |
| PR Reviews | [@BerriJ](https://github.com/BerriJ), [@copilot-pull-request-reviewer[bot]](https://github.com/copilot-pull-request-reviewer[bot]), [@fkiraly](https://github.com/fkiraly), [@giulianolmsk](https://github.com/giulianolmsk), [@simon-hirsch](https://github.com/simon-hirsch) |

## Acknowledgements & Disclosure

Simon Hirsch is employed at Statkraft and gratefully acknowledges support received from Statkraft for his Ph.D. studies. This work contains the author's opinion and not necessarily reflects Statkraft's position.

## Citation

If you use `ondil` in your research, please cite the following paper(s), depending on what you've used. Thank you!

```bibtex
@article{hirsch2024online,
  title={Online distributional regression},
  author={Hirsch, Simon and Berrisch, Jonathan and Ziel, Florian},
  journal={arXiv preprint arXiv:2407.08750},
  year={2024}
}

@article{hirsch2025online,
  title={Online Multivariate Regularized Distributional Regression for High-dimensional Probabilistic Electricity Price Forecasting},
  author={Hirsch, Simon},
  journal={arXiv preprint arXiv:2504.02518},
  year={2025}
}
```