"""
Online P-spline terms.

This example fits a smooth, heteroskedastic Gaussian model

    y ~ N(sin(2 x), exp(-0.5 + 0.3 x))

using the terms-based OnlineStructuredAdditiveDistributionRegressor with
penalized B-spline (P-spline) terms for both the mean and the (log-) scale.
The penalty strength of each term is selected via EDF-based AIC over a
geometric grid, and the model is then updated online with new data.
"""

import matplotlib.pyplot as plt
import numpy as np

from ondil.distributions import Normal
from ondil.estimators import OnlineStructuredAdditiveDistributionRegressor
from ondil.terms import InterceptTerm, PSplineTerm

# Simulate heteroskedastic data
rng = np.random.default_rng(42)
n = 1000
x = np.sort(rng.uniform(-2, 2, n))
mu = np.sin(2 * x)
sigma = np.exp(-0.5 + 0.3 * x)
y = mu + sigma * rng.normal(size=n)
X = x[:, None]

# Batch fit on the first part of the data
n0 = 750
estimator = OnlineStructuredAdditiveDistributionRegressor(
    distribution=Normal(),
    terms={
        0: [InterceptTerm(), PSplineTerm(feature=0, n_splines=20, ic="aic")],
        1: [InterceptTerm(), PSplineTerm(feature=0, n_splines=20, ic="aic")],
    },
)
estimator.fit(X[:n0], y[:n0])

# Online update with the remaining observations
estimator.update(X[n0:], y[n0:])

# Inspect the fitted smooths
for param in (0, 1):
    spline_term = estimator.terms_[param][1]
    print(
        f"Parameter {param}: selected lambda = {spline_term.lambda_selected_:.3g}, "
        f"edf = {spline_term.edf_:.2f}"
    )

fitted = estimator.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(x, y, s=5, alpha=0.3, color="grey")
axes[0].plot(x, mu, label="true $\\mu$", color="black")
axes[0].plot(x, fitted[:, 0], label="fitted $\\mu$", color="tab:red")
axes[0].legend()
axes[0].set_title("Location")
axes[1].plot(x, sigma, label="true $\\sigma$", color="black")
axes[1].plot(x, fitted[:, 1], label="fitted $\\sigma$", color="tab:red")
axes[1].legend()
axes[1].set_title("Scale")
fig.tight_layout()
plt.show()
